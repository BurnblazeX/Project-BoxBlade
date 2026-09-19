import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, float, vec3, uniform, instanceIndex, storage, texture3D,
  positionWorld, abs, sign, floor, max, texture, uv
} from 'three/tsl';
import { GRID_DIM, VOXEL_METRES } from './boxgrid.js';

// --- Phase B: WebGPU compute plumbing and the DDA on the GPU ---

// Ceiling on march length. The grid is 144 voxels on a side, so a full body
// diagonal crosses at most 3*144 boundaries; this bounds the loop for the
// shader compiler without ever truncating a legitimate ray.
const MAX_STEPS = 3 * GRID_DIM;

// grid.data stores 1 for occupied, but a RedFormat/UnsignedByteType texture is
// sampled NORMALISED - so an occupied voxel arrives in the shader as 1/255, not
// 1. Comparing against 0.5 meant the test could never pass and the march never
// reported a hit, which showed up as the GPU disagreeing with the CPU on every
// ray that should have been occluded. Half a quantisation step is the threshold.
const OCCUPIED = 0.5 / 255;

// ---------------------------------------------------------------------------
// 1. Compute smoke test
//
// The smallest thing that proves the whole compute path works end to end:
// dispatch a kernel, have it write a storage buffer, read it back, check the
// numbers. Run this before anything depends on compute, because a failure here
// is an environment problem, not a bug in the lighting.
// ---------------------------------------------------------------------------
export async function runComputeSmokeTest(renderer, count = 64) {
  const buffer = new THREE.StorageBufferAttribute(count, 1);
  const target = storage(buffer, 'float', count);

  const kernel = Fn(() => {
    // Deliberately not a constant: index-dependent output catches a dispatch
    // that runs once, or runs with the wrong workgroup count.
    target.element(instanceIndex).assign(instanceIndex.toFloat().mul(2).add(1));
  })().compute(count);

  await renderer.computeAsync(kernel);
  const read = new Float32Array(await renderer.getArrayBufferAsync(buffer));

  let mismatches = 0;
  let firstBad = null;
  for (let i = 0; i < count; i++) {
    const want = i * 2 + 1;
    if (read[i] !== want) {
      mismatches++;
      if (firstBad === null) firstBad = { index: i, got: read[i], want };
    }
  }

  const result = { ok: mismatches === 0, count, mismatches, firstBad, sample: [...read.slice(0, 8)] };
  console.log(result.ok
    ? `[gpu] compute smoke test PASSED - ${count} values written and read back`
    : `[gpu] compute smoke test FAILED - ${mismatches}/${count} wrong, first: ${JSON.stringify(firstBad)}`);
  return result;
}

// ---------------------------------------------------------------------------
// 2. Occupancy as a 3D texture
//
// A Data3DTexture rather than a storage buffer, because the shading pass that
// consumes it is a fragment shader. Nearest filtering throughout: occupancy is
// binary, and interpolating it would smear solid into empty and soften exactly
// the hard voxel edge the art direction is built on.
// ---------------------------------------------------------------------------
export function createOccupancyTexture(grid) {
  const tex = new THREE.Data3DTexture(grid.data, GRID_DIM, GRID_DIM, GRID_DIM);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

export function updateOccupancyTexture(tex) {
  tex.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// 3. DDA in TSL
//
// A direct port of marchOccupancy() in boxgrid.js - Amanatides-Woo, stepping in
// metres along a normalised direction. The CPU version is the reference and is
// checked against brute-force sampling in the test suite; this must agree with
// it, which is what runDDAParity() below exists to confirm.
//
// Returns 1.0 when the ray hit occupancy before maxDist, 0.0 otherwise.
// ---------------------------------------------------------------------------
export const marchOccupancyTSL = Fn(([occTex, gridOrigin, origin, dir, maxDist]) => {
  const dim = float(GRID_DIM);
  const vs = float(VOXEL_METRES);

  // Continuous voxel-space position, then the cell containing it.
  const p = origin.sub(gridOrigin).div(vs);
  const v = floor(p).toVar();

  const stepv = sign(dir).toVar();
  // A zero component would divide by zero; a large finite number stands in for
  // the infinity the CPU version uses, and never wins the comparison below.
  const BIG = float(1e20);
  const tDelta = vec3(
    vs.div(max(abs(dir.x), float(1e-8))),
    vs.div(max(abs(dir.y), float(1e-8))),
    vs.div(max(abs(dir.z), float(1e-8)))
  ).toVar();

  // Distance to the first boundary on each axis: the far side of the current
  // cell when stepping positive, the near side when stepping negative.
  const boundary = vec3(
    v.x.add(max(stepv.x, float(0))),
    v.y.add(max(stepv.y, float(0))),
    v.z.add(max(stepv.z, float(0)))
  ).mul(vs).add(gridOrigin);
  const tMax = vec3(
    boundary.x.sub(origin.x).div(dir.x),
    boundary.y.sub(origin.y).div(dir.y),
    boundary.z.sub(origin.z).div(dir.z)
  ).toVar();
  // Axes the ray does not travel along must never be chosen as the next step.
  If(abs(dir.x).lessThan(float(1e-8)), () => { tMax.x.assign(BIG); tDelta.x.assign(BIG); });
  If(abs(dir.y).lessThan(float(1e-8)), () => { tMax.y.assign(BIG); tDelta.y.assign(BIG); });
  If(abs(dir.z).lessThan(float(1e-8)), () => { tMax.z.assign(BIG); tDelta.z.assign(BIG); });

  const t = float(0).toVar();
  const hit = float(0).toVar();
  const first = float(1).toVar();

  Loop({ start: 0, end: MAX_STEPS, type: 'int', condition: '<' }, () => {
    // Leaving the grid means open sky for this cascade.
    If(v.x.lessThan(float(0)).or(v.y.lessThan(float(0))).or(v.z.lessThan(float(0)))
       .or(v.x.greaterThanEqual(dim)).or(v.y.greaterThanEqual(dim)).or(v.z.greaterThanEqual(dim)), () => {
      Break();
    });

    // The origin voxel is the shading surface itself; counting it would make
    // every lit surface shadow itself.
    If(first.equal(float(0)), () => {
      const uvw = v.add(float(0.5)).div(dim);
      If(texture3D(occTex, uvw).r.greaterThan(float(OCCUPIED)), () => {
        hit.assign(float(1));
        Break();
      });
    });
    first.assign(float(0));

    If(t.greaterThan(maxDist), () => { Break(); });

    // Advance along whichever axis reaches its next boundary first.
    If(tMax.x.lessThanEqual(tMax.y).and(tMax.x.lessThanEqual(tMax.z)), () => {
      v.x.addAssign(stepv.x); t.assign(tMax.x); tMax.x.addAssign(tDelta.x);
    }).ElseIf(tMax.y.lessThanEqual(tMax.z), () => {
      v.y.addAssign(stepv.y); t.assign(tMax.y); tMax.y.addAssign(tDelta.y);
    }).Else(() => {
      v.z.addAssign(stepv.z); t.assign(tMax.z); tMax.z.addAssign(tDelta.z);
    });
  });

  return hit;
});

// ---------------------------------------------------------------------------
// 4. CPU/GPU parity
//
// Runs the same rays through the GPU kernel and returns the results, so they
// can be diffed against marchOccupancy() from boxgrid.js. The whole reason the
// CPU DDA was written first was to have something trustworthy to check this
// against - a shader that silently disagrees is very hard to notice by eye.
// ---------------------------------------------------------------------------
export async function runGPUMarch(renderer, occTex, grid, rays) {
  const n = rays.length;
  const originBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);
  const dirBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);
  const outBuf = new THREE.StorageBufferAttribute(n, 1);

  for (let i = 0; i < n; i++) {
    const r = rays[i];
    const len = Math.hypot(r.dir.x, r.dir.y, r.dir.z) || 1;
    originBuf.array.set([r.origin.x, r.origin.y, r.origin.z, 0], i * 4);
    dirBuf.array.set([r.dir.x / len, r.dir.y / len, r.dir.z / len, r.maxDist], i * 4);
  }

  const origins = storage(originBuf, 'vec4', n);
  const dirs = storage(dirBuf, 'vec4', n);
  const out = storage(outBuf, 'float', n);
  const gridOrigin = uniform(new THREE.Vector3(grid.origin.x, grid.origin.y, grid.origin.z));

  const kernel = Fn(() => {
    const o = origins.element(instanceIndex);
    const d = dirs.element(instanceIndex);
    out.element(instanceIndex).assign(
      marchOccupancyTSL(occTex, gridOrigin, o.xyz, d.xyz, d.w)
    );
  })().compute(n);

  await renderer.computeAsync(kernel);
  return new Float32Array(await renderer.getArrayBufferAsync(outBuf));
}

// ---------------------------------------------------------------------------
// 5. Shadow shading node
//
// Phase B shades per screen pixel, not into an atlas - that is Phase C. One
// light, one shadow ray, marched through occupancy. Enough to prove traversal
// works against real geometry before the atlas is built on top of it.
// ---------------------------------------------------------------------------
export function createShadowColorNode({ occTex, grid, lightPos, ambient = 0.35 }) {
  const gridOrigin = uniform(new THREE.Vector3(grid.origin.x, grid.origin.y, grid.origin.z));
  const light = uniform(lightPos);

  const node = Fn(() => {
    const toLight = light.sub(positionWorld);
    const dist = toLight.length();
    // Offset the ray start slightly toward the light. The fragment sits exactly
    // on a voxel face, and floating point can place it either side; a nudge
    // keeps it out of the surface it belongs to.
    const startPoint = positionWorld.add(toLight.div(dist).mul(float(VOXEL_METRES * 0.5)));
    const shadowed = marchOccupancyTSL(occTex, gridOrigin, startPoint, toLight.div(dist), dist);
    return float(1).sub(shadowed.mul(float(1 - ambient)));
  })();

  return { node, lightUniform: light, gridOriginUniform: gridOrigin };
}

// Swaps the terrain to a node material whose colour is its own albedo modulated
// by the marched shadow term, keeping the original so it can be restored.
export function applyShadowMaterial(mesh, shadowNode) {
  if (!mesh.userData.originalMaterial) mesh.userData.originalMaterial = mesh.material;
  const base = mesh.userData.originalMaterial;

  const mat = new THREE.MeshBasicNodeMaterial();
  const albedo = base.map ? texture(base.map, uv()) : vec3(1, 1, 1);
  // Per-instance tint (arena greying) lives in instanceColor, so it has to be
  // carried across or battle mode loses its reachable-tile shading.
  mat.colorNode = albedo.mul(shadowNode);
  mesh.material = mat;
  return mat;
}

export function restoreOriginalMaterial(mesh) {
  if (!mesh.userData.originalMaterial) return false;
  if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
  mesh.material = mesh.userData.originalMaterial;
  return true;
}
