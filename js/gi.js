import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, float, vec2, vec3, vec4, int, uvec3, uniform, instanceIndex,
  storage, texture3D, textureStore, max, select
} from 'three/tsl';
import { HIT_EPS, MIN_STEP, cascadeVoxelMetres } from './boxgrid.js';
import {
  decodeFieldTSL, ringUV, createConeTraceSunTSL, createPointLightsTSL,
  createSunUniforms, SURFACE_BIAS_VOXELS, SHADOW_FADE_START, EDGE_FADE_VOXELS
} from './gpu.js';
import {
  LPV_DIM, LPV_CELLS, LPV_CELL_VOXELS, FACE_DIRS, SOLID_THRESHOLD,
  DEFAULT_LPV_SPREAD, DEFAULT_LPV_SLICES
} from './lpv.js';

// --- The LPV on the GPU: solid, inject, propagate, resolve ---
//
// lpv.js has the why and the CPU reference; this is the kernels. Four storage
// buffers and one 3D texture:
//
//   solid   float   how much of each cell is geometry        every frame
//   E       vec4    injected light (rgb)                     one slice a frame
//   A, B    vec4    propagated light, ping-ponged            N iterations a frame
//   tex     RGBA16F what the shading pass samples, trilinear  every frame
//
// The texture is written by a resolve pass that DILATES into solid cells: a
// solid cell takes the mean of its air neighbours. The surfaces the shading pass
// samples near are exactly where trilinear filtering would otherwise blend in
// the zeros inside the ground and darken every contact.

const NO_SKIP = vec2(1e9, 1e9);
const D = LPV_DIM;

// instanceIndex to cell coordinates, as ints.
function cellOf(index) {
  const i = int(index).toVar();
  return { i, x: i.mod(int(D)), y: i.div(int(D)).mod(int(D)), z: i.div(int(D * D)) };
}

// The neighbour one step along (dx, dy, dz), or this cell when that is off the
// volume - the CPU reference's edge rule.
function neighbour(c, dx, dy, dz) {
  const nx = c.x.add(int(dx)), ny = c.y.add(int(dy)), nz = c.z.add(int(dz));
  const inside = nx.greaterThanEqual(int(0)).and(ny.greaterThanEqual(int(0)))
    .and(nz.greaterThanEqual(int(0))).and(nx.lessThan(int(D)))
    .and(ny.lessThan(int(D))).and(nz.lessThan(int(D)));
  const n = nx.add(ny.mul(int(D))).add(nz.mul(int(D * D)));
  return { inside, index: select(inside, n, c.i) };
}

// level: which cascade the volume lies over. C0 gives 0.75 m cells over 18 m,
// C1 1.5 m cells over 36 m - the same 24^3 and six voxels a cell either way,
// because each cascade is the same 144^3 at a doubled voxel size.
export function createLPV({ cascades, lights, level = 0, biasUniform = null,
                            fadeStartUniform = null, edgeFadeUniform = null }) {
  const c0 = cascades[level];
  const cellMetres = LPV_CELL_VOXELS * cascadeVoxelMetres(level);
  const buf = (itemSize) => new THREE.StorageBufferAttribute(LPV_CELLS, itemSize);
  const solidBuf = buf(1), eBuf = buf(4), aBuf = buf(4), bBuf = buf(4);
  const solid = storage(solidBuf, 'float', LPV_CELLS);
  const E = storage(eBuf, 'vec4', LPV_CELLS);
  const A = storage(aBuf, 'vec4', LPV_CELLS);
  const B = storage(bBuf, 'vec4', LPV_CELLS);

  const tex = new THREE.Storage3DTexture(D, D, D);
  tex.type = THREE.HalfFloatType;
  tex.format = THREE.RGBAFormat;
  tex.generateMipmaps = false;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;

  const u = {
    // World-space min corner of the volume: C0's origin, tracked by the CPU.
    origin: uniform(new THREE.Vector3()),
    spread: uniform(float(DEFAULT_LPV_SPREAD)),
    albedo: uniform(new THREE.Vector3(0.3, 0.4, 0.2)),
    sunGain: uniform(float(1)),
    slice: uniform(int(0)),
    slices: uniform(int(DEFAULT_LPV_SLICES)),
    shift: uniform(new THREE.Vector3())
  };
  const sun = createSunUniforms(new THREE.Vector3(0, 1, 0));
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const fadeStart = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const edgeFade = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));

  const fieldAt = p => {
    const vp = p.sub(c0.origin).div(c0.voxel);
    return decodeFieldTSL(texture3D(c0.tex, ringUV(vp, c0.ring)).r, c0.range);
  };
  // Centre of a cell, in world metres.
  const centreOf = c => u.origin.add(
    vec3(float(c.x), float(c.y), float(c.z)).add(float(0.5)).mul(float(cellMetres)));

  // --- solid: 27 field samples a cell, at voxel centres spread through it ---
  const solidKernel = Fn(() => {
    const c = cellOf(instanceIndex);
    const base = u.origin.add(vec3(float(c.x), float(c.y), float(c.z))
                                .mul(float(cellMetres))).toVar();
    const count = float(0).toVar();
    const vox = cellMetres / LPV_CELL_VOXELS;
    for (const oz of [1, 3, 5]) for (const oy of [1, 3, 5]) for (const ox of [1, 3, 5]) {
      const p = base.add(vec3((ox + 0.5) * vox, (oy + 0.5) * vox, (oz + 0.5) * vox));
      count.addAssign(select(fieldAt(p).lessThan(float(0)), float(1), float(0)));
    }
    solid.element(c.i).assign(count.div(float(27)));
  })().compute(LPV_CELLS).setName(`LPV${level}.Solid`);

  // --- inject: light leaving the surfaces around each air cell ---
  //
  // Six probes along the axes, each up to one cell out. Where one lands on a
  // surface, that surface's direct light - the sun's cone trace and the whole
  // light list, the same functions the shading pass calls - times its albedo is
  // light this cell receives from that side. The terrain is boxes, so a probe
  // along an axis meets a face square on and the surface normal is -dir.
  const coneTrace = createConeTraceSunTSL(cascades);
  const pointLights = createPointLightsTSL({ cascades, lights, biasUniform: bias,
                                             fadeStartUniform: fadeStart,
                                             edgeFadeUniform: edgeFade, cards: null });
  const maxProbe = float(cellMetres);
  const sunReach = float(12);

  const injectKernel = Fn(() => {
    const c = cellOf(instanceIndex);
    If(c.i.mod(u.slices).equal(u.slice), () => {
      const out = vec3(0, 0, 0).toVar();
      const centre = centreOf(c).toVar();
      const air = solid.element(c.i).lessThan(float(SOLID_THRESHOLD))
                    .and(fieldAt(centre).greaterThan(float(0)));
      If(air, () => {
        // A loop, not six unrolled copies: each face calls the sun trace and the
        // whole light loop, and unrolling would put six of those in one kernel.
        // Faces in FACE_DIRS order: +x -x +y -y +z -z.
        Loop({ start: int(0), end: int(6), type: 'int', condition: '<', name: 'face' },
             ({ face }) => {
          const axis = face.div(int(2));
          const sgn = select(face.mod(int(2)).equal(int(0)), float(1), float(-1));
          const dir = vec3(select(axis.equal(int(0)), sgn, float(0)),
                           select(axis.equal(int(1)), sgn, float(0)),
                           select(axis.equal(int(2)), sgn, float(0))).toVar();
          const t = float(0).toVar();
          const hit = float(0).toVar();
          Loop({ start: 0, end: 24, type: 'int', condition: '<' }, () => {
            const d = fieldAt(centre.add(dir.mul(t)));
            If(d.lessThan(float(HIT_EPS)), () => { hit.assign(float(1)); Break(); });
            t.addAssign(max(d, float(MIN_STEP)).mul(c0.voxel));
            If(t.greaterThan(maxProbe), () => { Break(); });
          });
          If(hit.greaterThan(float(0)), () => {
            const p = centre.add(dir.mul(t)).toVar();
            const n = dir.negate().toVar();
            const ndotl = max(n.dot(sun.dir), float(0)).toVar();
            const sunLit = float(0).toVar();
            If(ndotl.greaterThan(float(0)).and(u.sunGain.greaterThan(float(0))), () => {
              const origin = p.add(n.mul(bias.mul(c0.voxel))).toVar();
              sunLit.assign(coneTrace(origin, sun.dir, sunReach, sun.coneRadius,
                                      fadeStart, edgeFade).x.mul(ndotl).mul(u.sunGain));
            });
            const lit = vec3(sunLit, sunLit, sunLit).add(pointLights(p, n, NO_SKIP, n));
            out.addAssign(lit.mul(u.albedo));
          });
        });
      });
      E.element(c.i).assign(vec4(out.div(float(6)), 0));
    });
  })().compute(LPV_CELLS).setName(`LPV${level}.Inject`);

  // --- propagate: one iteration, src -> dst (lpv.js propagateStep) ---
  const propagate = (src, dst) => Fn(() => {
    const c = cellOf(instanceIndex);
    const open = float(1).sub(solid.element(c.i)).toVar();
    const sum = vec3(0, 0, 0).toVar();
    for (const [dx, dy, dz] of FACE_DIRS) {
      const nb = neighbour(c, dx, dy, dz);
      const w = select(nb.inside, float(1).sub(solid.element(nb.index)), open);
      sum.addAssign(src.element(nb.index).xyz.mul(w));
    }
    const next = E.element(c.i).xyz.add(sum.mul(u.spread.div(float(6)))).mul(open);
    dst.element(c.i).assign(vec4(next, 0));
  })().compute(LPV_CELLS).setName(`LPV${level}.Propagate`);
  const aToB = propagate(A, B), bToA = propagate(B, A);

  // --- resolve: A into the texture, normalised, dilated into solid cells ---
  const resolveKernel = Fn(() => {
    const c = cellOf(instanceIndex);
    const own = A.element(c.i).xyz.toVar();
    const isSolid = solid.element(c.i).greaterThanEqual(float(SOLID_THRESHOLD));
    const outc = own.toVar();
    If(isSolid, () => {
      const sum = vec3(0, 0, 0).toVar();
      const wsum = float(0).toVar();
      for (const [dx, dy, dz] of FACE_DIRS) {
        const nb = neighbour(c, dx, dy, dz);
        const w = select(nb.inside, float(1).sub(solid.element(nb.index)), float(0));
        sum.addAssign(A.element(nb.index).xyz.mul(w));
        wsum.addAssign(w);
      }
      outc.assign(select(wsum.greaterThan(float(0)), sum.div(max(wsum, float(1e-6))), own));
    });
    // (1 - spread): see DEFAULT_LPV_SPREAD - brightness independent of reach.
    textureStore(tex, uvec3(c.x, c.y, c.z),
                 vec4(outc.mul(float(1).sub(u.spread)), 1));
  })().compute(LPV_CELLS).setName(`LPV${level}.Resolve`);

  // --- shift: when C0 re-origins, move A by whole cells (lpv.js shiftVolume) ---
  const shiftKernel = Fn(() => {
    const c = cellOf(instanceIndex);
    const sx = c.x.add(int(u.shift.x)), sy = c.y.add(int(u.shift.y)), sz = c.z.add(int(u.shift.z));
    const inside = sx.greaterThanEqual(int(0)).and(sy.greaterThanEqual(int(0)))
      .and(sz.greaterThanEqual(int(0))).and(sx.lessThan(int(D)))
      .and(sy.lessThan(int(D))).and(sz.lessThan(int(D)));
    const s = sx.add(sy.mul(int(D))).add(sz.mul(int(D * D)));
    B.element(c.i).assign(select(inside, A.element(select(inside, s, int(0))), vec4(0)));
  })().compute(LPV_CELLS).setName(`LPV${level}.Shift`);
  const copyBA = Fn(() => {
    A.element(instanceIndex).assign(B.element(instanceIndex));
  })().compute(LPV_CELLS).setName(`LPV${level}.Copy`);
  const clearA = Fn(() => {
    A.element(instanceIndex).assign(vec4(0));
    E.element(instanceIndex).assign(vec4(0));
  })().compute(LPV_CELLS).setName(`LPV${level}.Clear`);

  let frame = 0;
  // Kept on the CPU: the uniform is overwritten with 1 on a full inject.
  let sliceCount = DEFAULT_LPV_SLICES;

  return {
    tex, uniforms: u, sun, level, cellMetres,
    // What the shading pass needs to sample it: see lpvSampleTSL in gpu.js.
    binding: { tex, origin: u.origin, extent: LPV_DIM * cellMetres },

    // Injection slices: 1/N of the cells re-injected each frame.
    get slices() { return sliceCount; },
    set slices(n) { sliceCount = Math.max(1, Math.round(n)); },

    // One frame. shift is lpv.js lpvShift's answer, 'reset' for a jump the
    // volume cannot follow, or null when C0 did not move.
    update(renderer, { shift = null, iterations = 8, fullInject = false } = {}) {
      if (shift === 'reset') {
        renderer.compute(clearA);
        fullInject = true;
      } else if (shift) {
        u.shift.value.set(shift.x, shift.y, shift.z);
        renderer.compute(shiftKernel);
        renderer.compute(copyBA);
        fullInject = true;
      }
      renderer.compute(solidKernel);
      const slices = Math.max(1, sliceCount);
      // A full inject is one dispatch with a single slice covering every cell:
      // the old injection is in the wrong cells after a shift.
      u.slices.value = fullInject ? 1 : slices;
      u.slice.value = fullInject ? 0 : frame % slices;
      renderer.compute(injectKernel);
      frame++;
      // Even, so the answer always ends in A, where resolve reads it.
      const n = Math.max(2, iterations + (iterations % 2));
      for (let k = 0; k < n; k += 2) {
        renderer.compute(aToB);
        renderer.compute(bToA);
      }
      renderer.compute(resolveKernel);
    },

    dispose() {
      tex.dispose();
    }
  };
}
