
import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, float, vec3, uniform, instanceIndex, storage, texture3D,
  positionWorld, abs, sign, floor, max, texture, uv,
  attribute, varying, vec2, vec4, ivec2, textureStore, min,
  positionView, normalize, cross, clamp
} from 'three/tsl';
import { GRID_DIM, VOXEL_METRES, createBoxGridAt,
         HIT_EPS, MIN_STEP, MAX_TRACE_STEPS } from './boxgrid.js';
import { PAGE, TEXELS_PER_PAGE, FACES_PER_BLOCK } from './atlas.js';
import { SHADOW_DIM, CASCADE_COUNT, cascadeDepth, cascadeTexelMetres,
         BAYER4, SUN_ANGULAR_SIZE, readbackLayout, worldToShadow } from './csm.js';

// --- WebGPU compute plumbing and the sphere trace on the GPU ---

// The step ceiling, the hit epsilon and the minimum step all come from
// boxgrid.js rather than being restated here. They are the difference between
// the CPU reference and the shader agreeing and not, and bxb.parity() exists to
// catch exactly that - so there must be one definition, not two.

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
// 2. The distance field as a 3D texture
//
// A Data3DTexture rather than a storage buffer, because the passes that consume
// it want a sampler: the field is read at continuous positions along a ray, not
// at integer cells.
// ---------------------------------------------------------------------------
export function createDistanceTexture(grid) {
  const tex = new THREE.Data3DTexture(grid.data, GRID_DIM, GRID_DIM, GRID_DIM);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  // LINEAR, and this is the one place in the renderer where that is right.
  // Interpolating occupancy was meaningless - it smeared solid into empty. A
  // distance field interpolates exactly: between a solid voxel centre at -0.5
  // and its air neighbour at +0.5, the zero-crossing lands on the shared face,
  // which is where the surface is. It is also what gives a slope's zero-crossing
  // a sub-voxel position. Radiance stays nearest; only occlusion is smooth.
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

export function updateDistanceTexture(tex) {
  tex.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// 3. Sphere tracing in TSL
//
// The port of sphereTrace() in boxgrid.js, and the technique the design doc
// specifies: step by the distance stored at the current point, which is the
// radius of a sphere guaranteed to be empty, so the step can never overshoot a
// surface. The binary Amanatides-Woo DDA it replaced is still in boxgrid.js as
// an independent cross-check - two unrelated traversals agreeing on the same
// geometry is worth more than either one being carefully reviewed.
//
// Why this is what slopes need: the DDA could only ever answer "is this VOXEL
// solid", so an occluder's edge was pinned to a voxel boundary. Sampling the
// field linearly puts the zero-crossing wherever the bake put it, including
// partway through a voxel, which is the whole mechanism behind slopes and
// stairs arriving without sub-blocks.
//
// The decode is one multiply-add. An R8 texture samples normalised to [0,1] and
// the field is encoded over [-1,+1] voxels, so distance = sample * 2 - 1. The
// normalisation that silently broke the binary version - where an occupied
// voxel arrived as 1/255 and never cleared a 0.5 threshold - cannot recur here,
// because the full byte range is now meaningful rather than two values in it.
//
// Returns 1.0 when the ray reached a surface before maxDist, 0.0 otherwise.
// ---------------------------------------------------------------------------
export const traceDistanceTSL = Fn(([sdfTex, gridOrigin, origin, dir, maxDist]) => {
  const dim = float(GRID_DIM);
  const vs = float(VOXEL_METRES);
  const t = float(0).toVar();
  const hit = float(0).toVar();

  Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
    // Continuous voxel-space position of the current point. No cell index and
    // no per-axis boundary bookkeeping - that was all the DDA needed.
    const vp = origin.add(dir.mul(t)).sub(gridOrigin).div(vs);

    // Leaving the grid means open sky for this cascade. Until cascades exist
    // this is also where a long shadow truncates, which is what the directional
    // sun's CSM is for.
    If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
       .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
       .or(vp.z.greaterThanEqual(dim)), () => {
      Break();
    });

    const d = texture3D(sdfTex, vp.div(dim)).r.mul(float(2)).sub(float(1));
    If(d.lessThan(float(HIT_EPS)), () => {
      hit.assign(float(1));
      Break();
    });

    // MIN_STEP is the floor that guarantees progress. The field is clamped to
    // one voxel, so a ray running parallel to a wall sits at a small constant
    // distance and would otherwise inch forward until the step cap. Running out
    // of steps reports no-hit, which resolves a grazing ray as lit - the
    // conservative answer, and the one nobody can see.
    t.addAssign(max(d, float(MIN_STEP)).mul(vs));
    If(t.greaterThan(maxDist), () => { Break(); });
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
      traceDistanceTSL(occTex, gridOrigin, o.xyz, d.xyz, d.w)
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
// The sun is DIRECTIONAL: every surface marches along one shared direction for
// a capped distance, not toward a point. Treating it as a point light at
// (10,20,10) meant each pixel marched ~160 voxels to reach it, which is what
// killed the framerate, and put the light nearly overhead so every shadow hid
// underneath the block casting it.
//
// Capping the march is the perf lever. At 12.5cm voxels a 12m cap is ~96 steps
// worst case, and anything further away contributes a shadow too faint to
// matter at this art scale.
export function createShadowColorNode({ occTex, grid, sunDirection, maxDistance = 12, ambient = 0.35 }) {
  const gridOrigin = uniform(new THREE.Vector3(grid.origin.x, grid.origin.y, grid.origin.z));
  // Held as a uniform whose .value is the live vector, so moving the sun is a
  // uniform write rather than a material rebuild.
  const sunDir = uniform(sunDirection.clone().normalize());
  const maxDist = uniform(float(maxDistance));

  const node = Fn(() => {
    // Half a voxel along the ray: the fragment sits exactly on a voxel face and
    // floating point can place it either side, which would self-shadow.
    const start = positionWorld.add(sunDir.mul(float(VOXEL_METRES * 0.5)));
    const shadowed = traceDistanceTSL(occTex, gridOrigin, start, sunDir, maxDist);
    return float(1).sub(shadowed.mul(float(1 - ambient)));
  })();

  return { node, sunDirUniform: sunDir, gridOriginUniform: gridOrigin, maxDistUniform: maxDist };
}

// The occupancy grid covers 12x12 blocks, so it has to travel with the player
// or shading only works where it happened to be built. Repopulating in place
// and rewriting the origin uniform is enough - the texture object and the
// material are unchanged, so nothing rebuilds.
export function followShadowGrid(grid, tex, gridOriginUniform, originBlock) {
  createBoxGridAt(originBlock.x, originBlock.z, grid);
  gridOriginUniform.value.set(grid.origin.x, grid.origin.y, grid.origin.z);
  tex.needsUpdate = true;
  return grid;
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

// ---------------------------------------------------------------------------
// 6. Phase C: the texel atlas
//
// Same march, different place to stand. Instead of one ray per screen pixel in
// a fragment shader, one ray per TEXEL in a compute pass that writes into an
// atlas; the fragment shader then costs a single nearest fetch. atlas.js owns
// the addressing and the geometry - see its header for why this is the fork the
// architecture turns on.
//
// RGBA8 rather than R8: r8unorm is not a guaranteed storage-texture format in
// WebGPU (it needs texture-formats-tier1), and the spare channels are where
// coloured radiance goes in the next phase, so the width is not wasted.
// ---------------------------------------------------------------------------
export function createTexelAtlas(layout) {
  const tex = new THREE.StorageTexture(layout.width, layout.height);
  // Nearest is the entire point: a shadow edge must land on a texel boundary,
  // not be smeared across one. Bilinear here would undo the art direction.
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.generateMipmaps = false;
  return tex;
}

// Pages are addressed arithmetically, so pages belonging to faces that are
// never shaded (buried interiors) exist but are never written, and a storage
// texture starts with undefined contents. One clear to full brightness means an
// unshaded page reads as plain lit albedo instead of whatever was in memory.
export async function clearTexelAtlas(renderer, atlas, layout) {
  const w = layout.width;
  const kernel = Fn(() => {
    const x = instanceIndex.mod(w);
    const y = instanceIndex.div(w);
    textureStore(atlas, ivec2(x, y), vec4(1, 1, 1, 1));
  })().compute(layout.width * layout.height);
  await renderer.computeAsync(kernel);
  return layout.width * layout.height;
}

// One invocation per texel of every job. The job buffers carry a per-face basis
// computed on the CPU (texel (0,0) centre, plus the two per-texel steps), which
// is why there is no branch on face index anywhere in here.
export function createAtlasShadePass({ atlas, occTex, grid, jobs, layout,
                                       sunDirection, maxDistance = 12, ambient = 0.35,
                                       capacity = jobs.count, sunShadow = null,
                                       shadowOnly = false }) {
  // Sized with headroom so walking into denser geometry rewrites the buffers
  // instead of rebuilding the kernel, which would mean a shader lookup and a
  // visible hitch every few steps.
  const n = Math.max(1, capacity, jobs.count);
  const originBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);
  const uBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);
  const vBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);

  const origins = storage(originBuf, 'vec4', n);
  const uSteps = storage(uBuf, 'vec4', n);
  const vSteps = storage(vBuf, 'vec4', n);

  const gridOrigin = uniform(new THREE.Vector3(grid.origin.x, grid.origin.y, grid.origin.z));
  const sunDir = uniform(sunDirection.clone().normalize());
  const maxDist = uniform(float(maxDistance));
  const pagesX = float(layout.pagesX);

  const kernel = Fn(() => {
    const jobIdx = instanceIndex.div(TEXELS_PER_PAGE);
    const texelIdx = instanceIndex.mod(TEXELS_PER_PAGE);
    const tx = texelIdx.mod(PAGE).toFloat();
    const ty = texelIdx.div(PAGE).toFloat();

    const o = origins.element(jobIdx);
    const us = uSteps.element(jobIdx).xyz;
    const vs = vSteps.element(jobIdx).xyz;
    const pos = o.xyz.add(us.mul(tx)).add(vs.mul(ty));

    // The face normal, for free: the two texel steps are orthogonal and lie in
    // the face, and atlas.js orders them so their cross product points outward
    // on all six faces. Checked per face in the tests, because a flipped normal
    // would light the inside of the world.
    const n = normalize(cross(us, vs));

    // Lambert. Without it an unshadowed wall facing away from the sun reads as
    // bright as one facing it, which is what made the first atlas pass look flat
    // even where the shadows were correct.
    const ndotl = max(n.dot(sunDir), float(0));

    // Either technique returns visibility, 1 lit and 0 occluded, so they are
    // interchangeable here - which is the point of doing this in texel space.
    const visible = sunShadow
      ? sunShadow(pos, n)
      : float(1).sub(traceDistanceTSL(occTex, gridOrigin, pos, sunDir, maxDist));
    // shadowOnly writes the raw visibility term. Ambient and N.L both compress
    // the shadowed range toward the middle, so acne that is plain here is
    // invisible in the shaded result - which is why it has been hard to name.
    const lit = shadowOnly
      ? visible
      : float(ambient).add(visible.mul(ndotl).mul(float(1 - ambient)));

    // o.w is the page index; unpack it to the page's texel origin in the atlas.
    const px = o.w.mod(pagesX);
    const py = o.w.div(pagesX).floor();
    const coord = ivec2(px.mul(float(PAGE)).add(tx).toInt(),
                        py.mul(float(PAGE)).add(ty).toInt());
    textureStore(atlas, coord, vec4(lit, lit, lit, 1));
  })().compute(n * TEXELS_PER_PAGE);

  const pass = { kernel, sunDirUniform: sunDir, gridOriginUniform: gridOrigin,
                 maxDistUniform: maxDist, buffers: { originBuf, uBuf, vBuf },
                 jobCount: n, capacity: n };
  updateShadeJobs(pass, jobs);
  return pass;
}

// Job buffers are rewritten in place when the footprint moves, so the kernel,
// its bindings and the material all survive. Only valid while the new job count
// is no larger than the buffers the pass was built with - runAtlasShade reports
// false when it is not, and main.js rebuilds the pass.
export function updateShadeJobs(pass, jobs) {
  if (jobs.count > pass.jobCount) return false;
  pass.buffers.originBuf.array.set(jobs.origins);
  pass.buffers.uBuf.array.set(jobs.uSteps);
  pass.buffers.vBuf.array.set(jobs.vSteps);
  pass.buffers.originBuf.needsUpdate = true;
  pass.buffers.uBuf.needsUpdate = true;
  pass.buffers.vBuf.needsUpdate = true;
  // Retiring the slots the new footprint does not use. Two things make a dead
  // slot free rather than merely harmless: a negative page index puts the store
  // outside the atlas, where WGSL discards it, and an origin far outside the
  // occupancy grid makes the march break on its first iteration. So the pass is
  // always dispatched at full capacity and the surplus costs nothing, which is
  // what lets the footprint move without rebuilding the kernel.
  const a = pass.buffers.originBuf.array;
  for (let i = jobs.count; i < pass.jobCount; i++) {
    a[i * 4] = 1e6; a[i * 4 + 1] = 1e6; a[i * 4 + 2] = 1e6; a[i * 4 + 3] = -1;
  }
  return true;
}

// The whole point of the atlas: this does NOT run every frame. It runs when the
// light moves or the footprint moves, and the render pass reads the result for
// however many frames follow.
export async function runAtlasShade(renderer, pass) {
  await renderer.computeAsync(pass.kernel);
}

// ---------------------------------------------------------------------------
// The read side.
//
// Every value here has to be the exact mirror of what the compute pass wrote,
// and the two only meet on screen - so the mapping lives in atlas.js and is
// checked headlessly, and this is the transcription of it.
// ---------------------------------------------------------------------------
export function createAtlasColorNode({ atlas, layout }) {
  // instance_index is a vertex-stage builtin in WGSL, so the page address has
  // to be resolved in the vertex shader and carried across. It is constant over
  // a face, so interpolating it is exact rather than merely close.
  const pageIndex = instanceIndex.toFloat().mul(float(FACES_PER_BLOCK))
                      .add(attribute('faceIndex', 'float'));
  const pagesX = float(layout.pagesX);
  const pageOrigin = varying(vec2(
    pageIndex.mod(pagesX).mul(float(PAGE)),
    pageIndex.div(pagesX).floor().mul(float(PAGE))
  ));

  // floor() must happen AFTER interpolation, or every fragment on the face
  // would share whichever texel the vertex happened to land in.
  const t = min(floor(uv().mul(float(PAGE))), float(PAGE - 1));
  const atlasUV = pageOrigin.add(t).add(float(0.5))
                    .div(vec2(layout.width, layout.height));
  return texture(atlas, atlasUV).r;
}

// The terrain is one InstancedMesh, so its BoxGeometry's UVs are shared by every
// block; what distinguishes one face's page from another is instance_index plus
// this attribute. BoxGeometry emits 6 faces of 4 vertices in the order +X, -X,
// +Y, -Y, +Z, -Z, which is the order atlas.js FACE_NORMALS is written in.
export function addFaceIndexAttribute(geometry) {
  const verts = geometry.attributes.position.count;
  const per = verts / FACES_PER_BLOCK;
  const a = new Float32Array(verts);
  for (let i = 0; i < verts; i++) a[i] = Math.floor(i / per);
  geometry.setAttribute('faceIndex', new THREE.BufferAttribute(a, 1));
  return geometry;
}

// ---------------------------------------------------------------------------
// 7. Cascaded shadow map for the sun
//
// Three render targets, a depth-only override material, and a sampler used by
// the atlas pass. See csm.js for why the resolution is locked to the texel grid
// and why mixing a depth map with ray-marched local lights does not show.
//
// The caster set is an explicit list of proxies in a dedicated scene - see
// createShadowScene below. Terrain today, models when they land; camera-facing
// billboards stay out, since one would otherwise cast a shadow that swings with
// the camera.
// ---------------------------------------------------------------------------

export function createCascadeTargets() {
  const targets = [];
  for (let i = 0; i < CASCADE_COUNT; i++) {
    const rt = new THREE.RenderTarget(SHADOW_DIM, SHADOW_DIM, {
      // HALF float, not full. r32float is not a filterable texture format in
      // core WebGPU, so binding it to a sampler can fail validation even with
      // nearest filtering, and it is not blendable either. r16float is both.
      // Precision is ample: half float resolves ~1/2048 near 1.0, which over
      // the 54 m depth range is about 2.6 cm - far inside the 19 cm
      // normal-offset bias, so it cannot cause acne. Lateral resolution is the
      // scarce resource here and that is pinned to the texel grid on purpose.
      type: THREE.HalfFloatType,
      format: THREE.RedFormat,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      depthBuffer: true,
      generateMipmaps: false
    });
    rt.texture.name = 'csm' + i;
    targets.push(rt);
  }
  return targets;
}

// Writes linear distance along the light view direction, normalised over the
// cascade depth range. Deliberately the same expression the sampler uses, so the
// two cannot drift apart: an orthographic camera at the cascade origin with
// near = 0 makes -positionView.z equal dot(p - camPos, forward).
// ONE MATERIAL PER CASCADE, not one shared material whose `far` is rewritten
// between renders. Three renders happen in a single frame, and a uniform mutated
// between them is only guaranteed to reach the GPU per draw if the backend
// flushes it there. If it does not, C1 and C2 store depths normalised by C0's
// range while the lookup divides by their own - a systematic offset that reads
// as a large flat region wrongly in shadow, and looks nothing like a uniform
// buffer problem. A material per cascade costs three tiny pipelines and removes
// the question entirely.
export function createLightDepthMaterials() {
  const mats = [];
  for (let i = 0; i < CASCADE_COUNT; i++) mats.push(createLightDepthMaterial());
  return mats;
}

export function createLightDepthMaterial() {
  const farUniform = uniform(float(1));
  const material = new THREE.MeshBasicNodeMaterial();
  material.colorNode = positionView.z.negate().div(farUniform);
  // Thin geometry has to occlude from both sides, and a shadow caster has no
  // meaningful front face.
  material.side = THREE.DoubleSide;
  // NoBlending is mandatory, not a micro-optimisation: R32Float is not a
  // blendable format in WebGPU, so leaving the default normal blending on makes
  // the render pipeline itself invalid and the depth pass never runs. Blending a
  // depth would be meaningless anyway - the nearest caster wins, which is what
  // the depth test already decides.
  material.blending = THREE.NoBlending;
  material.transparent = false;
  return { material, farUniform };
}

export function createCascadeCameras() {
  const cams = [];
  for (let i = 0; i < CASCADE_COUNT; i++) {
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    cams.push(cam);
  }
  return cams;
}

// One orthographic camera per cascade, rebuilt from the fit whenever the sun or
// the view centre moves. The UV transform handed to the shader is the camera's
// OWN projection-view matrix rather than a basis reconstructed by hand - that is
// the difference between a shadow map that lands correctly and one that comes
// out mirrored at some sun angles, and it is not worth being clever about.
export function updateCascadeCameras(cascades, cameras) {
  for (let i = 0; i < cascades.length; i++) {
    const c = cascades[i];
    const cam = cameras[i];
    cam.left = -c.halfExtent; cam.right = c.halfExtent;
    cam.top = c.halfExtent; cam.bottom = -c.halfExtent;
    cam.near = 0; cam.far = c.depth;
    cam.position.set(c.position.x, c.position.y, c.position.z);
    cam.up.set(c.up.x, c.up.y, c.up.z);
    cam.lookAt(c.position.x + c.forward.x, c.position.y + c.forward.y,
               c.position.z + c.forward.z);
    cam.updateMatrixWorld(true);
    cam.updateProjectionMatrix();
  }
}

// A dedicated scene holding proxies of the casters, rather than the main scene
// with an override material and a layer mask.
//
// That earlier approach stacked two things that each looked reasonable and could
// not be told apart when the result was a blank map: whether WebGPURenderer
// honours scene.overrideMaterial for a NodeMaterial, and whether the layer mask
// was reaching the instanced terrain. Neither is worth depending on. A proxy
// InstancedMesh SHARES the terrain geometry and its instanceMatrix buffer, so
// this costs one draw call and no memory, the caster set is an explicit list
// instead of a mask, and the depth material is bound directly where it cannot be
// silently ignored.
export function createShadowScene() {
  return { scene: new THREE.Scene(), proxies: new Map() };
}

// `source` may be an InstancedMesh or a plain Mesh. The proxy shares geometry
// and, for instanced sources, the very same instanceMatrix attribute - so moving
// or re-tinting terrain needs no sync step here.
export function addShadowCaster(shadowScene, source, depthMaterial) {
  if (shadowScene.proxies.has(source)) return shadowScene.proxies.get(source);

  let proxy;
  if (source.isInstancedMesh) {
    proxy = new THREE.InstancedMesh(source.geometry, depthMaterial, source.count);
    proxy.instanceMatrix = source.instanceMatrix;
  } else {
    proxy = new THREE.Mesh(source.geometry, depthMaterial);
  }
  proxy.frustumCulled = false;
  proxy.matrixAutoUpdate = false;
  proxy.matrix.copy(source.matrixWorld);
  proxy.matrixWorld.copy(source.matrixWorld);
  shadowScene.proxies.set(source, proxy);
  shadowScene.scene.add(proxy);
  return proxy;
}

export function syncShadowCasters(shadowScene) {
  for (const [source, proxy] of shadowScene.proxies) {
    if (source.isInstancedMesh) proxy.count = source.count;
    proxy.matrixWorld.copy(source.matrixWorld);
    proxy.visible = source.visible !== false;
  }
}

export function renderCascades(renderer, shadowScene, cameras, targets, depth) {
  const prevTarget = renderer.getRenderTarget();
  const prevClear = renderer.getClearColor(new THREE.Color());
  const prevAlpha = renderer.getClearAlpha();

  syncShadowCasters(shadowScene);
  // A texel the depth pass never rasterised must read as "further away than
  // anything", not as zero - zero is the near plane and would shadow everything
  // behind it.
  renderer.setClearColor(0xffffff, 1);
  renderer.autoClear = true;

  // `depth` is the array of per-cascade materials. The proxies' material is
  // swapped rather than duplicating the proxy meshes, so this stays one draw
  // call per cascade over shared geometry.
  for (let i = 0; i < cameras.length; i++) {
    depth[i].farUniform.value = cameras[i].far;
    for (const proxy of shadowScene.proxies.values()) proxy.material = depth[i].material;
    renderer.setRenderTarget(targets[i]);
    renderer.clear();
    renderer.render(shadowScene.scene, cameras[i]);
  }

  renderer.setRenderTarget(prevTarget);
  renderer.setClearColor(prevClear, prevAlpha);
}

// Per-cascade uniforms, held live so moving the sun is a uniform write plus a
// depth re-render rather than a kernel rebuild.
export function createCascadeUniforms() {
  const u = [];
  for (let i = 0; i < CASCADE_COUNT; i++) {
    u.push({
      projView: uniform(new THREE.Matrix4()),
      camPos: uniform(new THREE.Vector3()),
      forward: uniform(new THREE.Vector3()),
      right: uniform(new THREE.Vector3()),
      up: uniform(new THREE.Vector3()),
      depth: uniform(float(1)),
      texel: uniform(float(cascadeTexelMetres(i)))
    });
  }
  return u;
}

const _pv = new THREE.Matrix4();
export function writeCascadeUniforms(uniforms, cascades, cameras) {
  for (let i = 0; i < cascades.length; i++) {
    _pv.multiplyMatrices(cameras[i].projectionMatrix, cameras[i].matrixWorldInverse);
    uniforms[i].projView.value.copy(_pv);
    uniforms[i].camPos.value.set(cascades[i].position.x, cascades[i].position.y,
                                 cascades[i].position.z);
    uniforms[i].forward.value.set(cascades[i].forward.x, cascades[i].forward.y,
                                  cascades[i].forward.z);
    uniforms[i].right.value.set(cascades[i].right.x, cascades[i].right.y,
                                cascades[i].right.z);
    uniforms[i].up.value.set(cascades[i].up.x, cascades[i].up.y, cascades[i].up.z);
    uniforms[i].depth.value = cascades[i].depth;
    uniforms[i].texel.value = cascades[i].texel;
  }
}

// The 4x4 Bayer matrix as a tiny texture rather than shader arithmetic. The
// bit-interleave formula for it is easy to get subtly wrong and impossible to
// spot by eye in a dither pattern, whereas a 16-byte lookup is exact and the
// values are checked entry by entry in the tests. Repeat wrapping makes the
// lookup work for any integer texel coordinate with no modulo in the shader.
// WEBGPU V IS FLIPPED. A WebGPU texture's row 0 is its TOP row, while NDC y =
// +1 is also the top, so mapping clip space to texture coordinates needs
// v = 1 - (ndc.y * 0.5 + 0.5) and not the WebGL-convention v = ndc.y * 0.5 + 0.5.
// three's own ShadowNode does exactly this - `shadowCoord.y.oneMinus()`, commented
// "follow webgpu standards". Without it the depth lookup is mirrored vertically
// across the map, so every shadow is fetched from the wrong side of the light's
// view and lands as a large misplaced dark region rather than as a wrong-looking
// edge. Stated once here because it applies identically to all three lookups.
function clipToShadowUV(clip) {
  return vec2(clip.x.mul(float(0.5)).add(float(0.5)),
              float(0.5).sub(clip.y.mul(float(0.5))));
}

export function createBayerTexture() {
  const tex = new THREE.DataTexture(
    new Uint8Array(BAYER4.map(v => v * 16)), 4, 4, THREE.RedFormat);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

// --- Distance-based softening ---
//
// The replacement for PCF against a depth map, and the same rule the
// sphere-traced local lights follow (doc 6.1): the sample footprint widens with
// how far the light travelled between occluder and receiver, because that is how
// a penumbra physically grows. Contact stays hard; a shadow cast from far away
// spreads. No separate blur pass, no PCSS-style distance-to-blocker search
// beyond the cheap estimate below.
//
// A fixed 4x4 tap grid whose SPACING scales with the estimated penumbra. Fixed
// count keeps the kernel uniform-cost and fully unrolled; when the penumbra is
// small the taps collapse onto a single texel and the result is a hard binary
// edge for free, with no branch to write.
const PCF_TAPS = 4;             // per axis, so 16 samples
const BLOCKER_TAPS = 3;         // per axis, for the penumbra estimate
const MIN_SPACING = 0.5;        // shadow texels
const MAX_SPACING = 6.0;
// Normal-offset bias, in shadow texels. Offsetting along the face normal rather
// than biasing depth is possible here precisely because the atlas job knows the
// exact face normal, so this removes acne without the peter-panning that a depth
// bias large enough to work would cause.
export const NORMAL_BIAS_TEXELS = 0.5;

// The receiver plane's depth gradient across the map, per shadow TEXTURE texel,
// and the correction from an arbitrary point onto the texel centre.
//
// This is what replaces a guessed bias. A depth map rasterised with nearest
// filtering stores the depth at the TEXEL CENTRE, but the receiver sits
// somewhere inside that texel, so the comparison is between two different points
// on the surface. Under a 23 degree sun the ground moves 0.295 m of depth per
// 12.5 cm texel, so that mismatch is +/-0.147 m - the same order as any bias
// worth setting, which is why a constant one flips between acne and peter-panning
// as the cascade slides and the sub-texel phase changes under it.
//
// The receiver is a plane through p with normal n, so the gradient is exact
// rather than estimated: a point p + a*right + b*up stays on the plane only at
// c = -(a*(right.n) + b*(up.n)) / (forward.n). Correcting onto the texel centre
// makes the comparison exact for any planar surface - which every block face is
// - leaving only half-float quantisation to absorb. Atlas shading is what makes
// this available at all: the job already carries the exact face normal, where a
// screen-space pass would have to reconstruct it from derivatives.
//
// gv is NEGATED relative to gu because WebGPU's texture v runs opposite the
// light's up axis (see clipToShadowUV). forward.n goes to zero as a face turns
// edge-on to the light, which would blow the gradient up; those faces have N.L
// of zero and contribute nothing, so clamping the denominator is safe.
function receiverPlane(suv, right, up, forward, n, texelSize, depthRange) {
  const nf = min(forward.dot(n), float(-0.05));
  const gu = right.dot(n).div(nf).negate().mul(texelSize).div(depthRange).toVar();
  const gv = up.dot(n).div(nf).mul(texelSize).div(depthRange).toVar();
  const st = suv.mul(float(SHADOW_DIM));
  const frac = st.sub(st.floor().add(float(0.5)));
  return { gu, gv, toCentre: gu.mul(frac.x.negate()).add(gv.mul(frac.y.negate())) };
}

// Soft shadow from one cascade. Returns 1 lit, 0 fully shadowed.
const cascadeShadowTSL = Fn(([tex, projView, camPos, right, up, forward,
                              depthRange, texelSize, p, n, angular, bias]) => {
  // A small normal offset still helps with the half-float quantisation, but it
  // is no longer what prevents acne - see the gradient below.
  const biased = p.add(n.mul(texelSize.mul(bias)));

  const clip = projView.mul(vec4(biased, float(1)));
  const suv = clipToShadowUV(clip).toVar();
  // Linear depth along the light, the same expression the depth pass wrote.
  const receiver = biased.sub(camPos).dot(forward).div(depthRange).toVar();
  const inv = float(1 / SHADOW_DIM);

  // RECEIVER-PLANE DEPTH BIAS.
  //
  // Every tap in the kernel lands somewhere else on the map, where the receiving
  // surface itself is at a different depth. Comparing all of them against the
  // depth under the ORIGINAL point is what caused the acne: on ground under a
  // 20 degree sun the surface moves ~34 cm in depth per shadow texel, so a tap
  // six texels out was 2 m off and read as occluded. A constant bias cannot fix
  // that, because the error grows with the kernel - and the distance-based
  // softening exists precisely to grow the kernel.
  //
  // The receiver is a plane through p with normal n, so its depth gradient
  // across the map is exact rather than estimated: a point p + a*right + b*up
  // stays on the plane only at c = -(a*(right.n) + b*(up.n)) / (forward.n).
  // Offsetting each tap by that removes planar self-shadowing at any kernel
  // size. Atlas shading is what makes this available at all - a screen-space
  // pass would have to reconstruct the normal from derivatives, whereas here the
  // job already carries the exact face normal.
  //
  // forward.n goes to zero as the face turns edge-on to the light, which would
  // blow the gradient up; those faces have N.L of zero and contribute nothing,
  // so clamping the denominator is safe and bounds the whole expression.
  //
  // gv IS NEGATED relative to gu. The taps step in TEXTURE space, and WebGPU's v
  // runs opposite the light's up axis (see clipToShadowUV), so a +oy texture
  // offset is a step along -up in the world. Pairing an un-negated gradient with
  // a flipped offset axis tilts the bias plane the wrong way along v, which
  // reintroduces exactly the planar acne this bias exists to remove - and only
  // on one axis, which makes it look like a directional artifact rather than a
  // sign error.
  const rp = receiverPlane(suv, right, up, forward, n, texelSize, depthRange);
  const gu = rp.gu, gv = rp.gv;
  // Snap the receiver onto the texel centre before any tap, so every tap in the
  // kernel compares like with like.
  receiver.addAssign(rp.toCentre);

  // Blocker search: the average depth of whatever is genuinely in front of the
  // receiving plane is what sets the penumbra width.
  const blockerSum = float(0).toVar();
  const blockerCount = float(0).toVar();
  Loop({ start: 0, end: BLOCKER_TAPS, type: 'int', condition: '<' }, ({ i }) => {
    Loop({ start: 0, end: BLOCKER_TAPS, type: 'int', condition: '<' }, ({ i: j }) => {
      const ox = i.toFloat().sub(float((BLOCKER_TAPS - 1) / 2)).mul(float(2));
      const oy = j.toFloat().sub(float((BLOCKER_TAPS - 1) / 2)).mul(float(2));
      const expected = receiver.add(gu.mul(ox)).add(gv.mul(oy));
      const d = texture(tex, suv.add(vec2(ox, oy).mul(inv))).r;
      If(d.lessThan(expected), () => {
        blockerSum.addAssign(d);
        blockerCount.addAssign(float(1));
      });
    });
  });

  // Nothing in front means lit, and no filtering to do.
  const lit = float(1).toVar();
  If(blockerCount.greaterThan(float(0)), () => {
    const blocker = blockerSum.div(blockerCount);
    // Separation in metres, then the penumbra in SHADOW TEXELS. Dividing by the
    // cascade texel size is what makes the same physical penumbra a smaller
    // kernel in a coarser cascade - the lockstep doc 7.3 asks for, and cheaper
    // exactly where detail matters least.
    const separation = receiver.sub(blocker).max(float(0)).mul(depthRange);
    const spacing = clamp(separation.mul(angular).div(texelSize),
                          float(MIN_SPACING), float(MAX_SPACING));

    const occluded = float(0).toVar();
    Loop({ start: 0, end: PCF_TAPS, type: 'int', condition: '<' }, ({ i }) => {
      Loop({ start: 0, end: PCF_TAPS, type: 'int', condition: '<' }, ({ i: j }) => {
        const ox = i.toFloat().sub(float((PCF_TAPS - 1) / 2)).mul(spacing);
        const oy = j.toFloat().sub(float((PCF_TAPS - 1) / 2)).mul(spacing);
        const expected = receiver.add(gu.mul(ox)).add(gv.mul(oy));
        const d = texture(tex, suv.add(vec2(ox, oy).mul(inv))).r;
        If(d.lessThan(expected), () => { occluded.addAssign(float(1)); });
      });
    });
    lit.assign(float(1).sub(occluded.div(float(PCF_TAPS * PCF_TAPS))));
  });

  return lit;
});

// The plain comparison: one tap, receiver-plane biased, nothing else. This is
// the shadow to get right BEFORE the soft one, because every artifact the soft
// path can produce it can produce too, only harder to attribute - a widening
// kernel turns a small bias error into a large dark region, which is what the
// first pass at this looked like.
const cascadeShadowHardTSL = Fn(([tex, projView, camPos, right, up, forward,
                                  depthRange, texelSize, p, n, bias,
                                  slopeScale]) => {
  const biased = p.add(n.mul(texelSize.mul(bias)));
  const clip = projView.mul(vec4(biased, float(1)));
  const suv = clipToShadowUV(clip);
  const receiver = biased.sub(camPos).dot(forward).div(depthRange);
  const d = texture(tex, suv).r;
  // Correct the receiver onto the texel centre the map actually stores, which
  // makes the comparison EXACT for a planar surface rather than approximately
  // right with a slop term. What is left to absorb is only half-float
  // quantisation - which is relative, so a constant in NORMALISED depth covers
  // every cascade: 0.001 is a few half-float steps near the middle of the range,
  // and slopeScale tunes it without touching the geometry.
  const rp = receiverPlane(suv, right, up, forward, n, texelSize, depthRange);
  const expected = receiver.add(rp.toCentre);
  const eps = slopeScale.mul(float(0.001));
  return d.lessThan(expected.sub(eps)).select(float(0), float(1));
});

// The comparison, unthresholded. Returns (receiver - stored) * scale + 0.5, so
// mid grey is a perfect match, darker means the receiver sits behind what the
// map stored (which is what produces a shadow) and brighter means in front.
//
// This is the measurement that separates the two explanations for a large wrong
// shadow: a bias that is merely too small gives a diff a few thousandths either
// side of 0.5, whereas a transform or normalisation error gives a large constant
// offset. Thresholded to lit/shadowed those look identical, which is why tuning
// bias against the symptom has been going in circles.
const cascadeDiffTSL = Fn(([tex, projView, camPos, forward, depthRange, texelSize,
                            p, n, bias, scale]) => {
  const biased = p.add(n.mul(texelSize.mul(bias)));
  const clip = projView.mul(vec4(biased, float(1)));
  const suv = clipToShadowUV(clip);
  const receiver = biased.sub(camPos).dot(forward).div(depthRange);
  const d = texture(tex, suv).r;
  return clamp(receiver.sub(d).mul(scale).add(float(0.5)), float(0), float(1));
});

// Full CSM lookup: pick the finest cascade containing the point with room for
// the kernel, filter in it, then quantise and dither the soft tail.
//
// A point beyond every cascade reads as lit - open sky - which is the same
// convention a ray leaving the grid follows, so the two techniques degrade the
// same way at their outer limit.
export function createCsmShadowTSL(textures, uniforms, angularUniform, bayerTex,
                                   stepsUniform, soft = false,
                                   biasUniform = null, slopeUniform = null,
                                   levelDebug = false, diffUniform = null) {
  const bias = biasUniform || uniform(float(NORMAL_BIAS_TEXELS));
  const slopeScale = slopeUniform || uniform(float(0.75));
  return Fn(([p, n]) => {
    const result = float(1).toVar();
    const done = float(0).toVar();

    // Unrolled on the host: the cascade count is a compile-time constant and a
    // texture binding cannot be selected by a runtime index anyway.
    for (let i = 0; i < CASCADE_COUNT; i++) {
      const u = uniforms[i];
      const clip = u.projView.mul(vec4(p, float(1)));
      const suv = clipToShadowUV(clip);
      // Margin wide enough that the kernel still samples inside the map - and
      // NO WIDER. The hard path takes one tap, so it needs about a texel; using
      // the soft path's worst-case kernel for it cost 18% of every edge, which
      // pushed near-screen ground out of C0 and into cascades four times
      // coarser than it needed. Acne is worst in the coarse cascades, so an
      // over-wide margin shows up as banding rather than as anything obviously
      // to do with cascade selection.
      const m = float(((soft ? PCF_TAPS * MAX_SPACING : 1) + 2) / SHADOW_DIM);
      const inside = suv.x.greaterThan(m).and(suv.x.lessThan(float(1).sub(m)))
        .and(suv.y.greaterThan(m)).and(suv.y.lessThan(float(1).sub(m)))
        .and(clip.z.greaterThan(float(0))).and(clip.z.lessThan(float(1)));

      If(done.equal(float(0)).and(inside), () => {
        result.assign(soft
          ? cascadeShadowTSL(textures[i], u.projView, u.camPos, u.right, u.up,
                             u.forward, u.depth, u.texel, p, n, angularUniform,
                             bias)
          : cascadeShadowHardTSL(textures[i], u.projView, u.camPos, u.right, u.up,
                                 u.forward, u.depth, u.texel, p, n,
                                 bias, slopeScale));
        if (levelDebug) result.assign(float((i + 1) / (CASCADE_COUNT + 1)));
        if (diffUniform) {
          result.assign(cascadeDiffTSL(textures[i], u.projView, u.camPos,
                                       u.forward, u.depth, u.texel, p, n,
                                       bias, diffUniform));
        }
        done.assign(float(1));
      });
    }

    // Quantise the soft tail into discrete steps rather than leaving a smooth
    // gradient: under flat-shaded pixel art a gradient reads as a rendering bug,
    // where stepped and dithered reads as intentional (doc 6.1).
    //
    // The dither is indexed by INTEGER WORLD TEXEL, not screen position. Screen
    // dithering crawls across surfaces under camera motion; anchored to the
    // world it is painted onto them - the same argument as the atlas itself. The
    // world texel is used rather than the page texel so that a page moving
    // within the atlas cannot change the pattern on it.
    // A hard shadow is already two values, so there is no tail to quantise and
    // nothing for the dither to break up - applying it anyway would only punch
    // holes in a binary edge.
    if (levelDebug || diffUniform || !soft) return result;

    const wt = p.div(float(VOXEL_METRES)).floor();
    const dither = texture(bayerTex, wt.xz.add(vec2(0, wt.y)).mul(float(0.25))).r;
    return clamp(result.mul(stepsUniform).add(dither).floor().div(stepsUniform),
                 float(0), float(1));
  });
}

// ---------------------------------------------------------------------------
// 8. Looking at a cascade
//
// Two independent ways to look, on purpose, because the first version of this
// had ONE way and it lied. A readback viewer reported every texel as zero and
// the reason had nothing to do with the depth pass: WebGPU pads readback rows to
// 256 bytes, so a 144-wide R16 map comes back with a 512-byte (256-element) row
// stride, and deriving the stride as length/(w*h) gives 1.77. Indexing a typed
// array at a fractional position yields undefined, and undefined decodes to
// exactly 0. A broken decode is indistinguishable from a broken renderer if the
// decode is the only witness.
//
// So: the DISPLAY path is now a blit that samples the texture the same way the
// shadow lookup does. If the blit shows structure, the map is good and any
// remaining fault is in the comparison. The readback stays for numbers, with the
// padding handled properly, and the two act as a check on each other.
// ---------------------------------------------------------------------------

// --- The blit: draw a cascade to the corner of the screen ---
//
// A screen-space quad in its own scene and camera, rendered after the main pass
// with the clear suppressed. The depth value is shown BANDED rather than as a
// plain ramp: the cascade spans 54 m and the terrain occupies a thin slice of
// it, so a linear grey ramp of a working map looks like a flat white square -
// which is exactly the misreading that cost the last round. fract(d * bands)
// turns any variation at all into visible contour stripes without needing to
// know the range in advance.
export function createCascadeBlit(targets, uniforms) {
  const level = { value: 0 };
  const bands = uniform(float(40));
  const marker = uniform(new THREE.Vector3());
  const showMarker = uniform(float(1));
  const texNode = texture(targets[0].texture);
  // The projection-view of the cascade being shown. Swapped by setLevel, and it
  // is the SAME matrix the shadow lookup uses - so a marker placed with it tests
  // the real world-to-map transform, not a reconstruction of it.
  const projView = uniform(new THREE.Matrix4());

  const material = new THREE.MeshBasicNodeMaterial();
  material.depthTest = false;
  material.depthWrite = false;
  material.blending = THREE.NoBlending;
  material.transparent = false;
  material.colorNode = Fn(() => {
    // p is (u, v) in the LIGHT's axes, with +v up the screen. WebGPU's texture v
    // runs the other way (see clipToShadowUV), so the sample is taken flipped
    // while the legend and the marker below stay in light-axis space. That is
    // what makes the overlay readable as the light's own view of the world
    // rather than as an upside-down one.
    const p = uv().toVar();
    const d = texNode.sample(vec2(p.x, float(1).sub(p.y))).r.toVar();
    const stripe = d.mul(bands).fract();
    // Cleared texels (nothing rasterised) and exact zeros are called out in
    // colour, because those are the two failure modes and neither should be
    // left to be inferred from a shade of grey.
    const cleared = d.greaterThan(float(0.999));
    const zero = d.lessThan(float(1e-4));
    const grey = vec3(stripe, stripe, stripe);
    const out = cleared.select(vec3(0.25, 0, 0.45),
                zero.select(vec3(0.8, 0, 0), grey)).toVar();

    // --- Axis legend ---
    //
    // A thin strip along the map's +u edge in RED and its +v edge in GREEN,
    // drawn in the SAME uv space the texture is sampled in. So the strips label
    // the map's own axes as displayed: if the green strip appears at the bottom
    // of the overlay rather than the top, the DISPLAY is flipped. That separates
    // a display flip from a wrong world-to-map transform, which is a distinction
    // that cost a whole round of guessing when the picture was unlabelled.
    const edge = float(0.02);
    If(p.x.greaterThan(float(1).sub(edge)), () => { out.assign(vec3(0.9, 0.1, 0.1)); });
    If(p.y.greaterThan(float(1).sub(edge)), () => { out.assign(vec3(0.1, 0.9, 0.1)); });

    // --- Marker ---
    //
    // A crosshair at a known world position, projected with the cascade's own
    // matrix. If it lands on that thing's silhouette in the map, the world-to-map
    // transform is right. It moves WITH the image under a display flip, by
    // design: the legend above is what catches the display, this catches the
    // transform, and one picture answers both.
    const clip = projView.mul(vec4(marker, float(1)));
    const muv = clip.xy.div(clip.w).mul(float(0.5)).add(float(0.5));
    const dx = p.x.sub(muv.x).abs();
    const dy = p.y.sub(muv.y).abs();
    const arm = float(0.06), thick = float(0.004);
    const onCross = dx.lessThan(thick).and(dy.lessThan(arm))
                      .or(dy.lessThan(thick).and(dx.lessThan(arm)));
    If(onCross.and(showMarker.greaterThan(float(0.5))), () => {
      out.assign(vec3(1.0, 0.85, 0.0));
    });

    return vec4(out, float(1));
  })();

  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  quad.frustumCulled = false;
  scene.add(quad);

  return {
    scene, camera, material, bands, level, targets, marker, showMarker,
    visible: false,
    setLevel(i) {
      level.value = i;
      texNode.value = targets[i].texture;
      if (uniforms) projView.value.copy(uniforms[i].projView.value);
    },
    // The cascades refit as the player moves, so the matrix has to be re-read
    // every frame the overlay is up, not just when the level changes.
    sync() {
      if (uniforms) projView.value.copy(uniforms[level.value].projView.value);
    },
    setMarker(p) {
      marker.value.set(p.x, p.y, p.z);
    }
  };
}

// Called from the frame loop after the main render. Scissored to a corner so the
// game stays visible next to it - the whole point is to compare the map against
// the scene it was built from.
export function drawCascadeBlit(renderer, blit, size = 288) {
  if (!blit || !blit.visible) return;
  blit.sync();
  const prevAutoClear = renderer.autoClear;
  const vp = new THREE.Vector4();
  renderer.getViewport(vp);
  const sc = new THREE.Vector4();
  renderer.getScissor(sc);
  const prevScissorTest = renderer.getScissorTest();

  const w = Math.min(size, vp.z), h = Math.min(size, vp.w);
  renderer.autoClear = false;
  renderer.setViewport(vp.z - w - 8, 8, w, h);
  renderer.setScissor(vp.z - w - 8, 8, w, h);
  renderer.setScissorTest(true);
  renderer.render(blit.scene, blit.camera);

  renderer.setScissorTest(prevScissorTest);
  renderer.setScissor(sc.x, sc.y, sc.z, sc.w);
  renderer.setViewport(vp.x, vp.y, vp.z, vp.w);
  renderer.autoClear = prevAutoClear;
}

// --- The readback: numbers, with the row padding handled ---

// Half floats come back as raw 16-bit patterns, so they need decoding before any
// of it means anything.
function halfToFloat(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

let viewCanvas = null;

export async function readCascade(renderer, targets, level = 0,
                                  depthRange = null, doc = document) {
  const rt = targets[level];
  const raw = await renderer.readRenderTargetPixelsAsync(rt, 0, 0, SHADOW_DIM, SHADOW_DIM);
  const isHalf = raw instanceof Uint16Array;
  const layout = readbackLayout(raw.length, raw.BYTES_PER_ELEMENT, SHADOW_DIM, SHADOW_DIM);
  if (!layout) {
    // Refuse to invent a stride. Guessing here is what produced a confident
    // report of an all-zero map from a buffer that was never being read.
    const msg = `[csm] cannot decode readback: ${raw.length} elements of ` +
                `${raw.BYTES_PER_ELEMENT} bytes does not match any padded ` +
                `${SHADOW_DIM}x${SHADOW_DIM} layout. Use the blit instead.`;
    console.warn(msg);
    return { verdict: msg, undecodable: true, rawLength: raw.length };
  }

  const n = SHADOW_DIM * SHADOW_DIM;
  const depth = new Float32Array(n);
  let lo = Infinity, hi = -Infinity;
  let rawLo = Infinity, rawHi = -Infinity;
  let zeros = 0, cleared = 0, between = 0;

  for (let y = 0; y < SHADOW_DIM; y++) {
    const row = y * layout.rowElements;
    for (let x = 0; x < SHADOW_DIM; x++) {
      const e = raw[row + x * layout.channels];
      const v = isHalf ? halfToFloat(e) : e;
      depth[y * SHADOW_DIM + x] = v;
      if (v < rawLo) rawLo = v;
      if (v > rawHi) rawHi = v;
      if (Math.abs(v) < 1e-6) zeros++;
      else if (v > 0.999) cleared++;
      else { between++; if (v < lo) lo = v; if (v > hi) hi = v; }
    }
  }

  if (!viewCanvas) {
    viewCanvas = doc.createElement('canvas');
    viewCanvas.width = SHADOW_DIM;
    viewCanvas.height = SHADOW_DIM;
    Object.assign(viewCanvas.style, {
      position: 'fixed', bottom: '8px', left: '8px', zIndex: '9999',
      // Pinned in px because the project stylesheet has a blanket
      // canvas { width: 100%; height: 100% } for the renderer canvas.
      width: (SHADOW_DIM * 2) + 'px', height: (SHADOW_DIM * 2) + 'px',
      border: '1px solid #444', background: '#000',
      imageRendering: 'pixelated', pointerEvents: 'none'
    });
    doc.body.appendChild(viewCanvas);
  }

  const ctx = viewCanvas.getContext('2d');
  const img = ctx.createImageData(SHADOW_DIM, SHADOW_DIM);
  const span = (hi - lo) || 1;
  for (let i = 0; i < n; i++) {
    const v = depth[i];
    // No row flip: a WebGPU readback's row 0 is the texture's TOP row, which is
    // also a canvas's row 0. The flip that used to be here turned the picture
    // upside down, disagreeing with the blit for no reason.
    const o = i * 4;
    let r, g, b;
    if (Math.abs(v) < 1e-6)      { r = 200; g = 0;  b = 0; }    // exact zero
    else if (v > 0.999)          { r = 40;  g = 0;  b = 70; }   // cleared
    else { r = g = b = Math.round(255 * (1 - (v - lo) / span)); }
    img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);

  const range = depthRange || cascadeDepth(level);
  const constant = (rawHi - rawLo) < 1e-6;
  const stats = {
    cascade: level, dim: SHADOW_DIM,
    format: isHalf ? 'half' : 'float',
    channels: layout.channels, rowElements: layout.rowElements,
    constant, rawMin: +rawLo.toFixed(6), rawMax: +rawHi.toFixed(6),
    pctGeometry: +(between / n * 100).toFixed(1),
    pctCleared: +(cleared / n * 100).toFixed(1),
    pctZero: +(zeros / n * 100).toFixed(1),
    nearestMetres: between ? +(lo * range).toFixed(2) : null,
    furthestMetres: between ? +(hi * range).toFixed(2) : null,
    depthRangeMetres: +range.toFixed(1)
  };

  let verdict;
  if (constant && Math.abs(rawLo) < 1e-6) {
    verdict = 'BROKEN: every texel is 0. The clear did not land on the target ' +
              'and the pass wrote nothing.';
  } else if (constant && rawLo > 0.999) {
    verdict = 'BROKEN: every texel is the clear value. The clear works, so the ' +
              'draw is the problem - no caster, or the camera sees none of it.';
  } else if (constant) {
    verdict = `BROKEN: every texel is ${stats.rawMin}. Something writes a ` +
              'constant, so the depth expression is not varying with position.';
  } else if (between === 0) {
    verdict = 'BROKEN: no texel holds a plausible depth - only zeros and/or the ' +
              'clear value, in patches.';
  } else {
    verdict = `OK: ${stats.pctGeometry}% of the map holds geometry at ` +
              `${stats.nearestMetres}..${stats.furthestMetres} m of a ` +
              `${stats.depthRangeMetres} m range.`;
  }
  console.log(`[csm] cascade ${level}: ${verdict}`, stats);
  stats.verdict = verdict;
  return stats;
}

export function hideCascadeView() {
  if (viewCanvas) { viewCanvas.remove(); viewCanvas = null; return true; }
  return false;
}

// --- Numeric probe: what the lookup compares, in metres ---
//
// A picture of the comparison saturates, so it can say "they disagree" but not
// by how much - and that is the whole question. A bias failure is centimetres; a
// transform failure is metres. This reads the one texel the lookup would read
// and prints both sides of the comparison, so the answer is a number.
export async function probeCascade(renderer, targets, cascade, p, n, level = 0) {
  const rt = targets[level];
  const raw = await renderer.readRenderTargetPixelsAsync(rt, 0, 0, SHADOW_DIM, SHADOW_DIM);
  const layout = readbackLayout(raw.length, raw.BYTES_PER_ELEMENT, SHADOW_DIM, SHADOW_DIM);
  if (!layout) return { error: 'readback layout not recognised' };

  const biased = {
    x: p.x + n.x * cascade.texel * NORMAL_BIAS_TEXELS,
    y: p.y + n.y * cascade.texel * NORMAL_BIAS_TEXELS,
    z: p.z + n.z * cascade.texel * NORMAL_BIAS_TEXELS
  };
  const sh = worldToShadow(cascade, biased);
  // The texture v is the complement of the light-space v - the same flip the
  // shader applies, taken from the same definition.
  const tv = 1 - sh.v;
  const tx = Math.floor(sh.u * SHADOW_DIM);
  const ty = Math.floor(tv * SHADOW_DIM);
  if (tx < 0 || ty < 0 || tx >= SHADOW_DIM || ty >= SHADOW_DIM) {
    return { error: 'probe point is outside cascade ' + level, u: sh.u, v: sh.v };
  }

  const e = raw[ty * layout.rowElements + tx * layout.channels];
  const stored = raw instanceof Uint16Array ? halfToFloat(e) : e;
  const diff = sh.depth - stored;
  const out = {
    cascade: level,
    texel: [tx, ty],
    u: +sh.u.toFixed(4), v: +sh.v.toFixed(4),
    receiver: +sh.depth.toFixed(6),
    stored: +stored.toFixed(6),
    diffMetres: +(diff * cascade.depth).toFixed(3),
    depthRangeMetres: +cascade.depth.toFixed(1),
    storedIsClear: stored > 0.999
  };
  out.verdict = out.storedIsClear
    ? 'the map is CLEARED at this texel - nothing was rasterised here, so this ' +
      'point should read as lit'
    : Math.abs(out.diffMetres) < 0.5
      ? 'receiver and map agree to within half a metre: this is a BIAS problem'
      : 'receiver and map disagree by ' + out.diffMetres + ' m: this is a ' +
        'TRANSFORM problem, and no bias value will fix it';
  console.log('[csm probe]', out.verdict, out);
  return out;
}
