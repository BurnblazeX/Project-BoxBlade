import * as THREE from 'three/webgpu';
import { Fn, If, float, vec3, uniform, instanceIndex, storage, floor, abs, vec2, vec4, ivec2, clamp, uniformArray, int, select, textureLoad, screenCoordinate, Discard, uint, atomicStore, atomicAdd } from 'three/tsl';
import { VOXEL_METRES } from '../core/field.js';
import { terrainSurfaceTSL } from './terrain.js';
import { outsideShadeTSL } from './occlusion.js';
import { MATERIAL_LAYERS } from '../core/materials.js';
import { createRoomMeter } from './roommeter.js';

// ---------------------------------------------------------------------------
// Texel-rate shading: the terrain shaded once per texel, not once per pixel
//
// Everything the terrain shades is texel-locked - one texel, one answer - so
// every pixel inside a texel runs the identical marches and gets the identical
// colour. At play zoom a 12.5 cm texel covers many pixels; profiling put the
// torch, the mirrors and the reflection at ~3 ms of the frame, almost all of
// it repeats.
//
// So the terrain is drawn twice:
//
//   1  a LOW-RES pass (1/scale per axis) with the full shading, writing
//      vec4(colour, texel id) to a float target. Each low-res pixel is one
//      sample of whatever texel lies under it.
//   2  the normal full-res pass, which for each pixel computes only its own
//      texel id (cheap: the lock and a hash), and takes the colour of the
//      first sample in the 3x3 around it with the same id. Only a pixel with
//      no matching sample - a texel smaller than the sample spacing, at a far
//      zoom or a grazing angle - shades itself, in full.
//
// Nothing is approximated: a match means the sample's texel IS this pixel's
// texel, and a texel's colour is the same everywhere in it, bit for bit. No
// atlas, no paging, no invalidation - the cache is rebuilt every frame, so it
// is always the current frame's answer.
//
// The id is the world texel coordinate, each axis mod 128, plus the face (6):
// 7 + 7 + 7 + 3 bits, exact as a float. It only has to tell apart the handful
// of texels in a 3x3 window of samples, which are never 16 m apart.
// Offset by 2 so no id is 0 (cleared) or 1 (the sky's alpha).
export const TEXEL_ID_OFFSET = 2;

export const texelIdTSL = (p, n) => {
  // Half a texel inside the face, so the along-normal floor is not taken on
  // the plane itself.
  const t = p.sub(n.mul(float(VOXEL_METRES * 0.5))).div(float(VOXEL_METRES)).floor();
  const wrap = v => v.sub(floor(v.div(float(128))).mul(float(128)));
  const face = select(abs(n.x).greaterThan(float(0.5)), select(n.x.greaterThan(float(0)), float(0), float(1)),
               select(abs(n.y).greaterThan(float(0.5)), select(n.y.greaterThan(float(0)), float(2), float(3)),
                      select(n.z.greaterThan(float(0)), float(4), float(5))));
  return wrap(t.x).add(wrap(t.y).mul(float(128))).add(wrap(t.z).mul(float(16384)))
    .add(face.mul(float(2097152))).add(float(TEXEL_ID_OFFSET));
};

// This terrain fragment's texel id - the same lock the shading takes.
export const terrainTexelIdTSL = () => {
  const surf = terrainSurfaceTSL();
  return select(surf.on, slopeTexelIdTSL(surf), texelIdTSL(surf.p, surf.n));
};

// A slope texel's id: its rows are finer than the voxels, so two can share a
// voxel - it is named by its block and indices instead, face 6. Only the
// 3 x 3 window has to tell texels apart, so each block coordinate keeps its
// low bits: x (2) with the row (5), z (3) with the column (4), y (7).
function slopeTexelIdTSL(surf) {
  const wrap = (v, m) => v.sub(floor(v.div(float(m))).mul(float(m)));
  const b = surf.block;
  const fx = wrap(b.x, 4).mul(float(32)).add(surf.slopeTexel.y);
  const fz = wrap(b.z, 8).mul(float(16)).add(surf.slopeTexel.x);
  return fx.add(wrap(b.y, 128).mul(float(128))).add(fz.mul(float(16384)))
    .add(float(6 * 2097152)).add(float(TEXEL_ID_OFFSET));
}

// The low-res pass's output: the full shading and the texel it belongs to.
// negate: glass under split shading - its id negative, so the split kernels
// pass it through instead of shading it as terrain.
export function createTexelCacheWriteNode(shade, negate = false) {
  const id = terrainTexelIdTSL();
  return vec4(shade, negate ? id.negate() : id);
}

// --- Split shading ---
//
// With it on, the low-res pass shades nothing: it writes each sample's
// texel-locked position and texel id (createTexelGBufferNode), and compute
// kernels light the samples a feature at a time (createShadowColorNode,
// splitKernels): base (surface, AO, GI, sun), point lights, mirror light,
// and a final kernel (reflections, the clamped highlights). Each kernel is a
// small program, so it runs with the registers it needs - one shader holding
// every feature ran every texel at the occupancy of its worst path. The
// lookup reads their result. A slope's plane or a glowing model cube writes
// no sample (id 0): its pixels fall to the miss pass, which keeps the full
// shader. capacity: samples (vec4 each) per buffer.
//
// The misses too (2026-10-14): a full-res pass of the same G-buffer materials
// (idRT) gives every pixel its texel; a resolve kernel finds each pixel's
// low-res sample, or appends the pixel to a miss list, and records which
// sample colours it (pixelSample). The kernels light low-res samples and
// misses as one index range - lowCap samples, then missCap misses - and the
// terrain's own draw reads its pixel's sample (createSplitReadNode). No
// prepass, no stencil, no miss pass: the full shaders are never compiled.
export function createSplitShading(lowCap, missCap, fullCap, half = true) {
  const capacity = lowCap + missCap;
  const buffer = (name, n = capacity, type = 'vec4', item = 4) => {
    const attr = new THREE.StorageBufferAttribute(type === 'vec4' ? n
      : type === 'uvec4' ? new Uint32Array(n * 4) : new Int32Array(n), item);
    return { attr, rw: storage(attr, type, n).setName(name),
             ro: storage(attr, type, n).toReadOnly().setName(name + 'Read') };
  };
  const countAttr = new THREE.StorageBufferAttribute(new Uint32Array(1), 1);
  return {
    capacity, lowCap, missCap, fullCap,
    result: buffer('bxbSplitResult'),   // colour so far, texel id (0: none)
    // The light on the inside's surfaces, for revealed faces (gpu/roommeter.js).
    room: createRoomMeter(),
    // Between kernels: the ambient a reflection blends with, and highlights
    // so far (clamped at the end) - six half floats in one uvec4. fp16 is
    // ~0.05%, a tenth of an 8-bit step before Fresnel scales it down; texel-
    // locked, so flat per texel - nothing to band.
    // half false: full precision, two vec4 buffers (bxb.splithalf()).
    half,
    light: half ? buffer('bxbSplitLight', capacity, 'uvec4') : null,
    env: half ? null : buffer('bxbSplitEnv'),
    spec: half ? null : buffer('bxbSplitSpec'),
    missSamples: buffer('bxbSplitMisses', missCap),             // a miss's position, id
    pixelSample: buffer('bxbSplitPixel', fullCap, 'int', 1),    // per pixel; -1 none
    missCount: { attr: countAttr,
                 rw: storage(countAttr, 'uint', 1).toAtomic().setName('bxbSplitMissCount'),
                 ro: storage(countAttr, 'uint', 1).toReadOnly().setName('bxbSplitMissCountRead') },
    width: uniform(int(1)),             // low-res
    count: uniform(int(0)),             // low-res samples this frame
    fullWidth: uniform(int(1)),
    fullCount: uniform(int(0)),
    viewPos: uniform(new THREE.Vector3()),
    rt: null,                           // the low-res target's texture, set by main.js
    idRT: null                          // the full-res one's
  };
}

// Per material, its slope planes' first layers + 1 (0: the side stretched):
// x slope, y half slope - render.js fillInstances' modelInfo.x, as a table.
export const slopeLayerTable = uniformArray(MATERIAL_LAYERS.faces.map(f =>
  new THREE.Vector4(f.slope[0] + 1, f.slope[1] + 1, 0, 0)), 'vec4').setName('bxbSlopeLayers');

// The resolve, before the kernels: [clear the miss count, per pixel find
// its sample]. cache: texelCache (scale, size).
export function createSplitResolve(split, cache) {
  const clear = Fn(() => { atomicStore(split.missCount.rw.element(int(0)), uint(0)); })()
    .compute(1).setName('splitMissClear');
  const resolve = Fn(() => {
    const idx = int(instanceIndex).toVar();
    If(idx.lessThan(split.fullCount), () => {
      const px = vec2(float(idx.mod(split.fullWidth)), float(idx.div(split.fullWidth))).toVar();
      const s = textureLoad(split.idRT, ivec2(px)).toVar();
      const out = int(-1).toVar();
      If(abs(s.w).greaterThan(float(1.5)), () => {
        // The lookup's own 3x3, on the low-res ids (glass ones negative too).
        const lc = floor(px.add(float(0.5)).mul(cache.scale)).toVar();
        const hi = cache.size.sub(float(1));
        for (const [dx, dy] of TEXEL_CACHE_TAPS) {
          If(out.lessThan(int(0)), () => {
            const c = clamp(lc.add(vec2(dx, dy)), vec2(0, 0), hi);
            const i = int(c.y).mul(split.width).add(int(c.x)).toVar();
            If(i.lessThan(split.count), () => {
              If(textureLoad(split.rt, ivec2(c)).w.equal(s.w), () => { out.assign(i); });
            });
          });
        }
        If(out.lessThan(int(0)), () => {
          const slot = int(atomicAdd(split.missCount.rw.element(int(0)), uint(1))).toVar();
          If(slot.lessThan(int(split.missCap)), () => {
            split.missSamples.rw.element(slot).assign(s);
            out.assign(int(split.lowCap).add(slot));
          }).Else(() => {
            // The list is full: the nearest low-res sample, not nothing.
            const c = clamp(lc, vec2(0, 0), hi);
            out.assign(int(c.y).mul(split.width).add(int(c.x)));
          });
        });
      });
      split.pixelSample.rw.element(idx).assign(out);
    });
  })().compute(split.fullCap).setName('splitResolve');
  return [clear, resolve];
}

// The terrain's (and glass's) draw under split shading: its pixel's sample -
// or, where this fragment is another texel than the pixel's centre (an MSAA
// edge), its own in the 3x3 low-res window; failing both, the pixel's.
export function createSplitReadNode(cache, split) {
  return Fn(() => {
    const id = terrainTexelIdTSL().toVar();
    const px = floor(screenCoordinate.xy).toVar();
    const si = split.pixelSample.ro.element(int(px.y).mul(split.fullWidth).add(int(px.x))).toVar();
    const out = vec3(0, 0, 0).toVar();
    const found = float(0).toVar();
    If(si.greaterThanEqual(int(0)), () => {
      const r = split.result.ro.element(si).toVar();
      out.assign(r.xyz);
      If(r.w.equal(id), () => { found.assign(float(1)); });
    });
    const lc = floor(screenCoordinate.xy.mul(cache.scale)).toVar();
    const hi = cache.size.sub(float(1));
    for (const [dx, dy] of TEXEL_CACHE_TAPS) {
      If(found.equal(float(0)), () => {
        const c = clamp(lc.add(vec2(dx, dy)), vec2(0, 0), hi);
        const s = split.result.ro.element(int(c.y).mul(split.width).add(int(c.x))).toVar();
        If(s.w.equal(id), () => { out.assign(s.xyz); found.assign(float(1)); });
      });
    }
    return out;
  })();
}

// The low-res pass's output under split shading: position and texel id -
// negative for glass, which has kernels of its own.
export function createTexelGBufferNode(glass = false, legacy = false) {
  return Fn(() => {
    const surf = terrainSurfaceTSL();
    if (legacy) {
      // A08//00a: slopes and glowing cubes write nothing (they are misses).
      const id = texelIdTSL(surf.p, surf.n);
      return select(surf.on.or(surf.emissive), vec4(0, 0, 0, 0), vec4(surf.p, glass ? id.negate() : id));
    }
    const id = select(surf.on, slopeTexelIdTSL(surf), texelIdTSL(surf.p, surf.n));
    return vec4(surf.p, glass ? id.negate() : id);
  })();
}

// The full-res lookup. cache: { tex, scale (vec2 low/full), size (vec2 low) }.
//
// A pixel with no matching sample is DISCARDED here, not shaded: the full
// shading lives in a separate miss pass (createTexelMissNode), drawn after,
// where a stencil test keeps it off every pixel this pass filled. It cannot
// live in a branch of this shader - measured, a never-taken branch holding the
// full shading cost 2.5 ms of 3.5, because a GPU sizes a shader's registers for
// its worst path, and every pixel then runs with that fewer in flight.
const TEXEL_CACHE_TAPS = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                          [1, 1], [-1, 1], [1, -1], [-1, -1]];

export function createTexelCacheLookupNode(cache, split = null) {
  // Under split shading the colours are in the kernels' result buffer.
  const load = c => (split
    ? split.result.ro.element(int(c.y).mul(split.width).add(int(c.x)))
    : textureLoad(cache.tex, ivec2(c)));
  return Fn(() => {
    const id = terrainTexelIdTSL().toVar();
    const lc = floor(screenCoordinate.xy.mul(cache.scale)).toVar();
    const hi = cache.size.sub(float(1));
    const out = vec3(0, 0, 0).toVar();
    const found = float(0).toVar();
    // Unrolled on purpose: nine loads, the centre first, each skipped once a
    // match is in hand.
    for (const [dx, dy] of TEXEL_CACHE_TAPS) {
      If(found.equal(float(0)), () => {
        const c = clamp(lc.add(vec2(dx, dy)), vec2(0, 0), hi);
        const s = load(c).toVar();
        If(s.w.equal(id), () => { out.assign(s.xyz); found.assign(float(1)); });
      });
    }
    If(found.equal(float(0)), () => { Discard(); });
    return out;
  })();
}

// The miss pass's colour: the full shading - or, in the miss view, red, so
// which pixels shade themselves can be seen.
export function createTexelMissNode(cache, shade) {
  return cache.showMisses ? vec3(1, 0, 0) : shade;
}

// Swaps the terrain to a node material whose colour is its own albedo modulated
// by the marched shadow term, keeping the original so it can be restored.
export function applyShadowMaterial(mesh, shadowNode, albedo = null) {
  const mat = createShadowMaterial(mesh, shadowNode, albedo);
  if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
  mesh.material = mat;
  return mat;
}

// The same material, built but not assigned - so it can be compiled off-screen
// while the current one keeps drawing.
// albedo: the surface colour node - for the terrain, its per-block layer with
// whiteworld mixed in (render.js). White when null.
export function createShadowMaterial(mesh, shadowNode, albedo = null) {
  if (!mesh.userData.originalMaterial) mesh.userData.originalMaterial = mesh.material;
  const mat = new THREE.MeshBasicNodeMaterial();
  // .rgb, because the shadow node is a vec3 now that a coloured light adds to it.
  // Per-instance tint (arena greying) lives in instanceColor, so it has to be
  // carried across or battle mode loses its reachable-tile shading.
  // Darkened outside a building the player is in - the view only, after the
  // shading (gpu/occlusion.js outsideShadeTSL).
  mat.colorNode = (albedo || vec3(1, 1, 1)).mul(shadowNode).mul(outsideShadeTSL());
  return mat;
}

export function restoreOriginalMaterial(mesh) {
  if (!mesh.userData.originalMaterial) return false;
  if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
  mesh.material = mesh.userData.originalMaterial;
  return true;
}
