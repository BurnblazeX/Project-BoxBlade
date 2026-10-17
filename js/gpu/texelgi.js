import * as THREE from 'three/webgpu';
import { If, float, vec3, uniform, storage, floor, max, abs, vec2, length, uniformArray, int, select, uint, ivec3, bitXor, bitAnd, bitOr, packHalf2x16, unpackHalf2x16, min, smoothstep } from 'three/tsl';
import { GRID_DIM, VOXEL_METRES } from '../core/boxgrid.js';
import { mix32TSL } from './hitcache.js';

// --- Texel GI: a radiance cache on the block faces ---
//
// Every box face is cut into patches of PATCH_TEXELS x PATCH_TEXELS texels (a
// block face is 4 x 4 of them). Each patch integrates the light arriving over
// its hemisphere from a FIXED set of GI_DIRS directions (a Fibonacci spiral,
// uniform over the hemisphere) - no random numbers anywhere, so a patch's
// answer is the same every frame the light is: nothing to denoise. The
// directions are dealt into GI_SLICES slices, every GI_SLICES-th one, so each
// slice covers the whole hemisphere; a frame traces one slice (frame mod
// GI_SLICES) for every patch on screen and keeps each slice's latest sum, and
// the patch's light is the sum of all of them: always the full GI_DIRS-ray
// quadrature, with each slice at most GI_SLICES frames old.
//
// What a ray meets gives back its light: its albedo times the sky, the sun
// (with its shadow), the light list unshadowed and its own patch light - so
// bounces chain, one more a frame. A glowing model cube gives its colour; a
// miss nothing (the sky is the ambient's, under the AO). Stored as L1
// spherical harmonics in world axes, so the texel reading it evaluates it at
// its own normal-mapped normal - the bumps catch the bounce too.
//
// A texel reads the four patches nearest its centre on its face, bilinearly
// (patchLightTSL) - one answer per texel, still.
//
// Patches live in a hash table (GI_SLOTS, two probes). The split base kernel
// finds the patches on screen: each sample stamps its patch's slot with its
// check by atomicMax (the stamps are cleared each frame), and the first lists
// the patch. A collision goes to the higher check; the lower tries its second
// slot. The patch kernel (one thread a listed patch) traces the patch's next
// slice; the sum kernel after it adds the slices up - in a dispatch of its
// own, so no ray ever reads a sum while it is being written (that tore, and
// shimmered).
// A patch's side in texels by level: 3 within C1 (C0 too), 6 within C2, 12
// (a whole block face) past it. Each step blends over the inner cascade's
// outer GI_BLEND_VOXELS (createShadowColorNode giLevel), where a texel reads -
// and lists - both levels' patches.
export const PATCH_TEXELS = 3;

const PATCH_SIDES = [3, 6, 12];

export const GI_BLEND_VOXELS = 16;

// Past C2 a ray has no field to march, so the bounce ends there: it fades out
// over C2's outer GI_FADE_VOXELS rather than stopping at an edge.
export const GI_FADE_VOXELS = 24;

// A slot's FILLED word: slices filled (bits 0-15), next slice (16-19), FRESH
// - new this frame, so the sum kernel takes its first sum as is, unsmoothed.
export const GI_FRESH = 1 << 20;

// A slot a patch has not traced for this many frames may go to a new patch.
export const GI_STALE_FRAMES = 4;

export const GI_DIRS = 128;

export const GI_SLICES = 16;                // 8 rays a slice, a slice a frame

export const GI_SLICE_MASK = (1 << GI_SLICES) - 1;

// A patch's light is the first moment of what arrives, sum L w dw - three
// vec3 (x, y, z), five u32 of half floats (patchLightTSL reads it).
export const GI_WORDS = 5;

export const GI_WORD_INDEX = [0, 1, 2, 3, 4];

export const GI_SLOTS = 1 << 17;

export const GI_LIST_CAP = 65536;

export const DEFAULT_TEXEL_GI_REACH = 8;   // metres a GI ray marches

// u32 a slot: 16 slices x 5, the sum x 5, check, then the slices filled and
// the next slice to trace (FILLED) - a patch's own count, so it steps through
// its slices however often it is on screen - then the frame it last traced.
export const GI_STRIDE = 96;

export const GI_SUM = GI_SLICES * GI_WORDS, GI_CHECK = GI_SUM + GI_WORDS, GI_FILLED = GI_CHECK + 1, GI_LAST = GI_FILLED + 1;

export const GI_STAMPS = 1 + GI_LIST_CAP * 4;     // queue: [count, list (4 u32 each), stamps, claims]

export const GI_CLAIMS = GI_STAMPS + GI_SLOTS;     // patches that took a slot this frame (bxb.gistats)

// Per slot, the patch's ANCHOR: of its texels seen this frame, the one nearest
// its centre - (squared distance << 8) | (a * 16 + b), by atomicMin, so the
// same texel wins every frame. The patch's rays start there: its centre can
// be under a pane, a half block or a model cube that covers part of it, and
// rays from inside geometry came out the other side with that side's light.
export const GI_ANCHORS = GI_CLAIMS + 1;

// The directions, about +z: z uniform in [0, 1] (equal solid angle), the
// golden angle round.
const GI_DIR_TABLE = Array.from({ length: GI_DIRS }, (_, i) => {
  const z = 1 - (i + 0.5) / GI_DIRS, r = Math.sqrt(1 - z * z), a = i * Math.PI * (3 - Math.sqrt(5));
  return new THREE.Vector4(r * Math.cos(a), r * Math.sin(a), z, 0);
});

export function createTexelGI() {
  const queueAttr = new THREE.StorageBufferAttribute(new Uint32Array(GI_ANCHORS + GI_SLOTS), 1);
  const dataAttr = new THREE.StorageBufferAttribute(new Uint32Array(GI_SLOTS * GI_STRIDE), 1);
  return {
    queueAttr, dataAttr,
    queue: storage(queueAttr, 'uint', GI_ANCHORS + GI_SLOTS).toAtomic().setName('bxbGIQueue'),
    claimsAt: GI_CLAIMS,
    data: storage(dataAttr, 'uint', GI_SLOTS * GI_STRIDE).setName('bxbGIPatches'),
    dataRead: storage(dataAttr, 'uint', GI_SLOTS * GI_STRIDE).toReadOnly().setName('bxbGIPatchesRead'),
    dirs: uniformArray(GI_DIR_TABLE, 'vec4').setName('bxbGIDirs'),
    frame: uniform(uint(0)),
    reach: uniform(float(DEFAULT_TEXEL_GI_REACH)),
    // 1: the bounce shows red where a texel finds no patch at all.
    debug: uniform(int(0)),
    // How far a patch's shown light moves toward its new sum this frame:
    // 1 - exp(-dt / response), set by main.js from the frame's own time - so
    // the smoothing is the same at 60 and 240 fps.
    blend: uniform(float(1)),
    // A/B switches for the last fixes, uniforms so they flip without a
    // rebuild (Alt+V -> GI): 1 is the fix, 0 the behaviour before it.
    ab: {
      hitShadows: uniform(int(1)),   // lamps shadowed at a ray's hit
      physical: uniform(int(1)),     // the hit bounces albedo (1 - F)(1 - metal) + F
      fresh: uniform(int(1)),        // the shading reads recently traced patches only
      anchor: uniform(int(1))        // rays start at the anchor texel, not the centre
    },
    strength: null,      // the GI strength uniform, set by main.js
    kernels: null        // { reset, patches }: built with the shading (createShadowColorNode)
  };
}

// A face's patch, from a point on it (texel-locked) and its axis normal:
// { pmin: its min voxel (tangent axes on the patch grid, the normal axis the
// solid voxel's), face (0-5), slots [two probes], check (never 0) }.
export const faceIndexTSL = n => select(abs(n.x).greaterThan(float(0.5)), select(n.x.greaterThan(float(0)), uint(0), uint(1)),
                          select(abs(n.y).greaterThan(float(0.5)), select(n.y.greaterThan(float(0)), uint(2), uint(3)),
                                 select(n.z.greaterThan(float(0)), uint(4), uint(5))));

// lvl (uint 0-2): the patch's size level - a 3- and a 6-texel patch at one
// corner are different patches.
export function patchKeyTSL(pmin, face, lvl) {
  const ip = ivec3(pmin);
  const h = mix32TSL(bitXor(bitXor(uint(ip.x).mul(uint(73856093)), uint(ip.y).mul(uint(19349663))),
                            bitXor(bitXor(uint(ip.z).mul(uint(83492791)), face.mul(uint(2654435761))),
                                   lvl.mul(uint(0x27d4eb2f))))).toVar();
  const h2 = mix32TSL(bitXor(h, uint(0x68e31da4)));
  return { slots: [int(bitAnd(h, uint(GI_SLOTS - 1))), int(bitAnd(h2, uint(GI_SLOTS - 1)))],
           check: bitOr(mix32TSL(bitXor(h, uint(0x9e3779b9))), uint(1)).toVar() };
}

// A level's patch side in texels, as a float.
export const patchSideTSL = lvl => select(lvl.equal(uint(0)), float(PATCH_SIDES[0]),
                                   select(lvl.equal(uint(1)), float(PATCH_SIDES[1]), float(PATCH_SIDES[2])));

// The voxel behind a face point, and the face's two tangent axes.
export function faceFrameTSL(p, n) {
  const q = float(VOXEL_METRES);
  const t = floor(p.sub(n.mul(q.mul(float(0.5)))).div(q)).toVar();
  const ax = abs(n).toVar();
  const e1 = vec3(float(1).sub(ax.x), ax.x, float(0)).toVar();
  const e2 = vec3(float(0), ax.z, float(1).sub(ax.z)).toVar();
  return { t, ax, e1, e2 };
}

// The first moment (3 vec3: the x, y and z parts, each rgb) to and from
// GI_WORDS u32 of half floats.
export const shPackTSL = c => [
  packHalf2x16(vec2(c[0].x, c[0].y)), packHalf2x16(vec2(c[0].z, c[1].x)), packHalf2x16(vec2(c[1].y, c[1].z)),
  packHalf2x16(vec2(c[2].x, c[2].y)), packHalf2x16(vec2(c[2].z, float(0)))];

export const shUnpackTSL = w => {
  const a = w.map(x => unpackHalf2x16(x));
  return [vec3(a[0].x, a[0].y, a[1].x), vec3(a[1].y, a[2].x, a[2].y), vec3(a[3].x, a[3].y, a[4].x)];
};

// The bounce light at a texel (p on its face, n its axis normal), seen by
// shading normal nS: the four nearest patches on the face, bilinear, those
// the table holds only. In the diffuse's units (E / pi), never below 0.
// lvl: the patch level there (createShadowColorNode's giLevel). fresh: only
// patches traced in the last GI_STALE_FRAMES - what the shading reads. A patch
// that stopped being seen (a block placed over it, a face removed) keeps its
// last light in the table, and the texels beside it blended that in for good:
// the edges by new geometry never followed the light. A ray's hit reads any
// (fresh false): off screen is still light that bounces.
export function patchLightTSL(tg, p, n, nS, lvl, buf = tg.dataRead, fresh = true) {
  const { t, ax, e1, e2 } = faceFrameTSL(p, n);
  const face = faceIndexTSL(n);
  const side = patchSideTSL(lvl).toVar();
  const u = t.add(float(0.5)).div(side).sub(float(0.5)).toVar();
  const i0 = floor(u).toVar();
  const fa = u.sub(i0).dot(e1).toVar(), fb = u.sub(i0).dot(e2).toVar();
  const keepT = t.mul(ax).toVar(), tang = vec3(1, 1, 1).sub(ax).toVar();
  const sum = [0, 1, 2].map(() => vec3(0, 0, 0).toVar());
  const wsum = float(0).toVar();
  for (const [da, db] of [[0, 0], [1, 0], [0, 1], [1, 1]]) {
    const pmin = keepT.add(i0.add(e1.mul(float(da))).add(e2.mul(float(db))).mul(side).mul(tang));
    const w = (da ? fa : float(1).sub(fa)).mul(db ? fb : float(1).sub(fb)).toVar();
    const key = patchKeyTSL(pmin, face, lvl);
    const found = int(-1).toVar();
    for (const slot of key.slots) {
      let ok = buf.element(slot.mul(int(GI_STRIDE)).add(int(GI_CHECK))).equal(key.check);
      if (fresh) {
        ok = ok.and(tg.frame.sub(buf.element(slot.mul(int(GI_STRIDE)).add(int(GI_LAST))))
                      .lessThanEqual(uint(GI_STALE_FRAMES)).or(tg.ab.fresh.equal(int(0))));
      }
      If(found.lessThan(int(0)).and(ok), () => { found.assign(slot); });
    }
    If(found.greaterThanEqual(int(0)).and(w.greaterThan(float(0))), () => {
      const base = found.mul(int(GI_STRIDE)).add(int(GI_SUM)).toVar();
      const c = shUnpackTSL(GI_WORD_INDEX.map(k => buf.element(base.add(int(k)))));
      for (let k = 0; k < 3; k++) sum[k].addAssign(c[k].mul(w));
      wsum.addAssign(w);
    });
  }
  const inv = float(1).div(max(wsum, float(1e-6)));
  // The light's first moment, sum L w dw, dotted with the normal, over pi:
  // exactly the cosine-weighted irradiance at the
  // face's own normal, and its linear change as a normal map tilts it. Not
  // the full SH irradiance (L0 + L1 convolved): fed from one hemisphere, that
  // gave a ray along the face a third of a head-on one's weight instead of
  // none - and rays skimming the floor, landing on whatever the torch lit,
  // made the bounce a field of bright patch-sized squares.
  const light = sum[0].mul(nS.x).add(sum[1].mul(nS.y)).add(sum[2].mul(nS.z)).mul(float(1 / Math.PI));
  const out = max(light.mul(inv), vec3(0, 0, 0));
  return select(tg.debug.equal(int(1)).and(wsum.lessThanEqual(float(0))), vec3(4, 0, 0), out);
}

// Texel GI's patch level at p: { lvl (uint) 0 inside C1, 1 inside C2, else
// 2; next, the level past it; blend, how much of next to take - 0 inside,
// rising to 1 over the cascade's outer GI_BLEND_VOXELS (x and z: the world
// is shallower than any cascade) }.
export const texelGILevelTSL = cascades => p => {
  const at = c => {
    const vp = p.sub(c.origin).div(c.voxel).toVar();
    const dim = float(GRID_DIM);
    const inside = vp.x.greaterThanEqual(float(0)).and(vp.y.greaterThanEqual(float(0)))
      .and(vp.z.greaterThanEqual(float(0))).and(vp.x.lessThan(dim)).and(vp.y.lessThan(dim)).and(vp.z.lessThan(dim));
    const edge = min(min(vp.x, vp.z), min(dim.sub(vp.x), dim.sub(vp.z)));
    return { inside, edge };
  };
  const a = at(cascades[Math.min(1, cascades.length - 1)]);
  const b = at(cascades[Math.min(2, cascades.length - 1)]);
  const lvl = select(a.inside, uint(0), select(b.inside, uint(1), uint(2))).toVar();
  const edge = select(a.inside, a.edge, b.edge);
  const blend = select(lvl.lessThan(uint(2)),
    float(1).sub(smoothstep(float(0), float(GI_BLEND_VOXELS), edge)), float(0)).toVar();
  // 1 inside C1, fading to 0 across C2's outer band, 0 past it.
  const fade = select(a.inside, float(1),
    select(b.inside, smoothstep(float(0), float(GI_FADE_VOXELS), b.edge), float(0))).toVar();
  return { lvl, next: min(lvl.add(uint(1)), uint(2)).toVar(), blend, fade };
};
