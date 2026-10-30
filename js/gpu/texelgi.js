import * as THREE from 'three/webgpu';
import { tableGroup, frameValues } from './uploads.js';
import { Fn, If, Loop, float, vec3, uniform, storage, floor, max, abs, vec2, int, select, uint, sqrt, cos, sin, ivec3, bitXor, bitAnd, bitOr, packHalf2x16, unpackHalf2x16, min, smoothstep, mix, clamp, exp, normalize, cross, instanceIndex, shiftLeft, shiftRight, atomicAdd, atomicLoad, atomicStore, atomicMax, atomicMin, uniformArray, step, length } from 'three/tsl';
import { GRID_DIM, VOXEL_METRES } from '../core/field.js';
import { MAX_GI_SPRITES } from '../core/lpv.js';
import { mix32TSL } from './hitcache.js';
import { createReflectTraceTSL, giTraceHitTSL } from './trace.js';
import { hitSurfaceTSL, hitTexelTSL, inHitBoundsTSL, skyOpenTSL } from './volume.js';
import { glassRayTransmitTSL } from './glass.js';
import { REFLECT_ROUGH_END, REFLECT_ROUGH_START, decodeSpecularTSL, fresnelRoughTSL } from './specular.js';
import { NO_SKIP } from './cards.js';
import { skyShTSL, sunColourUniform } from './sky.js';
import { smokeInscatterTSL } from './smokeshadow.js';
import { leafInscatterTSL } from './leafvolume.js';

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
// What a ray meets gives back its light: its albedo times the sun (with its
// shadow), the light list with its shadows and its own patch light - so
// bounces chain, one more a frame. A glowing model cube gives its colour. A
// miss is the sky along the ray - but only if the ray truly escaped: where
// it ended (out of reach, or out of the field) is above its column's roof
// (volume.js skyOpenTSL). One that ran out of reach indoors brings nothing,
// so a roofed room gets the sky only through its doors and windows - through
// glass, tinted by the metres the ray spent in it. The patch's light is then
// the sky AND the bounce, and replaces the sky ambient at the texels reading
// it (ab.sky). Rays that hit within the AO's reach are also summed apart as
// the NEAR band (ab.near): the texel's own cones then decide how much of the
// patch's near and far light it sees (patchLightTSL).
// Stored in world axes as the first moment (L1) and the total (L0), so the
// texel reading it evaluates it at its own normal-mapped normal - the bumps
// catch the bounce too.
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
// The sprites are in it as the LPV has them (core/lpv.js, "Sprites in the
// LPV"): a box per card caster, its quad's width square and its height tall.
// A ray that enters one before its hit takes the sprite's light - its mean
// albedo lit as a face turned back along the ray, by the one light evaluation
// the hit would have had - over what lies behind, by the sprite's opaque
// fraction. What lies behind then keeps only its cached direct light.
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

// ... and DIRECT_OK: its direct light (GI_DIRECT) is worked out - cleared when
// a new patch takes the slot.
export const GI_DIRECT_OK = 1 << 21;

// A slot a patch has not traced for this many frames may go to a new patch.
export const GI_STALE_FRAMES = 4;

export const GI_DIRS = 128;

const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

export const GI_SLICES = 16;                // 8 rays a slice, a slice a frame

export const GI_SLICE_MASK = (1 << GI_SLICES) - 1;

// A patch's light: the first moment of what arrives, sum L w dw - three vec3
// (x, y, z) - its total, sum L dw, and the near band's irradiance at the face
// normal, sum a L cos dw (a ray hitting at t within the AO reach counts
// a = 1 - t / reach of its light as near, as the cones weigh blocking), then
// the rays' own near occlusion, sum a cos dw (a scalar, in a vec3's x): six
// parts, eight u32 of half floats (patchLightTSL reads them).
export const GI_WORDS = 8;

export const GI_VEC3 = 6;

export const GI_WORD_INDEX = [0, 1, 2, 3, 4, 5, 6, 7];

export const GI_SLOTS = 1 << 17;

export const GI_LIST_CAP = 65536;

export const DEFAULT_TEXEL_GI_REACH = 8;   // metres a GI ray marches

// u32 a slot: 16 slices x 8, the sum x 8, check, then the slices filled and
// the next slice to trace (FILLED) - a patch's own count, so it steps through
// its slices however often it is on screen - then the frame it last traced.
// Then the patch's DIRECT light (2 u32, half floats): the sun and the light
// list at its anchor, shadowed - what a GI ray landing on it takes instead of
// lighting the hit itself (patchDirectTSL) - and in its spare half V_p, the
// cone AO's visibility at the anchor (giDirect). Then FILT (8 u32): the light
// the shading reads - the sum blurred with its plane's neighbours (giFilter).
// 149 used, padded to 152.
export const GI_STRIDE = 152;

export const GI_SUM = GI_SLICES * GI_WORDS, GI_CHECK = GI_SUM + GI_WORDS, GI_FILLED = GI_CHECK + 1, GI_LAST = GI_FILLED + 1;

export const GI_DIRECT = GI_LAST + 1;

export const GI_FILT = GI_DIRECT + 2;

// The direct light is refreshed every this many frames by default, a whole
// workgroup of listed patches at a time (createTexelGIKernels' giDirect).
export const DEFAULT_GI_DIRECT_EVERY = 4;

export const GI_STAMPS = 1 + GI_LIST_CAP * 4;     // queue: [count, list (4 u32 each), stamps, claims]

export const GI_CLAIMS = GI_STAMPS + GI_SLOTS;     // patches that took a slot this frame (bxb.gistats)

// Per slot, the patch's ANCHOR: of its texels seen this frame, the one nearest
// its centre - (squared distance << 8) | (a * 16 + b), by atomicMin, so the
// same texel wins every frame. The patch's rays start there: its centre can
// be under a pane, a half block or a model cube that covers part of it, and
// rays from inside geometry came out the other side with that side's light.
export const GI_ANCHORS = GI_CLAIMS + 1;

// The directions, about +z: direction k at z = 1 - (k + 0.5) / GI_DIRS (z
// uniform in [0, 1] - equal solid angle) and k golden angles round - worked
// out in the patch kernel, where each patch turns the set its own way.

export function createTexelGI() {
  const queueAttr = new THREE.StorageBufferAttribute(new Uint32Array(GI_ANCHORS + GI_SLOTS), 1);
  const dataAttr = new THREE.StorageBufferAttribute(new Uint32Array(GI_SLOTS * GI_STRIDE), 1);
  return {
    queueAttr, dataAttr,
    queue: storage(queueAttr, 'uint', GI_ANCHORS + GI_SLOTS).toAtomic().setName('bxbGIQueue'),
    claimsAt: GI_CLAIMS,
    data: storage(dataAttr, 'uint', GI_SLOTS * GI_STRIDE).setName('bxbGIPatches'),
    dataRead: storage(dataAttr, 'uint', GI_SLOTS * GI_STRIDE).toReadOnly().setName('bxbGIPatchesRead'),
    frame: frameValues.giFrame,
    reach: uniform(float(DEFAULT_TEXEL_GI_REACH)),
    // Frames between refreshes of a patch's direct light (1: every frame).
    directEvery: uniform(int(DEFAULT_GI_DIRECT_EVERY)),
    // 1: the bounce shows red where a texel finds no patch at all.
    debug: uniform(int(0)),
    // How far a patch's shown light moves toward its new sum this frame:
    // 1 - exp(-dt / response), set by main.js from the frame's own time - so
    // the smoothing is the same at 60 and 240 fps.
    blend: frameValues.giBlend,
    // A/B switches for the last fixes, uniforms so they flip without a
    // rebuild (Alt+V -> GI): 1 is the fix, 0 the behaviour before it.
    ab: {
      hitShadows: uniform(int(1)),   // lamps shadowed at a ray's hit
      physical: uniform(int(1)),     // the hit bounces albedo (1 - F)(1 - metal) + F
      fresh: uniform(int(1)),        // the shading reads recently traced patches only
      anchor: uniform(int(1)),       // rays start at the anchor texel, not the centre
      sky: uniform(int(1)),          // escaping rays bring the sky; it replaces the sky ambient
      direct: uniform(int(1)),       // a hit on a patch takes its cached direct light
      rotate: uniform(int(1)),       // each patch turns the direction set its own way
      filter: uniform(int(1)),       // the shading reads each patch blurred with its neighbours
      near: uniform(int(1))          // the cones split the patch light into near and far (with sky)
    },
    // The sprites the rays meet, three vec4 each: (centre, half width),
    // (albedo, half height), (opaque fraction, 0, 0, 0). Written each frame
    // (render/features/gi.js stepTexel).
    sprites: uniformArray(new Array(MAX_GI_SPRITES * 3).fill(0).map(() => new THREE.Vector4()), 'vec4')
      .setName('bxbTexelGISprites').setGroup(tableGroup),
    spriteCount: uniform(int(0)),
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

// The first moment, the total, the near band and the near occlusion (GI_VEC3
// parts: the x, y and z parts, then L0, then near, each rgb, then the
// occlusion in x) to and from GI_WORDS u32 of half floats.
export const shPackTSL = c => [
  packHalf2x16(vec2(c[0].x, c[0].y)), packHalf2x16(vec2(c[0].z, c[1].x)), packHalf2x16(vec2(c[1].y, c[1].z)),
  packHalf2x16(vec2(c[2].x, c[2].y)), packHalf2x16(vec2(c[2].z, c[3].x)), packHalf2x16(vec2(c[3].y, c[3].z)),
  packHalf2x16(vec2(c[4].x, c[4].y)), packHalf2x16(vec2(c[4].z, c[5].x))];

export const shUnpackTSL = w => {
  const a = w.map(x => unpackHalf2x16(x));
  return [vec3(a[0].x, a[0].y, a[1].x), vec3(a[1].y, a[2].x, a[2].y), vec3(a[3].x, a[3].y, a[4].x),
          vec3(a[4].y, a[5].x, a[5].y), vec3(a[6].x, a[6].y, a[7].x), vec3(a[7].y, 0, 0)];
};

// The near/far split's ratios clamp at NEAR_RATIO_MAX, and the open and
// occluded shares are floored at NEAR_FLOOR / NEAR_OCC_FLOOR.
export const NEAR_RATIO_MAX = 2;

export const NEAR_FLOOR = 0.05;

export const NEAR_OCC_FLOOR = 1e-3;

// The bounce light at a texel (p on its face, n its axis normal), seen by
// shading normal nS: the four nearest patches on the face, bilinear, those
// the table holds only. In the diffuse's units (E / pi), never below 0.
// lvl: the patch level there (createShadowColorNode's giLevel). fresh: only
// patches traced in the last GI_STALE_FRAMES - what the shading reads. A patch
// that stopped being seen (a block placed over it, a face removed) keeps its
// last light in the table, and the texels beside it blended that in for good:
// the edges by new geometry never followed the light. A ray's hit reads any
// (fresh false): off screen is still light that bounces.
// offFace: nS may lean far off the face (a sprite reading the ground under
// it) - with ab.sky, read the L0 + L1 irradiance instead of the first moment.
// near: { vis } - the texel's own cone visibility V_t (null: none). With
// ab.near and ab.sky each patch's light is split by its near band. The rays
// give the band's share k of the light and their own near occlusion O_p, so
// the near light per unit of occlusion is k / O_p and the far light per unit
// of openness (1 - k) / (1 - O_p) - one measure, the rays'. The cones only
// say how much more the texel is boxed in than its anchor: O_t = O_p + V_p -
// V_t, V_p the cones at the anchor (giDirect). The light is scaled by
// (1 - k)(1 - O_t) / (1 - O_p) + k O_t / O_p, ratios clamped: where the
// texel sees what its anchor sees, the patch's light itself - the near
// occluders its rays met are not darkened a second time by the cones. A
// share, not a difference (total(nS) - near(n) went below zero on bumps), and
// never V_p against the rays' near light: the cones' 1 - V_p could be ~0.05
// where the rays had met a bright wall, and dividing by it turned texel-to-
// texel differences into a grid. A patch whose V_p is not worked out yet
// counts as V_p = V_t.
// Returns { light, found } - found 1 where any of the four is in the table.
export function patchLightTSL(tg, p, n, nS, lvl, buf = tg.dataRead, fresh = true, offFace = false, near = null) {
  const { t, ax, e1, e2 } = faceFrameTSL(p, n);
  const face = faceIndexTSL(n);
  const side = patchSideTSL(lvl).toVar();
  const u = t.add(float(0.5)).div(side).sub(float(0.5)).toVar();
  const i0 = floor(u).toVar();
  const fa = u.sub(i0).dot(e1).toVar(), fb = u.sub(i0).dot(e2).toVar();
  const keepT = t.mul(ax).toVar(), tang = vec3(1, 1, 1).sub(ax).toVar();
  const sum = [0, 1, 2, 3].map(() => vec3(0, 0, 0).toVar());
  const wsum = float(0).toVar();
  const nearOn = near ? tg.ab.near.equal(int(1)).and(tg.ab.sky.equal(int(1))) : null;
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
      const base = found.mul(int(GI_STRIDE)).add(int(GI_FILT)).toVar();
      const c = shUnpackTSL(GI_WORD_INDEX.map(k => buf.element(base.add(int(k)))));
      if (near) {
        // k: the near band's share of the light at the face normal (0..1 a
        // colour); op: the rays' near occlusion (their cosine weights sum to pi).
        const sb = found.mul(int(GI_STRIDE)).toVar();
        const vpOk = bitAnd(buf.element(sb.add(int(GI_FILLED))), uint(GI_DIRECT_OK)).notEqual(uint(0));
        const vt = near.vis;
        const vp = select(vpOk, unpackHalf2x16(buf.element(sb.add(int(GI_DIRECT + 1)))).y, vt).toVar();
        const op = clamp(c[5].x.div(float(Math.PI)), float(0), float(1)).toVar();
        const ot = clamp(op.add(vp).sub(vt), float(0), float(1)).toVar();
        const rf = min(float(1).sub(ot).div(max(float(1).sub(op), float(NEAR_FLOOR))), float(NEAR_RATIO_MAX)).toVar();
        const rn = min(ot.div(max(op, float(NEAR_OCC_FLOOR))), float(NEAR_RATIO_MAX)).toVar();
        const atN = c[0].mul(n.x).add(c[1].mul(n.y)).add(c[2].mul(n.z));
        const k = clamp(c[4].div(max(atN, vec3(1e-6, 1e-6, 1e-6))), vec3(0, 0, 0), vec3(1, 1, 1));
        const split = mix(vec3(rf, rf, rf), vec3(rn, rn, rn), k);
        const wf = select(nearOn, split, vec3(1, 1, 1)).mul(w).toVar();
        for (let k2 = 0; k2 < 4; k2++) sum[k2].addAssign(c[k2].mul(wf));
      } else {
        for (let k = 0; k < 4; k++) sum[k].addAssign(c[k].mul(w));
      }
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
  const m1 = sum[0].mul(nS.x).add(sum[1].mul(nS.y)).add(sum[2].mul(nS.z)).toVar();
  // Off the face: the L0 + L1 SH irradiance, E = L0 / 4 + (L1 . n) / 2 -
  // a sprite facing sideways under open sky sees half of it, not none.
  const light = (offFace
    ? select(tg.ab.sky.equal(int(1)), sum[3].mul(float(0.25)).add(m1.mul(float(0.5))), m1) : m1)
    .mul(float(1 / Math.PI));
  const out = max(light.mul(inv), vec3(0, 0, 0));
  return { light: select(tg.debug.equal(int(1)).and(wsum.lessThanEqual(float(0))), vec3(4, 0, 0), out),
           found: select(wsum.greaterThan(float(0)), float(1), float(0)) };
}

// The direct light cached on the patch holding the texel at p (n its axis
// normal) at level lvl: { light, ok } - ok 0 where the table has no such
// patch, or its direct light is not worked out yet. One patch, not four: the
// bounce off it is spread over a whole hemisphere of the patch receiving it.
export function patchDirectTSL(p, n, lvl, buf) {
  const { t, ax } = faceFrameTSL(p, n);
  const side = patchSideTSL(lvl);
  const pmin = t.add(floor(t.div(side)).mul(side).sub(t).mul(vec3(1, 1, 1).sub(ax)));
  const key = patchKeyTSL(pmin, faceIndexTSL(n), lvl);
  const found = int(-1).toVar();
  for (const slot of key.slots) {
    const b = slot.mul(int(GI_STRIDE));
    const ok = buf.element(b.add(int(GI_CHECK))).equal(key.check)
      .and(bitAnd(buf.element(b.add(int(GI_FILLED))), uint(GI_DIRECT_OK)).notEqual(uint(0)));
    If(found.lessThan(int(0)).and(ok), () => { found.assign(slot); });
  }
  const light = vec3(0, 0, 0).toVar();
  If(found.greaterThanEqual(int(0)), () => {
    const b = found.mul(int(GI_STRIDE)).add(int(GI_DIRECT)).toVar();
    const a = unpackHalf2x16(buf.element(b)), c = unpackHalf2x16(buf.element(b.add(int(1))));
    light.assign(vec3(a.x, a.y, c.x));
  });
  return { light, ok: found.greaterThanEqual(int(0)) };
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

// --- The patch kernels: each listed patch traces this frame's slice ---
//
// reset empties the list before the split kernels fill it; patches runs after
// them, sums after that, then direct (each due patch's direct light). Reads and writes the table through its read-write
// binding only - a dispatch may not bind one buffer both ways. Built with the
// shading (gpu/shade.js createShadowColorNode), whose lighting a ray's hit
// takes - ctx:
//   texelGI, cascades, terrain   the tables, the field, the terrain textures
//   trace        the plain reflection trace, if the shading has one
//   glass        the terrain textures in a world with glass (else false): the
//                rays then march the glass too, and tint by it
//   bias, sky, ambient, shadeMode, sun, sunOn, sunShadow, sunVisZero,
//   sunDirect, skyIrradiance, pointLight, mirrorTrace   as the shading's
//   giRead       the shading's bounce read, { light, cov } (blended levels)
//   smoke        gpu/smokeshadow.js smokeShared, or null: each ray's light is
//                dimmed by the cloud spells it crossed on its way
//   cones, nearReach   the shading's AO cones (createAOTSL) and their reach:
//                the near band is the rays hitting within it, V_p the cones'
//                visibility at the anchor (giDirect). null: no near band
export function createTexelGIKernels({ texelGI, cascades, terrain, trace, glass, bias, sky, ambient,
                                       shadeMode, sun, sunOn, sunShadow, sunVisZero, sunDirect,
                                       skyIrradiance, pointLight, mirrorTrace, giRead,
                                       cones = null, nearReach = null, smoke = null, leaves = null }) {
  const plainTrace = glass ? null : trace || createReflectTraceTSL(cascades);
  const glassTrace = glass ? createReflectTraceTSL(cascades, false, true) : null;
  const abSky = float(texelGI.ab.sky);
  const level = texelGILevelTSL(cascades);
  // The sky's radiance along d, in the ambient's units: its cosine-weighted
  // integral over a hemisphere, over pi, is skyIrradiance.
  const skyRadiance = d => (sky ? skyShTSL(sky, d).mul(ambient).mul(sky.gain) : vec3(ambient, ambient, ambient));
  const q = texelGI.queue, D = texelGI.data;
  // What lit smoke is lit by, for the rays crossing it: the sun and the sky,
  // evenly - each where the world lets it reach the smoke (gpu/smokeshadow.js).
  const smokeIn = smoke ? smokeInscatterTSL(smoke, terrain) : null;
  // ... and the trees' leaves, the same way (gpu/leafvolume.js).
  const leafIn = leaves ? leafInscatterTSL(terrain, leaves) : null;
  const smokeSun = smoke ? sunColourUniform.mul(float(1).sub(ambient)).mul(float(sunOn ? 1 : 0)) : null;
  const smokeSky = smoke ? skyIrradiance(vec3(0, 1, 0)).add(skyIrradiance(vec3(0, -1, 0))).mul(float(0.5)) : null;
  const reset = Fn(() => {
    const i = int(instanceIndex);
    atomicStore(q.element(int(GI_STAMPS).add(i)), uint(0));
    atomicStore(q.element(int(GI_ANCHORS).add(i)), uint(0xffffffff));
    If(i.equal(int(0)), () => {
      atomicStore(q.element(int(0)), uint(0));
      atomicStore(q.element(int(GI_CLAIMS)), uint(0));
    });
  })().compute(GI_SLOTS).setName('giPatchReset');
  // A listed patch: { slot, face, pmin, key, mine } - mine only where the
  // slot's stamp is this patch's (a higher check took it otherwise).
  const listed = i => {
    const e = int(1).add(i.mul(int(4))).toVar();
    const w0 = atomicLoad(q.element(e)).toVar();
    const slot = int(bitAnd(w0, uint(0xfffff))).toVar();
    const face = bitAnd(shiftRight(w0, uint(20)), uint(15)).toVar();
    const lvl = bitAnd(shiftRight(w0, uint(24)), uint(3)).toVar();
    const pmin = vec3(float(int(atomicLoad(q.element(e.add(int(1)))))),
                      float(int(atomicLoad(q.element(e.add(int(2)))))),
                      float(int(atomicLoad(q.element(e.add(int(3))))))).toVar();
    const key = patchKeyTSL(pmin, face, lvl);
    const mine = atomicLoad(q.element(int(GI_STAMPS).add(slot))).equal(key.check);
    return { slot, face, lvl, pmin, key, mine };
  };
  // A listed patch's axis normal, and its anchor texel's centre on the face
  // (GI_ANCHORS) - the patch's own centre only if no texel offered one.
  const anchorOf = (slot, face, lvl, pmin) => {
    const fi = int(face), axis = fi.div(int(2));
    const sgn = select(fi.mod(int(2)).equal(int(0)), float(1), float(-1));
    const n = vec3(select(axis.equal(int(0)), sgn, float(0)), select(axis.equal(int(1)), sgn, float(0)),
                   select(axis.equal(int(2)), sgn, float(0))).toVar();
    const ax = abs(n).toVar();
    const qm = float(VOXEL_METRES);
    const e1 = vec3(float(1).sub(ax.x), ax.x, float(0)), e2 = vec3(float(0), ax.z, float(1).sub(ax.z));
    const anc = atomicLoad(q.element(int(GI_ANCHORS).add(slot))).toVar();
    const half = patchSideTSL(lvl).mul(float(0.5)).sub(float(0.5));
    const noAnchor = anc.equal(uint(0xffffffff)).or(texelGI.ab.anchor.equal(int(0)));
    const ta = select(noAnchor, half, float(bitAnd(shiftRight(anc, uint(4)), uint(15))));
    const tb = select(noAnchor, half, float(bitAnd(anc, uint(15))));
    const pc = pmin.add(ax.mul(float(0.5))).add(e1.mul(ta.add(float(0.5)))).add(e2.mul(tb.add(float(0.5))))
                 .mul(qm).add(n.mul(qm.mul(float(0.5)))).toVar();
    return { n, ax, pc };
  };
  // The sun and the light list at p (normal n), shadowed, added to light: a
  // patch's direct light, or - where no patch holds it - a ray hit's. skip:
  // the card left out (a sprite's own).
  const directAt = (p, n, light, skip = NO_SKIP) => {
    if (sunOn) {
      const vis = sunVisZero();
      If(n.dot(sun.dir).greaterThan(float(0)), () => { vis.assign(sunShadow(p, n, skip)); });
      light.addAssign(sunColourUniform.mul(sunDirect(vis, n, shadeMode)));
    }
    // The light list WITH its shadows, as the surface itself is lit on screen
    // (the shadow budget). Unshadowed, a lamp behind a wall lit the ground in
    // front of it at the hit, and that bounced onto every face round the
    // wall: light through walls.
    if (pointLight) {
      const mode = select(texelGI.ab.hitShadows.equal(int(1)), float(0), float(2));
      light.addAssign(pointLight.withSpecular(p, n, skip, n, null, mode));
    }
  };
  // The nearest sprite box the ray (o, d) enters before tMax, from outside
  // (a patch under a sprite's feet starts in its box: the contact disc darkens
  // it instead). { t (tMax: none), i (-1: none) }.
  const spriteOnRay = (o, d, tMax) => {
    const best = tMax.toVar(), found = int(-1).toVar();
    const inv = vec3(1, 1, 1).div(mix(d, vec3(1e-7, 1e-7, 1e-7), step(abs(d), vec3(1e-7, 1e-7, 1e-7)))).toVar();
    Loop({ start: int(0), end: texelGI.spriteCount, type: 'int', condition: '<', name: 'gsp' }, ({ gsp }) => {
      const A = texelGI.sprites.element(gsp.mul(int(3))).toVar();
      const half = vec3(A.w, texelGI.sprites.element(gsp.mul(int(3)).add(int(1))).w, A.w);
      const a = A.xyz.sub(half).sub(o).mul(inv), b = A.xyz.add(half).sub(o).mul(inv);
      const lo = min(a, b), hi = max(a, b);
      const tIn = max(max(lo.x, lo.y), lo.z), tOut = min(min(hi.x, hi.y), hi.z);
      If(tIn.greaterThan(float(0)).and(tIn.lessThanEqual(tOut)).and(tIn.lessThan(best)), () => {
        best.assign(tIn); found.assign(gsp);
      });
    });
    return { t: best, i: found };
  };
  // A patch's own turn of the direction set, fixed by its key: an angle round
  // the normal and a shift within each direction's band of z (both kept
  // stratified). Neighbours then sample between each other's directions, and
  // a small bright thing that fell between one patch's rays and on the
  // next's - blotches, patch-sized - becomes fine variation the filter evens.
  // Never changes from frame to frame. ab.rotate 0: the shared set.
  const turnOf = check => {
    const on = float(texelGI.ab.rotate);
    return { a: float(bitAnd(check, uint(0xffff))).mul(float(2 * Math.PI / 65536)).mul(on).toVar(),
             z: float(bitAnd(shiftRight(check, uint(16)), uint(0xffff))).div(float(65536)).sub(float(0.5))
                  .mul(on).toVar() };
  };
  const patches = Fn(() => {
    const i = int(instanceIndex).toVar();
    const count = min(int(atomicLoad(q.element(int(0)))), int(GI_LIST_CAP));
    If(i.lessThan(count), () => {
      const { slot, face, lvl, pmin, key, mine } = listed(i);
      If(mine, () => {
        const { n, ax, pc } = anchorOf(slot, face, lvl, pmin);
        const qm = float(VOXEL_METRES);
        const helper = select(ax.y.lessThan(float(0.5)), vec3(0, 1, 0), vec3(1, 0, 0));
        const T = normalize(cross(helper, n)).toVar();
        const B = cross(n, T).toVar();
        const origin = pc.add(n.mul(bias.mul(qm))).toVar();
        const base = slot.mul(int(GI_STRIDE)).toVar();
        // A slot newly this patch's: no slices yet (the sum kernel rewrites
        // the sum this frame, so nothing needs clearing).
        If(D.element(base.add(int(GI_CHECK))).notEqual(key.check), () => {
          D.element(base.add(int(GI_CHECK))).assign(key.check);
          D.element(base.add(int(GI_FILLED))).assign(uint(GI_FRESH));
          atomicAdd(q.element(int(GI_CLAIMS)), uint(1));
        });
        D.element(base.add(int(GI_LAST))).assign(texelGI.frame);
        const state = D.element(base.add(int(GI_FILLED))).toVar();
        const have = bitAnd(state, uint(GI_SLICE_MASK)).toVar();
        const next = int(bitAnd(shiftRight(state, uint(16)), uint(GI_SLICES - 1))).toVar();
        // The next slice in turn - and, for a patch new to the table, every
        // slice it lacks: it starts at the full GI_DIRS-ray answer instead of
        // one slice scaled up (8 rays: a rough guess that refined over the
        // next frames, and shimmered wherever patches were new).
        Loop({ start: int(0), end: int(GI_SLICES), type: 'int', condition: '<', name: 'gsl' }, ({ gsl }) => {
        If(gsl.equal(next).or(bitAnd(have, shiftLeft(uint(1), uint(gsl))).equal(uint(0))), () => {
        const sl = gsl;
        const c = Array.from({ length: GI_VEC3 }, () => vec3(0, 0, 0).toVar());
        const turn = turnOf(key.check);
        Loop({ start: int(0), end: int(GI_DIRS / GI_SLICES), type: 'int', condition: '<', name: 'gk' }, ({ gk }) => {
          // Direction k of the set (GI_DIRS above), turned by this patch's
          // own angle and shifted within its band of z by its own share.
          const k = float(sl.add(gk.mul(int(GI_SLICES))));
          const z = float(1).sub(k.add(float(0.5)).add(turn.z).div(float(GI_DIRS)));
          const ang = k.mul(float(GOLDEN_ANGLE)).add(turn.a);
          const rr = sqrt(max(float(1).sub(z.mul(z)), float(0)));
          const d = vec3(rr.mul(cos(ang)), rr.mul(sin(ang)), z);
          const dir = normalize(T.mul(d.x).add(B.mul(d.y)).add(n.mul(d.z))).toVar();
          const traced = glassTrace ? giTraceHitTSL(glassTrace(origin, dir, texelGI.reach).toVar()) : null;
          const hit = (traced ? traced.hit : plainTrace(origin, dir, texelGI.reach)).toVar();
          const hitAt = origin.add(dir.mul(max(hit.w, float(0)))).toVar();
          const L = vec3(0, 0, 0).toVar();
          // A sprite before the hit (above), and the one light evaluation of
          // the ray: at the sprite, or at a hit with no cached direct light.
          const sp = spriteOnRay(origin, dir, select(hit.w.greaterThanEqual(float(0)), hit.w, texelGI.reach));
          const onSprite = sp.i.greaterThanEqual(int(0)).toVar();
          const need = int(0).toVar();
          const dp = vec3(0, 0, 0).toVar(), dn = vec3(0, 1, 0).toVar();
          const dMul = vec3(0, 0, 0).toVar();
          If(hit.w.greaterThanEqual(float(0)).and(inHitBoundsTSL(hitAt.sub(hit.xyz.mul(float(VOXEL_METRES * 0.5))))), () => {
            const ht = hitTexelTSL(terrain, hitAt, hit.xyz, dir);
            const hp = ht.p.toVar(), nh = ht.n.toVar();
            const hs = hitSurfaceTSL(terrain, hp, nh, null);
            If(hs.emissive.greaterThan(float(0.5)), () => { L.assign(hs.raw.mul(texelGI.strength)); }).Else(() => {
              // The hit's own patch carries its sky (and its bounce), where
              // the table has one. Where not, with ab.sky, the sky only if
              // the hit is not under a roof - an unoccluded sky on every
              // wall a ray met lit rooms from inside.
              const g = giRead(hp, nh, nh, D);
              const open = skyOpenTSL(terrain, hp.add(nh.mul(qm.mul(float(0.5)))));
              const keep = float(1).sub(abSky.mul(g.cov.add(float(1).sub(g.cov).mul(float(1).sub(open)))));
              const light = skyIrradiance(nh).mul(keep).toVar();
              // The sun and the lamps: the hit patch's cached direct light,
              // where the table has it - one light loop a patch every few
              // frames, not one a ray. Else lit here, as the patch would be.
              const cached = patchDirectTSL(hp, nh, level(hp).lvl, D);
              If(cached.ok.and(texelGI.ab.direct.equal(int(1))), () => { light.addAssign(cached.light); })
                .ElseIf(onSprite.not(), () => { need.assign(int(1)); dp.assign(hp); dn.assign(nh); });
              light.addAssign(g.light);
              // What it bounces: its diffuse share, (1 - F)(1 - metal) of its
              // albedo - none for a metal, whose albedo is its F0 - plus the
              // specular share F, but only the part too rough for the mirror
              // light when that is on (it throws the smooth part itself, and
              // counting it here too was double). As the LPV's blockBounce.
              // Times the GI strength: a gain on what each bounce gives back,
              // not on the sky.
              const sd = decodeSpecularTSL(hs.spec, hs.raw);
              const F = fresnelRoughTSL(sd.f0, max(nh.dot(dir.negate()), float(1e-4)), sd.alpha).toVar();
              const kd = vec3(1, 1, 1).sub(F).mul(float(1).sub(sd.metal));
              const specShare = mirrorTrace
                ? F.mul(smoothstep(float(REFLECT_ROUGH_START), float(REFLECT_ROUGH_END), sd.alpha)) : F;
              dMul.assign(select(texelGI.ab.physical.equal(int(1)), hs.raw.mul(kd).add(specShare), hs.raw)
                            .mul(texelGI.strength));
              L.assign(light.mul(dMul));
            });
          }).ElseIf(hit.w.lessThan(float(0)), () => {
            // A miss: the sky - if the ray truly escaped. Where it ended (out
            // of reach, or out of the field) must be above its column's roof.
            const end = origin.add(dir.mul(texelGI.reach));
            L.assign(skyRadiance(dir).mul(abSky).mul(skyOpenTSL(terrain, end)));
          });
          // The sprite: a face turned back along the ray, lit by the sky over
          // it and (below) the sun and lamps, its own card left out.
          const sA = texelGI.sprites.element(max(sp.i, int(0)).mul(int(3))).toVar();
          const nBack = vec3(dir.x.negate(), float(0), dir.z.negate()).toVar();
          const sn = select(length(nBack).greaterThan(float(0.1)), normalize(nBack), dir.negate()).toVar();
          const sLight = vec3(0, 0, 0).toVar();
          If(onSprite, () => {
            need.assign(int(1)); dp.assign(sA.xyz); dn.assign(sn);
            sLight.assign(skyIrradiance(sn).mul(skyOpenTSL(terrain, sA.xyz)));
          });
          If(need.equal(int(1)), () => {
            const dl = vec3(0, 0, 0).toVar();
            directAt(dp, dn, dl, select(onSprite, sA.xz, NO_SKIP));
            If(onSprite, () => { sLight.addAssign(dl); }).Else(() => { L.addAssign(dl.mul(dMul)); });
          });
          If(onSprite, () => {
            const sB = texelGI.sprites.element(sp.i.mul(int(3)).add(int(1)));
            const cover = texelGI.sprites.element(sp.i.mul(int(3)).add(int(2))).x;
            L.assign(mix(L, sLight.mul(sB.xyz).mul(texelGI.strength), cover));
          });
          // Through glass: tinted by the metres it spent in it.
          if (traced) L.mulAssign(glassRayTransmitTSL(glass, traced.glass, origin, dir));
          // Through cloud spells: to the hit (or the sprite), or the whole reach for a ray that
          // escaped - held back by the smoke, and given the light of the lit
          // smoke it crossed (a bounce, so times the GI strength).
          if (smoke) {
            const sc = smokeIn(origin, dir, sp.t,
                               sun.dir, smokeSun, smokeSky).toVar();
            L.assign(L.mul(exp(sc.w.negate())).add(sc.xyz.mul(texelGI.strength)));
          }
          if (leaves) {
            const lc = leafIn(origin, dir, sp.t, sun.dir, sunColourUniform.mul(float(1).sub(ambient)).mul(float(sunOn ? 1 : 0)),
                              skyIrradiance(vec3(0, 1, 0))).toVar();
            L.assign(L.mul(exp(lc.w.negate())).add(lc.xyz.mul(texelGI.strength)));
          }
          c[0].addAssign(L.mul(dir.x));
          c[1].addAssign(L.mul(dir.y));
          c[2].addAssign(L.mul(dir.z));
          c[3].addAssign(L);
          // The near band: a hit within the cones' reach, weighed as they
          // weigh blocking (1 - t / reach), at the face normal (d.z).
          if (nearReach) {
            const a = select(hit.w.greaterThanEqual(float(0)),
              max(float(1).sub(hit.w.div(nearReach)), float(0)), float(0));
            c[4].addAssign(L.mul(a.mul(d.z)));
            c[5].addAssign(vec3(a.mul(d.z), 0, 0));
          }
        });
        // Uniform over the hemisphere: each ray stands for 2 pi / GI_DIRS sr.
        const wq = float(2 * Math.PI / GI_DIRS);
        const packed = shPackTSL(c.map(x => x.mul(wq)));
        const sb = base.add(sl.mul(int(GI_WORDS))).toVar();
        packed.forEach((w, k) => D.element(sb.add(int(k))).assign(w));
        });
        });
        D.element(base.add(int(GI_FILLED))).assign(bitOr(bitOr(uint(GI_SLICE_MASK), bitAnd(state, uint(GI_FRESH | GI_DIRECT_OK))),
          shiftLeft(bitAnd(uint(next.add(int(1))), uint(GI_SLICES - 1)), uint(16))));
      });
    });
  })().compute(GI_LIST_CAP).setName('giPatches');
  // Each listed patch's slices summed - after the patch kernel, so its rays
  // read only sums no thread is writing.
  const sums = Fn(() => {
    const i = int(instanceIndex).toVar();
    const count = min(int(atomicLoad(q.element(int(0)))), int(GI_LIST_CAP));
    If(i.lessThan(count), () => {
      const { slot, key, mine } = listed(i);
      If(mine.and(D.element(slot.mul(int(GI_STRIDE)).add(int(GI_CHECK))).equal(key.check)), () => {
        const base = slot.mul(int(GI_STRIDE)).toVar();
        const filled = bitAnd(D.element(base.add(int(GI_FILLED))), uint(GI_SLICE_MASK)).toVar();
        // The sum of every slice traced so far, scaled up while some are
        // still missing (a patch new to the table).
        const sum = Array.from({ length: GI_VEC3 }, () => vec3(0, 0, 0).toVar());
        const have = float(0).toVar();
        Loop({ start: int(0), end: int(GI_SLICES), type: 'int', condition: '<', name: 'gs' }, ({ gs }) => {
          If(bitAnd(filled, shiftLeft(uint(1), uint(gs))).notEqual(uint(0)), () => {
            const o = base.add(gs.mul(int(GI_WORDS))).toVar();
            const v = shUnpackTSL(GI_WORD_INDEX.map(k => D.element(o.add(int(k)))));
            for (let k = 0; k < GI_VEC3; k++) sum[k].addAssign(v[k]);
            have.addAssign(float(1));
          });
        });
        const scale = float(GI_SLICES).div(max(have, float(1)));
        // Shown: smoothed toward the new sum by blend (the response time) -
        // or the new sum itself while the patch is fresh.
        const state = D.element(base.add(int(GI_FILLED))).toVar();
        const fresh = bitAnd(state, uint(GI_FRESH)).notEqual(uint(0));
        const a = select(fresh, float(1), texelGI.blend).toVar();
        const old = shUnpackTSL(GI_WORD_INDEX.map(k => D.element(base.add(int(GI_SUM + k)))));
        const out = shPackTSL(sum.map((x, k) => mix(old[k], x.mul(scale), a)));
        out.forEach((w, k) => D.element(base.add(int(GI_SUM + k))).assign(w));
        D.element(base.add(int(GI_FILLED))).assign(bitAnd(state, uint(~GI_FRESH >>> 0)));
      });
    });
  })().compute(GI_LIST_CAP).setName('giSums');
  // Each listed patch's direct light, when due: new to the slot, or its turn -
  // every directEvery frames, a whole workgroup at a time, so a warp marches
  // together or not at all (a patch keeps its place in the list from frame
  // to frame, near enough). The rays read it next frame.
  const direct = Fn(() => {
    const i = int(instanceIndex).toVar();
    const count = min(int(atomicLoad(q.element(int(0)))), int(GI_LIST_CAP));
    If(i.lessThan(count), () => {
      const { slot, face, lvl, pmin, key, mine } = listed(i);
      const base = slot.mul(int(GI_STRIDE)).toVar();
      If(mine.and(D.element(base.add(int(GI_CHECK))).equal(key.check)), () => {
        const state = D.element(base.add(int(GI_FILLED))).toVar();
        const turn = uint(i.div(int(64))).add(texelGI.frame).mod(uint(texelGI.directEvery));
        If(bitAnd(state, uint(GI_DIRECT_OK)).equal(uint(0)).or(turn.equal(uint(0))), () => {
          const { n, pc } = anchorOf(slot, face, lvl, pmin);
          const light = vec3(0, 0, 0).toVar();
          directAt(pc, n, light);
          // V_p: the cones' visibility at the anchor, as a texel there takes
          // it (patchLightTSL's near split).
          const vp = cones ? cones(pc, n, n, NO_SKIP).vis : float(1);
          const b = base.add(int(GI_DIRECT));
          D.element(b).assign(packHalf2x16(vec2(light.x, light.y)));
          D.element(b.add(int(1))).assign(packHalf2x16(vec2(light.z, vp)));
          D.element(base.add(int(GI_FILLED))).assign(bitOr(state, uint(GI_DIRECT_OK)));
        });
      });
    });
  })().compute(GI_LIST_CAP).setName('giDirect');
  // Each listed patch's light blurred with its eight neighbours on its plane
  // (the same face and level; 4 the patch, 2 a side, 1 a corner - over the
  // ones the table holds, fresh), into FILT, which the shading and the rays'
  // hits read. Another plane never mixes in, so a corner stays a corner.
  // After the sums: it reads only sums no thread is writing. ab.filter 0: the
  // patch's own sum.
  const filter = Fn(() => {
    const i = int(instanceIndex).toVar();
    const count = min(int(atomicLoad(q.element(int(0)))), int(GI_LIST_CAP));
    If(i.lessThan(count), () => {
      const { slot, face, lvl, pmin, key, mine } = listed(i);
      const base = slot.mul(int(GI_STRIDE)).toVar();
      If(mine.and(D.element(base.add(int(GI_CHECK))).equal(key.check)), () => {
        const sumOf = b => shUnpackTSL(GI_WORD_INDEX.map(k => D.element(b.add(int(GI_SUM + k)))));
        const sum = sumOf(base).map(v => v.mul(float(4)).toVar());
        const wsum = float(4).toVar();
        If(texelGI.ab.filter.equal(int(1)), () => {
          const axis = int(face).div(int(2));
          const e1 = select(axis.equal(int(0)), vec3(0, 1, 0), vec3(1, 0, 0));
          const e2 = select(axis.equal(int(2)), vec3(0, 1, 0), vec3(0, 0, 1));
          const side = patchSideTSL(lvl);
          Loop({ start: int(0), end: int(9), type: 'int', condition: '<', name: 'gf' }, ({ gf }) => {
            const da = gf.mod(int(3)).sub(int(1)), db = gf.div(int(3)).sub(int(1));
            If(gf.notEqual(int(4)), () => {
              const nk = patchKeyTSL(pmin.add(e1.mul(float(da)).add(e2.mul(float(db))).mul(side)), face, lvl);
              const found = int(-1).toVar();
              for (const s of nk.slots) {
                const b = s.mul(int(GI_STRIDE));
                const ok = D.element(b.add(int(GI_CHECK))).equal(nk.check)
                  .and(texelGI.frame.sub(D.element(b.add(int(GI_LAST)))).lessThanEqual(uint(GI_STALE_FRAMES)));
                If(found.lessThan(int(0)).and(ok), () => { found.assign(s); });
              }
              If(found.greaterThanEqual(int(0)), () => {
                const w = select(da.equal(int(0)).or(db.equal(int(0))), float(2), float(1));
                const v = sumOf(found.mul(int(GI_STRIDE)));
                for (let k = 0; k < GI_VEC3; k++) sum[k].addAssign(v[k].mul(w));
                wsum.addAssign(w);
              });
            });
          });
        });
        shPackTSL(sum.map(v => v.div(wsum))).forEach((w, k) => D.element(base.add(int(GI_FILT + k))).assign(w));
      });
    });
  })().compute(GI_LIST_CAP).setName('giFilter');
  return { reset, patches, sums, filter, direct };
}

// --- Reading the patches ---
//
// The shading's reads, given texelGILevelTSL's giLevel:
//   giRead(p, n, nS, buf, fresh, offFace, near) -> { light, cov }: the bounce
//     at a texel, blended across a level step. buf: the read-only table
//     (shading: fresh patches only), or the patch kernel's read-write one (a
//     ray's hit: any). cov: how much of it the table answered (0 where no
//     patch, fading out past C2), for the sky ambient to fill the rest. near:
//     { vis }, the texel's cone visibility - patchLightTSL's near split.
//   texelAmbient(p, n, nS, fresh, offFace, near) -> { light, keep }: with
//     ab.sky the patches hold the sky - keep is what is left of the sky
//     ambient where it reads them.
//   abSky, abNear: the A/Bs as floats; abNear is ab.near with ab.sky - the
//     cones split the patch light rather than multiplying it.
export function createTexelGIReadTSL(texelGI, giLevel) {
  const giRead = (p, n, nS, buf = texelGI.dataRead, fresh = buf === texelGI.dataRead, offFace = false,
                  near = null) => {
    const L = giLevel(p);
    const a = patchLightTSL(texelGI, p, n, nS, L.lvl, buf, fresh, offFace, near);
    const out = a.light.toVar(), cov = a.found.toVar();
    If(L.blend.greaterThan(float(0)), () => {
      const b = patchLightTSL(texelGI, p, n, nS, L.next, buf, fresh, offFace, near);
      out.assign(mix(out, b.light, L.blend));
      cov.assign(mix(cov, b.found, L.blend));
    });
    return { light: out.mul(L.fade), cov: cov.mul(L.fade) };
  };
  const abSky = float(texelGI.ab.sky);
  const abNear = float(texelGI.ab.near).mul(abSky);
  const texelAmbient = (p, n, nS = n, fresh = true, offFace = false, near = null) => {
    const r = giRead(p, n, nS, texelGI.dataRead, fresh, offFace, near);
    return { light: r.light, keep: float(1).sub(abSky.mul(r.cov)) };
  };
  return { giRead, texelAmbient, abSky, abNear };
}

// --- Listing the patches on screen ---
//
// Returns registerPatch(p, n): the split base kernel lists the patch of the
// sample at p (locked on a box face, n its axis normal) for the patch kernel -
// and in a level blend band, the next level's too. giLevel: texelGILevelTSL's.
export function createPatchRegisterTSL(texelGI, giLevel) {
  const registerLevel = (p, n, lvl) => {
    const { t, ax, e1, e2 } = faceFrameTSL(p, n);
    const side = patchSideTSL(lvl);
    const pmin = t.add(floor(t.div(side)).mul(side).sub(t).mul(vec3(1, 1, 1).sub(ax))).toVar();
    // This texel within the patch, and how far from its centre (doubled, squared).
    const ta = t.sub(pmin).dot(e1), tb = t.sub(pmin).dot(e2);
    const da = ta.mul(float(2)).sub(side.sub(float(1))), db = tb.mul(float(2)).sub(side.sub(float(1)));
    const anchor = bitOr(shiftLeft(uint(da.mul(da).add(db.mul(db))), uint(8)),
                         uint(ta.mul(float(16)).add(tb))).toVar();
    const face = faceIndexTSL(n).toVar();
    const key = patchKeyTSL(pmin, face, lvl);
    const mine = key.check;
    const done = int(0).toVar();
    const q = texelGI.queue, R = texelGI.dataRead;
    // The slot to stamp first: the one this patch holds; else a free or
    // stale one; else the first. Owners keep their slots, so two patches on
    // one slot no longer race for it (and lose their light) every frame.
    const [s1, s2] = key.slots;
    const own = k => R.element(k.mul(int(GI_STRIDE)).add(int(GI_CHECK)));
    const open = k => own(k).equal(uint(0)).or(
      texelGI.frame.sub(R.element(k.mul(int(GI_STRIDE)).add(int(GI_LAST)))).greaterThan(uint(GI_STALE_FRAMES)));
    const secondFirst = own(s2).equal(mine).or(own(s1).notEqual(mine).and(open(s1).not()).and(open(s2)));
    const first = select(secondFirst, s2, s1).toVar(), other = select(secondFirst, s1, s2).toVar();
    for (const slot of [first, other]) {
      If(done.equal(int(0)), () => {
        // Through int(), as the miss list's atomicAdd: uint() of a uint is
        // no conversion at all, and the atomic's old value was lost - every
        // sample listed its patch again (108k a frame, the list overflowed).
        const old = uint(int(atomicMax(q.element(int(GI_STAMPS).add(slot)), mine))).toVar();
        // Listed already (by this patch): done. Free, or held by a lower
        // check: list it. Held by a higher one: the next probe.
        If(old.equal(mine), () => {
          atomicMin(q.element(int(GI_ANCHORS).add(slot)), anchor);
          done.assign(int(1));
        })
          .ElseIf(old.lessThan(mine), () => {
            atomicMin(q.element(int(GI_ANCHORS).add(slot)), anchor);
            const at = int(atomicAdd(q.element(int(0)), uint(1))).toVar();
            If(at.lessThan(int(GI_LIST_CAP)), () => {
              const e = int(1).add(at.mul(int(4)));
              atomicStore(q.element(e), bitOr(bitOr(uint(slot), shiftLeft(face, uint(20))), shiftLeft(lvl, uint(24))));
              atomicStore(q.element(e.add(int(1))), uint(int(pmin.x)));
              atomicStore(q.element(e.add(int(2))), uint(int(pmin.y)));
              atomicStore(q.element(e.add(int(3))), uint(int(pmin.z)));
            });
            done.assign(int(1));
          });
      });
    }
  };
  // p's patch, and in a blend band the next level's as well.
  const registerPatch = (p, n) => {
    const L = giLevel(p);
    registerLevel(p, n, L.lvl);
    If(L.blend.greaterThan(float(0)), () => registerLevel(p, n, L.next));
  };
  return registerPatch;
}
