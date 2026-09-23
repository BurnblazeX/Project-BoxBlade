import { signedDistanceTransform, facingToward } from './silhouette.js';
import { encodeDistance } from './boxgrid.js';

// --- Sprite shadows as analytic cards, one facing per light ---
//
// The distance field is GEOMETRY, shared by every light, so a card baked into it
// points one way and every other light gets the wrong silhouette. The way out is
// to stop baking sprites at all: keep them outside the field and intersect each
// shadow ray with a card turned to face THAT light.
//
// Cheaper than it sounds. A card is a plane, so the intersection is O(1) per
// sprite per RAY - not per march step, which is what made voxelising them cost
// anything. And being analytic it has no voxel size, so the cascade-scale seam
// that a baked silhouette has cannot arise here at all.
//
// The softening is not a second mechanism either: the cone trace's visibility is
// min(d / (R*t)), and that is exactly what this computes with d read from the
// silhouette's own distance transform at the hit point.
//
// CPU reference. gpu.js carries the TSL port, and the two are meant to be diffed
// the way sphereTrace and traceDistanceTSL already are.

// Pixels of margin around the mask so the soft falloff has somewhere to grow.
// The cone radius at the far end of a sun march is a few sprite pixels, so eight
// is comfortably past where the falloff has reached 1.
export const CARD_PAD = 8;

// rgba (optional): the sprite's own pixels, top-down, w x h. Only reflections
// need it - a reflection ray that lands on a card reads its colour.
export function createCard({ mask, w, h, widthMetres, heightMetres, rgba = null }) {
  const pw = w + CARD_PAD * 2, ph = h + CARD_PAD * 2;
  const padded = new Uint8Array(pw * ph);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) padded[(y + CARD_PAD) * pw + (x + CARD_PAD)] = 1;
    }
  }
  return {
    dt: signedDistanceTransform(padded, pw, ph),
    w: pw, h: ph, cols: w, rows: h,
    widthMetres, heightMetres, rgba,
    // Native sprite resolution, NOT reduced to any voxel size. That reduction is
    // what a baked silhouette needs and what made it cascade-dependent; an
    // analytic card has no grid to match.
    metresPerPixel: widthMetres / w
  };
}

// Signed distance to the silhouette, in METRES, at a point on the card measured
// from its centre. Positive outside.
//
// Bilinear, and clamped to the padded rect - a sample past the pad is further
// from the shape than the falloff can reach, so the border value is already the
// right answer there.
export function cardDistance(card, a, b, flip = false) {
  const mpp = card.metresPerPixel;
  const cu = (flip ? -a : a) / mpp + card.cols / 2 + CARD_PAD;
  // Image rows run top-down, b runs up.
  const cv = card.rows / 2 - b / mpp + CARD_PAD;

  const fu = cu - 0.5, fv = cv - 0.5;
  const x0 = Math.floor(fu), y0 = Math.floor(fv);
  const tx = fu - x0, ty = fv - y0;
  const cl = (i, n) => (i < 0 ? 0 : i >= n ? n - 1 : i);
  const xa = cl(x0, card.w), xb = cl(x0 + 1, card.w);
  const ya = cl(y0, card.h), yb = cl(y0 + 1, card.h);
  const ra = ya * card.w, rb = yb * card.w;
  const top = card.dt[ra + xa] + (card.dt[ra + xb] - card.dt[ra + xa]) * tx;
  const bot = card.dt[rb + xa] + (card.dt[rb + xb] - card.dt[rb + xa]) * tx;
  return (top + (bot - top) * ty) * mpp;
}

// The sprite pixel at card-local (a, b) metres, as [column, row] of the
// sprite image (row 0 at the top), or null off the sprite. What a reflection
// ray that crossed the card's plane there sees - nearest, because the art is
// pixel art. Same mapping as cardDistance, minus the pad and the filter; the
// TSL hit test in gpu.js (createCardHitTSL) mirrors this.
export function cardPixelAt(card, a, b, flip = false) {
  const mpp = card.metresPerPixel;
  const col = Math.floor((flip ? -a : a) / mpp + card.cols / 2);
  const row = Math.floor(card.rows / 2 - b / mpp);
  if (col < 0 || col >= card.cols || row < 0 || row >= card.rows) return null;
  return [col, row];
}

// The card's orientation for a given light, as its width direction in XZ.
//
// Per LIGHT and per CARD: a point light is in a different direction from every
// sprite, so this is evaluated from the sprite's own position, not from a global
// light vector. Shares facingToward with the baked path so the two cannot
// disagree about handedness.
export function cardFacingFor(centre, lightX, lightZ) {
  return facingToward(lightX - centre.x, lightZ - centre.z);
}

// How much of a light this card leaves visible from `from`, along `dir`.
//
// 1 is unoccluded, 0 fully blocked, between is the penumbra. coneSlope is the
// light's angular radius as a slope, so coneSlope * t is the cone's radius at
// the hit - the same quantity createConeTraceSunTSL divides by.
// Where a card's shadow starts dissolving as the ray runs out of reach, as a
// fraction of maxDist. The same constant and the same reasoning as the terrain's
// SHADOW_FADE_START: a march ends either because it hit something or because it
// ran past its cap, and only the first is a fact about the world.
export const CARD_FADE_START = 0.75;

// How far a card is allowed to cast for a DIRECTIONAL light, in metres.
//
// Deliberately not the march cap. That 12 m is a property of stepping through
// the distance field - how far we are willing to walk before giving up - and a
// card has no traversal at all: it is one plane intersection whatever the range.
// Inheriting the march's cap therefore bought nothing and cost a straight line
// across the ground perpendicular to the sun, where every sprite's shadow ended
// at once. At a low sun a 4.5 m tree casts nearly 12 m, so the cut landed right
// where the shadow was still going.
//
// Sized past the outermost cascade so the fade sits far outside anything the
// player can see. A point light keeps its own cap - the distance to the light -
// because that one IS physical: nothing past a lamp can shadow it.
export const CARD_SUN_REACH = 64;

// Cascade LOD - the mirror of the block in createCardsTSL. lod's integer part is
// the cascade the card stands in, its fraction how far into the band toward
// the next. Past C0 the hit point snaps to that level's voxel grid; across the
// band the two resolutions mix; past the outermost level the card fades out.
// C0's voxel size and the level count are copies, so this module stays free of
// the grid; cards.test.mjs asserts they match boxgrid.js.
export const CARD_LOD_C0_METRES = 0.125;
export const CARD_LOD_LEVELS = 3;
// Per light: lodShift draws its cards that many levels finer than their
// cascade, lodMax caps how coarse they get. The sun uses shift 1, max 1 - full
// resolution through C1's range, C1 quality in C2's, never C2's 50 cm. Point
// lights use shift 0 and the full ladder.
export const SUN_CARD_LOD_SHIFT = 1;
export const SUN_CARD_LOD_MAX = 1;
const lodQuantum = (l, shift, lodMax) => {
  const ql = Math.min(Math.max(l - shift, 0), lodMax);
  return ql < 0.5 ? 0 : CARD_LOD_C0_METRES * 2 ** ql;
};
const lodSnap = (v, half, q) => (q > 0 ? (Math.floor((v + half) / q) + 0.5) * q - half : v);

export function cardVisibility({ card, centre, facing, from, dir, maxDist,
                                 coneSlope = 0, flip = false,
                                 fadeStart = CARD_FADE_START, lod = 0, lodShift = 0,
                                 lodMax = CARD_LOD_LEVELS - 1 }) {
  const nx = -facing.uz, nz = facing.ux;
  const denom = dir.x * nx + dir.z * nz;
  // Edge-on. Cannot happen for a card built to face this light, but a locked or
  // stale facing can produce it, and a near-zero divisor would otherwise throw
  // the hit point to infinity.
  if (Math.abs(denom) < 1e-6) return 1;

  const t = ((centre.x - from.x) * nx + (centre.z - from.z) * nz) / denom;
  // Behind the receiver, or past the light: not between them, so not an
  // occluder. The maxDist cap is what stops a card beyond a point light from
  // shadowing it.
  if (t <= 0 || t >= maxDist) return 1;

  const hx = from.x + dir.x * t - centre.x;
  const hy = from.y + dir.y * t - centre.y;
  const hz = from.z + dir.z * t - centre.z;

  const a = hx * facing.ux + hz * facing.uz;

  // The transform, and then the card's own OUTLINE, and take whichever is
  // further. The transform saturates at CARD_PAD pixels - one metre - because
  // that is all the margin it was given, and past that it reports "one metre"
  // however far away the point really is. That is fine while the cone is narrow,
  // and wrong the moment it is not: at a 64 m reach the sun's cone is nearly
  // four metres across, so d/(R*t) came out below one EVERYWHERE and every card
  // dimmed the whole map. The distance field had exactly this failure at +-1
  // voxel; see DISTANCE_RANGE in boxgrid.js.
  //
  // The silhouette is inside its own rectangle, so the distance to that
  // rectangle is a valid floor for the distance to the shape - exact far away,
  // negative (and so ignored) anywhere inside. No extra padding, no bigger
  // atlas, and the far field is right by construction rather than by having
  // guessed a large enough margin.
  const hw = card.widthMetres / 2, hh = card.heightMetres / 2;
  const distAt = (sa, sb) => {
    const ox = Math.abs(sa) - hw, oy = Math.abs(sb) - hh;
    const rx = ox > 0 ? ox : 0, ry = oy > 0 ? oy : 0;
    //
    // OUTSIDE ONLY. The rectangle distance is zero anywhere inside, and taking
    // the max against zero there would wipe out the negative distances that ARE
    // the silhouette - every solid texel would read as exactly on the surface,
    // so nothing would cast at all.
    const outside = Math.sqrt(rx * rx + ry * ry);
    const dt = cardDistance(card, sa, sb, flip);
    return outside > 0 ? Math.max(dt, outside) : dt;
  };
  const l0 = Math.floor(lod), f = lod - l0;
  const q0 = lodQuantum(l0, lodShift, lodMax), q1 = lodQuantum(l0 + 1, lodShift, lodMax);
  let d = distAt(lodSnap(a, hw, q0), lodSnap(hy, hh, q0));
  if (f > 0 && l0 < CARD_LOD_LEVELS - 1 && q1 !== q0) {
    d += (distAt(lodSnap(a, hw, q1), lodSnap(hy, hh, q1)) - d) * f;
  }
  const lodFade = l0 >= CARD_LOD_LEVELS - 1 ? 1 - f : 1;

  const r = coneSlope * t;
  const raw = r <= 0 ? (d < 0 ? 0 : 1)
                     : (d / r < 0 ? 0 : d / r > 1 ? 1 : d / r);

  // FADE, do not cut. A card's plane faces the light, so t is very nearly the
  // distance from the receiver to the caster - which means the maxDist cap falls
  // as a straight line across the ground perpendicular to the light, and a tree's
  // shadow simply STOPS along it. That reads as a diagonal slice taken out of the
  // shadow rather than as a range limit.
  //
  // So weight how much a card is allowed to darken by how close its hit is to the
  // cap, exactly as fadeWeightTSL does for the field. Smoothstep because its
  // derivative vanishes at both ends, so neither the start of the fade nor the cap
  // itself shows as a crease. The shadow still ends - it now ends by fading, the
  // way one too faint to see would.
  const e0 = fadeStart * maxDist;
  const x = maxDist > e0 ? (t - e0) / (maxDist - e0) : 1;
  const xc = x < 0 ? 0 : x > 1 ? 1 : x;
  const fade = 1 - xc * xc * (3 - 2 * xc);
  return 1 - (1 - raw) * fade * lodFade;
}

// Every card against one light. Multiplied rather than min'd: two sprites
// overlapping a ray each take their own bite out of it, which is what separate
// occluders do, where min would let the nearer one hide the further.
export function cardsVisibility(cards, from, dir, maxDist, coneSlope = 0, lodShift = 0,
                                lodMax = CARD_LOD_LEVELS - 1) {
  let vis = 1;
  for (const c of cards) {
    if (c.enabled === false) continue;
    vis *= cardVisibility({
      card: c.card, centre: c.centre, facing: c.facing,
      from, dir, maxDist, coneSlope, flip: c.flip, lod: c.lod || 0, lodShift, lodMax
    });
    if (vis <= 0) return 0;
  }
  return vis;
}

// --- Getting the cards to the GPU ---
//
// The distance transforms go into ONE atlas texture rather than one texture per
// sprite, because a shader cannot index a texture by a loop variable - and the
// loop over cards is the whole point. An atlas turns "which texture" into "which
// rectangle", which is just arithmetic on a uniform.
//
// Stored R8 with the SAME squared encoding the boxGrid uses, so decodeFieldTSL
// reads it unchanged and there is one distance encoding in the renderer rather
// than two that could drift. Range is CARD_PAD pixels: past that the falloff has
// already reached 1, so nothing is lost by saturating there.

export const CARD_RANGE = CARD_PAD;

// Shelf packing, one row of cards. Sprite DTs are tens of pixels across and
// there are a handful of distinct ones, so a single row stays well inside any
// texture limit and the packing needs no cleverness to justify itself.
export function buildCardAtlas(cards) {
  let width = 0, height = 0;
  for (const c of cards) { width += c.w; height = Math.max(height, c.h); }
  if (width === 0) {
    return { data: new Uint8Array(1), colour: new Uint8Array(4), width: 1, height: 1, rects: [] };
  }

  const data = new Uint8Array(width * height);
  // FAR, not zero. Cards shorter than the atlas leave rows unwritten, and an
  // unwritten byte of 0 decodes to -CARD_RANGE - maximally SOLID. Bilinear
  // filtering reaches those texels from the edge of a short card, so the default
  // has to mean "nothing near", which is 255.
  data.fill(255);
  // The sprites' colour, same layout: each at its rect's origin plus CARD_PAD,
  // the offset its mask sits at in the distance transform - so one rect and
  // one pixel mapping address both. Transparent wherever there is no sprite.
  const colour = new Uint8Array(width * height * 4);
  const rects = [];
  let x = 0;
  for (const c of cards) {
    for (let y = 0; y < c.h; y++) {
      for (let i = 0; i < c.w; i++) {
        data[y * width + x + i] = encodeDistance(c.dt[y * c.w + i], CARD_RANGE);
      }
    }
    if (c.rgba) {
      for (let y = 0; y < c.rows; y++) {
        const row = c.rgba.subarray(y * c.cols * 4, (y + 1) * c.cols * 4);
        colour.set(row, ((y + CARD_PAD) * width + x + CARD_PAD) * 4);
      }
    }
    rects.push({ x, y: 0, w: c.w, h: c.h });
    x += c.w;
  }
  return { data, colour, width, height, rects };
}

// Four vec4 per card instance, flat, ready for a uniform array.
//
//   0  centre xyz,                flip as -1 or +1
//   1  facing ux, uz,             half width and height in metres
//   2  atlas rect x, y, w, h,     in pixels
//   3  metres per pixel,          columns, rows, cascade lod
//
// Returns how many were written. Instances past `capacity` are dropped rather
// than growing the buffer: the cap is on how many cards are TESTED per light,
// which is a frame-cost budget, and it has nothing to do with how many sprites
// exist in the world.
export function packCardInstances(instances, atlas, out, capacity) {
  const n = Math.min(instances.length, capacity);
  for (let i = 0; i < n; i++) {
    const it = instances[i];
    const c = it.card, r = atlas.rects[it.typeIndex];
    const o = i * 16;
    out[o + 0] = it.centre.x; out[o + 1] = it.centre.y;
    out[o + 2] = it.centre.z; out[o + 3] = it.flip ? -1 : 1;
    out[o + 4] = it.facing.ux; out[o + 5] = it.facing.uz;
    out[o + 6] = c.widthMetres / 2; out[o + 7] = c.heightMetres / 2;
    out[o + 8] = r.x; out[o + 9] = r.y; out[o + 10] = r.w; out[o + 11] = r.h;
    out[o + 12] = c.metresPerPixel; out[o + 13] = c.cols; out[o + 14] = c.rows;
    // Cascade LOD: level + how far into the band toward the next. 0 is full
    // resolution, which is also what an instance with no lod gets.
    out[o + 15] = it.lod || 0;
  }
  return n;
}

// Which cards can possibly occlude this light, nearest first.
//
// Two rejections, both cheap and both done on the CPU once per light per frame
// rather than per fragment:
//
//   out of reach   - a card further from the light than the light's own radius
//                    cannot be lit by it, so it cannot cast from it either.
//   too small to see - beyond some distance a sprite's shadow is a smudge; the
//                    sort puts those last so the cap drops them first.
//
// The cap is a frame-cost budget on cards TESTED, not a limit on sprites in the
// world. A scene can hold any number; what is bounded is how many compete to
// shadow one light in one frame.
export function cullCardsForLight(instances, lightPos, radiusMetres, capacity) {
  const keep = [];
  const r2 = radiusMetres * radiusMetres;
  for (const it of instances) {
    if (it.enabled === false) continue;
    const dx = it.centre.x - lightPos.x;
    const dy = it.centre.y - lightPos.y;
    const dz = it.centre.z - lightPos.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 > r2) continue;
    keep.push({ it, d2 });
  }
  keep.sort((a, b) => a.d2 - b.d2);
  return keep.slice(0, capacity).map(k => k.it);
}
