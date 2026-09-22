import { BLOCK_METRES } from './world.js';

// --- Dynamic lights: one 4-bit level, and everything else derived from it ---
//
// Every light that is not the sun carries a LEVEL of 0..15 and nothing else. One
// number sets both how far it reaches and how bright it is, because those are
// not independent in practice - a torch that lit a small circle blindingly and a
// torch that lit a large circle faintly are the same torch seen at two radii,
// and giving them separate knobs invites a combination that reads as neither.
//
//   level 0    off, and off means OFF - no march, no cost
//   level 15   brightest, reaching 15 blocks = 22.5 m
//
// Four bits because that is what a block-lit world stores per voxel when light
// is eventually propagated rather than marched, and picking the wire format now
// means the marched version and the stored version cannot disagree about what a
// level MEANS. It is also why the radius is a whole number of blocks: a level is
// a block count, so the two representations land on the same geometry.
//
// The sun is deliberately outside this. It has no position, no radius and no
// falloff, and folding it in would mean inventing a level for something whose
// distinguishing property is that distance does not attenuate it.
export const LIGHT_BITS = 4;
export const LIGHT_LEVELS = 1 << LIGHT_BITS;        // 16
export const MAX_LIGHT_LEVEL = LIGHT_LEVELS - 1;    // 15

export function clampLevel(level) {
  return Math.max(0, Math.min(MAX_LIGHT_LEVEL, Math.round(level || 0)));
}

// The level IS the radius in blocks. Stated as its own function because the
// shader mirrors this line and nothing else about the convention.
export function lightRadiusBlocks(level) {
  return clampLevel(level);
}

export function lightRadiusMetres(level) {
  return lightRadiusBlocks(level) * BLOCK_METRES;
}

// Peak brightness at the source, 0..1. Linear in the level, so the levels are
// evenly spaced to the eye at the low end where they are actually used.
export function lightIntensity(level) {
  return clampLevel(level) / MAX_LIGHT_LEVEL;
}

// WINDOWED AND LOGARITHMIC, not inverse-square and not quadratic.
//
// (1 - d/R)^2 was right about the endpoints and wrong everywhere between them:
// it spends nearly all its brightness in the first metre, so a level-12 torch
// reaching 18 m looked like a level-4 torch at 6 m. Squared falloff is a
// DARKENING curve - half the radius is a quarter the light.
//
// log2(1 + u), u = 1 - d/R, keeps both endpoints exact - 1 at the source, 0 at
// the radius - while sitting far above the square across the whole middle: at
// half the radius it gives 0.58 against 0.25. The torch reads as bright out to
// most of its range and then falls off, which is what a torch does, and the
// level still means exactly what it meant about reach.
export function lightFalloff(distanceMetres, level) {
  const r = lightRadiusMetres(level);
  if (r <= 0) return 0;
  const u = Math.max(0, 1 - distanceMetres / r);
  return Math.log2(1 + u);
}

// What a surface facing the light straight on receives at this distance.
export function lightContribution(distanceMetres, level) {
  return lightIntensity(level) * lightFalloff(distanceMetres, level);
}

// --- Softening: the emitter has a SIZE, and that is the whole of it ---
//
// A light's penumbra is not a filter applied to its shadow; it is the shadow,
// sampled over the area of the thing casting it. So a dynamic light needs one
// more number than its level: how big the emitter is, in metres. A flame is
// about a hand across.
//
// This is a STYLE KNOB, not a physical constant, for the same reason
// SUN_ANGULAR_SIZE is: a physically-sized flame at 12.5 cm texels spreads a
// shadow edge by a fraction of a texel, which is correct and invisible.
export const LIGHT_SOURCE_RADIUS = 0.25;

// THE ONE WAY A POINT LIGHT DIFFERS FROM THE SUN, and it is worth stating
// plainly because it inverts the intuition the sun builds.
//
// The sun's cone has a fixed angular width - its penumbra depends only on how
// far the occluder is from the receiver, because the sun is infinitely far away
// and its apparent size never changes. A nearby light is the opposite: the cone
// from a receiver to the emitter OPENS toward the light and closes to a point at
// the receiver, so the same occluder softens more the closer it sits to the
// flame, and the same flame softens less the further away it is.
//
// Carry a torch up to a wall and its shadow sharpens. Hold the torch close
// behind something and the shadow blurs out. Neither is true of the sun, and
// both come from this one line.
//
// The cone's cross-section at distance t along the ray has radius
// sourceRadius * t / D, so the SLOPE is sourceRadius / D - which is exactly the
// shape the sun's cone trace already takes (radius = R * t). That is why this
// reuses the sun's marcher verbatim instead of needing one of its own: same
// expression, a slope computed per fragment rather than held in a uniform.
export function coneSlope(distanceToLight, sourceRadius = LIGHT_SOURCE_RADIUS) {
  return sourceRadius / Math.max(distanceToLight, 1e-4);
}

// The penumbra an occluder smears, in metres: the width of that cone where the
// occluder sits. Nothing in the shader evaluates this - the cone trace produces
// the width as a consequence of its geometry - and it exists so the property can
// be asserted numerically rather than judged from a screenshot, which is how the
// sun's softening went wrong the first time.
export function pointPenumbraMetres(occluderDistance, distanceToLight,
                                    sourceRadius = LIGHT_SOURCE_RADIUS) {
  if (occluderDistance <= 0) return 0;
  return 2 * sourceRadius * occluderDistance / Math.max(distanceToLight, 1e-4);
}

// The torch the player carries. #FFD800 - a warm yellow that reads as flame
// against the world's greens without going orange enough to look like fire
// damage.
export const TORCH_COLOUR = 0xffd800;
export const TORCH_LEVEL = 12;

// How far above the character's feet the flame sits, in metres. Roughly hand
// height on a 1.5 m block, so contact shadows fall the way a held light's do
// rather than radiating from the ground.
export const TORCH_HEIGHT = 0.9;

export function colourToRGB(hex) {
  return {
    r: ((hex >> 16) & 255) / 255,
    g: ((hex >> 8) & 255) / 255,
    b: (hex & 255) / 255
  };
}
