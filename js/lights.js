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
export const TORCH_HEIGHT = 1.5;

// How far in FRONT of the character the flame sits, in metres - toward the
// camera, which is where the sprite faces.
//
// Together with TORCH_HEIGHT this is what keeps the flame out of its holder's
// own silhouette card: held high and out front, it clears the figure without a
// push along the card's normal - which, being tied to the sun rather than the
// camera, landed the flame at different distances on either side of Bob.
export const TORCH_FORWARD = 0.4;

// How far to the character's own side the flame is held, in metres.
//
// A torch is held in a HAND, not out of the middle of the chest, so it sits off
// to one side - and the side it sits on has to follow the character. Bob's
// sprite mirrors when he turns around, so everything about him swaps sides; a
// flame that stayed put while the art flipped would read as the torch passing
// through him.
//
// Kept well inside the figure's half-width (0.625 m at torch height) so the
// flame stays over the body rather than floating out past the shoulder.
export const TORCH_SIDE = 0.3;

export function colourToRGB(hex) {
  return {
    r: ((hex >> 16) & 255) / 255,
    g: ((hex >> 8) & 255) / 255,
    b: (hex & 255) / 255
  };
}

// --- The light list: lights are DATA, not kernel variants ---
//
// Every point light in the scene is a row in one uniform array, and the shading
// kernel loops over however many rows are live. Adding, moving, recolouring or
// removing a light is a write to that array - never a material rebuild - and a
// scene with no point lights runs a loop of zero iterations.
//
// Sixteen is the §7.1 budget. It is a cap on lights SHADED per frame, not on
// lights that exist: a scene may place any number and the nearest-first order
// the caller supplies decides which sixteen are live.
export const MAX_LIGHTS = 16;

// Three vec4 per light:
//   0  position xyz, level
//   1  colour rgb,   source radius (metres)
//   2  card start,   card count, shadow weight, card weight
//      - its slice of the shared card buffer, how much of its shadow is
//        applied (0 = unshadowed, no march), and how much of the SPRITE part
//        of that shadow is (0 = terrain only, no card test) - see the budgets
//        below
export const LIGHT_VEC4S = 3;
export const LIGHT_FLOATS = LIGHT_VEC4S * 4;

// Point lights share one card buffer, each taking a slice. The per-light cap is
// the same 64 the sun has; the total is what keeps the uniform block small
// (256 cards is 16 KB) - a scene with sixteen lights all crowded by sprites
// hands the later lights fewer cards, rather than growing the buffer.
export const POINT_CARD_CAPACITY = 256;

// '#ffd800', 'ffd800', 0xffd800 - all the same colour.
export function parseColour(c) {
  if (typeof c === 'number') return c & 0xffffff;
  if (typeof c === 'string') {
    const n = parseInt(c.replace(/^#/, ''), 16);
    if (Number.isFinite(n)) return n & 0xffffff;
  }
  return TORCH_COLOUR;
}

// The lights that will actually be shaded: lit ones only, at most `capacity`,
// in the order given. Level 0 is off, and off costs nothing - it never reaches
// the array, so the kernel's loop does not even visit it.
export function liveLights(lights, capacity = MAX_LIGHTS) {
  const out = [];
  for (const l of lights) {
    if (!l || clampLevel(l.level) <= 0) continue;
    out.push(l);
    if (out.length >= capacity) break;
  }
  return out;
}

// One light into its three vec4. cardStart/cardCount are the slice of the
// shared card buffer that was turned to face THIS light.
export function packLight(out, index, light, cardStart = 0, cardCount = 0,
                          shadowWeight = 1, cardWeight = 1) {
  const o = index * LIGHT_FLOATS;
  const { r, g, b } = colourToRGB(parseColour(light.colour));
  out[o + 0] = light.position.x;
  out[o + 1] = light.position.y;
  out[o + 2] = light.position.z;
  out[o + 3] = clampLevel(light.level);
  out[o + 4] = r; out[o + 5] = g; out[o + 6] = b;
  out[o + 7] = light.sourceRadius ?? LIGHT_SOURCE_RADIUS;
  out[o + 8] = cardStart;
  out[o + 9] = cardCount;
  out[o + 10] = Math.max(0, Math.min(1, shadowWeight));
  out[o + 11] = Math.max(0, Math.min(1, cardWeight));
  return out;
}

// --- Perf: the contribution cutoff ---
//
// A light's contribution is falloff x N.L, and across the outer part of its
// radius that is a sliver of light that still paid for a full march. The cutoff
// takes it off the top: (amount - cutoff) / (1 - cutoff), floored at zero. A
// remap rather than a threshold, so the light still reaches zero continuously -
// a plain `amount > cutoff` gate would leave a ring of brightness `cutoff`
// where the march stops.
//
// 1/100 is under a third of one 8-bit step at full brightness, and the whole
// band it removes is the part of a light nobody can see.
export const DEFAULT_LIGHT_CUTOFF = 1 / 100;

export function cutAmount(amount, cutoff = DEFAULT_LIGHT_CUTOFF) {
  if (cutoff <= 0) return Math.max(0, amount);
  return Math.max(0, amount - cutoff) / (1 - cutoff);
}

// --- Perf: the shadow budget ---
//
// How many point lights get a shadow march each frame. The rest still light
// surfaces, unshadowed. Cost then scales with the budget rather than the light
// count, which is what a scene full of lamps needs, and once GI lands the LPV
// carries the unshadowed lights' bounce regardless.
//
// Which lights: the ones that matter most where the player is looking - the
// contribution each would make at the focus point, as lightContribution
// computes it. A light that does not reach the focus scores its level alone,
// below every light that does, so a bright far lamp still outranks a dim one.
export const DEFAULT_SHADOW_BUDGET = 12;

// The card budget: of the lights that march, how many also test sprite cards.
// A narrower cut than the shadow budget, and a safe one: a light over it still
// marches the field, so walls still block it - only sprites stop casting its
// shadow. Picked by the same score, so it is the most important few.
export const DEFAULT_CARD_BUDGET = 6;

// Seconds for a light's shadow to fade in or out when it enters or leaves the
// budget, so walking past a lamp does not pop its shadows on and off.
export const SHADOW_FADE_SECONDS = 0.25;

export function shadowScore(light, focus) {
  const dx = light.position.x - focus.x;
  const dy = light.position.y - focus.y;
  const dz = light.position.z - focus.z;
  const c = lightContribution(Math.hypot(dx, dy, dz), light.level);
  // Contribution is 0..1; reaching lights land above 1, the rest below it.
  return c > 0 ? 1 + c : clampLevel(light.level) / (MAX_LIGHT_LEVEL + 1);
}

// The set of lights (by identity) that get shadows this frame.
export function pickShadowed(lights, focus, budget = DEFAULT_SHADOW_BUDGET) {
  const n = Math.max(0, Math.floor(budget));
  if (n >= lights.length) return new Set(lights);
  const ranked = lights.map(l => ({ l, s: shadowScore(l, focus) }))
                       .sort((a, b) => b.s - a.s);
  return new Set(ranked.slice(0, n).map(r => r.l));
}

// Step a shadow weight toward 1 (in the budget) or 0 (out), linearly, taking
// SHADOW_FADE_SECONDS for the full swing.
export function easeShadowWeight(w, shadowed, dt, fade = SHADOW_FADE_SECONDS) {
  const target = shadowed ? 1 : 0;
  if (fade <= 0) return target;
  const step = dt / fade;
  return target > w ? Math.min(target, w + step) : Math.max(target, w - step);
}
