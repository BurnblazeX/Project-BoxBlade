import { file, section, ok, near, truthy, falsy } from './lib/harness.mjs';
import { BLOCK_METRES } from '../js/world.js';
import {
  LIGHT_BITS, LIGHT_LEVELS, MAX_LIGHT_LEVEL, clampLevel,
  lightRadiusBlocks, lightRadiusMetres, lightIntensity, lightFalloff,
  lightContribution, TORCH_COLOUR, colourToRGB,
  LIGHT_SOURCE_RADIUS, coneSlope, pointPenumbraMetres,
  MAX_LIGHTS, LIGHT_FLOATS, POINT_CARD_CAPACITY, liveLights, packLight, parseColour,
  cutAmount, DEFAULT_LIGHT_CUTOFF, shadowScore, pickShadowed, easeShadowWeight,
  SHADOW_FADE_SECONDS
} from '../js/lights.js';

file('lights.test.mjs - the 4-bit level every dynamic light carries');

section('the wire format');
// Four bits is the whole contract. If this ever stops being 16 levels, a stored
// light value and a marched one start meaning different things, and the two
// would disagree only where they overlap - which is the hardest kind of bug to
// see in a lit scene.
ok('four bits', LIGHT_BITS, 4);
ok('sixteen levels', LIGHT_LEVELS, 16);
ok('brightest is 15', MAX_LIGHT_LEVEL, 15);

section('a level is a radius in blocks');
// Stated as an identity rather than a formula, because that is what makes a
// marched light and a future propagated one land on the same geometry: level 15
// reaches 15 blocks, not "about 15 blocks after attenuation".
ok('level 15 reaches 15 blocks', lightRadiusBlocks(15), 15);
ok('level 8 reaches 8 blocks', lightRadiusBlocks(8), 8);
near('15 blocks is 22.5 m', lightRadiusMetres(15), 22.5, 1e-12);
near('and one block is one block', lightRadiusMetres(1), BLOCK_METRES, 1e-12);

section('out-of-range levels are clamped, not wrapped');
// A level arriving as 16 must not read as 0. Wrapping a brightness is the kind
// of thing that shows up once, in one room, months later.
ok('above the top clamps down', clampLevel(99), 15);
ok('below zero clamps up', clampLevel(-4), 0);
ok('fractions round', clampLevel(7.6), 8);
ok('undefined is off', clampLevel(undefined), 0);

section('level 0 is genuinely off');
ok('no radius', lightRadiusMetres(0), 0);
ok('no intensity', lightIntensity(0), 0);
ok('no falloff at any distance', lightFalloff(0, 0), 0);
ok('and none at the source either', lightContribution(0, 0), 0);

section('the falloff reaches zero AT the radius');
// Windowed rather than inverse-square, and this is the assertion that matters:
// physical falloff never reaches zero, so it has to be cut off somewhere, and
// the cut shows as a ring. (1 - d/R)^2 arrives at zero with a vanishing
// derivative, so the light ends exactly where the level says it does and the
// boundary cannot be seen.
for (const level of [1, 8, 15]) {
  const r = lightRadiusMetres(level);
  near(`level ${level} is full brightness at the source`,
       lightFalloff(0, level), 1, 1e-12);
  ok(`level ${level} is exactly zero at its radius`, lightFalloff(r, level), 0);
  ok(`level ${level} stays zero past it`, lightFalloff(r * 2, level), 0);
  // Vanishing derivative at the edge: the last tenth of the radius contributes
  // less than a hundredth of the brightness.
  // Logarithmic, so it stays BRIGHT across the middle instead of spending
  // everything in the first metre - that is the whole change. Half the radius
  // must be well over the square's 0.25.
  truthy(`level ${level} is still bright at half its radius`,
         lightFalloff(r * 0.5, level) > 0.5);
  truthy(`level ${level} still fades before the edge`,
         lightFalloff(r * 0.9, level) < 0.15);
  truthy(`level ${level} falls monotonically`,
    [0, 0.25, 0.5, 0.75, 1].every((f, i, a) =>
      i === 0 || lightFalloff(r * f, level) < lightFalloff(r * a[i - 1], level)));
}

section('brightness scales with the level too');
// One number sets reach AND brightness, so a dim light is also a small one.
// Two knobs would allow "blinding over one block", which reads as a bug.
near('level 15 is full', lightIntensity(15), 1, 1e-12);
near('level 8 is a bit over half', lightIntensity(8), 8 / 15, 1e-12);
truthy('a higher level is brighter at the same distance',
       lightContribution(3, 15) > lightContribution(3, 8));
truthy('and reaches further',
       lightRadiusMetres(15) > lightRadiusMetres(8));
// A point outside the smaller light's radius is lit by the larger one only.
ok('past the small light, nothing', lightContribution(13, 8), 0);
truthy('past the small light, the big one still carries',
       lightContribution(13, 15) > 0);

section('the torch colour survives the round trip');
const c = colourToRGB(TORCH_COLOUR);
ok('#FFD800 red is full', c.r, 1);
near('green is 0xD8/255', c.g, 0xd8 / 255, 1e-12);
ok('blue is none', c.b, 0);
falsy('and it is not greyscale', c.r === c.g && c.g === c.b);

section('softening comes from the emitter having a size');
// The cone from a receiver to the emitter is linear in t, exactly as the sun's
// is - radius = slope * t - which is why the torch reuses the sun's marcher
// rather than getting one of its own. All that differs is the slope.
near('the cone opens to the flame radius AT the light',
     coneSlope(4) * 4, LIGHT_SOURCE_RADIUS, 1e-12);
near('and to half of it halfway there',
     coneSlope(4) * 2, LIGHT_SOURCE_RADIUS / 2, 1e-12);
truthy('a bigger flame is a wider cone', coneSlope(4, 0.5) > coneSlope(4, 0.25));

section('a point light inverts the sun s intuition');
// THIS is the assertion worth having. The sun's penumbra depends only on
// occluder-to-receiver separation, because its apparent size never changes. A
// nearby light's cone closes to a point at the receiver and opens toward the
// flame, so the SAME occluder softens less as the torch backs away. If this ever
// reads the other way round, the sun's fixed-angle cone has been left in.
{
  const near2 = pointPenumbraMetres(1, 2);
  const far = pointPenumbraMetres(1, 6);
  truthy('the same occluder softens LESS as the torch backs away', far < near2);
  near('and by the ratio of the distances', near2 / far, 3, 1e-9);
}
// Contact stays hard at any light distance - an occluder touching the receiver
// has no room for a penumbra, which is what keeps a box sitting on the floor
// looking like it is sitting on the floor.
for (const d of [1, 5, 20]) {
  ok(`contact is hard with the torch at ${d} m`, pointPenumbraMetres(0, d), 0);
}
// And it grows with separation, the one thing it DOES share with the sun.
truthy('a further occluder smears more',
       pointPenumbraMetres(3, 8) > pointPenumbraMetres(1, 8));
// An occluder right at the flame is maximally soft: the cone is the full disc.
near('an occluder at the light smears the whole flame',
     pointPenumbraMetres(4, 4), LIGHT_SOURCE_RADIUS * 2, 1e-12);
falsy('a zero-size flame casts no penumbra at all',
      pointPenumbraMetres(2, 5, 0) > 0);

section('the light list');
// Sixteen is the §7.1 budget, and the card buffer the point lights share has to
// hold at least the sun's per-light cap for one of them.
ok('sixteen lights', MAX_LIGHTS, 16);
ok('three vec4 per light', LIGHT_FLOATS, 12);
truthy('the shared card buffer holds a full light', POINT_CARD_CAPACITY >= 64);

ok('hex string with #', parseColour('#ff8800'), 0xff8800);
ok('hex string without', parseColour('4da6ff'), 0x4da6ff);
ok('number', parseColour(0x123456), 0x123456);
ok('garbage falls back to the torch', parseColour('zz'), TORCH_COLOUR);

const at = (x) => ({ x, y: 1, z: 0 });
const L = (level, x = 0) => ({ position: at(x), level, colour: 0xffffff });
// Off is level 0, and off must never reach the GPU - that is what makes an
// unlit torch free rather than a march multiplied by zero.
ok('level 0 is dropped', liveLights([L(0), L(5), L(0)]).length, 1);
ok('capped at the budget', liveLights(Array.from({ length: 20 }, () => L(3))).length, 16);
{
  const a = L(4, 1), b = L(9, 2);
  const live = liveLights([L(0), a, b]);
  truthy('order is kept - the caller decides priority', live[0] === a && live[1] === b);
}

{
  const out = new Float32Array(LIGHT_FLOATS * 2);
  packLight(out, 1, { position: { x: 3, y: 4, z: 5 }, level: 20,
                      colour: '#ff0000', sourceRadius: 0.5 }, 7, 9);
  const o = LIGHT_FLOATS;
  ok('row 1 lands at its offset, row 0 untouched', out[0], 0);
  ok('position', [out[o], out[o + 1], out[o + 2]].join(), '3,4,5');
  ok('level is clamped on the way in', out[o + 3], 15);
  ok('colour as linear 0..1', [out[o + 4], out[o + 5], out[o + 6]].join(), '1,0,0');
  ok('source radius', out[o + 7], 0.5);
  ok('card slice', [out[o + 8], out[o + 9]].join(), '7,9');
  ok('shadow weight defaults to fully shadowed', out[o + 10], 1);
  packLight(out, 0, { position: { x: 0, y: 0, z: 0 }, level: 3 });
  near('source radius defaults to the flame', out[7], LIGHT_SOURCE_RADIUS, 1e-6);
}

section('the contribution cutoff');
// A remap, not a gate: the light must still reach zero continuously, or the
// edge of the march shows as a ring.
ok('zero cutoff changes nothing', cutAmount(0.3, 0), 0.3);
ok('at the cutoff it is exactly zero', cutAmount(DEFAULT_LIGHT_CUTOFF), 0);
ok('below it, zero', cutAmount(DEFAULT_LIGHT_CUTOFF / 2), 0);
near('full brightness stays full', cutAmount(1), 1, 1e-12);
truthy('continuous just above the cutoff',
       cutAmount(DEFAULT_LIGHT_CUTOFF + 1e-6) < 1e-5);
truthy('monotonic', cutAmount(0.4) < cutAmount(0.5));

section('the shadow budget');
{
  const focus = { x: 0, y: 0, z: 0 };
  const lamp = (x, level) => ({ position: { x, y: 0, z: 0 }, level });
  const near1 = lamp(1, 8), mid = lamp(6, 8), far = lamp(40, 15), dim = lamp(40, 2);
  truthy('a light reaching the focus outranks one that does not',
         shadowScore(mid, focus) > shadowScore(far, focus));
  truthy('nearer outranks further at the same level',
         shadowScore(near1, focus) > shadowScore(mid, focus));
  truthy('out of reach, brighter still outranks dimmer',
         shadowScore(far, focus) > shadowScore(dim, focus));
  const picked = pickShadowed([dim, far, mid, near1], focus, 2);
  truthy('budget 2 picks the two that reach', picked.has(near1) && picked.has(mid) && picked.size === 2);
  ok('budget 0 shadows nothing', pickShadowed([near1, mid], focus, 0).size, 0);
  ok('a budget past the count shadows everything', pickShadowed([near1, mid], focus, 16).size, 2);
}
{
  const half = SHADOW_FADE_SECONDS / 2;
  near('fades in over the fade time', easeShadowWeight(0, true, half), 0.5, 1e-12);
  near('and out', easeShadowWeight(1, false, half), 0.5, 1e-12);
  ok('never overshoots', easeShadowWeight(0.9, true, 10), 1);
  ok('never undershoots', easeShadowWeight(0.1, false, 10), 0);
  ok('zero fade snaps', easeShadowWeight(0, true, 0.001, 0), 1);
}
