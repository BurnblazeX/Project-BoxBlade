import { file, section, ok, near, truthy, falsy, inRange, note } from './lib/harness.mjs';
import { World, getVoxelKey, BLOCK_METRES } from '../js/world.js';
import { createBoxGridAt, VOXEL_METRES, sphereTrace, DISTANCE_RANGE } from '../js/boxgrid.js';
import {
  SUN_ANGULAR_SIZE, penumbraMetres, penumbraTexels, coneRadius, sunBasis,
  coneOffsets, coneDirections, coneVisibility, coneTraceVisibility,
  coneTraceSample, GOLDEN_ANGLE,
  fadeWeight, SHADOW_FADE_START, EDGE_FADE_VOXELS,
  sunShade, DEFAULT_AMBIENT, GROUND_REFERENCE_FLOOR,
  BAYER4, bayer4, quantiseShadow, worldTexelIndex, texelLock
} from '../js/sun.js';

file('sun.test.mjs - the marched cone and its distance-based softening');

const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const len = a => Math.hypot(a.x, a.y, a.z);

// A 20 degree sun, the angle bxb.sun() defaults to because it is where shadows
// are long enough to read. Azimuth 0 puts it straight down +Z, which keeps the
// geometry in the tests below one-dimensional and the failures legible.
const EL = 20 * Math.PI / 180;
const SUN = { x: 0, y: Math.sin(EL), z: Math.cos(EL) };

// The geometry tests below use a 45 degree sun instead, and the reason is the
// range limit rather than convenience. At 20 degrees a 3-block wall casts a
// shadow 12.4 m long - past the 12 m march cap and most of the way across the
// 18 m footprint - so the far end of the penumbra is truncated by the field
// rather than resolved, and the measurement would be of the grid's edge. At 45
// degrees the shadow is as long as the wall is tall, so a 1 to 3 block wall
// stays comfortably inside both limits and the number being measured is the
// penumbra. That this restriction exists at all is the CSM's remaining argument;
// see the last section.
const GEL = 45 * Math.PI / 180;
const GSUN = { x: 0, y: Math.sin(GEL), z: Math.cos(GEL) };

section('the cone basis');
// The sun direction and its two tangents are written together as a set, because
// a basis left over from a previous sun tilts the penumbra off to one side - a
// failure that looks like a softening bug and is actually a stale uniform.
{
  const b = sunBasis(SUN);
  near('forward is normalised', len(b.forward), 1, 1e-12);
  near('tangent is normalised', len(b.tangent), 1, 1e-12);
  near('bitangent is normalised', len(b.bitangent), 1, 1e-12);
  near('tangent is perpendicular to the sun', dot(b.tangent, b.forward), 0, 1e-12);
  near('bitangent is perpendicular to the sun', dot(b.bitangent, b.forward), 0, 1e-12);
  near('the two tangents are perpendicular', dot(b.tangent, b.bitangent), 0, 1e-12);
}
// The degenerate case the shader would need a branch for, which is why the basis
// is built on the CPU: a sun straight up has no obvious tangent, and choosing
// world up as the reference axis would collapse the cross product to zero.
for (const [label, dir] of [
  ['straight up', { x: 0, y: 1, z: 0 }],
  ['straight down', { x: 0, y: -1, z: 0 }],
  ['along +X', { x: 1, y: 0, z: 0 }],
  ['along +Z', { x: 0, y: 0, z: 1 }]
]) {
  const b = sunBasis(dir);
  near(`the basis survives a sun pointing ${label}`, len(b.tangent), 1, 1e-12);
  near(`  and stays perpendicular to it`, dot(b.tangent, b.forward), 0, 1e-12);
}
near('an unnormalised direction is normalised', len(sunBasis({ x: 3, y: 4, z: 0 }).forward), 1, 1e-12);

section('the sample pattern');
ok('one ray is the cone axis itself', JSON.stringify(coneOffsets(1)), JSON.stringify([[0, 0]]));
// This is what makes rays=1 reproduce the old hard shadow exactly, so the soft
// result always has a reference to be judged against rather than an opinion.
{
  const d = coneDirections(SUN, 1, SUN_ANGULAR_SIZE);
  near('so a single ray points exactly at the sun', dot(d[0], SUN), 1, 1e-12);
}
{
  const n = 16;
  const offs = coneOffsets(n);
  ok(`${n} rays give ${n} offsets`, offs.length, n);
  truthy('every offset is inside the unit disc',
    offs.every(([u, v]) => Math.hypot(u, v) <= 1 + 1e-12));
  // Uniform in AREA, not in radius: half the samples must fall inside r =
  // 1/sqrt(2). The naive r = i/n crowds the rim and biases the penumbra outward,
  // which is visible as a shadow edge that is too dark too far out.
  const inner = offs.filter(([u, v]) => Math.hypot(u, v) <= Math.SQRT1_2).length;
  ok('half the samples fall in the inner half of the disc by area', inner, n / 2);
  // Evenly spread in angle, so n samples buy n samples of gradient rather than
  // several clustered ones.
  near('the golden angle is what spreads them', GOLDEN_ANGLE, 2.399963, 1e-5);
}
{
  const dirs = coneDirections(SUN, 32, SUN_ANGULAR_SIZE);
  truthy('every cone direction is unit length',
    dirs.every(d => Math.abs(len(d) - 1) < 1e-12));
  // Inside the cone means within the HALF-angle of the axis, because the
  // constant is the sun's angular diameter.
  const half = SUN_ANGULAR_SIZE / 2;
  truthy('every direction is inside the sun\'s angular radius',
    dirs.every(d => Math.acos(Math.min(1, dot(d, SUN))) <= half + 1e-9));
  // And the set is CENTRED on the sun: a mean that drifts off-axis would move
  // the whole shadow rather than soften it.
  const m = dirs.reduce((a, d) => ({ x: a.x + d.x, y: a.y + d.y, z: a.z + d.z }), { x: 0, y: 0, z: 0 });
  const ml = len(m);
  near('and the set is centred on the sun direction', dot(m, SUN) / ml, 1, 1e-6);
  // Rotating the pattern must not move it off-axis either - that rotation is
  // applied per texel to break up banding, so a drift would show as the shadow
  // itself dithering rather than its edge.
  const rot = coneDirections(SUN, 32, SUN_ANGULAR_SIZE, 1.234);
  const rm = rot.reduce((a, d) => ({ x: a.x + d.x, y: a.y + d.y, z: a.z + d.z }), { x: 0, y: 0, z: 0 });
  near('a rotated pattern is still centred on the sun', dot(rm, SUN) / len(rm), 1, 1e-6);
  truthy('but the rotation does move the individual samples',
    rot.some((d, i) => Math.abs(d.x - dirs[i].x) > 1e-6));
}
near('the cone radius is the tangent of the half-angle',
     coneRadius(SUN_ANGULAR_SIZE), Math.tan(SUN_ANGULAR_SIZE / 2), 1e-12);

section('the penumbra formula');
// The whole of "distance-based softening" as arithmetic. Nothing in the shader
// evaluates this - the cone sampling produces it as a consequence of its
// geometry - so this is the prediction the marched result is checked against.
near('contact casts no penumbra', penumbraMetres(0), 0, 1e-12);
near('a 4 m separation spreads by angular size x separation',
     penumbraMetres(4, 0.12), 0.48, 1e-12);
near('and that is 3.8 texels, which is what makes 0.12 the chosen style knob',
     penumbraTexels(4, 0.12), 0.48 / VOXEL_METRES, 1e-12);
truthy('the penumbra grows monotonically with separation',
  [0, 1, 2, 4, 8].every((s, i, a) => i === 0 || penumbraMetres(a[i]) > penumbraMetres(a[i - 1])));
near('a negative separation is clamped rather than inverted', penumbraMetres(-3), 0, 1e-12);

section('the marched cone against real geometry');
// The claim under test: softness comes from the geometry, not from a filter. So
// build one wall, march the ground behind it, and measure the shadow edge. If
// the edge is the same width close to the wall as far from it, the cone is not
// doing what the whole change was for.
//
// A wall at a fixed z with the sun down +Z at 20 degrees casts along -Z... but
// the sun VECTOR points toward the sun, so the shadow falls on the -Z side and
// is measured walking away from the wall.
const WALL_Z = 10;   // near the far edge, so the whole shadow lands on the grid

function wallGrid(height) {
  World.clear();
  // Ground under the whole footprint, plus a wall of the given height across it.
  for (let x = 0; x < 12; x++) {
    for (let z = 0; z < 12; z++) {
      World.set(getVoxelKey(x, 0, z), { solid: true, walkable: true, occupant: null });
    }
    for (let y = 1; y <= height; y++) {
      World.set(getVoxelKey(x, y, WALL_Z), { solid: true, walkable: false, occupant: null });
    }
  }
  return createBoxGridAt(0, 0);
}

// Ground surface height in world metres: block y = 0 spans [-0.75, +0.75], so its
// top face is at +0.75.
const GROUND_Y = 0.5 * BLOCK_METRES;
const UP = { x: 0, y: 1, z: 0 };

// Visibility along a line of ground texels walking away from the wall, in the
// direction the shadow falls.
function groundProfile(grid, rays, { angular = SUN_ANGULAR_SIZE, sun = GSUN,
                                     from = WALL_Z - 1, to = 0, cone = false,
                                     fade = null } = {}) {
  const out = [];
  const step = VOXEL_METRES;
  // Starts INSIDE the shadow and walks out into the light. Starting on the lit
  // side instead makes edgeWidth() find its upper threshold at the first sample
  // and report a width of zero - which reads as "the technique produces a hard
  // edge" rather than as "the profile was walked the wrong way".
  const z0 = (from + 0.5) * BLOCK_METRES;
  const z1 = (to - 0.5) * BLOCK_METRES;
  for (let z = z0; z >= z1; z -= step) {
    const p = { x: 6 * BLOCK_METRES, y: GROUND_Y, z };
    out.push({
      z,
      v: cone
        ? coneTraceVisibility(grid, p, UP, sun, {
            angular, maxDistance: 12,
            ...(fade ? { fadeStart: fade[0], edgeFade: fade[1] } : {})
          })
        : coneVisibility(grid, p, UP, sun, {
            rays, angular, maxDistance: 12,
            ...(fade ? { fadeStart: fade[0], edgeFade: fade[1] } : {})
          })
    });
  }
  return out;
}

// Width of the transition from shadow to light, in metres: the span over which
// visibility climbs through the middle of its range. Measured between 0.15 and
// 0.85 rather than 0 and 1, because the tails are where a single extra ray
// clipping the wall corner lands and they are not what the eye reads as the edge.
function edgeWidth(profile) {
  const lo = profile.findIndex(s => s.v > 0.15);
  let hi = -1;
  for (let i = 0; i < profile.length; i++) if (profile[i].v >= 0.85) { hi = i; break; }
  if (lo < 0 || hi < 0 || hi < lo) return null;
  return Math.abs(profile[hi].z - profile[lo].z);
}

{
  const grid = wallGrid(2);
  const hard = groundProfile(grid, 1);
  const soft = groundProfile(grid, 32);

  // First, the thing that must not change: both agree about where the umbra is.
  // The soft path is the hard path sampled over an area, so a disagreement here
  // is the marcher, not the softening.
  const fullyDark = p => p.filter(s => s.v < 0.01).length;
  truthy('the wall casts a shadow at all', fullyDark(hard) > 0);
  truthy('and the soft cone finds the same umbra', fullyDark(soft) > 0);
  note(`umbra: ${fullyDark(hard)} texels hard, ${fullyDark(soft)} texels fully dark soft`);

  // One ray is binary by construction: there is nothing to average.
  const values = new Set(hard.map(s => s.v));
  truthy('one ray gives a binary edge - only lit or shadowed',
    [...values].every(v => v === 0 || v === 1));
  truthy('and the cone gives intermediate values instead',
    soft.some(s => s.v > 0.02 && s.v < 0.98));

  const hw = edgeWidth(hard), sw = edgeWidth(soft);
  note(`edge width: hard ${hw === null ? 'n/a' : hw.toFixed(3)} m, soft ${sw.toFixed(3)} m`);
  truthy('the cone edge is wider than the single ray\'s', sw > (hw || 0));
}

// The actual claim of "distance-based": a taller wall means the shadow edge sits
// further from its occluder, so it must be softer. This is the property the
// depth-map path kept getting wrong, and it is one number either way.
{
  const widths = [];
  for (const h of [1, 2, 3]) {
    const grid = wallGrid(h);
    const w = edgeWidth(groundProfile(grid, 48));
    widths.push({ h, w });
    note(`wall ${h} blocks (${(h * BLOCK_METRES).toFixed(2)} m): edge ${w === null ? 'n/a' : w.toFixed(3)} m`);
  }
  truthy('every wall height produced a measurable edge', widths.every(x => x.w !== null));
  truthy('a taller wall casts a softer edge, which is the whole claim',
    widths[1].w > widths[0].w && widths[2].w > widths[1].w);

  // And it is the RIGHT width, not merely a growing one. The occluder is the
  // wall top; the shadow it casts lands roughly h/tan(elevation) away along the
  // ground, so the occluder-receiver separation is that hypotenuse. Checked
  // loosely: the cone is sampled, the ground is quantised to 12.5 cm texels, and
  // the edge threshold is a judgement - but an order of magnitude out would mean
  // the angular size is not reaching the shader as an angle.
  for (const { h, w } of widths) {
    const top = h * BLOCK_METRES;
    const separation = top / Math.sin(GEL);
    const predicted = penumbraMetres(separation);
    // Plus or minus a texel on top of the 2x, because the ground is sampled at
    // 12.5 cm and the edge cannot be resolved finer than that. A 1-block wall's
    // predicted penumbra is two texels wide, so it sits ON the resolution floor
    // of the lighting - which is not a defect of the measurement but the reason
    // 0.12 rad was chosen: anything smaller and even a tall caster's penumbra
    // would round away to a hard edge.
    inRange(`a ${h}-block wall's edge is within 2x (+/- a texel) of the ` +
            `predicted ${predicted.toFixed(2)} m`, w,
            Math.max(0, predicted * 0.5 - VOXEL_METRES),
            predicted * 2 + VOXEL_METRES);
  }
}

// The sun's angular size is a style knob, so it has to actually be the knob.
{
  const grid = wallGrid(3);
  const narrow = edgeWidth(groundProfile(grid, 48, { angular: 0.03 }));
  const wide = edgeWidth(groundProfile(grid, 48, { angular: 0.24 }));
  truthy('both angular sizes produced a measurable edge', narrow !== null && wide !== null);
  note(`angular 0.03 -> ${narrow === null ? 'n/a' : narrow.toFixed(3)} m, ` +
       `0.24 -> ${wide === null ? 'n/a' : wide.toFixed(3)} m`);
  truthy('a larger angular size gives a wider penumbra', wide > narrow * 2);
  // And the reason 0.12 was chosen over anything physical: the real sun is
  // 0.0093 rad, which at this scale is a third of a texel and invisible.
  truthy('the physical sun would be under one texel wide at 4 m',
    penumbraTexels(4, 0.0093) < 1);
}

// Open sky must stay fully lit. This is the failure mode the textbook SDF cone
// trace has against this field - min(d/(k*t)) reads 0.69 in open air at 3 m,
// because the field is clamped to one voxel - and the reason the cone is sampled
// rather than estimated from the stored distance.
{
  const grid = wallGrid(3);
  // Well clear of the wall's 4.5 m shadow at a 45 degree sun, on the lit side.
  for (const z of [0, 1, 2]) {
    const p = { x: 6 * BLOCK_METRES, y: GROUND_Y, z: (z - 0.5) * BLOCK_METRES };
    const v = coneVisibility(grid, p, UP, GSUN, { rays: 48 });
    ok(`open ground ${((WALL_Z - z) * BLOCK_METRES).toFixed(1)} m from the wall ` +
       `is fully lit, not partly`, v, 1);
  }
  // Tucked against the wall, deep in its umbra, nothing gets through.
  const p = { x: 6 * BLOCK_METRES, y: GROUND_Y, z: (WALL_Z - 1.2) * BLOCK_METRES };
  ok('and ground tucked against the wall is fully dark',
     coneVisibility(grid, p, UP, GSUN, { rays: 48 }), 0);
}

// The bias is along the FACE NORMAL, which is what makes it independent of sun
// angle. Along the ray it would scale with N.L, so a low sun grazing a surface
// would barely clear the face and the surface would shadow itself in stripes.
{
  const grid = wallGrid(1);
  for (const el of [5, 20, 60, 85]) {
    const e = el * Math.PI / 180;
    const sun = { x: 0, y: Math.sin(e), z: Math.cos(e) };
    // Open ground far from the wall, on the lit side: no geometry can occlude it,
    // so anything below 1 here is the surface shadowing itself.
    const p = { x: 6 * BLOCK_METRES, y: GROUND_Y, z: 0 };
    ok(`flat ground does not self-shadow at a ${el} degree sun`,
       coneVisibility(grid, p, UP, sun, { rays: 16 }), 1);
  }
}

section('the cone trace against the sampled reference');
// The point of building the sampled path first: it is the reference the cheap one
// is judged against. Sampling the disc with 48 rays is what a soft shadow IS -
// the fraction of the light the point can see - so wherever the single-ray cone
// trace disagrees, the cone trace is the one that is wrong, and by how much is
// the price of the sixteenfold saving.
{
  const grid = wallGrid(2);
  const sampled = groundProfile(grid, 48);
  const coned = groundProfile(grid, 1, { cone: true });
  ok('both profiles cover the same ground', coned.length, sampled.length);

  // First the two things that must match exactly, because they are not about
  // softening at all: full shadow and full light.
  const bothDark = sampled.filter((s, i) => s.v === 0 && coned[i].v === 0).length;
  const sampledDark = sampled.filter(s => s.v === 0).length;
  const sampledLit = sampled.filter(s => s.v === 1).length;
  const litAgree = sampled.filter((s, i) => s.v === 1 && coned[i].v === 1).length;
  ok('the two agree on every fully shadowed texel', bothDark, sampledDark);
  // Within a texel on the lit side, and the cone is the conservative one: it sees
  // the nearest surface rather than disc coverage, so at the outer lip of the
  // penumbra it still reports a trace of shadow where the sampled version has
  // already resolved to fully lit. Erring dark at the edge of a shadow is the
  // right direction to err.
  inRange('and on every fully lit one but at most a texel of conservatism',
          sampledLit - litAgree, 0, 1);
  truthy('with a real umbra and a real lit region to agree about',
    sampledDark > 4 && sampledLit > 4);

  // Then the penumbra itself, where they are allowed to differ. The cone trace
  // sees the NEAREST surface, not how much of the disc that surface covers, so a
  // difference here is expected - it is the magnitude that matters.
  let worst = 0, sum = 0;
  for (let i = 0; i < sampled.length; i++) {
    const d = Math.abs(sampled[i].v - coned[i].v);
    sum += d;
    if (d > worst) worst = d;
  }
  const mean = sum / sampled.length;
  const off = sampled.filter((s, i) => Math.abs(s.v - coned[i].v) > 0.25).length;
  note(`cone vs 48-ray reference: mean |diff| ${mean.toFixed(3)}, worst ` +
       `${worst.toFixed(3)}, ${off}/${sampled.length} texels off by >0.25`);
  inRange('the mean disagreement across the whole profile is small', mean, 0, 0.08);
  // The worst single texel is allowed to be large, because it lands where the two
  // models genuinely differ - the umbra lip, where disc coverage falls off a cliff
  // and nearest-surface does not. What must stay small is HOW MANY texels are
  // there: a handful is the price of one ray, a band of them would mean the cone
  // is softening the wrong region.
  inRange('and only a handful of texels differ materially', off, 0, 6);

  // WHICH WAY it errs, which is the useful half of the comparison. The cone reads
  // the distance to the nearest SURFACE, and near a wall's top edge that is
  // SMALLER than the lateral clearance past the silhouette - the corner is closer
  // to the ray than the edge it has to clear. A smaller d means a smaller
  // d/(R*t), so the cone comes out DARKER than the truth at the umbra lip, pushing
  // the umbra outward and narrowing the penumbra rather than widening it.
  //
  // Asserted in this direction on purpose: erring dark is the safe direction for a
  // shadow, and a cone that erred BRIGHT would mean light leaking out of an umbra,
  // which is a sign error rather than a cheaper model.
  const leak = sampled.filter((s, i) => coned[i].v > s.v + 0.25).length;
  const overdark = sampled.filter((s, i) => coned[i].v < s.v - 0.25).length;
  note(`the cone leaks light on ${leak} texels and over-darkens on ${overdark}`);
  ok('the cone never leaks light out of an umbra, which would be a sign error', leak, 0);
  truthy('it errs dark at the umbra lip instead', overdark > 0);

  // Both must be monotonic along the ground: visibility rises from the umbra out
  // to open sky and never dips back. A non-monotonic profile is the signature of
  // the field disagreeing with itself, which is what a stale clear band or a
  // mismatched decode produces.
  const monotone = p => p.every((s, i) => i === 0 || s.v >= p[i - 1].v - 1e-9);
  truthy('the sampled profile rises monotonically out of the shadow', monotone(sampled));
  truthy('and so does the cone trace', monotone(coned));
}

// The cone trace has to show the same distance-based growth, or it is not a
// substitute for the sampled one - it is just a cheaper blur.
{
  const widths = [];
  for (const h of [1, 2, 3]) {
    const grid = wallGrid(h);
    const w = edgeWidth(groundProfile(grid, 1, { cone: true }));
    widths.push({ h, w });
    note(`cone trace, wall ${h} blocks: edge ${w === null ? 'n/a' : w.toFixed(3)} m`);
  }
  truthy('every wall height produced a measurable edge', widths.every(x => x.w !== null));
  truthy('a taller wall casts a softer edge here too',
    widths[1].w > widths[0].w && widths[2].w > widths[1].w);

  // How the cone's penumbra compares with the sampled one at the same geometry.
  // It comes out NARROWER, for the reason above, and the ratio is worth pinning
  // down because it is what you would tune the angular size against if you
  // shipped the cone: a visible mismatch between the two techniques at the same
  // bxb.soften() setting is this, not a bug.
  const ratios = [];
  for (const { h, w } of widths) {
    const ref = edgeWidth(groundProfile(wallGrid(h), 48));
    if (ref) ratios.push(w / ref);
  }
  const meanRatio = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  note(`cone penumbra is ${(meanRatio * 100).toFixed(0)}% of the sampled width ` +
       `(${ratios.map(r => r.toFixed(2)).join(', ')})`);
  inRange('the cone under-softens relative to the reference, by under half',
          meanRatio, 0.5, 1.05);

  for (const { h, w } of widths) {
    const separation = (h * BLOCK_METRES) / Math.sin(GEL);
    const predicted = penumbraMetres(separation);
    inRange(`a ${h}-block wall's cone edge is within 2x (+/- a texel) of the ` +
            `predicted ${predicted.toFixed(2)} m`, w,
            Math.max(0, predicted * 0.5 - VOXEL_METRES),
            predicted * 2 + VOXEL_METRES);
  }
}

// THE FAILURE THE WIDENED FIELD EXISTS TO PREVENT. At DISTANCE_RANGE = 1 the
// stored distance saturated at 12.5 cm, so d/(R*t) in open air at 3 m came out at
// 0.125/(0.06*3) = 0.69 and the entire world dimmed by 31% with nothing casting
// anything. The range is 8 so the expression only saturates past 1/0.06 = 16.7 m,
// which is beyond the 12 m march cap - so there is nowhere a ray can reach where
// open sky reads as anything but lit.
{
  const grid = wallGrid(1);
  // Open ground on the lit side, at increasing distance from the only geometry
  // above the ground plane.
  for (const z of [0, 2, 4, 6]) {
    const p = { x: 6 * BLOCK_METRES, y: GROUND_Y, z: (z - 0.5) * BLOCK_METRES };
    ok(`open ground reads exactly lit at z=${z}, not merely nearly`,
       coneTraceVisibility(grid, p, UP, GSUN, { maxDistance: 12 }), 1);
  }
  // The headroom, stated: how far a receiver can be from everything before the
  // cone starts reporting shadow from empty space.
  const saturates = (DISTANCE_RANGE * VOXEL_METRES) / coneRadius(SUN_ANGULAR_SIZE);
  note(`the cone saturates at ${saturates.toFixed(1)} m of clearance, against a ` +
       `12 m march cap - so the margin is ${(saturates - 12).toFixed(1)} m`);
  truthy('the saturation distance is beyond the march cap, which is why the ' +
         'range is 8 and not 4', saturates > 12);
  // And the counterfactual, so the reason is recorded rather than asserted: at
  // the old range this margin was deeply negative.
  const old = (1 * VOXEL_METRES) / coneRadius(SUN_ANGULAR_SIZE);
  truthy(`at the old +-1 voxel range it saturated at ${old.toFixed(1)} m, far ` +
         `inside the cap - which is why this technique could not be used`, old < 12);
}

section('fading out where the march runs out of reach');
// The largest jump between neighbouring texels. A shadow that ends on a line has
// one big one; a shadow that dissolves has none.
const biggestJump = p => p.reduce((m, s, i) =>
  i === 0 ? m : Math.max(m, Math.abs(s.v - p[i - 1].v)), 0);
// A march ends for three reasons and only one is geometry: it hit something, it
// ran past maxDist, or it left the grid. The last two are answers about the
// FIELD, and both used to resolve to "fully lit" - so the texel whose ray reached
// a wall at 11.9 m came out black and its neighbour whose ray would have reached
// it at 12.1 m came out lit. That is a step discontinuity in the middle of open
// ground, and it is the hard straight edge seen at the far end of a long shadow.
{
  near('deep inside the march nothing is faded',
       fadeWeight({ x: 72, y: 72, z: 72 }, 1, 12), 1, 1e-9);
  near('at the distance cap it is fully faded',
       fadeWeight({ x: 72, y: 72, z: 72 }, 12, 12), 0, 1e-9);
  near('and hard against the grid boundary too',
       fadeWeight({ x: 0, y: 72, z: 72 }, 1, 12), 0, 1e-9);
  truthy('the distance fade is monotonic across the whole march',
    [0, 3, 6, 7.5, 9, 10.5, 12].every((t, i, a) => i === 0 ||
      fadeWeight({ x: 72, y: 72, z: 72 }, a[i], 12) <=
      fadeWeight({ x: 72, y: 72, z: 72 }, a[i - 1], 12) + 1e-12));
  // Smoothstep rather than linear, so neither end of the fade shows as a crease:
  // the derivative vanishes at both, which a linear ramp's does not.
  const midT = (SHADOW_FADE_START + 1) / 2 * 12;
  const mid = fadeWeight({ x: 72, y: 72, z: 72 }, midT, 12);
  inRange('and eases rather than ramping, so its own ends are invisible', mid, 0.4, 0.6);
  // Turning it off has to be exact, because that is how the hard cut gets
  // confirmed as the cause rather than assumed.
  near('bxb.fade(1, 0) disables both fades exactly',
       fadeWeight({ x: 0, y: 0, z: 0 }, 11.99, 12, 1, 0), 1, 1e-9);
}

// The behavioural test, and the one that matters: build a shadow long enough to
// be truncated by the 12 m cap, then look for a cliff in the profile.
{
  // A 3-block wall at a 20 degree sun throws a shadow 12.4 m long, so its far end
  // needs more than 12 m of ray travel and is cut off. This is the exact geometry
  // that produced the reported artifact.
  // Explicitly WIDE, not the shipped default. See the note below for why those
  // differ - this block is testing that the distance fade can remove a cap
  // cliff, which is a different question from whether the default does.
  const grid = wallGrid(3);
  const opts = { sun: SUN, cone: true };
  const faded = groundProfile(grid, 1, { ...opts, fade: [0.6, EDGE_FADE_VOXELS] });
  const hard = groundProfile(grid, 1, { ...opts, fade: [1, 0] });

  const hj = biggestJump(hard), fj = biggestJump(faded);
  note(`biggest neighbouring-texel jump: ${hj.toFixed(3)} unfaded, ${fj.toFixed(3)} faded`);
  truthy('without the fade the truncated shadow really does end on a cliff', hj > 0.3);
  truthy('and the fade removes it', fj < hj * 0.6);
  inRange('leaving no step a viewer would read as an edge', fj, 0, 0.35);

  // The fade must not eat the shadow it is smoothing. The umbra close to the
  // wall is well inside every limit and has to be untouched.
  const nearWall = faded.slice(0, 8);
  truthy('the umbra next to its caster is still fully dark',
    nearWall.every(s => s.v < 0.02));
  const bothDark = faded.filter((s, i) => s.v < 0.02 && hard[i].v < 0.02).length;
  const hardDark = hard.filter(s => s.v < 0.02).length;
  // The umbra does get shorter, and by a predictable amount: the fade spans the
  // last (1 - start) of the march, which at this sun angle lands on the ground as
  // (1 - start) * maxDist * cos(elevation) metres. Asserting against that rather
  // than a guessed fraction, because if the loss ever stops matching the fade
  // width, something other than the fade is eating the shadow.
  const lost = hardDark - bothDark;
  const predicted = (1 - 0.6) * 12 * Math.cos(EL) / VOXEL_METRES;
  note(`umbra shortened by ${lost} texels; the fade spans ${predicted.toFixed(0)}`);
  inRange('and the umbra it costs matches the width of the fade itself',
          lost, predicted * 0.5, predicted * 1.5);
}

// WHAT THE SHIPPED DEFAULT ACTUALLY COVERS, which is not what I first assumed.
//
// SHADOW_FADE_START is 0.97, so the distance fade spans the last 36 cm of a 12 m
// ray and does essentially nothing to a cap cliff - asserted here rather than
// glossed, because the wide-fade block above could otherwise be read as proof
// that the default handles it.
//
// That is deliberate. The hard edge actually seen in the game was the GRID
// BOUNDARY, not the distance cap: the footprint is 18 m and follows the player,
// so a shadow running away from them leaves the grid long before any ray runs out
// of distance. The edge fade below is what fixes that, and it is fully on at the
// default. A wide distance fade would cost metres of real umbra at the end of
// every long shadow to smooth a cliff that is rarely reached - a bad trade, so
// the distance fade is left narrow and available via bxb.fade() for scenes that
// do reach the cap.
{
  const grid = wallGrid(3);
  const atDefault = groundProfile(grid, 1, { sun: SUN, cone: true });
  const jump = atDefault.reduce((m, s, i) =>
    i === 0 ? m : Math.max(m, Math.abs(s.v - atDefault[i - 1].v)), 0);
  note(`at the shipped fade start of ${SHADOW_FADE_START}, the cap cliff is ${jump.toFixed(3)}`);
  truthy('the default does NOT smooth a distance-cap cliff, and is not meant to',
    jump > 0.25);
  const wide = biggestJump(groundProfile(grid, 1, {
    sun: SUN, cone: true, fade: [0.6, EDGE_FADE_VOXELS] }));
  truthy('bxb.fade(0.6) is what to reach for if a scene does reach the cap',
    jump > wide * 2);
}

// The grid boundary, which truncates independently of the cap and IS what the
// default covers: a ray can leave the footprint at any distance, including
// immediately, so fading by travelled distance alone would not catch it.
{
  const grid = wallGrid(2);
  // Walk OUT of the footprint along the lit side. Ground near the far boundary
  // has rays that exit almost at once.
  const prof = [];
  for (let b = 11.4; b >= 8; b -= 0.1) {
    const p = { x: b * BLOCK_METRES, y: GROUND_Y, z: 4 * BLOCK_METRES };
    prof.push(coneTraceVisibility(grid, p, UP, SUN, { maxDistance: 12 }));
  }
  truthy('nothing near the grid edge reports a sudden shadow',
    prof.every((v, i) => i === 0 || Math.abs(v - prof[i - 1]) < 0.35));
}

section('the shading formula');
// The bug this section exists for: the formula was written twice, the atlas copy
// applied Lambert and the per-pixel copy did not, and the same ground came out
// 0.572 on one path and 1.0 on the other. One function now, and these are the
// properties that make the difference visible if it ever forks again.
{
  const up = { x: 0, y: 1, z: 0 };
  const sunUp = s => Math.sin(s * Math.PI / 180);

  // Shadowed is ambient, lit is brighter, in every mode. Anything else is not a
  // shading formula.
  for (const mode of ['ground', 'lambert', 'flat']) {
    near(`fully shadowed ground falls to ambient in '${mode}'`,
         sunShade(0, sunUp(20), sunUp(20), DEFAULT_AMBIENT, mode), DEFAULT_AMBIENT, 1e-12);
    truthy(`and lit ground is brighter than that in '${mode}'`,
      sunShade(1, sunUp(20), sunUp(20), DEFAULT_AMBIENT, mode) > DEFAULT_AMBIENT);
  }

  // THE REPORTED SYMPTOM, as a number. Raw Lambert at a low sun multiplies every
  // up-facing surface - nearly all of them, in a top-down game - by sin(elevation).
  near('raw lambert leaves lit ground at 0.57 under a 20 degree sun, which is ' +
       'what read as permanent partial shade',
       sunShade(1, sunUp(20), sunUp(20), 0.35, 'lambert'), 0.572, 0.01);
  near("and 'flat' is the 1.0 that bxb.shadows always produced",
       sunShade(1, sunUp(20), sunUp(20), 0.35, 'flat'), 1, 1e-12);

  // The fix, and the property worth having: in 'ground' mode the sun's ANGLE
  // controls shadow direction and length, not the scene's brightness.
  for (const el of [20, 35, 50, 90]) {
    near(`lit flat ground reads fully lit at a ${el} degree sun`,
         sunShade(1, sunUp(el), sunUp(el), 0.35, 'ground'), 1, 1e-9);
  }
  // Below the reference floor it stops tracking, on purpose: a sun on the horizon
  // would otherwise divide every surface up to full brightness and collapse the
  // differentiation between faces entirely.
  truthy('but a sun near the horizon does not brighten everything to white',
    sunShade(1, sunUp(5), sunUp(5), 0.35, 'ground') < 1);

  // Faces must still differentiate, or 'ground' is just 'flat' with extra steps -
  // and losing that is what made the first atlas pass look flat.
  {
    const el = 35, s = sunUp(el);
    const facingAway = sunShade(1, -0.9, s, 0.35, 'ground');
    const facingSun = sunShade(1, 0.9, s, 0.35, 'ground');
    near('a face turned away from the sun still falls to ambient', facingAway, 0.35, 1e-12);
    truthy('while one facing it is fully lit', facingSun > 0.9);
    truthy('so the modes are not secretly the same', facingSun - facingAway > 0.5);
  }

  // Monotonic in visibility, so a penumbra is a gradient rather than a fold.
  truthy('brightness rises monotonically with visibility',
    [0, 0.25, 0.5, 0.75, 1].every((v, i, a) => i === 0 ||
      sunShade(a[i], 0.5, sunUp(30)) > sunShade(a[i - 1], 0.5, sunUp(30))));
  ok('ambient of 0 means a shadow is fully black',
     sunShade(0, 1, 1, 0, 'ground'), 0);
}

section('not overwriting a shadow with a non-answer');
// THE FLICKER, and why it was not a softening problem at all.
//
// The footprint follows the player, so walking eventually drops the geometry
// casting a shadow out of the field. The receiving ground is still inside the
// footprint, so it keeps being re-shaded - and its rays now find nothing, so it
// re-shades as LIT. Walk back and the shadow returns. Same ground, different
// answer every few steps, and because a footprint edge is straight while a shadow
// is not, the boundary between the two crawls across the ground as a staircase.
//
// The distinction that fixes it: a ray that ran off the edge of the field did not
// discover that the texel is lit. It discovered NOTHING. The atlas declines to
// store a non-answer over an answer, so the shadow measured while the caster was
// in range survives the caster leaving.
{
  const grid = wallGrid(3);
  const inShadow = { x: 6 * BLOCK_METRES, y: GROUND_Y, z: (WALL_Z - 2) * BLOCK_METRES };

  // Ground in the wall's umbra, with the wall well inside the field.
  const hit = coneTraceSample(grid, inShadow, UP, GSUN, { maxDistance: 12 });
  truthy('a ray that reaches its caster concludes', hit.concluded);
  truthy('and reports the shadow', hit.visible < 0.02);

  // Open ground whose ray leaves the grid without meeting anything. It has not
  // established that the texel is lit - only that it could not tell.
  const open = { x: 6 * BLOCK_METRES, y: GROUND_Y, z: 0 };
  const miss = coneTraceSample(grid, open, UP, { x: 0, y: 0.35, z: -1 },
                               { maxDistance: 12 });
  falsy('a ray that runs off the edge of the field concludes nothing', miss.concluded);

  // Spending the whole distance budget IS a conclusion - anything past maxDistance
  // is out of scope by design rather than unknown - so a short cap must not be
  // mistaken for ignorance and stop the atlas ever updating.
  const capped = coneTraceSample(grid, inShadow, UP, GSUN, { maxDistance: 0.5 });
  truthy('but spending the whole distance budget does conclude', capped.concluded);

  // The property the store relies on: every texel that reports a shadow reports a
  // conclusion too. If a shadow could ever arrive unconcluded it would never be
  // stored, and the caching would silently do nothing.
  {
    let umbra = 0, umbraUnconcluded = 0, penumbra = 0, penumbraUnconcluded = 0;
    // Stepped by TEXEL, not by block: the penumbra here is well under a metre,
    // so block-fraction steps land only two samples inside it and the ratio below
    // would be measuring the sweep rather than the shadow.
    for (let z = (WALL_Z - 1) * BLOCK_METRES; z >= -0.5 * BLOCK_METRES; z -= VOXEL_METRES) {
      const p = { x: 6 * BLOCK_METRES, y: GROUND_Y, z };
      const r = coneTraceSample(grid, p, UP, GSUN, { maxDistance: 12 });
      if (r.visible < 0.02) { umbra++; if (!r.concluded) umbraUnconcluded++; }
      else if (r.visible < 0.98) { penumbra++; if (!r.concluded) penumbraUnconcluded++; }
    }
    truthy('the sweep found umbra and penumbra to check', umbra > 4 && penumbra > 4);
    // The umbra is the part that must always be storable. A ray that reached an
    // occluder solid enough to black the texel out has, by definition, concluded -
    // so if this were ever non-zero the caching would silently do nothing for the
    // exact case it exists for.
    ok('every fully shadowed texel concluded, so all of them are storable',
       umbraUnconcluded, 0);
    // A PENUMBRA texel can legitimately fail to conclude: the cone picked up
    // partial occlusion and then ran off the edge before it could find out
    // whether there was more. Those keep their previous value rather than being
    // overwritten, which is the conservative answer - it is only ever the outer
    // lip of a penumbra sitting on the field boundary, never the umbra.
    note(`${penumbraUnconcluded}/${penumbra} penumbra texels did not conclude`);
    inRange('and only a few penumbra texels at the boundary do not',
            penumbraUnconcluded, 0, Math.ceil(penumbra * 0.25));
  }
}

section('quantise and dither');
// A smooth gradient under flat-shaded pixel art reads as a rendering bug; the
// soft tail is stepped so it reads as intentional, and dithered so the step
// boundary breaks up across texels instead of forming a contour line.
ok('fully lit stays fully lit', quantiseShadow(1, 4), 1);
ok('fully dark stays fully dark', quantiseShadow(0, 4), 0);
ok('four steps put a half-lit texel on a step boundary', quantiseShadow(0.5, 4), 0.5);
ok('values between steps snap down', quantiseShadow(0.4, 4), 0.25);
ok('the dither is what carries a value up to the next step',
   quantiseShadow(0.4, 4, 0.7), 0.5);
ok('out-of-range input is clamped, not wrapped', quantiseShadow(1.4, 4), 1);
ok('and so is negative', quantiseShadow(-0.2, 4), 0);
{
  const n = 8;
  const steps = new Set();
  for (let i = 0; i <= 40; i++) steps.add(quantiseShadow(i / 40, n));
  truthy(`${n} steps produce no more than ${n + 1} distinct levels`, steps.size <= n + 1);
}

ok('the Bayer matrix is 4x4', BAYER4.length, 16);
ok('and is a permutation of 0..15', new Set(BAYER4).size, 16);
near('its mean is a half, so the dither adds no net brightness',
     BAYER4.reduce((a, b) => a + b, 0) / 16 / 16, 0.46875, 1e-12);
inRange('bayer4 stays in [0,1)', bayer4(3, 3), 0, 0.9999);
ok('and tiles with period 4', bayer4(1, 2), bayer4(5, 6));
ok('negative texel coordinates wrap rather than reading undefined',
   bayer4(-1, -1), bayer4(3, 3));

// Anchored to the world texel, not the page texel: a page moving within the
// atlas must not change the pattern painted on it, and a screen-anchored pattern
// would crawl across surfaces as the camera moves.
{
  const a = worldTexelIndex({ x: 0, y: 0, z: 0 });
  ok('the origin is texel 0', a.tx, 0);
  const b = worldTexelIndex({ x: VOXEL_METRES * 3.5, y: 0, z: 0 });
  ok('and a point 3.5 voxels along x is texel 3', b.tx, 3);
  // y folds into the same axis as z so that stacked faces do not share a pattern
  // row, which would draw a visible seam up a wall.
  const c = worldTexelIndex({ x: 0, y: VOXEL_METRES * 2, z: 0 });
  ok('height shifts the pattern so stacked faces differ', c.ty, 2);
}

section('the range limit, stated rather than hidden');
// The one thing the CSM did that this does not. A sun ray has no termination
// distance, so it truncates where the field's footprint ends and the shadow
// simply stops. This is a property of how far the grid reaches, not of the
// traversal or the softening, and the test exists so it is a known number rather
// than a surprise at the screen edge.
{
  const grid = wallGrid(6);
  // Straight up the +Y axis from above open ground: the march runs out of grid
  // before it runs out of the 12 m cap, and open sky is the conservative answer.
  //
  // Started a metre clear of the surface deliberately. From exactly ON the ground
  // the field reads zero, which is below HIT_EPS, so the very first sample is a
  // hit and the ray never travels at all - which is the self-shadowing that
  // SURFACE_BIAS_VOXELS exists to prevent, and what coneVisibility applies for
  // every shading ray. Reaching for sphereTrace directly here means applying it
  // by hand, so the lift is explicit rather than assumed.
  const p = { x: 6 * BLOCK_METRES, y: GROUND_Y + 1, z: 0 };
  const far = sphereTrace(grid, p, { x: 0, y: 1, z: 0 }, 12);
  falsy('a ray leaving the grid reports no hit - open sky', far.hit);
  const reach = 12 * VOXEL_METRES * 12;  // 144 voxels of footprint
  note(`the footprint reaches ${reach.toFixed(1)} m on a side, so a shadow ` +
       `longer than that truncates; at a 20 degree sun that is a caster ` +
       `${(reach * Math.tan(EL)).toFixed(1)} m tall`);
  truthy('the footprint is smaller than the 12 m march cap on the diagonal',
    Math.hypot(reach, reach) > 12);
}

section('texel lock: one texel, one position, bit for bit');
{
  // Two fragments of the same ground texel, their interpolated y off by a
  // rounding error either way - the drift that let a texel reflect bands.
  const up = { x: 0, y: 1, z: 0 };
  const a = texelLock({ x: 1.01, y: 0.75 - 1e-7, z: 2.02 }, up);
  const b = texelLock({ x: 1.12, y: 0.75 + 1e-7, z: 2.11 }, up);
  ok('same texel, identical x', a.x, b.x);
  ok('same texel, identical y (on the face plane)', a.y, b.y);
  ok('same texel, identical z', a.z, b.z);
  ok('y is exactly the face', a.y, 0.75);
  near('x is the texel centre', a.x, 1.0625, 1e-12);
  const side = texelLock({ x: 3.75 + 2e-7, y: 1.3, z: 0.4 }, { x: 1, y: 0, z: 0 });
  ok('a wall face snaps onto its plane', side.x, 3.75);
}
