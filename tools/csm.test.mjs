import { file, section, ok, near, truthy, falsy, inRange, note } from './lib/harness.mjs';
import {
  SHADOW_DIM, CASCADE_COUNT, cascadeTexelMetres, cascadeExtent, cascadeDepth,
  lightBasis, fitCascade, worldToShadow, selectCascade,
  penumbraTexels, bayer4, BAYER4, quantiseShadow, worldTexelIndex,
  SUN_ANGULAR_SIZE, readbackLayout, verticalSpan, shadowUVFromNDC
} from '../js/csm.js';
import { VOXEL_METRES } from '../js/boxgrid.js';
import { BLOCK_METRES } from '../js/world.js';
import { faceBasis, FACE_NORMALS, FACES_PER_BLOCK } from '../js/atlas.js';

file('csm.test.mjs - cascaded sun shadows and distance-based softening');

const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const len = a => Math.hypot(a.x, a.y, a.z);

section('cascades step in lockstep with the texel grid');
// Doc 7.3: grid resolution is locked 1:1 to texel resolution AT EVERY LoD level,
// so a cascade that doubles its voxel size must be marched by texels at the
// matching mip. Keeping each cascade texel a whole number of boxGrid voxels is
// what makes that possible at all.
ok('three cascades', CASCADE_COUNT, 3);
near('C0 texel is one boxGrid voxel', cascadeTexelMetres(0), VOXEL_METRES, 1e-12);
near('C1 doubles it', cascadeTexelMetres(1), VOXEL_METRES * 2, 1e-12);
near('C2 doubles again', cascadeTexelMetres(2), VOXEL_METRES * 4, 1e-12);
for (let i = 0; i < CASCADE_COUNT; i++) {
  const ratio = cascadeTexelMetres(i) / VOXEL_METRES;
  ok(`C${i} texel is a whole number of voxels`, ratio, 1 << i);
}
// And this has to reproduce the cascade table in doc 4 exactly.
near('C0 covers one chunk, 18 m', cascadeExtent(0), 18, 1e-9);
near('C1 covers 36 m', cascadeExtent(1), 36, 1e-9);
near('C2 covers 72 m', cascadeExtent(2), 72, 1e-9);
ok('the map is the same 144 as C0', SHADOW_DIM, 144);
note(`3 x ${SHADOW_DIM}^2 R16F = ${(3 * SHADOW_DIM ** 2 * 2 / 1024).toFixed(0)} KB total`);

section('the light basis is orthonormal for any sun');
for (const sun of [{ x: 1, y: 1, z: 1 }, { x: 0, y: 1, z: 0 }, { x: -3, y: 0.4, z: 2 },
                   { x: 0, y: -1, z: 0 }, { x: 5, y: 0, z: 0 }]) {
  const b = lightBasis(sun);
  const tag = `sun ${sun.x},${sun.y},${sun.z}`;
  near(`${tag}: forward is unit`, len(b.forward), 1, 1e-9);
  near(`${tag}: right is unit`, len(b.right), 1, 1e-9);
  near(`${tag}: up is unit`, len(b.up), 1, 1e-9);
  near(`${tag}: right perpendicular to forward`, dot(b.right, b.forward), 0, 1e-9);
  near(`${tag}: up perpendicular to forward`, dot(b.up, b.forward), 0, 1e-9);
  near(`${tag}: right perpendicular to up`, dot(b.right, b.up), 0, 1e-9);
  // The camera has to look ALONG the sun direction reversed, since the sun
  // vector points toward the sun. Getting this backwards lights the night side.
  truthy(`${tag}: forward opposes the sun vector`, dot(b.forward, sun) < 0);
}
// Straight overhead is the degenerate case for a world-up basis.
const overhead = lightBasis({ x: 0, y: 1, z: 0 });
near('an overhead sun still produces a valid basis', len(overhead.right), 1, 1e-9);

section('a fitted cascade contains what it should');
const sun = { x: 0.6, y: 0.35, z: 0.6 };
const centre = { x: 10, y: 1.5, z: 10 };
const c0 = fitCascade(0, centre, sun);

const sc = worldToShadow(c0, centre);
inRange('the centre lands in the middle of the map (u)', sc.u, 0.49, 0.51);
inRange('the centre lands in the middle of the map (v)', sc.v, 0.49, 0.51);
// The depth window is centred on the authorable vertical MID-RANGE, not on the
// query point's height, so it is the mid-height point that sits at 0.5. The
// player's own height lands wherever it lands, and must simply be inside.
inRange('the centre is inside the depth range', sc.depth, 0, 1);
// The depth window's own centre - the camera position pushed half the range
// along the view direction - must sit at the authorable vertical mid-height.
// That is the property that makes the range cover the whole span symmetrically,
// and it is exact rather than approximate.
const windowCentre = {
  x: c0.position.x + c0.forward.x * c0.depth / 2,
  y: c0.position.y + c0.forward.y * c0.depth / 2,
  z: c0.position.z + c0.forward.z * c0.depth / 2
};
near('the depth window is centred on the authorable vertical mid-height',
     windowCentre.y, verticalSpan().mid, 1e-9);
near('so the window centre is exactly halfway through the depth range',
     worldToShadow(c0, windowCentre).depth, 0.5, 1e-9);

// A point one cascade-texel to the light-space right must move exactly one
// texel across the map. If this drifts, so does every shadow edge.
const oneRight = {
  x: centre.x + c0.right.x * c0.texel,
  y: centre.y + c0.right.y * c0.texel,
  z: centre.z + c0.right.z * c0.texel
};
near('one texel of world motion is one texel of map motion',
     (worldToShadow(c0, oneRight).u - sc.u) * SHADOW_DIM, 1, 1e-6);

// Depth must be linear in distance along the light, because that is exactly
// what the depth pass writes and the comparison is a plain subtraction.
const along = {
  x: centre.x + c0.forward.x * 3,
  y: centre.y + c0.forward.y * 3,
  z: centre.z + c0.forward.z * 3
};
near('depth is linear along the light direction',
     worldToShadow(c0, along).depth - sc.depth, 3 / c0.depth, 1e-9);

section('texel snapping');
// Without this, moving the view slides the map by a fraction of a texel and
// every shadow edge re-quantises. In texel-space shading that does not shimmer
// continuously - it flickers the instant a page is re-shaded, which is worse.
const a = fitCascade(0, { x: 10, y: 1.5, z: 10 }, sun);
const b = fitCascade(0, { x: 10.01, y: 1.5, z: 10.02 }, sun);
near('a sub-texel move does not move the map at all (right)',
     dot(a.position, a.right) - dot(b.position, b.right), 0, 1e-9);
near('nor vertically in light space',
     dot(a.position, a.up) - dot(b.position, b.up), 0, 1e-9);
const far = fitCascade(0, { x: 10 + cascadeTexelMetres(0) * 4, y: 1.5, z: 10 }, sun);
truthy('but a multi-texel move does',
       Math.abs(dot(a.position, a.right) - dot(far.position, far.right)) > 1e-6);
// Whatever it snaps to must still be a whole number of texels from the origin.
const snapped = dot(a.position, a.right) / a.texel;
near('the snapped origin is a whole number of texels',
     Math.abs(snapped - Math.round(snapped)), 0, 1e-6);

section('cascade selection falls through outward');
const cascades = [0, 1, 2].map(i => fitCascade(i, centre, sun));
ok('a point at the centre uses the finest cascade', selectCascade(cascades, centre), 0);
// Displaced LATERALLY in light space. Displacing it in world space is not the
// same test: most of a diagonal offset can land along the light direction,
// which changes depth rather than map position and legitimately stays in C0.
const lateral = 12;
const mid = { x: centre.x + c0.right.x * lateral, y: centre.y + c0.right.y * lateral,
              z: centre.z + c0.right.z * lateral };
ok('a point past C0 falls through to exactly C1', selectCascade(cascades, mid), 1);
const outer = { x: centre.x + c0.right.x * 30, y: centre.y + c0.right.y * 30,
                z: centre.z + c0.right.z * 30 };
ok('and further out, to C2', selectCascade(cascades, outer), 2);
ok('a point past every cascade reports none, and reads as open sky',
   selectCascade(cascades, { x: 500, y: 0, z: 500 }), -1);
// The reason this replaces the sphere trace: the trace could not answer past
// 18 m at all, because a ray leaving C0 reads as unshadowed.
truthy('coverage now reaches well past one chunk', cascadeExtent(2) > 4 * 18 * 0.9);

section('softening grows with distance, and shrinks with cascade coarseness');
// Doc 6.1: the sample footprint widens with the distance the light travelled
// between occluder and receiver, because that is how a penumbra grows.
const near0 = penumbraTexels(0.5, 0.5 - 0.1 / c0.depth, c0);
const far0 = penumbraTexels(0.5, 0.5 - 4.0 / c0.depth, c0);
truthy('contact is nearly hard', near0 < 0.5);
truthy('a distant caster spreads', far0 > near0 * 8);
note(`10 cm separation -> ${near0.toFixed(2)} texels, 4 m -> ${far0.toFixed(2)} texels`);
near('the penumbra scales linearly with separation', far0 / near0, 40, 1e-6);

// The same physical penumbra must be a SMALLER kernel in a coarser cascade -
// that is what keeps cost down where detail matters least, and it falls out of
// expressing the radius in shadow texels rather than metres.
const c2 = cascades[2];
const far2 = penumbraTexels(0.5, 0.5 - 4.0 / c2.depth, c2);
truthy('a coarser cascade needs fewer texels for the same penumbra', far2 < far0);
near('specifically four times fewer, matching its four-times texel',
     far0 / far2, 4, 1e-6);

section('quantised, dithered falloff');
// A smooth gradient under flat-shaded pixel art reads as a rendering bug;
// stepped and dithered reads as intentional.
ok('fully lit stays lit', quantiseShadow(1, 4), 1);
ok('fully dark stays dark', quantiseShadow(0, 4), 0);
const levels = new Set();
for (let i = 0; i <= 40; i++) levels.add(quantiseShadow(i / 40, 4));
inRange('a continuous ramp collapses to a handful of levels', levels.size, 2, 5);
truthy('and they are all in range', [...levels].every(v => v >= 0 && v <= 1));
ok('out-of-range input is clamped, not wrapped', quantiseShadow(2.5, 4), 1);
ok('and on the dark side too', quantiseShadow(-2.5, 4), 0);
// The dither has to be able to push a value across a step boundary, or it does
// nothing at all.
truthy('dither shifts values across step boundaries',
       quantiseShadow(0.24, 4, 0.9) > quantiseShadow(0.24, 4, 0));

section('the Bayer matrix');
ok('16 entries', BAYER4.length, 16);
ok('every value 0..15 appears exactly once', new Set(BAYER4).size, 16);
truthy('and they are exactly 0..15',
       BAYER4.slice().sort((x, y) => x - y).every((v, i) => v === i));
ok('the lookup normalises to [0,1)', bayer4(0, 0), 0);
near('the largest entry is 15/16', bayer4(0, 3), 15 / 16, 1e-12);
// Repeat wrapping in the shader depends on this holding for any integer.
ok('it tiles on 4 in x', bayer4(5, 2), bayer4(1, 2));
ok('it tiles on 4 in y', bayer4(2, 9), bayer4(2, 1));
ok('and negative coordinates wrap rather than going out of bounds',
   bayer4(-1, -1), bayer4(3, 3));

section('the dither is anchored to the world, not the screen or the atlas');
// Screen-indexed dithering crawls across surfaces under camera motion. Anchored
// to the world texel it is painted onto them - the same argument as the atlas.
// Using the WORLD texel rather than the page texel also means a page moving
// within the atlas cannot change the pattern on it.
const t1 = worldTexelIndex({ x: 4, y: 1.5, z: 9 });
const t2 = worldTexelIndex({ x: 4 + VOXEL_METRES * 4, y: 1.5, z: 9 });
ok('four texels along, the pattern repeats', bayer4(t1.tx, t1.ty), bayer4(t2.tx, t2.ty));
const t3 = worldTexelIndex({ x: 4 + VOXEL_METRES, y: 1.5, z: 9 });
truthy('one texel along, it does not', bayer4(t1.tx, t1.ty) !== bayer4(t3.tx, t3.ty));
const t4 = worldTexelIndex({ x: 4.02, y: 1.5, z: 9.01 });
ok('and a sub-texel move changes nothing', bayer4(t1.tx, t1.ty), bayer4(t4.tx, t4.ty));

section('the face normal the shader derives is the outward one');
// The atlas kernel takes the face normal as normalize(cross(uStep, vStep)) - no
// extra buffer, no branch on face. That only works if atlas.js orders the two
// steps consistently across all six faces. A flipped normal would invert the
// Lambert term and light the inside of the world, on that face alone.
for (let f = 0; f < FACES_PER_BLOCK; f++) {
  const bs = faceBasis(f);
  const n = FACE_NORMALS[f];
  const c = {
    x: bs.uStep.y * bs.vStep.z - bs.uStep.z * bs.vStep.y,
    y: bs.uStep.z * bs.vStep.x - bs.uStep.x * bs.vStep.z,
    z: bs.uStep.x * bs.vStep.y - bs.uStep.y * bs.vStep.x
  };
  const l = len(c);
  near(`face ${f}: cross(u,v) points outward`,
       (c.x * n.x + c.y * n.y + c.z * n.z) / l, 1, 1e-9);
}

section('the sun angular size is a style knob, and documented as one');
truthy('larger than the real sun, so penumbrae are actually visible',
       SUN_ANGULAR_SIZE > 0.0093);
inRange('but still small enough to read as a sun, not an area light',
        SUN_ANGULAR_SIZE, 0.0093, 0.2);

// --- Readback padding ---
//
// The bug this guards: WebGPU pads copyTextureToBuffer rows to 256 bytes, and
// deriving the row stride as length/(w*h) gave 1.77 for a 144-wide R16 map.
// Fractional indices into a typed array are undefined, undefined decoded to 0,
// and a healthy depth map was reported as entirely empty. Cheap arithmetic, and
// it cost a whole debugging round.
section('reading a cascade back accounts for the 256-byte row padding');
{
  // 144 texels of R16: 288 bytes per row, padded to 512 -> 256 elements.
  const half1 = (144 - 1) * 512 / 2 + 144;
  const l = readbackLayout(half1, 2, 144, 144);
  truthy('R16 144x144 readback layout is recognised', l);
  ok('R16 144x144 decodes as one channel', l.channels, 1);
  ok('R16 144x144 row stride is 256 elements, not 144', l.rowElements, 256);

  // The naive stride that caused the failure.
  const naive = half1 / (144 * 144);
  falsy('length/(w*h) is an integer - it is not, and that was the bug',
        Number.isInteger(naive));
  near('the old naive stride was 1.772', naive, 1.7723765, 1e-6);

  // RGBA8: 576 bytes per row, padded to 768.
  const rgba8 = (144 - 1) * 768 + 144 * 4;
  const l8 = readbackLayout(rgba8, 1, 144, 144);
  ok('RGBA8 144x144 decodes as four channels', l8 && l8.channels, 4);
  ok('RGBA8 144x144 row stride is 768 bytes', l8 && l8.rowElements, 768);

  // A width that already lands on the alignment needs no padding at all.
  const w64 = (64 - 1) * 256 / 4 + 64;
  const l64 = readbackLayout(w64, 4, 64, 64);
  ok('a 64-wide R32 map is already aligned', l64 && l64.rowElements, 64);

  // Anything that fits nothing must refuse rather than guess: an invented
  // stride carries false confidence, which is what did the damage.
  ok('an unmatchable length returns null', readbackLayout(12345, 2, 144, 144), null);
  ok('an empty readback returns null', readbackLayout(0, 2, 144, 144), null);

  // The last texel must sit inside the buffer - the off-by-one the spec's size
  // formula ((h-1)*row + w*bpt, not h*row) exists for.
  const last = (144 - 1) * l.rowElements + (144 - 1) * l.channels;
  truthy('the last texel index lies inside the buffer', last < half1);

  // Defaults come from SHADOW_DIM, so the cascade size needs no restating.
  const d = readbackLayout((SHADOW_DIM - 1) * 512 / 2 + SHADOW_DIM, 2);
  ok('defaults to the cascade dimensions', d && d.rowElements, 256);
}

// --- The depth range has to follow the sun ---
//
// The bug this guards: cascadeDepth was `extent + vertical * 2`, which ignored
// the sun angle. At the project's default sun (10, 6, 10) - 23 degrees - it gave
// 54 m where 88.5 m is needed, so casters near the top or bottom of the
// authorable range clipped out of the depth pass and their shadows vanished.
//
// The test is written as a BRUTE-FORCE CONTAINMENT CHECK rather than a repeat of
// the formula, so it fails if the formula is wrong rather than agreeing with it.
section('the cascade depth range contains every caster, at any sun angle');
{
  const VERTICAL = (12) * BLOCK_METRES;      // Y_MIN..Y_MAX inclusive, 18 m
  const yLo = -3 * BLOCK_METRES, yHi = 9 * BLOCK_METRES;

  const suns = [
    { label: 'the default sun, 23 deg', s: { x: 10, y: 6, z: 10 } },
    { label: 'a high sun, 60 deg',      s: { x: 1, y: 1.73, z: 0 } },
    { label: 'straight overhead',       s: { x: 0, y: 1, z: 0 } },
    { label: 'a low sun, 10 deg',       s: { x: 10, y: 1.76, z: 0 } },
    { label: 'a low sun from -z',       s: { x: 0, y: 1.76, z: -10 } }
  ];

  for (const { label, s } of suns) {
    const c = fitCascade(0, { x: 0, y: 0, z: 0 }, s);
    // Every corner of the footprint, at the top and bottom of the vertical
    // range, must land strictly inside 0..1 of the depth range. The corners are
    // the extremes, so checking them checks the whole box.
    let worstLo = Infinity, worstHi = -Infinity;
    for (const su of [-0.5, 0.5]) {
      for (const sr of [-0.5, 0.5]) {
        for (const y of [yLo, yHi]) {
          // Build the world point with the given light-space lateral offsets
          // and the given height, then ask where its depth falls.
          const u = su * c.extent, r = sr * c.extent;
          // y = u*up.y + r*right.y + depth*forward.y  ->  solve for depth
          const fy = c.forward.y;
          if (Math.abs(fy) < 1e-6) continue;
          const d = (y - u * c.up.y - r * c.right.y) / fy;
          const p = {
            x: c.right.x * r + c.up.x * u + c.forward.x * d,
            y: c.right.y * r + c.up.y * u + c.forward.y * d,
            z: c.right.z * r + c.up.z * u + c.forward.z * d
          };
          const sh = worldToShadow(c, p);
          if (sh.depth < worstLo) worstLo = sh.depth;
          if (sh.depth > worstHi) worstHi = sh.depth;
        }
      }
    }
    truthy(label + ': the nearest corner is inside the near plane', worstLo >= -1e-9);
    truthy(label + ': the furthest corner is inside the far plane', worstHi <= 1 + 1e-9);
    // Tight, not merely sufficient: a padded range throws away half-float
    // precision, and a short one clips casters out of the pass entirely.
    near(label + ': and the range is exactly filled, not padded',
         worstHi - worstLo, 1, 1e-9);
  }

  // Overhead is the tight case: nothing but the vertical range is needed, and
  // the range must not be inflated past it.
  near('overhead needs only the vertical range',
       cascadeDepth(0, { x: 0, y: 1, z: 0 }), VERTICAL, 1e-6);

  // The default sun is the case that was broken, pinned to the number.
  near('the default sun needs 88.5 m, not the 54 m the guess gave',
       cascadeDepth(0, { x: 10, y: 6, z: 10 }), 88.513, 0.01);
  truthy('which is well past the old extent + vertical*2 guess',
         cascadeDepth(0, { x: 10, y: 6, z: 10 }) > cascadeExtent(0) + VERTICAL * 2);

  // A lower sun always needs a longer range - monotonic, with no crossover.
  let prev = 0;
  for (const yy of [8, 6, 4, 2, 1]) {
    const d = cascadeDepth(0, { x: 10, y: yy, z: 10 });
    truthy('a lower sun (y=' + yy + ') needs a longer range', d > prev);
    prev = d;
  }

  // A grazing sun must be clamped rather than running to infinity, or one sunset
  // frame allocates a kilometre of depth and half float stops resolving.
  const graze = cascadeDepth(0, { x: 1, y: 0, z: 0 });
  truthy('a sun on the horizon is clamped, not infinite', Number.isFinite(graze));
  truthy('and clamped to something a half float can still resolve', graze < 1000);

  // Coarser cascades cover more ground, so they need more depth - the same
  // lockstep that ties resolution to the texel grid.
  const sun = { x: 10, y: 6, z: 10 };
  truthy('C1 needs a longer range than C0', cascadeDepth(1, sun) > cascadeDepth(0, sun));
  truthy('C2 needs a longer range than C1', cascadeDepth(2, sun) > cascadeDepth(1, sun));

  // fitCascade must use the sun-aware range, not a stale fixed one.
  near('fitCascade carries the sun-aware depth',
       fitCascade(0, { x: 0, y: 0, z: 0 }, sun).depth, cascadeDepth(0, sun), 1e-9);
}

// --- The basis must match the camera that rasterises the map ---
//
// The bug this guards: `right` was built as cross(up, forward), off the view
// direction. A three camera looks down its own -Z, so Matrix4.lookAt builds its
// x axis as cross(up, z) with z = -forward. The two differ by a sign, so `right`
// came out NEGATED for every sun direction and the shadow map was mirrored along
// u. `up` was unaffected, which is why it read as a flip rather than a transpose.
//
// This test RE-DERIVES three's lookAt independently rather than restating
// lightBasis, so it fails when the basis is wrong instead of agreeing with it.
section('the light basis matches the orthographic camera three builds from it');
{
  const cr = (a, b) => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  });
  const nm = (v) => {
    const l = Math.hypot(v.x, v.y, v.z) || 1;
    return { x: v.x / l, y: v.y / l, z: v.z / l };
  };
  const dt = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;

  const suns = [
    { label: 'the default sun', s: { x: 10, y: 6, z: 10 } },
    { label: 'straight overhead', s: { x: 0, y: 1, z: 0 } },
    { label: 'from -x -z', s: { x: -5, y: 2, z: -3 } },
    { label: 'from +x -z', s: { x: 5, y: 2, z: -3 } },
    { label: 'a grazing sun', s: { x: 1, y: 0.1, z: 0 } },
    { label: 'along +z only', s: { x: 0, y: 3, z: 8 } }
  ];

  for (const { label, s } of suns) {
    const b = lightBasis(s);
    // Matrix4.lookAt(eye, target, up), with the camera's up set to b.up and the
    // target one unit along b.forward - exactly what updateCascadeCameras does.
    const z = nm({ x: -b.forward.x, y: -b.forward.y, z: -b.forward.z });
    const x = nm(cr(b.up, z));
    const y = cr(z, x);

    near(label + ': right matches the camera x axis', dt(x, b.right), 1, 1e-9);
    near(label + ': up matches the camera y axis', dt(y, b.up), 1, 1e-9);
    // The sign is the whole bug, so assert it as a sign and not just a magnitude.
    truthy(label + ': right is not the NEGATED camera x axis', dt(x, b.right) > 0);

    // A camera basis is right-handed about its own +Z, which is -forward.
    near(label + ': right x up = -forward (camera handedness)',
         dt(cr(b.right, b.up), b.forward), -1, 1e-9);

    // Orthonormal, or the map is sheared as well as mirrored.
    near(label + ': right is unit length', Math.hypot(b.right.x, b.right.y, b.right.z), 1, 1e-9);
    near(label + ': up is unit length', Math.hypot(b.up.x, b.up.y, b.up.z), 1, 1e-9);
    near(label + ': right . up = 0', dt(b.right, b.up), 0, 1e-9);
    near(label + ': right . forward = 0', dt(b.right, b.forward), 0, 1e-9);
    near(label + ': up . forward = 0', dt(b.up, b.forward), 0, 1e-9);

    // `right` is the horizontal axis by construction, and cascadeDepth relies on
    // right.y being exactly zero to drop a term from its bound.
    near(label + ': right is horizontal, so cascadeDepth may drop its term',
         b.right.y, 0, 1e-12);
  }

  // A concrete orientation check, worked by hand: with the sun in the +x +z
  // quadrant the light looks toward -x -z, and moving WEST (-x) must not move
  // the same way in the map as moving NORTH (-z). If right were mirrored, one of
  // these two would land on the wrong side of centre.
  const c = fitCascade(0, { x: 0, y: 0, z: 0 }, { x: 10, y: 6, z: 10 });
  const at = (x, z) => worldToShadow(c, { x, y: 0, z });
  const o = at(0, 0);
  truthy('moving +x moves the same direction in u as the camera x axis',
         Math.sign(at(5, 0).u - o.u) === Math.sign(c.right.x));
  truthy('moving +z moves the same direction in u as the camera x axis',
         Math.sign(at(0, 5).u - o.u) === Math.sign(c.right.z));
  truthy('u and v are not the same axis', Math.abs(at(5, 0).u - o.u) > 1e-6 ||
         Math.abs(at(5, 0).v - o.v) > 1e-6);
}

// --- Clip space to texture coordinates, against a hand-built projection ---
//
// The bug this guards: the shadow lookup mapped clip space to texture space with
// the WebGL convention v = ndc.y * 0.5 + 0.5. WebGPU's texture row 0 is the TOP
// row, so v runs opposite the light's up axis and the correct mapping carries a
// flip. Without it every depth fetch came from the vertically mirrored position
// in the map, which reads as big misplaced dark regions rather than as an
// upside-down picture.
//
// The projection matrix is BUILT BY HAND here - ortho times lookAt, the same way
// three does it - so this is an independent check of worldToShadow rather than a
// restatement of it.
section('clip space maps to texture coordinates with webgpu\'s flipped v');
{
  const cr = (a, b) => ({
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  });
  const nm = (v) => {
    const l = Math.hypot(v.x, v.y, v.z) || 1;
    return { x: v.x / l, y: v.y / l, z: v.z / l };
  };
  const dt = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;

  const sun = { x: 10, y: 6, z: 10 };
  const c = fitCascade(0, { x: 4, y: 1.5, z: -7 }, sun);

  // three's camera basis, rebuilt from scratch.
  const z = nm({ x: -c.forward.x, y: -c.forward.y, z: -c.forward.z });
  const xa = nm(cr(c.up, z));
  const ya = cr(z, xa);

  // An orthographic camera's NDC, worked directly rather than through a matrix:
  // view coords are (dot(d, x), dot(d, y), dot(d, z)) for d = p - eye, and the
  // ortho projection scales x and y by 1/halfExtent.
  const ndcOf = (p) => {
    const d = { x: p.x - c.position.x, y: p.y - c.position.y, z: p.z - c.position.z };
    return { x: dt(d, xa) / c.halfExtent, y: dt(d, ya) / c.halfExtent };
  };

  const probes = [
    { label: 'the cascade centre', p: { x: 4, y: 1.5, z: -7 } },
    { label: 'a point up the light axis', p: { x: 4 + c.up.x * 6, y: 1.5 + c.up.y * 6, z: -7 + c.up.z * 6 } },
    { label: 'a point along the light right', p: { x: 4 + c.right.x * 5, y: 1.5, z: -7 + c.right.z * 5 } },
    { label: 'an off-axis point', p: { x: -2, y: 9, z: 3 } }
  ];

  for (const { label, p } of probes) {
    const ndc = ndcOf(p);
    const tex = shadowUVFromNDC(ndc.x, ndc.y);
    const sh = worldToShadow(c, p);
    near(label + ': u agrees with worldToShadow', tex.u, sh.u, 1e-9);
    // The flip is the point: texture v is the COMPLEMENT of the light-space v.
    near(label + ': texture v is 1 - the light-space v', tex.v, 1 - sh.v, 1e-9);
  }

  // Stated as a direction, because a symmetric probe set would pass either way:
  // moving along the light's UP axis must DECREASE the texture v.
  const base = { x: 4, y: 1.5, z: -7 };
  const upward = { x: base.x + c.up.x * 3, y: base.y + c.up.y * 3, z: base.z + c.up.z * 3 };
  const vBase = shadowUVFromNDC(ndcOf(base).x, ndcOf(base).y).v;
  const vUp = shadowUVFromNDC(ndcOf(upward).x, ndcOf(upward).y).v;
  truthy('moving along the light up axis decreases the texture v', vUp < vBase);
  truthy('and the light-space v increases, which is the disagreement', 
         worldToShadow(c, upward).v > worldToShadow(c, base).v);

  // The centre of the map is the one place the flip is invisible, which is why a
  // centre-only check would have missed this entirely.
  const mid = shadowUVFromNDC(0, 0);
  near('the map centre is v = 0.5 either way', mid.v, 0.5, 1e-12);
  near('and u = 0.5', mid.u, 0.5, 1e-12);

  // The corners, spelled out, so the orientation is documented by assertion.
  near('ndc y = +1 (top) is texture v = 0', shadowUVFromNDC(0, 1).v, 0, 1e-12);
  near('ndc y = -1 (bottom) is texture v = 1', shadowUVFromNDC(0, -1).v, 1, 1e-12);
  near('ndc x = -1 (left) is texture u = 0', shadowUVFromNDC(-1, 0).u, 0, 1e-12);
  near('ndc x = +1 (right) is texture u = 1', shadowUVFromNDC(1, 0).u, 1, 1e-12);
}

// --- Texel-centre correction, not a guessed bias ---
//
// A depth map stores the depth at the TEXEL CENTRE, but the receiver sits
// somewhere inside that texel. That mismatch is what a bias was being asked to
// cover, and at the project's sun it is the same size as the bias itself, so no
// constant value works: it flips between acne and peter-panning as the cascade
// slides. Correcting onto the texel centre with the exact receiver-plane
// gradient removes it. These numbers are the case for doing that.
section('sub-texel depth error is the size of the bias, so it must be corrected');
{
  const sun = { x: 10, y: 6, z: 10 };
  const c = fitCascade(0, { x: 0, y: 0, z: 0 }, sun);
  const ground = { x: 0, y: 1, z: 0 };

  // Depth change of flat ground across ONE shadow texel, in metres.
  const perTexel = Math.abs(c.texel * c.up.y / c.forward.y);
  near('flat ground moves ~0.295 m of depth per 12.5 cm texel', perTexel, 0.295, 0.005);
  // Worst case is half a texel either side of the centre.
  truthy('so a half-texel sampling error is ~0.147 m', Math.abs(perTexel / 2 - 0.147) < 0.005);

  // The probe measured 0.258 m of margin at the player's feet. An error of the
  // same order as the margin is exactly the regime where a constant bias cannot
  // separate the two cases.
  truthy('which is the same order as the measured 0.258 m margin',
         perTexel / 2 > 0.258 * 0.3 && perTexel / 2 < 0.258 * 3);

  // The gradient is EXACT for a plane: stepping one texel along the light's up
  // axis must change the receiver depth by exactly the plane's own gradient.
  const step = { x: c.up.x * c.texel, y: c.up.y * c.texel, z: c.up.z * c.texel };
  const p0 = { x: 0, y: 0, z: 0 };
  // Stay on the ground plane: move along the horizontal projection instead, so
  // this measures the plane's gradient rather than a move off it.
  const horiz = Math.hypot(step.x, step.z) || 1;
  const onPlane = { x: p0.x + step.x / horiz * c.texel / Math.abs(c.up.y === 1 ? 1 : 1),
                    y: 0,
                    z: p0.z + step.z / horiz * c.texel };
  const d0 = worldToShadow(c, p0).depth;
  const d1 = worldToShadow(c, onPlane).depth;
  truthy('moving along the ground changes the receiver depth', Math.abs(d1 - d0) > 1e-9);

  // Half-float quantisation is RELATIVE, so the epsilon that absorbs it is a
  // constant in normalised depth and needs no per-cascade tuning. Check it is
  // comfortably larger than the step but far smaller than a texel of slope.
  const halfFloatStep = 0.5 / 2048;
  const eps = 0.75 * 0.001;
  truthy('the epsilon covers the half-float step', eps > halfFloatStep);
  for (const level of [0, 1, 2]) {
    const cl = fitCascade(level, { x: 0, y: 0, z: 0 }, sun);
    const slopePerTexel = Math.abs(cl.texel * cl.up.y / cl.forward.y) / cl.depth;
    truthy('C' + level + ': the epsilon is far below one texel of slope',
           eps < slopePerTexel * 0.5);
  }
}
