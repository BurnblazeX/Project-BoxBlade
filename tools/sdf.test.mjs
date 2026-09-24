import { file, section, ok, near, truthy, falsy, inRange, note } from './lib/harness.mjs';
import { createTestArea, BLOCK_METRES } from '../js/world.js';
import {
  createBoxGrid, gridIndex, scrollForHandoff, applyHandoff, TEXELS_PER_BLOCK, GRID_DIM, VOXEL_METRES,
  DISTANCE_RANGE, FAR_BYTE, encodeDistance, decodeDistance,
  distanceAt, isOccupied, sampleDistance, boxDistance,
  sphereTrace, marchOccupancy, voxelCentreToWorld, createBoxGridAt
} from '../js/boxgrid.js';

file('sdf.test.mjs - signed distance field and sphere tracing');

section('the byte encoding');
// An R8 texture samples normalised to [0,1] and the field stores c over [-1,+1],
// so the decode is c = sample*2-1 with nothing to divide by 255. A silent
// normalisation mistake here is exactly what broke the binary version, where an
// occupied voxel arrived as 1/255.
//
// The mapping is SQUARED: c = sign(d)*sqrt(|d|/RANGE), so resolution concentrates
// at the zero crossing - which is where the geometry is, and where a slope's
// sub-voxel position will live - and thins out far away, where all a ray needs is
// a rough answer to how much room there is.
ok('the far end of the range is the top of the byte', encodeDistance(DISTANCE_RANGE), FAR_BYTE);
ok('and the deep end is the bottom', encodeDistance(-DISTANCE_RANGE), 0);
near('a surface sits mid-range', encodeDistance(0), 127.5, 1);
ok('beyond the range clamps rather than wrapping', encodeDistance(DISTANCE_RANGE + 1), FAR_BYTE);
ok('and clamps on the solid side too', encodeDistance(-DISTANCE_RANGE - 1), 0);
truthy('the encoding is monotonic, which is why the bake can min raw bytes',
  encodeDistance(-0.7) < encodeDistance(-0.2) && encodeDistance(-0.2) < encodeDistance(0.6));
truthy('and stays monotonic across the whole range, not just near the surface',
  [-8, -4, -1, -0.1, 0, 0.1, 1, 4, 8].every((d, i, a) =>
    i === 0 || encodeDistance(a[i]) >= encodeDistance(a[i - 1])));

// THE PRECISION TRADE, as numbers rather than as a claim. A linear map over +-8
// would have cost 8x resolution everywhere including at the surface; the squared
// one is finer than the old +-1 linear map where it matters and coarser only
// where nothing reads it.
const quantumAt = d => Math.abs(decodeDistance(encodeDistance(d) + 1) -
                                decodeDistance(encodeDistance(d)));
inRange('at the surface a byte step is under a hundredth of a voxel',
        quantumAt(0), 0, 0.01);
truthy('which is FINER than the old +-1 linear map managed (1/127.5)',
       quantumAt(0) < 1 / 127.5);
inRange('at one voxel out it is a few hundredths', quantumAt(1), 0, 0.08);
inRange('and at the far end an eighth of a voxel, which is plenty for stepping',
        quantumAt(DISTANCE_RANGE - 0.01), 0, 0.2);
inRange('a round trip near the surface is essentially exact',
  Math.abs(decodeDistance(encodeDistance(0.25)) - 0.25), 0, 0.01);

// The shader has its own copy of the decode in TSL (decodeFieldTSL), and the CPU
// reference is only worth having if the two read the same byte the same way.
// This is the arithmetic, transcribed, so a divergence fails here rather than
// looking like a shadow bug.
const shaderDecode = byte => {
  const c = (byte / 255) * 2 - 1;
  return c * Math.abs(c) * DISTANCE_RANGE;
};
for (const d of [-8, -2, -0.5, 0, 0.5, 2, 8]) {
  near(`the shader decode agrees with the CPU one at ${d} voxels`,
       shaderDecode(encodeDistance(d)), decodeDistance(encodeDistance(d)), 0.02);
}

section('box distance');
// Exact, and the seam where every future block type plugs in: a slope is this
// function replaced, nothing else.
near('a point on a face is at zero', boxDistance(2, 0, 0, 0, 0, 0, 2, 2, 2), 0, 1e-12);
near('outside, along an axis', boxDistance(5, 0, 0, 0, 0, 0, 2, 2, 2), 3, 1e-12);
near('outside, past a corner', boxDistance(4, 4, 4, 0, 0, 0, 2, 2, 2),
     Math.sqrt(12), 1e-12);
near('inside is negative, and is the distance to the nearest face',
     boxDistance(1, 0, 0, 0, 0, 0, 2, 2, 2), -1, 1e-12);
near('the centre is the half-extent away', boxDistance(0, 0, 0, 0, 0, 0, 2, 2, 2), -2, 1e-12);

section('the baked field');
createTestArea(40, 40);
const g = createBoxGrid(0, 0);

ok('two bytes per voxel (opaque, glass), 5.70 MB', (g.data.length / 1048576).toFixed(2), '5.70');
// Ground only: createTestArea lays the slab, the wall and pillar come from
// addTestElevation and sit outside chunk (0,0) anyway.
ok('unsupported ground fills its top half only', g.occupiedCount,
   144 * (TEXELS_PER_BLOCK / 2) * TEXELS_PER_BLOCK * TEXELS_PER_BLOCK);

// Voxel 47 is the top voxel of the ground slab, 48 the air above it.
// Tolerance is 0.02 rather than 0.01 because half a voxel out is exactly where
// the squared encoding is coarsest in relative terms - see the precision table
// above. The error is 1.5 mm at this scale.
near('the top solid voxel centre is half a voxel inside', distanceAt(g, 0, 47, 0), -0.5, 0.02);
near('the air voxel above it is half a voxel outside', distanceAt(g, 0, 48, 0), 0.5, 0.02);
near('the bottom of the half-filled slab is also a face', distanceAt(g, 0, 42, 0), -0.5, 0.02);
// Twelve voxels above the surface, so past the 8-voxel clamp: saturated.
near('a voxel beyond the range reads as far air', distanceAt(g, 0, 60, 0), DISTANCE_RANGE, 0.01);
// And the point of widening it: two voxels out is now a REAL distance rather
// than a saturated one. This is the number that stayed frozen at 1 through every
// attempt to widen the range, because the bake had the old clamp inlined.
near('but two voxels out is a real distance, not a saturated one',
     distanceAt(g, 0, 50, 0), 2.5, 0.05);
truthy('the field grades monotonically away from the surface',
  [48, 50, 52, 54, 56].every((vy, i, a) =>
    i === 0 || distanceAt(g, 0, a[i], 0) > distanceAt(g, 0, a[i - 1], 0)));
near('out of bounds reads as open air', distanceAt(g, -1, 0, 0), DISTANCE_RANGE, 1e-12);

section('isOccupied is now the sign of the field');
truthy('solid where the slab is', isOccupied(g, 0, 47, 0) && isOccupied(g, 0, 42, 0));
falsy('air above it', isOccupied(g, 0, 48, 0));
falsy('air below the half-filled part', isOccupied(g, 0, 41, 0));
falsy('out of bounds is not solid', isOccupied(g, -1, 0, 0));

section('the zero-crossing lands exactly on the surface');
// This is the assertion that makes linear filtering correct rather than merely
// tolerable. If it drifts, every shadow edge drifts with it by the same amount.
const faceY = g.origin.y + 48 * VOXEL_METRES;   // top face of the ground slab
near('zero exactly at the block face', sampleDistance(g, 0.5, faceY, 0.5), 0, 1e-6);
truthy('negative just below', sampleDistance(g, 0.5, faceY - VOXEL_METRES * 0.4, 0.5) < 0);
truthy('positive just above', sampleDistance(g, 0.5, faceY + VOXEL_METRES * 0.4, 0.5) > 0);
// Across the surface the field is a true distance: one voxel of travel is one
// unit of distance, which is what lets sphere tracing step by the value it reads.
// The 2.3% shortfall is the byte, not a bug, and it is read AT +-0.5 voxels -
// the coarsest point of the squared mapping in relative terms, where one step is
// about 0.03 of a voxel, or 4 mm on the ground. Sampled closer to the face, where
// occlusion is actually decided, the same measurement is near exact.
near('the gradient is one per voxel across the surface',
     sampleDistance(g, 0.5, faceY + VOXEL_METRES * 0.5, 0.5)
     - sampleDistance(g, 0.5, faceY - VOXEL_METRES * 0.5, 0.5), 1, 0.04);
// Sampling CLOSER to the face does not tighten that number, and it is worth
// knowing why: the field is trilinear between voxel centres, so its gradient is
// constant across the cell and is fixed by the two stored centre values whatever
// point inside the cell you read. The centres sit at +-0.5 voxels, so the
// encoding's resolution there is what caps the gradient - permanently, for any
// sampling distance.
//
// What the squared mapping actually buys is the ZERO CROSSING's position, which is
// what occlusion is decided by and what a slope's sub-voxel placement will be.
// That is the thing to assert.
{
  // Bisect for where the interpolated field changes sign, and compare with the
  // face the bake was given.
  let lo = faceY - VOXEL_METRES, hi = faceY + VOXEL_METRES;
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2;
    if (sampleDistance(g, 0.5, mid, 0.5) < 0) lo = mid; else hi = mid;
  }
  const crossing = (lo + hi) / 2;
  near('the zero crossing lands on the face to within a hundredth of a voxel',
       (crossing - faceY) / VOXEL_METRES, 0, 0.01);
}
// The small shortfall is the byte, not a bug. What matters is that the gradient
// now SURVIVES away from the face instead of flattening immediately: at +-1 voxel
// of range the gradient one voxel past the face had already collapsed to 0.75,
// which is what made sphere tracing barely better than a DDA. It should now stay
// near one out to several voxels.
near('and the gradient survives two voxels out, where the old clamp killed it',
     sampleDistance(g, 0.5, faceY + VOXEL_METRES * 3, 0.5)
     - sampleDistance(g, 0.5, faceY + VOXEL_METRES * 2, 0.5), 1, 0.06);
inRange('flattening only arrives at the far end of the range',
        sampleDistance(g, 0.5, faceY + VOXEL_METRES * 12, 0.5)
        - sampleDistance(g, 0.5, faceY + VOXEL_METRES * 11, 0.5), 0, 0.5);

section('sphere tracing');
const surface = voxelCentreToWorld(g, 0, 48, 0);   // in air, just above ground
truthy('up from open ground escapes the grid',
       sphereTrace(g, surface, { x: 0, y: 1, z: 0 }, 20).escaped);
truthy('down from open ground hits it',
       sphereTrace(g, surface, { x: 0, y: -1, z: 0 }, 20).hit);
falsy('a zero-length direction is handled',
      sphereTrace(g, surface, { x: 0, y: 0, z: 0 }, 10).hit);
falsy('a ray shorter than the gap cannot reach anything',
      sphereTrace(g, surface, { x: 0, y: -1, z: 0 }, 0.01).hit);

// The wall in the test area is at x=20, z=8..14, y=1..2 - outside chunk (0,0),
// so use the pillar's chunk instead for an occluder test.
section('it agrees with the binary DDA it replaced');
// Two unrelated traversals over the same field. Near-total agreement is worth
// more than either being carefully reviewed; the residue is rays that clip the
// outermost voxel column, where the DDA tests the cell it is in before checking
// bounds and the trace tests the point after stepping. Both defensible.
let agree = 0, disagree = 0, exhausted = 0;
let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
const N = 2000;
for (let i = 0; i < N; i++) {
  const o = { x: rnd() * 16, y: rnd() * 4, z: rnd() * 16 };
  const d = { x: rnd() * 2 - 1, y: rnd() * 2 - 1, z: rnd() * 2 - 1 };
  const t = sphereTrace(g, o, d, 12);
  if (t.exhausted) exhausted++;
  if (marchOccupancy(g, o, d, 12, false).hit === t.hit) agree++; else disagree++;
}
inRange('at least 99% of random rays agree', agree / N, 0.99, 1);
ok('and no ray ever runs out of steps', exhausted, 0);
note(`${agree}/${N} agree with the DDA, ${disagree} differ (grid-boundary rays)`);

section('step efficiency');
// The reason the doc asks for sphere tracing. A long diagonal is where the DDA
// is worst: it stops at every axis boundary it crosses, ~1.7 per voxel of
// travel, while the trace covers a clamped voxel per step in open air.
const from = { x: 1, y: 2.5, z: 1 };
const dir = { x: 1, y: 0.15, z: 1 };
const trace = sphereTrace(g, from, dir, 16);
const dda = marchOccupancy(g, from, dir, 16, false);
ok('both reach the same verdict on a long diagonal', trace.hit, dda.hit);
truthy('and the trace uses no more steps than the DDA', trace.steps <= dda.steps);
note(`long diagonal: sphere trace ${trace.steps} steps, DDA ${dda.steps}`);

section('re-origining rebuilds the field exactly');
// The bake clears only the ranges it wrote last time, expanded by the field's
// one-voxel band. Getting that expansion wrong leaves a stale rim that reads as
// a phantom occluder, so an in-place rebuild must be byte-identical to a fresh one.
const reused = createBoxGrid(1, 1);
createBoxGrid(0, 0, reused);
const fresh = createBoxGrid(0, 0);
ok('same occupied count', reused.occupiedCount, fresh.occupiedCount);
truthy('and byte-for-byte identical, with no stale band left behind',
       reused.data.every((v, i) => v === fresh.data[i]));

section('a re-origin that slides is a scroll, not a rebake');
// Compared in LOGICAL coordinates: the field is a ring in x and z, so a scrolled
// grid holds the same voxels as a fresh bake at a different physical offset.
const sameLogical = (a, b) => {
  for (let vz = 0; vz < GRID_DIM; vz++)
    for (let vy = 0; vy < GRID_DIM; vy++)
      for (let vx = 0; vx < GRID_DIM; vx++) {
        // Both channels: opaque (R) and glass (G).
        const ia = gridIndex(a, vx, vy, vz) * 2, ib = gridIndex(b, vx, vy, vz) * 2;
        if (a.data[ia] !== b.data[ib] || a.data[ia + 1] !== b.data[ib + 1]) return false;
      }
  return true;
};
// The hitch this removes: a full bake is 18.8 ms at C0, which at 240 Hz is four
// or five dropped frames on every block stepped. Sliding the overlap into place
// and baking only the strip that scrolled in is about a twelfth of that.
//
// The whole safety argument for it is this assertion. A scroll is only allowed
// to be faster, never different - if it drifts, the symptom is a stale rim of
// distances reading as an occluder that is not there, which is very hard to see
// and very easy to blame on the shadow code. Byte-for-byte or it is wrong.
for (const [dx, dz] of [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-2, 3], [5, -4]]) {
  const slid = createBoxGridAt(10, 10);
  createBoxGridAt(10 + dx, 10 + dz, slid);
  const baked = createBoxGridAt(10 + dx, 10 + dz);
  truthy(`scrolling by (${dx}, ${dz}) matches a fresh bake byte for byte`,
         sameLogical(slid, baked));
  ok(`  and keeps the occupancy count exact`, slid.occupiedCount, baked.occupiedCount);
}
// The coarse level scrolls by its own voxel stride, not C0's - a level that
// slid by the wrong number of voxels would look almost right, which is the
// worst way for this to fail.
{
  const slid = createBoxGridAt(10, 10, null, 1);
  createBoxGridAt(12, 14, slid, 1);
  const baked = createBoxGridAt(12, 14, null, 1);
  truthy('C1 scrolls by its own voxel size',
         sameLogical(slid, baked));
}
// C2 has 3 voxels per block, so half a block is 1.5 voxels. filledBlocks must
// still be whole voxel coordinates - fractional starts had the boxGrid overlay
// indexing the grid between voxels and dropping most of C2's surface.
{
  const c2 = createBoxGridAt(0, 0, null, 2);
  truthy('C2 block ranges start on whole voxels', c2.filledBlocks.every(Number.isInteger));
}

// The field worker's hand-off: a mirror scrolls and bakes, only the strips come
// back, and the main grid applies them. Must land byte-identical (logically) to
// a fresh bake - including after the mirror is reset to a DIFFERENT ring, which
// is what a main-thread full bake does to it.
{
  const main = createBoxGridAt(10, 10);
  let mirror = createBoxGridAt(10, 10);
  let ok1 = true;
  const moves = [[11, 10], [11, 11], [12, 12], 'reset', [13, 12], [12, 11], [60, 60], [61, 60]];
  let at = [12, 12];
  for (const mv of moves) {
    if (mv === 'reset') {
      createBoxGridAt(at[0] + 3, at[1], main);        // main full-bakes elsewhere...
      createBoxGridAt(at[0], at[1], main);            // ...and back, keeping its ring
      mirror = createBoxGridAt(at[0], at[1]);          // mirror resets to ring 0
      continue;
    }
    at = mv;
    applyHandoff(main, scrollForHandoff(mirror, mv[0], mv[1]));
    if (!sameLogical(main, createBoxGridAt(mv[0], mv[1]))) ok1 = false;
  }
  truthy('worker hand-off matches a fresh bake, across rings and a jump', ok1);
}
// A jump too far to share anything has to fall back to a full bake rather than
// scrolling in garbage from the far side of the buffer.
{
  const jumped = createBoxGridAt(10, 10);
  createBoxGridAt(60, 60, jumped);
  const baked = createBoxGridAt(60, 60);
  truthy('and a jump past the footprint rebakes instead',
         sameLogical(jumped, baked));
}
