import { file, section, ok, near, truthy, falsy, inRange, note } from './lib/harness.mjs';
import { createTestArea, BLOCK_METRES } from '../js/world.js';
import {
  createBoxGrid, TEXELS_PER_BLOCK, GRID_DIM, VOXEL_METRES,
  DISTANCE_RANGE, FAR_BYTE, encodeDistance, decodeDistance,
  distanceAt, isOccupied, sampleDistance, boxDistance,
  sphereTrace, marchOccupancy, voxelCentreToWorld
} from '../js/boxgrid.js';

file('sdf.test.mjs - signed distance field and sphere tracing');

section('the byte encoding');
// The whole point of this mapping: an R8 texture samples normalised to [0,1],
// and the field spans [-1,+1] voxels, so the shader decode is sample*2-1 with
// nothing to divide by 255. A silent normalisation mistake here is exactly what
// broke the binary version, where an occupied voxel arrived as 1/255.
ok('the far end of the range is the top of the byte', encodeDistance(1), FAR_BYTE);
ok('and the deep end is the bottom', encodeDistance(-1), 0);
near('a surface sits mid-range', encodeDistance(0), 127.5, 1);
ok('beyond the range clamps rather than wrapping', encodeDistance(9), FAR_BYTE);
ok('and clamps on the solid side too', encodeDistance(-9), 0);
truthy('the encoding is monotonic, which is why the bake can min raw bytes',
  encodeDistance(-0.7) < encodeDistance(-0.2) && encodeDistance(-0.2) < encodeDistance(0.6));
inRange('a round trip is within half a quantisation step',
  Math.abs(decodeDistance(encodeDistance(0.25)) - 0.25), 0, DISTANCE_RANGE / 255);
near('the shader decode agrees with the CPU one',
  encodeDistance(-0.5) / 255 * 2 - 1, -0.5, 1 / 255);

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

ok('still one byte per voxel, 2.85 MB', (g.data.length / 1048576).toFixed(2), '2.85');
// Ground only: createTestArea lays the slab, the wall and pillar come from
// addTestElevation and sit outside chunk (0,0) anyway.
ok('unsupported ground fills its top half only', g.occupiedCount,
   144 * (TEXELS_PER_BLOCK / 2) * TEXELS_PER_BLOCK * TEXELS_PER_BLOCK);

// Voxel 47 is the top voxel of the ground slab, 48 the air above it.
near('the top solid voxel centre is half a voxel inside', distanceAt(g, 0, 47, 0), -0.5, 0.01);
near('the air voxel above it is half a voxel outside', distanceAt(g, 0, 48, 0), 0.5, 0.01);
near('the bottom of the half-filled slab is also a face', distanceAt(g, 0, 42, 0), -0.5, 0.01);
near('a voxel two clear of anything reads as far air', distanceAt(g, 0, 60, 0), 1, 0.01);
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
near('the gradient is one per voxel across the surface',
     sampleDistance(g, 0.5, faceY + VOXEL_METRES * 0.5, 0.5)
     - sampleDistance(g, 0.5, faceY - VOXEL_METRES * 0.5, 0.5), 1, 0.01);
// The 0.4% shortfall is the byte, not a bug: +-0.5 quantises to +-0.498 because
// one step of the encoding is 1/127.5 of a voxel, or 0.1 mm on the ground.
// Further out the +-1 clamp takes over, and it does so immediately - one voxel
// past the face the gradient has already dropped to 0.75. This is the honest
// cost of the byte format and the reason sphere tracing here is a modest win
// rather than a large one: in open air every step is capped at a single voxel.
inRange('but the clamp flattens it beyond one voxel',
        sampleDistance(g, 0.5, faceY + VOXEL_METRES * 2, 0.5)
        - sampleDistance(g, 0.5, faceY + VOXEL_METRES, 0.5), 0, 0.5);

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
