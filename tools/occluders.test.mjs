import { file, section, ok, truthy, falsy, note } from './lib/harness.mjs';
import { World, createTestArea, getVoxelKey } from '../js/world.js';
import {
  createBoxGridAt, populateDistanceField, isOccupied, sphereTrace,
  worldToVoxel, voxelIndex, gridIndex, GRID_DIM, VOXEL_METRES
} from '../js/boxgrid.js';
import {
  createOccluder, placeOccluder, applyOccluders,
  PROXY_WIDTH, PROXY_HEIGHT
} from '../js/occluders.js';

file('occluders.test.mjs - dynamic occluders in the distance field');

// A flat 12x12 ground slab, so anything solid above y=0 is the proxy and nothing
// else. Deliberately no walls: a test that cannot tell a character's imprint
// from a pillar's is not testing the character.
World.clear();
createTestArea(12, 12);

function freshGrid() {
  return createBoxGridAt(0, 0, null, 0);
}

// A byte-for-byte snapshot. The strongest available statement about the restore
// path is not "the shadow went away" but "the field is indistinguishable from
// one that never had an occluder in it".
function snapshot(grid) {
  return Uint8Array.from(grid.data);
}

function differingBytes(a, b) {
  let n = 0;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++;
  return n;
}

// Where a proxy standing at the centre of the slab should put solid voxels.
// Chest height rather than the centre of the box, so the test is reading the
// body and not an edge case at the boundary.
const STAND = { x: 9, z: 9 };          // metres, mid-slab
const CHEST = 1.0;                      // metres above the ground

function chestVoxel(grid, x = STAND.x, z = STAND.z) {
  return worldToVoxel(grid, x, CHEST, z);
}

section('a proxy writes itself into the field');

const grid = freshGrid();
const clean = snapshot(grid);
const at = chestVoxel(grid);

falsy('nothing solid at chest height before', isOccupied(grid, at.vx, at.vy, at.vz));

const bob = placeOccluder(createOccluder('bob'), STAND.x, 0.75, STAND.z);
// The slab's top surface is at y = 0.75 m (a 1.5 m block centred on 0), so the
// feet go there - not at 0, which would bury the proxy to the knees.
truthy('applying a new occluder reports a change', applyOccluders(grid, [bob]));
truthy('solid at chest height after', isOccupied(grid, at.vx, at.vy, at.vz));

section('the proxy is the size it was asked for');

// Walk out along x from the centre and find where solid stops. Half-width is
// 0.3 m = 2.4 voxels, so the last solid voxel centre sits within that of the
// proxy's own centre.
let solidRun = 0;
for (let i = 0; i < 12; i++) {
  if (!isOccupied(grid, at.vx + i, at.vy, at.vz)) break;
  solidRun++;
}
note(`${solidRun} solid voxels from centre outward, half-width is ` +
     `${(PROXY_WIDTH / 2 / VOXEL_METRES).toFixed(1)} voxels`);
ok('half-width reads as 2 or 3 voxels', solidRun >= 2 && solidRun <= 3, true);

// Vertically: feet at 0.75 m, head at 0.75 + 1.7 = 2.45 m. Sample above the head
// and below the feet.
const above = worldToVoxel(grid, STAND.x, 0.75 + PROXY_HEIGHT + 0.3, STAND.z);
falsy('nothing solid above the head', isOccupied(grid, above.vx, above.vy, above.vz));

section('it actually blocks a ray');

// Straight down onto the proxy from above - the shadow test in miniature.
const from = { x: STAND.x, y: 0.75 + PROXY_HEIGHT + 1.5, z: STAND.z };
const down = sphereTrace(grid, from, { x: 0, y: -1, z: 0 }, 4);
truthy('a ray from above is blocked by the proxy', down.hit);
// And one a metre to the side is not, so the block above is the proxy rather
// than the ground the ray would eventually reach anyway.
const beside = sphereTrace(grid, { x: STAND.x + 1.2, y: from.y, z: STAND.z },
                           { x: 0, y: -1, z: 0 }, 1.2);
falsy('a ray beside it is not', beside.hit);

section('a still proxy costs nothing');

falsy('re-applying an unmoved occluder reports no change',
      applyOccluders(grid, [bob]));

// Sub-voxel movement must not either - this is the whole reason the box is
// snapped, and the reason a walking character does not re-bake every frame.
placeOccluder(bob, STAND.x + VOXEL_METRES * 0.2, 0.75, STAND.z);
falsy('a fifth-of-a-voxel step reports no change', applyOccluders(grid, [bob]));

placeOccluder(bob, STAND.x + VOXEL_METRES * 1.1, 0.75, STAND.z);
truthy('a full voxel step does report a change', applyOccluders(grid, [bob]));

section('removing it restores the field exactly');

// Back to where it started, then away entirely.
placeOccluder(bob, STAND.x, 0.75, STAND.z);
applyOccluders(grid, [bob]);
truthy('removal reports a change', applyOccluders(grid, []));

const after = snapshot(grid);
const diff = differingBytes(clean, after);
note(`${diff} bytes differ from a field that never had an occluder`);
ok('field is byte-for-byte what it was before the occluder', diff, 0);
falsy('nothing solid at chest height again', isOccupied(grid, at.vx, at.vy, at.vz));

section('the band is cleared, not just the box');

// The failure this guards against is subtle and does not look like a leftover
// shadow: a shell of short distances around where the proxy stood makes rays
// creep through open air. Byte equality above already covers it, but name the
// shell specifically so a regression reports itself as "the band was not
// cleared" rather than as one of 2.9 million bytes.
//
// Against the CLEAN snapshot, not against FAR_BYTE. This voxel sits 0.25 m above
// the slab, which is two voxels - well inside the eight-voxel band - so the
// ground itself gives it a graded distance and "far from anything" was never
// what it should read. What has to hold is that it reads what it would have read
// had no proxy ever stood there.
const shell = worldToVoxel(grid, STAND.x + 0.5, CHEST, STAND.z);
const shellIdx = gridIndex(grid, shell.vx, shell.vy, shell.vz);
note(`shell voxel holds ${grid.data[shellIdx]}, clean field holds ${clean[shellIdx]}`);
ok('a voxel in the old proxy band is back to its static value',
   grid.data[shellIdx], clean[shellIdx]);

section('two proxies do not erase each other');

const a = placeOccluder(createOccluder('a'), STAND.x, 0.75, STAND.z);
const b = placeOccluder(createOccluder('b'), STAND.x + 0.9, 0.75, STAND.z);
applyOccluders(grid, [a, b]);

const aAt = chestVoxel(grid, STAND.x, STAND.z);
const bAt = chestVoxel(grid, STAND.x + 0.9, STAND.z);
truthy('a is solid', isOccupied(grid, aAt.vx, aAt.vy, aAt.vz));
truthy('b is solid', isOccupied(grid, bAt.vx, bAt.vy, bAt.vz));

// Move a, close enough that its restore rect covers b. b did not move, so only
// the re-min over the whole rect set puts it back - this is the case that fails
// if the rewrite loop is narrowed to dirty occluders only.
placeOccluder(a, STAND.x - 0.5, 0.75, STAND.z);
truthy('moving a reports a change', applyOccluders(grid, [a, b]));
truthy('b survives a moving through its band',
       isOccupied(grid, bAt.vx, bAt.vy, bAt.vz));

const aMoved = chestVoxel(grid, STAND.x - 0.5, STAND.z);
truthy('a is solid at its new position', isOccupied(grid, aMoved.vx, aMoved.vy, aMoved.vz));
// The gap between where a used to be and where b still is. a moved in -x, so
// its vacated ground is on the +x side of its old centre: a's old right edge is
// at 9.3 m and b's left edge at 9.6 m, which puts 9.5 m in clear air between
// them. Checking -x would land on a's NEW body and prove nothing.
const gap = chestVoxel(grid, STAND.x + 0.5, STAND.z);
falsy('the ground a vacated is clear, and b is not touching it',
      isOccupied(grid, gap.vx, gap.vy, gap.vz));

applyOccluders(grid, []);
ok('both removed restores the field exactly', differingBytes(clean, snapshot(grid)), 0);

section('a re-origin carries proxies with it');

// The two ways a scroll can go wrong, and neither shows up without moving the
// grid: the stored box is in grid coordinates and goes stale, or the strip bake
// wipes the imprint of a proxy standing in what slid in.
const g2 = createBoxGridAt(0, 0, null, 0);
const walker = placeOccluder(createOccluder('w'), 9, 0.75, 9);
applyOccluders(g2, [walker]);
const before = chestVoxel(g2, 9, 9);
truthy('solid before the scroll', isOccupied(g2, before.vx, before.vy, before.vz));

// Step the footprint one block. The proxy does not move in the world.
createBoxGridAt(1, 0, g2, 0);
const moved = chestVoxel(g2, 9, 9);
falsy('a stationary proxy needs no rewrite after a scroll',
      applyOccluders(g2, [walker]));
truthy('and is still solid at the same WORLD position',
       isOccupied(g2, moved.vx, moved.vy, moved.vz));

// Now a proxy standing in the strip that is about to scroll in. Its imprint
// cannot exist yet, so the strip bake cannot destroy it - place it, scroll, and
// check the grid notices it has to be written into the freshly baked strip.
const edge = createBoxGridAt(0, 0, null, 0);
// A world x near the far edge of the footprint, so one block of scroll brings
// fresh ground under it.
const ex = 16.5;
const stander = placeOccluder(createOccluder('s'), ex, 0.75, 9);
applyOccluders(edge, [stander]);
createBoxGridAt(1, 0, edge, 0);
applyOccluders(edge, [stander]);
const ev = chestVoxel(edge, ex, 9);
truthy('a proxy over the scrolled-in strip is still solid',
       isOccupied(edge, ev.vx, ev.vy, ev.vz));

section('a full re-bake drops every imprint');

const g3 = createBoxGridAt(0, 0, null, 0);
applyOccluders(g3, [placeOccluder(createOccluder('x'), 9, 0.75, 9)]);
// A jump too far to slide is a full bake, which clears everything including the
// proxies. The state has to be dropped with it or the next diff thinks they are
// already written.
createBoxGridAt(60, 60, g3, 0);
ok('occluder state is cleared by a full bake', g3.occluderBoxes, null);
