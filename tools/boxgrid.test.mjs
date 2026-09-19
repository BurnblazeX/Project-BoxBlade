import { file, section, ok, near, truthy, falsy, inRange, note } from './lib/harness.mjs';
import { createTestArea, World, getVoxelKey, BLOCK_METRES } from '../js/world.js';
import { createBoxGrid, createBoxGridAt, gridOriginFor, GRID_DIM, VOXEL_METRES, TEXELS_PER_BLOCK, worldToVoxel,
         voxelCentreToWorld, isOccupied, inBounds, marchOccupancy, isInShadow } from '../js/boxgrid.js';

file('boxgrid.test.mjs - occupancy grid and DDA traversal');

const solid = () => ({ solid: true, walkable: true, materialId: null, occupant: null, triggerId: null });
createTestArea(36, 36);
// A wall inside chunk (0,0) so the shadow tests have an occluder nearby.
for (let y = 1; y <= 2; y++) World.set(getVoxelKey(6, y, 6), solid());
const g = createBoxGrid(0, 0);

section('frozen constants - the 1:1 texel/voxel lock');
ok('12 texels per block', TEXELS_PER_BLOCK, 12);
ok('grid is 144 cubed', GRID_DIM, 144);
near('voxel is 12.5cm', VOXEL_METRES, 0.125);
ok('C0 covers exactly one chunk', GRID_DIM / TEXELS_PER_BLOCK, 12);
ok('occupancy is 2.85 MB', (g.data.length / 1048576).toFixed(2), '2.85');

section('voxelisation');
// Ground blocks have nothing beneath them, so only their top half fills:
// 12*12*6 voxels each instead of 12^3. Wall blocks sit on ground, so they are
// supported and fill completely.
const HALF = TEXELS_PER_BLOCK * TEXELS_PER_BLOCK * (TEXELS_PER_BLOCK / 2);
const FULL = TEXELS_PER_BLOCK ** 3;
ok('only in-chunk blocks fill (144 half ground + 2 full wall)', g.occupiedCount, 144 * HALF + 2 * FULL);
ok('half a block is exactly half the voxels', HALF * 2, FULL);
truthy('ground fills only its top half, vy 42..47', isOccupied(g, 0, 42, 0) && isOccupied(g, 0, 47, 0));
falsy('lower half of unsupported ground is skipped', isOccupied(g, 0, 36, 0) || isOccupied(g, 0, 41, 0));
truthy('a supported block still fills completely', isOccupied(g, 72, 48, 72) && isOccupied(g, 72, 59, 72));
falsy('air directly above ground', isOccupied(g, 0, 48, 0));
falsy('nothing below ground', isOccupied(g, 0, 35, 0));
truthy('last voxel of the chunk is filled', isOccupied(g, 143, 47, 143));
truthy('wall block (6,2,6) lands at vy 60..71', isOccupied(g, 72, 60, 72) && isOccupied(g, 72, 71, 72));
falsy('out of bounds reads as empty', isOccupied(g, -1, 0, 0));
falsy('inBounds rejects the far edge', inBounds(144, 0, 0));
ok('a neighbouring chunk picks up only its slice of the x=20 wall',
   createBoxGrid(1, 0).occupiedCount, 144 * HALF + 8 * FULL);

section('grid origin follows the mode');
ok('battle snaps to the chunk', JSON.stringify(gridOriginFor('battle', { x: 20, z: 10 })), '{"x":12,"z":0}');
ok('explore centres on the player', JSON.stringify(gridOriginFor('explore', { x: 20, z: 10 })), '{"x":14,"z":4}');
ok('an explore-centred grid holds the same block count as a chunk-aligned one',
   createBoxGridAt(0, 0).occupiedCount, createBoxGrid(0, 0).occupiedCount);
truthy('a player-centred grid puts the player near the middle',
   (() => { const g = createBoxGridAt(...Object.values(gridOriginFor('explore', { x: 20, z: 10 })));
            const v = worldToVoxel(g, 20 * BLOCK_METRES, 0.75, 10 * BLOCK_METRES);
            return v.vx > 60 && v.vx < 84 && v.vz > 60 && v.vz < 84; })());

section('repopulating in place matches a fresh build');
const fresh = createBoxGridAt(12, 0);
const reused = createBoxGridAt(0, 0);
createBoxGridAt(12, 0, reused); // re-origin the existing buffer
ok('same occupied count', reused.occupiedCount, fresh.occupiedCount);
truthy('and byte-identical occupancy - the scoped clear leaves no stale voxels',
       reused.data.every((v, i) => v === fresh.data[i]));

section('coordinate transforms');
const c = worldToVoxel(g, 5 * BLOCK_METRES, 0, 5 * BLOCK_METRES);
ok('block centre lands mid-span', `${c.vx},${c.vy},${c.vz}`, '66,42,66');
const rt = voxelCentreToWorld(g, 66, 42, 66);
ok('voxel -> world -> voxel roundtrip', worldToVoxel(g, rt.x, rt.y, rt.z).vx, 66);
ok('standing height is the first air voxel', worldToVoxel(g, 9, 0.75, 9).vy, 48);

section('DDA traversal');
const surface = { x: 5 * BLOCK_METRES, y: 0.75, z: 5 * BLOCK_METRES };
truthy('up from open ground escapes the grid', marchOccupancy(g, surface, { x: 0, y: 1, z: 0 }, 20).escaped);
truthy('down from open ground hits it', marchOccupancy(g, surface, { x: 0, y: -1, z: 0 }, 20, false).hit);
falsy('origin voxel is skipped, so surfaces do not self-shadow',
      marchOccupancy(g, { x: 9, y: 0, z: 9 }, { x: 1, y: 0, z: 0 }, 0.05).hit);
falsy('a zero-length direction is handled', marchOccupancy(g, surface, { x: 0, y: 0, z: 0 }, 10).hit);

section('shadow geometry');
const beside    = { x: 7 * BLOCK_METRES, y: 0.75, z: 6 * BLOCK_METRES };
const further   = { x: 8 * BLOCK_METRES, y: 0.75, z: 6 * BLOCK_METRES };
const lightSide = { x: 4 * BLOCK_METRES, y: 0.75, z: 6 * BLOCK_METRES };
const lowLight  = { x: 3 * BLOCK_METRES, y: 3 * BLOCK_METRES, z: 6 * BLOCK_METRES };
const overhead  = { x: 7 * BLOCK_METRES, y: 30, z: 6 * BLOCK_METRES };
truthy('wall shadows the tile beside it', isInShadow(g, beside, lowLight));
truthy('the shadow extends further along', isInShadow(g, further, lowLight));
falsy('a tile on the light side is lit', isInShadow(g, lightSide, lowLight));
falsy('the same tile is lit from overhead', isInShadow(g, beside, overhead));

section('DDA vs brute force - the GPU port must match this');
const dir = { x: 1, y: 0.2, z: 1 };
const len = Math.hypot(dir.x, dir.y, dir.z);
const n = { x: dir.x / len, y: dir.y / len, z: dir.z / len };
const origin = { x: 0.1, y: 0.8, z: 0.1 };
const dda = marchOccupancy(g, origin, dir, 18);
let brute = null;
for (let s = 0; s <= 18; s += 0.001) {
  const v = worldToVoxel(g, origin.x + n.x * s, origin.y + n.y * s, origin.z + n.z * s);
  if (isOccupied(g, v.vx, v.vy, v.vz)) { brute = { s, v }; break; }
}
truthy('brute force found an occluder to compare against', !!brute);
ok('both agree on the first occluder voxel',
   `${dda.voxel.vx},${dda.voxel.vy},${dda.voxel.vz}`,
   `${brute.v.vx},${brute.v.vy},${brute.v.vz}`);
near('and on the distance, to within one sample', dda.distance, brute.s, 0.002);
note(`DDA ${dda.steps} steps / ${dda.distance.toFixed(4)}m vs brute force ${brute.s.toFixed(4)}m at 1mm sampling`);
inRange('step count tracks distance, not grid size', dda.steps, 100, 200);
