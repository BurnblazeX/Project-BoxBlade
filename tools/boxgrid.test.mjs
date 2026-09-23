import { file, section, ok, near, truthy, falsy, inRange, note } from './lib/harness.mjs';
import { createTestArea, World, getVoxelKey, BLOCK_METRES, CHUNK_SIZE } from '../js/world.js';
import { CASCADE_COUNT, cascadeVoxelsPerBlock, cascadeVoxelMetres, cascadeBlocks,
         cascadeExtentMetres, cascadeRangeVoxels, DISTANCE_RANGE_METRES,
         createBoxGrid, createBoxGridAt, gridOriginFor, GRID_DIM, VOXEL_METRES, TEXELS_PER_BLOCK, worldToVoxel,
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
ok('explore centres on the player when no sun is given',
   JSON.stringify(gridOriginFor('explore', { x: 20, z: 10 })), '{"x":14,"z":4}');
// --- Leaning the footprint up-sun ---
//
// A shadow ray travels TOWARD the sun, so every occluder that can darken ground
// the player sees lies up-sun of it. Centred, the 12-block window reaches 9 m
// each way and spends half of itself down-sun, where nothing can cast onto
// anything visible - which is why a long shadow got cut off by a diagonal line
// once the player walked away from its caster.
{
  const P = { x: 20, z: 10 };
  const centred = gridOriginFor('explore', P);

  // The lean follows AZIMUTH only. Elevation sets how long a shadow is, not
  // which way the footprint should reach.
  for (const [label, dir, dx, dz] of [
    ['+X', { x: 1, y: 0.4, z: 0 }, 2, 0],
    ['-X', { x: -1, y: 0.4, z: 0 }, -2, 0],
    ['+Z', { x: 0, y: 0.4, z: 1 }, 0, 2],
    ['-Z', { x: 0, y: 0.4, z: -1 }, 0, -2]
  ]) {
    const o = gridOriginFor('explore', P, dir, 2);
    ok(`a sun toward ${label} leans the footprint that way`,
       JSON.stringify(o), JSON.stringify({ x: centred.x + dx, z: centred.z + dz }));
  }
  // Elevation must not change it: these differ only in height.
  ok('the lean ignores the sun elevation',
     JSON.stringify(gridOriginFor('explore', P, { x: 1, y: 0.1, z: 1 }, 3)),
     JSON.stringify(gridOriginFor('explore', P, { x: 1, y: 9.0, z: 1 }, 3)));
  // A sun straight overhead has no azimuth to lean toward, and normalising it
  // would divide by zero.
  ok('a sun directly overhead falls back to centred',
     JSON.stringify(gridOriginFor('explore', P, { x: 0, y: 1, z: 0 }, 3)),
     JSON.stringify(centred));
  ok('and a zero bias is the centred footprint too',
     JSON.stringify(gridOriginFor('explore', P, { x: 1, y: 0.4, z: 1 }, 0)),
     JSON.stringify(centred));
  ok('battle stays chunk-aligned whatever the sun does',
     JSON.stringify(gridOriginFor('battle', P, { x: 1, y: 0.4, z: 1 }, 4)),
     JSON.stringify(gridOriginFor('battle', P)));

  // THE POINT OF THE WHOLE THING: a caster up-sun that a centred footprint would
  // have dropped is still inside the leaned one. This is the artifact, as a test.
  {
    const sun = { x: 1, y: Math.tan(20 * Math.PI / 180) * Math.SQRT1_2, z: 1 };
    const inside = (o, bx, bz) =>
      bx >= o.x && bx < o.x + CHUNK_SIZE && bz >= o.z && bz < o.z + CHUNK_SIZE;
    // Seven blocks up-sun along both axes - past the centred window's six.
    // Bias 4 rather than 2, because the lean is a DISTANCE along the azimuth and
    // not a per-axis amount: a diagonal sun spends it as 0.707 of the bias on
    // each axis, so 4 buys 3 blocks each way and 2 would buy only 1. Worth
    // knowing when picking a value with bxb.gridbias() - a diagonal sun needs
    // about 1.4x what an axis-aligned one does for the same reach per axis.
    const casterX = P.x + 7, casterZ = P.z + 7;
    const leaned = gridOriginFor('explore', P, sun, 4);
    falsy('a caster 7 blocks up-sun falls outside a centred footprint',
      inside(centred, casterX, casterZ));
    truthy('and is inside the leaned one, which is the whole point',
      inside(leaned, casterX, casterZ));
    // And the cost, stated rather than hidden: the same distance down-sun is now
    // outside. That ground is still visible, and keeps its last shaded value.
    truthy('while the same distance down-sun is what it gives up',
      inside(centred, P.x - 6, P.z - 6) && !inside(leaned, P.x - 6, P.z - 6));

    // The diagonal split, asserted so the 1.4x is not a surprise later.
    const diag = gridOriginFor('explore', P, { x: 1, y: 0.4, z: 1 }, 4);
    const axis = gridOriginFor('explore', P, { x: 1, y: 0.4, z: 0 }, 4);
    ok('an axis-aligned sun spends the whole bias on one axis',
       axis.x - centred.x, 4);
    ok('a diagonal sun spends 0.707 of it on each', diag.x - centred.x, 3);
    ok('and symmetrically on the other', diag.z - centred.z, 3);
  }
}

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


section('cascades');
// C1 exists because a shadow ray travels toward the sun, and a receiver at the
// far end of a long shadow ran out of C0 before reaching what casts it. Every
// level is the same 144^3 buffer at a different voxel size, so each step coarser
// doubles the reach for the same 2.99 MB.
ok('there are three levels', CASCADE_COUNT, 3);
ok('C0 is the texel-locked one', cascadeVoxelsPerBlock(0), TEXELS_PER_BLOCK);
ok('and C1 is half resolution', cascadeVoxelsPerBlock(1), TEXELS_PER_BLOCK / 2);
near('C0 voxels are 12.5 cm', cascadeVoxelMetres(0), 0.125, 1e-12);
near('C1 voxels are 25 cm', cascadeVoxelMetres(1), 0.25, 1e-12);
ok('C0 covers 12 blocks', cascadeBlocks(0), 12);
ok('C1 covers 24', cascadeBlocks(1), 24);
near('which is 18 m', cascadeExtentMetres(0), 18, 1e-12);
near('and 36 m', cascadeExtentMetres(1), 36, 1e-12);
// C2: same 144^3 buffer again, 50 cm voxels, 72 m across. Each step coarser
// doubles the reach for the same 2.99 MB, which is the whole argument for a
// cascade - and 72 m is past anything a 12 m march cap can reach, so this is
// the level at which range stops being the limitation.
ok('and C2 is quarter resolution', cascadeVoxelsPerBlock(2), TEXELS_PER_BLOCK / 4);
near('C2 voxels are 50 cm', cascadeVoxelMetres(2), 0.5, 1e-12);
ok('C2 covers 48 blocks', cascadeBlocks(2), 48);
near('which is 72 m', cascadeExtentMetres(2), 72, 1e-12);
// 3 voxels per block still divides evenly, so C2 stays in phase with C0 - the
// next halving would not, which is where this ladder stops.
truthy('C2 is still a whole number of voxels per block',
       Number.isInteger(cascadeVoxelsPerBlock(2)));
truthy('and still in phase with C0',
       cascadeVoxelsPerBlock(0) % cascadeVoxelsPerBlock(2) === 0);
// The clamp is constant in WORLD space at every level, so a ray crossing a seam
// does not change brightness - the cone's penumbra saturates on metres, not
// voxels.
near('C2 clamps at the same metre as C0',
     cascadeRangeVoxels(2) * cascadeVoxelMetres(2),
     cascadeRangeVoxels(0) * cascadeVoxelMetres(0), 1e-12);
// Halving rather than any other ratio, so a C1 voxel is a whole number of C0
// voxels and the two fields stay in phase.
truthy('each level is a whole multiple of the finer one',
  cascadeVoxelsPerBlock(0) % cascadeVoxelsPerBlock(1) === 0);

// THE CLAMP IS CONSTANT IN WORLD SPACE. Holding it at 8 voxels everywhere would
// have made C1 cost MORE to bake than C0 - four times the blocks at (6+16)^3
// each - and would have doubled the cone trace's saturation distance at C1,
// putting a visible brightness step at the seam.
for (let l = 0; l < CASCADE_COUNT; l++) {
  near(`C${l}'s range is ${DISTANCE_RANGE_METRES} m in world space`,
       cascadeRangeVoxels(l) * cascadeVoxelMetres(l), DISTANCE_RANGE_METRES, 1e-12);
}
ok('which is 8 voxels at C0', cascadeRangeVoxels(0), 8);
ok('and 4 at C1', cascadeRangeVoxels(1), 4);

// CONTAINMENT, which is the property the whole fall-through rests on. The trace
// marches C0, and when the ray leaves it, continues in C1 from the same t. If C0
// were not wholly inside C1 there would be a gap at the seam that a ray crosses
// blind, and occluders sitting in it would cast nothing.
{
  const contains = (outer, lo, inner, li) => {
    const os = cascadeBlocks(lo), is = cascadeBlocks(li);
    return outer.x <= inner.x && outer.z <= inner.z &&
           outer.x + os >= inner.x + is && outer.z + os >= inner.z + is;
  };
  let worst = null;
  for (const mode of ['explore', 'battle']) {
    for (let px = -13; px <= 13; px++) {
      for (let pz = -13; pz <= 13; pz++) {
        for (const [sx, sz] of [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1]]) {
          const sun = { x: sx, y: 0.4, z: sz };
          const p = { x: px, z: pz };
          const c0 = gridOriginFor(mode, p, sun, 2, 0);
          const c1 = gridOriginFor(mode, p, sun, 2, 1);
          if (!contains(c1, 1, c0, 0)) worst = { mode, px, pz, sx, sz, c0, c1 };
        }
      }
    }
  }
  falsy('C1 wholly contains C0 at every position, sun and mode' +
        (worst ? ` - failed at ${JSON.stringify(worst)}` : ''), worst);
}

// C1 snaps to a coarser stride so it does not rebuild as often as C0. Without
// this a cascade costs its full bake on every block stepped, which is most of
// the reason to have one gone.
{
  const sun = { x: 1, y: 0.4, z: 0 };
  // Re-origins across n single-block steps - counted as changes, so where the
  // walk starts relative to a stride boundary does not matter.
  const moves = (level, n) => {
    let count = 0, prev = null;
    for (let i = 0; i <= n; i++) {
      const o = gridOriginFor('explore', { x: i, z: 0 }, sun, 2, level);
      const key = o.x + ',' + o.z;
      if (prev !== null && key !== prev) count++;
      prev = key;
    }
    return count;
  };
  ok('C0 re-origins on every block stepped', moves(0, 16), 16);
  ok('C1 only on every second one', moves(1, 16), 8);
}

// Centred on the player: rounded to the stride, not floored toward -x/-z.
{
  for (let level = 0; level < CASCADE_COUNT; level++) {
    const half = cascadeBlocks(level) / 2;
    let worst = 0;
    for (let px = 0; px < 16; px++) {
      const o = gridOriginFor('explore', { x: px, z: px }, null, 0, level);
      worst = Math.max(worst, Math.abs(o.x + half - px), Math.abs(o.z + half - px));
    }
    truthy(`C${level} stays within half a stride of the player`, worst <= (1 << level) / 2);
  }
}

// And the field itself bakes sanely at the coarse level - same world, half the
// voxels per block, so roughly an eighth the occupied voxel count.
{
  createTestArea(12, 12);
  const g0 = createBoxGridAt(0, 0, null, 0);
  const g1 = createBoxGridAt(0, 0, null, 1);
  ok('C0 records its level', g0.level, 0);
  ok('C1 records its level', g1.level, 1);
  near('C1 voxels are twice the size', g1.voxelSize, g0.voxelSize * 2, 1e-12);
  truthy('both fields found geometry', g0.occupiedCount > 0 && g1.occupiedCount > 0);
  // Eight times fewer voxels per block, but C1's footprint holds four times the
  // blocks - and this test area only fills C0's, so the ratio lands near 1/8.
  inRange('C1 holds about an eighth the occupied voxels over the same blocks',
          g1.occupiedCount / g0.occupiedCount, 0.1, 0.16);
}
