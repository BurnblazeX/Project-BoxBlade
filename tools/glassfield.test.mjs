import { file, section, ok, near, truthy } from './lib/harness.mjs';
import { createTestArea, World, getVoxelKey } from '../js/world.js';
import { createBoxGridAt, populateDistanceField, distanceAt, glassDistanceAt, gridIndex,
         worldToVoxel, GRID_DIM, FIELD_CHANNELS, FAR_BYTE, TEXELS_PER_BLOCK } from '../js/boxgrid.js';
import { isGlassMaterial } from '../js/materials.js';

file('glassfield.test.mjs - two distances per voxel: opaque (R) and glass (G)');

createTestArea(36, 36);
// No material is glass yet, so the tests mark their own: grid.isGlass.
const isGlass = b => b && b.materialId === 'testGlass';
const bake = (x = 0, z = 0) => {
  const g = createBoxGridAt(x, z);
  g.isGlass = isGlass;
  populateDistanceField(g);
  return g;
};

section('layout');
ok('two channels', FIELD_CHANNELS, 2);
ok('the glass material is glass', isGlassMaterial('glass'), true);
ok('stained and frosted glass are glass', isGlassMaterial('stainedGlass') && isGlassMaterial('frostedGlass'), true);
ok('the others are not', isGlassMaterial('grass') || isGlassMaterial('marble') ||
   isGlassMaterial('ironPlate'), false);

section('no glass: G is empty, R is the field as before');
const plain = bake();
let gFar = true;
for (let i = 1; i < plain.data.length; i += 2) if (plain.data[i] !== FAR_BYTE) { gFar = false; break; }
truthy('every glass byte reads far', gFar);

// A glass block on open ground, and an opaque block on top of it.
const G = { x: 5, y: 1, z: 5 };
World.set(getVoxelKey(G.x, G.y, G.z), { solid: true, materialId: 'testGlass' });
World.set(getVoxelKey(G.x, G.y + 1, G.z), { solid: true, materialId: 'grass' });
const withGlass = bake();
const centreVoxel = (g, b) => {
  const per = TEXELS_PER_BLOCK;
  return { vx: (b.x - g.originBlock.x) * per + per / 2,
           vy: (b.y - g.originBlock.y) * per + per / 2,
           vz: (b.z - g.originBlock.z) * per + per / 2 };
};

section('glass is in G, not in R');
const c = centreVoxel(withGlass, G);
near('G at the glass centre: half a block inside', glassDistanceAt(withGlass, c.vx, c.vy, c.vz),
     -TEXELS_PER_BLOCK / 2 + 0.5, 0.2);
truthy('R at the glass centre is not solid', distanceAt(withGlass, c.vx, c.vy, c.vz) > 0);
near('G just outside a side face: half a voxel',
     glassDistanceAt(withGlass, c.vx - TEXELS_PER_BLOCK / 2 - 1, c.vy, c.vz), 0.5, 0.05);

section('an opaque block on glass keeps its full height in R');
const top = centreVoxel(withGlass, { x: G.x, y: G.y + 1, z: G.z });
truthy('its lower half is solid', distanceAt(withGlass, top.vx, top.vy - 4, top.vz) < 0);

section('R elsewhere is untouched by the glass');
// Far from both new blocks the opaque field matches the plain bake.
let same = true;
for (let vz = 100; vz < 110 && same; vz++)
  for (let vy = 0; vy < GRID_DIM && same; vy++)
    for (let vx = 100; vx < 110; vx++)
      if (distanceAt(withGlass, vx, vy, vz) !== distanceAt(plain, vx, vy, vz)) { same = false; break; }
truthy('opaque distances 10+ m away are identical', same);

section('scrolling carries both channels');
const slid = bake(0, 0);
createBoxGridAt(1, 0, slid);   // the scroll path: strip bake with grid.isGlass
createBoxGridAt(1, 2, slid);
const fresh = bake(1, 2);
let match = true;
for (let vz = 0; vz < GRID_DIM && match; vz++)
  for (let vy = 0; vy < GRID_DIM && match; vy++)
    for (let vx = 0; vx < GRID_DIM; vx++) {
      const a = gridIndex(slid, vx, vy, vz) * 2, b = gridIndex(fresh, vx, vy, vz) * 2;
      if (slid.data[a] !== fresh.data[b] || slid.data[a + 1] !== fresh.data[b + 1]) { match = false; break; }
    }
truthy('a scrolled grid with glass matches a fresh bake, both channels', match);

World.delete(getVoxelKey(G.x, G.y, G.z));
World.delete(getVoxelKey(G.x, G.y + 1, G.z));

section('the field worker bakes glass as glass');
{
  // The worker's copy of the world comes from worldMirrorEntries; baked from
  // it, glass must land in G exactly as it does from the real World. It used
  // to be keys only, which baked glass as rock in every strip the worker made.
  const { worldMirrorEntries, loadWorldMirror } = await import('../js/world.js');
  createTestArea(36, 36);
  // Origin (6, 6): the footprint covers blocks 6-17, so the cube at x 12 is in it.
  const real = createBoxGridAt(6, 6);
  const entries = worldMirrorEntries(World);
  ok('carries the material', entries.find(([k]) => k === getVoxelKey(12, 1, 13))?.[1], 'glass');
  loadWorldMirror(entries);
  const mirrored = createBoxGridAt(6, 6);
  let same = true;
  for (let i = 0; i < real.data.length; i++) if (real.data[i] !== mirrored.data[i]) { same = false; break; }
  truthy('a bake from the mirror matches the real one byte for byte', same);
  const c = centreVoxel(mirrored, { x: 12, y: 1, z: 13 });
  truthy('the cube is inside the footprint', c.vx >= 0 && c.vx < GRID_DIM && c.vz >= 0 && c.vz < GRID_DIM);
  truthy('the glass cube is not opaque in the mirror bake', distanceAt(mirrored, c.vx, c.vy, c.vz) > 0);
  truthy('and is glass there', glassDistanceAt(mirrored, c.vx, c.vy, c.vz) < 0);
  createTestArea(36, 36);   // leave World as the real thing for later files
}

section('through the cube - the CPU mirror of the glass march');
{
  const { marchThroughGlass, VOXEL_METRES } = await import('../js/boxgrid.js');
  const { BLOCK_METRES } = await import('../js/world.js');
  createTestArea(36, 36);
  const g = createBoxGridAt(6, 6);            // covers the cube at (12, 1, 13)
  const cx = 12 * BLOCK_METRES, cy = 1 * BLOCK_METRES, cz = 13 * BLOCK_METRES;
  const half = BLOCK_METRES / 2, lift = 0.75 * VOXEL_METRES;
  // Straight down from just under the top face: through the glass, onto the grass.
  const down = marchThroughGlass(g, { x: cx, y: cy + half - lift, z: cz }, { x: 0, y: -1, z: 0 }, 12);
  ok('down: ends on an opaque surface', down.kind, 'opaque');
  near('down: at the cube floor', down.distance, BLOCK_METRES - lift, 0.1);
  ok('down: the grass faces up', down.normal && down.normal.y, 1);
  // Sideways from just inside the -x face: exits the +x face.
  const across = marchThroughGlass(g, { x: cx - half + lift, y: cy, z: cz }, { x: 1, y: 0, z: 0 }, 12);
  ok('across: leaves the glass', across.kind, 'exit');
  near('across: after a block of glass', across.distance, BLOCK_METRES - lift, 0.1);
  ok('across: the exit face points +x', across.normal && across.normal.x, 1);
  // Diagonally up and out through the top.
  const up = marchThroughGlass(g, { x: cx, y: cy, z: cz }, { x: 0.3, y: 1, z: 0 }, 12);
  ok('up: leaves through the top', up.kind, 'exit');
  ok('up: the exit face points up', up.normal && up.normal.y, 1);
}

section('exact face normals at edges and corners');
{
  const { blockFaceNormal } = await import('../js/boxgrid.js');
  const { BLOCK_METRES: B } = await import('../js/world.js');
  const face = (x, y, z, d, leaving) => {
    const n = blockFaceNormal({ x, y, z }, d, leaving);
    return `${n.x},${n.y},${n.z}`;
  };
  // A block centred at the origin spans -0.75..0.75. Leaving its -x face a
  // hair below the top edge, heading down and left: the -x face, not the top.
  ok('exit near the top edge of a side face', face(-0.75, 0.74, 0, { x: -0.6, y: -0.8, z: 0 }, true), '-1,0,0');
  // Leaving through the top a hair inside the side edge, heading up: the top.
  ok('exit near the side edge of the top face', face(-0.74, 0.75, 0, { x: -0.1, y: 0.99, z: 0 }, true), '0,1,0');
  // Hitting the ground under glass at its foot, heading down: the ground faces up.
  ok('hit on the ground at the glass foot', face(-0.74, -0.75, 0.74, { x: -0.3, y: -0.9, z: 0.3 }, false), '0,1,0');
  // Exactly on a corner, the plane the ray crosses most squarely wins.
  ok('a corner, crossed mostly along z', face(0.75, 0.75, 0.75, { x: 0.1, y: 0.1, z: 0.99 }, true), '0,0,1');
  ok('boundaries repeat every block', face(-0.75 + 3 * B, 0.2, 0.1, { x: -1, y: 0, z: 0 }, true), '-1,0,0');
}

section('the walk through glass, block by block (the shader\'s path, on the CPU)');
{
  const { walkThroughGlass, refractVec } = await import('../js/boxgrid.js');
  const { BLOCK_METRES: B } = await import('../js/world.js');
  createTestArea(36, 36);
  const ior = 1.5;
  const norm = v => { const l = Math.hypot(v.x, v.y, v.z); return { x: v.x / l, y: v.y / l, z: v.z / l }; };
  const into = (view, n) => refractVec(norm(view), n, 1 / ior);
  // Seen from above and in front, as the play camera sees it: the view ray
  // points down and away (-y, -z).
  const view = { x: 0.2, y: -0.8, z: -0.55 };
  const up = { x: 0, y: 1, z: 0 };

  // The window: x 13-15, y 1-2, z 11. Its top face at y = 2.5 blocks.
  const topY = 2.5 * B;
  // A top texel at the BACK edge of the window's top face: the ray heads down
  // and away, meets the back face steeply, and reflects - a light pipe. It
  // must reach the ground, not run out of reflections.
  const back = walkThroughGlass({ x: 14 * B, y: topY, z: 11 * B - 0.7 }, up, into(view, up), ior);
  ok('window, back edge of the top face: ends on the ground', back.kind, 'opaque');
  truthy('after reflecting inside at least once', back.bounces >= 1);
  ok('the ground faces up', back.normal && back.normal.y, 1);

  // A top texel at the FRONT edge: straight down to the ground, no reflection.
  const front = walkThroughGlass({ x: 14 * B, y: topY, z: 11 * B + 0.7 }, up, into(view, up), ior);
  ok('window, front edge of the top face: ends on the ground', front.kind, 'opaque');

  // The left rim of the top face: the ray heads away from a side face, then
  // down - still the ground.
  const left = walkThroughGlass({ x: 13 * B - 0.7, y: topY, z: 11 * B }, up, into(view, up), ior);
  ok('window, left rim of the top face: ends on the ground', left.kind, 'opaque');

  // The window's front face, mid-height: through the slab and out the back.
  const fwd = { x: 0, y: 0, z: 1 };
  const mid = walkThroughGlass({ x: 14 * B, y: 1.5 * B, z: 11 * B + 0.75 }, fwd, into(view, fwd), ior);
  ok('window, front face: leaves through the back', mid.kind, 'exit');
  ok('the exit face points -z', mid.normal && mid.normal.z, -1);
  near('after one block of glass, or a little more', mid.length, B / Math.abs(into(view, fwd).z), 1e-6);

  // The cube's top face, every texel of it: nothing trapped.
  let trapped = 0;
  for (let i = 0; i < 12; i++) for (let j = 0; j < 12; j++) {
    const px = 12 * B - 0.75 + (i + 0.5) * B / 12, pz = 13 * B - 0.75 + (j + 0.5) * B / 12;
    if (walkThroughGlass({ x: px, y: 1.5 * B, z: pz }, up, into(view, up), ior).kind === 'trapped') trapped++;
  }
  ok('the cube top: no texel trapped', trapped, 0);
}

section('shadow rays through glass - metres of glass crossed');
{
  const { glassOnRay } = await import('../js/boxgrid.js');
  const { BLOCK_METRES: B } = await import('../js/world.js');
  createTestArea(36, 36);
  const g = createBoxGridAt(6, 6);
  const ground = 0.5 * B + 0.75 * 0.125;       // just above the grass, lifted as the shader does
  // Straight up from the grass under the cube: through the whole cube.
  const under = glassOnRay(g, { x: 12 * B, y: ground, z: 13 * B }, { x: 0, y: 1, z: 0 }, 12);
  ok('under the cube: not blocked', under.blocked, false);
  near('under the cube: about a block of glass', under.inGlass, B, 0.25);
  // Straight up beside it (between it and the frosted cube at x 10): none.
  const beside = glassOnRay(g, { x: 11 * B, y: ground, z: 13 * B }, { x: 0, y: 1, z: 0 }, 12);
  near('beside the cube: no glass', beside.inGlass, 0, 1e-9);
  // A slanted ray through a corner of the cube crosses less than the middle.
  const corner = glassOnRay(g, { x: 12 * B + 0.6, y: ground, z: 13 * B }, { x: 0.7, y: 1, z: 0 }, 12);
  truthy('through an edge: less glass than straight through the middle',
         corner.inGlass > 0 && corner.inGlass < under.inGlass);
  // Under the window, towards a low sun behind it: two blocks tall, one thick.
  const window = glassOnRay(g, { x: 14 * B, y: ground, z: 11 * B + 0.8 }, { x: 0, y: 0.5, z: -1 }, 12);
  truthy('toward a low sun through the window: glass crossed', window.inGlass > 0.5);
}

section('the overlay sees glass (debug.js draws it from these)');
{
  const { isGlassVoxel, isOccupied, scrollForHandoff, applyHandoff } = await import('../js/boxgrid.js');
  createTestArea(36, 36);
  const g = createBoxGridAt(6, 6);
  truthy('glass blocks are recorded', g.glassBlocks.length >= 3 * 3);
  const c = centreVoxel(g, { x: 12, y: 1, z: 13 });
  truthy('a voxel inside the cube is glass', isGlassVoxel(g, c.vx, c.vy, c.vz));
  truthy('and not opaque', !isOccupied(g, c.vx, c.vy, c.vz));
  // The worker's hand-off carries them to the main grid.
  const mirror = createBoxGridAt(6, 6);
  const main = createBoxGridAt(6, 6);
  const m = scrollForHandoff(mirror, 7, 6);
  applyHandoff(main, m);
  ok('carried through the hand-off', main.glassBlocks.length, mirror.glassBlocks.length);
}

section('light through glass loses what the surfaces reflect, at the real angle');
{
  const { glassSurfaceTransmit } = await import('../js/materials.js');
  const f0 = 10 / 255;
  near('head-on: (1 - 0.04)^2', glassSurfaceTransmit(f0, 1), (1 - f0) ** 2, 1e-12);
  truthy('a low sun (80 degrees off the normal) loses much more',
         glassSurfaceTransmit(f0, Math.cos(80 * Math.PI / 180)) < 0.6);
  near('grazing: nothing gets through', glassSurfaceTransmit(f0, 0), 0, 1e-12);
  // What one face reflects plus what it lets through is all of it.
  const cos = Math.cos(70 * Math.PI / 180);
  const F = f0 + (1 - f0) * (1 - cos) ** 5;
  near('reflected + transmitted at one face = 1', F + Math.sqrt(glassSurfaceTransmit(f0, cos)), 1, 1e-12);
}
