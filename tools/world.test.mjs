import { file, section, ok, near, truthy, falsy } from './lib/harness.mjs';
import { World, getVoxelKey, createTestArea, createEntity, isStandable, isSolid, getColumnTop,
         findPath, getReachableVoxels, enterBattle, exitBattle, addToInventory, currentArena,
         getBlock, stepLevel, surfaceLevel, surfaceHeight, isSolidAt, blockShape, blockBoxes, edgeLevel,
         slopeLevel, groundHeightAt, followGround, blockSpan,
         pathDistance, attackRange, getAbilityModifier,
         BLOCK_METRES, CHUNK_SIZE, CHUNK_HEIGHT, Y_MIN, Y_MAX } from '../js/world.js';

file('world.test.mjs - vertical world, standability, pathfinding');

createTestArea(36, 36);

section('layout constants');
ok('block is 1.5m', BLOCK_METRES, 1.5);
ok('chunk footprint is 12', CHUNK_SIZE, 12);
ok('chunk is 12 levels tall', CHUNK_HEIGHT, 12);
ok('authorable span is -3..8', `${Y_MIN}..${Y_MAX}`, '-3..8');

section('generation - surface only, no sandbox fill');
// + 12: the glass test bed (a cube, a 3 x 2 window, a 2 x 2 stained window,
// a frosted cube). + 7: the half-tile bed (two steps, an arch of two pillars
// and a lintel). + 3: the slope bed (a slope, a half slope, a high one).
// + 3: the stair bed (two stairs and the marble block under the upper one).
ok('ground layer + test features only', World.size, 36 * 36 + 14 + 3 + 16 + 14 + 12 + 12 + 7 + 3 + 3);

section('glass test bed');
ok('the cube is glass', World.get(getVoxelKey(12, 1, 13))?.materialId, 'glass');
ok('the window is glass, 2 tall', World.get(getVoxelKey(14, 2, 11))?.materialId, 'glass');
ok('the stained window', World.get(getVoxelKey(17, 2, 13))?.materialId, 'stainedGlass');
ok('the frosted cube', World.get(getVoxelKey(10, 1, 13))?.materialId, 'frostedGlass');

section('reflection test bed');
ok('floor patch is iron plate', World.get('5,0,14').materialId, 'ironPlate');
ok('its wall is marble', World.get('5,2,17').materialId, 'marble');
ok('ordinary ground is grass', World.get('20,0,20').materialId, 'grass');
truthy('the iron floor is walkable', isStandable(5, 0, 14));
ok('iron plates stand either side', `${World.get('1,1,15').materialId},${World.get('9,2,15').materialId}`,
   'ironPlate,ironPlate');
truthy('nothing generated below ground', [...World.keys()].every(k => Number(k.split(',')[1]) >= 0));

section('standability is derived, not stored');
truthy('open ground is standable', isStandable(8, 0, 8));
falsy('empty air is not', isStandable(8, 5, 8));
falsy('ground under the wall has no headroom', isStandable(20, 0, 10));
truthy('top of the wall is standable', isStandable(20, 2, 10));
truthy('top of the raised platform is standable', isStandable(30, 1, 30));
falsy('under the platform has no headroom', isStandable(30, 0, 30));
ok('column top on open ground', getColumnTop(8, 8), 0);
ok('column top on the wall', getColumnTop(20, 10), 2);
ok('column top on the pillar', getColumnTop(26, 20), 3);
ok('column top off the map', getColumnTop(500, 500), null);

section('occupancy blocks traversal');
truthy('solid where the wall is', isSolid(20, 1, 10));
falsy('not solid above the wall', isSolid(20, 3, 10));

section('pathfinding steps by half a block, no more');
truthy('flat path found', findPath({x:8,y:0,z:8}, {x:14,y:0,z:8}).length > 0);
{
  const up = findPath({x:8,y:0,z:8}, {x:30,y:1,z:30});
  truthy('climbs to the platform', up.length > 0);
  truthy('by the half-tile steps', up.some(n => n.x === 27 && n.y === 1 && (n.z === 29 || n.z === 30)));
  truthy('and back down', findPath({x:30,y:1,z:30}, {x:8,y:0,z:8}).length > 0);
}
ok('cannot climb the pillar (a full block, no stairs)', findPath({x:8,y:0,z:10}, {x:26,y:3,z:20}).length, 0);
ok('cannot enter the wall footprint', findPath({x:8,y:0,z:8}, {x:20,y:0,z:10}).length, 0);
const around = findPath({x:8,y:0,z:11}, {x:25,y:0,z:11});
truthy('routes around the wall', around.length > 0);
falsy('never passes through it', around.some(n => n.x === 20 && n.z >= 8 && n.z <= 14));
truthy('stays on one level', around.every(n => n.y === 0));

section('half tiles');
ok('the steps are half-bottoms', blockShape(getBlock(27, 1, 29)), 'halfBottom');
ok('the lintel is a half-top', blockShape(getBlock(24, 2, 25)), 'halfTop');
ok('a block without a shape is full', blockShape(getBlock(8, 0, 8)), 'full');
ok('a half-bottom stands half a block up', surfaceLevel(27, 1, 29), 1.5);
ok('ground stands at level 1', surfaceLevel(8, 0, 8), 1);
ok('surface height in metres', surfaceHeight(27, 1, 29), 1.5);
ok('ground surface height is unchanged', surfaceHeight(8, 0, 8), 0.75);
truthy('a half-bottom is standable', isStandable(27, 1, 29));
falsy('the ground under a step is not', isStandable(27, 0, 29));
truthy('walk under the lintel: 1.5 m of headroom', isStandable(24, 0, 25));
ok('step up onto a half-bottom', stepLevel(26, 0, 29, 27, 29), 1);
ok('step up again onto the platform', stepLevel(27, 1, 29, 28, 29), 1);
ok('no step onto a full block', stepLevel(19, 0, 10, 20, 10), null);
ok('step down from the platform', stepLevel(28, 1, 29, 27, 29), 1);
truthy('solid in a half-bottom lower half', isSolidAt(27 * 1.5, 1.2, 29 * 1.5));
falsy('air in its upper half', isSolidAt(27 * 1.5, 1.8, 29 * 1.5));
falsy('a point on its top face is outside', isSolidAt(27 * 1.5, 1.5, 29 * 1.5));
truthy('solid in the lintel', isSolidAt(24 * 1.5, 3.4, 25 * 1.5));
falsy('air under it', isStandable(24, 1, 25));
{
  // Headroom over a half-bottom: a half-top above leaves 1.5 m; a full
  // block or another half-bottom leaves 0.75.
  const put = (k, b) => World.set(k, { solid: true, walkable: true, materialId: 'grass', ...b });
  put('40,0,40', { shape: 'halfBottom' });
  put('40,1,40', { shape: 'halfTop' });
  truthy('half-bottom under a half-top: standable', isStandable(40, 0, 40));
  put('40,1,40', { shape: 'halfBottom' });
  falsy('under a half-bottom: not', isStandable(40, 0, 40));
  put('40,1,40', {});
  falsy('under a full block: not', isStandable(40, 0, 40));
  put('41,0,40', {});
  put('41,1,40', { shape: 'halfTop' });
  falsy('a full block under a half-top: not', isStandable(41, 0, 40));
  // Glass is always whole.
  put('42,0,40', { shape: 'halfBottom', materialId: 'glass' });
  ok('glass ignores a shape', blockShape(getBlock(42, 0, 40)), 'full');
  for (const k of ['40,0,40', '40,1,40', '41,0,40', '41,1,40', '42,0,40']) World.delete(k);
}

section('slopes: planes, baked as staircases under them');
{
  const slope = getBlock(29, 1, 27), half = getBlock(33, 1, 29), high = getBlock(32, 1, 29);
  ok('a slope bakes as 11 steps (the first has no height)', blockBoxes(slope).length, 11);
  ok('a half slope as 11', blockBoxes(half).length, 11);
  ok('a high half slope as 12 (on its half block)', blockBoxes(high).length, 12);
  truthy('every step is one texel deep',
    [slope, half, high].every(b => blockBoxes(b).every(q =>
      Math.abs((q[1] - q[0]) * 12 - 1) < 1e-9 || Math.abs((q[5] - q[4]) * 12 - 1) < 1e-9)));
  // THE PLANE IS NEVER INSIDE ITS OWN STEPS: every step's top is at or under
  // the plane across its whole width, and meets it at its downhill edge.
  const under = b => blockBoxes(b).every(q => {
    const corners = [[q[0], q[4]], [q[1], q[4]], [q[0], q[5]], [q[1], q[5]]];
    const levels = corners.map(([x, z]) => slopeLevel(b, x, z));
    return levels.every(l => q[3] <= l + 1e-9) && levels.some(l => Math.abs(q[3] - l) < 1e-9);
  });
  truthy('each step lies under the plane and touches it', [slope, half, high].every(under));
  for (const facing of ['N', 'E', 'S', 'W']) {
    const b = { materialId: 'grass', shape: 'slope', facing };
    truthy(`... facing ${facing} too`, under(b));
  }
  // Facing S climbs toward +z: the step at the +z edge is the tallest.
  const tall = blockBoxes(slope).reduce((a, b) => (b[3] > a[3] ? b : a));
  ok('facing S, the tallest step is at the +z edge', `${tall[4]},${tall[5]},${tall[3]}`, `${11 / 12},1,${11 / 12}`);
  // Drawn from its plane's top: the head is level with the next block's top.
  ok('a slope spans its whole cell', blockSpan(slope).join(','), '0,1');
  ok('a half slope the lower half', blockSpan(half).join(','), '0,0.5');
  ok('a high half slope the whole cell', blockSpan(high).join(','), '0,1');
  ok('the plane at the foot and head', `${slopeLevel(slope, 0.5, 0)},${slopeLevel(slope, 0.5, 1)}`, '0,1');
  ok('a half slope\'s plane', `${slopeLevel(half, 1, 0.5)},${slopeLevel(half, 0, 0.5)}`, '0,0.5');
  ok('a high half slope\'s plane', `${slopeLevel(high, 1, 0.5)},${slopeLevel(high, 0, 0.5)}`, '0.5,1');
  // Edges: the slope joins the ground at its foot and the platform at its head.
  ok('slope foot (toward -z) at ground level', edgeLevel(29, 1, 27, 0, -1), 1);
  ok('slope head (toward +z) at platform level', edgeLevel(29, 1, 27, 0, 1), 2);
  ok('across it, its middle', edgeLevel(29, 1, 27, 1, 0), 1.5);
  ok('stand at its middle', surfaceLevel(29, 1, 27), 1.5);
  ok('onto the slope from its foot', stepLevel(29, 0, 26, 29, 27), 1);
  ok('onto the platform from its head', stepLevel(29, 1, 27, 29, 28), 1);
  ok('down the slope from the platform', stepLevel(29, 1, 28, 29, 27), 1);
  const up = findPath({x:29,y:0,z:22}, {x:29,y:1,z:30});
  truthy('a path up the slope', up.length > 0 && up.some(n => n.x === 29 && n.z === 27 && n.y === 1));
  // The gentle ramp: ground (1) -> half slope (1 .. 1.5) -> high (1.5 .. 2) -> platform (2).
  ok('half slope foot (east) at ground level', edgeLevel(33, 1, 29, 1, 0), 1);
  ok('half slope head meets the high one', edgeLevel(33, 1, 29, -1, 0), edgeLevel(32, 1, 29, 1, 0));
  ok('high half slope head at platform level', edgeLevel(32, 1, 29, -1, 0), 2);
  const ramp = findPath({x:35,y:0,z:29}, {x:30,y:1,z:29});
  truthy('a path up the gentle ramp',
    ramp.some(n => n.x === 33 && n.y === 1) && ramp.some(n => n.x === 32 && n.y === 1));
  truthy('solid under the plane at the head', isSolidAt(29 * 1.5, 2.1, 27 * 1.5 + 0.7));
  falsy('air above the foot', isSolidAt(29 * 1.5, 1.5 + 0.5, 27 * 1.5 - 0.7));
  falsy('a foot a little under the plane is clear with clearance',
    isSolidAt(29 * 1.5, 1.5 + 0.5 * 1.5 - 0.2, 27 * 1.5 + 0.375, 0.3));

  // Standing on it: the ground follows the plane, continuous from the ground
  // up to the platform - no half-block hops.
  const g = z => groundHeightAt({ x: 29, y: 0, z: 26 }, 29 * 1.5, z);
  const B = 1.5;
  ok('on the ground before it', g(26 * B), 0.75);
  near('at its foot, the ground\'s height', g(26.5 * B - 1e-6), 0.75, 1e-5);
  near('just onto it, still the ground\'s height', g(26.5 * B + 1e-6), 0.75, 1e-5);
  near('halfway up, half a block higher', groundHeightAt({ x: 29, y: 1, z: 27 }, 29 * B, 27 * B), 1.5, 1e-9);
  near('at its head, the platform\'s height',
       groundHeightAt({ x: 29, y: 1, z: 27 }, 29 * B, 27.5 * B - 1e-6), 2.25, 1e-5);
  let worst = 0, prev = null;
  for (let i = 0; i <= 300; i++) {
    const z = (26 + 3 * i / 300) * B;
    const gp = { x: 29, y: Math.round(z / B) === 26 ? 0 : 1, z: Math.round(z / B) };
    const h = groundHeightAt(gp, 29 * B, z);
    if (prev !== null) worst = Math.max(worst, Math.abs(h - prev));
    prev = h;
  }
  truthy('ground to slope to platform: no jump bigger than the walk', worst < 0.02);

  // followGround: a ledge is eased, a slope followed.
  const sprite = { position: { x: 26 * B, y: 0.75, z: 29 * B }, userData: {} };
  followGround(sprite, { x: 26, y: 0, z: 29 }, 1 / 60);
  ok('standing on the ground', sprite.position.y, 0.75);
  sprite.position.x = 27 * B;                       // onto the half-bottom step
  followGround(sprite, { x: 27, y: 1, z: 29 }, 1 / 60);
  truthy('a ledge is not snapped', sprite.position.y < 1.4);
  for (let i = 0; i < 60; i++) followGround(sprite, { x: 27, y: 1, z: 29 }, 1 / 60);
  near('and is eased onto it', sprite.position.y, 1.5, 1e-3);
}

section('entity occupancy is respected');
const blocker = createEntity({ id:'blocker', gridPos:{x:9,y:0,z:8} });
World.get('9,0,8').occupant = blocker.id;
ok('occupied tile is not enterable', findPath({x:8,y:0,z:8}, {x:9,y:0,z:8}).length, 0);
World.get('9,0,8').occupant = null;

section('battle arena carves a full chunk');
const { arena, bounds } = enterBattle({x:8,y:0,z:8});
ok('arena covers the chunk footprint', arena.size, 12 * 12);
ok('arena spans the full vertical range', `${bounds.minY}..${bounds.maxY}`, '-3..8');
truthy('reachable set stays on one level',
  [...getReachableVoxels({x:8,y:0,z:8}, 3).keys()].every(k => k.split(',')[1] === '0'));
exitBattle();

section('distance functions stay distinct');
ok('pathDistance is Manhattan', pathDistance({x:0,y:0,z:0}, {x:2,y:0,z:3}), 5);
ok('attackRange is Chebyshev (allows diagonals)', attackRange({x:0,y:0,z:0}, {x:2,y:0,z:3}), 3);

section('entity helpers');
ok('ability modifier curve', `${getAbilityModifier(8)},${getAbilityModifier(10)},${getAbilityModifier(15)}`, '-1,0,2');
const e = createEntity({ id:'inv', gridPos:{x:0,y:0,z:0} });
addToInventory(e, 'gold_coin', 5);
addToInventory(e, 'gold_coin', 3);
addToInventory(e, 'rope', 1);
ok('stacks the same item', e.inventory.find(i => i.itemId === 'gold_coin').quantity, 8);
ok('keeps distinct items separate', e.inventory.length, 2);


section('findPath: identical to the old array search, ties included');
{
  // The search as it was before the heap (2026-09-25): an open set scanned in
  // full every step, first tile with the lowest f wins. The new one must give
  // exactly these paths - the same tiles in the same order.
  const isEnterable = (x, y, z) => isStandable(x, y, z) && !getBlock(x, y, z).occupant;
  function oldFindPath(start, end, allowDiagonals = false) {
    const openSet = [start];
    const cameFrom = new Map();
    const gScore = new Map();
    gScore.set(getVoxelKey(start.x, start.y, start.z), 0);
    const getH = (a, b) => {
      const dx = Math.abs(a.x - b.x);
      const dz = Math.abs(a.z - b.z);
      return allowDiagonals ? (Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz)) : (dx + dz);
    };
    const fScore = new Map();
    fScore.set(getVoxelKey(start.x, start.y, start.z), getH(start, end));
    const dirs = [{x: 0, z: -1, cost: 1}, {x: 0, z: 1, cost: 1}, {x: -1, z: 0, cost: 1}, {x: 1, z: 0, cost: 1}];
    if (allowDiagonals) {
      dirs.push({x: -1, z: -1, cost: Math.SQRT2}, {x: 1, z: -1, cost: Math.SQRT2},
                {x: -1, z: 1, cost: Math.SQRT2}, {x: 1, z: 1, cost: Math.SQRT2});
    }
    while (openSet.length > 0) {
      let current = openSet[0];
      let lowestIndex = 0;
      let currentKey = getVoxelKey(current.x, current.y, current.z);
      for (let i = 1; i < openSet.length; i++) {
        const nodeKey = getVoxelKey(openSet[i].x, openSet[i].y, openSet[i].z);
        if ((fScore.get(nodeKey) ?? Infinity) < (fScore.get(currentKey) ?? Infinity)) {
          current = openSet[i]; lowestIndex = i; currentKey = nodeKey;
        }
      }
      if (current.x === end.x && current.y === end.y && current.z === end.z) {
        const path = [];
        let currNode = current;
        let currKey = getVoxelKey(currNode.x, currNode.y, currNode.z);
        while (cameFrom.has(currKey)) {
          path.unshift(currNode);
          currNode = cameFrom.get(currKey);
          currKey = getVoxelKey(currNode.x, currNode.y, currNode.z);
        }
        return path;
      }
      openSet.splice(lowestIndex, 1);
      for (const dir of dirs) {
        // Levels as the new search steps them (stepLevel, 2026-09-26).
        const ny = stepLevel(current.x, current.y, current.z, current.x + dir.x, current.z + dir.z);
        if (ny === null) continue;
        const neighbor = { x: current.x + dir.x, y: ny, z: current.z + dir.z };
        const neighborKey = getVoxelKey(neighbor.x, neighbor.y, neighbor.z);
        if (currentArena && !currentArena.has(neighborKey)) continue;
        if (!isEnterable(neighbor.x, neighbor.y, neighbor.z)) continue;
        if (allowDiagonals && dir.cost > 1) {
          const corner = (nx, nz) => {
            const cy = stepLevel(current.x, current.y, current.z, nx, nz);
            return cy !== null && isEnterable(nx, cy, nz);
          };
          if (!corner(current.x + dir.x, current.z)) continue;
          if (!corner(current.x, current.z + dir.z)) continue;
        }
        const tentative = (gScore.get(currentKey) ?? Infinity) + dir.cost;
        if (tentative < (gScore.get(neighborKey) ?? Infinity)) {
          cameFrom.set(neighborKey, current);
          gScore.set(neighborKey, tentative);
          fScore.set(neighborKey, tentative + getH(neighbor, end));
          if (!openSet.find(n => n.x === neighbor.x && n.y === neighbor.y && n.z === neighbor.z)) {
            openSet.push(neighbor);
          }
        }
      }
    }
    return [];
  }

  const tiles = [];
  for (let x = 0; x < 36; x++) for (let z = 0; z < 36; z++) {
    const y = getColumnTop(x, z);
    if (y !== null) tiles.push({ x, y, z });
  }
  const same = (a, b) => a.length === b.length &&
    a.every((n, i) => n.x === b[i].x && n.y === b[i].y && n.z === b[i].z);
  // Deterministic sample: every 37th tile as a start, every 13th as an end -
  // near, far, blocked, around the wall, up the steps onto the platform.
  let pairs = 0, mismatches = 0, ties = 0;
  const run = () => {
    for (let i = 0; i < tiles.length; i += 37) for (let j = 0; j < tiles.length; j += 13) {
      for (const diag of [false, true]) {
        const a = oldFindPath(tiles[i], tiles[j], diag), b = findPath(tiles[i], tiles[j], diag);
        pairs++;
        if (!same(a, b)) mismatches++;
        if (a.length > 1) ties++;
      }
    }
  };
  run();
  ok(`${pairs} start/end pairs, 4-way and diagonal: the same paths`, mismatches, 0);
  truthy('the sample includes real multi-step paths', ties > 50);
  // In battle the arena bounds the search.
  enterBattle({ x: 13, y: 0, z: 13 });
  pairs = 0; mismatches = 0;
  run();
  exitBattle();
  ok(`${pairs} pairs inside a battle arena: the same paths`, mismatches, 0);
}

section('levels: the world as data');
{
  const { serializeLevel, loadLevel, newBlock, WorldModels, ModelBoxes, removeWorldModel,
          addWorldModel, mountWorldModel, clearWorldModels, LEVEL_FORMAT } = await import('../js/world.js');
  const { parseBBModel } = await import('../js/bbmodel.js');
  const { readFileSync } = await import('node:fs');
  const models = new Map(['decor_chair', 'decor_wallTorch'].map(n => [n, parseBBModel(
    readFileSync(new URL(`../assets/objects/${n}.bbmodel`, import.meta.url), 'utf8'), n)]));
  createTestArea(36, 36);
  clearWorldModels();
  addWorldModel(models.get('decor_chair'), 6, 0, 8, 1);
  mountWorldModel(models.get('decor_wallTorch'), 19, 2, 11, 'E');
  const saved = serializeLevel();
  ok('its format', saved.format, LEVEL_FORMAT);
  ok('every block', saved.blocks.length, World.size);
  ok('a slope keeps its shape and facing', JSON.stringify(saved.blocks.find(b => b[0] === 29 && b[1] === 1 && b[2] === 27)),
     '[29,1,27,"grass","slope","S"]');
  ok('a plain block is four numbers and a material', JSON.stringify(saved.blocks.find(b => b[0] === 0 && b[2] === 0)),
     '[0,0,0,"grass"]');
  ok('the models', JSON.stringify(saved.models),
     '[["decor_chair",6,0,8,1],["decor_wallTorch",19,2,11,0,"E"]]');
  // Round trip: through JSON, into an emptied world, and out again the same.
  const text = JSON.stringify(saved);
  World.clear(); clearWorldModels();
  ok('loads with nothing missing', loadLevel(JSON.parse(text), models).length, 0);
  ok('and saves back identically', JSON.stringify(serializeLevel()), text);
  truthy('the chair takes its tile again', !!World.get('6,0,8').occupant);
  ok('the models bake again', ModelBoxes.length,
     models.get('decor_chair').cubes.length + models.get('decor_wallTorch').cubes.filter(c => !c.light).length);
  ok('an unknown model is reported, not placed',
     JSON.stringify(loadLevel({ format: LEVEL_FORMAT, blocks: [[0, 0, 0, 'grass']], models: [['nope', 0, 0, 0, 0]] }, models)),
     '["nope"]');
  let threw = false; try { loadLevel({ blocks: [] }, models); } catch { threw = true; }
  truthy('a file that is not a level is refused', threw);

  // Removing a model frees its tile and its boxes.
  World.clear(); clearWorldModels();
  World.set('1,0,1', newBlock('grass'));
  const c = addWorldModel(models.get('decor_chair'), 1, 0, 1, 0);
  truthy('removed', removeWorldModel(c));
  ok('nothing left to bake', ModelBoxes.length, 0);
  falsy('its tile is free', !!World.get('1,0,1').occupant);
  falsy('removing it twice does nothing', removeWorldModel(c));

  ok('a full block has no shape', newBlock('grass', 'full', 'N').shape, undefined);
  ok('only a slope keeps a facing', `${newBlock('grass', 'halfTop', 'N').facing},${newBlock('grass', 'slope', 'W').facing}`,
     'undefined,W');
  createTestArea(36, 36);
}

section('stairs');
{
  const { blockShape: shapeOf, hasFacing, isCompound, isSlope: slopeOf, groundHeightAt: ground } = await import('../js/world.js');
  const lower = getBlock(18, 1, 12), upper = getBlock(19, 2, 12);
  ok('marble stairs', `${lower.materialId},${shapeOf(lower)},${lower.facing}`, 'marble,stairs,E');
  truthy('stairs turn', hasFacing('stairs'));
  falsy('stairs are no slope plane', slopeOf(lower));
  truthy('two boxes', isCompound(lower));
  ok('facing E, the upper half is the +x half', JSON.stringify(blockBoxes(lower)[1]), '[0.5,1,0.5,1,0,1]');
  ok('its foot (toward -x) at half height', edgeLevel(18, 1, 12, -1, 0), 1.5);
  ok('its head at the top', edgeLevel(18, 1, 12, 1, 0), 2);
  ok('onto the stairs from the ground', stepLevel(17, 0, 12, 18, 12), 1);
  ok('onto the upper stairs', stepLevel(18, 1, 12, 19, 12), 2);
  ok('onto the wall top', stepLevel(19, 2, 12, 20, 12), 2);
  const up = findPath({ x: 12, y: 0, z: 12 }, { x: 20, y: 2, z: 10 });
  truthy('a path up the stairs onto the wall', up.length > 0 && up.some(n => n.x === 19 && n.y === 2 && n.z === 12));
  // The treads under a point: the downhill half at half height, the uphill at full.
  const B = 1.5;
  ok('the lower tread', ground({ x: 18, y: 1, z: 12 }, 18 * B - 0.3, 12 * B), (1.5 - 0.5) * B);
  ok('the upper tread', ground({ x: 18, y: 1, z: 12 }, 18 * B + 0.3, 12 * B), (2 - 0.5) * B);
  truthy('solid under the upper tread', isSolidAt(18 * B + 0.3, 2.1, 12 * B));
  falsy('air over the lower tread', isSolidAt(18 * B - 0.3, 2.1, 12 * B));
}
