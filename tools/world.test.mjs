import { file, section, ok, truthy, falsy } from './lib/harness.mjs';
import { World, createTestArea, createEntity, isStandable, isSolid, getColumnTop,
         findPath, getReachableVoxels, enterBattle, exitBattle, addToInventory,
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
ok('ground layer + test features only', World.size, 36 * 36 + 14 + 3 + 16 + 14 + 12);

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

section('pathfinding is strictly same-level');
truthy('flat path found', findPath({x:8,y:0,z:8}, {x:14,y:0,z:8}).length > 0);
ok('cannot climb to the platform', findPath({x:8,y:0,z:8}, {x:30,y:1,z:30}).length, 0);
ok('cannot enter the wall footprint', findPath({x:8,y:0,z:8}, {x:20,y:0,z:10}).length, 0);
const around = findPath({x:8,y:0,z:11}, {x:25,y:0,z:11});
truthy('routes around the wall', around.length > 0);
falsy('never passes through it', around.some(n => n.x === 20 && n.z >= 8 && n.z <= 14));
truthy('stays on one level', around.every(n => n.y === 0));

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
