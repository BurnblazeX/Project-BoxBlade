import { file, section, ok } from './lib/harness.mjs';
import { World, newBlock, BLOCK_METRES as B } from '../js/core/world.js';
import { playerCell, buildInterior, interiorTargets, easeCells, rekeyCells, maskIndex, maskOrigin,
         CELL_EASE } from '../js/core/interior.js';

file('interior.test.mjs - roofs and front walls under a roof');

// A two-storey house: ground at y 0, a 7x7 room (walls at +-3) on y 1..3, a
// floor slab at y 4, an upper storey's walls on y 5..6, a roof at y 7.
// The player at the room's centre, standing in cell y 1.
const put = (x, y, z, m = 'stone') => World.set(`${x},${y},${z}`, newBlock(m));
function house() {
  World.clear();
  for (let x = -4; x <= 4; x++) for (let z = -4; z <= 4; z++) {
    put(x, 0, z, 'grass');
    put(x, 4, z);
    put(x, 7, z);
    const edge = Math.abs(x) === 3 || Math.abs(z) === 3;
    if (edge && Math.abs(x) <= 3 && Math.abs(z) <= 3) for (let y = 1; y <= 6; y++) if (y !== 4) put(x, y, z);
  }
}
const feet = { x: 0, y: 0.5 * B, z: 0 };
const cell = playerCell(feet);
const ahead = { x: 0, z: -1 };   // the camera at +z, looking along -z
const at = (room, x, y, z) => maskIndex(room.origin, x, y, z);

section('finding the building');
ok('the player stands in cell y 1', JSON.stringify(cell), JSON.stringify({ x: 0, y: 1, z: 0 }));
World.clear();
put(0, 0, 0, 'grass');
ok('in the open: no interior', buildInterior(cell), null);
house();
const room = buildInterior(cell);
ok('under the slab: the roof starts there', room.roofY, 4);
const roof = new Set(room.roof);
ok('the slab, the storey above and the roof are one roof',
   [at(room, 0, 4, 0), at(room, 3, 5, 0), at(room, 0, 7, 0)].every(i => roof.has(i)), true);
ok('the floor is not', roof.has(at(room, 0, 0, 0)), false);
const walls = new Map(room.walls.map(w => [w.i, w]));
ok('the room\'s walls above the stub are walls', walls.has(at(room, 0, 2, 3)) && walls.has(at(room, 0, 3, -3)), true);
ok('the stub row is not', walls.has(at(room, 0, 1, 3)), false);

section('the inside');
const insideOf = r => new Set(r.inside.map(c => `${c.x},${c.z}`));
const inside = insideOf(room);
ok('the room and its walls', ['0,0', '2,-2', '3,0', '-3,3'].every(k => inside.has(k)), true);
ok('not the eaves past the walls', inside.has('4,0') || inside.has('0,-4'), false);
World.delete('0,1,3'); World.delete('0,2,3');
const door = insideOf(buildInterior(cell));
ok('a door: the step under the eaves outside it', door.has('0,4'), true);
ok('... but not the eaves elsewhere', door.has('4,0'), false);
ok('on the step: inside, through the door', insideOf(buildInterior({ x: 0, y: 1, z: 4 })).has('0,0'), true);
const inDoor = buildInterior({ x: 0, y: 1, z: 3 });
ok("in the doorway: the roof is the building's, not the door's top", inDoor.roofY, 4);
ok('... the walls beside the door stay walls', new Set(inDoor.roof).has(at(inDoor, 1, 3, 3)), false);
ok("... the door's top lifts with its wall", interiorTargets(inDoor, ahead).has(at(inDoor, 0, 3, 3)), true);
ok('under the eaves by a wall: not under a roof', buildInterior({ x: 4, y: 1, z: 0 }), null);
house();

section('which walls lift');
const t = interiorTargets(room, ahead);
ok('the roof', t.get(at(room, 0, 4, 0)), 1);
ok('the wall facing the camera', t.get(at(room, 0, 2, 3)), 1);
ok('... to its corners', t.has(at(room, 3, 2, 3)) && t.has(at(room, -3, 2, 3)), true);
ok('not the wall behind', t.has(at(room, 0, 2, -3)), false);
ok('not the side walls, even their near ends', t.has(at(room, 3, 2, 0)) || t.has(at(room, 3, 2, 2)), false);
ok('not the stub', t.has(at(room, 0, 1, 3)), false);
const turned = interiorTargets(room, { x: -1, z: 0 });
ok('a quarter turn: the next wall lifts instead', [turned.has(at(room, 3, 2, 0)), turned.has(at(room, 0, 2, 3))].join(),
   'true,false');
const corner = interiorTargets(room, { x: -Math.SQRT1_2, z: -Math.SQRT1_2 });
ok('from a corner (battle): both walls facing it',
   [corner.has(at(room, 0, 2, 3)), corner.has(at(room, 3, 2, 0)), corner.has(at(room, 3, 2, 3))].join(), 'true,true,true');
ok('... not the two behind', corner.has(at(room, 0, 2, -3)) || corner.has(at(room, -3, 2, 0)), false);
ok('the walls kept: only the roof', interiorTargets(room, ahead, false).has(at(room, 0, 2, 3)), false);
World.delete('3,2,0');   // a window in the side wall, over its sill
const windowed = buildInterior(cell), w = interiorTargets(windowed, ahead);
ok('a window in a side wall: the wall beside it stays', w.has(at(windowed, 3, 2, 1)) || w.has(at(windowed, 3, 3, 1)), false);
ok('... and over it', w.has(at(windowed, 3, 3, 0)), false);
ok('... the front wall still lifts', w.has(at(windowed, 0, 2, 3)), true);
house();
ok('no interior: nothing', interiorTargets(null, ahead).size, 0);

section('easing');
{
  const v = new Map();
  const target = new Map([[5, 1]]);
  ok('a cell starts fading', easeCells(v, target, CELL_EASE / 2), true);
  ok('... halfway', v.get(5), 0.5);
  easeCells(v, target, CELL_EASE);
  ok('... done', v.get(5), 1);
  ok('held: nothing moves', easeCells(v, target, 0.1), false);
  easeCells(v, new Map(), CELL_EASE);
  ok('target gone: faded back and dropped', v.has(5), false);
  const a = maskOrigin({ x: 0, y: 1, z: 0 }), b = maskOrigin({ x: 1, y: 1, z: 0 });
  const moved = rekeyCells(new Map([[maskIndex(a, 2, 3, 4), 0.7]]), a, b);
  ok('the mask moved with the player: same cell, new index', moved.get(maskIndex(b, 2, 3, 4)), 0.7);
}
World.clear();
