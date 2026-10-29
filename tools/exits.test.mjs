import { file, section, ok, truthy, falsy } from './lib/harness.mjs';
import { exitBox, exitAt, exitsWithout, validExit, createExitTracker } from '../js/core/exits.js';

file('exits.test.mjs - exits between regions (World plan P2, step 3)');

const door = { cells: [10, 0, 4, 11, 0, 4], target: 'test_b', spawn: 'start' };
const gate = { cells: [0, 2, 0, 0, 2, 0], target: 'dev_level', spawn: 'east' };

section('boxes');
{
  ok('corners in any order', exitBox([3, 1, 9], [1, 0, 7]).join(), '1,0,7,3,1,9');
  ok('a cell of the box', exitAt([door, gate], 11, 0, 4), door);
  ok('another exit', exitAt([door, gate], 0, 2, 0), gate);
  ok('beside it: none', exitAt([door, gate], 12, 0, 4), null);
  ok('a level up: none', exitAt([door], 10, 1, 4), null);
  ok('no list', exitAt(undefined, 0, 0, 0), null);
  ok('removing by a cell drops the exit it touches', exitsWithout([door, gate], [10, 0, 4, 10, 0, 4]).length, 1);
  ok('a cell off every exit drops none', exitsWithout([door, gate], [5, 0, 5, 5, 0, 5]).length, 2);
}

section('read from a file');
{
  ok('a good one, its box put in order', JSON.stringify(validExit({ cells: [2, 0, 2, 1, 0, 1], target: 'a' })),
     '{"cells":[1,0,1,2,0,2],"target":"a","spawn":"start"}');
  ok('no target', validExit({ cells: [0, 0, 0, 0, 0, 0] }), null);
  ok('a broken box', validExit({ cells: [[1, 1, 0]], target: 'b' }), null);
  ok('fractions', validExit({ cells: [0.5, 0, 0, 1, 0, 0], target: 'b' }), null);
}

section('which exit is offered');
{
  const t = createExitTracker();
  ok('standing on one: offered', t.update([door], { x: 10, y: 0, z: 4 }), door);
  ok('off it: none', t.update([door], { x: 9, y: 0, z: 4 }), null);
  t.disarm();
  ok('arrived on one: not offered back', t.update([door], { x: 10, y: 0, z: 4 }), null);
  falsy('still disarmed while on it', t.armed);
  ok('stepping off arms it', t.update([door], { x: 9, y: 0, z: 4 }), null);
  truthy('armed', t.armed);
  ok('and back on: offered', t.update([door], { x: 11, y: 0, z: 4 }), door);
}
