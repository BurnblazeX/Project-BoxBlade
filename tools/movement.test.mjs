import * as THREE from 'three/webgpu';
import { file, section, ok, near } from './lib/harness.mjs';
import { occupantAt, setOccupantAt } from '../js/core/world.js';
import { World, getVoxelKey, BLOCK_METRES as B } from '../js/core/world.js';
import { followPath, walk, vacate, occupy, faceSprite, getSpriteWorldPos } from '../js/game/movement.js';

file('movement.test.mjs - characters walking and holding their block');

// A flat 6 x 6 floor at y 0, plus whatever a test adds.
function floor() {
  World.clear();
  for (let x = 0; x < 6; x++) for (let z = 0; z < 6; z++) put(x, 0, z);
}
function put(x, y, z) {
  World.set(getVoxelKey(x, y, z), { solid: true, walkable: true, materialId: 'grass', occupant: null, triggerId: null });
}
const at = (x, y, z) => World.get(getVoxelKey(x, y, z));
function character(id, x, z) {
  const e = { id, gridPos: { x, y: 0, z } };
  occupy(e);
  const sprite = { position: getSpriteWorldPos(e.gridPos),
                   material: { map: { repeat: { x: 1 }, offset: { x: 0 } } } };
  return { e, sprite };
}

section('occupancy');
floor();
{
  const a = character('a', 1, 1), b = character('b', 2, 1);
  ok('each holds its block', occupantAt(getVoxelKey(1, 0, 1)) + occupantAt(getVoxelKey(2, 0, 1)), 'ab');
  a.e.gridPos = { x: 2, y: 0, z: 1 };
  vacate(a.e);
  ok("vacating never clears another's", occupantAt(getVoxelKey(2, 0, 1)), 'b');
}

section('following a path');
floor();
{
  const { e, sprite } = character('p', 0, 0);
  const node = { x: 1, y: 0, z: 0 };
  const path = [node];
  // 8 m/s for 0.1 s is 0.8 m: not there yet.
  ok('short of the node: still walking', followPath(e, sprite, path, 8, 0.1, 0), false);
  near('moved 0.8 m along x', sprite.position.x, 0.8);
  ok('arrival-only occupancy stays put', occupantAt(getVoxelKey(0, 0, 0)), 'p');
  ok('facing right, walking +x at heading 0', sprite.material.map.repeat.x, -1);
  ok('arrives', followPath(e, sprite, path, 8, 0.1, 0), true);
  near('snapped onto the node', sprite.position.x, B);
  ok('the node became its gridPos', e.gridPos, node);
  ok('path consumed', path.length, 0);
  ok('the old block is free', occupantAt(getVoxelKey(0, 0, 0)), null);
  ok('the new block is held', occupantAt(getVoxelKey(1, 0, 0)), 'p');
}
floor();
{
  const { e, sprite } = character('p', 0, 0);
  const path = [{ x: 1, y: 0, z: 0 }];
  followPath(e, sprite, path, 8, 0.1, 0, true);   // 0.8 m: past the column edge at 0.75 m
  ok('continuous: the column it is over holds it', occupantAt(getVoxelKey(1, 0, 0)), 'p');
  ok('continuous: and the one it left does not', occupantAt(getVoxelKey(0, 0, 0)), null);
  ok('continuous: gridPos follows', e.gridPos.x, 1);
}

section('walking on the keys');
floor();
{
  const { e, sprite } = character('p', 1, 1);
  walk(e, sprite, { d: true }, 0, 0.1);
  near('d at heading 0 walks +x', sprite.position.x, B + 0.8);
  ok('into the next column', e.gridPos.x, 2);
  ok('which it holds', occupantAt(getVoxelKey(2, 0, 1)), 'p');
}
floor();
{
  const { e, sprite } = character('p', 1, 1);
  character('q', 2, 1);
  walk(e, sprite, { d: true }, 0, 0.1);
  near("blocked by another's block", sprite.position.x, B);
  sprite.position.x = B + 0.7;   // still column 1, at its edge
  walk(e, sprite, { d: true, s: true }, 0, 0.1);
  ok('diagonal into it slides along z', sprite.position.z > B && sprite.position.x === B + 0.7, true);
}
floor();
{
  put(2, 1, 1);   // a full block up: a jump, not a step
  const { e, sprite } = character('p', 1, 1);
  walk(e, sprite, { d: true }, 0, 0.1);
  near('a full block up blocks the walk', sprite.position.x, B);
}

section('facing');
{
  const s = { material: { map: { repeat: { x: 1 }, offset: { x: 0 } } } };
  faceSprite(s, true);
  ok('right mirrors the art', `${s.material.map.repeat.x},${s.material.map.offset.x}`, '-1,1');
  faceSprite(s, false);
  ok('left restores it', `${s.material.map.repeat.x},${s.material.map.offset.x}`, '1,0');
}
World.clear();
