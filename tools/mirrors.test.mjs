import { file, section, ok, near, truthy, falsy } from './lib/harness.mjs';
import { World, createTestArea, BLOCK_METRES } from '../js/world.js';
import { buildMirrors, packMirrors, isReflective, mirrorReach, MAX_MIRRORS, MIRROR_SUN_BIT } from '../js/mirrors.js';

file('mirrors.test.mjs - reflective surfaces as rectangles');

section('which materials reflect');
falsy('no _s never does', isReflective(null));
falsy('fully rough does not', isReflective(new Uint8Array([0, 10, 0, 255])));
truthy('one polished texel is enough', isReflective(new Uint8Array([0, 10, 0, 255, 245, 230, 0, 255])));

section('the test bed');
createTestArea(36, 36);
const rects = buildMirrors(World, new Set(['ironPlate', 'marble']));
const floor = rects.filter(r => r.axis === 1 && r.sign === 1 && r.plane === 0.5 * BLOCK_METRES
                               && r.min[0] < 10 * BLOCK_METRES);
// The iron patch is 5 x 5 at ground level: its top is one rectangle.
const iron = floor.find(r => r.max[1] <= 16.5 * BLOCK_METRES);
truthy('the iron floor top is found', !!iron);
ok('as one rectangle, 5 blocks wide', (iron.max[0] - iron.min[0]) / BLOCK_METRES, 5);
ok('and 5 deep', (iron.max[1] - iron.min[1]) / BLOCK_METRES, 5);
falsy('its sides are buried - no side faces', rects.some(r =>
  r.axis !== 1 && r.plane >= 2.5 * BLOCK_METRES && r.plane <= 7.5 * BLOCK_METRES
  && r.min[1] < 0.5 * BLOCK_METRES && r.max[1] <= 16.5 * BLOCK_METRES && r.axis === 0));
// The marble wall: 7 long, 2 tall, at z = 17. Its floor-facing side is -z.
const face = rects.find(r => r.axis === 2 && r.sign === -1 && r.plane === 16.5 * BLOCK_METRES);
truthy('the wall face toward the floor is found', !!face);
ok('7 blocks long', (face.max[0] - face.min[0]) / BLOCK_METRES, 7);
ok('2 blocks tall', (face.max[1] - face.min[1]) / BLOCK_METRES, 2);
truthy('the wall bottom sits on blocks - no downward face',
       !rects.some(r => r.axis === 1 && r.sign === -1));
const plateIn = rects.find(r => r.axis === 0 && r.sign === 1 && r.plane === 1.5 * BLOCK_METRES);
truthy('the west plate faces the floor (+x)', !!plateIn);
ok('3 blocks long', (plateIn.max[1] - plateIn.min[1]) / BLOCK_METRES, 3);
truthy('the east plate faces it back (-x)',
       rects.some(r => r.axis === 0 && r.sign === -1 && r.plane === 8.5 * BLOCK_METRES));
truthy('few rectangles for the whole bed', rects.length <= 20);

section('what a mirror can reflect');
{
  const floorTop = iron;   // faces up
  const high = { dir: [0.5, 0.7, 0.5], on: true }, below = { dir: [0.5, -0.7, 0.5], on: true };
  truthy('a floor facing the sun has the sun bit', mirrorReach(floorTop, high, []).mask & MIRROR_SUN_BIT);
  ok('with the sun below it, nothing', mirrorReach(floorTop, below, []).mask, 0);
  ok('with the sun off, nothing', mirrorReach(floorTop, { dir: [0, 1, 0], on: false }, []).mask, 0);
  const cx = (iron.min[0] + iron.max[0]) / 2, cz = (iron.min[1] + iron.max[1]) / 2;
  const torch = { x: cx, y: 3, z: cz, radius: 10 };
  const far = { x: cx + 100, y: 3, z: cz, radius: 10 };
  const under = { x: cx, y: -3, z: cz, radius: 10 };
  const m = mirrorReach(floorTop, { dir: [0, 1, 0], on: false }, [far, torch, under]).mask;
  ok('only the torch above it, in range, gets a bit', m, 1 << 1);
  const box = mirrorReach(floorTop, high, []).box;
  truthy('the sun box holds the mirror', box[0][1] <= floorTop.plane && box[1][1] >= floorTop.plane);
  truthy('and reaches up along the reflected sun', box[1][1] > floorTop.plane + 10);
}

section('packing');
const out = new Float32Array(MAX_MIRRORS * 16);
const all = packMirrors(rects, { x: 5 * BLOCK_METRES, y: 0, z: 14 * BLOCK_METRES }, out, MAX_MIRRORS,
                        { dir: [0, 1, 0], on: true });
truthy('mirrors facing up the sun are packed', all > 0);
ok('nearest first - the floor under the point', out[0], 1);
ok('nothing lit, nothing packed',
   packMirrors(rects, { x: 0, y: 0, z: 0 }, out, MAX_MIRRORS, { dir: [0, 1, 0], on: false }, []), 0);
