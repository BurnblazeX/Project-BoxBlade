import { file, section, ok, note } from './lib/harness.mjs';
import { ring, volumeOffset, enteredBoxes, ringRuns, writeVolumeBox } from '../js/core/blockvolume.js';
import { createChunkWorld } from '../js/core/chunks.js';
import { newBlock, CHUNK_SIZE, Y_MIN, CHUNK_HEIGHT } from '../js/core/world.js';
import { windowAxis } from '../js/core/follow.js';

file('blockvolume.test.mjs - the block volume as a ring: strips, not rebuilds (World plan P2a)');

section('the ring');
{
  ok('a negative block wraps', ring(-1, 128), 127);
  ok('a multiple is 0', ring(-256, 128), 0);
  ok('one run that fits', JSON.stringify(ringRuns(130, 137, 128)), '[[2,9]]');
  ok('two where it wraps', JSON.stringify(ringRuns(124, 131, 128)), '[[124,127],[0,3]]');
  ok('a dual half sits size[2] along', volumeOffset([0, Y_MIN, 0], [4, 2, 4], 1, Y_MIN, 1, true) -
     volumeOffset([0, Y_MIN, 0], [4, 2, 4], 1, Y_MIN, 1), 4 * 2 * 4 * 4);
}

section('entered strips');
{
  ok('a move east brings in one strip', JSON.stringify(enteredBoxes([0, Y_MIN, 0], [8, Y_MIN, 0], [128, 48, 128])),
     '[[128,135,0,127]]');
  ok('a diagonal move: an x strip, and a z strip over the rest',
     JSON.stringify(enteredBoxes([0, Y_MIN, 0], [-8, Y_MIN, 16], [128, 48, 128])), '[[-8,-1,16,143],[0,119,128,143]]');
  ok('no overlap: everything', enteredBoxes([0, Y_MIN, 0], [500, Y_MIN, 0], [128, 48, 128]), null);
}

section('strips land exactly where a full build puts the bytes');
{
  // A world wider than a small window: ground, a wall, a dual block, a high
  // column for the sky tops, blocks out of reach on both sides.
  const world = createChunkWorld({ size: CHUNK_SIZE, yMin: Y_MIN, height: CHUNK_HEIGHT });
  for (let x = -40; x < 100; x++) for (let z = -30; z < 70; z++) world.setAt(x, 0, z, newBlock((x * 7 + z) % 3 ? 'grass' : 'marble'));
  for (let y = 1; y < 6; y++) world.setAt(10, y, 10, newBlock('marble'));
  world.setAt(33, 1, -12, newBlock('marble', 'halfBottom', null, 'glass'));
  world.setAt(-35, 20, 60, newBlock('grass', 'slope', 'E'));
  const hits = { index: new Map([['12,1,12', { header: 300, giIndex: 2 }], ['-30,1,50', { header: 7, giIndex: 0 }]]) };
  const size = [32, CHUNK_HEIGHT, 32];
  const make = () => ({ data: new Uint8Array(size[0] * size[1] * size[2] * 8), skyTop: new Uint8Array(size[0] * size[2] * 2),
                        size });
  const full = lo => {
    const v = { ...make(), lo };
    writeVolumeBox(v, world, hits, [lo[0], lo[0] + size[0] - 1, lo[2], lo[2] + size[2] - 1], true);
    return v;
  };
  // Walk a focus around and back, the window following as render.js's does.
  const live = full([0, Y_MIN, 0]);
  let lo = [0, Y_MIN, 0], moves = 0, same = true, firstBad = null;
  const path = [];
  for (let t = 0; t < 400; t++) path.push([Math.round(30 * Math.sin(t / 17) + 25), Math.round(40 * Math.cos(t / 23) + 15)]);
  for (const [fx, fz] of path) {
    const next = [windowAxis(lo[0], fx, size[0], -41, 100, 8, 4), Y_MIN, windowAxis(lo[2], fz, size[2], -31, 70, 8, 4)];
    if (next[0] === lo[0] && next[2] === lo[2]) continue;
    const boxes = enteredBoxes(lo, next, size);
    lo = next;
    moves++;
    live.lo = lo;
    if (!boxes) { Object.assign(live, full(lo)); continue; }
    for (const box of boxes) writeVolumeBox(live, world, hits, box);
    const want = full(lo);
    if (same && (Buffer.compare(Buffer.from(live.data), Buffer.from(want.data)) ||
                 Buffer.compare(Buffer.from(live.skyTop), Buffer.from(want.skyTop)))) { same = false; firstBad = lo.join(); }
  }
  note(`${moves} moves`);
  ok('the window moved, many times', moves > 20, true);
  ok('after every move: the same bytes as built from nothing', same ? 'same' : `differs at ${firstBad}`, 'same');
}
