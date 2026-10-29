import { file, section, ok } from './lib/harness.mjs';
import { createSmokeCloud, sphereCubes, sphereMeasure, smokeAt, smokeRadius, smokeAppear, SMOKE_SPHERE,
         SMOKE_RADIUS, SMOKE_CUBE, CLOUD_SPELLS, MAX_CAST_LEVEL, SMOKE_MAX_RADIUS,
         SMOKE_BOX, SMOKE_BOX_FROM } from '../js/core/smoke.js';
import { BLOCK_METRES } from '../js/core/world.js';

file('smoke.test.mjs - cloud spells: a volume of cubes round the emitter');

section('the sphere');
const S = SMOKE_SPHERE;
const at = (i, j, k) => S.some(c => c[0] === i && c[1] === j && c[2] === k);
const layer = j => S.filter(c => c[1] === j);
const top = Math.max(...S.map(c => c[1]));
ok('symmetric across x and z', S.every(([i, j, k]) => at(-i, j, k) && at(i, j, -k) && at(k, j, i)), true);
ok('symmetric up and down about the emitter', S.every(([i, j, k]) => at(i, -1 - j, k)), true);
ok('17 cubes across at the middle', layer(0).filter(c => c[2] === 0).length, 17);
ok('no nub: the top layer is a cap, not one cube', layer(top).length > 1, true);
ok('the top is not flat: each layer up is smaller', [...Array(top + 1).keys()].every(j => j === 0 || layer(j).length <= layer(j - 1).length), true);

const ground = (x, y) => y < 0.75;              // everything below the top of block row 0
const tile = { gx: 0, gy: 0, gz: 0 };
const fog = CLOUD_SPELLS.fogCloud;
const keyOf = c => c.slice(4, 7).join();

section('volume on open ground');
const c = createSmokeCloud(tile, ground);
const upper = S.filter(([, j]) => j >= 0).length;
ok('the volume is the half above ground', c.volume, upper);
ok('the cloud holds its volume', c.cubes.length, c.volume);
ok('exactly the upper half sphere', c.cubes.every(([, , , m]) => m <= 1) && c.cubes.every(([, y]) => y > 0.75), true);
ok('the emitter is the top face\'s middle', c.centre.join(), [0, 0.75, 0].join());
ok('the first cube taken is over the emitter', c.cubes[0].slice(0, 3).join(), [0, 0.75 + SMOKE_CUBE / 2, 0].join());
ok('taken from the inside out', c.cubes.every((q, n) => n === 0 || q[3] >= c.cubes[n - 1][3] - 1e-12), true);
ok('it rolls out from the emitter', smokeAppear(c, c.cubes[0][3]) < smokeAppear(c, c.cubes[c.cubes.length - 1][3]), true);
ok('the block over the emitter is full smoke', smokeAt(c, 0, 1, 0), 1);
const surface = c.cubes.filter(q => q[7]).length;
ok('only some cubes are on the surface', surface > 0 && surface < c.cubes.length, true);
ok('a cube in the middle is not', c.cubes.find(([x, y, z]) => x === 0 && z === 0 && Math.abs(y - 2.5) < 1e-9)[7], 0);
ok('one on the ground is not open to air', c.cubes[0][7], 0);
ok('... it presses on the ground below (face -y)', c.cubes[0][8], 1 << 3);

section('the underground part never spills');
// A floor three blocks thick: the lower half is all solid.
const thick = (x, y) => y < 0.75 && y > -4;
const t = createSmokeCloud(tile, thick);
ok('the same volume as on thin ground', t.volume, upper);
ok('no cube past the radius', t.cubes.every(([, , , m]) => m <= 1), true);
// One layer of ground over nothing - as a level often is: the void under it is underground too.
const layer1 = (x, y) => y < 0.75 && y >= -0.75;
const v = createSmokeCloud(tile, layer1);
ok('a one-layer floor over a void: the same volume', v.volume, upper);
ok('... and none of it below', v.cubes.every(([, y]) => y > 0.75), true);
// On a pillar's top: below the emitter's plane is air, and smoke - no spill there either.
const pillar = (x, y, z) => Math.abs(x) <= 0.75 && Math.abs(z) <= 0.75 && y < 0.75 && y > -9;
const p = createSmokeCloud(tile, pillar);
ok('on a pillar the air below takes smoke', p.cubes.some(([, y]) => y < 0.75), true);
ok('... reached round the edge: farther than in a straight line', p.cubes.filter(([, y]) => y < 0.75)
   .every(([x, y, z, travel]) => travel > Math.sqrt(sphereMeasure(x / SMOKE_CUBE, (y - 0.75) / SMOKE_CUBE - 0.5, z / SMOKE_CUBE, p.radius)) - 1e-6), true);
ok('... and the cloud keeps its volume', p.cubes.length, p.volume);

section('it flows round corners');
// A wall at x in [1.5, 3), full height, with a doorway at z in [0, 1.5), y up to 3.
const doorWall = (x, y, z) => ground(x, y) || (x >= 1.5 && x < 3 && !(z >= 0 && z < 1.5 && y < 3));
const d = createSmokeCloud(tile, doorWall, fog, 3);   // upcast: a cloud big enough to reach past the wall
const behind = d.cubes.filter(([x, , z]) => x >= 3 && z < -0.75);
ok('smoke reaches behind the wall through the doorway', behind.length > 0, true);
ok('... travelling round: farther than the straight line', behind.every(([x, y, z, travel]) =>
   travel > Math.sqrt(sphereMeasure(x / SMOKE_CUBE, (y - 0.75) / SMOKE_CUBE - 0.5, z / SMOKE_CUBE, d.radius)) + 1e-6), true);
ok('... so it rolls in there later', Math.min(...behind.map(q => q[3])) > Math.min(...d.cubes.filter(([x, , z]) => x >= 3 && z >= 0 && z < 1.5).map(q => q[3])), true);
// A corridor 1 block wide and 2 high along +x: the volume runs down it past the radius.
const corridor = (x, y, z) => y < 0.75 || y >= 3.75 || Math.abs(z) > 0.75;
// Darkness: small enough that its volume fits the corridor inside the cloud's box (64 cubes long).
const run = createSmokeCloud(tile, corridor, CLOUD_SPELLS.darkness);
ok('in a corridor it keeps its volume', run.cubes.length, run.volume);
ok('... and runs past its radius', Math.max(...run.cubes.map(([x]) => Math.abs(x))) > run.radius * SMOKE_CUBE * 1.5, true);

section('volume round the geometry');
// A wall filling x in [1.5, 3) above the ground.
const wall = (x, y) => ground(x, y) || (x >= 1.5 && x < 3);
const w = createSmokeCloud(tile, wall);
ok('the same volume as in the open', w.cubes.length, c.volume);
ok('no cube inside the wall', w.cubes.every(([x]) => !(x >= 1.5 && x < 3)), true);
ok('the smoke the wall took goes past the radius, this side', w.cubes.some(([, , , m]) => m > 1), true);
ok('... never through the wall', w.cubes.every(([x]) => x < 1.5), true);
// The same wall as glass: seen through, so the cubes on it are open, not hidden.
const pressed = cs => cs.filter(([x]) => x > 1.5 - 1.5 * SMOKE_CUBE && x < 1.5);
const glassWall = createSmokeCloud(tile, wall, fog, fog.level, (x, y) => !ground(x, y) && x >= 1.5 && x < 3);
ok('against a stone wall the cubes inside hide', pressed(w.cubes).some(c => !c[7] && c[8]), true);
ok('against glass they show', pressed(glassWall.cubes).length > 0 && pressed(glassWall.cubes).every(c => c[7] === 1 && c[8] === 0), true);
ok('... and the cloud is the same', glassWall.cubes.length, w.cubes.length);
// Under a low roof: a slab of solid at y in [2.25, 3).
const roof = (x, y) => ground(x, y) || (y >= 2.25 && y < 3);
const r = createSmokeCloud(tile, roof);
ok('under a roof it spreads out wide instead', r.cubes.length === c.volume && r.cubes.every(([, y]) => y < 2.25), true);
const buried = createSmokeCloud(tile, () => true);
ok('cast inside solid: nothing', buried.cubes.length, 0);

section('upcasting');
// BG3's measured Fog Cloud radii (m), levels 1-6; the cubes are a half-integer at or under each.
[4.5, 6, 8, 10, 10.5, 12].forEach((m, n) => {
  const r = smokeRadius(fog, n + 1) * SMOKE_CUBE;
  ok(`level ${n + 1}: just under BG3's ${m} m`, r <= m && r > m - SMOKE_CUBE, true);
});
ok('past 6th: half a metre a level', smokeRadius(fog, 9) - smokeRadius(fog, 6), 3);
ok('no lower than its own level', smokeRadius(fog, 0), SMOKE_RADIUS);
ok('no higher than 9th', smokeRadius(fog, 12), smokeRadius(fog, MAX_CAST_LEVEL));
ok('the others do not grow when upcast', smokeRadius(CLOUD_SPELLS.cloudkill, 9), CLOUD_SPELLS.cloudkill.radius);
ok('every radius a half-integer of cubes', Object.values(CLOUD_SPELLS).every(s =>
   [...Array(MAX_CAST_LEVEL).keys()].every(l => (smokeRadius(s, l + 1) * 2) % 2 === 1)), true);
const up = createSmokeCloud(tile, ground, fog, 3);
ok('an upcast cloud is the bigger half sphere', up.cubes.length, sphereCubes(smokeRadius(fog, 3)).filter(([, j]) => j >= 0).length);
ok('... and rolls out for longer', up.growSeconds > c.growSeconds, true);
ok('the largest cloud fits its box', 2 * Math.ceil(SMOKE_MAX_RADIUS + 0.5) <= SMOKE_BOX && -SMOKE_BOX_FROM >= Math.ceil(SMOKE_MAX_RADIUS + 0.5), true);
ok('cubes sit on the cube grid', c.cubes.every(([x, , z]) => Math.abs(x / SMOKE_CUBE - Math.round(x / SMOKE_CUBE)) < 1e-9 &&
   Math.abs(z / SMOKE_CUBE - Math.round(z / SMOKE_CUBE)) < 1e-9), true);
ok('no two cubes in one box', new Set(w.cubes.map(keyOf)).size, w.cubes.length);
ok('measure is 1 at the rim', sphereMeasure(6.5, -0.5, 0, 6.5), 1);

section('smoke takes up space');
{
  const { reshapeSmokeCloud, smokeCells, bodyCells, modelCells, cellKey } = await import('../js/core/smoke.js');
  const cells = cl => new Set(cl.cubes.map(q => cellKey(q[4], q[5], q[6])));
  // Bob: a body 1.5 m square and 3 m tall, standing on block (2, 0, 0).
  const bob = x => bodyCells([x * BLOCK_METRES - 0.75, 0.75, -0.75, x * BLOCK_METRES + 0.75, 3.75, 0.75]);
  ok('a body standing on a block: 3 x 3 cells, 6 high', bob(2).size, 54);
  const a = createSmokeCloud(tile, ground);
  const before = cells(a);
  ok('a cast keeps spare cells past its volume', a.candidates > a.volume, true);
  ok('walking in: changed', reshapeSmokeCloud(a, bob(2)), true);
  ok('... no smoke where the body stands', [...bob(2)].some(k => cells(a).has(k)), false);
  ok('... the volume is kept', a.cubes.length, a.volume);
  ok('... the smoke went to the rim', [...cells(a)].filter(k => !before.has(k)).length, [...bob(2)].filter(k => before.has(k)).length);
  ok('... the hole is seen: faces on the body are open', a.cubes.some(q => q[7] && Math.abs(q[0] - 3) < 1 && q[1] < 3.75), true);
  ok('standing still: nothing to do', reshapeSmokeCloud(a, bob(2)), false);
  reshapeSmokeCloud(a, null);
  ok('walking out: the hole fills and the rim draws back', [...cells(a)].every(k => before.has(k)) && cells(a).size === before.size, true);
  ok('gameplay sees it too: the block over the emitter is full again', smokeAt(a, 0, 1, 0), 1);
  // Cast round a body already there.
  const round = createSmokeCloud(tile, ground, fog, 1, undefined, { blocked: bob(1) });
  ok('cast round a body: none on it, the whole volume', [...bob(1)].some(k => cells(round).has(k)) || round.cubes.length !== round.volume, false);

  // Two clouds side by side: the second grows round the first.
  const first = createSmokeCloud(tile, ground);
  const second = createSmokeCloud({ gx: 4, gy: 0, gz: 0 }, ground, fog, 1, undefined, { taken: smokeCells(first) });
  const shared = [...cells(second)].filter(k => cells(first).has(k)).length;
  ok('two clouds never share a cell', shared, 0);
  ok('... the second still holds its volume', second.cubes.length, second.volume);
  ok('... pushed away from the first', second.cubes.some(q => q[0] > 4 * BLOCK_METRES + c.radius * SMOKE_CUBE * 0.9), true);
  ok('... where they meet, faces show (other smoke is seen through)',
     second.cubes.some(q => q[7] && first.cubes.some(p => Math.abs(p[0] - q[0]) === SMOKE_CUBE && p[1] === q[1] && p[2] === q[2])), true);
  // A body walking into the first never pushes it into the second.
  reshapeSmokeCloud(first, bob(3), smokeCells(second));
  ok('displaced smoke keeps out of the other cloud', [...cells(first)].some(k => cells(second).has(k)), false);

  // Models: a table top over the emitter (2 texels thick) blocks; a carpet (1 texel) does not.
  const table = [[-6, 18, -6, 6, 20, 6]];            // world texels: y 18..20 is 2.25..2.5 m
  const carpet = [[-6, 6, -6, 6, 7, 6]];             // y 6..7: on the ground's top (0.75 m)
  ok('a model filling half a cell blocks it', modelCells([-2, 0, -2], 8, table).has(cellKey(0, 5, 0)), true);
  ok('a carpet (a quarter of a cell) does not', modelCells([-2, 0, -2], 8, carpet).size, 0);
  const under = createSmokeCloud(tile, ground, fog, 1, undefined, { models: table });
  ok('smoke keeps out of the model', [...modelCells([-32, -30, -32], 64, table)].some(k => cells(under).has(k)), false);
  ok('... and holds its volume round it', under.cubes.length, under.volume);
}

section('reshaping step by step ends where a fresh cast would');
{
  const { reshapeSmokeCloud, bodyCells, cellKey } = await import('../js/core/smoke.js');
  const walk = x => bodyCells([x - 0.75, 0.75, -1, x + 0.75, 3.75, 0.5]);
  const live = createSmokeCloud(tile, ground);
  for (let x = -6; x <= 3; x += 0.25) reshapeSmokeCloud(live, walk(x));
  const fresh = createSmokeCloud(tile, ground, fog, 1, undefined, { blocked: walk(3) });
  // Faces only: a cast round a body ends its roll-out farther out (its gMax), so travel differs.
  const byCell = cl => new Map(cl.cubes.map(q => [cellKey(q[4], q[5], q[6]), q.slice(7, 9).join()]));
  const a = byCell(live), b = byCell(fresh);
  ok('the same cells', a.size === b.size && [...a.keys()].every(k => b.has(k)), true);
  ok('each with the same faces', [...a].every(([k, v]) => b.get(k) === v), true);
  ok('the same smoke per block', [...live.blocks].every(([k, v]) => fresh.blocks.get(k) === v) && live.blocks.size === fresh.blocks.size, true);
}

section('other clouds passed as one set a cloud (as the renderer does)');
{
  const { smokeCells, cellKey } = await import('../js/core/smoke.js');
  const one = createSmokeCloud(tile, ground), two = createSmokeCloud({ gx: 0, gy: 0, gz: 5 }, ground, fog, 1, undefined, { taken: [smokeCells(one)] });
  const three = createSmokeCloud({ gx: 4, gy: 0, gz: 2 }, ground, fog, 1, undefined, { taken: [smokeCells(one), smokeCells(two)] });
  const of = cl => new Set(cl.cubes.map(q => cellKey(q[4], q[5], q[6])));
  const clash = (p, q) => [...of(p)].some(k => of(q).has(k));
  ok('three clouds, no cell shared', clash(one, two) || clash(one, three) || clash(two, three), false);
  ok('... each holds its volume', [one, two, three].every(cl => cl.cubes.length === cl.volume), true);
}

section('sprites hold their silhouettes, turned, not their boxes');
{
  const { silhouetteRadii, profileCells, bodyCells, cellKey, reshapeSmokeCloud, createSmokeCloud: cast } = await import('../js/core/smoke.js');
  // A 12 x 24 px figure on a 1.5 x 3 m quad: a 2 px neck, 6 px shoulders, 2 px legs, nothing above the head.
  const W = 12, H = 24, rgba = new Uint8Array(W * H * 4);
  const fill = (r0, r1, c0, c1) => { for (let r = r0; r < r1; r++) for (let c = c0; c < c1; c++) rgba[(r * W + c) * 4 + 3] = 255; };
  fill(4, 8, 5, 7);       // head, 2 px
  fill(8, 16, 3, 9);      // body, 6 px
  fill(16, 24, 5, 7);     // legs
  const s = silhouetteRadii(rgba, W, H, 1.5, 3);
  ok('a row reaches from the middle to its farthest opaque pixel', s.radii[H - 1 - 10], 3 * 1.5 / W);
  ok('empty rows reach nothing', s.radii[H - 1], 0);
  const at = { x: 0, y: 0.75, z: 0 };
  const turned = profileCells({ ...at, ...s });
  const boxed = bodyCells([-0.75, 0.75, -0.75, 0.75, 3.75, 0.75]);
  ok('under half the cells of its box', turned.size < boxed.size / 2, true);
  ok('the legs: one column', [...turned].filter(k => k === cellKey(0, 2, 0) || k === cellKey(1, 2, 0)).length, 1);
  ok('above the head is free', turned.has(cellKey(0, 7, 0)), false);
  // Off a cell's centre (walking between two), a thin part touches both cells it straddles.
  const mid = profileCells({ x: 0.25, y: 0.75, z: 0, ...s });
  ok('between two cells: both', mid.has(cellKey(0, 2, 0)) && mid.has(cellKey(1, 2, 0)), true);
  // Smoke fills round it: the cloud keeps its volume, none in the figure.
  const cl = cast(tile, ground, fog, 1, undefined, { blocked: turned });
  ok('smoke fills round the figure', cl.cubes.length === cl.volume && !cl.cubes.some(q => turned.has(cellKey(q[4], q[5], q[6]))), true);
}

section('cubes grow in and shrink out as they come and go');
{
  const { reshapeSmokeCloud, pruneFading, bodyCells, SMOKE_CAST_BORN, SMOKE_GROW_IN } = await import('../js/core/smoke.js');
  const cl = createSmokeCloud(tile, ground);
  ok('a cast\'s cubes grow with the roll-out', cl.cubes.every(q => q[11] === SMOKE_CAST_BORN), true);
  const body = bodyCells([2.25, 0.75, -0.75, 3.75, 3.75, 0.75]);
  reshapeSmokeCloud(cl, body, null, 10);
  ok('the cubes that went shrink out, died now', cl.fading.length > 0 && cl.fading.every(f => f.died === 10), true);
  ok('the cubes that came are born now', cl.cubes.filter(q => q[11] === 10).length, cl.fading.length);
  ok('their neighbours keep their birth', cl.cubes.filter(q => q[11] === SMOKE_CAST_BORN).length, cl.volume - cl.fading.length);
  ok('still shrinking: kept', pruneFading(cl, 10 + SMOKE_GROW_IN / 2), false);
  ok('done shrinking: dropped', pruneFading(cl, 10 + SMOKE_GROW_IN) && cl.fading.length === 0, true);
}

section('a body moving through smoke: the rim holds still until it settles');
{
  const { reshapeSmokeCloud, bodyCells, cellKey } = await import('../js/core/smoke.js');
  const cl = createSmokeCloud(tile, ground);
  const of = c => new Set(c.cubes.map(q => cellKey(q[4], q[5], q[6])));
  const cast = of(cl);
  // Straddling and centred in turn - a column, then two - as a body walking does.
  let rimMoved = false, inBody = false, t = 0;
  for (let x = -4; x <= 4; x += 0.25) {
    const body = bodyCells([x - 0.4, 0.75, -0.4, x + 0.4, 3.75, 0.4]);
    reshapeSmokeCloud(cl, body, null, t += 0.1, false);
    if ([...of(cl)].some(k => !cast.has(k))) rimMoved = true;
    if ([...of(cl)].some(k => body.has(k))) inBody = true;
  }
  ok('while it moves, no cell past the cast\'s shape', rimMoved, false);
  ok('... and none where it stands', inBody, false);
  ok('... short by what it holds', cl.cubes.length < cl.volume, true);
  const stand = bodyCells([3.6, 0.75, -0.4, 4.4, 3.75, 0.4]);
  reshapeSmokeCloud(cl, stand, null, t += 1, true);
  ok('settled: the volume again, the rim grown once', cl.cubes.length === cl.volume && [...of(cl)].some(k => !cast.has(k)), true);
  reshapeSmokeCloud(cl, null, null, t += 1, false);
  ok('moving off: the hole refills at once, the grown rim waits', [...cast].every(k => of(cl).has(k)) && cl.cubes.length > cl.volume, true);
  reshapeSmokeCloud(cl, null, null, t += 1, true);
  ok('settled with nothing there: the cast shape exactly', of(cl).size === cast.size && [...of(cl)].every(k => cast.has(k)), true);
}

section('a cube a newcomer covers stays seen until it has grown over it');
{
  const { reshapeSmokeCloud, expireCovered, bodyCells, cellKey, SMOKE_GROW_IN } = await import('../js/core/smoke.js');
  const cl = createSmokeCloud(tile, ground);
  const body = bodyCells([-0.4, 0.75, -0.4, 0.4, 3.75, 0.4]);
  reshapeSmokeCloud(cl, body, null, 5, false);              // a hole: its walls are seen
  const walls = cl.cubes.filter(q => q[7]).length;
  reshapeSmokeCloud(cl, null, null, 6, false);              // it fills: the walls are covered again
  const kept = cl.cubes.filter(q => q[12] > 6);
  ok('the hole\'s walls stay seen while the fill grows', kept.length > 0 && kept.every(q => !q[7]), true);
  ok('... until it has grown', kept.every(q => q[12] > 6 + SMOKE_GROW_IN), true);
  ok('before then nothing expires', expireCovered(cl, 6.1), false);
  ok('after, they go back to unseen', expireCovered(cl, 7) && cl.cubes.every(q => !q[12]), true);
  ok('more than the walls were not kept', kept.length <= walls, true);
}
