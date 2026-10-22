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
