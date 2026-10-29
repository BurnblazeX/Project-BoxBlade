import { file, section, ok, truthy, throws, note } from './lib/harness.mjs';
import { readFileSync } from 'node:fs';
import { migrateLevel, levelRows, packRegion, unpackRegion, encodeRegion, decodeRegion,
         LEVEL_FORMAT, LEVEL_VERSION } from '../js/core/region.js';
import { World, Region, newBlock, serializeLevel, loadLevel, clearWorldModels, Y_MIN, Y_MAX,
         CHUNK_SIZE } from '../js/core/world.js';

file('region.test.mjs - regions on disk, level format v2 (World plan P2)');

const shape = World.shape;
const v1 = {
  format: LEVEL_FORMAT, version: 1,
  blocks: [[0, 0, 0, 'grass'], [1, 0, 0, 'grass'], [-1, Y_MIN, -13, 'marble'], [30, Y_MAX, 5, 'grass', 'slope', 'S'],
           [5, 1, 5, 'marble', 'halfBottom', null, 'glass']],
  models: [['decor_chair', 0, 0, 0, 1]]
};
const sorted = rows => rows.map(r => JSON.stringify(r)).sort().join(' ');

section('v1 levels migrate');
{
  const level = migrateLevel(v1, shape);
  ok('to version 2', level.version, LEVEL_VERSION);
  ok('bounds from the blocks', JSON.stringify(level.region.bounds), '[-1,30,-13,5]');
  ok('no spawns, no exits', JSON.stringify([level.spawns, level.exits]), '[{},[]]');
  ok('a palette of the kinds of block, 0 empty', level.palette.length, 5);
  ok('a chunk per column of chunks used', level.chunks.length, 3);
  ok('every row comes back', sorted(levelRows(level).blocks), sorted(v1.blocks));
  ok('the models kept', JSON.stringify(level.models), JSON.stringify(v1.models));
  throws('a block outside the vertical range is refused',
         () => migrateLevel({ format: LEVEL_FORMAT, blocks: [[0, Y_MAX + 1, 0, 'grass']] }, shape));
  throws('a newer version is refused', () => migrateLevel({ format: LEVEL_FORMAT, version: 99, blocks: [] }, shape));
  throws('a file that is not a level is refused', () => migrateLevel({ blocks: [] }, shape));
}

section('bytes: packed, gzipped and back');
{
  const level = migrateLevel({ ...v1, region: { name: 'test', bounds: [-20, 40, -20, 40], seaLevel: -2 },
                               spawns: { start: [0, 1, 0] }, exits: [{ cells: [1, 1, 0, 1, 1, 0], target: 'b', spawn: 'start' }] }, shape);
  const back = unpackRegion(packRegion(level));
  ok('the header comes back', JSON.stringify([back.region, back.spawns, back.exits, back.models]),
     JSON.stringify([level.region, level.spawns, level.exits, level.models]));
  ok('authored bounds wider than the blocks are kept', JSON.stringify(back.region.bounds), '[-20,40,-20,40]');
  ok('the blocks come back', sorted(levelRows(back).blocks), sorted(v1.blocks));
  const gz = await encodeRegion(level);
  ok('a .bxr is gzip', gz[0] === 0x1f && gz[1] === 0x8b, true);
  ok('and decodes to the same', sorted(levelRows(await decodeRegion(gz)).blocks), sorted(v1.blocks));
  ok('bytes already inflated are read as they are', (await decodeRegion(packRegion(level))).chunks.length, level.chunks.length);
  throws('a file that is not a region is refused', () => unpackRegion(new TextEncoder().encode('{"format":"x"}')));
  const bad = packRegion(level);
  bad[bad.length - 1] = 0xff;            // the last run's type, past the palette
  throws('a corrupt chunk is refused', () => unpackRegion(bad));
}

section('another chunk shape converts');
{
  const level = migrateLevel(v1, { size: 16, yMin: -8, height: 48 });
  ok('chunked at 16', level.chunk.size, 16);
  const back = migrateLevel(level, shape);
  ok('and back to the world\'s', back.chunk.size, CHUNK_SIZE);
  ok('the same blocks', sorted(levelRows(back).blocks), sorted(v1.blocks));
}

section('the world: saved and loaded through a .bxr');
{
  clearWorldModels();
  const missing = loadLevel(v1, new Map(), 'fallback');
  ok('the chair is reported missing (no models given)', JSON.stringify(missing), '["decor_chair"]');
  ok('the name falls back to the one given', Region.name, 'fallback');
  ok('every block in the world', World.size, v1.blocks.length);
  ok('a slope keeps its shape and facing', JSON.stringify([World.getAt(30, Y_MAX, 5).shape, World.getAt(30, Y_MAX, 5).facing]),
     '["slope","S"]');
  ok('a dual block its second material', World.getAt(5, 1, 5).second, 'glass');
  World.set('60,0,60', newBlock('grass'));
  const saved = serializeLevel();
  ok('bounds grow to a block placed outside them', JSON.stringify(saved.region.bounds), '[-1,60,-13,60]');
  ok('the palette holds only types in use', saved.palette.length, 5);
  World.delete('60,0,60');
  const before = sorted(levelRows(serializeLevel()).blocks);
  const bytes = await encodeRegion(serializeLevel());
  World.clear();
  loadLevel(await decodeRegion(bytes), new Map());
  ok('a round trip through the file leaves the world the same', sorted(levelRows(serializeLevel()).blocks), before);
  ok('emptied chunks are not saved', serializeLevel().chunks.length, 3);
}

section('the Dev Level');
{
  const url = new URL('../levels/dev_level.bxr', import.meta.url);
  let bytes = null;
  try { bytes = readFileSync(url); } catch { /* not converted yet */ }
  truthy('levels/dev_level.bxr exists', bytes);
  if (bytes) {
    const level = await decodeRegion(new Uint8Array(bytes));
    ok('is named', level.region.name, 'dev_level');
    truthy('holds its blocks', levelRows(level).blocks.length > 2000);
    note(`${bytes.length} bytes, ${level.chunks.length} chunks, ${level.palette.length - 1} kinds of block`);
  }
}

section('a large region: size and speed');
{
  // 384 x 384 blocks of ground four to six deep, hills, a road, a building
  // every 24 blocks, and the surface a random mix of four materials, cell by
  // cell - worse for runs than any painted map: ~700k blocks.
  const hash = (x, z) => { let h = Math.imul(x, 374761393) + Math.imul(z, 668265263); h = Math.imul(h ^ (h >>> 13), 1274126177); return (h ^ (h >>> 16)) >>> 0; };
  const top = ['grass', 'grass', 'dirt', 'gravel'];
  const blocks = [];
  for (let x = 0; x < 384; x++) {
    for (let z = 0; z < 384; z++) {
      const h = Math.round(Math.sin(x / 40) * Math.cos(z / 50) * 2) + 2;
      for (let y = -3; y <= h - 2; y++) blocks.push([x, y, z, y === h - 2 ? (z % 64 < 3 ? 'path' : top[hash(x, z) & 3]) : 'dirt']);
      if (x % 24 < 6 && z % 24 < 6) for (let y = h - 1; y <= h + 2; y++) blocks.push([x, y, z, 'stone']);
    }
  }
  const level = migrateLevel({ format: LEVEL_FORMAT, blocks }, shape);
  let t = performance.now();
  const gz = await encodeRegion(level);
  const encodeMs = performance.now() - t;
  t = performance.now();
  const back = await decodeRegion(gz);
  const decodeMs = performance.now() - t;
  const json = JSON.stringify({ format: LEVEL_FORMAT, version: 1, blocks }).length;
  note(`${blocks.length} blocks: ${(gz.length / 1024).toFixed(0)} KB as a .bxr against ${(json / 1048576).toFixed(1)} MB of v1 JSON; ` +
       `encode ${encodeMs.toFixed(0)} ms, decode ${decodeMs.toFixed(0)} ms`);
  truthy('under a megabyte', gz.length < 1 << 20);
  ok('every chunk back', back.chunks.length, level.chunks.length);
  t = performance.now();
  World.clear();
  loadLevel(back, new Map());
  const loadMs = performance.now() - t;
  note(`into the world in ${loadMs.toFixed(0)} ms`);
  ok('every block in the world', World.size, blocks.length);
  truthy('decoded and loaded in under a second', decodeMs + loadMs < 1000);
  World.clear();
}

section('regions: A, then B, then A again is A loaded fresh (World plan P2 §4.5)');
{
  const { WorldModels, ModelBoxes, occupantAt, regionSpawn } = await import('../js/core/world.js');
  const { parseBBModel } = await import('../js/core/bbmodel.js');
  const { readdirSync } = await import('node:fs');
  const dir = new URL('../assets/objects/', import.meta.url);
  const models = new Map(readdirSync(dir).filter(f => f.endsWith('.bbmodel')).map(f => {
    const n = f.replace('.bbmodel', '');
    return [n, parseBBModel(readFileSync(new URL(f, dir), 'utf8'), n)];
  }));
  const read = async n => decodeRegion(new Uint8Array(readFileSync(new URL(`../levels/${n}.bxr`, import.meta.url))));
  const a = await read('dev_level'), b = await read('test_b');
  const state = () => JSON.stringify({
    blocks: levelRows(serializeLevel()).blocks, region: Region,
    models: WorldModels.map(m => [m.name, m.x, m.y, m.z, m.yaw, m.wall]), boxes: ModelBoxes,
    occupied: WorldModels.map(m => occupantAt(`${m.tile[0]},${m.tile[1]},${m.tile[2]}`))
  });
  World.clear(); clearWorldModels();
  ok('A loads with every model found', loadLevel(a, models, 'dev_level').length, 0);
  const fresh = state();
  ok('B loads with every model found', loadLevel(b, models, 'test_b').length, 0);
  ok('B is B: its region', Region.name, 'test_b');
  ok('nothing of A is left in B', World.size, 26154);
  ok('B\'s spawn by name', JSON.stringify(regionSpawn('hut')), '[19,0,15]');
  ok('an unknown spawn is the start', JSON.stringify(regionSpawn('nowhere')), '[12,0,30]');
  loadLevel(a, models, 'dev_level');
  ok('A again is A fresh: blocks, region, models, their boxes and tiles', state(), fresh);
  World.clear(); clearWorldModels();
}
