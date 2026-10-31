import { file, section, ok, truthy, throws, note } from './lib/harness.mjs';
import { createChunkWorld, blocksIn, LAYERS } from '../js/core/chunks.js';
import { World, newBlock, getVoxelKey, occupantAt, setOccupantAt, Y_MIN, Y_MAX, CHUNK_HEIGHT,
         CHUNK_SIZE } from '../js/core/world.js';
import { materialProps, PROP_KEYS, PROP_MATERIALS, DEFAULT_PROPS } from '../js/core/matprops.js';
import { MATERIALS } from '../js/core/materials.js';

file('chunks.test.mjs - the world stored by chunk (World plan P3)');

const make = () => createChunkWorld({ size: CHUNK_SIZE, yMin: Y_MIN, height: CHUNK_HEIGHT });

section('answers as the Map did');
{
  const w = make(), m = new Map();
  // Across chunk borders, both signs, every height.
  const cells = [[0, 0, 0], [-1, 0, -1], [11, Y_MAX, 12], [-13, Y_MIN, 5], [25, 7, -40], [12, 0, 0]];
  cells.forEach(([x, y, z], i) => {
    const b = newBlock(i % 2 ? 'grass' : 'marble', i === 3 ? 'slab' : null);
    w.set(`${x},${y},${z}`, b);
    m.set(`${x},${y},${z}`, b);
  });
  ok('size', w.size, m.size);
  ok('every key reads back the same block', cells.every(([x, y, z]) => {
    const k = `${x},${y},${z}`;
    return JSON.stringify(w.get(k)) === JSON.stringify((({ occupant, ...b }) => b)(m.get(k)));
  }), true);
  ok('the same keys', [...w.keys()].sort().join(' '), [...m.keys()].sort().join(' '));
  ok('a missing cell is undefined', w.get('3,3,3'), undefined);
  ok('has', w.has('25,7,-40') && !w.has('25,8,-40'), true);
  ok('getAt agrees', w.getAt(-13, Y_MIN, 5), w.get(`-13,${Y_MIN},5`));
  ok('delete', w.delete('0,0,0') && !w.has('0,0,0') && w.size === m.size - 1, true);
  ok('deleting nothing is false', w.delete('0,0,0'), false);
  throws('above the world is refused', () => w.set(`0,${Y_MAX + 1},0`, newBlock('grass')));
  ok('... and reads as nothing', w.get(`0,${Y_MAX + 1},0`), undefined);
  w.clear();
  ok('clear', `${w.size} ${[...w.keys()].length}`, '0 0');
}

section('block types are shared and frozen');
{
  const w = make();
  w.set('0,0,0', newBlock('grass'));
  w.set('5,3,9', newBlock('grass'));
  w.set('6,3,9', newBlock('grass', 'slope', 'N'));
  ok('two grass blocks are one object', w.get('0,0,0') === w.get('5,3,9'), true);
  ok('a slope is its own type', w.get('6,3,9') !== w.get('0,0,0'), true);
  ok('three blocks, two types', w.stats().types, 2);
  truthy('frozen: a write to one cannot reach every grass block', Object.isFrozen(w.get('0,0,0')));
  ok('the occupant is not part of the type', 'occupant' in w.get('0,0,0'), false);
}

section('occupancy is its own table');
{
  World.clear();
  World.set('1,0,1', newBlock('grass'));
  World.set('2,0,1', newBlock('grass'));
  setOccupantAt('1,0,1', 'a');
  ok('one cell held, its twin type free', `${occupantAt('1,0,1')} ${occupantAt('2,0,1')}`, 'a null');
  World.delete('1,0,1');
  ok('emptying the cell frees it', occupantAt('1,0,1'), null);
  setOccupantAt('2,0,1', 'b');
  World.clear();
  ok('clearing the world frees every cell', occupantAt('2,0,1'), null);
}

section('windowed walks');
{
  const w = make();
  for (let x = -30; x <= 30; x += 3) for (let z = -30; z <= 30; z += 3) w.set(`${x},0,${z}`, newBlock('grass'));
  const box = [-7, 8, 0, 13];
  const inBox = [...blocksIn(w, box)].map(([c]) => c.join()).sort();
  const expect = [...w.keys()].filter(k => { const [x, , z] = k.split(',').map(Number);
    return x >= box[0] && x <= box[1] && z >= box[2] && z <= box[3]; }).sort();
  ok('a box gets exactly the blocks in it', inBox.join(' '), expect.join(' '));
  const viaMap = [...blocksIn(new Map([...w.entries()]), box)].map(([c]) => c.join()).sort();
  ok('a plain Map gets the same', viaMap.join(' '), expect.join(' '));
  let n = 0;
  w.forEachBlockIn(box[0], box[1], box[2], box[3], () => n++);
  ok('forEachBlockIn agrees', n, expect.length);
}

section('layers');
{
  const w = make();
  w.set('4,2,4', newBlock('grass'));
  ok('a layer reads 0 before any write', w.layerAt('water', 4, 2, 4), 0);
  w.setLayerAt('water', 4, 2, 4, 144);
  ok('and back after', w.layerAt('water', 4, 2, 4), 144);
  ok('only the cell written', w.layerAt('water', 5, 2, 4), 0);
  w.setLayerAt('air', 40, 10, 40, 3);
  ok('a layer can sit where no block is (air, water)', w.layerAt('air', 40, 10, 40), 3);
  throws('an unknown layer is refused', () => w.setLayerAt('lava', 0, 0, 0, 1));
  ok('the layers Gameplay §3.3 names', LAYERS.join(), 'coating,coatingAmount,temperature,air,airDensity,fuel,water');
}

section('material properties');
{
  ok('every built-in material has its own row', MATERIALS.every(m => PROP_MATERIALS.includes(m.id)), true);
  const inRange = id => PROP_KEYS.every(k => { const v = materialProps(id)[k];
    return Number.isInteger(v) && v >= 0 && v <= (k === 'conductive' ? 1 : 7); });
  ok('all 3-bit, conductive 1-bit', [...PROP_MATERIALS, 'nonesuch'].every(inRange), true);
  ok('an unknown material takes the defaults', materialProps('nonesuch'), DEFAULT_PROPS);
  ok('iron conducts, grass does not', `${materialProps('ironPlate').conductive}${materialProps('grass').conductive}`, '10');
}

section('a million blocks (a 512-block region)');
{
  const w = make();
  const t0 = performance.now();
  // 512 x 512 columns, a surface block and two under it, varied a little.
  const types = [newBlock('grass'), newBlock('marble'), newBlock('grass', 'slab')];
  for (let x = -256; x < 256; x++) for (let z = -256; z < 256; z++) {
    const top = ((x * 7 + z * 13) & 7) - 2;
    for (let y = top - 2; y <= top; y++) w.setAt(x, y, z, types[(x + y + z) & 1 ? 0 : (y === top ? 2 : 1)]);
  }
  const built = performance.now() - t0;
  const s = w.stats();
  ok('786,432 blocks held', s.blocks, 512 * 512 * 3);
  ok('three types', s.types, 3);
  const t1 = performance.now();
  let n = 0;
  w.forEachBlockIn(-64, 63, -64, 63, () => n++);
  const window = performance.now() - t1;
  ok('a 128-block window walks only its chunks', n, 128 * 128 * 3);
  note(`built in ${built.toFixed(0)} ms; cells ${(s.cellBytes / 1048576).toFixed(1)} MB in ${s.chunks} chunks ` +
       `(a Map held ~150 B a block: ~${(s.blocks * 150 / 1048576).toFixed(0)} MB); 128-block window walked in ${window.toFixed(1)} ms`);
}
