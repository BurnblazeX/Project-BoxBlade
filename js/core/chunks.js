// --- The world, stored by chunk (World plan P3, Gameplay §3.3) ---
//
// Until 2026-10-28 World was a Map from a "x,y,z" string to a block object:
// fine for the Dev Level's 2,270 blocks, about 150 bytes a block, and every
// question about a box of the world meant walking all of it. A region of
// 512 x 512 blocks holds around a million. So the world is now chunks of
// size x height x size cells (12 x 48 x 12), each a Uint16Array of block
// types, made when something is first put in them and dropped when emptied.
//
// A block TYPE is what a block is - its material, shape, facing, second half -
// interned once in a shared palette and frozen: every grass block is the same
// object. What belongs to one cell and changes in play lives beside the
// palette, never on it: who stands there (world.js occupantAt, a side table)
// and the simulation's per-chunk layers (layerAt below).
//
// It answers as the Map did - get, set, has, delete, keys, values, entries,
// iteration, size, clear, keyed by "x,y,z" - so no caller had to change; the
// hot paths use the coordinate forms (getAt, hasAt) and the windowed walk
// (forEachBlockIn), which visits only the chunks a box overlaps.
//
// Cells are indexed x + size * (z + size * (y - yMin)), as Gameplay §3.3 set
// out. Iteration runs chunk by chunk in the order chunks were made, cells in
// index order - not the Map's insertion order, which nothing relied on.

// The per-chunk layers the simulation will keep (Gameplay §3.3), one byte a
// cell each, made on the first write to them: empty until their systems come.
export const LAYERS = ['coating', 'coatingAmount', 'temperature', 'air', 'airDensity', 'fuel', 'water'];

// A block's type: every own property but the per-cell ones, in a fixed order.
const PER_CELL = new Set(['occupant']);
function signature(block) {
  return JSON.stringify(Object.keys(block).filter(k => !PER_CELL.has(k)).sort().map(k => [k, block[k]]));
}

// "x,y,z" to numbers without split and map: World.get is on paths that run
// every tick (pathfinding, movement).
function parseKey(key) {
  const a = key.indexOf(','), b = key.indexOf(',', a + 1);
  return [Number(key.slice(0, a)), Number(key.slice(a + 1, b)), Number(key.slice(b + 1))];
}

export function createChunkWorld({ size = 12, yMin, height }) {
  const yMax = yMin + height - 1;
  const cellsPer = size * size * height;
  const chunks = new Map();          // chunk key -> { cx, cz, cells, count, layers }
  const palette = [null];            // type index -> frozen block (0: empty)
  const types = new Map();           // signature -> type index
  const indexOf = new WeakMap();     // palette object -> type index: a type set back skips the signature
  let count = 0;

  const chunkKey = (cx, cz) => (cx + 32768) * 65536 + (cz + 32768);
  const floorDiv = v => Math.floor(v / size);
  const cellOf = (x, y, z) => (x - floorDiv(x) * size) + size * ((z - floorDiv(z) * size) + size * (y - yMin));
  const inY = y => Number.isInteger(y) && y >= yMin && y <= yMax;

  function intern(block) {
    const known = indexOf.get(block);
    if (known !== undefined) return known;
    const sig = signature(block);
    let t = types.get(sig);
    if (t === undefined) {
      const copy = {};
      for (const k of Object.keys(block)) if (!PER_CELL.has(k)) copy[k] = block[k];
      t = palette.length;
      if (t > 65535) throw new Error('chunks: more than 65535 kinds of block');
      palette.push(Object.freeze(copy));
      types.set(sig, t);
      indexOf.set(copy, t);
    }
    return t;
  }

  function chunkFor(x, z, make) {
    const cx = floorDiv(x), cz = floorDiv(z), k = chunkKey(cx, cz);
    let c = chunks.get(k);
    if (!c && make) {
      c = { cx, cz, cells: new Uint16Array(cellsPer), count: 0, layers: {} };
      chunks.set(k, c);
    }
    return c;
  }

  function getAt(x, y, z) {
    if (!inY(y)) return undefined;
    const c = chunkFor(x, z, false);
    if (!c) return undefined;
    const t = c.cells[cellOf(x, y, z)];
    return t ? palette[t] : undefined;
  }

  function setAt(x, y, z, block) {
    if (!inY(y)) throw new Error(`chunks: y ${y} is outside ${yMin}..${yMax}`);
    const c = chunkFor(x, z, true);
    const i = cellOf(x, y, z);
    if (!c.cells[i]) { c.count++; count++; }
    c.cells[i] = intern(block);
  }

  function deleteAt(x, y, z) {
    if (!inY(y)) return false;
    const c = chunkFor(x, z, false);
    if (!c) return false;
    const i = cellOf(x, y, z);
    if (!c.cells[i]) return false;
    c.cells[i] = 0;
    c.count--; count--;
    if (c.count === 0 && Object.keys(c.layers).length === 0) chunks.delete(chunkKey(c.cx, c.cz));
    return true;
  }

  // Every block of one chunk: fn(x, y, z, block).
  function eachInChunk(c, fn) {
    const x0 = c.cx * size, z0 = c.cz * size;
    const cells = c.cells;
    for (let i = 0; i < cellsPer; i++) {
      const t = cells[i];
      if (!t) continue;
      const lx = i % size, lz = Math.floor(i / size) % size, ly = Math.floor(i / (size * size));
      fn(x0 + lx, yMin + ly, z0 + lz, palette[t]);
    }
  }

  // [[x, y, z], block] for the chunks given, cells in index order.
  function* cellsOf(list, box) {
    for (const c of list) {
      const x0 = c.cx * size, z0 = c.cz * size;
      for (let i = 0; i < cellsPer; i++) {
        const t = c.cells[i];
        if (!t) continue;
        const x = x0 + (i % size), z = z0 + (Math.floor(i / size) % size);
        if (box && (x < box[0] || x > box[1] || z < box[2] || z > box[3])) continue;
        yield [[x, yMin + Math.floor(i / (size * size)), z], palette[t]];
      }
    }
  }
  // The chunks a box overlaps, those that exist.
  function chunksIn(x0, x1, z0, z1) {
    const out = [];
    for (let cx = floorDiv(x0); cx <= floorDiv(x1); cx++) {
      for (let cz = floorDiv(z0); cz <= floorDiv(z1); cz++) {
        const c = chunks.get(chunkKey(cx, cz));
        if (c) out.push(c);
      }
    }
    return out;
  }

  function* entries() {
    for (const c of chunks.values()) {
      const x0 = c.cx * size, z0 = c.cz * size;
      for (let i = 0; i < cellsPer; i++) {
        const t = c.cells[i];
        if (!t) continue;
        const x = x0 + (i % size), z = z0 + (Math.floor(i / size) % size), y = yMin + Math.floor(i / (size * size));
        yield [`${x},${y},${z}`, palette[t]];
      }
    }
  }

  // --- The per-chunk layers ---
  function layerAt(name, x, y, z) {
    if (!inY(y)) return 0;
    const c = chunkFor(x, z, false);
    const l = c && c.layers[name];
    return l ? l[cellOf(x, y, z)] : 0;
  }
  function setLayerAt(name, x, y, z, value) {
    if (!LAYERS.includes(name)) throw new Error(`chunks: no layer ${name}`);
    if (!inY(y)) throw new Error(`chunks: y ${y} is outside ${yMin}..${yMax}`);
    const c = chunkFor(x, z, true);
    (c.layers[name] ||= new Uint8Array(cellsPer))[cellOf(x, y, z)] = value;
  }

  return {
    // --- As the Map answered ---
    get: key => { const [x, y, z] = parseKey(key); return getAt(x, y, z); },
    has: key => { const [x, y, z] = parseKey(key); return getAt(x, y, z) !== undefined; },
    set(key, block) { const [x, y, z] = parseKey(key); setAt(x, y, z, block); return this; },
    delete: key => { const [x, y, z] = parseKey(key); return deleteAt(x, y, z); },
    clear() { chunks.clear(); count = 0; },
    get size() { return count; },
    entries,
    *keys() { for (const [k] of entries()) yield k; },
    *values() { for (const [, b] of entries()) yield b; },
    [Symbol.iterator]: entries,
    forEach(fn) { for (const [k, b] of entries()) fn(b, k, this); },

    // --- By coordinates, and by box ---
    getAt, setAt, deleteAt,
    hasAt: (x, y, z) => getAt(x, y, z) !== undefined,
    forEachBlock(fn) { for (const c of chunks.values()) eachInChunk(c, fn); },
    // Blocks with x0 <= x <= x1 and z0 <= z <= z1, visiting only the chunks
    // that box overlaps.
    forEachBlockIn(x0, x1, z0, z1, fn) {
      for (let cx = floorDiv(x0); cx <= floorDiv(x1); cx++) {
        for (let cz = floorDiv(z0); cz <= floorDiv(z1); cz++) {
          const c = chunks.get(chunkKey(cx, cz));
          if (c) eachInChunk(c, (x, y, z, b) => { if (x >= x0 && x <= x1 && z >= z0 && z <= z1) fn(x, y, z, b); });
        }
      }
    },
    // [[x, y, z], block] over a box of columns (x0..x1, z0..z1, inclusive),
    // or the whole world with no box - for the windowed builders (blocksIn).
    cellsIn(box = null) {
      return cellsOf(box ? chunksIn(box[0], box[1], box[2], box[3]) : chunks.values(), box);
    },
    layerAt, setLayerAt,
    // What it holds: chunks, blocks, kinds of block, and the bytes the cells
    // and layers take.
    stats() {
      let layerBytes = 0;
      for (const c of chunks.values()) for (const l of Object.values(c.layers)) layerBytes += l.byteLength;
      return { chunks: chunks.size, blocks: count, types: palette.length - 1,
               cellBytes: chunks.size * cellsPer * 2, layerBytes };
    },
    // The chunk geometry, for anything that walks by chunk.
    shape: { size, yMin, yMax, height }
  };
}

// [[x, y, z], block] for every block in a box of columns - box [x0, x1, z0,
// z1], inclusive; null for all of them - from a chunk world's windowed walk,
// or from any Map of "x,y,z" keys (the tests build those).
export function* blocksIn(world, box = null) {
  if (world.cellsIn) { yield* world.cellsIn(box); return; }
  for (const [key, block] of world.entries()) {
    const c = key.split(',').map(Number);
    if (box && (c[0] < box[0] || c[0] > box[1] || c[2] < box[2] || c[2] > box[3])) continue;
    yield [c, block];
  }
}
