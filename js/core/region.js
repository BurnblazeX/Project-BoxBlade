// --- Regions on disk: level format v3 (World plan P2 §4.1) ---
//
// A level was JSON with a row per block, [x, y, z, materialId, ...]: 47 KB for
// the Dev Level, but 15-20 MB for a 512 x 512 region a few layers deep, and a
// parse that makes an object per block. A region is now one .bxr file:
//
//   gzip of:
//     'BXR' 0          magic
//     u32              header length in bytes
//     header           UTF-8 JSON (below)
//     per chunk:       i16 cx, i16 cz, u16 runs, then runs x (u16 length, u16 type)
//
// all little-endian. The header holds everything but the blocks:
//   { format: 'boxblade-level', version: 3,
//     region:  { name, bounds: [x0, x1, z0, z1] inclusive, seaLevel },
//     spawns:  { name: [x, y, z] },
//     exits:   [{ cells: [x0, y0, z0, x1, y1, z1], target, spawn }] (core/exits.js),
//     models:  as v1 ([name, x, y, z, yaw] or [name, x, y, z, 0, wall]),
//     chunk:   { size, yMin, height },
//     palette: [null, [part, ...], ...] }
// A palette row is a tile's parts (world.js tileParts), each part
// [materialId, shape?, facing?, state?, at?]. v1 and v2 rows were one block,
// [materialId, shape?, facing?, second?], in halves: legacyParts turns them
// into parts on the Grid (a half is a slab, a third of the tile).
// A chunk's cells run in the chunk store's own order (core/chunks.js: x, then
// z, then y), so a load decodes runs straight into its Uint16Arrays, and whole
// layers of ground are a run each. Type 0 is empty; the palette is the file's
// own, so block types can be added or renamed without breaking old regions.
//
// In memory a level is that header plus chunks: [{ cx, cz, cells }], cells a
// Uint16Array of palette indices. levelRows turns it into the readable form -
// the header with v1's block rows in place of palette and chunks - which is
// what tools/region.mjs dumps, and what migrateLevel takes back, as it takes
// any v1 level. Headless and DOM-free: the dev server (vite.config.js) and the
// tools read and write regions with it too.

export const LEVEL_FORMAT = 'boxblade-level';
export const LEVEL_VERSION = 3;
const MAGIC = [0x42, 0x58, 0x52, 0x00];   // 'BXR' 0

// A part's row: [materialId, shape?, facing?, state?, at?], trailing
// defaults left off.
export function partRow({ materialId, shape, facing, state, at }) {
  const row = [materialId ?? null, shape || null, facing || null, state || null, at || 0];
  while (row.length > 1 && !row[row.length - 1]) row.pop();
  return row;
}

// A v1 or v2 block row's tail, [materialId, shape?, facing?, second?], as
// parts. Halves become a slab or a pane a Grid cell thick, at the side the
// half was on; a dual block's second material fills the other two thirds. A
// slope is a high one, a half slope a medium one, a high half slope a low one
// on two slabs - the same ends, 0 to 1 over the two. Stairs are three steps.
const SIDE = { E: [2, 'E'], W: [0, 'E'], S: [18, 'S'], N: [0, 'S'] };
const OTHER_SIDES = { E: [0, 1], W: [1, 2], S: [0, 9], N: [9, 18] };
export function legacyParts([materialId, shape, facing, second]) {
  const m = materialId ?? null, f = facing || 'E';
  const slabs = (mat, ...ats) => ats.map(at => partRow({ materialId: mat, shape: 'slab', at }));
  switch (shape) {
    case 'halfBottom': return [...slabs(m, 0), ...(second != null ? slabs(second, 3, 6) : [])];
    case 'halfTop': return [...slabs(m, 6), ...(second != null ? slabs(second, 0, 3) : [])];
    case 'halfVertical': {
      const [at, axis] = SIDE[f] || SIDE.E;
      const rest = second != null ? (OTHER_SIDES[f] || OTHER_SIDES.E).map(a =>
        partRow({ materialId: second, shape: 'pane', facing: axis, at: a })) : [];
      return [partRow({ materialId: m, shape: 'pane', facing: axis, at }), ...rest];
    }
    case 'middleVertical': {
      const ns = f === 'N' || f === 'S';
      return [partRow({ materialId: m, shape: 'pane', facing: ns ? 'S' : 'E', at: ns ? 9 : 1 })];
    }
    case 'slope': return [partRow({ materialId: m, shape: 'slope', facing: f, state: 'high' })];
    case 'halfSlope': return [partRow({ materialId: m, shape: 'slope', facing: f, state: 'medium' })];
    case 'halfSlopeHigh': return [...slabs(m, 0, 3), partRow({ materialId: m, shape: 'slope', facing: f, state: 'low', at: 6 })];
    case 'stairs': return [partRow({ materialId: m, shape: 'steps', facing: f })];
    default: return [[m]];
  }
}

const header = level => ({
  format: LEVEL_FORMAT, version: LEVEL_VERSION,
  region: { name: level.region?.name ?? null, bounds: level.region?.bounds ?? null,
            seaLevel: level.region?.seaLevel ?? null },
  spawns: level.spawns || {}, exits: level.exits || [], models: level.models || []
});

// Bounds of the cells a level's chunks hold: [x0, x1, z0, z1], or null if none.
export function chunkBounds(chunks, { size, height }) {
  const cellsPer = size * size * height;
  let b = null;
  for (const { cx, cz, cells } of chunks) {
    for (let i = 0; i < cellsPer; i++) {
      if (!cells[i]) continue;
      const x = cx * size + (i % size), z = cz * size + (Math.floor(i / size) % size);
      if (!b) b = [x, x, z, z];
      else { if (x < b[0]) b[0] = x; if (x > b[1]) b[1] = x; if (z < b[2]) b[2] = z; if (z > b[3]) b[3] = z; }
    }
  }
  return b;
}
// The union of two bounds, either possibly null.
export function unionBounds(a, b) {
  if (!a) return b ? [...b] : null;
  if (!b) return [...a];
  return [Math.min(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.max(a[3], b[3])];
}

// Any level - v1, the readable v2 or v3 form, or v2 or v3 in memory - as v3
// in memory, chunked to `shape` ({ size, yMin, height }, the world's:
// World.shape). A v1 level's bounds come from its blocks; it has no spawns or
// exits. Rows before v3 are halves, made parts (legacyParts).
export function migrateLevel(level, shape) {
  if (!level || level.format !== LEVEL_FORMAT) throw new Error('not a BoxBlade level');
  const version = level.version ?? 1;
  if (version > LEVEL_VERSION) throw new Error(`level version ${level.version} is newer than this game's ${LEVEL_VERSION}`);
  const { size, yMin, height } = shape;
  const same = level.chunk && level.chunk.size === size && level.chunk.yMin === yMin && level.chunk.height === height;
  if (level.chunks && same) {
    if (version === LEVEL_VERSION) return level;
    return { ...level, version: LEVEL_VERSION, palette: level.palette.map(r => r && legacyParts(r)) };
  }
  const rows = level.chunks ? levelRows(level).blocks : level.blocks;
  if (!Array.isArray(rows)) throw new Error('not a BoxBlade level');
  const out = { ...header(level), chunk: { size, yMin, height }, palette: [null], chunks: [] };
  const types = new Map(), chunks = new Map();
  const cellsPer = size * size * height;
  for (const [x, y, z, ...tail] of rows) {
    if (y < yMin || y >= yMin + height) throw new Error(`block at y ${y} is outside ${yMin}..${yMin + height - 1}`);
    const row = version >= 3 ? tail.map(r => partRow({ materialId: r[0], shape: r[1], facing: r[2], state: r[3], at: r[4] }))
                              : legacyParts(tail), sig = JSON.stringify(row);
    let t = types.get(sig);
    if (t === undefined) { t = out.palette.length; out.palette.push(row); types.set(sig, t); }
    const cx = Math.floor(x / size), cz = Math.floor(z / size), key = `${cx},${cz}`;
    let c = chunks.get(key);
    if (!c) { c = { cx, cz, cells: new Uint16Array(cellsPer) }; chunks.set(key, c); out.chunks.push(c); }
    c.cells[(x - cx * size) + size * ((z - cz * size) + size * (y - yMin))] = t;
  }
  out.region.bounds = unionBounds(out.region.bounds, chunkBounds(out.chunks, shape));
  return out;
}

// The readable form: the header, with a block row per block - [x, y, z,
// part, ...] (v1's [x, y, z, materialId, ...] for a level before v3).
export function levelRows(level) {
  const { size, yMin, height } = level.chunk;
  const cellsPer = size * size * height, blocks = [];
  for (const { cx, cz, cells } of level.chunks) {
    for (let i = 0; i < cellsPer; i++) {
      const t = cells[i];
      if (!t) continue;
      blocks.push([cx * size + (i % size), yMin + Math.floor(i / (size * size)),
                   cz * size + (Math.floor(i / size) % size), ...level.palette[t]]);
    }
  }
  return { ...header(level), blocks };
}

// --- Bytes ---

// The uncompressed file.
export function packRegion(level) {
  const { size, height } = level.chunk;
  const cellsPer = size * size * height;
  const head = new TextEncoder().encode(JSON.stringify({ ...header(level), chunk: level.chunk, palette: level.palette }));
  // Runs first, to size the buffer.
  const records = [];
  let bytes = 8 + head.length;
  for (const { cx, cz, cells } of level.chunks) {
    if (cells.length !== cellsPer) throw new Error('chunk cells do not match the chunk shape');
    const runs = [];
    for (let i = 0; i < cellsPer;) {
      const t = cells[i];
      let j = i + 1;
      while (j < cellsPer && cells[j] === t) j++;
      runs.push(j - i, t);
      i = j;
    }
    if (runs.length === 2 && runs[1] === 0) continue;   // all empty
    records.push(cx, cz, runs);
    bytes += 6 + runs.length * 2;
  }
  const out = new Uint8Array(bytes), view = new DataView(out.buffer);
  out.set(MAGIC, 0);
  view.setUint32(4, head.length, true);
  out.set(head, 8);
  let p = 8 + head.length;
  for (let r = 0; r < records.length; r += 3) {
    const runs = records[r + 2];
    view.setInt16(p, records[r], true);
    view.setInt16(p + 2, records[r + 1], true);
    view.setUint16(p + 4, runs.length / 2, true);
    p += 6;
    for (const v of runs) { view.setUint16(p, v, true); p += 2; }
  }
  return out;
}

export function unpackRegion(bytes) {
  if (bytes.length < 8 || MAGIC.some((m, i) => bytes[i] !== m)) throw new Error('not a BoxBlade region');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headLength = view.getUint32(4, true);
  const level = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headLength)));
  if (level.format !== LEVEL_FORMAT) throw new Error('not a BoxBlade region');
  const { size, height } = level.chunk;
  const cellsPer = size * size * height;
  level.chunks = [];
  let p = 8 + headLength;
  while (p < bytes.length) {
    const cx = view.getInt16(p, true), cz = view.getInt16(p + 2, true), runs = view.getUint16(p + 4, true);
    p += 6;
    const cells = new Uint16Array(cellsPer);
    let i = 0;
    for (let r = 0; r < runs; r++, p += 4) {
      const length = view.getUint16(p, true), t = view.getUint16(p + 2, true);
      if (i + length > cellsPer || t >= level.palette.length) throw new Error('region chunk is corrupt');
      if (t) cells.fill(t, i, i + length);
      i += length;
    }
    if (i !== cellsPer) throw new Error('region chunk is corrupt');
    level.chunks.push({ cx, cz, cells });
  }
  return level;
}

async function gzip(bytes, inflate) {
  const stream = new Blob([bytes]).stream()
    .pipeThrough(inflate ? new DecompressionStream('gzip') : new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// The .bxr file's bytes.
export async function encodeRegion(level) {
  return gzip(packRegion(level), false);
}
// From a .bxr file's bytes. Bytes already inflated - a server that sent the
// file with Content-Encoding: gzip - are taken as they are.
export async function decodeRegion(bytes) {
  bytes = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  return unpackRegion(bytes[0] === 0x1f && bytes[1] === 0x8b ? await gzip(bytes, true) : bytes);
}
