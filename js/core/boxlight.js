// --- Box light: basic mode's lighting, one answer per BoxGrid box ---
//
// The shadows-off path has no field to march, so its light is worked out on
// the BoxGrid instead (core/boxgrid.js: three boxes a block, half a metre):
// every box of air that touches a solid face - a SURFACE cell - gets one sun
// light, one sky openness and one sum of lamp light, and every face reads the
// cell in front of it. Hard and stepped by design: a shadow edge falls on a
// box boundary, a face shows three by three squares of light.
//
// Visibility is a straight walk through the boxes (Amanatides & Woo's voxel
// traversal) from the cell's centre toward the sun, toward each lamp in reach,
// and along SKY_DIRS for the sky. It starts in AIR - the cell in front of the
// face - so a face never shadows itself the way a walk from a solid voxel's
// centre does. Sprites are not boxes: their cards are tested on the GPU
// (gpu/boxlight.js), as the shaded path tests them.
//
// Per block, three u32 (buildBoxMasks):
//   0  solid - 27 bits, a box solid when its centre is (world.js isSolidAt)
//   1  glass - 27 bits, the boxes of glass; light passes, tinted
//   2  where its boxes' tints start in the tint list (glass blocks only)
// A walk entering glass from outside it takes that box's tint once - a pane
// crossed, not a thickness. Each box's tint is the mean linear colour of its
// third of the texture, so stained glass throws its panes' colours.
// A slope's box is solid only when it lies wholly under the plane, so the box
// in front of every texel of the plane is open. Placed models (world.js
// ModelBoxes) cast where a box's centre is inside one of their cubes; every
// box a cube overlaps is a surface cell, so a chair's faces have light to read.
//
// CPU and pure: the GPU kernels mirror boxRay, boxSky and boxLampAmount, and
// the tests check them here.
import { World, ModelBoxes, BLOCK_METRES, Y_MIN, Y_MAX, isSolidAt, partAt, blockShape, blockFacing, isDual,
         isSlope } from './world.js';
import { BOX_CELLS, BOX_METRES } from './boxgrid.js';
import { windowAxis } from './follow.js';
import { blocksIn } from './chunks.js';
import { isGlassMaterial } from './materials.js';
import { clampLevel, lightRadiusMetres, MAX_LIGHT_LEVEL, DEFAULT_LIGHT_CUTOFF } from './lights.js';

// The lit region's x and z, in blocks: past it, light is not worked out (a
// face out there reads the sun unshadowed and no lamps). The world's own
// bounds, one block of margin, at most this a side - 288 boxes across, and
// as tall as the levels in use (13 levels: 288 x 39 x 288, 26 MB at
// RGBA16F). A bigger map needs the region to follow the camera (World plan P2).
export const BOX_LIGHT_MAX_BLOCKS = 96;

// u32s a block in the block data.
export const BOX_BLOCK_U32 = 3;

// The sky's walks: straight up, six at 60 degrees up, six at 30 between them.
// A cell's openness is the share that leave without meeting anything solid
// (glass passes them at its mean tint), within SKY_REACH boxes.
export const SKY_RAYS = 13;
export const SKY_REACH = 64;
export function skyDir(k) {
  if (k === 0) return [0, 1, 0];
  const high = k <= 6;
  const e = (high ? 60 : 30) * Math.PI / 180;
  const a = ((high ? k - 1 : k - 7) * 60 + (high ? 0 : 30)) * Math.PI / 180;
  return [Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)];
}

// Surface cells pack into one u32: x | y << 10 | z << 20, local boxes.
const PACK_BITS = 10;
const PACK_MASK = (1 << PACK_BITS) - 1;

const FULL_MASK = 2 ** 27 - 1;

// The bit of box (i, j, k), each 0..2, in its block's mask.
export const boxBit = (i, j, k) => i + BOX_CELLS * (j + BOX_CELLS * k);

// The region over a world: blocks lo .. lo + size - 1. Across, the world's
// own bounds with a block of margin; up and down, the levels it uses with a
// block of air under the lowest and over the highest (a bottom face reads the
// box below it, a top face the one above), within Y_MIN .. Y_MAX + 1. Fitted
// rather than the whole 48 levels: 288 x 147 x 288 boxes would be ~97 MB at
// RGBA16F (World plan P1). Placed models count by the blocks their cubes
// reach, so a tall one on the highest block keeps its air. box0 is the world
// box index of local box 0 - block lo's lowest box (blocks hold boxes
// 3b-1..3b+1).
//
// A world wider than BOX_LIGHT_MAX_BLOCKS gets a window of that many around
// focus (a block position; World plan P2a, core/follow.js), kept while focus
// stays in its middle: prev is the last region's [x, z] first block. Same
// dims wherever it sits, so moving it rebuilds the cells, not the kernels.
// bounds: the world's across, for render/features/boxlight.js to tell, each
// frame, whether the window wants moving without scanning the world.
export function boxLightRegion(world = World, modelBoxes = ModelBoxes, focus = null, prev = null) {
  let x0 = Infinity, z0 = Infinity, x1 = -Infinity, z1 = -Infinity;
  let y0 = Infinity, y1 = -Infinity;
  for (const key of world.keys()) {
    const [x, y, z] = key.split(',').map(Number);
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
    if (z < z0) z0 = z; if (z > z1) z1 = z;
  }
  // A cube's texels to the blocks it spans: block b covers texels 12b - 6 ..
  // 12b + 6.
  const blockOfTexel = t => Math.floor((t + 6) / 12);
  for (const c of modelBoxes) {
    y0 = Math.min(y0, blockOfTexel(c[1]));
    y1 = Math.max(y1, blockOfTexel(c[4] - 1e-6));
  }
  if (x0 > x1) { x0 = z0 = 0; x1 = z1 = 0; }
  if (y0 > y1) { y0 = y1 = 0; }
  const ly = Math.max(Y_MIN, y0 - 1), hy = Math.min(Y_MAX + 1, y1 + 1);
  const fit = (a, b, f, p) => {
    a -= 1; b += 1;
    const n = Math.min(b - a + 1, BOX_LIGHT_MAX_BLOCKS);
    if (n === b - a + 1 || f === null || f === undefined) return [Math.round((a + b) / 2 - (n - 1) / 2), n];
    return [windowAxis(p, f, n, a, b), n];
  };
  const [lx, nx] = fit(x0, x1, focus && focus.x, prev && prev[0]);
  const [lz, nz] = fit(z0, z1, focus && focus.z, prev && prev[1]);
  const lo = [lx, ly, lz], size = [nx, hy - ly + 1, nz];
  return { lo, size, box0: lo.map(v => BOX_CELLS * v - 1), dims: size.map(v => v * BOX_CELLS),
           clipped: nx < x1 - x0 + 3 || nz < z1 - z0 + 3, bounds: { x0, x1, z0, z1 } };
}

const srgbToLinear = c => {
  const v = c / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
};

// One glass box's tint, linear rgb 0..1: the mean of its third of the face
// texture (rgba: top-down sRGB, square). Across the face, column u is box i
// or, for glass whose face is crossed along x, box k; row v is box j from
// the top. Nearly clear texels (alpha < 128) are left out, as averageAlbedo
// does; a third with none takes white.
export function glassBoxTint(rgba, i, j, k, alongX) {
  const n = Math.round(Math.sqrt(rgba.length / 4)), s = n / BOX_CELLS;
  const col = alongX ? k : i, row = BOX_CELLS - 1 - j;
  let r = 0, g = 0, b = 0, w = 0;
  for (let y = Math.floor(row * s); y < Math.ceil((row + 1) * s); y++) {
    for (let x = Math.floor(col * s); x < Math.ceil((col + 1) * s); x++) {
      const o = (y * n + x) * 4;
      if (rgba[o + 3] < 128) continue;
      r += srgbToLinear(rgba[o]); g += srgbToLinear(rgba[o + 1]); b += srgbToLinear(rgba[o + 2]); w++;
    }
  }
  return w ? [r / w, g / w, b / w] : [1, 1, 1];
}

const packTint = ([r, g, b]) => {
  const q = v => Math.max(0, Math.min(255, Math.round(v * 255)));
  return (q(r) | (q(g) << 8) | (q(b) << 16)) >>> 0;
};
export const unpackTint = v => [(v & 255) / 255, ((v >>> 8) & 255) / 255, ((v >>> 16) & 255) / 255];

// Whether a glass block's face is crossed along x: a pane facing east or west,
// or a whole block in a wall that runs along z (its neighbours are at z +- 1).
function crossedAlongX(block, c, world) {
  const shape = blockShape(block);
  if (shape === 'halfVertical' || shape === 'middleVertical') {
    const f = blockFacing(block);
    return f === 'E' || f === 'W';
  }
  const has = (dx, dz) => world.has(`${c[0] + dx},${c[1]},${c[2] + dz}`);
  return !(has(1, 0) || has(-1, 0)) && (has(0, 1) || has(0, -1));
}

// Every block's three u32, x fastest, then y, then z, the tint list, and the
// open boxes that are cells whatever is beside them (touch: packed local
// boxes) - a slope's, and those a model cube overlaps without filling.
// glassPixels(materialId): its albedo's rgba (top-down sRGB), or null while it
// has not loaded - then its boxes are tinted white. models: cubes as
// [x0, y0, z0, x1, y1, z1] in world texels (an eighth of a metre).
export function buildBoxMasks(region, world = World, glassPixels = () => null, models = ModelBoxes) {
  const [nx, ny, nz] = region.size;
  const data = new Uint32Array(nx * ny * nz * BOX_BLOCK_U32);
  const tints = [0];
  const touch = [];
  const box = [region.lo[0], region.lo[0] + nx - 1, region.lo[2], region.lo[2] + nz - 1];
  for (const [c, block] of blocksIn(world, box)) {
    const l = c.map((v, a) => v - region.lo[a]);
    if (l.some((v, a) => v < 0 || v >= region.size[a])) continue;
    const at = ((l[2] * ny + l[1]) * nx + l[0]) * BOX_BLOCK_U32;
    // Most blocks are whole and opaque: all 27 boxes solid.
    const whole = blockShape(block) === 'full' && !isDual(block);
    if (whole && !isGlassMaterial(block.materialId)) { data[at] = FULL_MASK; continue; }
    let solid = 0, glass = 0;
    // A slope's box is tested at its top: solid only under the plane throughout.
    const lift = isSlope(block) ? BOX_METRES / 2 - 1e-4 : 0;
    for (let k = 0; k < BOX_CELLS; k++) for (let j = 0; j < BOX_CELLS; j++) for (let i = 0; i < BOX_CELLS; i++) {
      // Box centres sit at -1/3, 0, 1/3 of a block from the block's.
      const wx = (c[0] + (i - 1) / BOX_CELLS) * BLOCK_METRES;
      const wy = (c[1] + (j - 1) / BOX_CELLS) * BLOCK_METRES;
      const wz = (c[2] + (k - 1) / BOX_CELLS) * BLOCK_METRES;
      if (!isSolidAt(wx, wy + lift, wz)) continue;
      const part = partAt(wx, wy, wz, world);
      const id = part ? part.materialId : block.materialId;
      if (isGlassMaterial(id)) glass |= 1 << boxBit(i, j, k);
      else solid |= 1 << boxBit(i, j, k);
    }
    data[at] = solid >>> 0;
    data[at + 1] = glass >>> 0;
    // A slope's open boxes are all cells: the gap between its plane and the
    // staircase under it can be a box with nothing solid beside it.
    if (lift) {
      for (let b = 0; b < BOX_CELLS ** 3; b++) {
        if ((solid >>> b) & 1) continue;
        const i = b % BOX_CELLS, j = Math.floor(b / BOX_CELLS) % BOX_CELLS, k = Math.floor(b / BOX_CELLS ** 2);
        touch.push(packCell(l[0] * BOX_CELLS + i, l[1] * BOX_CELLS + j, l[2] * BOX_CELLS + k));
      }
    }
    if (glass) {
      data[at + 2] = tints.length;
      const rgba = glassPixels(isGlassMaterial(block.materialId) ? block.materialId : block.second);
      const alongX = crossedAlongX(block, c, world);
      for (let b = 0; b < BOX_CELLS ** 3; b++) {
        const i = b % BOX_CELLS, j = Math.floor(b / BOX_CELLS) % BOX_CELLS, k = Math.floor(b / BOX_CELLS ** 2);
        tints.push(rgba ? packTint(glassBoxTint(rgba, i, j, k, alongX)) : packTint([1, 1, 1]));
      }
    }
  }
  // Placed models: solid where a box's centre is inside a cube, touched
  // wherever a cube overlaps it.
  const T = BLOCK_METRES / 12;
  for (const cube of models) {
    const lo = [0, 1, 2].map(a => cube[a] * T), hi = [0, 1, 2].map(a => cube[a + 3] * T);
    const from = lo.map(v => Math.floor(v / BOX_METRES - 0.5) + 1);
    const to = hi.map(v => Math.ceil(v / BOX_METRES + 0.5) - 1);
    for (let z = from[2]; z <= to[2]; z++) for (let y = from[1]; y <= to[1]; y++) for (let x = from[0]; x <= to[0]; x++) {
      const l = [x, y, z].map((v, a) => v - region.box0[a]);
      if (l.some((v, a) => v < 0 || v >= region.dims[a])) continue;
      const inside = [x, y, z].every((v, a) => v * BOX_METRES >= lo[a] && v * BOX_METRES < hi[a]);
      if (!inside) { touch.push(packCell(...l)); continue; }
      const b = l.map(v => Math.floor(v / BOX_CELLS));
      const at = ((b[2] * ny + b[1]) * nx + b[0]) * BOX_BLOCK_U32;
      data[at] = (data[at] | (1 << boxBit(l[0] - b[0] * BOX_CELLS, l[1] - b[1] * BOX_CELLS, l[2] - b[2] * BOX_CELLS))) >>> 0;
    }
  }
  return { data, tints: Uint32Array.from(tints), touch: Uint32Array.from(touch) };
}

// What local box (x, y, z) is: 0 air, 1 solid, 2 glass. Off the region is air.
export function boxKind(region, masks, x, y, z) {
  const d = region.dims;
  if (x < 0 || y < 0 || z < 0 || x >= d[0] || y >= d[1] || z >= d[2]) return 0;
  const bx = Math.floor(x / BOX_CELLS), by = Math.floor(y / BOX_CELLS), bz = Math.floor(z / BOX_CELLS);
  const at = ((bz * region.size[1] + by) * region.size[0] + bx) * BOX_BLOCK_U32;
  const bit = boxBit(x - bx * BOX_CELLS, y - by * BOX_CELLS, z - bz * BOX_CELLS);
  if ((masks.data[at] >>> bit) & 1) return 1;
  return (masks.data[at + 1] >>> bit) & 1 ? 2 : 0;
}

export const boxSolid = (region, masks, x, y, z) => boxKind(region, masks, x, y, z) === 1;

// Glass box (x, y, z)'s tint, linear rgb.
function boxTint(region, masks, x, y, z) {
  const bx = Math.floor(x / BOX_CELLS), by = Math.floor(y / BOX_CELLS), bz = Math.floor(z / BOX_CELLS);
  const at = ((bz * region.size[1] + by) * region.size[0] + bx) * BOX_BLOCK_U32;
  return unpackTint(masks.tints[masks.data[at + 2] + boxBit(x - bx * BOX_CELLS, y - by * BOX_CELLS, z - bz * BOX_CELLS)]);
}

// The surface cells, packed - boxes of air or glass beside a solid or glass
// one (a wall's face behind a pane is lit through it), and every box a model
// cube touches, and its neighbours - and
// top: one over the highest solid or glass box's y (local); a sun walk above
// it has nothing left to cross.
export function buildBoxCells(region, masks) {
  const [dx, dy, dz] = region.dims;
  const [nx, ny, nz] = region.size;
  // The masks unpacked to one byte a box, with a box of air all round, so the
  // neighbour reads need no bounds tests.
  const px = dx + 2, py = dy + 2, plane = px * py;
  const occ = new Uint8Array(plane * (dz + 2));
  let top = 0;
  for (let bz = 0; bz < nz; bz++) for (let by = 0; by < ny; by++) for (let bx = 0; bx < nx; bx++) {
    const at = ((bz * ny + by) * nx + bx) * BOX_BLOCK_U32;
    const solid = masks.data[at], glass = masks.data[at + 1];
    if (!(solid | glass)) continue;
    for (let k = 0; k < BOX_CELLS; k++) for (let j = 0; j < BOX_CELLS; j++) for (let i = 0; i < BOX_CELLS; i++) {
      const bit = boxBit(i, j, k);
      const kind = (solid >>> bit) & 1 ? 1 : (glass >>> bit) & 1 ? 2 : 0;
      if (!kind) continue;
      const y = by * BOX_CELLS + j;
      occ[(bz * BOX_CELLS + k + 1) * plane + (y + 1) * px + bx * BOX_CELLS + i + 1] = kind;
      if (y + 1 > top) top = y + 1;
    }
  }
  // A model's touched boxes: 3, open, but they make their neighbours cells.
  for (const v of masks.touch || []) {
    const [x, y, z] = unpackCell(v);
    const o = (z + 1) * plane + (y + 1) * px + x + 1;
    if (!occ[o]) occ[o] = 3;
  }
  const out = [];
  for (let z = 0; z < dz; z++) for (let y = 0; y < dy; y++) {
    let o = (z + 1) * plane + (y + 1) * px + 1;
    for (let x = 0; x < dx; x++, o++) {
      if (occ[o] === 1) continue;
      if (occ[o] === 3 || (occ[o + 1] | occ[o - 1] | occ[o + px] | occ[o - px] | occ[o + plane] | occ[o - plane])) {
        out.push(packCell(x, y, z));
      }
    }
  }
  return { cells: Uint32Array.from(out), top };
}

export const packCell = (x, y, z) => (x | (y << PACK_BITS) | (z << (2 * PACK_BITS))) >>> 0;
export const unpackCell = v => [v & PACK_MASK, (v >>> PACK_BITS) & PACK_MASK, v >>> (2 * PACK_BITS)];

// The walk from local box c's centre along dir (unit; box space is metres
// scaled). Returns the light that gets through, rgb: 0 once anything solid is
// crossed, each pane of glass entered multiplying in its box's tint. It ends
// at maxT boxes along the ray, the box `target` (a lamp's own box), y = top
// (the sun: nothing above), or the region's edge.
// The GPU's walk (gpu/boxlight.js boxRayTSL) is this, line for line.
export function boxRay(region, masks, c, dir, maxT = Infinity, top = Infinity, target = null) {
  let [x, y, z] = c;
  const s = dir.map(v => (v > 0 ? 1 : v < 0 ? -1 : 0));
  const td = dir.map(v => (Math.abs(v) > 1e-6 ? 1 / Math.abs(v) : 1e30));
  const tm = td.map(v => v * 0.5);
  const d = region.dims;
  const steps = d[0] + d[1] + d[2] + 3;
  // Starting in glass (a cell behind a pane): through it already.
  let inGlass = boxKind(region, masks, x, y, z) === 2;
  const T = inGlass ? boxTint(region, masks, x, y, z) : [1, 1, 1];
  for (let n = 0; n < steps; n++) {
    let t;
    if (tm[0] <= tm[1] && tm[0] <= tm[2]) { t = tm[0]; x += s[0]; tm[0] += td[0]; }
    else if (tm[1] <= tm[2]) { t = tm[1]; y += s[1]; tm[1] += td[1]; }
    else { t = tm[2]; z += s[2]; tm[2] += td[2]; }
    if (t >= maxT) break;
    if (target && x === target[0] && y === target[1] && z === target[2]) break;
    if (y >= top) break;
    if (x < 0 || y < 0 || z < 0 || x >= d[0] || y >= d[1] || z >= d[2]) break;
    const kind = boxKind(region, masks, x, y, z);
    if (kind === 1) return [0, 0, 0];
    if (kind === 2 && !inGlass) {
      const g = boxTint(region, masks, x, y, z);
      T[0] *= g[0]; T[1] *= g[1]; T[2] *= g[2];
    }
    inGlass = kind === 2;
  }
  return T;
}

// The sun on cell c, rgb: 0 in shadow, the glass's colour through glass.
export function boxSun(region, masks, top, c, sunDir) {
  if (sunDir[1] <= 0) return [0, 0, 0];
  return boxRay(region, masks, c, sunDir, Infinity, top);
}

// The sky over cell c, 0..1: the mean over SKY_DIRS of what gets through
// (glass at its tint's mean).
export function boxSky(region, masks, top, c) {
  let sum = 0;
  for (let k = 0; k < SKY_RAYS; k++) {
    const T = boxRay(region, masks, c, skyDir(k), SKY_REACH, top);
    sum += (T[0] + T[1] + T[2]) / 3;
  }
  return sum / SKY_RAYS;
}

// Local box centre in world metres.
export const boxCellCentre = (region, c) => c.map((v, a) => (v + region.box0[a]) * BOX_METRES);

// A lamp's light at cell c, before its colour, rgb: lights.js's falloff, cut
// as the shaded path cuts it, with no N.L (a cell has no one facing), times
// what the walk to it lets through.
export function boxLampAmount(region, masks, c, light, cutoff = DEFAULT_LIGHT_CUTOFF) {
  const level = clampLevel(light.level);
  const r = lightRadiusMetres(level);
  if (r <= 0) return [0, 0, 0];
  const p = boxCellCentre(region, c);
  const L = [light.position.x, light.position.y, light.position.z];
  const to = L.map((v, a) => v - p[a]);
  const dist = Math.max(Math.hypot(...to), 1e-4);
  const u = Math.min(1, Math.max(0, 1 - dist / r));
  const falloff = Math.log2(1 + u) * level / MAX_LIGHT_LEVEL;
  const amount = Math.max(0, falloff - cutoff) / (1 - cutoff);
  if (amount <= 0) return [0, 0, 0];
  const target = L.map((v, a) => Math.round(v / BOX_METRES) - region.box0[a]);
  return boxRay(region, masks, c, to.map(v => v / dist), dist / BOX_METRES, Infinity, target).map(v => v * amount);
}
