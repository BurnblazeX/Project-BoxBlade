// --- Interiors: the roof lifted, the room shown against the dark ---
//
// Under a roof, the building around the player opens the way Octopath
// Traveler shows a room: the roof and the storeys above are gone, and what is
// outside the building darkens, so the room stands alone. Found on the
// world's blocks, not on pixels.
//
//   the roof     the first opaque block straight above the player that is not
//                a lintel (a beam over an opening - a doorway's top, an arch:
//                air past both its ends across one axis, at its height and
//                below), and every opaque block connected to it at or above
//                that height (a flood fill, capped): the ceiling, the storeys
//                above, their walls, the roof itself - this building's,
//                however big or odd. In a doorway the roof is the building's,
//                not the top of the door with every wall above it.
//   the walls    opaque blocks under the roof's footprint (and its eaves, one
//                block out), from one block above the player's floor up to the
//                roof - the bottom row stays, a stub at the room's edge. Each
//                knows which of the room's sides it closes (wallSides): +x, -x,
//                +z, -z. The sides facing the camera are lifted, asked every
//                frame, so it follows the camera round: one side with the
//                explore camera square on, a corner's two with the battle one.
//   the inside   the columns of the room: a flood fill from the player's cell,
//                within the roof's footprint, through any column open at the
//                player's height or one above (a door, a window, past a
//                table) - its edge filled but not spread from - widened by
//                one column so its walls are inside too. Under the eaves
//                outside the walls is not under a roof at all.
//                What is outside it darkens - the view only: the lighting
//                still sees the whole world.
//
// Each cell eases toward its target on its own (easeCells), so a quarter turn
// of the camera fades one wall back in as the next fades out; a cell fully
// faded is not drawn at all (gpu/occlusion.js). A translucent roof is costly
// - two layers a pixel, and the texel cache holds one, so the other is shaded
// per pixel - so it is only ever translucent while it fades. The cells live
// in a mask texture around the player:
// INTERIOR_DIMS cells, the player's at INTERIOR_HALF, whose outermost layer
// is never written (reads outside clamp to it, so to nothing).
import { BLOCK_METRES, getBlock } from './world.js';
import { isGlassMaterial } from './materials.js';

export const INTERIOR_DIMS = [64, 32, 64];   // cells x, y, z
export const INTERIOR_HALF = [32, 8, 32];    // the player's cell within it
export const ROOF_REACH = 12;                // cells up looking for a roof
export const FILL_CAP = 20000;               // cells, so a mountain cannot stall the frame
export const FACING_COS = 0.38;              // a wall side faces the camera within ~67 degrees
export const CELL_EASE = 0.25;               // seconds for a cell to fade fully in or out
export const ROOF_DELAY = 0.25;              // under a roof this long before it fades
export const DEFAULT_INTERIOR = { dark: 0.9 };   // how far outside the building darkens, 0..1

// What a roof or a wall is made of: an opaque block. Glass is seen through already.
export function opaqueAt(x, y, z) {
  const b = getBlock(x, y, z);
  return !!b && !isGlassMaterial(b.materialId);
}

// The cell the player stands in: feet on its floor (a full block's top) or
// on a half block in it.
export function playerCell(feet) {
  const B = BLOCK_METRES;
  return { x: Math.round(feet.x / B), y: Math.floor(feet.y / B + 0.5 + 1e-3), z: Math.round(feet.z / B) };
}

// The mask's origin (its cell 0) for the player's cell, and a cell's index in
// it - x fastest, then y, then z, as a Data3DTexture lays it out - or -1
// outside its inner part.
export function maskOrigin(cell) {
  return { x: cell.x - INTERIOR_HALF[0], y: cell.y - INTERIOR_HALF[1], z: cell.z - INTERIOR_HALF[2] };
}
export function maskIndex(o, x, y, z) {
  const [W, H, D] = INTERIOR_DIMS;
  const i = x - o.x, j = y - o.y, k = z - o.z;
  if (i < 1 || j < 1 || k < 1 || i > W - 2 || j > H - 2 || k > D - 2) return -1;
  return i + W * (j + H * k);
}

const STEPS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

// A beam over an opening (see the top): across x or z it is at most two
// blocks through, with air past both ends - at its height and the one below.
function lintelAt(x, y, z, opaque) {
  for (const [dx, dz] of [[1, 0], [0, 1]]) {
    // Steps to the first air along s, within the beam's two blocks; 0: none.
    const end = s => {
      for (let k = 1; k <= 2; k++) if (!opaque(x + dx * s * k, y, z + dz * s * k)) return k;
      return 0;
    };
    const a = end(1), b = end(-1);
    if (a && b && a + b <= 3 && !opaque(x + dx * a, y - 1, z + dz * a) && !opaque(x - dx * b, y - 1, z - dz * b))
      return true;
  }
  return false;
}

// The interior around the player's cell, or null in the open:
//   { origin, roofY, roof: [index], walls: [{ i, x, z, sides }], inside: [{ x, z }] }
export function buildInterior(cell, opaque = opaqueAt) {
  let roofY = null;
  for (let y = cell.y + 1; y <= cell.y + ROOF_REACH; y++) {
    if (opaque(cell.x, y, cell.z) && !lintelAt(cell.x, y, cell.z, opaque)) { roofY = y; break; }
  }
  if (roofY === null) return null;
  const origin = maskOrigin(cell);
  // The roof: connected opaque blocks at or above its height.
  const roof = [], inRoof = new Set(), cols = new Map();   // column -> its lowest roof cell
  const start = maskIndex(origin, cell.x, roofY, cell.z);
  const queue = [[cell.x, roofY, cell.z]];
  inRoof.add(start);
  while (queue.length && roof.length < FILL_CAP) {
    const [x, y, z] = queue.pop();
    roof.push(maskIndex(origin, x, y, z));
    const c = `${x},${z}`;
    cols.set(c, Math.min(cols.get(c) ?? Infinity, y));
    for (const [dx, dy, dz] of STEPS) {
      const nx = x + dx, ny = y + dy, nz = z + dz;
      if (ny < roofY) continue;
      const i = maskIndex(origin, nx, ny, nz);
      if (i < 0 || inRoof.has(i) || !opaque(nx, ny, nz)) continue;
      inRoof.add(i);
      queue.push([nx, ny, nz]);
    }
  }
  // The walls: under the footprint and its eaves, above the stub, below the roof.
  let top = roofY;
  for (const y of cols.values()) top = Math.max(top, y);
  const walls = [], seen = new Set();
  for (const c of cols.keys()) {
    const [cx, cz] = c.split(',').map(Number);
    for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) {
      const x = cx + dx, z = cz + dz, key = `${x},${z}`;
      if (seen.has(key)) continue;
      seen.add(key);
      const ceiling = cols.get(key) ?? top;
      for (let y = cell.y + 1; y < ceiling; y++) {
        const i = maskIndex(origin, x, y, z);
        if (i >= 0 && !inRoof.has(i) && opaque(x, y, z)) walls.push({ i, x, z, sides: 0 });
      }
    }
  }
  const room = insideColumns(cell, cols, opaque);
  if (!room) return null;
  const wallCols = new Set(walls.map(w => `${w.x},${w.z}`));
  // The room's floor: the fill without the columns walls stand in - a
  // window's (the fill goes through it, over its sill) or a door's, under its
  // top. Counted as room, the wall beside a window would close the room
  // toward the camera, and lift.
  const floor = new Set([...room.core].filter(k => !wallCols.has(k)));
  const sidesOf = new Map();
  for (const w of walls) {
    const key = `${w.x},${w.z}`;
    if (!sidesOf.has(key)) sidesOf.set(key, wallSides(w.x, w.z, floor, wallCols));
    w.sides = sidesOf.get(key);
  }
  return { origin, roofY, roof, walls, inside: room.inside };
}

// The room's sides, as bits: +x, -x, +z, -z - the way out through that wall.
export const SIDE_DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];

// Which sides of the room a wall column closes (bits of SIDE_DIRS): out past
// it, through any thickness, is not the room, and in from it is - straight in
// through up to three blocks of wall, or diagonally at a corner, where the
// wall ends. A pillar in the room has the room past it too: none.
// core: the room's floor columns ('x,z') - its fill without the footprint's edge or walls.
function wallSides(x, z, core, wallCols) {
  const key = (a, b) => `${a},${b}`;
  let sides = 0;
  SIDE_DIRS.forEach(([dx, dz], bit) => {
    let k = 1;
    while (k <= 3 && wallCols.has(key(x + dx * k, z + dz * k))) k++;
    if (core.has(key(x + dx * k, z + dz * k))) return;
    for (let j = 1; j <= 3; j++) {
      const c = key(x - dx * j, z - dz * j);
      if (core.has(c)) { sides |= 1 << bit; return; }
      if (!wallCols.has(c)) break;
    }
    if (k === 1 && (core.has(key(x - dx + dz, z - dz + dx)) || core.has(key(x - dx - dz, z - dz - dx))))
      sides |= 1 << bit;
  });
  return sides;
}

// The room's columns (see the top): filled from the player's, within the
// footprint (cols: 'x,z' keys), then widened by one - { inside: [{ x, z }],
// core: the fill without the footprint's edge }. null when the fill never
// leaves the footprint's edge: the player is under the eaves, outside.
function insideColumns(cell, cols, opaque) {
  const open = (x, z) => !opaque(x, cell.y, z) || !opaque(x, cell.y + 1, z);
  const filled = new Set([`${cell.x},${cell.z}`]);
  const queue = [[cell.x, cell.z]];
  const SIDES = SIDE_DIRS;
  const core = new Set();
  while (queue.length && filled.size < FILL_CAP) {
    const [x, z] = queue.pop();
    // The footprint's edge (the eaves, a door's step) fills but spreads no
    // further - a door would let the fill run round the house under the
    // eaves - except from the player's own cell, so a door's step leads in.
    const edge = !SIDES.every(([dx, dz]) => cols.has(`${x + dx},${z + dz}`));
    if (!edge) core.add(`${x},${z}`);
    if (edge && (x !== cell.x || z !== cell.z)) continue;
    for (const [dx, dz] of SIDES) {
      const nx = x + dx, nz = z + dz, key = `${nx},${nz}`;
      if (filled.has(key) || !cols.has(key) || !open(nx, nz)) continue;
      filled.add(key);
      queue.push([nx, nz]);
    }
  }
  if (!core.size) return null;
  const inside = new Map();
  for (const key of filled) {
    const [x, z] = key.split(',').map(Number);
    for (let dx = -1; dx <= 1; dx++) for (let dz = -1; dz <= 1; dz++) inside.set(`${x + dx},${z + dz}`, [x + dx, z + dz]);
  }
  return { inside: [...inside.values()].map(([x, z]) => ({ x, z })), core };
}

// Each cell's target this frame: the roof's, and - with walls - the walls of
// the room's sides facing the camera (ahead: the camera's horizontal view, a
// unit {x, z}; the camera looks along it). A Map index -> 1.
export function interiorTargets(interior, ahead, walls = true) {
  const out = new Map();
  if (!interior) return out;
  for (const i of interior.roof) out.set(i, 1);
  if (!walls) return out;
  let facing = 0;
  SIDE_DIRS.forEach(([dx, dz], bit) => { if (-(dx * ahead.x + dz * ahead.z) > FACING_COS) facing |= 1 << bit; });
  for (const w of interior.walls) if (w.sides & facing) out.set(w.i, 1);
  return out;
}

// Every cell toward its target (missing: 0) by dt / CELL_EASE. values: a Map
// index -> 0..1, changed in place; dropped at 0. Returns whether any moved.
export function easeCells(values, targets, dt) {
  const step = dt / CELL_EASE;
  let moved = false;
  for (const [i, t] of targets) if (!values.has(i)) { values.set(i, 0); moved = true; }
  for (const [i, v] of values) {
    const t = targets.get(i) ?? 0;
    const n = t > v ? Math.min(t, v + step) : Math.max(t, v - step);
    if (n !== v) moved = true;
    if (n <= 0 && t === 0) values.delete(i);
    else values.set(i, n);
  }
  return moved;
}

// The mask moved with the player (origin a -> b): the same cells' values
// under their new indices; cells that left its inner part are dropped.
export function rekeyCells(values, a, b) {
  const [W, H] = INTERIOR_DIMS;
  const out = new Map();
  for (const [i, v] of values) {
    const x = i % W, y = Math.floor(i / W) % H, z = Math.floor(i / (W * H));
    const j = maskIndex(b, a.x + x, a.y + y, a.z + z);
    if (j >= 0) out.set(j, v);
  }
  return out;
}
