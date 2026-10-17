// --- Interiors: the roof and the front walls made see-through ---
//
// Under a roof, the building around the player turns translucent, the way
// Project Zomboid and The Sims cut away a house: found on the world's blocks,
// not on pixels.
//
//   the roof     the first opaque block straight above the player, and every
//                opaque block connected to it at or above that height (a flood
//                fill, capped): the ceiling, the storeys above, their walls,
//                the roof itself - this building's, however big or odd.
//   the walls    opaque blocks under the roof's footprint (and its eaves, one
//                block out), from one block above the player's floor up to the
//                roof - the bottom row stays, a stub at the room's edge. Only
//                those toward the camera from the player are front walls, and
//                that is asked every frame, so it follows the camera round.
//
// Each cell eases toward its target on its own (easeCells), so a quarter turn
// of the camera fades one wall back in as the next fades out. How see-through
// a cell is also falls off with its distance on screen from the player, 100%
// to 50% (gpu/occlusion.js). The cells live in a mask texture around the player:
// INTERIOR_DIMS cells, the player's at INTERIOR_HALF, whose outermost layer
// is never written (reads outside clamp to it, so to nothing).
import { BLOCK_METRES, getBlock } from './world.js';
import { isGlassMaterial } from './materials.js';

export const INTERIOR_DIMS = [64, 32, 64];   // cells x, y, z
export const INTERIOR_HALF = [32, 8, 32];    // the player's cell within it
export const ROOF_REACH = 12;                // cells up looking for a roof
export const FILL_CAP = 20000;               // cells, so a mountain cannot stall the frame
export const FRONT_MARGIN = 0.5;             // blocks toward the camera, past the player
export const CELL_EASE = 0.25;               // seconds for a cell to fade fully in or out
export const ROOF_DELAY = 0.25;              // under a roof this long before it fades
export const DEFAULT_INTERIOR = { near: 2.5, far: 9, min: 0.5 };   // metres on screen at the player; the far cells' cut

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

// The interior around the player's cell, or null in the open:
//   { origin, roofY, roof: [index], walls: [{ i, x, z }] }
export function buildInterior(cell, opaque = opaqueAt) {
  let roofY = null;
  for (let y = cell.y + 1; y <= cell.y + ROOF_REACH; y++) if (opaque(cell.x, y, cell.z)) { roofY = y; break; }
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
        if (i >= 0 && !inRoof.has(i) && opaque(x, y, z)) walls.push({ i, x, z });
      }
    }
  }
  return { origin, roofY, roof, walls };
}

// Each cell's target this frame: the roof's, and - with walls - the walls
// toward the camera from the player (feet, metres; ahead: the camera's
// horizontal view, a unit {x, z}). A Map index -> 1.
export function interiorTargets(interior, feet, ahead, walls = true) {
  const out = new Map();
  if (!interior) return out;
  for (const i of interior.roof) out.set(i, 1);
  if (!walls) return out;
  const B = BLOCK_METRES;
  for (const w of interior.walls) {
    const h = (w.x * B - feet.x) * ahead.x + (w.z * B - feet.z) * ahead.z;   // < 0 toward the camera
    if (h < -FRONT_MARGIN * B) out.set(w.i, 1);
  }
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
