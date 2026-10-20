// --- Picking by grid walk ---
//
// three's InstancedMesh raycast tests every instance - the whole terrain, every
// frame the mouse moves (it was most of the logic time while hovering). Here
// the instances' boxes are binned by grid cell once, and a pick walks the
// cells along the ray (Amanatides & Woo), testing only the boxes binned there.
//
// Same answer as the raycast picker it replaces (render/picking.js tileAt):
// the nearest box face the ray ENTERS, skipping side and bottom faces and
// going on past them - and a box the ray starts inside is never hit, as its
// faces all face away (the terrain material is front-sided).
//
// Instances are axis-aligned boxes: a unit box of side `size` (the terrain's
// BoxGeometry) under a matrix of scale and translation only. Scale 0 is a
// hidden instance (battle) and is left out. Cells are `size` wide and centred
// on multiples of it, as blocks are.

const EPS = 1e-6;

// matrices: column-major 4x4 per instance (an InstancedMesh's instanceMatrix
// array); count: instances in use.
export function buildPickGrid(matrices, count, size) {
  const boxes = new Float32Array(count * 6);
  const cells = new Map();
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const cellOf = v => Math.floor(v / size + 0.5);
  for (let i = 0; i < count; i++) {
    const m = i * 16;
    const sx = matrices[m], sy = matrices[m + 5], sz = matrices[m + 10];
    if (sx === 0 || sy === 0 || sz === 0) { boxes[i * 6] = NaN; continue; }
    const hx = Math.abs(sx) * size / 2, hy = Math.abs(sy) * size / 2, hz = Math.abs(sz) * size / 2;
    const cx = matrices[m + 12], cy = matrices[m + 13], cz = matrices[m + 14];
    const b = [cx - hx, cy - hy, cz - hz, cx + hx, cy + hy, cz + hz];
    boxes.set(b, i * 6);
    for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], b[a]); hi[a] = Math.max(hi[a], b[a + 3]); }
    // Every cell the box overlaps; a box ending exactly on a cell edge stays out of the next.
    const x0 = cellOf(b[0]), x1 = cellOf(b[3] - EPS), y0 = cellOf(b[1]), y1 = cellOf(b[4] - EPS);
    const z0 = cellOf(b[2]), z1 = cellOf(b[5] - EPS);
    for (let x = x0; x <= x1; x++) for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) {
      const k = `${x},${y},${z}`;
      const list = cells.get(k);
      if (list) list.push(i); else cells.set(k, [i]);
    }
  }
  return { boxes, cells, size, lo, hi };
}

// The ray against one box: { t, top } where it enters through a face, or null.
function enterBox(boxes, i, o, d) {
  const b = i * 6;
  if (Number.isNaN(boxes[b])) return null;
  let tNear = -Infinity, tFar = Infinity, axis = -1;
  for (let a = 0; a < 3; a++) {
    if (d[a] === 0) {
      if (o[a] < boxes[b + a] || o[a] > boxes[b + a + 3]) return null;
      continue;
    }
    let t0 = (boxes[b + a] - o[a]) / d[a], t1 = (boxes[b + a + 3] - o[a]) / d[a];
    if (t0 > t1) { const s = t0; t0 = t1; t1 = s; }
    if (t0 > tNear) { tNear = t0; axis = a; }
    if (t1 < tFar) tFar = t1;
  }
  if (axis < 0 || tNear > tFar || tNear < 0) return null;
  return { t: tNear, top: axis === 1 && d[1] < 0 };
}

// The instance whose top face the ray (origin o, direction d, both [x, y, z])
// meets first, past any side or bottom faces in the way; -1 for none.
// skip(i): an instance not shown (a lifted roof), passed through.
export function pickTop(grid, o, d, skip = null) {
  const { boxes, cells, size, lo, hi } = grid;
  // The ray within the instances' bounds.
  let tMin = 0, tMax = Infinity;
  for (let a = 0; a < 3; a++) {
    if (d[a] === 0) {
      if (o[a] < lo[a] || o[a] > hi[a]) return -1;
      continue;
    }
    let t0 = (lo[a] - o[a]) / d[a], t1 = (hi[a] - o[a]) / d[a];
    if (t0 > t1) { const s = t0; t0 = t1; t1 = s; }
    tMin = Math.max(tMin, t0); tMax = Math.min(tMax, t1);
  }
  if (!(tMin <= tMax)) return -1;

  // The walk, from the cell the ray is in at tMin.
  const cell = [0, 0, 0], step = [0, 0, 0], next = [0, 0, 0], delta = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const p = o[a] + d[a] * tMin;
    cell[a] = Math.floor(p / size + 0.5);
    if (d[a] > 0) {
      step[a] = 1; delta[a] = size / d[a];
      next[a] = ((cell[a] + 0.5) * size - o[a]) / d[a];
    } else if (d[a] < 0) {
      step[a] = -1; delta[a] = -size / d[a];
      next[a] = ((cell[a] - 0.5) * size - o[a]) / d[a];
    } else {
      delta[a] = Infinity; next[a] = Infinity;
    }
  }
  let tEnter = tMin;
  const hits = [];
  while (tEnter <= tMax + EPS) {
    const tExit = Math.min(next[0], next[1], next[2]);
    const list = cells.get(`${cell[0]},${cell[1]},${cell[2]}`);
    if (list) {
      // Faces entered within this cell; one entered past it is met again in its own.
      hits.length = 0;
      for (const i of list) {
        const h = enterBox(boxes, i, o, d);
        if (h && h.t >= tEnter - EPS && h.t < tExit + EPS) hits.push([h.t, i, h.top]);
      }
      hits.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      for (const [, i, top] of hits) if (top && !(skip && skip(i))) return i;
    }
    const a = next[0] === tExit ? 0 : next[1] === tExit ? 1 : 2;
    cell[a] += step[a];
    next[a] += delta[a];
    tEnter = tExit;
  }
  return -1;
}
