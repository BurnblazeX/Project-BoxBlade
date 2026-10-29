// --- Generated trees (the Dynamic Trees way; Research/Procedural_Trees.md) ---
//
// A tree grows by PULSES on the BoxGrid (half-metre cells, 4 texels): each
// pulse is a signal sent up from the root with an energy - steps it may take.
// At every branch cell it picks its next direction by weighted chance: up
// (upProbability, never below lowestBranch otherwise), straight on, and
// toward existing branch (its radius + 2) or air (1) - so growth follows the
// limbs already there. Where it steps into air, one new branch cell grows and
// the pulse ends. Age is how many pulses it has had.
//
// Thickness from the tips down, as the recursion unwinds - the pipe rule:
//   r = clamp(sqrt(r_child^2 + sum r_side^2) + taper, r, maxRadius)
// in texels; a new tip is twigRadius (one texel wide).
//
// Leaves: Dynamic Trees' hydration automaton. Twig cells emit 5; an air cell
// takes the first rule (H, C, R) where at least C of its six neighbours hold
// H or more... as Dynamic Trees' solver: at least C neighbours AT H. Repeated
// until it settles. Every cell left above 0 is leaf.
//
// Then as boxes on the texel grid: each branch cell a cube of its width,
// joined to its parent by a box of its own width, and straight runs of one
// width merged; the leaf cells merged into as few boxes as cover them.
//
// Pure and seeded: the same species, seed and age give the same tree.
import { createRng } from './rng.js';

export const CELL = 4;   // texels a BoxGrid cell

// Species. Lengths in cells unless said; radii in texels.
export const TREE_SPECIES = {
  oak: {
    energy: 18, upProbability: 1, lowestBranch: 4, taper: 0.12, twigRadius: 0.5, maxRadius: 3.5,
    pulses: 160,
    // Dynamic Trees' deciduous solver: [H, C, R].
    leafRules: [[5, 1, 4], [4, 2, 3], [3, 2, 2], [4, 1, 1], [3, 1, 1], [2, 1, 1]],
    textures: { bark: 'terrain_oakBark', top: 'terrain_oakBark_top', leaves: 'terrain_oakLeaves' },
    // The leaves' summer ramp, dark to light (the grey levels map onto it).
    ramp: [[0x38, 0x48, 0x20], [0x40, 0x60, 0x28], [0x48, 0x70, 0x28], [0x78, 0x90, 0x28]]
  }
};

const DIRS = [[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]];
const UP = 0, DOWN = 5;
const key = (x, y, z) => `${x},${y},${z}`;

// Grows a tree: { nodes: Map key -> { at: [x, y, z], r, parent }, leaves:
// Map key -> hydration, cells as [x, y, z] from the root cell (0, 0, 0, the
// first above the ground) }.
export function growTree(species, seed, pulses = species.pulses) {
  const rng = createRng('tree', seed);
  const nodes = new Map();
  nodes.set(key(0, 0, 0), { at: [0, 0, 0], r: species.twigRadius, parent: null });
  const radiusAt = (x, y, z) => { const n = nodes.get(key(x, y, z)); return n ? n.r : 0; };

  // One pulse from node n, coming in along `dir`, `steps` taken.
  function signal(n, dir, steps, energy) {
    if (energy <= 0) return { r: n.r, grew: false };
    const [x, y, z] = n.at;
    let next = UP;
    if (steps + 1 > species.lowestBranch) {
      const w = DIRS.map((d, i) => {
        if (i === DOWN) return 0;
        const from = dir !== null && d.every((v, a) => v === -DIRS[dir][a]);
        if (from) return 0;
        const cx = x + d[0], cy = y + d[1], cz = z + d[2];
        if (radiusAt(cx, cy, cz)) return radiusAt(cx, cy, cz) + 2;   // along an existing limb
        // New growth seeks light: away from crowding (branch cells round the
        // cell it would grow into), and sideways outward from the trunk - so
        // limbs spread over the crown instead of clumping at its middle.
        let crowd = 0;
        for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
          if ((a || b || c) && nodes.has(key(cx + a, cy + b, cz + c))) crowd++;
        }
        let p = 1;
        if (i === UP) p += species.upProbability;
        if (i === dir) p += 1;
        if (d[1] === 0 && d[0] * x + d[2] * z > 0) p *= 2;
        return p / (1 + crowd);
      });
      const total = w.reduce((a, b) => a + b, 0);
      let pick = rng.float() * total;
      next = w.findIndex(v => (pick -= v) < 0);
      if (next < 0) next = UP;
    }
    const d = DIRS[next], at = [x + d[0], y + d[1], z + d[2]];
    let child = nodes.get(key(...at));
    let grew = false;
    if (!child) {
      if (at[1] < 0) return { r: n.r, grew: false };
      child = { at, r: species.twigRadius, parent: n.at };
      nodes.set(key(...at), child);
      grew = true;
    } else if (child.parent && child.parent.join() === n.at.join()) {
      grew = signal(child, next, steps + 1, energy - 1).grew;
    } else {
      return { r: n.r, grew: false };     // into another limb: no loops
    }
    // Thickness, the pipe rule over this node's children.
    let area = 0;
    for (const dd of DIRS) {
      const c = nodes.get(key(x + dd[0], y + dd[1], z + dd[2]));
      if (c && c.parent && c.parent.join() === n.at.join()) area += c.r * c.r;
    }
    n.r = Math.min(species.maxRadius, Math.max(n.r, Math.sqrt(area) + species.taper));
    return { r: n.r, grew };
  }
  const root = nodes.get(key(0, 0, 0));
  for (let p = 0; p < pulses; p++) signal(root, null, 0, species.energy);

  // Leaves: twigs emit 5; the automaton over the air cells near them.
  const leaves = new Map();
  const twig = n => n.r < 1;
  const value = (x, y, z) => {
    const n = nodes.get(key(x, y, z));
    if (n) return twig(n) ? 5 : 0;
    return leaves.get(key(x, y, z)) || 0;
  };
  const near = new Set();
  for (const n of nodes.values()) {
    if (!twig(n)) continue;
    for (let dx = -3; dx <= 3; dx++) for (let dy = -2; dy <= 3; dy++) for (let dz = -3; dz <= 3; dz++) {
      const at = [n.at[0] + dx, n.at[1] + dy, n.at[2] + dz];
      if (at[1] >= 0 && !nodes.has(key(...at))) near.add(key(...at));
    }
  }
  for (let pass = 0; pass < 8; pass++) {
    let changed = false;
    const next = new Map();
    for (const k of near) {
      const [x, y, z] = k.split(',').map(Number);
      const counts = new Array(8).fill(0);
      for (const d of DIRS) counts[value(x + d[0], y + d[1], z + d[2])]++;
      let v = 0;
      for (const [h, c, r] of species.leafRules) if (counts[h] >= c) { v = r; break; }
      if (v) next.set(k, v);
      if (v !== (leaves.get(k) || 0)) changed = true;
    }
    leaves.clear();
    for (const [k, v] of next) leaves.set(k, v);
    if (!changed) break;
  }
  // A rounded crown: an ellipsoid over the branches, from the lowest branch
  // to above the top, as wide as they reach. Nothing outside it (no jagged
  // tufts); its upper shell filled whole (a smooth dome over the top); below
  // its middle, thinning toward the bottom by chance, where the branches show.
  let top = 0, reach = 0;
  for (const n of nodes.values()) { top = Math.max(top, n.at[1]); reach = Math.max(reach, Math.hypot(n.at[0], n.at[2])); }
  const cy = (species.lowestBranch + top + 2) / 2, ry = Math.max(1.5, (top + 2 - species.lowestBranch) / 2);
  const rx = Math.max(2, reach + 1.5);
  const measure = (x, y, z) => (x * x + z * z) / (rx * rx) + ((y - cy) / ry) ** 2;
  for (const k of near) {
    const [x, y, z] = k.split(',').map(Number);
    const m = measure(x, y, z), v = leaves.get(k) || 0;
    if (m > 1) { leaves.delete(k); continue; }
    if (y >= cy && m >= 0.55) { leaves.set(k, Math.max(v, 2)); continue; }
    if (!v) continue;
    const t = (y - species.lowestBranch) / Math.max(1, cy - species.lowestBranch);   // 0 at the lowest branch, 1 at the middle
    if (rng.float() >= Math.min(1, (v >= 3 ? 0.5 : 0.2) + t)) leaves.delete(k);
  }
  return { nodes, leaves };
}

// A box [x0, y0, z0, x1, y1, z1] in texels from the root cell's floor centre
// (the model's origin: the middle of the top of the block it stands on).
const cellCentre = ([x, y, z]) => [x * CELL, y * CELL + CELL / 2, z * CELL];
const around = (c, w) => { const lo = c - Math.floor(w / 2); return [lo, lo + w]; };

// Merges boxes that share a cross-section and touch along the third axis,
// and drops boxes inside others.
function mergeBoxes(boxes) {
  let list = boxes.map(b => b.slice());
  let merged = true;
  while (merged) {
    merged = false;
    list = list.filter((b, i) => !list.some((o, j) => j !== i && o[0] <= b[0] && o[1] <= b[1] && o[2] <= b[2] &&
      o[3] >= b[3] && o[4] >= b[4] && o[5] >= b[5] && (j < i || o.some((v, a) => v !== b[a]))));
    outer: for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const a = list[i], b = list[j];
        for (let ax = 0; ax < 3; ax++) {
          const o1 = (ax + 1) % 3, o2 = (ax + 2) % 3;
          if (a[o1] !== b[o1] || a[o1 + 3] !== b[o1 + 3] || a[o2] !== b[o2] || a[o2 + 3] !== b[o2 + 3]) continue;
          if (a[ax + 3] < b[ax] || b[ax + 3] < a[ax]) continue;
          a[ax] = Math.min(a[ax], b[ax]); a[ax + 3] = Math.max(a[ax + 3], b[ax + 3]);
          list.splice(j, 1);
          merged = true;
          break outer;
        }
      }
    }
  }
  return list;
}

// The boxes as model cubes (core/bbmodel.js's form), every face wrapped onto
// its tile (bbmodel.js transformCube): bark (tile 0) on the wood's sides, end
// grain (tile 1) on its tops and bottoms, leaves (tile 2) all round. Image
// rows run down, so v runs against y on the sides. The trunk's root box first.
const FACE_AXES = [[2, 1], [2, 1], [0, 2], [0, 2], [0, 1], [0, 1]];   // as bbmodel.js
export function treeCubes({ wood, leaves }) {
  const cube = (b, tileOf) => ({
    name: '', light: false, lightLevel: null, rects: [], min: b.slice(0, 3), max: b.slice(3),
    faces: FACE_AXES.map(([u, v], d) => ({
      u: { axis: u, o: 12 * tileOf(d), d: 1 }, v: { axis: v, o: 0, d: v === 1 ? -1 : 1 }, wrap: true }))
  });
  const root = wood.reduce((best, b) => (b[1] <= 0 && (!best || b[4] > best[4]) ? b : best), null);
  const woodSorted = root ? [root, ...wood.filter(b => b !== root)] : wood;
  return [...woodSorted.map(b => cube(b, d => (d === 2 || d === 3 ? 1 : 0))),
          ...leaves.map(b => cube(b, () => 2))];
}

// The tree as boxes: { wood: [box], leaves: [box] } (see above).
export function treeBoxes({ nodes, leaves }) {
  const width = n => Math.max(1, Math.round(2 * n.r));
  const wood = [];
  for (const n of nodes.values()) {
    const c = cellCentre(n.at), w = width(n);
    const cube = [0, 1, 2].map(a => around(c[a], w));
    // Down to the ground for the root.
    if (!n.parent) cube[1][0] = 0;
    wood.push([cube[0][0], cube[1][0], cube[2][0], cube[0][1], cube[1][1], cube[2][1]]);
    if (n.parent) {
      const p = cellCentre(n.parent);
      const lo = c.map((v, a) => Math.min(v, p[a])), hi = c.map((v, a) => Math.max(v, p[a]));
      const span = [0, 1, 2].map(a => (lo[a] === hi[a] ? around(c[a], w) : [lo[a], hi[a]]));
      wood.push([span[0][0], span[1][0], span[2][0], span[0][1], span[1][1], span[2][1]]);
    }
  }
  const woodBoxes = mergeBoxes(wood);
  // Leaf cells clear of the wood, merged greedily: runs along x, then
  // stacked along z, then along y.
  const inWood = c => {
    const lo = [c[0] * CELL - CELL / 2, c[1] * CELL, c[2] * CELL - CELL / 2];
    return woodBoxes.some(b => b[0] < lo[0] + CELL && b[3] > lo[0] && b[1] < lo[1] + CELL && b[4] > lo[1] &&
      b[2] < lo[2] + CELL && b[5] > lo[2]);
  };
  const cells = [...leaves.keys()].map(k => k.split(',').map(Number)).filter(c => !inWood(c));
  const leafBoxes = mergeBoxes(cells.map(c => [c[0] * CELL - CELL / 2, c[1] * CELL, c[2] * CELL - CELL / 2,
                                               c[0] * CELL + CELL / 2, c[1] * CELL + CELL, c[2] * CELL + CELL / 2]));
  return { wood: woodBoxes, leaves: leafBoxes, leafCells: cells.length };
}
