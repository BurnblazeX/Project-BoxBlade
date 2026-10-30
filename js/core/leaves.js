// --- Leaves: every placed tree's canopy, as cells on the BoxGrid ---
//
// A generated tree (core/treegen.js, core/models.js treeModel) is two things:
// its wood, model cubes like any model's, and its leaves - cells half a metre
// across, a BoxGrid box each when the tree stands on a full block. The leaves
// are not model cubes: gpu/leaves.js draws them as cubes that sway, textured
// by noise and coloured by the species' ramp.
//
// A tree draws its crown's shell (treegen.js shownLeafCells). Each cell is
// placed as its tree is: turned with it, standing where it stands.
import { TREE_SPECIES, CELL, LEAF_FACES } from './treegen.js';
import { quarterTurn, transformCube } from './bbmodel.js';
import { BLOCK_METRES } from './world.js';
import { boxIndex } from './boxgrid.js';

const TEXEL_METRES = BLOCK_METRES / 12;
const SPECIES = Object.values(TREE_SPECIES);
// The species in the order gpu/leaves.js's ramp table holds them.
export const LEAF_SPECIES = Object.keys(TREE_SPECIES);

// The leaf cubes of the placed models (core/world.js WorldModels), as vec4s:
// centre (world metres), then (drawn faces x 4 + species) + the cell's
// height in its crown (0 .. 0.999) - its drawn faces turned with the tree, a bit each in
// LEAF_FACES order; at most 4 species. Written into `out` (a Float32Array, 4
// a cube). Past its room, the nearest to `near` (metres) are kept. Returns
// { count, total }.
export function placedLeafCubes(models, out, near = null) {
  const capacity = out.length / 4;
  const cubes = [];
  for (const m of models) {
    const cells = m.model.leafCells;
    if (!cells || !m.at) continue;
    const species = Math.max(0, SPECIES.indexOf(m.model.species));
    const T = quarterTurn(1, m.yaw || 0);
    for (let b = 0; b < 3; b++) T.p[b] += m.at[b];
    // Each face's direction once turned: target axis b = s x source axis a.
    const faceTo = LEAF_FACES.map(d => {
      const w = [0, 0, 0];
      for (let a = 0; a < 3; a++) { const [b, s] = T.axisOf[a]; w[b] = s * d[a]; }
      return LEAF_FACES.findIndex(f => f.every((v, a) => v === w[a]));
    });
    for (const [x, y, z, height, open] of cells) {
      const c = [x * CELL, y * CELL + CELL / 2, z * CELL];
      const p = transformCube({ min: c, max: c, faces: [] }, T).min.map(v => v * TEXEL_METRES);
      let turned = 0;
      for (let f = 0; f < 6; f++) if (open & (1 << f)) turned |= 1 << faceTo[f];
      cubes.push([p[0], p[1], p[2], turned * 4 + species + Math.min(height, 0.999)]);
    }
  }
  const total = cubes.length;
  if (total > capacity && near) {
    const d = c => (c[0] - near[0]) ** 2 + (c[1] - near[1]) ** 2 + (c[2] - near[2]) ** 2;
    cubes.sort((a, b) => d(a) - d(b));
  }
  const count = Math.min(total, capacity);
  for (let i = 0; i < count; i++) out.set(cubes[i], 4 * i);
  return { count, total };
}

// Every leaf cell of the placed models, inside and out - what shades - as the
// BoxGrid boxes they are (core/boxgrid.js boxIndex; a tree on a slab rounds
// to the nearest): [x0, y0, z0, x1, ...].
export function placedLeafBoxes(models) {
  const out = [];
  for (const m of models) {
    const cells = m.model.leafAll;
    if (!cells || !m.at) continue;
    const T = quarterTurn(1, m.yaw || 0);
    for (let b = 0; b < 3; b++) T.p[b] += m.at[b];
    for (const [x, y, z] of cells) {
      const c = [x * CELL, y * CELL + CELL / 2, z * CELL];
      for (const v of transformCube({ min: c, max: c, faces: [] }, T).min) out.push(boxIndex(v * TEXEL_METRES));
    }
  }
  return out;
}

// The leaf density window (gpu/leafvolume.js): `out` (dims x*y*z bytes, x
// fastest) cleared, then 255 in every box of `boxes` inside the window whose
// min box is `origin`. bounds (6 numbers, optional): the min and max box of
// those inside, or min past max with none. Returns how many fell inside.
export function fillLeafDensity(boxes, origin, dims, out, bounds = null) {
  out.fill(0);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let n = 0;
  for (let i = 0; i < boxes.length; i += 3) {
    const x = boxes[i] - origin[0], y = boxes[i + 1] - origin[1], z = boxes[i + 2] - origin[2];
    if (x < 0 || y < 0 || z < 0 || x >= dims[0] || y >= dims[1] || z >= dims[2]) continue;
    out[x + dims[0] * (y + dims[1] * z)] = 255;
    for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], boxes[i + a]); hi[a] = Math.max(hi[a], boxes[i + a]); }
    n++;
  }
  if (bounds) {
    if (n) bounds.splice(0, 6, ...lo, ...hi);
    else bounds.splice(0, 6, 1, 1, 1, 0, 0, 0);
  }
  return n;
}
