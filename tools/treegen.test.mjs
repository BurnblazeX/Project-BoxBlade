import { file, section, ok, truthy } from './lib/harness.mjs';
import { growTree, treeBoxes, treeCubes, TREE_SPECIES, CELL } from '../js/core/treegen.js';
import { placeModel, modelInstanceData } from '../js/core/bbmodel.js';

file('treegen.test.mjs - generated trees (oak kit, pulses, pipe radius, leaf automaton, boxes)');

const oak = TREE_SPECIES.oak;
const a = growTree(oak, 7), b = growTree(oak, 7), c = growTree(oak, 8);
const shape = t => [...t.nodes.keys()].sort().join(';');

section('growth');
ok('the same seed, the same tree', shape(a), shape(b));
ok('another seed, another tree', shape(a) === shape(c), false);
ok('older is bigger', growTree(oak, 7, 40).nodes.size < a.nodes.size, true);
const root = a.nodes.get('0,0,0');
ok('the trunk is the thickest', [...a.nodes.values()].every(n => n.r <= root.r), true);
ok('... and no thicker than the species allows', root.r <= oak.maxRadius, true);
ok('a bare trunk up to the fork (the low limbs droop below it only well out)',
  [...a.nodes.values()].filter(n => n.at[1] < a.fork && n.at[0] ** 2 + n.at[2] ** 2 <= 9)
    .every(n => n.at[0] === 0 && n.at[2] === 0), true);
ok('... where two or more limbs leave it', [...a.nodes.values()].filter(n => n.parent && n.parent.join() === `0,${a.fork},0`).length >= 2, true);
ok('every node in the crown or on the trunk', [...a.nodes.values()].every(n => n.at[1] <= a.fork ||
  a.envelope.measure(...n.at) <= 1.3), true);
ok('every node but the root hangs from a neighbour', [...a.nodes.values()].every(n => !n.parent ||
  n.parent.reduce((s, v, i) => s + Math.abs(v - n.at[i]), 0) === 1), true);
// Pipe rule: a parent at least as thick as the root of its children's areas.
ok('thickness by the pipe rule', [...a.nodes.values()].every(n => {
  const kids = [...a.nodes.values()].filter(k => k.parent && k.parent.join() === n.at.join());
  return n.r + 1e-9 >= Math.min(oak.maxRadius, Math.sqrt(kids.reduce((s, k) => s + k.r * k.r, 0)));
}), true);

section('leaves');
truthy('a crown of leaves', a.leaves.size > 50);
const shown = k => a.nodes.has(k) && Math.round(2 * a.nodes.get(k).r) >= 2;
ok('no leaf where shown wood is', [...a.leaves.keys()].some(shown), false);
ok('... a twig too thin to show is leaf', [...a.nodes].filter(([, n]) => Math.round(2 * n.r) < 2).every(([k]) => a.leaves.has(k)), true);
{
  // Every leaf joined to shown wood through leaves: none floats.
  const near = k => { const [x, y, z] = k.split(',').map(Number);
    return [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]].map(([p, q, r]) => `${x + p},${y + q},${z + r}`); };
  const reached = new Set([...a.leaves.keys()].filter(k => near(k).some(shown)));
  const queue = [...reached];
  while (queue.length) for (const k of near(queue.pop())) if (a.leaves.has(k) && !reached.has(k)) { reached.add(k); queue.push(k); }
  ok('no floating leaves', reached.size, a.leaves.size);
  // A domed top: the highest layer of leaves much smaller than the widest.
  const layers = new Map();
  for (const k of a.leaves.keys()) { const y = Number(k.split(',')[1]); layers.set(y, (layers.get(y) || 0) + 1); }
  const top = Math.max(...layers.keys()), widest = Math.max(...layers.values());
  truthy('a domed top, not a flat one', layers.get(top) < widest / 3);
}
ok('none below ground', [...a.leaves.keys()].every(k => Number(k.split(',')[1]) >= 0), true);

section('the pulse kit');
const pine = { ...oak, kit: 'pulses', energy: 18, upProbability: 1, lowestBranch: 4 };
const p = growTree(pine, 7);
ok('the same seed, the same tree', shape(p), shape(growTree(pine, 7)));
ok('a bare trunk up to the lowest branch', [...p.nodes.values()].filter(n => n.at[1] < pine.lowestBranch - 1)
  .every(n => n.at[0] === 0 && n.at[2] === 0), true);
truthy('... and leaves', p.leaves.size > 20);

section('boxes');
const boxes = treeBoxes(a);
truthy('merged: fewer wood boxes than two a node', boxes.wood.length < 2 * a.nodes.size);
ok('leaf boxes cover every leaf cell clear of wood, once', boxes.leaves.reduce((s, x) =>
  s + (x[3] - x[0]) * (x[4] - x[1]) * (x[5] - x[2]), 0), boxes.leafCells * CELL ** 3);
ok('the trunk reaches the ground', boxes.wood.some(x => x[1] === 0), true);
const cubes = treeCubes(boxes);
ok('one cube a box', cubes.length, boxes.wood.length + boxes.leaves.length);
ok('the first cube is the trunk, on the ground', cubes[0].min[1], 0);
const placed = placeModel({ cubes, bounds: [] }, [120, 6, -36], 0).cubes;
ok('placed, a wrapped face keeps its tile', placed.every((q, i) => q.faces.every((f, d) => f.u.o === cubes[i].faces[d].u.o)), true);
ok('... and its cube is marked wrapped (bit 14)', (modelInstanceData(placed[0], { modelLayer: 0, tilesX: 3 }).info[0] >> 14) & 1, 1);
ok('bark on the sides and on top, leaves all round',
  [cubes[0].faces[0].u.o, cubes[0].faces[2].u.o, cubes[cubes.length - 1].faces[4].u.o].join(), '0,0,24');
const runX = treeCubes({ wood: [[0, 8, -1, 12, 10, 1]], leaves: [] })[0];
ok('a branch along x: its top grain (v) runs along x, flagged swapped',
  [runX.faces[2].v.axis, (modelInstanceData(runX, { modelLayer: 0, tilesX: 3 }).info[0] >> 12) & 1].join(), '0,1');
ok('... along z: unswapped', treeCubes({ wood: [[-1, 8, 0, 1, 10, 12]], leaves: [] })[0].faces[2].v.axis, 2);

section('canopy and root flare');
{
  // The dome's top closed: in its inner footprint (measure < 0.3 across),
  // the highest leaf or wood within 2.5 cells of the envelope's surface.
  let open = 0, cols = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const t = growTree(TREE_SPECIES.oak, seed), e = t.envelope, top = new Map();
    const mark = (x, y, z) => top.set(x + ',' + z, Math.max(top.get(x + ',' + z) ?? -99, y));
    for (const k of t.leaves.keys()) mark(...k.split(',').map(Number));
    for (const n of t.nodes.values()) mark(...n.at);
    for (let z = -Math.floor(e.R); z <= e.R; z++) for (let x = -Math.floor(e.R); x <= e.R; x++) {
      const q = (x * x + z * z) / (e.R * e.R);
      if (q > 0.3) continue;
      cols++;
      if (e.cy + e.up * Math.sqrt(1 - q) - (top.get(x + ',' + z) ?? -99) > 2.5) open++;
    }
  }
  truthy(`the crown top is closed (${open} of ${cols} columns open)`, open < 0.05 * cols);
  const foot = age => treeBoxes(growTree(TREE_SPECIES.oak, 3, age)).wood.filter(b => b[1] === 0);
  ok('no flare at full size', foot(160).length, 1);
  const old = foot(300);
  ok('an old oak foot: the trunk on a block a texel a side wider, 7 texels tall',
    old.map(b => `${b[3] - b[0]}x${b[4]}x${b[5] - b[2]}`).join(' ').replace(/^8x\d+x8/, '8xHx8'), '8xHx8 10x7x10');
  ok('... both centred on the tile', old.map(b => [b[0] + b[3], b[2] + b[5]].join()).join(' '), '0,0 0,0');
  truthy('... inside its tile', old.every(b => b[0] >= -6 && b[2] >= -6 && b[3] <= 6 && b[5] <= 6));
  // The skirt: the crown's underside lowest out near its edge, below the
  // fork; under the middle, no lower than the fork's height less a cell.
  let lowEdge = 0, lowMiddle = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const t = growTree(TREE_SPECIES.oak, seed), e = t.envelope;
    for (const k of t.leaves.keys()) {
      const [x, y, z] = k.split(',').map(Number);
      const q = (x * x + z * z) / (e.R * e.R);
      if (y < t.fork && q > 0.4) lowEdge++;
      if (y < t.fork - 1 && x * x + z * z <= 2) lowMiddle++;
    }
  }
  truthy(`leaves hang below the fork out toward the edge (${lowEdge})`, lowEdge > 200);
  ok('... but not round the trunk', lowMiddle, 0);
  ok('... and the trunk stays the first cube', treeCubes(treeBoxes(growTree(TREE_SPECIES.oak, 3, 300)))[0].max[1] > 8, true);
}

section('budget');
{
  // The shape (skirt, low limbs, closed canopy) spends what the oak spent
  // before it (2026-11-01: ~174 nodes, ~2600 leaf cells, ~2520 drawn).
  let nodes = 0, all = 0, drawn = 0;
  for (let seed = 1; seed <= 20; seed++) for (const age of [160, 260]) {
    const t = growTree(TREE_SPECIES.oak, seed, age), g = treeBoxes(t);
    nodes += t.nodes.size / 40; all += g.all.length / 40; drawn += g.cells.length / 40;
  }
  truthy(`wood within 5% of 174 nodes (${nodes.toFixed(0)})`, Math.abs(nodes / 174 - 1) < 0.05);
  truthy(`leaves within 5% of 2520 drawn (${drawn.toFixed(0)})`, Math.abs(drawn / 2520 - 1) < 0.05);
}

section('clearance');
{
  // A block and a third (4 cells, 2 m) over the ground clear of leaves and branches -
  // the trunk aside - on every oak with room for a crown over it; a
  // sapling keeps its own bare trunk. Under clearanceAge, none at all.
  let broken = 0, trees = 0, saplings = 0;
  for (let seed = 1; seed <= 20; seed++) for (const age of [100, 120, 160, 260]) {
    const t = growTree(TREE_SPECIES.oak, seed, age);
    const low = y => y < t.clearance;
    if (t.clearance < TREE_SPECIES.oak.clearance) saplings++;
    trees++;
    for (const k of t.leaves.keys()) if (low(Number(k.split(',')[1]))) broken++;
    for (const n of t.nodes.values()) if (low(n.at[1]) && (n.at[0] || n.at[2])) broken++;
  }
  ok('no leaf or branch under the clearance', broken, 0);
  truthy(`... the full clearance on all but the saplings (${saplings} of ${trees})`, saplings < trees / 3);
  const big = growTree(TREE_SPECIES.oak, 1, 160), small = growTree(TREE_SPECIES.oak, 2, 60);
  // Low limbs: wood out past three cells at the fork's height or under it.
  const lowLimb = t => [...t.nodes.values()].some(n => n.at[1] <= t.fork && n.at[0] ** 2 + n.at[2] ** 2 > 9);
  ok('a young oak (age 90) keeps no clearance', growTree(TREE_SPECIES.oak, 2, 90).clearance, 0);
  truthy('a full-size oak has low limbs', lowLimb(big));
  ok('... a sapling none', lowLimb(small), false);
}

section('closed crown sides');
{
  // Seen from each side, through the crown's middle band, the first leaf
  // lies within 2.5 cells of the envelope's surface - no windows onto the
  // branches inside.
  let open = 0, lines = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const t = growTree(TREE_SPECIES.oak, seed), e = t.envelope, R = Math.ceil(e.R) + 2;
    const solid = new Set(t.leaves.keys());
    for (const n of t.nodes.values()) solid.add(n.at.join());
    for (const [ax, sgn] of [[0, 1], [0, -1], [2, 1], [2, -1]]) {
      const at = (w, y, u) => (ax === 0 ? [sgn * w, y, u] : [u, y, sgn * w]);
      for (let y = Math.ceil(e.cy + e.up); y > t.fork; y--) for (let u = -R; u <= R; u++) {
        if ((ax === 0 ? e.measure(0, y, u) : e.measure(u, y, 0)) > 0.5) continue;
        lines++;
        let surf = 0, hit = null;
        for (let w = 0; w <= R; w++) if (e.measure(...at(w, y, u)) <= 1) surf = w;
        for (let w = R; w >= -R && hit === null; w--) if (solid.has(at(w, y, u).join())) hit = w;
        if (hit === null || surf - hit > 2.5) open++;
      }
    }
  }
  truthy(`the crown sides are closed (${open} of ${lines} sight lines open)`, open < 0.02 * lines);
}
