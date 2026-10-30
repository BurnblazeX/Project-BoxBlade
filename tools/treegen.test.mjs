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
ok('a bare trunk up to the fork', [...a.nodes.values()].filter(n => n.at[1] < a.fork)
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
ok('bark on the sides, end grain on top, leaves all round',
  [cubes[0].faces[0].u.o, cubes[0].faces[2].u.o, cubes[cubes.length - 1].faces[4].u.o].join(), '0,12,24');
