import { file, section, ok, truthy } from './lib/harness.mjs';
import { growTree, treeBoxes, treeCubes, TREE_SPECIES, CELL } from '../js/core/treegen.js';
import { placeModel, modelInstanceData } from '../js/core/bbmodel.js';

file('treegen.test.mjs - generated trees (pulses, pipe radius, leaf automaton, boxes)');

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
ok('a bare trunk up to the lowest branch', [...a.nodes.values()].filter(n => n.at[1] < oak.lowestBranch - 1)
  .every(n => n.at[0] === 0 && n.at[2] === 0), true);
ok('every node but the root hangs from a neighbour', [...a.nodes.values()].every(n => !n.parent ||
  n.parent.reduce((s, v, i) => s + Math.abs(v - n.at[i]), 0) === 1), true);
// Pipe rule: a parent at least as thick as the root of its children's areas.
ok('thickness by the pipe rule', [...a.nodes.values()].every(n => {
  const kids = [...a.nodes.values()].filter(k => k.parent && k.parent.join() === n.at.join());
  return n.r + 1e-9 >= Math.min(oak.maxRadius, Math.sqrt(kids.reduce((s, k) => s + k.r * k.r, 0)));
}), true);

section('leaves');
truthy('a crown of leaves', a.leaves.size > 50);
ok('no leaf where wood is', [...a.leaves.keys()].some(k => a.nodes.has(k)), false);
ok('none below ground', [...a.leaves.keys()].every(k => Number(k.split(',')[1]) >= 0), true);

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
