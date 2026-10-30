import { file, section, ok, truthy } from './lib/harness.mjs';
import { treeModel } from '../js/core/models.js';
import { placedLeafCubes, placedLeafBoxes, fillLeafDensity } from '../js/core/leaves.js';

file('leaves.test.mjs - trees\' leaves as cells (core/leaves.js, models.js treeModel)');

const tree = treeModel('tree_oak#7@160');

section('the tree model');
ok('its cubes are all wood: bark and end grain, no leaf tile', tree.cubes.every(c =>
  c.faces.every(f => f.u.o === 0 || f.u.o === 12)), true);
truthy('its leaves are cells', tree.leafCells.length > 100);
ok('each with its height in the crown, 0 to 1', tree.leafCells.every(c => c[3] >= 0 && c[3] <= 1), true);
{
  const top = Math.max(...tree.leafCells.map(c => c[1]));
  const mean = cells => cells.reduce((s, c) => s + c[3], 0) / cells.length;
  truthy('... higher up the crown, higher',
    mean(tree.leafCells.filter(c => c[1] >= top - 2)) > mean(tree.leafCells.filter(c => c[1] <= top / 2)));
  truthy('every leaf cell kept for the shading, inside ones too', tree.leafAll.length > tree.leafCells.length);
}

section('placed');
// Standing on block (4, 2, -3): its origin at the middle of the block's top.
const at = [4 * 12, 2 * 12 + 6, -3 * 12];
const out = new Float32Array(tree.leafCells.length * 4);
const r = placedLeafCubes([{ model: tree, yaw: 0, at }], out);
ok('every cell placed', `${r.count}/${r.total}`, `${tree.leafCells.length}/${tree.leafCells.length}`);
const cubes = [...Array(r.count)].map((_, i) => [...out.subarray(4 * i, 4 * i + 4)]);
ok('on the BoxGrid: centres on half metres', cubes.every(c => [0, 1, 2].every(a => Number.isInteger(c[a] * 2))), true);
ok('above the block it stands on', cubes.every(c => c[1] > 2 * 1.5 + 0.75), true);
ok('the species and the height in w', cubes.every(c => Math.floor(c[3]) % 4 === 0 && c[3] - Math.floor(c[3]) < 1), true);
ok('... and which faces are drawn', cubes.every(c => Math.floor(Math.floor(c[3]) / 4) > 0), true);
// Between two drawn cells one face, never two on one plane (they would fight).
const FACES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
const facesClear = arr => {
  const faces = new Map([...Array(r.count)].map((_, i) =>
    [[0, 1, 2].map(a => Math.round(arr[4 * i + a] * 2)).join(), Math.floor(Math.floor(arr[4 * i + 3]) / 4)]));
  return [...faces].every(([k, drawn]) => FACES.every((d, f) => {
    if (!(drawn & (1 << f))) return true;
    const other = faces.get(k.split(',').map((v, a) => Number(v) + d[a]).join());
    return other === undefined || !(other & (1 << (f ^ 1)));
  }));
};
ok('one face a plane between drawn cells', facesClear(out), true);
{
  // The shell: cells a face from air, and some further in - but not all.
  const shell = tree.leafCells.length, inner = tree.leafCells.filter(c => c[4] !== 0).length;
  truthy('the crown shell drawn, some inside cells among it, not every cell',
    shell === inner && shell < tree.leafAll.length &&
    tree.leafCells.some(([x, y, z]) => FACES.every(d => tree.leafAll.some(c => c[0] === x + d[0] && c[1] === y + d[1] && c[2] === z + d[2]))));
}
{
  const turned = new Float32Array(out.length);
  placedLeafCubes([{ model: tree, yaw: 1, at }], turned);
  const ring = arr => [...Array(r.count)].map((_, i) => {
    const dx = arr[4 * i] - 6, dz = arr[4 * i + 2] + 4.5;
    return Math.round((dx * dx + dz * dz) * 4) + ':' + arr[4 * i + 1];
  }).sort().join();
  ok('turned with its tree, about its trunk', ring(turned), ring(out));
  ok('... and actually turned', turned.some((v, i) => v !== out[i]), true);
  ok('... its drawn faces turned with it', facesClear(turned), true);
}

section('capacity');
{
  const small = new Float32Array(40);
  const near = [6, 20, -4.5];
  const s = placedLeafCubes([{ model: tree, yaw: 0, at }], small, near);
  ok('past its room, the room is filled', `${s.count}/${s.total}`, `10/${tree.leafCells.length}`);
  const d = c => (c[0] - near[0]) ** 2 + (c[1] - near[1]) ** 2 + (c[2] - near[2]) ** 2;
  const kept = Math.max(...[...Array(10)].map((_, i) => d(small.subarray(4 * i, 4 * i + 3))));
  ok('... by the nearest', cubes.filter(c => d(c) < kept).length <= 10, true);
}
section('the leaf volume');
{
  const boxes = placedLeafBoxes([{ model: tree, yaw: 0, at }]);
  ok('a box per leaf cell', boxes.length, tree.leafAll.length * 3);
  ok('... the drawn cubes among them', cubes.every(c => {
    const k = c.slice(0, 3).map(v => Math.round(v * 2));
    for (let i = 0; i < boxes.length; i += 3) if (boxes[i] === k[0] && boxes[i + 1] === k[1] && boxes[i + 2] === k[2]) return true;
    return false;
  }), true);
  const dims = [64, 48, 64], data = new Uint8Array(64 * 48 * 64);
  const origin = [boxes[0] - 32, boxes[1] - 24, boxes[2] - 32];
  const n = fillLeafDensity(boxes, origin, dims, data);
  ok('filled where the leaves are', data.reduce((s, v) => s + (v ? 1 : 0), 0), n);
  ok('... every one inside a window that holds the tree', n, tree.leafAll.length);
  data[0] = 7;
  ok('refilled from clear', fillLeafDensity([], origin, dims, data) + data[0], 0);
}
ok('models with no leaves add none', placedLeafCubes([{ model: { cubes: [] }, yaw: 0, at }], out).count, 0);
