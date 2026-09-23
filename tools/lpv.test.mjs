import { file, section, ok, near, truthy } from './lib/harness.mjs';
import { GRID_DIM, VOXEL_METRES } from '../js/boxgrid.js';
import {
  LPV_DIM, LPV_CELLS, LPV_CELL_VOXELS, LPV_CELL_METRES, LPV_EXTENT_METRES,
  cellIndex, cellCoords, propagateStep, lpvShift, shiftVolume, averageAlbedo
} from '../js/lpv.js';

file('lpv.test.mjs - the light propagation volume, CPU reference');

section('the volume sits exactly on C0');
ok('24 cells a side', LPV_DIM, 24);
ok('six voxels a cell', LPV_CELL_VOXELS * LPV_DIM, GRID_DIM);
near('0.75 m cells', LPV_CELL_METRES, 6 * VOXEL_METRES, 1e-12);
near('18 m across, as C0', LPV_EXTENT_METRES, GRID_DIM * VOXEL_METRES, 1e-12);
{
  const c = cellCoords(cellIndex(3, 7, 19));
  ok('index round-trips', [c.x, c.y, c.z].join(), '3,7,19');
}

// Iterate to convergence on a small volume - the rule does not depend on size.
const dim = 8, n = dim ** 3;
const run = (E, solid, alpha, iters = 400) => {
  let a = new Float32Array(n * 4), b = new Float32Array(n * 4);
  for (let i = 0; i < iters; i++) { propagateStep(a, E, solid, alpha, b, dim); [a, b] = [b, a]; }
  return a;
};
const at = (x, y, z) => x + y * dim + z * dim * dim;

section('steady state is normalised by (1 - spread)');
{
  const E = new Float32Array(n * 4), solid = new Float32Array(n);
  for (let i = 0; i < n; i++) E[i * 4] = 0.2;
  for (const alpha of [0.5, 0.9]) {
    const L = run(E, solid, alpha);
    // Off-volume neighbours hold the cell's own value, so uniform injection is
    // uniform everywhere - no dark rim at the edge of the volume.
    near(`uniform injection, spread ${alpha}: corner reads E after normalising`,
         L[at(0, 0, 0) * 4] * (1 - alpha), 0.2, 1e-4);
    near(`and so does the middle`, L[at(4, 4, 4) * 4] * (1 - alpha), 0.2, 1e-4);
  }
}

section('light spreads from a source, falling off with distance');
{
  const E = new Float32Array(n * 4), solid = new Float32Array(n);
  E[at(1, 4, 4) * 4 + 1] = 1;
  const L = run(E, solid, 0.9);
  const g = x => L[at(x, 4, 4) * 4 + 1];
  truthy('brightest at the source', g(1) > g(2));
  truthy('monotonic away from it', g(2) > g(4) && g(4) > g(6));
  truthy('but it does reach across', g(6) > 0);
  ok('only the channel injected', L[at(3, 4, 4) * 4], 0);
}

section('solid cells block, and hold nothing');
{
  const E = new Float32Array(n * 4), solid = new Float32Array(n);
  // A full wall at x = 4 between a source at x = 1 and the far side.
  for (let y = 0; y < dim; y++) for (let z = 0; z < dim; z++) solid[at(4, y, z)] = 1;
  E[at(1, 4, 4) * 4] = 1;
  const L = run(E, solid, 0.9);
  ok('the wall itself is dark', L[at(4, 4, 4) * 4], 0);
  ok('nothing leaks through a full wall', L[at(6, 4, 4) * 4], 0);
  truthy('the near side is lit', L[at(3, 4, 4) * 4] > 0);
}

section('shifting with C0');
{
  const s = lpvShift({ x: 0, y: -4.5, z: 0 }, { x: 1.5, y: -4.5, z: -3 });
  ok('one block is two cells', [s.x, s.y, s.z].join(), '2,0,-4');
  const s1 = lpvShift({ x: 0, y: 0, z: 0 }, { x: 3, y: 0, z: -1.5 }, 1.5);
  ok('over C1 a cell is a block', [s1.x, s1.y, s1.z].join(), '2,0,-1');
  ok('half a C1 cell starts over', lpvShift({ x: 0, y: 0, z: 0 }, { x: 0.75, y: 0, z: 0 }, 1.5), null);
  ok('a non-cell move starts over', lpvShift({ x: 0, y: 0, z: 0 }, { x: 0.3, y: 0, z: 0 }), null);
  ok('a jump past the volume starts over', lpvShift({ x: 0, y: 0, z: 0 }, { x: 18, y: 0, z: 0 }), null);
  ok('no prior origin starts over', lpvShift(null, { x: 0, y: 0, z: 0 }), null);

  const src = new Float32Array(n);
  for (let i = 0; i < n; i++) src[i] = i;
  const out = shiftVolume(src, { x: 2, y: 0, z: 0 }, new Float32Array(n), 1, dim);
  ok('cell c takes old cell c + shift', out[at(0, 3, 3)], at(2, 3, 3));
  ok('what slid in is empty', out[at(dim - 1, 3, 3)], 0);
}

section('albedo from a texture');
{
  const px = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255, 9, 9, 9, 0]);
  const a = averageAlbedo(px);
  near('mean in linear light', a.r, 0.5, 1e-6);
  near('blue too', a.b, 0.5, 1e-6);
  ok('transparent pixels are ignored', a.g, 0);
  truthy('sRGB mid-grey linearises dark', averageAlbedo(new Uint8Array([128, 128, 128, 255])).r < 0.25);
}
ok('cell count', LPV_CELLS, 13824);
