import { file, section, ok, near, truthy } from './lib/harness.mjs';
import { GRID_DIM, VOXEL_METRES } from '../js/boxgrid.js';
import {
  LPV_DIM, LPV_CELLS, LPV_CELL_VOXELS, LPV_CELL_METRES, LPV_EXTENT_METRES,
  cellIndex, cellCoords, propagateStep, lpvShift, shiftVolume, averageAlbedo,
  diffuseAlbedo, boxCellOverlap
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

section('diffuse albedo per material');
{
  // Two texels, one mid grey dielectric and one metal: only the dielectric
  // bounces, weighted by the half of the surface it is.
  const rgba = new Uint8Array([188, 188, 188, 255, 188, 188, 188, 255]);
  const all = diffuseAlbedo(rgba);
  const half = diffuseAlbedo(rgba, new Uint8Array([0, 10, 0, 255, 240, 230, 0, 255]));
  near('no _s: the plain mean', all.r, averageAlbedo(rgba).r, 1e-9);
  near('half metal bounces half', half.r, all.r / 2, 1e-9);
  ok('all metal bounces nothing',
     diffuseAlbedo(rgba, new Uint8Array([240, 230, 0, 255, 240, 230, 0, 255])).g, 0);
}

section('sprite box overlap');
{
  near('a box covering the cell fills it', boxCellOverlap([0, 0, 0], 1, [0.5, 0.5, 0.5], [1, 1, 1]), 1, 1e-9);
  near('half in x', boxCellOverlap([0, 0, 0], 1, [0, 0.5, 0.5], [0.5, 1, 1]), 0.5, 1e-9);
  ok('clear of the cell is zero', boxCellOverlap([0, 0, 0], 1, [3, 0.5, 0.5], [0.5, 0.5, 0.5]), 0);
}

section('specular share in the bounce (specularAlbedo)');
{
  const { specularAlbedo, glassCellTransmit, propagateStep, LPV_CELLS } = await import('../js/lpv.js');
  const tex = (n, rgba) => { const a = new Uint8Array(n * 4); for (let i = 0; i < n; i++) a.set(rgba, i * 4); return a; };
  const albedo = tex(4, [128, 128, 128, 255]);
  // Polished iron (LabPBR metal 230, smoothness 255): its F0, not nothing.
  const iron = specularAlbedo(albedo, tex(4, [255, 230, 0, 255]));
  truthy('polished iron bounces its reflectance', iron.r > 0.4 && iron.g > 0.4);
  // The same iron with the mirror light on: smooth, so all of it is the
  // mirror light's to carry - none left for here.
  const ironMirrored = specularAlbedo(albedo, tex(4, [255, 230, 0, 255]), { mirrored: true });
  near('... but none of it while the mirror light carries it', ironMirrored.r, 0, 1e-9);
  // Rough iron (smoothness 0): too rough for the mirror light, so all of it stays.
  const rough = specularAlbedo(albedo, tex(4, [0, 230, 0, 255]), { mirrored: true });
  near('rough iron keeps its share either way', rough.r, iron.r, 1e-9);
  // A dielectric with no _s: the default grey 0.04.
  near('no specular texture: 0.04', specularAlbedo(albedo).g, 10 / 255, 1e-9);

  section('glass tints the light the LPV carries through it');
  const clear = glassCellTransmit({ r: 1, g: 1, b: 1 }, 0.75, 1);
  near('clear glass passes everything', clear.r, 1, 1e-9);
  const red = glassCellTransmit({ r: 0.8, g: 0.2, b: 0.2 }, 0.75, 1);
  near('a half-block cell of red glass: tint^0.5', red.g, Math.sqrt(0.2), 1e-9);
  near('no glass in the cell: untouched', glassCellTransmit({ r: 0.8, g: 0.2, b: 0.2 }, 0.75, 0).g, 1, 1e-9);

  // propagateStep with trans: the tinted cell's light is multiplied by it.
  const dim = 3, cells = dim * dim * dim;
  const L = new Float32Array(cells * 4).fill(1), E = new Float32Array(cells * 4);
  const solid = new Float32Array(cells), out = new Float32Array(cells * 4);
  const trans = new Float32Array(cells * 4).fill(1);
  const mid = 1 + 1 * dim + 1 * dim * dim;
  trans[mid * 4 + 1] = 0.5;
  propagateStep(L, E, solid, 0.5, out, dim, trans);
  near('the glass cell\'s green is halved', out[mid * 4 + 1], out[mid * 4] * 0.5, 1e-9);
  const plain = new Float32Array(cells * 4);
  propagateStep(L, E, solid, 0.5, plain, dim);
  near('without trans nothing changes', plain[mid * 4], out[mid * 4], 1e-9);
}
