// Not a test - a measurement, run by hand. The widened field was justified by
// three numbers, so those numbers should be reproducible rather than remembered:
// what the bake now costs, how much of the field actually carries a usable
// distance, and how many steps a ray takes to cross open air.
//
//   node tools/field.bench.mjs
import { World, getVoxelKey } from '../js/world.js';
import { createBoxGridAt, populateDistanceField, sphereTrace, decodeDistance,
         GRID_DIM, VOXEL_METRES, DISTANCE_RANGE, CASCADE_COUNT,
         cascadeVoxelMetres, cascadeBlocks, cascadeExtentMetres,
         cascadeRangeVoxels } from '../js/boxgrid.js';

function groundWorld(n = 12) {
  World.clear();
  for (let x = 0; x < n; x++) {
    for (let z = 0; z < n; z++)
      World.set(getVoxelKey(x, 0, z), { solid: true, walkable: true, occupant: null });
    for (let y = 1; y <= 3; y++)
      World.set(getVoxelKey(x, y, 6), { solid: true, walkable: false, occupant: null });
  }
  return World.size;
}

const blocks = groundWorld(24);   // wide enough to fill C1's footprint too

function measure(level) {
  const grid = createBoxGridAt(0, 0, null, level);
  for (let i = 0; i < 5; i++) populateDistanceField(grid);
  const N = 20;
  const t0 = performance.now();
  for (let i = 0; i < N; i++) {
    grid.originBlock.x = i % 2;   // force a real re-bake, not a no-op
    populateDistanceField(grid);
  }
  const ms = (performance.now() - t0) / N;
  createBoxGridAt(0, 0, grid, level);

  let saturated = 0, graded = 0, solid = 0;
  const range = grid.range;
  // The opaque distance only: every other byte (G, glass) is interleaved.
  const voxels = grid.data.length / 2;
  for (let i = 0; i < grid.data.length; i += 2) {
    const b = grid.data[i];
    if (b === 255) saturated++;
    else if (b < 128) solid++;
    else graded++;
  }
  const pct = n => +(n / voxels * 100).toFixed(1);
  // C1 snaps to a 2-block stride, so it re-bakes half as often as C0 - what
  // matters per block stepped is the amortised figure, not the raw one.
  const everyNBlocks = 1 << level;
  return {
    level,
    voxelMetres: cascadeVoxelMetres(level),
    extentMetres: cascadeExtentMetres(level),
    rangeVoxels: cascadeRangeVoxels(level),
    bakeMs: +ms.toFixed(2),
    rebuildsEveryNBlocks: everyNBlocks,
    amortisedMsPerBlockStep: +(ms / everyNBlocks).toFixed(2),
    gradedExterior: pct(graded) + '%',
    saturatedAtMax: pct(saturated) + '%',
    solid: pct(solid) + '%'
  };
}

const levels = [];
for (let l = 0; l < CASCADE_COUNT; l++) levels.push(measure(l));
const grid = createBoxGridAt(0, 0, null, 0);

// Steps to cross the footprint diagonally through open air, which is what a sun
// ray does. At the old +-1 voxel clamp every step was one voxel.
function stepsToCross() {
  const o = { x: 0.2, y: 8, z: 0.2 };
  const d = { x: 0.577, y: 0.577, z: 0.577 };
  return sphereTrace(grid, o, d, 24).steps;
}

console.log(JSON.stringify({
  blocks,
  gridVoxels: GRID_DIM ** 3,
  levels,
  totalAmortisedMsPerBlockStep:
    +levels.reduce((a, l) => a + l.amortisedMsPerBlockStep, 0).toFixed(2),
  reachMetres: cascadeExtentMetres(CASCADE_COUNT - 1),
  maxStoredVoxels: +decodeDistance(255).toFixed(2),
  stepsAcrossOpenAir: stepsToCross(),
  // The threshold the range was chosen for: past this much clearance the cone
  // trace saturates and open sky would start reading as shadowed. Constant
  // across levels because the range is constant in WORLD space.
  coneSaturatesAtMetres: +((DISTANCE_RANGE * VOXEL_METRES) / Math.tan(0.12 / 2)).toFixed(1)
}, null, 2));
