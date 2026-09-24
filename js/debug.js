import * as THREE from 'three/webgpu';
import { createBoxGridAt, gridOriginFor, isOccupied, isGlassVoxel, GRID_DIM,
         voxelCentreToWorld, CASCADE_COUNT, cascadeVoxelMetres, cascadeBlocks,
         cascadeExtentMetres, SUN_BIAS_BLOCKS } from './boxgrid.js';

// --- boxGrid debug visualisation (Alt+G) ---
//
// Draws the occupancy grid as it actually exists in memory: one cube per
// occupied voxel at 12.5 cm, with a checkerboard tint so individual voxels stay
// distinguishable where they form a continuous surface.
// If voxelisation ever lands at the wrong offset or scale it is obvious here
// and invisible everywhere else.
//
// Only SURFACE voxels are drawn - occupied with at least one empty face
// neighbour. Interior voxels cannot be seen from outside and would multiply the
// instance count for nothing. Out-of-bounds counts as empty, so voxels on the
// grid boundary are surface too, which is what makes the grid's extent visible.

// A blue wireframe box per voxel was tried and reverted: ~45k instances of line
// geometry tanked the framerate. Voxels are distinguished instead by a
// checkerboard lightness flip on the per-instance colour, which costs nothing -
// no extra geometry, no extra draw call - combined with the gap left by drawing
// each cube smaller than its cell.
const SOLID_SCALE = 0.82;
const CHECKER_LIGHTNESS = 0.14;

let solidMesh = null;
const meshes = [];
let clipBoxes = [];
let isVisible = false;
let currentOrigin = null;

// ONE OVERLAY, BOTH CASCADES, NO OVERLAP.
//
// C0 is drawn where it exists and C1 only OUTSIDE C0's footprint - subtracted,
// not layered. They occupy the same world space, so drawing both in full would
// z-fight everywhere a ray actually marches the fine field, and the coarse
// cubes would hide the fine ones they enclose. Subtracting makes the overlay a
// picture of what a ray really sees: fine near the player, coarse further out,
// and the seam between them visible as the exact place the resolution halves.


function isSurfaceVoxel(grid, x, y, z) {
  if (!isOccupied(grid, x, y, z)) return false;
  return !isOccupied(grid, x + 1, y, z) || !isOccupied(grid, x - 1, y, z)
      || !isOccupied(grid, x, y + 1, z) || !isOccupied(grid, x, y - 1, z)
      || !isOccupied(grid, x, y, z + 1) || !isOccupied(grid, x, y, z - 1);
}

// Preallocated once. Crossing a chunk boundary then costs a matrix refill and
// a count change, not a fresh geometry, material and buffer allocation.
const MAX_INSTANCES = 70000;
let cachedGrid = null;
// Dedupe stamp: blocks share faces, so the same voxel is reachable from two
// ranges. A Set meant up to 250k hashed insertions per rebuild - by far the
// largest cost in the refill. A reused byte array plus a generation counter is
// O(1) with no allocation and no hashing.
const _seen = new Uint8Array(GRID_DIM * GRID_DIM * GRID_DIM);
let _generation = 0;
const _m = new THREE.Matrix4();
const _colour = new THREE.Color();

// occ: which channel's occupancy - the opaque field's, or the glass one's.
function surfaceMask(grid, x, y, z, occ = isOccupied) {
  if (!occ(grid, x, y, z)) return 0;
  // Bit per exposed face. A voxel exposed ONLY downwards can never be seen from
  // a camera that lives above the world, so it is skipped: that is the entire
  // underside of every slab.
  let mask = 0;
  if (!occ(grid, x + 1, y, z)) mask |= 1;
  if (!occ(grid, x - 1, y, z)) mask |= 2;
  if (!occ(grid, x, y + 1, z)) mask |= 4;
  if (!occ(grid, x, y, z + 1)) mask |= 8;
  if (!occ(grid, x, y, z - 1)) mask |= 16;
  return mask;
}

// Glass is drawn from the field's glass channel, pale cyan, so it reads as
// glass among the height-ramped rock.
const GLASS_HUE = 0.5, GLASS_SAT = 0.55, GLASS_LIGHT = 0.72;

function fill(mesh, grid, clip = null) {
  let i = 0;
  // Per LEVEL, not the C0 constant: a coarse cascade fills six voxels per block
  // per axis, and walking twelve would run off the end of each block's range
  // into its neighbour's.
  const per = grid.voxelsPerBlock;
  _generation = (_generation + 1) & 0xff;
  if (_generation === 0) { _seen.fill(0); _generation = 1; } // wrapped, retire old stamps

  // Rock first, then glass: each from its own block list and its own channel.
  for (const glass of [false, true]) {
  const b = glass ? (grid.glassBlocks || []) : grid.filledBlocks;
  const occ = glass ? isGlassVoxel : isOccupied;
  // Visit only the voxel ranges that blocks actually filled. Scanning all
  // 144^3 cells meant ~18M neighbour lookups per rebuild; this is ~12x fewer.
  for (let k = 0; k < b.length; k += 3) {
    const vx0 = b[k], vy0 = b[k + 1], vz0 = b[k + 2];
    for (let dz = 0; dz < per; dz++) {
      for (let dy = 0; dy < per && vy0 + dy < GRID_DIM; dy++) {
        for (let dx = 0; dx < per; dx++) {
          const vx = vx0 + dx, vy = vy0 + dy, vz = vz0 + dz;
          // Blocks outside the footprint are baked now, for their band alone -
          // see populateDistanceField - so their voxel ranges can start outside
          // the grid and must be skipped rather than wrapped.
          if (vx < 0 || vy < 0 || vz < 0 ||
              vx >= GRID_DIM || vz >= GRID_DIM) continue;
          if (!surfaceMask(grid, vx, vy, vz, occ)) continue;

          // Blocks overlap at shared faces, so the same voxel can be reached
          // from two ranges.
          const id = (vz * GRID_DIM + vy) * GRID_DIM + vx;
          if (_seen[id] === _generation) continue;
          _seen[id] = _generation;

          if (i >= MAX_INSTANCES) { mesh.count = i; return i; }
          const w = voxelCentreToWorld(grid, vx, vy, vz);
          // The subtraction. A coarse voxel whose centre falls inside C0's
          // footprint is already drawn, finer, by C0.
          if (clip && w.x > clip.x0 && w.x < clip.x1 &&
              w.z > clip.z0 && w.z < clip.z1) continue;
          _m.setPosition(w.x, w.y, w.z);
          mesh.setMatrixAt(i, _m);

          // Hue ramps with height so a slab's layering, and a block's 12-voxel
          // thickness, read at a glance. Lightness alternates on a 3D
          // checkerboard so neighbouring voxels never share a shade.
          const checker = ((vx + vy + vz) & 1) ? CHECKER_LIGHTNESS : 0;
          if (glass) _colour.setHSL(GLASS_HUE, GLASS_SAT, GLASS_LIGHT + checker * 0.5);
          else _colour.setHSL((vy / GRID_DIM) * 0.75, 0.75, 0.48 + checker);
          mesh.setColorAt(i, _colour);
          i++;
        }
      }
    }
  }
  }
  mesh.count = i;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return i;
}

function createMesh(voxelMetres) {
  const s = voxelMetres * SOLID_SCALE;
  // Unlit on purpose: this is an overlay reporting data, not lit geometry.
  const mesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(s, s, s), new THREE.MeshBasicMaterial(), MAX_INSTANCES);
  mesh.frustumCulled = false;
  mesh.setColorAt(0, _colour); // allocates the instanceColor buffer up front
  return mesh;
}

function disposeMeshes(scene) {
  for (const m of meshes) {
    if (!m) continue;
    scene.remove(m);
    m.geometry.dispose();
    m.material.dispose();
    m.dispose();
  }
  meshes.length = 0;
  solidMesh = null;
}

// Re-originating is split across two frames rather than done in one: the
// occupancy repopulate lands on one frame and the mesh refill on the next.
// Both halves are similar in cost, so alternating roughly halves the worst
// frame instead of paying for everything at once. The mesh shows the previous
// origin's voxels for that one intervening frame, which at walking pace is not
// perceptible.
let pendingOrigin = null;
let pendingMode = null;
let pendingPlayer = null;

// One cached grid PER LEVEL. createBoxGridAt() only initialises voxelSize,
// range and voxelsPerBlock when it allocates, so handing it a C0 buffer and
// asking for C1 would repopulate at the coarse stride while the grid still
// claimed to be fine - a field that is wrong by a factor of two in a way that
// looks plausible. They are also cheap to keep: two 3 MB buffers, and the debug
// overlay is not on during play.
const cachedGrids = [];

function beginRebuild(scene, origin, mode, sunDirection, bias) {
  for (let l = 0; l < CASCADE_COUNT; l++) {
    const o = l === 0 ? origin : gridOriginFor(mode, pendingPlayer, sunDirection, bias, l);
    cachedGrids[l] = createBoxGridAt(o.x, o.z, cachedGrids[l] || null, l);
  }
  cachedGrid = cachedGrids[0];
  // Each level is clipped by the one finer than it, so the ladder subtracts all
  // the way out: C0 solid, C1 filling the ring around it, C2 the ring around
  // that. Generic in CASCADE_COUNT rather than written twice, because a third
  // level arriving and being silently invisible is exactly what this overlay is
  // meant to catch.
  clipBoxes = cachedGrids.map((g, l) => {
    const side = cascadeExtentMetres(l);
    return { x0: g.origin.x, x1: g.origin.x + side,
             z0: g.origin.z, z1: g.origin.z + side };
  });
  if (!meshes.length) {
    for (let l = 0; l < CASCADE_COUNT; l++) {
      meshes[l] = createMesh(cascadeVoxelMetres(l));
      scene.add(meshes[l]);
    }
    solidMesh = meshes[0];
  }
  pendingOrigin = origin;
  pendingMode = mode;
  currentOrigin = origin;
}

function finishRebuild() {
  const counts = cachedGrids.map((g, l) => fill(meshes[l], g, l === 0 ? null : clipBoxes[l - 1]));
  const count = counts[0];
  console.log(
    `[boxGrid debug] ${pendingMode} | origin (${pendingOrigin.x},${pendingOrigin.z}) | ` +
    counts.map((c, l) =>
      `C${l} ${(cascadeVoxelMetres(l) * 100).toFixed(1)}cm to ` +
      `${cascadeExtentMetres(l)}m: ${c.toLocaleString()}`).join(' | ')
  );
  pendingOrigin = null;
  pendingMode = null;
  return count;
}

// Both halves at once. Used on enable, where there is nothing on screen yet and
// splitting would just show an empty frame.
function rebuild(scene, origin, mode, sunDirection, bias) {
  beginRebuild(scene, origin, mode, sunDirection, bias);
  return finishRebuild();
}

export function toggleBoxGridDebug(scene, playerGridPos, terrainMesh,
                                   mode = 'explore', opts = {}) {
  const { sunDirection = null, bias = SUN_BIAS_BLOCKS } = opts;
  isVisible = !isVisible;

  if (!isVisible) {
    disposeMeshes(scene);
    currentOrigin = null;
    pendingOrigin = null;
    pendingMode = null;
    if (terrainMesh) terrainMesh.visible = true;
    console.log('[boxGrid debug] off');
    return false;
  }

  pendingPlayer = playerGridPos;
  rebuild(scene, originFor(mode, playerGridPos, sunDirection, bias), mode,
          sunDirection, bias);

  // The terrain hides while the overlay is up. A voxel's top face is exactly
  // coplanar with its block's top face, so drawing both would z-fight across
  // the whole ground plane; substituting the view also makes it unambiguous
  // that what is on screen is grid data, not the world.
  if (terrainMesh) terrainMesh.visible = false;
  return `boxGrid overlay on - C0 at ${(cascadeVoxelMetres(0) * 100).toFixed(1)} cm ` +
         `over ${cascadeExtentMetres(0)} m, C1 at ` +
         `${(cascadeVoxelMetres(1) * 100).toFixed(1)} cm filling out to ` +
         `${cascadeExtentMetres(1)} m around it. C1 is drawn with C0 subtracted, ` +
         `so the seam IS where a ray's resolution halves.`;
}

// THE SAME ORIGIN THE MARCHER USES, sun lean and coarse-level snap included.
//
// This used to call gridOriginFor() with no sun, so the overlay drew a footprint
// CENTRED on the player while the field being marched leaned two blocks up-sun.
// An overlay whose whole job is "the field is where you think it is" was
// answering a question nobody asked, and it would have been worse at C1, which
// snaps to a two-block stride on top of the lean.
function originFor(mode, playerGridPos, sunDirection, bias) {
  return gridOriginFor(mode, playerGridPos, sunDirection, bias, 0);
}

// Called every frame. Cheap to call: compares two integers and returns unless
// the grid origin actually moved, or a split rebuild is half done. In explore
// that is once per block stepped (the footprint follows the player); in battle
// it is chunk-aligned and static for the whole fight.
export function refreshBoxGridDebug(scene, playerGridPos, mode = 'explore',
                                    opts = {}) {
  if (!isVisible) return false;

  // Second half of a split rebuild takes priority, so a rebuild always
  // completes before another can start.
  if (pendingOrigin) { finishRebuild(); return true; }

  const { sunDirection = null, bias = SUN_BIAS_BLOCKS } = opts;
  const origin = originFor(mode, playerGridPos, sunDirection, bias);
  if (currentOrigin && origin.x === currentOrigin.x && origin.z === currentOrigin.z) return false;
  pendingPlayer = playerGridPos;
  beginRebuild(scene, origin, mode, sunDirection, bias);
  return true;
}

export function isBoxGridDebugVisible() {
  return isVisible;
}

export function getBoxGridDebugOrigin() {
  return currentOrigin;
}
