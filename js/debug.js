import * as THREE from 'three/webgpu';
import { createBoxGridAt, gridOriginFor, isOccupied, GRID_DIM, VOXEL_METRES, TEXELS_PER_BLOCK, voxelCentreToWorld } from './boxgrid.js';

// --- boxGrid debug visualisation (Alt+X) ---
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
let isVisible = false;
let currentOrigin = null;

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

function surfaceMask(grid, x, y, z) {
  if (!isOccupied(grid, x, y, z)) return 0;
  // Bit per exposed face. A voxel exposed ONLY downwards can never be seen from
  // a camera that lives above the world, so it is skipped: that is the entire
  // underside of every slab.
  let mask = 0;
  if (!isOccupied(grid, x + 1, y, z)) mask |= 1;
  if (!isOccupied(grid, x - 1, y, z)) mask |= 2;
  if (!isOccupied(grid, x, y + 1, z)) mask |= 4;
  if (!isOccupied(grid, x, y, z + 1)) mask |= 8;
  if (!isOccupied(grid, x, y, z - 1)) mask |= 16;
  return mask;
}

function fill(mesh, grid) {
  let i = 0;
  const b = grid.filledBlocks;
  _generation = (_generation + 1) & 0xff;
  if (_generation === 0) { _seen.fill(0); _generation = 1; } // wrapped, retire old stamps

  // Visit only the voxel ranges that blocks actually filled. Scanning all
  // 144^3 cells meant ~18M neighbour lookups per rebuild; this is ~12x fewer.
  for (let k = 0; k < b.length; k += 3) {
    const vx0 = b[k], vy0 = b[k + 1], vz0 = b[k + 2];
    for (let dz = 0; dz < TEXELS_PER_BLOCK; dz++) {
      for (let dy = 0; dy < TEXELS_PER_BLOCK && vy0 + dy < GRID_DIM; dy++) {
        for (let dx = 0; dx < TEXELS_PER_BLOCK; dx++) {
          const vx = vx0 + dx, vy = vy0 + dy, vz = vz0 + dz;
          if (!surfaceMask(grid, vx, vy, vz)) continue;

          // Blocks overlap at shared faces, so the same voxel can be reached
          // from two ranges.
          const id = (vz * GRID_DIM + vy) * GRID_DIM + vx;
          if (_seen[id] === _generation) continue;
          _seen[id] = _generation;

          if (i >= MAX_INSTANCES) { mesh.count = i; return i; }
          const w = voxelCentreToWorld(grid, vx, vy, vz);
          _m.setPosition(w.x, w.y, w.z);
          mesh.setMatrixAt(i, _m);

          // Hue ramps with height so a slab's layering, and a block's 12-voxel
          // thickness, read at a glance. Lightness alternates on a 3D
          // checkerboard so neighbouring voxels never share a shade.
          const checker = ((vx + vy + vz) & 1) ? CHECKER_LIGHTNESS : 0;
          _colour.setHSL((vy / GRID_DIM) * 0.75, 0.75, 0.48 + checker);
          mesh.setColorAt(i, _colour);
          i++;
        }
      }
    }
  }
  mesh.count = i;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  return i;
}

function createMesh() {
  const s = VOXEL_METRES * SOLID_SCALE;
  // Unlit on purpose: this is an overlay reporting data, not lit geometry.
  const mesh = new THREE.InstancedMesh(
    new THREE.BoxGeometry(s, s, s), new THREE.MeshBasicMaterial(), MAX_INSTANCES);
  mesh.frustumCulled = false;
  mesh.setColorAt(0, _colour); // allocates the instanceColor buffer up front
  return mesh;
}

function disposeMeshes(scene) {
  if (!solidMesh) return;
  scene.remove(solidMesh);
  solidMesh.geometry.dispose();
  solidMesh.material.dispose();
  solidMesh.dispose();
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

function beginRebuild(scene, origin, mode) {
  cachedGrid = createBoxGridAt(origin.x, origin.z, cachedGrid);
  if (!solidMesh) {
    solidMesh = createMesh();
    scene.add(solidMesh);
  }
  pendingOrigin = origin;
  pendingMode = mode;
  currentOrigin = origin; // claimed now, so the next frame doesn't re-trigger
}

function finishRebuild() {
  const count = fill(solidMesh, cachedGrid);
  console.log(
    `[boxGrid debug] ${pendingMode} | origin (${pendingOrigin.x},${pendingOrigin.z}) | ` +
    `${GRID_DIM}^3 @ ${VOXEL_METRES * 100}cm | ` +
    `occupied ${cachedGrid.occupiedCount.toLocaleString()} | visible surface ${count.toLocaleString()} drawn`
  );
  pendingOrigin = null;
  pendingMode = null;
  return count;
}

// Both halves at once. Used on enable, where there is nothing on screen yet and
// splitting would just show an empty frame.
function rebuild(scene, origin, mode) {
  beginRebuild(scene, origin, mode);
  return finishRebuild();
}

export function toggleBoxGridDebug(scene, playerGridPos, terrainMesh, mode = 'explore') {
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

  rebuild(scene, gridOriginFor(mode, playerGridPos), mode);

  // The terrain hides while the overlay is up. A voxel's top face is exactly
  // coplanar with its block's top face, so drawing both would z-fight across
  // the whole ground plane; substituting the view also makes it unambiguous
  // that what is on screen is grid data, not the world.
  if (terrainMesh) terrainMesh.visible = false;
  return true;
}

// Called every frame. Cheap to call: compares two integers and returns unless
// the grid origin actually moved, or a split rebuild is half done. In explore
// that is once per block stepped (the footprint follows the player); in battle
// it is chunk-aligned and static for the whole fight.
export function refreshBoxGridDebug(scene, playerGridPos, mode = 'explore') {
  if (!isVisible) return false;

  // Second half of a split rebuild takes priority, so a rebuild always
  // completes before another can start.
  if (pendingOrigin) { finishRebuild(); return true; }

  const origin = gridOriginFor(mode, playerGridPos);
  if (currentOrigin && origin.x === currentOrigin.x && origin.z === currentOrigin.z) return false;
  beginRebuild(scene, origin, mode);
  return true;
}

export function isBoxGridDebugVisible() {
  return isVisible;
}

export function getBoxGridDebugOrigin() {
  return currentOrigin;
}
