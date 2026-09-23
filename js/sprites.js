import * as THREE from 'three/webgpu';
import { isSolid, BLOCK_METRES } from './world.js';

// --- Character quad vs. world geometry ---
//
// Characters depth-test against the world normally, so one standing behind a
// wall is properly occluded by it. The problem is that the quad is a full block
// wide, so its corners sit exactly BLOCK_METRES/2 from centre - the cell
// half-width. Any yaw puts them on the voxel boundary, and the backward lean
// pushes the top edge past it, so alongside a tall block the quad genuinely
// penetrates geometry and gets sliced open.
//
// Rather than give up depth testing everywhere, detect the penetration and only
// draw on top for the frames where it is actually happening. Occlusion stays
// correct the rest of the time.

export const CHARACTER_RENDER_ORDER = 10;

// Four corners plus the top-edge midpoint. The top edge is what leans, so it is
// where penetration starts; block-scale geometry cannot pass through the quad's
// interior without touching one of these.
export function makeClipSamples(width, height) {
  return [
    new THREE.Vector3(-width / 2, 0, 0),
    new THREE.Vector3( width / 2, 0, 0),
    new THREE.Vector3(-width / 2, height, 0),
    new THREE.Vector3( width / 2, height, 0),
    new THREE.Vector3(0, height, 0)
  ];
}

const _point = new THREE.Vector3();

// Takes a world matrix rather than a mesh so it can be exercised without a
// renderer or a scene graph.
export function isQuadClipping(matrixWorld, samples) {
  for (const local of samples) {
    _point.copy(local).applyMatrix4(matrixWorld);
    // Blocks are centred on b * BLOCK_METRES, so rounding maps a world point to
    // the cell that contains it.
    if (isSolid(
      Math.round(_point.x / BLOCK_METRES),
      Math.round(_point.y / BLOCK_METRES),
      Math.round(_point.z / BLOCK_METRES)
    )) return true;
  }
  return false;
}

export function updateCharacterClipping(mesh, samples) {
  mesh.updateMatrixWorld(true); // rotation was set this frame; the matrix is stale until asked
  const clipping = isQuadClipping(mesh.matrixWorld, samples);
  if (clipping === mesh.userData.drawingOnTop) return clipping; // only touch state on a transition

  mesh.userData.drawingOnTop = clipping;
  mesh.renderOrder = clipping ? CHARACTER_RENDER_ORDER : 0;
  // Lit materials come as a prebuilt pair. Flipping depth state on a lit
  // material means a fresh pipeline - a shader compile of 100 ms or more, the
  // first time a character leans into a wall - where swapping to a variant
  // compiled in advance costs nothing.
  const v = mesh.userData.depthVariants;
  if (v) { mesh.material = clipping ? v.onTop : v.normal; return clipping; }
  mesh.material.depthTest = !clipping;
  mesh.material.depthWrite = !clipping;
  // Depth state is baked into the render pipeline, so the material must be
  // rebuilt. Gated on the transition above, so this costs nothing per frame.
  mesh.material.needsUpdate = true;
  return clipping;
}
