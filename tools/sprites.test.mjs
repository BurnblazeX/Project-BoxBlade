import * as THREE from 'three/webgpu';
import { file, section, ok, truthy, falsy, note } from './lib/harness.mjs';
import { createTestArea, BLOCK_METRES } from '../js/world.js';
import { makeClipSamples, isQuadClipping, updateCharacterClipping, CHARACTER_RENDER_ORDER } from '../js/sprites.js';

file('sprites.test.mjs - character quad vs world geometry');

createTestArea(36, 36);
const samples = makeClipSamples(BLOCK_METRES, BLOCK_METRES * 2);
const TILT = THREE.MathUtils.degToRad(22.5);
const HEADINGS = [0, Math.PI / 2, Math.PI, 3 * Math.PI / 2, Math.PI / 4, 3 * Math.PI / 4];

// Builds the same world matrix main.js produces: positioned on the standing
// block's top face, yawed to the camera heading, leaned back by the tilt, in
// YXZ order.
function quadMatrix(gridPos, heading, tilt) {
  const obj = new THREE.Object3D();
  obj.rotation.order = 'YXZ';
  obj.position.set(
    gridPos.x * BLOCK_METRES,
    gridPos.y * BLOCK_METRES + BLOCK_METRES / 2,
    gridPos.z * BLOCK_METRES
  );
  obj.rotation.set(-tilt, heading, 0);
  obj.updateMatrixWorld(true);
  return obj.matrixWorld;
}

const clipsAtAnyHeading = (gridPos) =>
  HEADINGS.some(h => isQuadClipping(quadMatrix(gridPos, h, TILT), samples));
const clipsAtEveryHeading = (gridPos) =>
  HEADINGS.every(h => isQuadClipping(quadMatrix(gridPos, h, TILT), samples));

section('sample points');
ok('four corners plus the top midpoint', samples.length, 5);
ok('corners sit at half a block from centre', Math.abs(samples[0].x), BLOCK_METRES / 2);
ok('top samples at two blocks up', samples[2].y, BLOCK_METRES * 2);

section('open ground - depth testing must be preserved');
// The test world is bare ground away from the wall (x=20), pillar (26,20) and
// platform (28-31, 28-31).
falsy('standing in the open never clips', clipsAtAnyHeading({ x: 5, y: 0, z: 5 }));
falsy('nor does another open tile', clipsAtAnyHeading({ x: 10, y: 0, z: 25 }));
falsy('nor one a single tile from the wall line', clipsAtAnyHeading({ x: 18, y: 0, z: 11 }));

section('beside tall geometry - the case that was getting sliced');
// Wall occupies x=20, z=8..14, y=1..2. Standing directly alongside it, the
// yawed-and-leaned quad reaches into the wall cell at some headings.
truthy('adjacent to the 2-tall wall clips at some heading', clipsAtAnyHeading({ x: 19, y: 0, z: 11 }));
truthy('adjacent on the far side too', clipsAtAnyHeading({ x: 21, y: 0, z: 11 }));
truthy('beside the 3-tall pillar clips', clipsAtAnyHeading({ x: 25, y: 0, z: 20 }));

section('the fix is conditional, not blanket');
// The whole point of the correction: being near geometry must not disable depth
// testing at every heading, or occlusion is lost just as it was before.
falsy('open ground does not clip at any heading', clipsAtEveryHeading({ x: 5, y: 0, z: 5 }));
const wallAdjacentHeadings = HEADINGS.filter(h => isQuadClipping(quadMatrix({ x: 19, y: 0, z: 11 }, h, TILT), samples));
note(`beside the wall, ${wallAdjacentHeadings.length} of ${HEADINGS.length} headings clip`);
truthy('some headings clip beside the wall', wallAdjacentHeadings.length > 0);

section('material state transitions');
const mesh = {
  matrixWorld: quadMatrix({ x: 5, y: 0, z: 5 }, 0, TILT),
  updateMatrixWorld() {},
  material: { depthTest: true, depthWrite: true, needsUpdate: false },
  userData: { drawingOnTop: false },
  renderOrder: 0
};
updateCharacterClipping(mesh, samples);
truthy('open ground keeps depth testing on', mesh.material.depthTest);
truthy('and keeps depth writing on', mesh.material.depthWrite);
ok('and stays at default render order', mesh.renderOrder, 0);
falsy('no pipeline rebuild when nothing changed', mesh.material.needsUpdate);

// Pick a heading that actually clips rather than assuming one, so the
// transition asserts below always run.
const clippingHeading = HEADINGS.find(h => isQuadClipping(quadMatrix({ x: 19, y: 0, z: 11 }, h, TILT), samples));
mesh.matrixWorld = quadMatrix({ x: 19, y: 0, z: 11 }, clippingHeading, TILT);
const nowClipping = updateCharacterClipping(mesh, samples);
if (nowClipping) {
  falsy('clipping turns depth testing off', mesh.material.depthTest);
  ok('and promotes render order', mesh.renderOrder, CHARACTER_RENDER_ORDER);
  truthy('and rebuilds the pipeline once', mesh.material.needsUpdate);

  mesh.material.needsUpdate = false;
  updateCharacterClipping(mesh, samples);
  falsy('no rebuild while the state holds', mesh.material.needsUpdate);

  mesh.matrixWorld = quadMatrix({ x: 5, y: 0, z: 5 }, 0, TILT);
  updateCharacterClipping(mesh, samples);
  truthy('moving clear restores depth testing', mesh.material.depthTest);
  ok('and restores render order', mesh.renderOrder, 0);
} else {
  note('SKIPPED transition asserts: that heading did not clip');
}
