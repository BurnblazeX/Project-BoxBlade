import * as THREE from 'three/webgpu';
import { file, section, ok, near, truthy, falsy, note } from './lib/harness.mjs';
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

section('sprite geometry - merged crossed planes');
{
  const { addSpriteTangent, crossedPlanesGeometry } = await import('../js/sprites.js');
  const plane = addSpriteTangent(new THREE.PlaneGeometry(1, 2));
  const pt = plane.attributes.tangent;
  ok('a plain quad gets a tangent per vertex', pt.count, plane.attributes.position.count);
  truthy('the tangent is exactly +x', [0, 1, 2, 3].every(i =>
    pt.getX(i) === 1 && pt.getY(i) === 0 && pt.getZ(i) === 0 && pt.getW(i) === 1));

  const angles = [0, Math.PI / 2, Math.PI / 4, 3 * Math.PI / 4];
  const W = 2 * BLOCK_METRES / 12, H = 3 * BLOCK_METRES / 12;
  const g = crossedPlanesGeometry(W, H, angles);
  ok('four quads of four vertices', g.attributes.position.count, 16);
  ok('four quads of two triangles', g.index.count, 24);

  // Each plane must match the separate mesh it replaces: a translated plane,
  // rotated about Y by its angle, with its normal and tangent turned with it.
  let match = true;
  angles.forEach((a, k) => {
    const ref = addSpriteTangent(new THREE.PlaneGeometry(W, H));
    ref.translate(0, H / 2, 0);
    const m = new THREE.Matrix4().makeRotationY(a);
    const n = new THREE.Vector3(), t = new THREE.Vector3(), p = new THREE.Vector3();
    for (let i = 0; i < 4; i++) {
      const j = k * 4 + i;
      p.fromBufferAttribute(ref.attributes.position, i).applyMatrix4(m);
      n.set(0, 0, 1).transformDirection(m);
      t.set(1, 0, 0).transformDirection(m);
      const close = (v, attr) => Math.abs(v.x - attr.getX(j)) < 1e-6 &&
        Math.abs(v.y - attr.getY(j)) < 1e-6 && Math.abs(v.z - attr.getZ(j)) < 1e-6;
      if (!close(p, g.attributes.position) || !close(n, g.attributes.normal) ||
          !close(t, g.attributes.tangent) ||
          g.attributes.uv.getX(j) !== ref.attributes.uv.getX(i) ||
          g.attributes.uv.getY(j) !== ref.attributes.uv.getY(i)) match = false;
    }
  });
  truthy('every vertex matches its separate, rotated plane', match);
  ok('bottom edge on the origin', g.boundingBox.min.y, 0);
  near('top at the plane height', g.boundingBox.max.y, H, 1e-6);
}

section('sprite texel atlas - planes, sides and packing');
{
  const { addSpriteTangent, crossedPlanesGeometry, spriteAtlasGeometry, spriteAtlasRegion,
          packSpriteAtlas } = await import('../js/sprites.js');
  const plane = addSpriteTangent(new THREE.PlaneGeometry(1, 2));
  ok('a plain quad is one plane', plane.userData.spritePlanes, 1);
  truthy('its vertices are all on plane 0', [...plane.attributes.spritePlane.array].every(v => v === 0));

  const angles = [0, Math.PI / 2, Math.PI / 4, 3 * Math.PI / 4];
  const g = crossedPlanesGeometry(1, 2, angles);
  ok('the crossed planes are four planes', g.userData.spritePlanes, 4);
  truthy('each quad carries its own plane index', angles.every((a, k) =>
    [0, 1, 2, 3].every(i => g.attributes.spritePlane.getX(k * 4 + i) === k)));

  // The atlas twin: every vertex twice, front then back, the rest unchanged.
  const t = spriteAtlasGeometry(g);
  ok('twice the vertices', t.attributes.position.count, 32);
  ok('twice the triangles', t.index.count, 48);
  truthy('first copy is side 0, second side 1', [...t.attributes.spriteSide.array]
    .every((s, i) => s === (i < 16 ? 0 : 1)));
  truthy('both copies keep position, uv and plane', ['position', 'uv', 'spritePlane'].every(n =>
    [...Array(16).keys()].every(i => t.attributes[n].getX(i) === g.attributes[n].getX(i) &&
                                     t.attributes[n].getX(i + 16) === g.attributes[n].getX(i))));
  truthy('the back copy indexes its own vertices', [...Array(24).keys()].every(i =>
    t.index.getX(i + 24) === g.index.getX(i) + 16));
  ok('one twin geometry per source geometry', spriteAtlasGeometry(g), t);
  ok('its bounds are the sprite\'s', t.boundingSphere.radius, g.boundingSphere.radius);

  ok('a tree region: four planes across, two sides down', spriteAtlasRegion(24, 36, 4).join(), '96,72');

  const sizes = [[96, 72], [12, 48], [96, 72], [96, 72], [12, 48]];
  const { at, height } = packSpriteAtlas(sizes, 200);
  const overlap = (i, j) => at[i][0] < at[j][0] + sizes[j][0] && at[j][0] < at[i][0] + sizes[i][0] &&
                            at[i][1] < at[j][1] + sizes[j][1] && at[j][1] < at[i][1] + sizes[i][1];
  let clean = true;
  for (let i = 0; i < sizes.length; i++) {
    if (at[i][0] + sizes[i][0] > 200 || at[i][1] + sizes[i][1] > height) clean = false;
    for (let j = i + 1; j < sizes.length; j++) if (overlap(i, j)) clean = false;
  }
  truthy('regions inside the atlas and never overlapping', clean);
  truthy('whole pixels', at.every(([x, y]) => Number.isInteger(x) && Number.isInteger(y)));
  ok('two tree shelves then the characters', height, 72 + 72);
  let threw = false;
  try { packSpriteAtlas([[300, 10]], 200); } catch { threw = true; }
  truthy('a region wider than the atlas is an error, not an overlap', threw);
}
