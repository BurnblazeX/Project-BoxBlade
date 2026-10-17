import * as THREE from 'three/webgpu';
import { file, section, ok, near } from './lib/harness.mjs';
import { occlusionFrame, cellCut, pixelCut, cellAcross, cellRadial, playerHidden, hidesAt, cutTargets,
         DEFAULT_OCCLUSION } from '../js/core/occlusion.js';
import { maskOrigin, maskIndex, playerCell } from '../js/core/interior.js';
import { World, newBlock, BLOCK_METRES as B } from '../js/core/world.js';
import { CAMERA_CONFIGS } from '../js/render/camera.js';
import { occlusion, cellMask } from '../js/gpu/occlusion.js';
import { createOcclusion } from '../js/render/features/occlusion.js';

file('occlusion.test.mjs - the cut-out between the camera and the player');

section("the cut set (cells in blocks from the player's; +z toward the camera)");
{
  const { pitch, distance } = CAMERA_CONFIGS.explore;
  const ft = { x: 0, y: 0.5 * B, z: 0 };   // on the ground layer's top
  const ey = { x: 0, y: ft.y + distance * Math.sin(pitch), z: distance * Math.cos(pitch) };
  const fl = Math.hypot(ey.x - ft.x, ey.y - ft.y, ey.z - ft.z);
  const fw = { x: (ft.x - ey.x) / fl, y: (ft.y - ey.y) / fl, z: (ft.z - ey.z) / fl };
  const fr = occlusionFrame(ey, fw, ft);
  const may = (x, y, z) => cellCut({ x: x * B, y: y * B, z: z * B }, fr, DEFAULT_OCCLUSION);
  ok('may be cut: a wall in front, at and above their level, and the roof', may(0, 1, 1) + may(0, 2, 1) + may(0, 3, 0), 3);
  ok('never: the floor, a wall behind, far off the sight line', may(0, 0, 1) + may(0, 2, -1) + may(5, 2, 1), 0);
  // Per pixel, at points on faces (blocks): how cut, before the circle.
  const at = (x, y, z) => pixelCut({ x: x * B, y: y * B, z: z * B }, fr);
  ok("the face of a wall in front, above the player's head", at(0, 2.2, 1.5), 1);
  ok('the top of the floor ahead', at(0, 0.5, 1.5), 0);
  ok('a wall behind, however high', at(0, 2.5, -0.5) + at(0, 4, -0.5), 0);
  ok('the camera side of a block beside them, at their height', at(1, 1, 0.5), 0);
  const mid = at(0, 1.2, 1.5);
  ok('lower on a wall in front: ramping, not a step', mid > 0 && mid < 1, true);
  const across = (v, raw, n) => cellAcross({ x: v[0] * B, y: v[1] * B, z: v[2] * B },
    { x: raw[0], y: raw[1], z: raw[2] }, { x: n[0], y: n[1], z: n[2] });
  ok("the cell under a block's bottom face", JSON.stringify(across([0.5, 0.5, 1.5], [1, -1, 1], [0, -1, 0])),
     JSON.stringify({ x: 0, y: 0, z: B }));
  ok('... and behind its back face', JSON.stringify(across([-0.5, 1.5, 0.5], [-1, 1, -1], [0, 0, -1])),
     JSON.stringify({ x: 0, y: B, z: 0 }));
  ok("the circle's gradient: clear, solid past the edge, ramping between", [cellRadial(10, 100, 30), cellRadial(120, 100, 30),
     cellRadial(85, 100, 30)].join(), '1,0,0.5');
}

// The explore camera over feet at the origin, looking along -z; +z toward it.
const { pitch, distance } = CAMERA_CONFIGS.explore;
const eye = { x: 0, y: distance * Math.sin(pitch), z: distance * Math.cos(pitch) };
const right = { x: 1, z: 0 };
const feet = { x: 0, y: -0.5 * B, z: 0 };
const put = (x, y, z, m = 'stone') => World.set(`${x},${y},${z}`, newBlock(m));
const hidden = () => playerHidden(eye, feet, right);

section('is the player hidden');
World.clear();
ok('in the open', hidden(), false);
for (let y = 0; y < 3; y++) { put(1, y, 0); put(0, y, -1); }
ok('a tall wall beside and one behind hide nothing', hidden(), false);
put(0, 0, 1);
ok('a low block hiding only the feet is not enough', hidden(), false);
const wall = () => { for (let x = -1; x <= 1; x++) for (let y = 0; y < 6; y++) put(x, y, 1); };
wall();
ok('a wall hiding all of them is', hidden(), true);
World.clear();
for (let y = 0; y < 4; y++) put(0, y, 2, 'glass');
ok('glass hides nothing', hidesAt(0, 1 * B, 2 * B), false);
ok('... so a glass wall in front is seen through', hidden(), false);
World.clear();

section('the cut set, gathered along the sight line');
{
  wall();
  const fw = { x: -eye.x, y: feet.y - eye.y, z: -eye.z };
  const l = Math.hypot(fw.x, fw.y, fw.z);
  const fr = occlusionFrame(eye, { x: fw.x / l, y: fw.y / l, z: fw.z / l }, feet);
  const o = maskOrigin(playerCell(feet));
  const t = cutTargets(fr, DEFAULT_OCCLUSION, o);
  ok("the wall's blocks up the sight line", t.get(maskIndex(o, 0, 3, 1)) > 0, true);
  ok('not the air, not what is beside', [t.has(maskIndex(o, 0, 3, 0)), t.has(maskIndex(o, 3, 3, 1))].join(), 'false,false');
  World.clear();
}

section('feature');
{
  const camera = new THREE.PerspectiveCamera(45, 16 / 9);
  camera.position.set(eye.x, eye.y, eye.z);
  camera.lookAt(0, 0, 0);
  const renderer = { getDrawingBufferSize: v => v.set(1600, 900) };
  const feature = createOcclusion({ renderer, camera, feet: () => new THREE.Vector3(feet.x, feet.y, feet.z) });
  const cutMax = () => { let m = 0; for (let k = 1; k < cellMask.data.length; k += 2) m = Math.max(m, cellMask.data[k]); return m; };
  feature.update(1000);
  ok('in the open: nothing cut', cutMax(), 0);
  wall();
  feature.update(1100);
  feature.update(1200);
  ok('hidden: nothing cut for CUT_DELAY', cutMax(), 0);
  feature.update(1300);
  const first = cutMax();
  feature.update(1400);
  feature.update(1500);
  ok('... then each block eases in', first > 0 && first < 255 && cutMax() === 255, true);
  near('centred on the player, on screen', occlusion.centrePx.value.x, 800, 1e-3);
  ok('bxb.occlusion sets the circle', feature.commands.occlusion.run(1.5, 0.5).startsWith('occlusion cut-out on: radius 1.5 m'), true);
  feature.settings.settings[0].set(false);
  for (const t of [1600, 1700, 1800]) feature.update(t);
  ok('off by its setting: eased back out', cutMax(), 0);
  World.clear();
}
