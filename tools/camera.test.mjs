import * as THREE from 'three/webgpu';
import { file, section, ok, near } from './lib/harness.mjs';

// Every test file runs in one process: put back the window at the end.
const savedWindow = globalThis.window;
globalThis.window = { innerWidth: 1600, innerHeight: 900 };
const { createCameraRig, CAMERA_CONFIGS } = await import('../js/render/camera.js');

file('camera.test.mjs - the rig eases to each mode and follows its focus');

const run = (rig, seconds, mode, focus, keys = {}) => {
  for (let t = 0; t < seconds; t += 1 / 60) rig.update(1 / 60, mode, focus, keys);
};

section('explore: on Bob');
{
  const rig = createCameraRig(new THREE.Scene());
  const bob = new THREE.Vector3(6, 1, 9);
  run(rig, 3, 'explore', bob);
  near('the pivot reaches Bob', rig.pivot.position.distanceTo(bob), 0, 1e-3);
  near('at the explore distance', rig.camera.position.distanceTo(rig.pivot.position), CAMERA_CONFIGS.explore.distance, 1e-3);
  near('and pitch', rig.pitch, CAMERA_CONFIGS.explore.pitch, 1e-6);
}

section('battle: over the arena, narrow and far');
{
  const rig = createCameraRig(new THREE.Scene());
  rig.arenaCenter.set(9, 0, 9);
  run(rig, 4, 'battle', new THREE.Vector3(50, 0, 50));
  near('the pivot settles on the arena, not Bob', rig.pivot.position.distanceTo(rig.arenaCenter), 0, 1e-3);
  near('fov eased to battle', rig.camera.fov, CAMERA_CONFIGS.battle.fov, 1e-3);
  near('heading to its offset', rig.heading, CAMERA_CONFIGS.battle.headingOffset, 1e-6);
}

section('Q/E and free cam');
{
  const rig = createCameraRig(new THREE.Scene());
  rig.rotationStep = 1;
  near('a quarter turn', rig.settledHeading('explore'), Math.PI / 2, 1e-9);
  rig.freeCam = true;
  run(rig, 1, 'explore', new THREE.Vector3(30, 0, 30), { w: true });
  ok('free cam ignores Bob', rig.pivot.position.distanceTo(new THREE.Vector3(30, 0, 30)) > 20, true);
  // W at heading pi/2: screen-forward is -x.
  ok('W flies forward on screen', rig.pivot.position.x < -11 && Math.abs(rig.pivot.position.z) < 1e-6, true);
}

globalThis.window = savedWindow;
