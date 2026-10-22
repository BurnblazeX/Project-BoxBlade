// --- The camera rig ---
//
// One PerspectiveCamera on an orbit around a pivot, eased toward the mode's
// framing: close and steep in explore, following Bob; far and narrow in
// battle - near-orthographic - over the arena's centre. Q/E turn it a quarter
// at a time. The pose it has reached is read by everything that faces the
// camera: rig.heading, rig.pitch, rig.distance.
import * as THREE from 'three/webgpu';

// We use one PerspectiveCamera to allow buttery smooth lerping (no janky swaps).
// We mimic Orthographic for battle mode by lowering the FOV and pulling back distance.
export const CAMERA_CONFIGS = {
  explore: {
    fov: 45,
    distance: 22,
    pitch: Math.PI / 3.43,
    headingOffset: 0
  },
  battle: {
    fov: 8,
    distance: 125,
    pitch: Math.atan(1 / Math.sqrt(2)),
    headingOffset: Math.PI / 4
  }
};
const CAMERA_LERP_SPEED = 0.1;
const FREE_CAM_SPEED = 12;   // m/s

export function createCameraRig(scene) {
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
  const pivot = new THREE.Object3D();
  scene.add(pivot);

  const rig = {
    camera, pivot,
    // Battle: where the pivot settles (the encounter sets it).
    arenaCenter: new THREE.Vector3(),
    // Quarter turns, from Q/E.
    rotationStep: 0,
    // The pose so far, easing toward the mode's.
    fov: CAMERA_CONFIGS.explore.fov,
    pitch: CAMERA_CONFIGS.explore.pitch,
    heading: CAMERA_CONFIGS.explore.headingOffset,
    distance: CAMERA_CONFIGS.explore.distance,
    // Free cam (Alt+C, debug): detaches the camera from Bob - WASD flies the
    // pivot instead of walking him, Q/E still turn it. Only the CAMERA moves -
    // the cascades, the field overlay and everything else that follows the
    // player keep following player.gridPos, so flying away shows the field's
    // footprint staying put around Bob.
    freeCam: false,

    // The heading the mode's framing settles at: the basis Bob walks in.
    settledHeading(mode) {
      return (rig.rotationStep * Math.PI / 2) + CAMERA_CONFIGS[mode].headingOffset;
    },

    // One frame: the pivot toward focus (explore) or the arena (battle), or
    // flown by keys under free cam; the pose eased toward the mode's; the
    // camera placed on its orbit.
    update(dt, mode, focus, keys) {
      // The lerp factors were tuned as "per frame at 60 Hz". Converted to the
      // equivalent fraction for this frame's dt, so a 240 Hz screen eases at the
      // same speed as a 60 Hz one instead of four times faster.
      const ease = 1 - Math.pow(1 - CAMERA_LERP_SPEED, dt * 60);
      if (rig.freeCam) {
        fly(dt, mode, keys);
      } else if (mode === 'explore') {
        pivot.position.lerp(focus, ease);
      } else {
        pivot.position.lerp(rig.arenaCenter, ease);
      }

      const config = CAMERA_CONFIGS[mode];
      const targetHeading = rig.settledHeading(mode);

      // Lerp all camera properties smoothly
      rig.fov += (config.fov - rig.fov) * ease;
      rig.pitch += (config.pitch - rig.pitch) * ease;
      rig.heading += (targetHeading - rig.heading) * ease;
      rig.distance += (config.distance - rig.distance) * ease;

      camera.fov = rig.fov;
      camera.updateProjectionMatrix();

      const xzLen = rig.distance * Math.cos(rig.pitch);
      camera.position.x = pivot.position.x + xzLen * Math.sin(rig.heading);
      camera.position.y = pivot.position.y + rig.distance * Math.sin(rig.pitch);
      camera.position.z = pivot.position.z + xzLen * Math.cos(rig.heading);

      camera.lookAt(pivot.position);
    }
  };

  // Free cam: WASD moves the pivot, screen-relative - the same basis Bob walks in.
  function fly(dt, mode, keys) {
    let rx = 0, rz = 0;
    if (keys.w) rz -= 1;
    if (keys.s) rz += 1;
    if (keys.a) rx -= 1;
    if (keys.d) rx += 1;
    if (!rx && !rz) return;
    const len = Math.hypot(rx, rz);
    const h = rig.settledHeading(mode);
    const c = Math.cos(h), s = Math.sin(h);
    const step = FREE_CAM_SPEED * dt / len;
    pivot.position.x += (rx * c + rz * s) * step;
    pivot.position.z += (-rx * s + rz * c) * step;
  }

  return rig;
}
