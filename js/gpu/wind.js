// --- Wind: one direction and strength for everything that sways ---
//
// A global uniform for now (no weather yet): the leaves' sway wave travels
// along `dir` (x, z; unit length), as far as `strength` takes it. Grass and
// smoke can read the same when they sway.
import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';

export const wind = {
  dir: uniform(new THREE.Vector2(1, 0)),
  strength: uniform(1)
};

// Points the wind `degrees` round from +x toward +z.
export function setWindHeading(degrees) {
  const a = degrees * Math.PI / 180;
  wind.dir.value.set(Math.cos(a), Math.sin(a));
}
