// --- What is under the cursor ---
import * as THREE from 'three/webgpu';
import { worldInstancedMesh, keyForInstance } from './render.js';

export function createPicker(camera) {
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const aim = (clientX, clientY) => {
    mouse.x = (clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
  };

  return {
    // The grid cell whose top face is under the cursor, as { gx, gy, gz }.
    // This raycasts the terrain mesh rather than a single horizontal plane - with
    // real elevation there is no one ground height, and only a block's TOP face is
    // somewhere an entity can stand, so side-face hits on walls are skipped.
    // Instances carry no rotation, so the box geometry's local +Y normal is world up.
    // Hits arrive sorted by distance, making the first top face the nearest surface.
    tileAt(clientX, clientY) {
      if (!worldInstancedMesh) return null;
      aim(clientX, clientY);
      for (const hit of raycaster.intersectObject(worldInstancedMesh, false)) {
        if (!hit.face || hit.face.normal.y < 0.5) continue;
        const key = keyForInstance(hit.instanceId);
        if (!key) continue;
        const [gx, gy, gz] = key.split(',').map(Number);
        return { gx, gy, gz };
      }
      return null;
    },
    // The nearest of objects (and their children) under the cursor, or null.
    objectAt(clientX, clientY, objects) {
      aim(clientX, clientY);
      const hits = raycaster.intersectObjects(objects);
      return hits.length > 0 ? hits[0].object : null;
    }
  };
}
