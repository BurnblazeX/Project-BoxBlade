// --- What is under the cursor ---
import * as THREE from 'three/webgpu';
import { worldInstancedMesh, keyForInstance } from './render.js';
import { BLOCK_METRES } from '../core/world.js';
import { buildPickGrid, pickTop } from '../core/pickgrid.js';

export function createPicker(camera) {
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const aim = (clientX, clientY) => {
    mouse.x = (clientX / window.innerWidth) * 2 - 1;
    mouse.y = -(clientY / window.innerHeight) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
  };
  // The terrain's boxes binned by cell (core/pickgrid.js), rebuilt when its
  // instance matrices change - an edit, or battle hiding what is off the arena.
  // The mesh sits at the origin, so its instance matrices are world space.
  let grid = null, gridFor = null;
  const terrainGrid = () => {
    const m = worldInstancedMesh, v = m.instanceMatrix.version;
    if (!grid || gridFor.mesh !== m || gridFor.version !== v || gridFor.count !== m.count) {
      grid = buildPickGrid(m.instanceMatrix.array, m.count, BLOCK_METRES);
      gridFor = { mesh: m, version: v, count: m.count };
    }
    return grid;
  };

  return {
    // The grid cell whose top face is under the cursor, as { gx, gy, gz }.
    // This raycasts the terrain mesh rather than a single horizontal plane - with
    // real elevation there is no one ground height, and only a block's TOP face is
    // somewhere an entity can stand, so side-face hits on walls are skipped.
    // Instances carry no rotation, so the box geometry's local +Y normal is world up.
    // The nearest top face along the ray wins; side faces are passed through.
    // A grid walk, not three's raycast, which tests every instance.
    tileAt(clientX, clientY) {
      if (!worldInstancedMesh) return null;
      aim(clientX, clientY);
      const { origin: o, direction: d } = raycaster.ray;
      const id = pickTop(terrainGrid(), [o.x, o.y, o.z], [d.x, d.y, d.z]);
      const key = id >= 0 ? keyForInstance(id) : null;
      if (!key) return null;
      const [gx, gy, gz] = key.split(',').map(Number);
      return { gx, gy, gz };
    },
    // The nearest of objects (and their children) under the cursor, or null.
    objectAt(clientX, clientY, objects) {
      aim(clientX, clientY);
      const hits = raycaster.intersectObjects(objects);
      return hits.length > 0 ? hits[0].object : null;
    }
  };
}
