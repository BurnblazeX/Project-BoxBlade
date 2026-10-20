import * as THREE from 'three/webgpu';
import { file, section, ok, near } from './lib/harness.mjs';
import { maskOrigin, maskIndex, playerCell, ROOF_DELAY, CELL_EASE, DEFAULT_INTERIOR } from '../js/core/interior.js';
import { World, newBlock, BLOCK_METRES as B } from '../js/core/world.js';
import { occlusion, cellMask } from '../js/gpu/occlusion.js';
import { createOcclusion } from '../js/render/features/occlusion.js';

file('occlusion.test.mjs - the interiors feature: lifting a roof, darkening outside');

// A 7x7 room (walls at +-3) on y 1..3 under a slab at y 4, the player at its
// centre in cell y 1, the camera at +z looking down along -z.
const put = (x, y, z, m = 'stone') => World.set(`${x},${y},${z}`, newBlock(m));
function house() {
  World.clear();
  for (let x = -4; x <= 4; x++) for (let z = -4; z <= 4; z++) {
    put(x, 0, z, 'grass');
    put(x, 4, z);
    if ((Math.abs(x) === 3 || Math.abs(z) === 3) && Math.abs(x) <= 3 && Math.abs(z) <= 3)
      for (let y = 1; y <= 3; y++) put(x, y, z);
  }
}

section('feature');
{
  const feet = new THREE.Vector3(0, 0.5 * B, 0);
  const camera = new THREE.PerspectiveCamera(45, 16 / 9);
  camera.position.set(0, 10 * B, 10 * B);
  camera.lookAt(0, 0, 0);
  const feature = createOcclusion({ camera, feet: () => feet.clone() });
  const o = maskOrigin(playerCell(feet));
  const at = (x, y, z) => cellMask.data[maskIndex(o, x, y, z)];
  let t = 1000;
  const run = seconds => { for (let s = 0; s < seconds; s += 0.05) feature.update(t += 50); };

  World.clear();
  put(0, 0, 0, 'grass');
  run(1);
  ok('in the open: nothing lifted', at(0, 4, 0), 0);
  house();
  run(ROOF_DELAY / 2);
  ok('under a roof: nothing for ROOF_DELAY', at(0, 4, 0), 0);
  run(ROOF_DELAY + CELL_EASE + 0.1);
  ok('... then the roof lifts', at(0, 4, 0), 255);
  ok('the wall facing the camera with it', at(0, 2, 3), 255);
  ok('not the wall behind, nor the stub', at(0, 2, -3) + at(0, 1, 3), 0);
  near('outside darkens', occlusion.outsideDark.value, DEFAULT_INTERIOR.dark, 1e-6);
  ok('picking passes through the lifted roof', feature.hiddenAt('0,4,0') && !feature.hiddenAt('0,0,0'), true);
  feature.commands.interiors.run();
  run(CELL_EASE + 0.1);
  ok('off: grown back, the outside lit', [at(0, 4, 0), occlusion.outsideDark.value].join(), '0,0');
  feature.commands.interiors.run();
  World.clear();
}
