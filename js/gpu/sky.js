import * as THREE from 'three/webgpu';
import { tableGroup } from './uploads.js';
import { float, vec3, uniform, max, uniformArray, int } from 'three/tsl';

// --- The sky, as L2 SH (sky.js) ---
//
// Nine RGB coefficients, projected on the CPU each frame the sky changes, in
// one uniform array. Everything that wants the sky evaluates this: the ambient
// at the bent normal (irradiance), the background and reflection misses at a
// direction (radiance). gain scales irradiance so the noon sky lights an
// up-facing surface at exactly the ambient uniform - sky.js skyAmbientGain.
// The sun's light colour and strength (sky.js sunLight), linear. Module-level,
// so the shading pass and the LPV injection read the one value.
export const sunColourUniform = uniform(new THREE.Color(1, 1, 1));

export function createSkyBindings() {
  return {
    sh: uniformArray(new Array(9).fill(0).map(() => new THREE.Vector4()), 'vec4').setName('bxbSkySH').setGroup(tableGroup),
    gain: uniform(float(1))
  };
}

export function writeSkyBindings(b, sh, gain) {
  for (let i = 0; i < 9; i++) b.sh.array[i].set(sh[i][0], sh[i][1], sh[i][2], 0);
  b.gain.value = gain;
  return b;
}

// The mirror of evalSH in sky.js: same basis, same order, same band factors.
export function skyShTSL(b, d, irradiance = false) {
  const x = d.x, y = d.y, z = d.z;
  const Y = [
    float(0.282095),
    y.mul(0.488603), z.mul(0.488603), x.mul(0.488603),
    x.mul(y).mul(1.092548), y.mul(z).mul(1.092548), z.mul(z).mul(3).sub(1).mul(0.315392),
    x.mul(z).mul(1.092548), x.mul(x).sub(y.mul(y)).mul(0.546274)
  ];
  const band = [1, 2 / 3, 2 / 3, 2 / 3, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4];
  let out = null;
  for (let i = 0; i < 9; i++) {
    const k = irradiance ? band[i] : 1;
    const term = b.sh.element(int(i)).xyz.mul(Y[i].mul(float(k)));
    out = out ? out.add(term) : term;
  }
  return max(out, vec3(0, 0, 0));
}
