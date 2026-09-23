import { file, section, ok, near, truthy, falsy } from './lib/harness.mjs';
import { readFileSync } from 'node:fs';
import { MATERIALS, materialLayer, decodeSpecular, METAL_F0, DEFAULT_SPECULAR,
         BLOCK_TEXELS } from '../js/materials.js';

file('materials.test.mjs - block materials and LabPBR specular');

section('material table');
ok('grass is layer 0, the fallback', materialLayer('grass'), 0);
ok('no id falls back to layer 0', materialLayer(null), 0);
ok('an unknown id falls back to layer 0', materialLayer('nope'), 0);
ok('marble has its own layer', materialLayer('marble'), 1);
ok('iron plate has its own layer', materialLayer('ironPlate'), 2);
ok('ids are unique', new Set(MATERIALS.map(m => m.id)).size, MATERIALS.length);

section('LabPBR decode - dielectrics');
{
  const d = decodeSpecular(DEFAULT_SPECULAR);
  ok('default is fully rough', d.roughness, 1);
  near('default F0 is ~0.04', d.f0[0], 10 / 255);
  falsy('default is not metal', d.metal);
  ok('alpha 255 is no emission', d.emission, 0);
  const m = decodeSpecular([245, 10, 0, 255]);
  near('smoothness 245 -> roughness (1 - s)^2', m.roughness, (1 - 245 / 255) ** 2);
  near('porosity range is 0-64', decodeSpecular([0, 10, 64, 255]).porosity, 1);
  near('SSS starts at 65', decodeSpecular([0, 10, 65, 255]).sss, 0);
  near('SSS tops out at 255', decodeSpecular([0, 10, 255, 255]).sss, 1);
  ok('SSS range is not porosity', decodeSpecular([0, 10, 200, 255]).porosity, 0);
  near('alpha 254 is full emission', decodeSpecular([0, 10, 0, 254]).emission, 1);
}

section('LabPBR decode - metals');
{
  const iron = decodeSpecular([240, 230, 0, 255]);
  truthy('230 is a metal', iron.metal);
  ok('230 is iron', iron.metalName, 'iron');
  // Iron's F0 is a mid grey, a touch cooler in blue: ~0.53-0.56 in each channel.
  truthy('iron F0 is a bright grey', iron.f0.every(c => c > 0.45 && c < 0.65));
  const gold = decodeSpecular([240, 231, 0, 255]);
  truthy('gold F0 is warm (red > blue)', gold.f0[0] > gold.f0[2] + 0.3);
  const other = decodeSpecular([240, 250, 0, 255], [0.2, 0.4, 0.6]);
  ok('238-255 takes F0 from the albedo', other.f0.join(), '0.2,0.4,0.6');
  ok('every predefined metal has an F0', METAL_F0.length, 8);
}

// The authored files: every block texture is 12x12, and the two reflective
// materials carry a _s that decodes to what they are meant to be.
function png(name) {
  const b = readFileSync(new URL(`../assets/textures/${name}.png`, import.meta.url));
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
section('authored textures');
for (const m of MATERIALS) {
  const { w, h } = png(m.texture);
  ok(`${m.texture} is ${BLOCK_TEXELS}x${BLOCK_TEXELS}`, `${w}x${h}`, `${BLOCK_TEXELS}x${BLOCK_TEXELS}`);
}
for (const t of ['terrain_marble', 'terrain_ironPlate']) {
  for (const suffix of ['_n', '_s']) {
    const { w, h } = png(t + suffix);
    ok(`${t}${suffix} matches its albedo`, `${w}x${h}`, `${BLOCK_TEXELS}x${BLOCK_TEXELS}`);
  }
}
