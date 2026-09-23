import { file, section, ok, near, truthy } from './lib/harness.mjs';
import { spriteNormals, terrainNormals, encodeNormal, decodeNormal,
         SPRITE_BEVEL_PX } from '../js/normals.js';
import { createCard, CARD_PAD } from '../js/cards.js';

file('normals.test.mjs - generated tangent-space normal maps');

section('encoding');
{
  const buf = new Uint8Array(4);
  encodeNormal(buf, 0, 0, 0, 1);
  const [x, y, z] = decodeNormal(buf, 0);
  near('facing the viewer encodes to (0, 0, 1)', x, 0, 0.01);
  near('  y', y, 0, 0.01);
  near('  z', z, 1, 0.01);
}

section('sprite bevel');
// A 12x12 solid square: flat in the middle, tilted outward at every edge.
const W = 12, H = 12;
const card = createCard({ mask: new Uint8Array(W * H).fill(1), w: W, h: H,
                          widthMetres: 1, heightMetres: 1 });
const n = spriteNormals(card.dt, card.w, W, H, CARD_PAD);
// Output rows are bottom-up: row r of the image is output row H-1-r.
const at = (x, imgRow) => decodeNormal(n, ((H - 1 - imgRow) * W + x) * 4);

const mid = at(6, 6);
near('the middle faces the viewer', mid[2], 1, 0.02);
truthy('the left edge leans left', at(0, 6)[0] < -0.3);
truthy('the right edge leans right', at(W - 1, 6)[0] > 0.3);
truthy('the top edge leans UP (y is up, image rows run down)', at(6, 0)[1] > 0.3);
truthy('the bottom edge leans down', at(6, H - 1)[1] < -0.3);
// The raw byte, not the decode: DirectX puts an up-leaning normal's green
// BELOW the midpoint. A decode that flipped along with the encode would pass
// the checks above and still hand the shader an OpenGL map.
truthy('DirectX: the top edge stores green below 128',
       n[((H - 1 - 0) * W + 6) * 4 + 1] < 128);
ok('LabPBR: blue is texture AO, 255 when none is baked', n[(6 * W + 6) * 4 + 2], 255);
ok('LabPBR: alpha is height, 255 at the surface', n[(6 * W + 6) * 4 + 3], 255);
near('flat again past the bevel', at(SPRITE_BEVEL_PX + 2, 6)[2], 1, 0.02);

section('terrain from luminance');
{
  // A ramp brightening to the right: the surface rises toward +x, so its
  // normal leans toward -x. Wrapped, so test away from the seam.
  const w = 8, h = 4, rgba = new Uint8Array(w * h * 4);
  for (let row = 0; row < h; row++) for (let x = 0; x < w; x++) {
    const v = x * 30;
    rgba.set([v, v, v, 255], (row * w + x) * 4);
  }
  const t = terrainNormals(rgba, w, h);
  const [x, y] = decodeNormal(t, (1 * w + 3) * 4);
  truthy('a rising ramp tilts the normal against the slope', x < -0.05);
  near('and not sideways', y, 0, 0.01);
  const flat = terrainNormals(new Uint8Array(w * h * 4).fill(128), w, h);
  ok('a flat texture is flat everywhere', decodeNormal(flat, 20).slice(0, 3).map(v => Math.round(v)).join(), '0,0,1');
}
