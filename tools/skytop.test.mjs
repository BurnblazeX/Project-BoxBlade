import { file, section, ok, near } from './lib/harness.mjs';
import { buildSkyTops, encodeSkyTop, decodeSkyTop, roofTop, SKY_TOP_STEPS } from '../js/core/skytop.js';
import { BLOCK_METRES, Y_MIN, Y_MAX, newBlock } from '../js/core/world.js';

file('skytop.test.mjs - each column\'s roof, for rays that missed everything');

section('encoding');
const step = BLOCK_METRES / SKY_TOP_STEPS;
for (const y of [Y_MIN, 0, 3, Y_MAX]) {
  const top = (y + 0.5) * BLOCK_METRES;
  near(`a full block's top at y ${y} comes back`, decodeSkyTop(encodeSkyTop(top)), top, 1e-9);
}
const odd = 1.234;
const back = decodeSkyTop(encodeSkyTop(odd));
ok('an odd height rounds up, never down', back >= odd, true);
ok('... by less than a step', back - odd < step, true);
ok('no roof decodes as no height at all', decodeSkyTop(0), -Infinity);
ok('the highest top fits a byte', encodeSkyTop((Y_MAX + 0.5) * BLOCK_METRES) <= 255, true);

section('what is a roof');
ok('a full block reaches its cell top', roofTop(newBlock('stone')), 1);
ok('a low half reaches half way', roofTop(newBlock('stone', 'halfBottom')), 0.5);
ok('glass is no roof', roofTop(newBlock('glass')), null);
ok('a glass slope is no roof', roofTop(newBlock('glass', 'slope', 'E')), null);
ok('a dual block takes its non-glass half', roofTop(newBlock('glass', 'halfTop', null, 'stone')), 0.5);

section('the map');
const world = new Map([
  ['0,0,0', newBlock('grass')],     // ground
  ['0,3,0', newBlock('stone')],     // a roof over it
  ['1,0,0', newBlock('grass')],     // open ground beside it
  ['2,0,0', newBlock('grass')],
  ['2,4,0', newBlock('glass')],     // a skylight: no roof
  ['9,0,9', newBlock('stone')]      // outside the map's box
]);
const tops = buildSkyTops(world, [0, 0], [4, 2]);
const at = (x, z) => decodeSkyTop(tops[z * 4 + x]);
near('the roofed column: its highest block', at(0, 0), 3.5 * BLOCK_METRES, 1e-9);
near('the open column: its ground', at(1, 0), 0.5 * BLOCK_METRES, 1e-9);
near('under a skylight: the ground', at(2, 0), 0.5 * BLOCK_METRES, 1e-9);
ok('an empty column has none', tops[3], 0);
ok('blocks outside the box are left out', tops.length, 8);
