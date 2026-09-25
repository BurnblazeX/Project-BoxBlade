import { file, section, ok, near, truthy } from './lib/harness.mjs';
import { texelPixels, texelCacheSpacingFor, TEXEL_METRES, TEXEL_SPACING_MIN,
         TEXEL_SPACING_MAX } from '../js/texelcache.js';

file('texelcache.test.mjs - the texel cache follows the texel size on screen');

section('texel size on screen');
// The explore camera: fov 45 at 22 m. 1440 rows (1080 at DPR 1.33) puts a
// texel at ~9.9 px, which is what the screen shows.
near('1440 rows, explore camera', texelPixels(1440, 45, 22), 9.88, 0.02);
near('half the rows, half the size', texelPixels(720, 45, 22), texelPixels(1440, 45, 22) / 2, 1e-9);
near('twice as far, half the size', texelPixels(1440, 45, 44), texelPixels(1440, 45, 22) / 2, 1e-9);
near('a texel is an eighth of a metre', TEXEL_METRES, 0.125, 1e-9);

section('spacing');
ok('~9.9 px texels (measured best: 5)', texelCacheSpacingFor(texelPixels(1440, 45, 22)), 5);
ok('1080 rows: 4', texelCacheSpacingFor(texelPixels(1080, 45, 22)), 4);
ok('the battle camera matches explore at the same resolution',
   texelCacheSpacingFor(texelPixels(1440, 8, 125)), 5);
ok('tiny texels floor at the minimum', texelCacheSpacingFor(1), TEXEL_SPACING_MIN);
ok('huge texels cap at the maximum', texelCacheSpacingFor(100), TEXEL_SPACING_MAX);
ok('held near its rounding edge', texelCacheSpacingFor(2 * 5.6, 5), 5);
ok('moves once clearly past it', texelCacheSpacingFor(2 * 5.7, 5), 6);
ok('and back down the same way', texelCacheSpacingFor(2 * 4.3, 5), 4);
truthy('always a whole number of pixels', [3.1, 7.7, 9.9, 13.2].every(t =>
  Number.isInteger(texelCacheSpacingFor(t))));
