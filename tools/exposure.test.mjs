import { file, section, ok, near } from './lib/harness.mjs';
import { meteredLog2, targetEV, adaptEV, DEFAULT_EXPOSURE, EXPOSURE_KEY, METER_LOG_MIN,
         METER_LOG_SCALE, METER_LOG_MAX_Q } from '../js/core/exposure.js';

file('exposure.test.mjs - auto-exposure meters and adapts');

// What the GPU meter adds for one sample of luminance L (gpu/exposure.js).
const q = L => Math.min(METER_LOG_MAX_Q, Math.max(0, Math.floor(
  (Math.log2(Math.max(L, 2 ** METER_LOG_MIN)) - METER_LOG_MIN) * METER_LOG_SCALE)));

ok('the default bias is half a stop darker', DEFAULT_EXPOSURE.bias, -0.5);

section('metering');
ok('no terrain sampled: nothing', meteredLog2(0, 0), null);
near('one sample of middle grey', meteredLog2(q(0.18), 1), Math.log2(0.18), 1 / METER_LOG_SCALE);
near('the geometric mean, not the arithmetic', meteredLog2(q(1) + q(0.01), 2), Math.log2(0.1), 2 / METER_LOG_SCALE);
near('black counts as the floor', meteredLog2(q(0), 1), METER_LOG_MIN, 1e-9);
ok('the largest sum fits a u32', 4096 * METER_LOG_MAX_Q < 2 ** 32, true);

section('target');
const P = { ...DEFAULT_EXPOSURE, bias: 0 };   // the maths without the default bias
near('a middle-grey frame needs no change', targetEV(Math.log2(EXPOSURE_KEY), P), 0, 1e-9);
near('a frame 4x too dark: +2 EV', targetEV(Math.log2(EXPOSURE_KEY / 4), P), 2, 1e-9);
ok('a pitch-dark frame stops at the brightest', targetEV(-14, P), P.maxEV);
ok('a blinding one at the darkest', targetEV(6, P), P.minEV);
near('bias adds', targetEV(Math.log2(EXPOSURE_KEY), { ...P, bias: 0.5 }), 0.5, 1e-9);

section('adaptation');
near('one time constant: 63% of the way (brightening)', adaptEV(0, 1, P.brighten, P), 1 - Math.exp(-1), 1e-9);
near('darkening is faster', adaptEV(1, 0, P.darken, P), Math.exp(-1), 1e-9);
ok('no time, no change', adaptEV(0.3, 2, 0, P), 0.3);
ok('a zero time constant jumps', adaptEV(0, 2, 0.016, { ...P, brighten: 0 }), 2);
// The same second at 60 and at 240 fps lands in the same place.
let a = 0, b = 0;
for (let i = 0; i < 60; i++) a = adaptEV(a, 3, 1 / 60, P);
for (let i = 0; i < 240; i++) b = adaptEV(b, 3, 1 / 240, P);
near('frame-rate independent', a, b, 1e-9);
