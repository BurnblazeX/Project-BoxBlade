// --- Exposure: what the auto-exposure meters and how it adapts ---
//
// The meter (gpu/exposure.js) averages log2 luminance over a fixed grid of the
// terrain's low-res samples - the geometric mean, so one torch flame or one
// patch of sky cannot swing the whole frame. A GPU has no float atomics, so
// each sample adds its log2 luminance as fixed point: (log2 L - LOG_MIN) *
// LOG_SCALE, clamped to [0, LOG_MAX_Q]. Headless, so the maths tests in node.

export const METER_SAMPLES = 4096;           // a 64 x 64 grid over the low-res target
export const METER_GRID = 64;
export const METER_LOG_MIN = -16;            // luminance below 2^-16 counts as 2^-16
export const METER_LOG_SCALE = 256;
export const METER_LOG_MAX_Q = 32 * METER_LOG_SCALE - 1;   // up to 2^16

// The middle grey the frame's mean is brought to, before the bias.
export const EXPOSURE_KEY = 0.18;

export const DEFAULT_EXPOSURE = {
  bias: -0.5,       // EV added to the metered exposure (brighter +, darker -)
  minEV: -2,        // how far a bright scene is darkened
  maxEV: 5,         // how far a dark one is brightened: night stays darker than day
  brighten: 1.5,    // seconds to adapt toward brighter (eyes are slow into the dark)
  darken: 0.5       // seconds to adapt toward darker
};

// The mean log2 luminance from the meter's two words (fixed-point sum, sample
// count), or null when no sample was terrain.
export function meteredLog2(sum, count) {
  if (!count) return null;
  return sum / count / METER_LOG_SCALE + METER_LOG_MIN;
}

// The exposure, in EV (the output is scaled by 2^EV), that brings a frame of
// mean log2 luminance meanLog2 to middle grey, plus bias, within [minEV, maxEV].
export function targetEV(meanLog2, { bias, minEV, maxEV }) {
  const ev = Math.log2(EXPOSURE_KEY) - meanLog2 + bias;
  return Math.max(minEV, Math.min(maxEV, ev));
}

// One step of adaptation over dt seconds: an exponential ease toward target,
// slower when brightening (target above current) than when darkening -
// independent of the frame rate.
export function adaptEV(current, target, dt, { brighten, darken }) {
  const tau = target > current ? brighten : darken;
  if (!(tau > 0)) return target;
  return current + (target - current) * (1 - Math.exp(-Math.max(0, dt) / tau));
}
