// --- The sky: an authored gradient, handed to everything as L2 SH ---
//
// Not a physical model. Four terms, keyframed on a clock, so the palette is an
// art decision rather than the output of turbidity tuning:
//
//   sky(dir) = mix(horizon, zenith, saturate(dir.y))
//            + sunGlow * saturate(dir . sunDir)^k        (sun-side brightening)
//
// Below the horizon it holds the horizon colour.
//
// ONE REPRESENTATION. The gradient is projected to 9 RGB spherical harmonics
// (L2) once per frame, here on the CPU, and uploaded as one uniform. Every
// consumer evaluates that and nothing else:
//
//   ambient light      irradiance SH at the bent normal (gpu.js)
//   background pixels  radiance SH at the view direction
//   reflection misses  radiance SH at the ray direction
//
// L2 holds ~99% of a smooth sky's irradiance (Ramamoorthi & Hanrahan), which
// is what lighting needs. It is far too soft to LOOK at - no sun disc, a soft
// horizon - which is the fidelity this game has chosen not to need. A better
// model (Hosek-Wilkie) can replace skyRadiance() later; nothing past the
// projection knows which model filled it.
//
// Colours below are authored in sRGB and used linear.

const srgb = hex => {
  const c = [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255].map(v => v / 255);
  return c.map(v => (v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
};

// Hours 0-24. Night holds either side of midnight; dawn and dusk are short.
//   zenith, horizon  the gradient
//   sun              the sky's glow on the sun's side
//   light            the SUN'S OWN light: colour times strength, linear. Noon is
//                    exactly white 1, so the scene at noon is what it always was
const light = (hex, k) => srgb(hex).map(v => v * k);
export const SKY_KEYS = [
  { hour: 0,    name: 'night', zenith: srgb(0x070b1c), horizon: srgb(0x161c36), sun: srgb(0x000000),
    light: light(0x000000, 0) },
  { hour: 5,    name: 'night', zenith: srgb(0x070b1c), horizon: srgb(0x161c36), sun: srgb(0x000000),
    light: light(0x000000, 0) },
  { hour: 6.5,  name: 'dawn',  zenith: srgb(0x34467e), horizon: srgb(0xe8a47a), sun: srgb(0xffa060),
    light: light(0xffb27a, 0.7) },
  { hour: 12,   name: 'noon',  zenith: srgb(0x3b7dd6), horizon: srgb(0xb6d3ee), sun: srgb(0x5a5040),
    light: light(0xffffff, 1) },
  { hour: 17.5, name: 'dusk',  zenith: srgb(0x2c3470), horizon: srgb(0xe8784e), sun: srgb(0xff7038),
    light: light(0xff9a60, 0.7) },
  { hour: 19,   name: 'night', zenith: srgb(0x070b1c), horizon: srgb(0x161c36), sun: srgb(0x000000),
    light: light(0x000000, 0) },
  { hour: 24,   name: 'night', zenith: srgb(0x070b1c), horizon: srgb(0x161c36), sun: srgb(0x000000),
    light: light(0x000000, 0) }
];
// The hour the game starts at.
export const DEFAULT_SKY_HOUR = 9;
// The sun-side glow's tightness: saturate(dir . sun)^k.
export const SKY_SUN_POWER = 8;

const mix3 = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t,
                           a[2] + (b[2] - a[2]) * t];

// The palette at an hour: the two keys either side, lerped.
export function skyPalette(hour) {
  const h = ((hour % 24) + 24) % 24;
  let i = 0;
  while (i < SKY_KEYS.length - 2 && SKY_KEYS[i + 1].hour <= h) i++;
  const a = SKY_KEYS[i], b = SKY_KEYS[i + 1];
  const t = b.hour > a.hour ? (h - a.hour) / (b.hour - a.hour) : 0;
  return { zenith: mix3(a.zenith, b.zenith, t), horizon: mix3(a.horizon, b.horizon, t),
           sun: mix3(a.sun, b.sun, t), light: mix3(a.light, b.light, t) };
}

// The gradient, linear RGB. dir and sunDir unit vectors, y up.
export function skyRadiance(pal, dir, sunDir, k = SKY_SUN_POWER) {
  const up = Math.min(1, Math.max(0, dir[1]));
  const c = mix3(pal.horizon, pal.zenith, up);
  const s = Math.max(0, dir[0] * sunDir[0] + dir[1] * sunDir[1] + dir[2] * sunDir[2]) ** k;
  return [c[0] + pal.sun[0] * s, c[1] + pal.sun[1] * s, c[2] + pal.sun[2] * s];
}

// --- L2 spherical harmonics ---
//
// Real SH basis, the usual order: Y00; Y1-1 Y10 Y11; Y2-2 Y2-1 Y20 Y21 Y22.
export function shBasis(d) {
  const [x, y, z] = d;
  return [
    0.282095,
    0.488603 * y, 0.488603 * z, 0.488603 * x,
    1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3 * z * z - 1),
    1.092548 * x * z, 0.546274 * (x * x - y * y)
  ];
}
// Cosine-lobe convolution per band, divided by PI - so a sky of uniform
// radiance L evaluates to L, the units the rest of the shading is in.
export const SH_IRRADIANCE_BAND = [1, 2 / 3, 2 / 3, 2 / 3, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4];

// Numerical projection over a fixed lat-long grid, weighted by solid angle.
// 32 x 64 - 2048 samples, well under a millisecond a frame. The weights are
// normalised to sum to exactly 4 PI: the midpoint rule over-counts slightly,
// and uncorrected that brightens every sky by half a percent.
export const SH_GRID_ROWS = 32, SH_GRID_COLS = 64;
const gridWeights = new Map();
function rowWeights(rows, cols) {
  const key = rows + 'x' + cols;
  if (!gridWeights.has(key)) {
    const dTheta = Math.PI / rows, dPhi = 2 * Math.PI / cols;
    const w = Array.from({ length: rows }, (_, r) => Math.sin((r + 0.5) * dTheta) * dTheta * dPhi);
    const total = w.reduce((a, b) => a + b, 0) * cols;
    gridWeights.set(key, w.map(v => v * 4 * Math.PI / total));
  }
  return gridWeights.get(key);
}
export function projectSH(radiance, rows = SH_GRID_ROWS, cols = SH_GRID_COLS) {
  const sh = Array.from({ length: 9 }, () => [0, 0, 0]);
  const dTheta = Math.PI / rows, dPhi = 2 * Math.PI / cols;
  const weights = rowWeights(rows, cols);
  for (let r = 0; r < rows; r++) {
    const theta = (r + 0.5) * dTheta;
    const st = Math.sin(theta), w = weights[r];
    for (let c = 0; c < cols; c++) {
      const phi = (c + 0.5) * dPhi;
      const d = [st * Math.cos(phi), Math.cos(theta), st * Math.sin(phi)];
      const L = radiance(d), Y = shBasis(d);
      for (let i = 0; i < 9; i++) {
        sh[i][0] += L[0] * Y[i] * w; sh[i][1] += L[1] * Y[i] * w; sh[i][2] += L[2] * Y[i] * w;
      }
    }
  }
  return sh;
}

// Evaluate: radiance (irradiance = false) or cosine-convolved irradiance / PI.
// The CPU mirror of skyShTSL in gpu.js. Clamped at zero - an L2 fit rings
// slightly negative opposite a bright lobe.
export function evalSH(sh, d, irradiance = false) {
  const Y = shBasis(d);
  const out = [0, 0, 0];
  for (let i = 0; i < 9; i++) {
    const k = irradiance ? SH_IRRADIANCE_BAND[i] : 1;
    out[0] += sh[i][0] * Y[i] * k; out[1] += sh[i][1] * Y[i] * k; out[2] += sh[i][2] * Y[i] * k;
  }
  return out.map(v => Math.max(0, v));
}

const luminance = c => 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];

// The ambient scale. The flat ambient the renderer had was one number, tuned
// by eye at a daytime sun; the sky ambient keeps that number's meaning at
// noon. So the gain divides by what an up-facing surface receives from the
// NOON sky - and a dusk or night sky, being darker, lights less.
export const SKY_NOON_HOUR = 12;
export function skyAmbientGain() {
  const pal = skyPalette(SKY_NOON_HOUR);
  const sun = [0, 1, 0];
  const sh = projectSH(d => skyRadiance(pal, d, sun));
  return 1 / Math.max(1e-4, luminance(evalSH(sh, [0, 1, 0], true)));
}

// The sun's light for a palette and a sun direction: the key's colour, faded
// out as the sun crosses the horizon - so a sun placed below it by hand
// (bxb.sun) does not light the undersides of things.
export function sunLight(pal, sunDir) {
  const t = Math.min(1, Math.max(0, (sunDir[1] + 0.05) / 0.1));
  const f = t * t * (3 - 2 * t);
  return pal.light.map(v => v * f);
}

// Hour to sun direction, for bxb.time: rises in the east (+x) at 6, highest
// at noon, sets in the west at 18. tilt is the noon elevation's complement,
// in degrees - a sun that never quite reaches overhead reads better.
export function sunForHour(hour, noonElevationDeg = 60) {
  const a = (hour - 6) / 12 * Math.PI;                  // 0 at dawn, PI at dusk
  const e = noonElevationDeg * Math.PI / 180;
  const x = Math.cos(a), y = Math.sin(a) * Math.sin(e), z = Math.sin(a) * Math.cos(e);
  const len = Math.hypot(x, y, z);
  return [x / len, y / len, z / len];
}
