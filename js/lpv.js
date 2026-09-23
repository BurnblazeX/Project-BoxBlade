import { GRID_DIM } from './boxgrid.js';

// --- The light propagation volume: GI's long band (doc §6.2) ---
//
// A coarse, dense grid of RGB light in AIR, laid over a cascade and moving with
// it - one over C0 at half-block cells, one over C1 at full-block cells.
// Three steps a frame, all on the GPU (gi.js), defined here on the CPU so the
// kernels have a reference to be checked against:
//
//   solid     how much of each cell is geometry, from the distance field
//   inject    light leaving the surfaces around each air cell - the direct
//             light that lands on them (sun and every light in the list) times
//             their albedo. The lit SURFACE is the source, not the lamp: a lamp
//             injected at its own position only glows, where one injected where
//             it lands bleeds its colour off the wall it lights (§6.2)
//   propagate each air cell takes its own injection plus a share of what its
//             neighbours hold, blocked by solid cells
//
// Deliberately NOT the 12.5 cm surface-radiance grid the doc routes injection
// through. GI only ever reads this volume, trilinearly, at 0.75 m, so it is
// injected at 0.75 m directly - no 24 MB radiance grid, no bricking. The fine
// grid returns with reflections, which need its sharpness.
//
// Scalar RGB per cell rather than the SH or six-direction flux a classic LPV
// carries. Direction comes from WHERE the gather samples - at the hit point of
// each AO cone, not at the receiver (§6.2) - so a texel beside a red wall picks
// up red from the cones that struck it. What a scalar field gives up is light
// arriving at a point from one side only; at 0.75 m under blocky direct light,
// that is detail the look discards anyway.

// 24^3 over C0's 144^3: six voxels a cell, two cells a block, so a halfTile is
// exactly one cell and C0's whole-block re-origin is a whole-cell shift.
export const LPV_CELL_VOXELS = 6;
export const LPV_DIM = GRID_DIM / LPV_CELL_VOXELS;              // 24
export const LPV_CELLS = LPV_DIM * LPV_DIM * LPV_DIM;           // 13 824
export const LPV_CELL_METRES = LPV_CELL_VOXELS * 0.125;         // 0.75 m
export const LPV_EXTENT_METRES = LPV_DIM * LPV_CELL_METRES;     // 18 m, = C0
// Which cascades carry a volume: C0 at half-block cells, C1 at full-block cells
// (36 m). Past C1 there is no bounce - it fades out over C1's last cells.
export const LPV_LEVELS = 2;

// How much of a neighbour's light each iteration carries on. The steady state
// in open air is E / (1 - alpha), so the output is multiplied back by
// (1 - alpha): near a source the volume then reads roughly what was injected,
// and alpha only decides how FAR it spreads, not how bright it is.
export const DEFAULT_LPV_SPREAD = 0.75;
// Iterations a frame. Light advances one cell per iteration, so 8 crosses the
// 24-cell volume in three frames (§6.2).
export const DEFAULT_LPV_ITERATIONS = 8;
// Injection is the expensive pass - a shadow march per light per surface hit -
// and light that has not moved gives the same answer again, so each frame
// re-injects one slice of the cells. 4 slices refreshes everything every 4th
// frame, which at 60 fps is a torch lagging its bounce by ~50 ms.
export const DEFAULT_LPV_SLICES = 4;
// Scale on the gathered bounce before it is added to the direct light.
export const DEFAULT_GI_STRENGTH = 2.0;

// A cell counts as solid, for injection and for the dilated read-back, at half.
export const SOLID_THRESHOLD = 0.5;

export function cellIndex(x, y, z) {
  return x + y * LPV_DIM + z * LPV_DIM * LPV_DIM;
}

export function cellCoords(i) {
  return {
    x: i % LPV_DIM,
    y: Math.floor(i / LPV_DIM) % LPV_DIM,
    z: Math.floor(i / (LPV_DIM * LPV_DIM))
  };
}

// The six axis directions, in the order every pass walks them.
export const FACE_DIRS = [
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
];

// One propagation step, on the CPU - the reference the GPU kernel mirrors.
//
//   L'(c) = open(c) * ( E(c) + alpha * sum_f open(n_f) * L(n_f) / 6 )
//
// A neighbour off the edge of the volume counts as holding this cell's own
// value: zero there would darken a rim around the whole volume that follows the
// player, which is exactly the cascade-edge artifact the shadows fade to avoid.
//
// L and E are Float32Array(LPV_CELLS * 4) (rgb + unused), solid is
// Float32Array(LPV_CELLS) in 0..1. Writes into out, returns it.
export function propagateStep(L, E, solid, alpha, out, dim = LPV_DIM) {
  const at = (x, y, z) => x + y * dim + z * dim * dim;
  for (let z = 0; z < dim; z++) {
    for (let y = 0; y < dim; y++) {
      for (let x = 0; x < dim; x++) {
        const c = at(x, y, z);
        const open = 1 - solid[c];
        let r = 0, g = 0, b = 0;
        for (const [dx, dy, dz] of FACE_DIRS) {
          const nx = x + dx, ny = y + dy, nz = z + dz;
          const inside = nx >= 0 && ny >= 0 && nz >= 0 && nx < dim && ny < dim && nz < dim;
          const n = inside ? at(nx, ny, nz) : c;
          const w = inside ? 1 - solid[n] : open;
          r += w * L[n * 4]; g += w * L[n * 4 + 1]; b += w * L[n * 4 + 2];
        }
        const k = alpha / 6;
        out[c * 4] = open * (E[c * 4] + k * r);
        out[c * 4 + 1] = open * (E[c * 4 + 1] + k * g);
        out[c * 4 + 2] = open * (E[c * 4 + 2] + k * b);
        out[c * 4 + 3] = 0;
      }
    }
  }
  return out;
}

// How far a volume has to shift when its cascade re-origins, in cells, or null
// when the move is not a whole number of cells or leaves nothing to keep - both
// mean start over instead. cellMetres is the volume's cell size: 0.75 m over C0,
// 1.5 m over C1.
export function lpvShift(from, to, cellMetres = LPV_CELL_METRES) {
  if (!from) return null;
  const s = [to.x - from.x, to.y - from.y, to.z - from.z].map(d => d / cellMetres);
  if (s.some(v => Math.abs(v - Math.round(v)) > 1e-4)) return null;
  const r = s.map(Math.round);
  if (r.some(v => Math.abs(v) >= LPV_DIM)) return null;
  return { x: r[0], y: r[1], z: r[2] };
}

// The shift itself, on the CPU: cell c of the new volume is cell c + shift of
// the old one, or empty where that falls outside.
export function shiftVolume(src, shift, out, stride = 4, dim = LPV_DIM) {
  for (let z = 0; z < dim; z++) {
    for (let y = 0; y < dim; y++) {
      for (let x = 0; x < dim; x++) {
        const c = x + y * dim + z * dim * dim;
        const sx = x + shift.x, sy = y + shift.y, sz = z + shift.z;
        const inside = sx >= 0 && sy >= 0 && sz >= 0 && sx < dim && sy < dim && sz < dim;
        const s = sx + sy * dim + sz * dim * dim;
        for (let k = 0; k < stride; k++) out[c * stride + k] = inside ? src[s * stride + k] : 0;
      }
    }
  }
  return out;
}

// Mean colour of an RGBA8 image, linearised from sRGB - the albedo a block
// texture bounces, until the material system (§6.4) gives each block its own.
export function averageAlbedo(rgba) {
  const lin = c => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i + 3] < 128) continue;
    r += lin(rgba[i]); g += lin(rgba[i + 1]); b += lin(rgba[i + 2]); n++;
  }
  return n ? { r: r / n, g: g / n, b: b / n } : { r: 0.5, g: 0.5, b: 0.5 };
}
