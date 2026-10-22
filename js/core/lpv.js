import { GRID_DIM, cascadeVoxelMetres } from './field.js';
import { BLOCK_METRES } from './world.js';
import { DEFAULT_SPECULAR, decodeSpecular } from './materials.js';

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
// Which cascades carry a volume, at how many cells a block a side: all of C1
// at 2 (0.75 m cells, 36 m - C0 lies inside it) and C2 at 1 (1.5 m cells,
// 72 m), finest first. Past C2 there is no bounce - it fades out over C2's
// last cells. A finer volume keeps its reach in metres by itself: gi.js runs
// res x the iterations and spreads spread^(1/res) a cell. (Until 2026-10-21:
// C0 at 0.75 m over 18 m, C1 at 1.5 m. On 2026-10-22 C1 at 3 a block - the
// BoxGrid, 0.5 m, 72^3 - was tried and dropped: steep for next to no gain.)
export const LPV_VOLUMES = [{ level: 1, blockCells: 2 }, { level: 2, blockCells: 1 }];

// A volume's res for gpu/gi.js createLPV: its cells a side over the 24 its
// cascade has at LPV_CELL_VOXELS voxels a cell - 2 for both of the above
// (48^3 cells each).
export function lpvVolumeRes({ level, blockCells }) {
  return blockCells * LPV_CELL_VOXELS * cascadeVoxelMetres(level) / BLOCK_METRES;
}

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
// Bounces: the injection is the first (direct light off a surface); each one
// after it is the order before it, landing on the same surfaces and leaving
// again times their albedo (gi.js). One propagate dispatch carries every order,
// so a bounce costs ALU and memory, not dispatches.
export const GI_MAX_BOUNCES = 16;
export const DEFAULT_GI_BOUNCES = 2;
// GI resolution (gi.js createLPV res): a block face's cells a side, halved.
// 1 is the 0.75 m cell over C0 - 2 x 2 a block face; 1.5, 2 and 3 give 3 x 3,
// 4 x 4 and 6 x 6 over the same extent, at ~3.4x, 8x and 27x the cells. Each
// keeps a whole number of voxels a cell (6, 4, 3, 2) and of cells a block.
export const LPV_RES_OPTIONS = [1, 1.5, 2, 3];
// Scale on the gathered bounce before it is added to the direct light.
export const DEFAULT_GI_STRENGTH = 1.0;

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
// trans: optional, per cell, the rgb its light is multiplied by (glass - see
// glassCellTransmit); 4 floats a cell like L. null for none.
export function propagateStep(L, E, solid, alpha, out, dim = LPV_DIM, trans = null) {
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
        const tr = trans ? trans[c * 4] : 1, tg = trans ? trans[c * 4 + 1] : 1;
        const tb = trans ? trans[c * 4 + 2] : 1;
        out[c * 4] = open * (E[c * 4] + k * r) * tr;
        out[c * 4 + 1] = open * (E[c * 4 + 1] + k * g) * tg;
        out[c * 4 + 2] = open * (E[c * 4 + 2] + k * b) * tb;
        out[c * 4 + 3] = 0;
      }
    }
  }
  return out;
}

// --- Sky visibility: the fourth channel ---
//
// The light is RGB; the cell's fourth float carries how much of the sky it
// can see, V in [0, 1], for the sky ambient (gpu/shade.js) - so a roofed room
// is not lit by the sky through its roof, the LPV-mode answer to texel GI's
// escaping rays. Not light propagated by the same rule: that one loses what
// flows into solids, and an open cell beside a wall would read dim. Instead:
//
//   V'(c) = open(c) * trans(c) * ( 1                          sky-open cell
//                                  skySpread * mean_f V(n_f)   roofed cell )
//
// A sky-open cell (air above its column's roof - core/skytop.js) sees the
// sky. A roofed one sees what its air neighbours see, less a share a cell:
// the mean over its open neighbours (weighted by how open each is), not the
// sum over six - so an alcove off the open air is nearly as bright as the
// air, and a room through a door darkens a cell at a time. Off the volume a
// neighbour is the cell itself, as for the light.
// V, skyOpen (0/1 a cell), solid: Float32Array; trans: as propagateStep's
// (4 floats a cell, its rgb averaged) or null. Writes into out, returns it.
export const DEFAULT_LPV_SKY_SPREAD = 0.9;
export function propagateSkyStep(V, skyOpen, solid, skySpread, out, dim = LPV_DIM, trans = null) {
  const at = (x, y, z) => x + y * dim + z * dim * dim;
  for (let z = 0; z < dim; z++) {
    for (let y = 0; y < dim; y++) {
      for (let x = 0; x < dim; x++) {
        const c = at(x, y, z);
        const open = 1 - solid[c];
        let sum = 0, wsum = 0;
        for (const [dx, dy, dz] of FACE_DIRS) {
          const nx = x + dx, ny = y + dy, nz = z + dz;
          const inside = nx >= 0 && ny >= 0 && nz >= 0 && nx < dim && ny < dim && nz < dim;
          const n = inside ? at(nx, ny, nz) : c;
          const w = inside ? 1 - solid[n] : open;
          sum += w * V[n]; wsum += w;
        }
        const t = trans ? (trans[c * 4] + trans[c * 4 + 1] + trans[c * 4 + 2]) / 3 : 1;
        const seen = skyOpen[c] > 0.5 ? 1 : skySpread * sum / Math.max(wsum, 1e-4);
        out[c] = open * t * seen;
      }
    }
  }
  return out;
}

// How far a volume has to shift when its cascade re-origins, in cells, or null
// when the move is not a whole number of cells or leaves nothing to keep - both
// mean start over instead. cellMetres is the volume's cell size: 0.75 m over C0,
// 1.5 m over C1.
export function lpvShift(from, to, cellMetres = LPV_CELL_METRES, dim = LPV_DIM) {
  if (!from) return null;
  const s = [to.x - from.x, to.y - from.y, to.z - from.z].map(d => d / cellMetres);
  if (s.some(v => Math.abs(v - Math.round(v)) > 1e-4)) return null;
  const r = s.map(Math.round);
  if (r.some(v => Math.abs(v) >= dim)) return null;
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

// What a block material bounces DIFFUSELY: its mean albedo over the texels
// that are not metal (LabPBR specular green >= 230), times the fraction that
// is not. A metal has no diffuse term - its colour is all reflection - so an
// iron floor bounces almost nothing, and marble bounces its own dark grey.
// specRgba null means no _s texture: all dielectric.
export function diffuseAlbedo(rgba, specRgba = null) {
  const lin = c => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  let r = 0, g = 0, b = 0, n = 0, total = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i + 3] < 128) continue;
    total++;
    if (specRgba && specRgba[i + 1] >= 230) continue;
    r += lin(rgba[i]); g += lin(rgba[i + 1]); b += lin(rgba[i + 2]); n++;
  }
  if (!total) return { r: 0.5, g: 0.5, b: 0.5 };
  // The mean of the dielectric texels, weighted by how much of the surface
  // they are: n / total. Written out, that is their sum over the total.
  return { r: r / total, g: g / total, b: b / total };
}

// What a block material bounces SPECULARLY, into the LPV: its mean F0 over
// its opaque texels (metal's from the LabPBR table or its albedo, a dielectric's
// grey), as the diffuse share above is its mean diffuse albedo - so metal, which
// bounces nothing diffusely, bounces this. The LPV holds no direction, so a
// reflection goes in as a glow at the surface.
//
// mirrored: the mirror light (gpu/mirrorlight.js createMirrorLightTSL) carries a smooth
// texel's reflection to where it lands, and the GI then bounces it from THERE
// (gi.js) - so a smooth texel's share must not also go in here, or it would be
// counted twice. Each texel then counts only as far as it is too rough for the
// mirror light: 1 - smooth, over the same band (REFLECT_ROUGH_*).
export const REFLECT_ROUGH_START = 0.2;   // gpu/specular.js REFLECT_ROUGH_START
export const REFLECT_ROUGH_END = 0.45;    // gpu/specular.js REFLECT_ROUGH_END
export function specularAlbedo(rgba, specRgba = null, { mirrored = false } = {}) {
  const lin = c => {
    const v = c / 255;
    return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
  };
  const smoothstep = (a, b, x) => {
    const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
    return t * t * (3 - 2 * t);
  };
  let r = 0, g = 0, b = 0, total = 0;
  for (let i = 0; i < rgba.length; i += 4) {
    if (rgba[i + 3] < 128) continue;
    total++;
    const s = specRgba ? [specRgba[i], specRgba[i + 1], specRgba[i + 2], specRgba[i + 3]]
                       : DEFAULT_SPECULAR;
    const d = decodeSpecular(s, [lin(rgba[i]), lin(rgba[i + 1]), lin(rgba[i + 2])]);
    const w = mirrored ? smoothstep(REFLECT_ROUGH_START, REFLECT_ROUGH_END, d.roughness) : 1;
    r += d.f0[0] * w; g += d.f0[1] * w; b += d.f0[2] * w;
  }
  if (!total) return { r: 0.04, g: 0.04, b: 0.04 };
  return { r: r / total, g: g / total, b: b / total };
}

// How light through one cell of glass is tinted: the glass's tint through a
// block (its mean linear albedo), to the power of the cell's share of a block -
// the Beer-Lambert of gpu/glass.js glassTransmitTSL - blended by how much of the cell
// is glass. The LPV's propagation multiplies each cell's light by this
// (propagateStep's trans), so a room behind stained glass fills with its colour.
export function glassCellTransmit(tint, cellMetres, glassFraction, blockMetres = 1.5) {
  const k = cellMetres / blockMetres;
  const t = c => 1 + (Math.pow(Math.max(c, 1e-3), k) - 1) * glassFraction;
  return { r: t(tint.r), g: t(tint.g), b: t(tint.b) };
}

// --- Sprites in the LPV ---
//
// Sprites are not in the field, so the LPV cannot see them the way it sees the
// terrain. Each goes in as a lit box instead - its quad's width square, its
// height tall - which both INJECTS (its bounce colour, into the cells the box
// overlaps) and OCCLUDES (partial solidity in those cells, so light crossing a
// crowd is dimmed). fill is how much of that box counts: the sprite is a flat
// card, mostly air, so its opaque fraction times SPRITE_GI_FILL.
export const MAX_GI_SPRITES = 16;
export const SPRITE_GI_FILL = 0.5;

// The overlap of a cell (min corner, side) with an axis-aligned box (centre,
// half extents), as a fraction of the cell's volume. The CPU mirror of the
// kernel's test.
export function boxCellOverlap(cellMin, side, centre, half) {
  let f = 1;
  for (let a = 0; a < 3; a++) {
    const lo = Math.max(cellMin[a], centre[a] - half[a]);
    const hi = Math.min(cellMin[a] + side, centre[a] + half[a]);
    f *= Math.max(0, hi - lo) / side;
  }
  return f;
}
