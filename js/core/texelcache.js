// The texel cache's sample spacing (main.js renderTexelCache). Its own module
// so the numbers are testable headless - render.js loads textures through Vite.
import { BLOCK_METRES } from './world.js';
import { BLOCK_TEXELS } from './materials.js';

// The screen can be smaller than an emulated/rendered viewport. Always cover
// the actual drawing size too, including odd dimensions at spacing two.
export function splitCapacityFor(width, height) {
  const w = Math.max(1, Math.ceil(width)), h = Math.max(1, Math.ceil(height));
  const full = Math.min(w * h, 1 << 24);
  return { full, low: Math.min(Math.ceil(w / 2) * Math.ceil(h / 2), 1 << 22),
           miss: Math.ceil(full / 16), table: tableSlotsFor(full) };
}

// The visible-texel table (gpu/texel.js createSplitShading, table mode): one
// slot per TABLE_PIXELS_PER_SLOT pixels, a power of two. The texels on screen
// are far fewer than the pixels (today's resolve shades at most the low-res
// samples plus full / 16 misses, ~0.4 of the slots), so the table stays well
// under half full and its probe runs short. At most TABLE_MAX_SLOTS: the
// kernels go one thread a texel by an indirect dispatch, at most 65535
// workgroups of 64.
export const TABLE_PIXELS_PER_SLOT = 4;
export const TABLE_MIN_SLOTS = 1 << 12;
export const TABLE_MAX_SLOTS = 1 << 21;
// Slots a key may sit past its home - both how far an insert carries a key
// and how far a pixel looks for its own.
export const TABLE_PROBES = 8;
export function tableSlotsFor(pixels) {
  const want = Math.max(1, Math.ceil(pixels / TABLE_PIXELS_PER_SLOT));
  return Math.max(TABLE_MIN_SLOTS, Math.min(TABLE_MAX_SLOTS, 2 ** Math.ceil(Math.log2(want))));
}

// The texel cache's low-res pass samples every `spacing` pixels, and each
// sample shades its whole texel. Any spacing gives the same image - a pixel
// with no sample of its own texel nearby shades itself in the miss pass - so
// the spacing is purely a cost trade: too fine and each texel is shaded many
// times over (a 9 px texel at 4 px spacing: ~5 times), too coarse and small
// texels (far, or foreshortened faces) miss and shade per pixel.
// Measured at 2560x1440 (texels ~9.9 px across at the focus): spacing 4 2.49 ms,
// 5 2.30, 6 2.37, 8 3.04. The best sits at about half a texel: the smallest
// faces on screen are foreshortened to ~0.6 of a texel's width, and the lookup
// finds a sample only within one low-res pixel.
export const TEXEL_METRES = BLOCK_METRES / BLOCK_TEXELS;
export const TEXEL_SPACING_PER_TEXEL = 0.5;
export const TEXEL_SPACING_MIN = 2;
export const TEXEL_SPACING_MAX = 8;
// A texel's size in pixels at the camera's focus: a perspective camera at
// `distance` with vertical fov `fovDeg`, drawing `drawHeight` pixels.
export function texelPixels(drawHeight, fovDeg, distance) {
  return drawHeight * TEXEL_METRES / (2 * distance * Math.tan(fovDeg * Math.PI / 360));
}
// The spacing for a texel size. `current` is the spacing in use: it is kept
// until the ideal is clearly past its rounding edge, so a camera easing
// between modes does not resize the target back and forth.
export function texelCacheSpacingFor(texelPx, current = 0) {
  const want = texelPx * TEXEL_SPACING_PER_TEXEL;
  const clampS = s => Math.max(TEXEL_SPACING_MIN, Math.min(TEXEL_SPACING_MAX, s));
  if (current && Math.abs(want - current) < 0.65) return clampS(current);
  return clampS(Math.round(want));
}
