// The texel cache's sample spacing (main.js renderTexelCache). Its own module
// so the numbers are testable headless - render.js loads textures through Vite.
import { BLOCK_METRES } from './world.js';
import { BLOCK_TEXELS } from './materials.js';

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
