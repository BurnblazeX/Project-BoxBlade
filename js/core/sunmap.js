// --- The sun depth map: the far half of the sun's shadow ---
//
// The marched sun (gpu/sun.js) is soft and exact, but it only looks 12 m
// toward the sun - the cone's field estimate saturates past ~17 m, and every
// metre more is a step for every lit texel. With 48 levels a tower throws a
// shadow a hundred metres long (World plan P1), so the march alone cuts it off.
//
// The far half is a depth map from the sun over the whole region: each texel
// holds s = dot(p, L) - base of the first surface the sun meets, L the sun's
// direction (toward it). For a receiver at p, s_map - s_p is how far toward the
// sun its farthest blocker is - and a blocker counts here only past the
// distance the march already covered, so the two never shadow the same thing
// twice nor leave a band between them. Because that threshold is metres and
// not a depth bias, the map has none of the acne the old CSM fought (gpu/sun.js).
//
// Softened inside C2, hard past it (vistas). The cone's analytic penumbra is
// the look that is wanted, so the march carries as much as the preset can pay
// for (gpu/sunmap.js sunSoftReach: 12 m, C1, C2) and the map's own softening is
// a smooth bilinear grid, never dithered taps. Redrawn only when the sun turns
// or the world changes. Rendering side: render/features/sunmap.js and
// gpu/sunmap.js.
import { World, ModelBoxes, BLOCK_METRES } from './world.js';

// The penumbra's widest, in metres - for the march past where its cone outgrows
// the field's 1 m distance range (open air would read as half-shadowed), and
// for the map's softening, so a shadow is as soft on either side of the switch.
export const SUN_PENUMBRA_CAP = 0.75;

// Over the last metres of C2 the map's softening fades to its single hard tap,
// so there is no ring where the vistas begin.
export const SUN_MAP_SOFT_FADE = 3;

// How far the sun turns, in radians, before the map is redrawn. A running day
// moves it 0.125 degrees a second; 0.001 rad redraws about every half second
// and moves a shadow 100 m away by 10 cm a redraw.
export const SUN_MAP_REDRAW = 0.001;

// The nearest a blocker may be and still count for the map, in metres - where
// the march covered less (a receiver outside C2, a vista, covers nothing). A
// map texel stands for a patch of surface, and a receiver anywhere on that
// patch, or a softening tap up to the penumbra cap away, can read its own
// surface a little toward the sun; a block's worth of margin keeps that from
// ever shadowing itself.
export const SUN_MAP_MIN_REACH = 1.5;

// The softening grid's side, most: side x side bilinear taps. The preset
// picks its largest (render/features/sunmap.js); each shade uses as few as its
// penumbra needs (sunMapGrid).
export const SUN_MAP_MAX_SIDE = 8;

// The widest a softening grid's taps may be spread, in map texels. Each tap is
// already a one-texel bilinear ramp; spread further, a straight shadow edge
// shows as offset copies of itself - the stepping first seen 2026-10-28 with
// a 2x2 grid spread over a 1.5 m penumbra.
export const SUN_MAP_TAP_SPACING = 1.5;

// The grid for a penumbra radius (map texels) under a preset's largest side:
// { side, radius }. Enough taps that none is more than SUN_MAP_TAP_SPACING
// from the next; where the preset's grid cannot cover the penumbra that
// finely, the penumbra narrows to what it can - a narrower soft edge, never a
// stepped one. The shader computes the same.
export function sunMapGrid(radius, maxSide) {
  const side = Math.max(1, Math.min(maxSide, Math.ceil((2 * radius) / SUN_MAP_TAP_SPACING)));
  return { side, radius: Math.min(radius, (side * SUN_MAP_TAP_SPACING) / 2) };
}

// --- Where the map lies, and how fine it is (World plan P2a) ---
//
// The map follows the player rather than covering the region. A blocker
// projects onto the map at the same texel as the shadow it casts, so only the
// receivers need covering - a square of SUN_MAP_WINDOW either side of the
// player's projection, the screen and its margins - while the depth range
// still spans the whole world along the sun. A small world whose projection
// fits keeps the map fitted to it.
export const SUN_MAP_WINDOW = 48;

// Its texel: what lets the softening grid reach the cone's full penumbra
// without spreading its taps - side taps SUN_MAP_TAP_SPACING apart cover
// side x 0.75 texels, so 1 / side metres a texel reaches 0.75 m. Hard (no
// grid): the texel lock's 12.5 cm. The size in texels follows from the window.
export const SUN_MAP_HARD_TEXEL = 0.125;
export function sunMapTexelFor(side) {
  return side > 0 ? SUN_PENUMBRA_CAP / ((side * SUN_MAP_TAP_SPACING) / 2) : SUN_MAP_HARD_TEXEL;
}
export function sunMapSizeFor(frame, texel) {
  return Math.min(4096, Math.max(64, Math.ceil((2 * frame.half) / texel)));
}

// Metres a margin around what is drawn, so the edge of the map is never a
// surface's edge.
const MARGIN = 1;

const TEXEL_METRES = BLOCK_METRES / 12;

// The world's box in metres, { min: [x, y, z], max: [x, y, z] }: every block
// (centred on b * 1.5, so +-0.75 about it) and every placed model's cubes
// (texels of 12.5 cm). null for an empty world.
export function sunMapBounds(world = World, modelBoxes = ModelBoxes) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  const h = BLOCK_METRES / 2;
  for (const key of world.keys()) {
    const c = key.split(',').map(Number);
    for (let a = 0; a < 3; a++) {
      const m = c[a] * BLOCK_METRES;
      if (m - h < min[a]) min[a] = m - h;
      if (m + h > max[a]) max[a] = m + h;
    }
  }
  for (const b of modelBoxes) {
    for (let a = 0; a < 3; a++) {
      min[a] = Math.min(min[a], b[a] * TEXEL_METRES);
      max[a] = Math.max(max[a], b[a + 3] * TEXEL_METRES);
    }
  }
  return min[0] > max[0] ? null : { min, max };
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = v => { const l = Math.hypot(...v); return v.map(x => x / l); };

// The map's frame for a sun direction (toward the sun, any length): an
// orthographic view down -L over the bounds, square, with a metre of margin.
//   dir        L, unit
//   right, up  the view's x and y - as three's lookAt builds them from up
//              (world +y, or +z for a sun straight overhead)
//   centre     the bounds' middle
//   half       half the square's side, metres
//   eye        how far up-sun of the centre the camera sits
//   near, far  its clip range from there
//   base       under every s in the bounds, so a stored s is never 0 - the
//              cleared value, which reads as "nothing toward the sun"
// With focus ({ x, y, z } metres) and a world wider than the window, the
// map follows: half is windowHalf and the centre sits on the focus's
// projection, snapped to whole texels (texel metres) so the map's grid stays
// put on the world as it moves and shadow edges do not shimmer. follows says
// which, and plane is the centre's [right, up] coordinates.
export function sunMapFrame(bounds, sunDirection, focus = null, windowHalf = SUN_MAP_WINDOW, texel = 0) {
  const L = norm([sunDirection.x ?? sunDirection[0], sunDirection.y ?? sunDirection[1],
                  sunDirection.z ?? sunDirection[2]]);
  const hint = Math.abs(L[1]) > 0.999 ? [0, 0, 1] : [0, 1, 0];
  const right = norm(cross(hint, L));
  const up = cross(L, right);
  const centre = [0, 1, 2].map(a => (bounds.min[a] + bounds.max[a]) / 2);
  let half = 0, lo = Infinity, hi = -Infinity, sMin = Infinity;
  for (let i = 0; i < 8; i++) {
    const c = [0, 1, 2].map(a => ((i >> a) & 1 ? bounds.max[a] : bounds.min[a]));
    const d = c.map((v, a) => v - centre[a]);
    half = Math.max(half, Math.abs(dot(d, right)), Math.abs(dot(d, up)));
    lo = Math.min(lo, dot(d, L));
    hi = Math.max(hi, dot(d, L));
    sMin = Math.min(sMin, dot(c, L));
  }
  half += MARGIN;
  const eye = hi + MARGIN;
  const fitted = { dir: L, right, up, centre, half, eye, near: 0, far: eye - lo + MARGIN, base: sMin - MARGIN,
                   follows: false, plane: [dot(centre, right), dot(centre, up)] };
  if (!focus || half <= windowHalf) return fitted;
  const f = [focus.x, focus.y, focus.z];
  const snap = v => (texel > 0 ? Math.round(v / texel) * texel : v);
  const plane = [snap(dot(f, right)), snap(dot(f, up))];
  // The new centre: the snapped projection, at the fitted centre's depth along
  // L - so eye, near and far, measured along L, are unchanged.
  const along = dot(centre, L);
  const moved = [0, 1, 2].map(a => right[a] * plane[0] + up[a] * plane[1] + L[a] * along);
  return { ...fitted, centre: moved, half: windowHalf, follows: true, plane };
}

// Whether a following map's focus has left the middle of its window - a
// quarter of the half-window from its centre - and the map wants redrawing.
export function sunMapMoved(frame, focus, fraction = 0.25) {
  if (!frame || !frame.follows) return false;
  const f = [focus.x, focus.y, focus.z];
  return Math.abs(dot(f, frame.right) - frame.plane[0]) > frame.half * fraction ||
         Math.abs(dot(f, frame.up) - frame.plane[1]) > frame.half * fraction;
}

// Metres a map texel covers.
export function sunMapTexelMetres(frame, size) {
  return (2 * frame.half) / size;
}

// Whether the sun has turned far enough from where the map was drawn to draw
// it again. null (never drawn) is always stale.
export function sunMapStale(drawnDir, sunDirection, threshold = SUN_MAP_REDRAW) {
  if (!drawnDir) return true;
  const a = norm(drawnDir), b = norm([sunDirection.x, sunDirection.y, sunDirection.z]);
  return Math.acos(Math.min(1, Math.max(-1, dot(a, b)))) > threshold;
}

// The far map's verdict for one tap, as the shader computes it: a blocker
// counts in full past the march's reach, not at all before the fade start
// (fadeStart x reach, where the march's own occlusion fades out), and in
// between in proportion - the march's fade turned around, so the two add up.
export function farOcclusion(blocker, reach, fadeStart) {
  const from = reach * fadeStart;
  return Math.min(1, Math.max(0, (blocker - from) / Math.max(reach - from, 1e-3)));
}
