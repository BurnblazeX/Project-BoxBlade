import { BLOCK_METRES, Y_MIN, Y_MAX, CHUNK_SIZE } from './world.js';
import { VOXEL_METRES, TEXELS_PER_BLOCK } from './boxgrid.js';

// --- Cascaded shadow map for the directional sun ---
//
// The one place a per-light depth pass is accepted (doc 6.1b), and only for a
// single directional light. Local lights keep sphere tracing the distance field;
// do not generalise this back to them.
//
// Why a depth pass at all, when the whole architecture is built on ray marching:
// a long directional march is the case marching is worst at. The ray travels
// tens of metres, leaves C0, and reads as open sky - which is exactly the
// truncated shadow visible at the edge of the screen. A sun ray has no
// termination distance the way a torch does, so there is nothing to cap.
//
// Why mixing techniques does not show: shading happens in texel space, so the
// atlas quantises whatever it is fed. Sampling a depth map once per texel snaps
// it to the same 12.5 cm grid a ray would. Sun shadows and torch shadows come
// out at matching blockiness despite completely different underlying methods,
// which is normally the objection to mixing them and does not apply here.
//
// RESOLUTION IS LOCKED TO THE TEXEL GRID. Doc 7.3 requires cascades and texel
// mips to step together, so each cascade's shadow texel is a whole number of
// boxGrid voxels: 1, 2 and 4. At 144 texels per cascade that reproduces the
// cascade table in 4 exactly - 18 m, 36 m, 72 m - and makes the shadow map the
// same structure as the boxGrid rather than a parallel one with its own
// resolution to tune. Going finer would be discarded by the atlas anyway.

export const SHADOW_DIM = CHUNK_SIZE * TEXELS_PER_BLOCK;   // 144, as C0
export const CASCADE_COUNT = 3;

// Metres per shadow texel, per cascade: 12.5 cm, 25 cm, 50 cm.
export function cascadeTexelMetres(level) {
  return VOXEL_METRES * (1 << level);
}

// Side of the square each cascade covers: 18 m, 36 m, 72 m.
export function cascadeExtent(level) {
  return SHADOW_DIM * cascadeTexelMetres(level);
}

// The depth the ortho camera has to span, and it DEPENDS ON THE SUN ANGLE.
//
// The first version of this was `extent + vertical * 2`, a guess that ignored
// the sun entirely, and it was 40% short at a 23 degree sun. Casters above or
// below the middle of the range clipped out of the depth pass and their shadows
// simply vanished - which is not a subtle artifact, it is a missing shadow.
//
// The bound is exact, not padded. Work in the light basis: a point's world
// height is y = u * up.y + depth * forward.y, so depth = (u * up.y - y) /
// forward.y. Over the footprint u spans the extent and y spans the authorable
// vertical range, giving
//
//     depth = (extent * |up.y| + vertical) / |forward.y|
//
// Overhead sun: |up.y| = 0, |forward.y| = 1, so it collapses to the vertical
// range alone - 18 m, and nothing is wasted. Low sun: the 1/|forward.y| term
// takes over, because a grazing light genuinely does have to see that far to
// find the caster. At 23 degrees this is 88.5 m rather than the 54 m the guess
// produced.
//
// Precision is not the cost it looks like. Half float resolves ~1/2048 near
// 1.0, so even a 200 m range quantises to 10 cm - still inside the normal-offset
// bias. Lateral resolution is the scarce resource and that stays pinned to the
// texel grid.
const MIN_SUN_SINE = 0.08;   // ~4.6 degrees; below this the range would explode

// The authorable vertical span in world metres, and its MIDPOINT. The midpoint
// is not zero: y runs from Y_MIN*BLOCK to (Y_MAX+1)*BLOCK, which for -3..8 is
// -4.5..13.5 m, centred on 4.5 m. Centring the depth window on the player's
// height instead - which is what the first version did - clipped the near plane
// at every sun angle, because the span it had to cover was lopsided around it.
export function verticalSpan() {
  const lo = Y_MIN * BLOCK_METRES;
  const hi = (Y_MAX + 1) * BLOCK_METRES;
  return { lo, hi, range: hi - lo, mid: (lo + hi) / 2 };
}

export function cascadeDepth(level, sunDirection) {
  const vertical = verticalSpan().range;
  const extent = cascadeExtent(level);
  // No sun given: fall back to the overhead case, which is the tight bound for
  // a light straight down and the smallest honest answer available.
  if (!sunDirection) return vertical;
  const b = lightBasis(sunDirection);
  // right.y is always 0 - `right` is the horizontal cross(worldUp, forward) -
  // so only the `up` axis contributes a lateral term here.
  const sine = Math.max(Math.abs(b.forward.y), MIN_SUN_SINE);
  return (extent * Math.abs(b.up.y) + vertical) / sine;
}

const EPS = 1e-6;
const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a, b) => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x
});
function norm(v) {
  const l = Math.hypot(v.x, v.y, v.z) || 1;
  return { x: v.x / l, y: v.y / l, z: v.z / l };
}

// The light's orthonormal basis. `sunDirection` points TOWARD the sun, matching
// the convention the sphere-traced path uses, so the camera looks along its
// negation.
//
// DERIVED THE WAY A THREE CAMERA DERIVES ITS OWN, deliberately, because the
// camera is what actually rasterises the map and any disagreement shows up as a
// mirrored or upside-down shadow map. Matrix4.lookAt builds
//
//     z = normalize(eye - target)   which is -forward, since a camera looks down -Z
//     x = normalize(cross(up, z))
//     y = cross(z, x)
//
// The first version of this took `right = cross(up, forward)`, off the view
// direction rather than off the camera's +Z. Since z = -forward, that is
// cross(up, -z) = -x: right came out NEGATED for every sun direction, so the map
// was mirrored along u. up was unaffected, which is why the fault looked like a
// flip rather than an obvious transpose.
export function lightBasis(sunDirection) {
  const forward = norm({ x: -sunDirection.x, y: -sunDirection.y, z: -sunDirection.z });
  // A camera looks along -Z, so its third basis vector is the NEGATED view
  // direction. Everything else hangs off this, so it is stated once.
  const z = { x: -forward.x, y: -forward.y, z: -forward.z };
  // World up degenerates when the sun is directly overhead; any perpendicular
  // will do there, and Z is as good as another.
  const upRef = Math.abs(forward.y) > 1 - 1e-4 ? { x: 0, y: 0, z: 1 } : { x: 0, y: 1, z: 0 };
  const right = norm(cross(upRef, z));
  return { forward, right, up: cross(z, right) };
}

// Fit a cascade to a square centred on `centre`, in the light's basis.
//
// The centre is SNAPPED to whole shadow texels. Without it, moving the camera
// slides the map by a fraction of a texel and every shadow edge in the scene
// re-quantises, which in texel-space shading does not shimmer continuously - it
// flickers the moment a page is re-shaded, which is worse because it draws the
// eye. Snapping costs one round per axis and removes the whole class.
export function fitCascade(level, centre, sunDirection) {
  const basis = lightBasis(sunDirection);
  const extent = cascadeExtent(level);
  const depth = cascadeDepth(level, sunDirection);
  const texel = extent / SHADOW_DIM;

  // Snap in light space, then rebuild the world position from the snapped
  // components so the residual along `forward` does not matter (depth is
  // compared, not indexed).
  const cr = Math.round(dot(centre, basis.right) / texel) * texel;
  const cu = Math.round(dot(centre, basis.up) / texel) * texel;
  // The depth window is positioned so it spans the whole authorable vertical
  // range, which means solving for the light-space depth of the cascade's
  // lateral centre AT THE MID HEIGHT rather than at the player's feet:
  //   p.y = right.y*r + up.y*u + forward.y*d,  right.y = 0
  //   =>  d = (y - up.y*u) / forward.y
  const span = verticalSpan();
  const fy = basis.forward.y;
  const cf = Math.abs(fy) > EPS
    ? (span.mid - basis.up.y * cu) / fy
    : dot(centre, basis.forward);

  // Camera sits half the depth back along the view direction, so the whole
  // range in front of it is covered.
  const originF = cf - depth / 2;
  const position = {
    x: basis.right.x * cr + basis.up.x * cu + basis.forward.x * originF,
    y: basis.right.y * cr + basis.up.y * cu + basis.forward.y * originF,
    z: basis.right.z * cr + basis.up.z * cu + basis.forward.z * originF
  };

  return { level, ...basis, position, extent, depth, texel, halfExtent: extent / 2 };
}

// World position to a cascade's shadow-map coordinates. u and v are 0..1 across
// the map; depth is 0..1 across the cascade's depth range, which is what the
// depth pass writes, so the comparison is a plain subtraction.
export function worldToShadow(cascade, p) {
  const d = sub(p, cascade.position);
  return {
    u: dot(d, cascade.right) / cascade.extent + 0.5,
    v: dot(d, cascade.up) / cascade.extent + 0.5,
    depth: dot(d, cascade.forward) / cascade.depth
  };
}

// Which cascade to use: the finest one that contains the point with a margin
// wide enough for the soft filter's kernel to stay inside the map. Selecting by
// containment rather than by view distance means it stays correct when the
// cascades are centred somewhere other than the camera.
export function selectCascade(cascades, p, marginTexels = 4) {
  for (const c of cascades) {
    const s = worldToShadow(c, p);
    const m = marginTexels / SHADOW_DIM;
    if (s.u > m && s.u < 1 - m && s.v > m && s.v < 1 - m &&
        s.depth > 0 && s.depth < 1) {
      return c.level;
    }
  }
  return -1;   // beyond every cascade: unshadowed, open sky
}

// --- Distance-based softening ---
//
// The replacement for PCF against a depth map, and it is the same rule the
// sphere-traced lights use (doc 6.1): the sample footprint widens with the
// distance the light travelled to get here, because that is how a penumbra
// physically grows. A contact shadow stays hard; a shadow cast from far away
// spreads. No separate blur pass.
//
// Expressed in SHADOW TEXELS so it composes with the cascade lockstep: the same
// physical penumbra is a smaller kernel in a coarser cascade, which is both
// correct and cheaper exactly where detail matters least.
//
// lightAngularSize is the sun's angular diameter in radians, and it is a STYLE
// KNOB, not a physical constant. The real sun is ~0.0093 rad, which at 12.5 cm
// texels spreads a 4 m-high caster's shadow edge by 0.3 of a texel - correct,
// and completely invisible. Even 0.03 only reached about one texel.
//
// 0.12 puts a 4 m separation at ~3.8 texels, which is a penumbra you can
// actually see stepping outward across the ground while contact shadows stay
// hard. Tunable at runtime with bxb.soften().
export const SUN_ANGULAR_SIZE = 0.12;

export function penumbraTexels(receiverDepth, blockerDepth, cascade,
                               angularSize = SUN_ANGULAR_SIZE) {
  // Both are normalised over the cascade's depth range; the separation in metres
  // is what drives the penumbra.
  const separation = Math.max(0, (receiverDepth - blockerDepth)) * cascade.depth;
  return (separation * angularSize) / cascade.texel;
}

// A 4x4 Bayer matrix, and the reason it is indexed by INTEGER TEXEL COORDINATE
// rather than screen position: screen-space dithering crawls across surfaces
// under camera motion. Indexed by texel, the pattern is painted onto the
// surface and holds still - which is the same argument as the atlas itself.
export const BAYER4 = [
   0,  8,  2, 10,
  12,  4, 14,  6,
   3, 11,  1,  9,
  15,  7, 13,  5
];

export function bayer4(tx, ty) {
  const x = ((tx % 4) + 4) % 4;
  const y = ((ty % 4) + 4) % 4;
  return BAYER4[y * 4 + x] / 16;
}

// Quantise the soft falloff into discrete steps rather than leaving it a smooth
// gradient. Doc 6.1: a smooth gradient under flat-shaded pixel art reads as a
// rendering bug, whereas stepped and dithered reads as intentional. The dither
// is added before flooring, so the step boundary itself breaks up across texels
// instead of forming a visible contour line.
export function quantiseShadow(value, steps, dither = 0) {
  const v = Math.min(1, Math.max(0, value));
  return Math.min(1, Math.max(0, Math.floor(v * steps + dither) / steps));
}

// World texel coordinate, so the dither pattern is anchored to the world rather
// than to a page's position in the atlas. A page moving in the atlas must not
// change the pattern painted on it.
export function worldTexelIndex(p) {
  return {
    tx: Math.floor(p.x / VOXEL_METRES),
    ty: Math.floor(p.z / VOXEL_METRES) + Math.floor(p.y / VOXEL_METRES)
  };
}

// --- Reading a cascade back ---
//
// WebGPU requires copyTextureToBuffer rows to be a multiple of 256 bytes, so a
// readback buffer is PADDED and its length is not width*height*channels. At 144
// texels of R16 a row is 288 bytes, padded to 512, so the element stride per row
// is 256 and not 144. Deriving the stride as length/(w*h) gives 1.77; indexing a
// typed array at a fractional position returns undefined, and undefined decodes
// to zero - which is how a perfectly ordinary map got reported as entirely
// empty. This exists so that arithmetic has a test on it.
//
// Returns null rather than guessing when nothing fits: an invented stride is
// worse than no reading at all, because it comes with false confidence.
export function readbackLayout(rawLength, bytesPerElement, w = SHADOW_DIM, h = SHADOW_DIM) {
  for (const channels of [1, 2, 4]) {
    const bpt = channels * bytesPerElement;
    const rowBytes = Math.ceil(w * bpt / 256) * 256;
    const bytes = (h - 1) * rowBytes + w * bpt;
    if (Math.ceil(bytes / bytesPerElement) === rawLength) {
      return { channels, rowElements: rowBytes / bytesPerElement };
    }
  }
  return null;
}

// --- Clip space to shadow-map texture coordinates ---
//
// WEBGPU'S TEXTURE V RUNS OPPOSITE THE LIGHT'S UP AXIS. A WebGPU texture's row 0
// is its TOP row, and NDC y = +1 is also the top, so the correct mapping is
// v_texture = 0.5 - ndc.y * 0.5, not the WebGL-convention 0.5 + ndc.y * 0.5.
// three's own ShadowNode does the same thing - `shadowCoord.y.oneMinus()`,
// commented "follow webgpu standards".
//
// Getting this wrong mirrors the depth lookup vertically across the map, so
// every shadow is fetched from the wrong side of the light's view. It does not
// look like a flipped image; it looks like large dark regions in the wrong
// places, which is far harder to attribute to its cause.
//
// The TSL version in gpu.js (clipToShadowUV) MUST match this. This is the
// definition the tests pin, and it is the one to change if the convention ever
// does.
export function shadowUVFromNDC(ndcX, ndcY) {
  return { u: ndcX * 0.5 + 0.5, v: 0.5 - ndcY * 0.5 };
}
