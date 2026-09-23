import { VOXEL_METRES, sphereTrace, sampleDistance, worldToVoxelFloat,
         GRID_DIM, HIT_EPS, MIN_STEP, MAX_TRACE_STEPS } from './boxgrid.js';

// Defaults mirrored from gpu.js. See fadeWeightTSL there for why a march that
// ends at a LIMIT rather than at geometry has to fade out instead of stopping.
export const SHADOW_FADE_START = 0.97;
export const EDGE_FADE_VOXELS = 12;

// How much a sample at this point is allowed to darken. 1 well inside the
// march's reach, falling to 0 as it approaches the distance cap or the grid
// boundary. The CPU mirror of fadeWeightTSL.
export function fadeWeight(vp, t, maxDistance,
                           fadeStart = SHADOW_FADE_START,
                           edgeFade = EDGE_FADE_VOXELS) {
  const lo = Math.min(vp.x, vp.y, vp.z);
  const hi = Math.min(GRID_DIM - vp.x, GRID_DIM - vp.y, GRID_DIM - vp.z);
  // A zero width means no edge fade at all. Dividing by it gives 0/0 = NaN
  // exactly at the corner of the grid, and a NaN weight propagates into the
  // visibility as a black or white speck that no amount of tuning explains.
  const edge = edgeFade <= 0 ? 1
    : Math.min(1, Math.max(0, Math.min(lo, hi) / edgeFade));
  const a = fadeStart * maxDistance;
  // smoothstep(a, maxDistance, t), inverted.
  const u = Math.min(1, Math.max(0, (t - a) / Math.max(1e-6, maxDistance - a)));
  return edge * (1 - u * u * (3 - 2 * u));
}

// --- The sun: a marched cone, not a depth map ---
//
// The CSM is gone. Every argument for it was about RANGE - a sun ray has no
// termination distance, so a march truncates where the grid ends - and every
// argument against it was about everything else: a depth map needs a bias, and a
// bias needs a receiver-plane gradient, and a widening PCF kernel turns any
// residual bias error into a large dark region rather than a small one. Three
// interacting error sources to tune by eye, none of which the marched path has.
//
// What replaces it is what bxb.shadows always did - sphere-trace one ray toward
// the sun through the same distance field every other light uses - with the
// single ray widened into a CONE covering the sun's angular size. Softening then
// is not a filter applied to a shadow; it is the shadow, sampled over the area
// of the light that casts it.
//
// There are two ways to get the penumbra and both are here, because one is the
// reference for the other:
//
//   SAMPLED  - N rays over the solar disc, averaged. This is what a soft shadow
//              IS, so it is the ground truth, at N times the cost.
//   CONE     - one ray, min(d(t)/(R*t)) along it, using the stored distance as a
//              free estimate of how much room there is beside the ray.
//
// The cone trace needed the field widened before it could work at all. At the old
// DISTANCE_RANGE of one voxel, d saturated at 12.5 cm and a receiver 3 m from
// anything in OPEN AIR computed 0.125/(0.06*3) = 0.69 - 31% shadowed by nothing.
// At 8 voxels it only saturates past 16.6 m, beyond the march cap, so open sky
// reads as exactly lit everywhere a ray can reach.
//
// Either way the atlas is what makes it affordable: shading happens when the
// light or the footprint moves, not once per frame.

// The sun's angular DIAMETER in radians, and a STYLE KNOB rather than a physical
// constant. The real sun is ~0.0093 rad, which at 12.5 cm texels spreads a 4 m
// caster's shadow edge by 0.3 of a texel - correct, and completely invisible.
// Even 0.03 only reached about one texel.
//
// 0.12 puts a 4 m separation at ~3.8 texels: a penumbra you can watch stepping
// outward across the ground while contact shadows stay hard. Tunable at runtime
// with bxb.soften().
export const SUN_ANGULAR_SIZE = 0.12;

// Penumbra width in metres for a given occluder-receiver separation. This is the
// whole of "distance-based softening" as a formula, and nothing in the shader
// evaluates it - the cone sampling produces this width as a consequence of its
// geometry. It exists so the property can be asserted numerically in a test
// rather than judged by eye, which is how the CSM's softening went wrong.
export function penumbraMetres(separation, angular = SUN_ANGULAR_SIZE) {
  return Math.max(0, separation) * angular;
}

export function penumbraTexels(separation, angular = SUN_ANGULAR_SIZE) {
  return penumbraMetres(separation, angular) / VOXEL_METRES;
}

// Half-angle as a disc radius on the plane one unit along the sun direction.
// Every cone direction is normalize(forward + tangent*u*R + bitangent*v*R) for
// (u,v) in the unit disc, so R = tan(half-angle) is the only place the angular
// size enters.
export function coneRadius(angular = SUN_ANGULAR_SIZE) {
  return Math.tan(angular / 2);
}

// An orthonormal basis around the sun direction, built on the CPU because the
// sun moves on bxb.light() and not per texel. Doing it host-side also keeps the
// degenerate-axis choice out of the shader, where it would be a branch on every
// invocation to answer a question that is constant over the whole pass.
export function sunBasis(dir) {
  const len = Math.hypot(dir.x, dir.y, dir.z) || 1;
  const f = { x: dir.x / len, y: dir.y / len, z: dir.z / len };
  // Cross with whichever world axis is least parallel to the sun, so the basis
  // never collapses - including at the top of the sky, where the obvious choice
  // of up would.
  const ax = Math.abs(f.x), ay = Math.abs(f.y), az = Math.abs(f.z);
  const a = ay <= ax && ay <= az ? { x: 0, y: 1, z: 0 }
          : ax <= az ? { x: 1, y: 0, z: 0 } : { x: 0, y: 0, z: 1 };
  const t = {
    x: a.y * f.z - a.z * f.y,
    y: a.z * f.x - a.x * f.z,
    z: a.x * f.y - a.y * f.x
  };
  const tl = Math.hypot(t.x, t.y, t.z) || 1;
  t.x /= tl; t.y /= tl; t.z /= tl;
  const b = {
    x: f.y * t.z - f.z * t.y,
    y: f.z * t.x - f.x * t.z,
    z: f.x * t.y - f.y * t.x
  };
  return { forward: f, tangent: t, bitangent: b };
}

// Sample positions on the unit disc, as a Vogel spiral: r = sqrt((i+0.5)/n),
// theta = i * golden angle. Uniform in AREA, which is what a uniformly bright
// solar disc needs - the naive r = i/n crowds the rim and biases the penumbra
// outward.
//
// Deterministic rather than random, for three reasons: the pattern is identical
// on every re-shade, so walking does not make the penumbra shimmer; it has no
// clustering, so n samples buy n samples' worth of gradient; and the CPU can
// reproduce it exactly, which is what lets the shader be tested. The banding a
// fixed set would otherwise produce is broken up by rotating the whole pattern
// per texel - see coneDirections - and then hidden under the quantise step,
// which is coarser than the sample count anyway.
//
// n === 1 is the sun direction itself, not a point 0.7 of the way out. That is
// what makes rays=1 reproduce the old hard shadow exactly, which is the
// comparison every soft result is judged against.
export const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5));

export function coneOffsets(n) {
  if (n <= 1) return [[0, 0]];
  const out = [];
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt((i + 0.5) / n);
    const th = i * GOLDEN_ANGLE;
    out.push([r * Math.cos(th), r * Math.sin(th)]);
  }
  return out;
}

// The CPU mirror of what the shader computes, so the directions can be checked
// for being unit length, inside the cone, and centred on the sun.
export function coneDirections(dir, n, angular = SUN_ANGULAR_SIZE, rotation = 0) {
  const { forward, tangent, bitangent } = sunBasis(dir);
  const R = coneRadius(angular);
  const cs = Math.cos(rotation), sn = Math.sin(rotation);
  return coneOffsets(n).map(([u, v]) => {
    const du = (u * cs - v * sn) * R;
    const dv = (u * sn + v * cs) * R;
    const d = {
      x: forward.x + tangent.x * du + bitangent.x * dv,
      y: forward.y + tangent.y * du + bitangent.y * dv,
      z: forward.z + tangent.z * du + bitangent.z * dv
    };
    const l = Math.hypot(d.x, d.y, d.z) || 1;
    return { x: d.x / l, y: d.y / l, z: d.z / l };
  });
}

// The CPU reference for the whole technique, and the reason it exists is the
// same reason the CPU sphere trace exists: a shader that quietly disagrees about
// a penumbra is very hard to notice by eye, and "the softening looks wrong" was
// exactly the report that could never be pinned down on the depth-map path.
//
// This marches the same cone the kernel does, through the same field, using the
// same sphereTrace, so the penumbra width it produces can be asserted against
// penumbraMetres() in a headless test rather than judged from a screenshot.
//
// Returns visibility: 1 fully lit, 0 fully occluded, and in between the fraction
// of the solar disc the point can see.
export function coneVisibility(grid, p, n, sunDirection, {
  rays = 16, angular = SUN_ANGULAR_SIZE, maxDistance = 12,
  bias = 0.75, rotation = 0,
  fadeStart = SHADOW_FADE_START, edgeFade = EDGE_FADE_VOXELS
} = {}) {
  const origin = {
    x: p.x + n.x * bias * VOXEL_METRES,
    y: p.y + n.y * bias * VOXEL_METRES,
    z: p.z + n.z * bias * VOXEL_METRES
  };
  const dirs = coneDirections(sunDirection, rays, angular, rotation);
  let lit = 0;
  for (const d of dirs) {
    // Faded at the march's limits, exactly as the cone trace is - so a ray that
    // ran out of reach contributes a fraction of a shadow rather than none, and
    // the two techniques end a long shadow the same way.
    const r = sphereTrace(grid, origin, d, maxDistance);
    if (!r.hit) { lit += 1; continue; }
    const at = {
      x: origin.x + d.x * r.distance,
      y: origin.y + d.y * r.distance,
      z: origin.z + d.z * r.distance
    };
    const vp = worldToVoxelFloat(grid, at.x, at.y, at.z);
    lit += 1 - fadeWeight(vp, r.distance, maxDistance, fadeStart, edgeFade);
  }
  return lit / dirs.length;
}

// The CPU mirror of coneTraceSunTSL: the cheap technique, where one ray carries
// the whole penumbra because the field already knows how much room is beside it.
//
//   visibility = min over t of clamp( d(t) / (R * t) )
//
// R*t is the cone's radius in metres at distance t, so a surface the cone clears
// contributes 1, one it contains contributes 0, and one grazing its edge
// contributes the fraction - and because R*t grows with t, a distant occluder
// softens while contact stays hard. See gpu.js for why this needed the field
// widened to 8 voxels before it could work at all.
// Returns the visibility alone. Most callers only want that.
export function coneTraceVisibility(grid, p, n, sunDirection, opts = {}) {
  return coneTraceSample(grid, p, n, sunDirection, opts).visible;
}

// Returns the visibility AND whether the march concluded, which is the mirror of
// what the shader hands the atlas kernel.
//
// concluded is false only when the ray ran off the edge of the field without
// hitting anything - it did not discover that the texel is lit, it discovered
// nothing. That distinction is what stops the shadows flickering: the atlas
// declines to overwrite a stored answer with a non-answer, so a shadow measured
// while its caster was in range survives the caster leaving. See the store in
// createAtlasShadePass.
export function coneTraceSample(grid, p, n, sunDirection, {
  angular = SUN_ANGULAR_SIZE, maxDistance = 12, bias = 0.75,
  fadeStart = SHADOW_FADE_START, edgeFade = EDGE_FADE_VOXELS
} = {}) {
  const R = coneRadius(angular);
  const l = Math.hypot(sunDirection.x, sunDirection.y, sunDirection.z) || 1;
  const dir = { x: sunDirection.x / l, y: sunDirection.y / l, z: sunDirection.z / l };
  const o = {
    x: p.x + n.x * bias * VOXEL_METRES,
    y: p.y + n.y * bias * VOXEL_METRES,
    z: p.z + n.z * bias * VOXEL_METRES
  };

  // Starts one voxel out, not at zero: the cone has no radius at t = 0, so the
  // ratio would divide by zero, and the surface being left is the nearest thing
  // to the ray by construction.
  let t = VOXEL_METRES;
  let res = 1;
  for (let i = 0; i < MAX_TRACE_STEPS; i++) {
    const wx = o.x + dir.x * t, wy = o.y + dir.y * t, wz = o.z + dir.z * t;
    const v = worldToVoxelFloat(grid, wx, wy, wz);
    // Running off the edge of the field is the only exit that can conclude
    // nothing - and even then, only if the ray found no occlusion on the way.
    // res is a minimum over the path, so running out of field can only ever have
    // missed MORE occlusion, never less: a value below 1 is a real measurement
    // and an upper bound. A ray that found nothing has established only that it
    // could not tell, which is the case the flicker came from.
    if (v.x < 0 || v.y < 0 || v.z < 0 ||
        v.x >= GRID_DIM || v.y >= GRID_DIM || v.z >= GRID_DIM) {
      const out = Math.min(1, Math.max(0, res));
      return { visible: out, concluded: out < 0.999 };
    }

    const w = fadeWeight(v, t, maxDistance, fadeStart, edgeFade);
    const d = sampleDistance(grid, wx, wy, wz);
    // A real hit occludes fully, but only in proportion to w - a hit found at the
    // very end of the march is one the next texel along will miss entirely.
    if (d < HIT_EPS) {
      return { visible: Math.min(1, Math.max(0, Math.min(res, 1 - w))), concluded: true };
    }
    const ratio = Math.min(1, Math.max(0, (d * VOXEL_METRES) / (R * t)));
    res = Math.min(res, 1 - (1 - ratio) * w);
    t += Math.max(d, MIN_STEP) * VOXEL_METRES;
    // Spending the whole distance budget IS a conclusion: anything past maxDistance
    // is out of scope by design, not unknown.
    if (t > maxDistance) return { visible: Math.min(1, Math.max(0, res)), concluded: true };
  }
  return { visible: Math.min(1, Math.max(0, res)), concluded: true };
}

// --- The texel lock, mirrored from texelLockTSL ---
//
// The shading position, snapped to the centre of the world texel it falls in,
// across the face only - the component along the normal is left exact, because
// rounding that would move the origin off the surface the bias is measured from.
//
// This is what makes a shadow a property of the SURFACE rather than of the
// screen: every fragment inside one 12.5 cm texel marches the identical ray, so
// the shadow comes out in whole texels at the density of the sprite art and
// stays painted on the ground under any camera. It is the whole of what the
// texel atlas was for, without the atlas.
export function texelLock(p, n, q = VOXEL_METRES) {
  const snap = (v, axis) => {
    const a = Math.abs(axis);
    return Math.floor(v / q) * q + q * 0.5 + (v - (Math.floor(v / q) * q + q * 0.5)) * a;
  };
  return { x: snap(p.x, n.x), y: snap(p.y, n.y), z: snap(p.z, n.z) };
}

// --- The shading formula, mirrored from sunShadeTSL ---
//
// This is the arithmetic that was wrong: it existed twice in gpu.js, the atlas
// copy applied Lambert and the per-pixel copy did not, and at a 20 degree sun
// that made the same ground 0.572 under one and 1.0 under the other - the whole
// world reading as permanent partial shade on the atlas path. There is one copy
// in the shader now and this is its CPU mirror, so the disagreement is a test
// failure rather than something to notice by eye.
//
// ndotl is the raw dot of the face normal with the sun; sunUpDot is the sun
// direction's y, which is what a horizontal up-facing surface would receive.
export const GROUND_REFERENCE_FLOOR = 0.25;
export const DEFAULT_AMBIENT = 0.17;

export function sunShade(visible, ndotl, sunUpDot, ambient = DEFAULT_AMBIENT,
                         mode = 'ground') {
  const v = Math.min(1, Math.max(0, visible));
  if (mode === 'flat') return ambient + v * (1 - ambient);
  let nl = Math.max(0, ndotl);
  if (mode === 'ground') {
    nl = Math.min(1, nl / Math.max(sunUpDot, GROUND_REFERENCE_FLOOR));
  }
  return ambient + v * nl * (1 - ambient);
}

// A 4x4 Bayer matrix, and the reason it is indexed by INTEGER TEXEL COORDINATE
// rather than screen position: screen-space dithering crawls across surfaces
// under camera motion. Indexed by texel, the pattern is painted onto the
// surface and holds still - the same argument as the atlas itself.
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
