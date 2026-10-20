import { float, vec3, uniform, max, normalize, clamp, int, select, sqrt, pow, round } from 'three/tsl';
import { METAL_F0 } from '../core/materials.js';

// ---------------------------------------------------------------------------
// Specular (§6.3): LabPBR decode, GGX highlights, one reflection ray
//
// The terrain's specular texel (materials.js has the CPU mirror of the decode)
// becomes three things per fragment:
//
//   highlights  GGX from the sun and every point light, taking the SAME
//               visibility the diffuse term marched - no extra rays.
//   reflection  one ray along reflect(-V, N), marched through the field like
//               any shadow ray. Where it lands, the hit is shaded on the spot:
//               its block's own texture (looked up in the block-material
//               volume, render.js), the sun and the light list with their
//               shadow marches, and the LPV bounce. Past the field, the sky.
//   fresnel     Schlick, roughness-aware, weighting the reflection; the diffuse
//               is scaled by what the reflection took, and metals have none.
//
// There is no surface-radiance grid yet (§7.4), so the hit is lit on the fly
// rather than read from a cache. What that costs: one reflection march plus
// one sun march per reflective fragment, and nothing for any other.
//
// Everything is evaluated at the texel-locked position, the view vector too,
// so a highlight or a reflection lands in whole texels like the shadows do.
// ---------------------------------------------------------------------------
export const REFLECT_MAX_DISTANCE = 24;   // metres

// Reflections in reflections: how many surfaces a reflection ray may follow,
// 1 (the reflection alone) to REFLECT_MAX_BOUNCES. A uniform, so the debug
// slider is a write, not a rebuild (createShadowColorNode, nestedReflect).
export const REFLECT_MAX_BOUNCES = 16;

export const DEFAULT_REFLECT_BOUNCES = 2;

export const reflectBouncesUniform = uniform(int(DEFAULT_REFLECT_BOUNCES));

// A/B: a reflection ray that met nothing sees the sky only if it ended above
// its column's roof (gpu/shade.js skyMiss) - 0 shows the sky indoors too.
// A uniform: no rebuild.
export const reflectEscapeUniform = uniform(int(1));

// Rough surfaces fall back from the ray to the ambient estimate over this
// band of alpha (LabPBR roughness, (1 - smoothness)^2). Grass, at alpha 1,
// never marches.
export const REFLECT_ROUGH_START = 0.2;

export const REFLECT_ROUGH_END = 0.45;

// DEFAULT_SPECULAR as a texel: rough, so it reflects nothing further.
export const ROUGH_SPEC = [0, 10 / 255, 0, 1];

// GGX with a near-zero alpha is a delta: a highlight one texel wide and
// thousands bright, flickering in and out as the view moves. The sun is a disc,
// not a point, so its highlight on a mirror is a disc too - this is that floor.
export const SPEC_MIN_ALPHA = 0.03;

// Past this a highlight is clipped white anyway; the cap keeps a torch against
// polished metal from blooming into a flat white rectangle.
export const SPEC_MAX = 4;

// LabPBR specular texel (0-1 per channel) -> { alpha, f0, metal }.
export function decodeSpecularTSL(s, albedo) {
  const g = s.g.mul(float(255)).toVar();
  const metal = select(g.greaterThanEqual(float(229.5)), float(1), float(0)).toVar();
  // 230-237: the standard's predefined metals, F0 from materials.js's n/k table.
  const idx = clamp(round(g.sub(float(230))), float(0), float(7)).toVar();
  let table = vec3(...METAL_F0[7]);
  for (let i = 6; i >= 0; i--) table = select(idx.equal(float(i)), vec3(...METAL_F0[i]), table);
  // 238-255: some other metal - the albedo is its F0.
  const metalF0 = select(g.greaterThanEqual(float(237.5)), albedo, table);
  const f0 = select(metal.greaterThan(float(0.5)), metalF0, vec3(s.g, s.g, s.g)).toVar();
  const rough = float(1).sub(s.r);
  return { alpha: rough.mul(rough).toVar(), f0, metal };
}

// Fresnel for the reflection, Schlick with a roughness term: a rough surface's
// grazing reflection does not climb all the way to white.
export function fresnelRoughTSL(f0, nv, alpha) {
  const k = pow(float(1).sub(nv), float(5));
  return f0.add(max(vec3(float(1).sub(alpha)), f0).sub(f0).mul(k));
}

// GGX + height-correlated Smith + Schlick, times N.L and PI - the scale the
// diffuse is in here (albedo x light x N.L: a Lambert BRDF of albedo / PI with
// the PI folded into the light). Returns a function of the light direction,
// for the light loop to call per light.
export function specularTSL(n, v, alpha, f0) {
  const a = max(alpha, float(SPEC_MIN_ALPHA));
  const a2 = a.mul(a).toVar();
  const nv = max(n.dot(v), float(1e-4)).toVar();
  return l => {
    const h = normalize(l.add(v));
    const nl = max(n.dot(l), float(0));
    const nh = max(n.dot(h), float(0));
    const vh = max(v.dot(h), float(0));
    const dd = nh.mul(nh).mul(a2.sub(float(1))).add(float(1));
    const D = a2.div(float(Math.PI).mul(dd).mul(dd));
    const vis = float(0.5).div(max(
      nl.mul(sqrt(nv.mul(nv).mul(float(1).sub(a2)).add(a2)))
        .add(nv.mul(sqrt(nl.mul(nl).mul(float(1).sub(a2)).add(a2)))), float(1e-5)));
    const F = f0.add(vec3(1, 1, 1).sub(f0).mul(pow(float(1).sub(vh), float(5))));
    return F.mul(D.mul(vis).mul(nl).mul(float(Math.PI)));
  };
}
