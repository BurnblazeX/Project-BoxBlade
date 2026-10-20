import * as THREE from 'three/webgpu';
import { Fn, If, float, vec3, vec4, uniform, max, texture, vec2, normalize, clamp, cos, sin, int, mix } from 'three/tsl';
import { VOXEL_METRES } from '../core/boxgrid.js';
import { BAYER4, SUN_ANGULAR_SIZE, coneOffsets, coneRadius, sunBasis } from '../core/sun.js';
import { CARD_SUN_REACH } from '../core/cards.js';
import { layout } from './layout.js';
import { EDGE_FADE_VOXELS, SHADOW_FADE_START, SURFACE_BIAS_VOXELS } from './field.js';
import { glassRayTransmitTSL } from './glass.js';
import { createConeTraceSunTSL, createShadowTraceTSL, createReflectTraceTSL } from './trace.js';
import { createCardsTSL } from './cards.js';
import { skyOpenTSL } from './volume.js';

// --- The sun's march, past its reach ---
//
// The march stops at its cap (12 m) and fades out before it, and a ray that
// met nothing by then counts as lit. Under a low sun, a ray from a hall's floor
// travels further than that before it reaches the roof - so the sun lit the
// floor through the roof from sunrise until it climbed high enough. Where the
// ray stopped tells: still under a roof (core/skytop.js, volume.js
// skyOpenTSL), it has not reached the sun, and the sun is shadowed. A ray
// that left through a window or a door ends outside and stays lit; a shaft
// through a window further than the cap is lost (dark, not a leak).
// sunRoofUniform: the A/B - 0, the cap alone, as before. A uniform: no rebuild.
export const sunRoofUniform = uniform(int(1));

// 1, or 0 where a sun ray from origin along dir, reach long, ends under a
// roof. terrain: the terrain textures (their sky-top map), or null: always 1.
// What the LPV's sun marches take; the shading's go further (createSunFarTSL).
export function sunRoofTSL(terrain, origin, dir, reach) {
  if (!terrain || !terrain.skyTop) return float(1);
  return mix(float(1), skyOpenTSL(terrain, origin.add(dir.mul(reach))), float(sunRoofUniform));
}

// --- Past the cap, under a roof: the hard march on ---
//
// A ray still lit at the cap but under a roof is not given up on: it marches
// on, hard (no cone - past ~17 m the cone's field saturates and would shadow
// open air), to sunFarUniform metres in all - all of C1 by default. It finds
// the wall or the roof that shadows it, or leaves through a window, tinted by
// the glass it crossed - a stained window's light on a floor 30 m away. Ending
// that far out still under a roof, it is shadowed (sunRoofTSL). Only those
// rays march: a ray lit at the cap in the open is lit, as it always was.
export const DEFAULT_SUN_FAR = 36;
export const sunFarUniform = uniform(float(DEFAULT_SUN_FAR));

// Returns (origin, dir, reach) -> what the visibility from the soft march is
// multiplied by: 1 where it was not under a roof at the cap, else 0 or the
// glass's tint (a colour, with glass). null without the sky-top map.
function createSunFarTSL(cascades, terrain, glass) {
  if (!terrain || !terrain.skyTop) return null;
  const trace = createReflectTraceTSL(cascades, false, !!glass);
  return (origin, dir, reach) => {
    const out = (glass ? vec3(1, 1, 1) : float(1)).toVar();
    const end = origin.add(dir.mul(reach)).toVar();
    If(sunRoofUniform.equal(int(1)).and(skyOpenTSL(terrain, end).lessThan(float(0.5))), () => {
      const more = max(sunFarUniform.sub(reach), float(0)).toVar();
      const r = trace(end, dir, more).toVar();
      const t = glass ? r.y : r.w;
      const roofed = skyOpenTSL(terrain, end.add(dir.mul(more))).lessThan(float(0.5));
      If(t.greaterThanEqual(float(0)).or(roofed), () => { out.assign(glass ? vec3(0, 0, 0) : float(0)); })
        .Else(() => { if (glass) out.assign(glassRayTransmitTSL(glass, vec4(0, 0, r.z, r.w), end, dir)); });
    });
    return out;
  };
}

// ---------------------------------------------------------------------------
// The sun: a marched cone
//
// This replaced a cascaded shadow map, and the replacement is smaller than the
// thing it replaced by about seven hundred lines. See sun.js for the argument;
// the short version is that a depth map needs a bias, a bias needs a
// receiver-plane gradient to be stable under a moving cascade, and a softening
// kernel that widens with distance turns any leftover bias error into a large
// dark region rather than a small speckle. The marched path has none of those
// error sources, because it never compares two depths - it asks the distance
// field whether anything is in the way, which is the same question every other
// light in the renderer asks.
//
// The sun is the ONE light with no termination distance, which is the property
// the CSM existed for and the one thing this does not solve: a ray leaving the
// grid reads as open sky, so a long shadow truncates at the footprint edge. That
// is a RANGE problem, addressed by how far the field reaches, not by how the
// shadow is filtered - so it is deliberately not mixed into this.
// ---------------------------------------------------------------------------

// The dither source. Four by four, nearest, repeating: indexed by integer world
// texel, so the pattern is painted onto surfaces rather than crawling across
// them under camera motion.
export function createBayerTexture() {
  const tex = new THREE.DataTexture(
    new Uint8Array(BAYER4.map(v => v * 16)), 4, 4, THREE.RedFormat);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

// The soft sun. Returns visibility: 1 fully lit, 0 fully occluded, and the values
// between are the fraction of the solar disc this point can see.
//
// Signature is (position, normal) -> float, which is exactly what the atlas shade
// kernel already calls, so the sun technique is interchangeable at that seam -
// the whole point of shading in texel space.
//
// rayCount is a COMPILE-TIME argument, not a uniform: the loop is unrolled on the
// host so the cone offsets are literals and the kernel has no dynamic loop bound.
// Changing it is a kernel rebuild, which is why main.js tracks it alongside the
// other pass-shape flags.
// --- One shading formula, used by both paths ---
//
// This existed twice and the two copies disagreed. The atlas applied Lambert and
// the per-pixel path did not, so at a 20 degree sun flat ground came out at
// 0.35 + 1*sin(20)*0.65 = 0.572 under bxb.atlas() and 1.0 under bxb.shadows() -
// the same geometry, the same sun, 43% darker on one of them, reading as though
// the whole world were in partial shade. Unshaded atlas pages are cleared to 1.0,
// which is why the pages that had not been written yet looked RIGHT and the
// shaded ones looked wrong.
//
// So there is now one function and both call it. Two copies of a formula is a
// disagreement waiting to happen; two copies of a formula where one is a
// simplification of the other is a disagreement that has already happened.
//
// THE N.L PROBLEM, which is an art question rather than a bug. Lambert is
// physically right and it is also why the sun's elevation was controlling the
// whole scene's brightness: at 20 degrees every up-facing surface - which in a
// top-down game is nearly all of them - is multiplied by 0.34. Correct, and too
// dark, with no second light and no tonemapping to recover it.
//
//   'flat'       no N.L at all. What bxb.shadows always did. Bright, but an
//                unshadowed wall facing away from the sun reads exactly as bright
//                as one facing it, which is what made the first atlas look flat.
//   'lambert'    raw N.L. Physically right, dim at a low sun.
//   'ground'     N.L normalised against what a horizontal surface receives, so
//                flat ground under open sky is always fully lit and the sun's
//                ANGLE changes shadow direction and length rather than overall
//                brightness. Faces still differentiate: one turned away still
//                falls to ambient. This is the default.
//
// The reference is floored so that a sun on the horizon does not divide every
// surface up to full brightness and collapse the differentiation entirely.
export const GROUND_REFERENCE_FLOOR = 0.25;

export const DEFAULT_AMBIENT = 0.17;

// ao scales the AMBIENT term only - it is occlusion of light arriving from all
// around, and says nothing about the one direction the sun comes from.
export function sunShadeTSL({ visible, n, sunDir, ambientUniform, mode = 'ground', ao = null }) {
  const amb = ao ? ambientUniform.mul(ao) : ambientUniform;
  if (mode === 'flat') return amb.add(visible.mul(float(1).sub(ambientUniform)));

  const ndotl = max(n.dot(sunDir), float(0)).toVar();
  if (mode === 'ground') {
    // sunDir.y IS the N.L a horizontal up-facing surface would receive.
    const ref = max(sunDir.y, float(GROUND_REFERENCE_FLOOR));
    ndotl.assign(clamp(ndotl.div(ref), float(0), float(1)));
  }
  return amb.add(visible.mul(ndotl).mul(float(1).sub(ambientUniform)));
}

export function createSoftSunTSL({ cascades, sunDir, sunTangent,
                                  sunBitangent, maxDist, coneRadiusUniform,
                                  rayCount = 1, bayerTex = null,
                                  stepsUniform = null, biasUniform = null,
                                  cone = false, quantise = true,
                                  fadeStartUniform = null, edgeFadeUniform = null,
                                  cards = null, glass = false, terrain = null }) {
  // terrain: the terrain textures, for the sky-top map (sunRoofTSL); null: none.
  // glass: a world with glass - the terrain textures (for the tint of the
  // texel a ray enters glass through), or false. The visibility is then a
  // COLOUR (vec3), the light tinted by the glass it crossed.
  const offsets = coneOffsets(rayCount);
  const cardsFn = createCardsTSL(cards);
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const fadeStart = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const edgeFade = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));
  // Built once per pass, not per invocation: the cascade list is host-side, so
  // the level walk is unrolled into the kernel.
  const coneTrace = createConeTraceSunTSL(cascades, !!glass);
  const shadowTrace = createShadowTraceTSL(cascades);
  // The march on past the cap (createSunFarTSL): the cone path's sees glass;
  // the sampled disc's does not, as its own march does not.
  const far = createSunFarTSL(cascades, terrain, cone && glass);
  const outType = glass ? 'vec3' : 'float';
  const asOut = v => (glass ? vec3(v, v, v) : v);

  // The cone trace gets the whole penumbra from a single ray, so the sample
  // pattern, its per-texel rotation and the sample count are all irrelevant to
  // it - it is a different way of asking the question, not a cheaper sampling of
  // the same one. Kept as an early return rather than threaded through the loop
  // below so that neither path carries the other's machinery.
  if (cone) {
    return Fn(([p, n, skip]) => {
      // toVar, not a bare expression. TSL hoists a node into a variable at the
      // point it is first REUSED, and here that is inside the C1 loop - which
      // only runs when C0 did not conclude. Everywhere the march ended in C0,
      // the card test below then read an origin that was never assigned, (0,0,0),
      // and every sprite shadow vanished from that region.
      const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();
      const traced = coneTrace(origin, sunDir, maxDist, coneRadiusUniform,
                               fadeStart, edgeFade).toVar();
      const visible = traced.x.toVar();
      const through = glass ? glassRayTransmitTSL(glass, traced, origin, sunDir).toVar() : null;
      if (far) {
        const f = far(origin, sunDir, maxDist);
        if (glass) through.mulAssign(f); else visible.mulAssign(f);
      }
      // Sprites left the distance field so that each light could have them
      // facing IT - see createCardsTSL. Same ray, same cap, same cone radius, so
      // a character's penumbra and a wall's come out of one formula.
      if (cardsFn) {
        visible.mulAssign(cardsFn(origin, sunDir, float(CARD_SUN_REACH),
                                  coneRadiusUniform, skip));
      }
      // The cone trace is ANALYTIC: d/(R*t) is a continuous function of the
      // geometry, so its penumbra is already smooth and there is nothing here a
      // dither is fixing. Quantising it throws that away and replaces it with
      // banding, then breaks the banding up with a crosshatch. Off by default for
      // this path - it is a style choice, not a correctness one.
      if (!quantise || !stepsUniform || !bayerTex) {
        return glass ? through.mul(visible) : visible;
      }
      const wt = p.div(float(VOXEL_METRES)).floor();
      const dither = texture(bayerTex, wt.xz.add(vec2(0, wt.y)).mul(float(0.25))).r;
      const stepped = clamp(visible.mul(stepsUniform).add(dither).floor().div(stepsUniform),
                            float(0), float(1));
      return glass ? through.mul(stepped) : stepped;
    }).setLayout(layout('sunCone', outType, [
      { name: 'p', type: 'vec3' }, { name: 'n', type: 'vec3' }, { name: 'skip', type: 'vec2' }]));
  }

  return Fn(([p, n, skip]) => {
    // Lift off the face once, not per ray: the origin depends on the surface,
    // not on which part of the sun is being sampled.
    const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();

    // Rotate the whole sample pattern by a per-texel angle. Without this the
    // fixed offsets make the penumbra a set of rayCount+1 discrete bands that
    // are identical on every surface, which reads as contouring rather than as
    // softness. Anchored to the world texel for the same reason the quantise
    // dither is - a pattern that moves with the camera crawls.
    const rot = float(0).toVar();
    if (bayerTex) {
      const wt = p.div(float(VOXEL_METRES)).floor();
      rot.assign(texture(bayerTex, wt.xz.add(vec2(0, wt.y)).mul(float(0.25))).r
                   .mul(float(Math.PI * 2)));
    }
    const cs = cos(rot).toVar();
    const sn = sin(rot).toVar();

    const lit = float(0).toVar();
    for (let i = 0; i < offsets.length; i++) {
      const [u, v] = offsets[i];
      // Unrolled, so u and v are literals and the rotation is two multiply-adds.
      const du = cs.mul(float(u)).sub(sn.mul(float(v))).mul(coneRadiusUniform);
      const dv = sn.mul(float(u)).add(cs.mul(float(v))).mul(coneRadiusUniform);
      const dir = normalize(sunDir.add(sunTangent.mul(du)).add(sunBitangent.mul(dv)));
      // Returns weighted occlusion, so visibility is its complement.
      lit.addAssign(float(1).sub(shadowTrace(origin, dir, maxDist, fadeStart, edgeFade).x));
    }
    const visible = lit.div(float(offsets.length)).toVar();
    if (far) visible.mulAssign(far(origin, sunDir, maxDist));
    // The sampled path takes the cards once rather than per disc sample: the
    // card test is already analytic, so sampling it N times would return the
    // same answer N times and pay for it each one.
    if (cardsFn) {
      visible.mulAssign(cardsFn(origin, sunDir, float(CARD_SUN_REACH),
                                coneRadiusUniform, skip));
    }

    // Quantise the soft tail into discrete steps rather than leaving a smooth
    // gradient: under flat-shaded pixel art a gradient reads as a rendering bug,
    // where stepped and dithered reads as intentional (doc 6.1). A single ray is
    // already two values, so there is no tail to quantise and the dither would
    // only punch holes in a binary edge.
    // The sampled disc does not see glass (it is not the default path); it
    // answers in the same type as the cone so either can be swapped in.
    if (!quantise || !stepsUniform || !bayerTex || offsets.length === 1) {
      return asOut(visible);
    }

    const wt = p.div(float(VOXEL_METRES)).floor();
    const dither = texture(bayerTex, wt.xz.add(vec2(0, wt.y)).mul(float(0.25))).r;
    return asOut(clamp(visible.mul(stepsUniform).add(dither).floor().div(stepsUniform),
                       float(0), float(1)));
  }).setLayout(layout('sunSampled', outType, [
      { name: 'p', type: 'vec3' }, { name: 'n', type: 'vec3' }, { name: 'skip', type: 'vec2' }]));
}

// The uniforms the cone needs, as one object, because the sun direction and its
// two basis vectors must be written TOGETHER or the cone stops being centred on
// the light - a basis left over from the previous sun tilts the penumbra off to
// one side, which looks like a softening bug rather than a stale uniform.
export function createSunUniforms(sunDirection, angular = SUN_ANGULAR_SIZE) {
  const b = sunBasis(sunDirection);
  return {
    dir: uniform(new THREE.Vector3(b.forward.x, b.forward.y, b.forward.z)),
    tangent: uniform(new THREE.Vector3(b.tangent.x, b.tangent.y, b.tangent.z)),
    bitangent: uniform(new THREE.Vector3(b.bitangent.x, b.bitangent.y, b.bitangent.z)),
    coneRadius: uniform(float(coneRadius(angular))),
    angular
  };
}

export function writeSunUniforms(u, sunDirection, angular = u.angular) {
  const b = sunBasis(sunDirection);
  u.dir.value.set(b.forward.x, b.forward.y, b.forward.z);
  u.tangent.value.set(b.tangent.x, b.tangent.y, b.tangent.z);
  u.bitangent.value.set(b.bitangent.x, b.bitangent.y, b.bitangent.z);
  u.coneRadius.value = coneRadius(angular);
  u.angular = angular;
  return u;
}
