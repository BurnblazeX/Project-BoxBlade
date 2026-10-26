import { smokeTransmitTSL } from './smokeshadow.js';
import { tableGroup } from './uploads.js';
import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, vec3, uniform, max, abs, vec4, min, clamp, length, log2, uniformArray, int, select, mix } from 'three/tsl';
import { VOXEL_METRES } from '../core/field.js';
import { MAX_LIGHT_LEVEL, MAX_LIGHTS, LIGHT_VEC4S, DEFAULT_LIGHT_CUTOFF } from '../core/lights.js';
import { layout } from './layout.js';
import { EDGE_FADE_VOXELS, SHADOW_FADE_START, SURFACE_BIAS_VOXELS } from './field.js';
import { glassRayTransmitTSL } from './glass.js';
import { createConeTraceSunTSL } from './trace.js';
import { createCardsTSL } from './cards.js';

// --- Dynamic point lights, marched through the same field ---
//
// The reason a moving light is affordable at all, and it is worth being explicit
// because it is the opposite of how a shadow map behaves: the distance field is
// GEOMETRY, not light. Moving a torch invalidates nothing. There is no re-bake,
// no re-origin, no second cascade set, no shadow map to re-render from a new
// point of view - the light moves, and the next frame marches the same field it
// already had toward a different point. That is the whole cost.
//
// The march is capped at the distance to the light rather than at a fixed range,
// which is both correct and the cheapest possible cap: anything past the light
// cannot shadow it. Combined with the windowed falloff reaching zero exactly at
// the radius, a level-15 torch is a 22.5 m ray at worst and a level-0 torch is
// no ray at all.


// --- The light list, bound ---

// A light row's second vec4 (colour, source radius) with the radius made
// positive: its SIGN says whether the light is shadowed at secondary hits
// (lights.js packLight), which only the light loop reads.
export const lightRowBTSL = raw => vec4(raw.xyz, abs(raw.w));

// LIGHT_VEC4S vec4 per light, laid out as lights.js packLight writes them. A
// uniform array for the same reason the cards are one: 16 lights is 768 bytes,
// and a uniform read is the cheaper of the two in a per-fragment loop.
export function createLightBindings(capacity = MAX_LIGHTS) {
  return {
    capacity,
    data: uniformArray(new Array(capacity * LIGHT_VEC4S).fill(0)
                         .map(() => new THREE.Vector4()), 'vec4').setName('bxbLights').setGroup(tableGroup),
    count: uniform(int(0)),
    // lights.js cutAmount: the sliver of each light's reach not worth a march.
    cutoff: uniform(float(DEFAULT_LIGHT_CUTOFF))
  };
}

export function writeLightBindings(b, packed, count) {
  const v = b.data.array;
  const n = Math.min(count, b.capacity);
  for (let i = 0; i < n * LIGHT_VEC4S; i++) {
    v[i].set(packed[i * 4], packed[i * 4 + 1], packed[i * 4 + 2], packed[i * 4 + 3]);
  }
  b.count.value = n;
  return b;
}

// Returns Fn([p, n, skip, shadeN]) -> the summed colour of every live light.
// The body is the single torch's, unchanged, run once per row of the list.
// smoke: gpu/smokeshadow.js smokeShared, or null - every light is dimmed by
// the cloud spells between it and the point (smokeSoft: through their soft
// copy - the smoke lighting itself).
export function createPointLightsTSL({ cascades, lights, biasUniform, fadeStartUniform,
                                edgeFadeUniform, cards = null, glass = false, smoke = null,
                                smokeSoft = false }) {
  // Ranged: the point lights share one card buffer and each walks only the
  // slice that was turned to face it.
  const cardsFn = createCardsTSL(cards, { ranged: true });
  // The SUN'S cone marcher, unchanged. Its cone radius is slope * t, and a
  // point light's cone is also linear in t - it just has a different slope, and
  // one that varies per fragment rather than being a uniform. So there is one
  // soft-shadow marcher in the renderer, not two, and a penumbra that disagreed
  // between the sun and a torch would be a bug in one shared function rather
  // than a difference between two plausible ones.
  const trace = createConeTraceSunTSL(cascades, !!glass);
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const fadeStart = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const edgeFade = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));

  // n lifts the ray off the surface; shadeN is what N.L is taken against - the
  // normal-mapped normal. Separate because the lift has to follow the real
  // geometry, and for a sprite that is not the direction it is shaded as facing.
  //
  // brdf: null, or a function of the light direction returning the specular
  // response per unit of light (see specularTSL) - then the same loop, with the
  // same visibility, also sums each light's highlight.
  // second: 1 at a SECONDARY hit (what a reflection or the view through
  // glass lands on) - there only lights in the hit-shadow budget march
  // (packLight's radius sign); the rest light it unshadowed.
  const body = (p, n, skip, shadeN, brdf = null, second = null) => {
    const sum = vec3(0, 0, 0).toVar();
    const spec = brdf ? vec3(0, 0, 0).toVar() : null;
    // Lifted off the face along the NORMAL, the same as the sun's origin and for
    // the same reason - see SURFACE_BIAS_VOXELS. The same for every light.
    const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();

    // Named: the marcher and the card loop inside both use the default `i`, and
    // an inlined inner loop's `i` would shadow this one's.
    Loop({ start: int(0), end: lights.count, type: 'int', condition: '<', name: 'li' },
         ({ li }) => {
      const base = li.mul(int(LIGHT_VEC4S));
      // Read into variables up front, for the same shadowing reason: an
      // element() left as an expression is re-emitted at each use, inside the
      // inner loops.
      const A = lights.data.element(base).toVar();
      const Braw = lights.data.element(base.add(int(1))).toVar();
      const B = lightRowBTSL(Braw).toVar();
      const C = lights.data.element(base.add(int(2))).toVar();
      const Dw = lights.data.element(base.add(int(3))).toVar();
      const level = A.w;

      const toLight = A.xyz.sub(p).toVar();
      // max() rather than a branch: a fragment exactly at the light would divide
      // by zero, and the answer there is arbitrary anyway.
      const dist = max(length(toLight), float(1e-4)).toVar();
      const dir = toLight.div(dist).toVar();

      // lights.js: the level IS the radius in blocks, and the falloff reaches
      // zero at exactly that radius, so the light ends where the level says and
      // the boundary has no ring.
      const radius = level.toVar();   // metres (lights.js lightRadiusMetres)
      const u = clamp(float(1).sub(dist.div(max(radius, float(1e-4)))),
                      float(0), float(1));
      // log2(1 + u), mirroring lights.js lightFalloff - exact at both ends and
      // far brighter than a square across the middle, which is what a torch
      // looks like.
      const falloff = log2(float(1).add(u)).mul(level.div(float(MAX_LIGHT_LEVEL))).toVar();
      // shadeN null: a point in the air (gpu/gi.js fog light), lit from every side.
      const ndotl = shadeN ? max(shadeN.dot(dir), float(0)) : float(1);

      // lights.js coneSlope(): the cone from this fragment to the emitter opens
      // to sourceRadius at the light, so its slope is sourceRadius / D. Unlike
      // the sun's, this is per fragment - which is the whole difference between
      // a light 150 million km away and one in the player's hand.
      const slope = B.w.div(dist).toVar();
      // Capped at the nearer of the light and its own radius: nothing past
      // either can take away light that is already zero there.
      const reach = min(dist, radius).toVar();

      // Only march where the light can arrive at all. Outside the radius, or on
      // a face turned away, the result is multiplied by zero - and without this
      // gate every such fragment paid a march of up to the full radius, per
      // light. This is also the per-fragment cull §7.1 describes: a light that
      // does not reach here costs its distance test and nothing more.
      //
      // The cutoff remap (lights.js cutAmount) moves the gate inward, so the
      // faint outer band of the light is not marched either, and stays
      // continuous at the new edge rather than leaving a ring.
      const cut = lights.cutoff;
      const amount = max(falloff.mul(ndotl).sub(cut), float(0))
                       .div(float(1).sub(cut)).toVar();
      If(amount.greaterThan(float(0)), () => {
        // C.z is the shadow weight: 0 for a light outside the shadow budget,
        // which lights the surface unshadowed and marches nothing; between 0
        // and 1 while it fades in or out of the budget.
        // At a secondary hit, times the hit-shadow weight (D.x) - faded in
        // and out of that budget.
        const weight = second ? C.z.mul(select(second.lessThan(float(0.5)), float(1), Dw.x)) : C.z;
        // A colour in a world with glass: the light tinted by what it crossed.
        const visible = (glass ? vec3(1, 1, 1) : float(1)).toVar();
        const marches = second
          ? weight.greaterThan(float(0)).and(second.lessThan(float(0.5))
              .or(second.lessThan(float(1.5)).and(Braw.w.greaterThanEqual(float(0)))))
          : weight.greaterThan(float(0));
        If(marches, () => {
          const traced = trace(origin, dir, reach, slope, fadeStart, edgeFade).toVar();
          const v = traced.x.toVar();
          // Sprites are not in the field, so they are tested here - against the
          // cards turned to face THIS light. Same ray, same cap, same cone slope,
          // so a sprite's penumbra matches a wall's.
          // C.w is the card weight: 0 for a light outside the card budget,
          // which tests an empty slice. Faded like the shadow weight.
          if (cardsFn) {
            const start = int(C.x).toVar();
            const cardVis = cardsFn(origin, dir, reach, slope, skip,
                                    start, start.add(int(C.y)));
            v.mulAssign(mix(float(1), cardVis, C.w));
          }
          if (glass) {
            visible.assign(mix(vec3(1, 1, 1),
                               glassRayTransmitTSL(glass, traced, origin, dir).mul(v), weight));
          }
          else visible.assign(mix(float(1), v, weight));
        });
        if (smoke) visible.mulAssign(smokeTransmitTSL(smoke, origin, dir, reach, smokeSoft));
        sum.addAssign(B.xyz.mul(amount).mul(visible));
        // The highlight takes the light's falloff, not the cut-remapped
        // diffuse amount - brdf carries its own N.L.
        if (brdf) spec.addAssign(B.xyz.mul(falloff).mul(visible).mul(brdf(dir)));
      });
    });

    return brdf ? { diffuse: sum, spec } : sum;
  };

  // A plain function around the Fn, not the Fn itself: a TSL Fn is a proxy,
  // and properties hung on it read back as node members, not as these.
  const fn = Fn(([p, n, skip, shadeN]) => body(p, n, skip, shadeN))
    .setLayout(layout('pointLights', 'vec3', [
      { name: 'p', type: 'vec3' }, { name: 'n', type: 'vec3' },
      { name: 'skip', type: 'vec2' }, { name: 'shadeN', type: 'vec3' }]));
  const call = (p, n, skip, shadeN) => fn(p, n, skip, shadeN);
  // Called inline from inside another Fn, so their vars land in the caller.
  call.withSpecular = (p, n, skip, shadeN, brdf = null, second = null) => body(p, n, skip, shadeN, brdf, second);
  return call;
}
