import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, vec3, uniform, instanceIndex, storage, floor, max, texture, abs, vec4, clamp, smoothstep, length, log2, uniformArray, int, select, mix, sqrt, exp2, pow, round, reflect, refract } from 'three/tsl';
import { VOXEL_METRES } from '../core/field.js';
import { MAX_LIGHT_LEVEL, LIGHT_VEC4S } from '../core/lights.js';
import { BLOCK_METRES } from '../core/world.js';
import { blockFaceTSL as boxFaceTSL, texelLockTSL } from './terrain.js';
import { MIRROR_VEC4S, MIRROR_SUN_BIT, MIRROR_CACHE_VEC4S } from '../core/mirrors.js';
import { layout } from './layout.js';
import { EDGE_FADE_VOXELS, SHADOW_FADE_START } from './field.js';
import { glassBackGlare } from './glass.js';
import { createConeTraceSunTSL } from './trace.js';
import { blockLayerTSL } from './volume.js';
import { boxBumpedNormalTSL, labNormalTSL } from './normals.js';
import { NO_SKIP, createCardsTSL } from './cards.js';
import { sunColourUniform } from './sky.js';
import { REFLECT_MAX_DISTANCE, REFLECT_ROUGH_END, REFLECT_ROUGH_START, SPEC_MIN_ALPHA, decodeSpecularTSL } from './specular.js';
import { lightRowBTSL } from './pointlights.js';

// --- Mirrors: reflected sunlight, gathered (mirrors.js) ---
//
// The light a polished surface throws onto other surfaces. mirrors.js has the
// idea: from a receiving texel, the sun mirrored in a mirror's plane is one
// direction, D = reflect(L, N); if a ray that way lands on the mirror, the
// receiver sees the sun in it. Then it is lit by:
//
//   sun colour x sun scale x max(N_recv . D, 0)       as the direct sun, from D
//   x F                  the mirror TEXEL's Fresnel at the sun's angle, from its
//                        own _s: F0 or metal, so a seam that is rough and
//                        dielectric throws almost nothing and the patch on the
//                        wall carries the plate's pattern
//   x smooth             1 over the smooth band, 0 past REFLECT_ROUGH_END - a
//                        rough surface scatters rather than mirrors
//   x vis (mirror->sun)  the sun's own cone trace and cards at the hit
//   x vis (recv->mirror) a cone trace along D, as wide as the sun's, stopped
//                        short of the mirror so its own face is not a blocker
//
// Two marches, and only for a texel whose ray actually lands on a mirror;
// every other texel pays one plane test per rectangle. The sun only: a point
// light's mirror image depends on which plane, so it needs its own pass.
export function createMirrorBindings(capacity) {
  // The mirror texel cache (mirrors.js): written by the cache pass, read by the
  // shading. One buffer, a writable and a read-only view.
  const cache = new THREE.StorageBufferAttribute(MIRROR_CACHE_VEC4S, 4);
  return {
    capacity,
    data: uniformArray(new Array(capacity * MIRROR_VEC4S).fill(0).map(() => new THREE.Vector4()), 'vec4')
      .setName('bxbMirrors'),
    count: uniform(int(0)),
    cacheWrite: storage(cache, 'vec4', MIRROR_CACHE_VEC4S).setName('bxbMirrorCacheW'),
    cacheRead: storage(cache, 'vec4', MIRROR_CACHE_VEC4S).toReadOnly().setName('bxbMirrorCache')
  };
}

export function writeMirrorBindings(b, packed, count) {
  for (let i = 0; i < count * MIRROR_VEC4S; i++) {
    b.data.array[i].set(packed[i * 4], packed[i * 4 + 1], packed[i * 4 + 2], packed[i * 4 + 3]);
  }
  b.count.value = count;
  return b;
}

// --- The mirror light, as a factory ---
//
// Built by the shading pass (createShadowColorNode) and by the GI's injection
// (gi.js), which lights each probe hit with it too - so light thrown by a mirror
// bounces from where it lands, as all light does. Everything it needs comes in:
// the sun's uniforms and its shadow (sunShadow(p, n, skip), a number or, in a
// world with glass, a colour), sunLight() the sun's scale in the caller's
// units, and the card sets (null for none). Returns
// mirrorLight(p, n, nS, lift, skip) -> vec3, or null when there is nothing to
// throw.
export function createMirrorLightTSL({ cascades, terrain, mirrors, lights, sun, sunShadow,
                                      sunLight, sunOn, bias, fadeStartUniform = null,
                                      edgeFadeUniform = null, cards = null, lightCards = null,
                                      cache = false, sunColoured = false }) {
  // cache: read each mirror texel's own terms from the mirror texel cache
  // (mirrors.js) rather than working them out per receiver; the returned
  // function then carries .cacheKernel, the pass that fills it, which must run
  // each frame before anything shaded with this. sunColoured: sunShadow
  // returns a colour (a world with glass), so the cache keeps three channels.
  // Reflected light from the mirrors - see createMirrorBindings. Needs the
  // terrain arrays, for each mirror texel's own reflectivity.
  //
  // The sun AND the light list. A point light's mirror image is exact for a
  // plane, and every rectangle carries its plane: the light at P, reflected in
  // it, sits at P with the plane's axis flipped across the plane. A receiver
  // sees the light in that mirror when the line to the image crosses the plane
  // inside the rectangle; the light then arrives from the image's direction,
  // attenuated over the WHOLE path (receiver -> mirror -> light, which is the
  // distance to the image), and is shadowed on both legs.
  const mirrorTrace = mirrors && terrain && (sunOn || lights)
    ? createConeTraceSunTSL(cascades) : null;
  const mirrorFade = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const mirrorEdge = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));
  // The point lights' sprite cards, for the mirror->light leg. Ranged, as in
  // the light loop: each light walks its own slice.
  const mirrorCards = createCardsTSL(lightCards, { ranged: true });
  // And the sun's, for the sun's receiver->mirror leg. Both card sets are
  // turned to face the REAL light, not its mirror image. For a floor that is
  // exact - the reflection flips height only, so the direction across the
  // ground is unchanged. For an upright mirror the silhouette is a little off
  // its angle; for a sprite-sized shadow that does not show.
  const mirrorSunCards = createCardsTSL(cards);

  // What one mirror texel sends toward the receiver, for light arriving at H
  // from direction Lh: its Fresnel (its own _s, at the bumped normal's angle
  // to the light) times smoothness times the normal-map lobe. back is the way
  // from H to the receiver; minWidth the lobe floor - the source's angular
  // size. vec3; zero for a rough or a tilted texel.
  //
  // The texel's OWN normal matters: the ray was aimed with the flat face
  // normal, and a bevel or a rivet tilted off it sends the light somewhere
  // else. So the lobe weighs how closely where this texel really sends it,
  // reflect(-Lh, Nb), matches `back`. Flat texels pass, tilted ones go dark -
  // the normal map shows in the patch. (Their light lands somewhere this ray
  // is not looking; it is not redistributed.)
  // The same, in parts, for the back-face reflection, which needs the texel's
  // F0 (for its index), albedo (its tint) and Fresnel separately.
  const mirrorTexelParts = (H, N, Lh, back, minWidth) => {
    const f = boxFaceTSL(H, N);
    const layer = blockLayerTSL(terrain, f.block, N, H);
    const s = texture(terrain.specular, f.st).depth(int(layer)).level(int(0));
    const alb = texture(terrain.albedo, f.st).depth(int(layer)).level(int(0)).rgb.toVar();
    const d = decodeSpecularTSL(s, alb);
    const smooth = float(1).sub(smoothstep(float(REFLECT_ROUGH_START),
                                           float(REFLECT_ROUGH_END), d.alpha)).toVar();
    const lab = labNormalTSL(texture(terrain.normal, f.st).depth(int(layer)).level(int(0)));
    const Nb = boxBumpedNormalTSL(N, lab.xyz).toVar();
    const lb = max(Lh.dot(Nb), float(0)).toVar();
    const Rb = Lh.negate().add(Nb.mul(lb.mul(float(2))));
    const width = max(d.alpha, max(minWidth, float(SPEC_MIN_ALPHA)));
    const aligned = exp2(Rb.dot(back).sub(float(1)).div(width.mul(width)).mul(float(1.4427))).toVar();
    const F = d.f0.add(vec3(1, 1, 1).sub(d.f0).mul(pow(float(1).sub(lb), float(5)))).toVar();
    return { F, smooth, aligned, alb, f0: d.f0 };
  };
  // In two parts: what the texel is (its Fresnel, smoothness, where it sends
  // the light and how wide) - the same for every receiver, so the cache pass
  // stores it - and the lobe toward one receiver, `back`.
  const mirrorTexelTerms = (H, N, Lh, minWidth) => {
    const f = boxFaceTSL(H, N);
    const layer = blockLayerTSL(terrain, f.block, N, H);
    const s = texture(terrain.specular, f.st).depth(int(layer)).level(int(0));
    const alb = texture(terrain.albedo, f.st).depth(int(layer)).level(int(0)).rgb;
    const d = decodeSpecularTSL(s, alb);
    const smooth = float(1).sub(smoothstep(float(REFLECT_ROUGH_START),
                                           float(REFLECT_ROUGH_END), d.alpha));
    const lab = labNormalTSL(texture(terrain.normal, f.st).depth(int(layer)).level(int(0)));
    const Nb = boxBumpedNormalTSL(N, lab.xyz).toVar();
    const lb = max(Lh.dot(Nb), float(0)).toVar();
    const Rb = Lh.negate().add(Nb.mul(lb.mul(float(2))));
    const width = max(d.alpha, max(minWidth, float(SPEC_MIN_ALPHA)));
    const F = d.f0.add(vec3(1, 1, 1).sub(d.f0).mul(pow(float(1).sub(lb), float(5))));
    return { F, smooth, Rb, width };
  };
  const mirrorLobe = ({ F, smooth, Rb, width }, back) => {
    const aligned = exp2(Rb.dot(back).sub(float(1)).div(width.mul(width)).mul(float(1.4427)));
    return F.mul(smooth.mul(aligned));
  };
  const mirrorTexel = (H, N, Lh, back, minWidth) =>
    mirrorLobe(mirrorTexelTerms(H, N, Lh, minWidth), back);

  // The cache's layout (mirrors.js): per mirror texel, `stride` vec4 - the
  // sun's lobe and its shadow at the texel, then per light its Fresnel and
  // smoothness, its lobe direction and width, and the mirror->light leg's
  // visibility and cards.
  const cacheStride = () => int(2).add(lights ? lights.count.mul(int(3)) : int(0));
  const sunVisOf = v => (sunColoured ? v.xyz : v.x);

  // skip: the receiving sprite's own card, so it does not block the light
  // reaching itself. Terrain passes NO_SKIP.
  const mirrorLightFn = mirrorTrace ? Fn(([p, nS, lift, skip]) => {
    const sum = vec3(0, 0, 0).toVar();
    const origin = p.add(lift.mul(bias.mul(float(VOXEL_METRES)))).toVar();
    Loop({ start: int(0), end: mirrors.count, type: 'int', condition: '<', name: 'mi' },
         ({ mi }) => {
      const row = mi.mul(int(MIRROR_VEC4S));
      const A = mirrors.data.element(row).toVar();
      const B = mirrors.data.element(row.add(int(1))).toVar();
      // What this mirror can reflect, culled on the CPU (mirrors.js
      // mirrorReach): A.w is a mask - bit i for light i, MIRROR_SUN_BIT for the
      // sun - and C, E the box its reflected sun can reach. Measured: the
      // mirror loop was 0.58 ms of 2.35 at 1080p before this, most of it
      // texels no mirror could ever light.
      const mask = A.w;
      const bit = k => floor(mask.div(exp2(k))).mod(float(2)).greaterThan(float(0.5));
      const C = mirrors.data.element(row.add(int(2)));
      const E = mirrors.data.element(row.add(int(3)));
      const axis = int(A.x);
      const isX = axis.equal(int(0)), isY = axis.equal(int(1));
      const N = select(isX, vec3(A.y, 0, 0), select(isY, vec3(0, A.y, 0), vec3(0, 0, A.y))).toVar();
      const pick = v => select(isX, v.x, select(isY, v.y, v.z));
      const pickU = v => select(isX, v.y, v.x);                   // mirrors.js PLANE_AXES
      const pickV = v => select(isX.or(isY), v.z, v.y);
      // A snapped mirror texel's slot in the cache: its row and column on the
      // rectangle, from the rectangle's cache base (E.w).
      const texelSlot = H => {
        const q = float(VOXEL_METRES);
        const nu = round(B.z.sub(B.x).div(q));
        const iu = floor(pickU(H).div(q)).sub(round(B.x.div(q)));
        const iv = floor(pickV(H).div(q)).sub(round(B.y.div(q)));
        return int(E.w.add(iv.mul(nu)).add(iu)).mul(cacheStride());
      };
      // The receiver must be on the mirror's reflecting side.
      const side = pick(origin).sub(A.z).mul(A.y);
      If(side.greaterThan(float(0)), () => {
        // Where a ray from the receiver along dir crosses the plane: the
        // distance, or -1 if it misses the rectangle.
        const crossing = dir => {
          const da = pick(dir);
          const t = A.z.sub(pick(origin)).div(select(abs(da).greaterThan(float(1e-5)), da,
                                                     float(1e-5))).toVar();
          const H = origin.add(dir.mul(t));
          const hu = pickU(H), hv = pickV(H);
          const inside = t.greaterThan(float(VOXEL_METRES))
            .and(hu.greaterThanEqual(B.x)).and(hu.lessThanEqual(B.z))
            .and(hv.greaterThanEqual(B.y)).and(hv.lessThanEqual(B.w));
          return select(inside, t, float(-1));
        };

        // The back-face reflection of a glass rectangle (C.w: its thickness),
        // for light reaching the receiver along -D through the front face at
        // distance t: traced in (refract), across the slab to the back face,
        // reflected, back across, and out - where it must still be inside the
        // rectangle. The same lobe and visibility as the front reflection,
        // which it sits beside; weight (1 - F)^2 F, tinted over the slab twice.
        const backGlare = (D, t, Lh, minWidth) => {
          const w = vec3(0, 0, 0).toVar();
          If(glassBackGlare.greaterThan(float(0.5)).and(C.w.greaterThan(float(0))), () => {
            const H1 = origin.add(D.mul(t)).toVar();
            const parts = mirrorTexelParts(texelLockTSL(H1, N), N, Lh, D.negate(), minWidth);
            const sf = sqrt(clamp(parts.f0.x, float(0), float(0.5)));
            const ior = float(1).add(sf).div(float(1).sub(sf));
            const t1 = refract(D, N, float(1).div(ior)).toVar();
            const cosIn = max(abs(t1.dot(N)), float(1e-3));
            const s = C.w.div(cosIn).toVar();
            const H3 = H1.add(t1.mul(s)).add(reflect(t1, N).mul(s)).toVar();
            const hu = pickU(H3), hv = pickV(H3);
            const inside = hu.greaterThanEqual(B.x).and(hu.lessThanEqual(B.z))
              .and(hv.greaterThanEqual(B.y)).and(hv.lessThanEqual(B.w));
            If(inside, () => {
              const T = float(1).sub(parts.F.x);
              const absorb = pow(max(parts.alb, vec3(1e-3, 1e-3, 1e-3)),
                                 vec3(s, s, s).mul(float(2 / BLOCK_METRES)));
              w.assign(parts.F.mul(T.mul(T)).mul(absorb).mul(parts.smooth.mul(parts.aligned)));
            });
          });
          return w;
        };

        // --- the sun, mirrored ---
        if (sunOn) {
          const L = sun.dir;
          const D = L.sub(N.mul(L.dot(N).mul(float(2)))).toVar();     // reflect(L, N)
          const facing = max(nS.dot(D), float(0)).toVar();
          const inBox = origin.x.greaterThanEqual(C.x).and(origin.y.greaterThanEqual(C.y))
            .and(origin.z.greaterThanEqual(C.z)).and(origin.x.lessThanEqual(E.x))
            .and(origin.y.lessThanEqual(E.y)).and(origin.z.lessThanEqual(E.z));
          If(bit(float(Math.log2(MIRROR_SUN_BIT))).and(inBox)
               .and(facing.greaterThan(float(0))).and(L.dot(N).greaterThan(float(0))), () => {
            const t = crossing(D).toVar();
            If(t.greaterThan(float(0)).and(t.lessThan(float(REFLECT_MAX_DISTANCE))), () => {
              // The mirror texel: snapped, like every surface point, so one
              // mirror texel reflects one value.
              const Hl = texelLockTSL(origin.add(D.mul(t)), N).toVar();
              const slot = cache ? texelSlot(Hl).toVar() : null;
              const w = (cache ? mirrors.cacheRead.element(slot).xyz
                               : mirrorTexel(Hl, N, L, D.negate(), sun.coneRadius.mul(float(2)))).toVar();
              If(w.x.add(w.y).add(w.z).greaterThan(float(1e-3)), () => {
                const toMirror = mirrorTrace(origin, D, t.sub(float(VOXEL_METRES * 2)),
                                             sun.coneRadius, mirrorFade, mirrorEdge).x.toVar();
                // Sprites between the receiver and the mirror - Bob standing on
                // the floor cuts his shadow into its glare on the wall.
                if (mirrorSunCards) {
                  toMirror.mulAssign(mirrorSunCards(origin, D, t, sun.coneRadius, skip));
                }
                const toSun = cache ? sunVisOf(mirrors.cacheRead.element(slot.add(int(1))))
                                    : sunShadow(Hl, N, NO_SKIP);
                const wb = backGlare(D, t, L, sun.coneRadius.mul(float(2)));
                sum.addAssign(w.add(wb).mul(toMirror.mul(toSun).mul(facing))
                               .mul(sunLight()).mul(sunColourUniform));
              });
            });
          });
        }

        // --- every point light, mirrored ---
        if (lights) {
          Loop({ start: int(0), end: lights.count, type: 'int', condition: '<', name: 'ml' },
               ({ ml }) => {
            const base = ml.mul(int(LIGHT_VEC4S));
            const LA = lights.data.element(base).toVar();
            const LB = lightRowBTSL(lights.data.element(base.add(int(1)))).toVar();
            const LC = lights.data.element(base.add(int(2))).toVar();
            const mirrorW = lights.data.element(base.add(int(3))).y.toVar();   // faded in / out of the budget
            const P = LA.xyz, level = LA.w;
            // In the mask: on the reflecting side and within reach of it.
            If(bit(float(ml)), () => {
              // Its image: the plane's axis flipped across the plane. N carries
              // the rectangle's sign; times the sign again it is the bare axis.
              const image = P.add(N.mul(A.y).mul(A.z.sub(pick(P)).mul(float(2)))).toVar();
              const toImage = image.sub(origin).toVar();
              const dist = max(length(toImage), float(1e-4)).toVar();
              const D = toImage.div(dist).toVar();
              // The falloff over the whole path, as the light loop's (lights.js
              // lightFalloff): log2(1 + u), reaching zero at the radius.
              const radius = level.toVar();   // metres (lights.js lightRadiusMetres)
              const u = clamp(float(1).sub(dist.div(max(radius, float(1e-4)))), float(0), float(1));
              const falloff = log2(float(1).add(u)).mul(level.div(float(MAX_LIGHT_LEVEL))).toVar();
              const facing = max(nS.dot(D), float(0)).toVar();
              If(falloff.mul(facing).greaterThan(lights.cutoff), () => {
                const t = crossing(D).toVar();
                If(t.greaterThan(float(0)).and(t.lessThan(dist)), () => {
                  const Hl = texelLockTSL(origin.add(D.mul(t)), N).toVar();
                  const toLight = P.sub(Hl).toVar();
                  const legB = max(length(toLight), float(1e-4)).toVar();
                  const Lh = toLight.div(legB).toVar();
                  // The flame's angular size from the mirror is the lobe's floor.
                  const at = cache ? texelSlot(Hl).add(int(2)).add(ml.mul(int(3))).toVar() : null;
                  const c0 = cache ? mirrors.cacheRead.element(at).toVar() : null;
                  const c1 = cache ? mirrors.cacheRead.element(at.add(int(1))).toVar() : null;
                  const c2 = cache ? mirrors.cacheRead.element(at.add(int(2))).toVar() : null;
                  const w = (cache
                    ? mirrorLobe({ F: c0.xyz, smooth: c0.w, Rb: c1.xyz, width: c1.w }, D.negate())
                    : mirrorTexel(Hl, N, Lh, D.negate(), LB.w.div(legB))).toVar();
                  If(w.x.add(w.y).add(w.z).greaterThan(float(1e-3)), () => {
                    // Both legs shadowed, as the light loop shadows - the same
                    // budget weight (LC.z) and sprite cards (LC.x, LC.y, LC.w).
                    // The cone opens to the source over the whole path.
                    const slope = LB.w.div(dist);
                    const visible = float(1).toVar();
                    If(LC.z.greaterThan(float(0)), () => {
                      const v = mirrorTrace(origin, D, t.sub(float(VOXEL_METRES * 2)), slope,
                                            mirrorFade, mirrorEdge).x.toVar();
                      const lift2 = Hl.add(N.mul(bias.mul(float(VOXEL_METRES)))).toVar();
                      v.mulAssign(cache ? c2.x : mirrorTrace(lift2, Lh, legB, LB.w.div(legB),
                                                            mirrorFade, mirrorEdge).x);
                      if (mirrorCards) {
                        // Both legs: receiver->mirror (leaving the receiver's
                        // own card out) and mirror->light.
                        const start = int(LC.x).toVar();
                        const end = start.add(int(LC.y)).toVar();
                        const cardVis = mirrorCards(origin, D, t, slope, skip, start, end)
                          .mul(cache ? c2.y
                                     : mirrorCards(lift2, Lh, legB, LB.w.div(legB), NO_SKIP, start, end));
                        v.mulAssign(mix(float(1), cardVis, LC.w));
                      }
                      visible.assign(mix(float(1), v, LC.z));
                    });
                    const wb = backGlare(D, t, Lh, LB.w.div(legB));
                    sum.addAssign(w.add(wb).mul(LB.xyz).mul(falloff.mul(facing).mul(visible).mul(mirrorW)));
                  });
                });
              });
            });
          });
        }
      });
    });
    return sum;
  }).setLayout(layout('mirrorLight', 'vec3', [
    { name: 'p', type: 'vec3' }, { name: 'nS', type: 'vec3' }, { name: 'lift', type: 'vec3' },
    { name: 'skip', type: 'vec2' }]))
    : null;
  if (!mirrorLightFn) return null;
  const light = (p, n, nS, lift, skip = NO_SKIP) => mirrorLightFn(p, nS, lift, skip);
  if (cache) light.cacheKernel = createMirrorCacheKernel();
  return light;

  // The cache pass: one invocation per mirror texel of the packed rectangles
  // (count set each frame to mirrors.js packMirrors' cache.texels). Each works
  // out, at its texel's snapped centre - the very point a receiver's
  // texelLockTSL lands on, built by the same arithmetic - everything the
  // mirror loop above reads from the cache, with the same calls. Traces only
  // where a receiver would trace: behind the same gates, the lobe taken at its
  // largest (aligned <= 1), so no receiver ever reads a slot left unfilled.
  function createMirrorCacheKernel() {
    const out = mirrors.cacheWrite;
    return Fn(() => {
      const idx = float(instanceIndex).toVar();
      // Which rectangle: bases rise in packing order, so the last one at or
      // below this index.
      const r = int(0).toVar();
      Loop({ start: int(1), end: mirrors.count, type: 'int', condition: '<', name: 'mc' }, ({ mc }) => {
        If(mirrors.data.element(mc.mul(int(MIRROR_VEC4S)).add(int(3))).w.lessThanEqual(idx), () => {
          r.assign(mc);
        });
      });
      const row = r.mul(int(MIRROR_VEC4S));
      const A = mirrors.data.element(row).toVar();
      const B = mirrors.data.element(row.add(int(1))).toVar();
      const E = mirrors.data.element(row.add(int(3))).toVar();
      const mask = A.w;
      const bit = k => floor(mask.div(exp2(k))).mod(float(2)).greaterThan(float(0.5));
      const axis = int(A.x);
      const isX = axis.equal(int(0)), isY = axis.equal(int(1));
      const N = select(isX, vec3(A.y, 0, 0), select(isY, vec3(0, A.y, 0), vec3(0, 0, A.y))).toVar();
      // The texel's snapped centre, as texelLockTSL builds it: (k + 0.5) q
      // along the face, round(plane / q) q across it.
      const q = float(VOXEL_METRES);
      const nu = round(B.z.sub(B.x).div(q)).toVar();
      const local = idx.sub(E.w).toVar();
      const iv = floor(local.div(nu)).toVar();
      const iu = local.sub(iv.mul(nu));
      const hu = round(B.x.div(q)).add(iu).add(float(0.5)).mul(q);
      const hv = round(B.y.div(q)).add(iv).add(float(0.5)).mul(q);
      const hp = round(A.z.div(q)).mul(q);
      const Hl = vec3(select(isX, hp, hu),
                      select(isX, hu, select(isY, hp, hv)),
                      select(isX.or(isY), hv, hp)).toVar();
      const slot = int(instanceIndex).mul(cacheStride()).toVar();

      const wS = vec3(0, 0, 0).toVar();
      const vS = vec3(0, 0, 0).toVar();
      if (sunOn) {
        const L = sun.dir;
        const D = L.sub(N.mul(L.dot(N).mul(float(2)))).toVar();
        If(bit(float(Math.log2(MIRROR_SUN_BIT))).and(L.dot(N).greaterThan(float(0))), () => {
          wS.assign(mirrorTexel(Hl, N, L, D.negate(), sun.coneRadius.mul(float(2))));
          If(wS.x.add(wS.y).add(wS.z).greaterThan(float(1e-3)), () => {
            const v = sunShadow(Hl, N, NO_SKIP);
            vS.assign(sunColoured ? v : vec3(v, v, v));
          });
        });
      }
      out.element(slot).assign(vec4(wS, 0));
      out.element(slot.add(int(1))).assign(vec4(vS, 0));

      if (lights) {
        Loop({ start: int(0), end: lights.count, type: 'int', condition: '<', name: 'ml' }, ({ ml }) => {
          const base = ml.mul(int(LIGHT_VEC4S));
          const LA = lights.data.element(base).toVar();
          const LB = lightRowBTSL(lights.data.element(base.add(int(1)))).toVar();
          const LC = lights.data.element(base.add(int(2))).toVar();
          If(bit(float(ml)), () => {
            const toLight = LA.xyz.sub(Hl).toVar();
            const legB = max(length(toLight), float(1e-4)).toVar();
            const Lh = toLight.div(legB).toVar();
            const terms = mirrorTexelTerms(Hl, N, Lh, LB.w.div(legB));
            const F = terms.F.toVar(), smooth = terms.smooth.toVar();
            const vis = float(1).toVar(), cardVis = float(1).toVar();
            const Fs = F.mul(smooth);
            If(Fs.x.add(Fs.y).add(Fs.z).greaterThan(float(1e-3)).and(LC.z.greaterThan(float(0))), () => {
              const lift2 = Hl.add(N.mul(bias.mul(float(VOXEL_METRES)))).toVar();
              vis.assign(mirrorTrace(lift2, Lh, legB, LB.w.div(legB), mirrorFade, mirrorEdge).x);
              if (mirrorCards) {
                const start = int(LC.x).toVar();
                const end = start.add(int(LC.y)).toVar();
                cardVis.assign(mirrorCards(lift2, Lh, legB, LB.w.div(legB), NO_SKIP, start, end));
              }
            });
            const at = slot.add(int(2)).add(ml.mul(int(3)));
            out.element(at).assign(vec4(F, smooth));
            out.element(at.add(int(1))).assign(vec4(terms.Rb, terms.width));
            out.element(at.add(int(2))).assign(vec4(vis, cardVis, 0, 0));
          });
        });
      }
    })().compute(1);
  }
}
