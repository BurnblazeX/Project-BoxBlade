import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, vec3, uniform, floor, max, texture, abs, vec2, clamp, smoothstep, length, uniformArray, int, select, mix, exp2 } from 'three/tsl';
import { VOXEL_METRES, CASCADE_COUNT } from '../core/field.js';
import { CARD_RANGE, CARD_PAD, CARD_FADE_START, CARD_SUN_REACH } from '../core/cards.js';
import { layout } from './layout.js';
import { decodeFieldTSL } from './field.js';

// ---------------------------------------------------------------------------
// Analytic sprite cards - the GPU port of cards.js
//
// Sprites are NOT in the distance field. The field is geometry shared by every
// light, so a card baked into it points one way and every other light gets the
// wrong silhouette; the fix is to keep sprites out of it and intersect each
// shadow ray with a card turned to face THAT light.
//
// This is the mirror of cardVisibility() in cards.js and has to stay one, the
// way traceDistanceTSL mirrors sphereTrace - bxb.cardparity() diffs them.
//
// Cheap for a reason worth being explicit about: a card is a PLANE, so this is
// one intersection per card per RAY. Marching a voxelised sprite costs a test
// per card per STEP, tens of times more, and buys nothing the plane does not
// already give exactly. Being analytic it also has no voxel size, so the
// cascade-scale seam a baked silhouette suffers cannot arise at all.
//
// Layout, four vec4 per card, packed by packCardInstances():
//   0  centre xyz, flip (-1 or +1)
//   1  facing ux, uz, half width, half height   (metres)
//   2  atlas rect x, y, w, h                    (pixels)
//   3  metres per pixel, cols, rows, cascade lod (see the LOD block below)
// ---------------------------------------------------------------------------

export const MAX_CARDS = 64;

// The sun's cards reach far past the march cap - see CARD_SUN_REACH.
export const CARD_SUN_MAX = CARD_SUN_REACH;

// name: the buffer's name in WGSL, unique per shader - see bufferName.
export function createCardBindings(capacity = MAX_CARDS, name = 'bxbCards') {
  return {
    capacity,
    // Four vec4 per card, flat. A uniform array rather than a storage buffer:
    // 64 cards is 4 KB, far inside the uniform limit, and a uniform read is the
    // cheaper of the two in a per-fragment loop.
    data: uniformArray(new Array(capacity * 4).fill(0).map(() => new THREE.Vector4()),
                       'vec4').setName(name),
    count: uniform(int(0)),
    // How many levels finer than its cascade this light draws a card - see the
    // LOD block in createCardsTSL. The sun uses 1, point lights 0.
    lodShift: uniform(float(0)),
    lodMax: uniform(float(CASCADE_COUNT - 1)),   // coarsest quality level
    atlas: null,
    atlasSize: uniform(new THREE.Vector2(1, 1))
  };
}

export function createCardAtlasTexture(atlas) {
  const tex = new THREE.DataTexture(atlas.data, atlas.width, atlas.height,
                                    THREE.RedFormat, THREE.UnsignedByteType);
  // LINEAR for the same reason the 3D field is: interpolating a DISTANCE is
  // exact, where interpolating occupancy would smear solid into empty. It is
  // also what lets one texel of silhouette resolve a smooth penumbra edge.
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

// Push a packed Float32Array into the uniform array.
export function writeCardBindings(b, packed, count) {
  const v = b.data.array;
  for (let i = 0; i < count * 4; i++) {
    v[i].set(packed[i * 4], packed[i * 4 + 1], packed[i * 4 + 2], packed[i * 4 + 3]);
  }
  b.count.value = count;
  return b;
}

// How much of one light the cards leave visible along a ray.
//
// Multiplied rather than min'd, matching cardsVisibility(): two sprites
// overlapping a ray each take their own bite, where min would let the nearer
// one hide the further.
//
// ranged: the Fn takes two more arguments, the first card and one past the last,
// for a buffer shared between lights where each owns a slice - see the point
// light loop. Unranged it walks the whole buffer, as the sun's does.
export function createCardsTSL(b, { ranged = false } = {}) {
  // No atlas means no cards in this pass at all, and the caller drops the
  // multiply rather than compiling a loop that can never run.
  if (!b || !b.atlas) return null;
  // skip is the XZ of the card to leave out: a sprite shading itself must not
  // be occluded by its own card. Terrain passes a point far off the map.
  const body = (from, dir, maxDist, coneSlope, skip, start, end) => {
    const vis = float(1).toVar();

    Loop({ start, end, type: 'int', condition: '<' }, ({ i }) => {
      const base = i.mul(int(4));
      const A = b.data.element(base);
      const B = b.data.element(base.add(int(1)));
      const C = b.data.element(base.add(int(2)));
      const D = b.data.element(base.add(int(3)));

      const centre = A.xyz;
      const flip = A.w;
      const ux = B.x, uz = B.y;
      // The card's normal is its width turned 90 degrees, same as the CPU side.
      const nx = uz.negate(), nz = ux;

      const denom = dir.x.mul(nx).add(dir.z.mul(nz));
      // Edge-on. A card built to face this light cannot be, but a stale facing
      // can, and a near-zero divisor throws the hit point to infinity.
      If(abs(denom).greaterThan(float(1e-6))
           .and(length(centre.xz.sub(skip)).greaterThan(float(0.05))), () => {
        const t = centre.x.sub(from.x).mul(nx)
                  .add(centre.z.sub(from.z).mul(nz)).div(denom);
        // Between the receiver and the light, or it is not an occluder. The
        // maxDist cap is what stops a sprite BEYOND a torch from shadowing it.
        If(t.greaterThan(float(0)).and(t.lessThan(maxDist)), () => {
          const h = from.add(dir.mul(t)).sub(centre);
          const a = h.x.mul(ux).add(h.z.mul(uz)).mul(flip);
          const bH = h.y;

          const mpp = D.x, cols = D.y, rows = D.z, lod = D.w;
          const rx = C.x, ry = C.y, rw = C.z, rh = C.w;
          const hw = B.z, hh = B.w;

          // EARLY OUT, exact. The card's plane is infinite, so nearly every
          // ray crosses it somewhere - usually metres off the card, where the
          // answer is "fully lit" - and every such crossing used to read the
          // silhouette texture anyway. That was the single largest cost in the
          // frame (1.7 ms of 3.7 with the torch on, measured). The distance
          // field is never less than the distance to the card's rectangle, less
          // the largest cascade snap below; past the cone's radius plus that,
          // the visibility is exactly 1. Skip it.
          const outside = length(vec2(max(abs(a).sub(hw), float(0)),
                                      max(abs(bH).sub(hh), float(0))));
          const maxSnap = float(VOXEL_METRES).mul(exp2(b.lodMax));
          If(outside.lessThanEqual(coneSlope.mul(t).add(maxSnap)), () => {

            // Signed distance to the silhouette at card-local (sa, sb), metres.
            const distAt = (sa, sb) => {
              // Card-local metres to atlas pixels, then to UV. CARD_PAD is the
              // margin the falloff grows into; past it the border value already
              // reads as fully clear.
              const pu = sa.div(mpp).add(cols.mul(float(0.5))).add(float(CARD_PAD));
              const pv = rows.mul(float(0.5)).sub(sb.div(mpp)).add(float(CARD_PAD));
              // Inset by half a texel on every side. Bilinear filtering samples the
              // four texels AROUND the coordinate, so clamping to the rect edge
              // still reaches one texel outside it - into the neighbouring card, or
              // into the unwritten rows below a short one. That bleed is what put
              // spurious solid bands across the ground.
              const su = clamp(pu, float(0.5), rw.sub(float(0.5)));
              const sv = clamp(pv, float(0.5), rh.sub(float(0.5)));
              const uvp = vec2(rx.add(su).div(b.atlasSize.x),
                               ry.add(sv).div(b.atlasSize.y));
              // Same decode as the field's, because it is the same encoding - one
              // distance format in the renderer, not two that can drift.
              const dPix = decodeFieldTSL(texture(b.atlas, uvp).r, float(CARD_RANGE));
              // The transform saturates at CARD_PAD pixels, so far from the card it
              // under-reports the distance and a wide cone reads shadow everywhere.
              // The silhouette sits inside its own rectangle, so that rectangle's
              // distance is a valid floor - exact far away, negative and ignored
              // inside. See cardVisibility for the same step and why it is needed.
              // Outside only: the rectangle distance is zero inside, and maxing
              // against zero there would erase the negative distances that are the
              // silhouette itself.
              const dRect = length(vec2(max(abs(sa).sub(hw), float(0)),
                                        max(abs(sb).sub(hh), float(0))));
              const dtM = dPix.mul(mpp);
              return select(dRect.greaterThan(float(0)), max(dtM, dRect), dtM);
            };

            // CASCADE LOD - the card's resolution follows the level it stands in,
            // as the terrain's does. lod is set per card on the CPU: its integer
            // part is the level, its fraction how far into the band toward the
            // next one. Past C0 the hit point is snapped to that level's voxel
            // grid, so a distant sprite's shadow is as coarse as the ground's
            // around it; across the band the two resolutions crossfade, and past
            // the outermost level the card fades out as terrain shadows do.
            const l0 = floor(lod), f = lod.sub(l0);
            // Quality level: the cascade minus this light's shift, capped at its
            // lodMax - see SUN_CARD_LOD_SHIFT in cards.js.
            const quantum = l => {
              const ql = clamp(l.sub(b.lodShift), float(0), b.lodMax);
              return select(ql.lessThan(float(0.5)), float(0), float(VOXEL_METRES).mul(exp2(ql)));
            };
            const snap = (v, half, q) => select(q.greaterThan(float(0)),
              v.add(half).div(q).floor().add(float(0.5)).mul(q).sub(half), v);
            const q0 = quantum(l0), q1 = quantum(l0.add(float(1)));
            const d = distAt(snap(a, hw, q0), snap(bH, hh, q0)).toVar();
            const lastLevel = float(CASCADE_COUNT - 1);
            // Only where the two resolutions actually differ - under the sun
            // the C0/C1 band is full resolution on both sides.
            If(f.greaterThan(float(0)).and(l0.lessThan(lastLevel)).and(q1.notEqual(q0)), () => {
              d.assign(mix(d, distAt(snap(a, hw, q1), snap(bH, hh, q1)), f));
            });
            const lodFade = select(l0.greaterThanEqual(lastLevel), float(1).sub(f), float(1));

            const r = coneSlope.mul(t);
            // Hard when the cone has no width, otherwise d / (R*t) - the very
            // expression createConeTraceSunTSL uses, so a card's penumbra and a
            // wall's come from one formula rather than two that look alike.
            const soft = clamp(d.div(max(r, float(1e-6))), float(0), float(1));
            const hard = select(d.lessThan(float(0)), float(0), float(1));
            const raw = select(r.greaterThan(float(0)), soft, hard);

            // FADE, do not cut - the mirror of cardVisibility's tail, and the same
            // reasoning as fadeWeightTSL. A card faces the light, so t is very
            // nearly the distance from receiver to caster: the maxDist cap lands as
            // a straight line across the ground and a tree's shadow simply stops
            // along it, which reads as a diagonal slice cut out of the shadow.
            const fade = float(1).sub(
              smoothstep(maxDist.mul(float(CARD_FADE_START)), maxDist, t));
            vis.mulAssign(float(1).sub(float(1).sub(raw).mul(fade).mul(lodFade)));
          });
        });
      });
    });

    return vis;
  };
  const inputs = [
    { name: 'rayFrom', type: 'vec3' }, { name: 'dir', type: 'vec3' },
    { name: 'maxDist', type: 'float' }, { name: 'coneSlope', type: 'float' },
    { name: 'skip', type: 'vec2' }];
  return ranged
    ? Fn(([rayFrom, dir, maxDist, coneSlope, skip, start, end]) =>
        body(rayFrom, dir, maxDist, coneSlope, skip, start, end))
      .setLayout(layout('cardsRanged', 'float', [...inputs, { name: 'start', type: 'int' }, { name: 'end', type: 'int' }]))
    : Fn(([rayFrom, dir, maxDist, coneSlope, skip]) =>
        body(rayFrom, dir, maxDist, coneSlope, skip, int(0), b.count))
      .setLayout(layout('cards', 'float', inputs));
}

// Terrain has no card of its own to skip.
export const NO_SKIP = vec2(1e9, 1e9);

// --- Sprites in reflections: the card hit test ---
//
// Sprites are not in the field, so a reflection ray cannot land on one by
// marching. It is tested against the VIEW cards instead - one per sprite,
// turned the way the drawn billboard faces (toward the camera's heading) - and
// where it crosses one inside the sprite's opaque pixels, that is a hit. The
// colour comes from the colour atlas (buildCardAtlas), addressed by the same
// rect as the silhouette; the pixel mapping mirrors cardPixelAt in cards.js.
//
// Nearest-neighbour and a hard alpha cut, like the sprite's own draw: a
// reflected sprite is the same pixel art, not a filtered copy of it.
export function createCardColourTexture(atlas) {
  const tex = new THREE.DataTexture(atlas.colour, atlas.width, atlas.height,
                                    THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

// Returns a function, called inline from the shading Fn:
//   (from, dir, maxDist) -> { col, t, skip }
// col the sprite's linear colour, t the distance along the ray (< 0 for no
// hit nearer than maxDist), skip the card's XZ - so the sprite's own lighting
// can leave its own card out of its shadow test.
export function createCardHitTSL(b, colourTex) {
  if (!b || !colourTex) return null;
  return (from, dir, maxDist) => {
    const best = maxDist.toVar();
    const t = float(-1).toVar();
    const col = vec3(0, 0, 0).toVar();
    const skip = vec2(1e9, 1e9).toVar();
    Loop({ start: int(0), end: b.count, type: 'int', condition: '<', name: 'vc' },
         ({ vc }) => {
      const base = vc.mul(int(4));
      const A = b.data.element(base).toVar();
      const B = b.data.element(base.add(int(1))).toVar();
      const C = b.data.element(base.add(int(2))).toVar();
      const D = b.data.element(base.add(int(3))).toVar();
      const ux = B.x, uz = B.y;
      const nx = uz.negate(), nz = ux;
      const denom = dir.x.mul(nx).add(dir.z.mul(nz));
      If(abs(denom).greaterThan(float(1e-6)), () => {
        const th = A.x.sub(from.x).mul(nx).add(A.z.sub(from.z).mul(nz)).div(denom).toVar();
        If(th.greaterThan(float(0)).and(th.lessThan(best)), () => {
          const h = from.add(dir.mul(th)).sub(A.xyz);
          const a = h.x.mul(ux).add(h.z.mul(uz)).mul(A.w);
          const mpp = D.x, cols = D.y, rows = D.z;
          const pc = floor(a.div(mpp).add(cols.mul(float(0.5))));
          const pr = floor(rows.mul(float(0.5)).sub(h.y.div(mpp)));
          If(pc.greaterThanEqual(float(0)).and(pc.lessThan(cols))
               .and(pr.greaterThanEqual(float(0))).and(pr.lessThan(rows)), () => {
            const st = vec2(C.x.add(float(CARD_PAD)).add(pc).add(float(0.5)).div(b.atlasSize.x),
                            C.y.add(float(CARD_PAD)).add(pr).add(float(0.5)).div(b.atlasSize.y));
            const c = texture(colourTex, st).level(int(0)).toVar();
            If(c.a.greaterThanEqual(float(0.5)), () => {
              best.assign(th);
              t.assign(th);
              col.assign(c.rgb);
              skip.assign(A.xz);
            });
          });
        });
      });
    });
    return { col, t, skip };
  };
}
