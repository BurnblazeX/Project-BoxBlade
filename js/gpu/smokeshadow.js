// --- The smoke as the whole renderer sees it: shadows and bounce through cloud spells ---
//
// smokeShared   one binding (a module singleton, as volume.js hitBounds): per
//               cloud slot (MAX_SMOKE_CLOUDS), which BoxGrid boxes are smoke
//               (SMOKE_BOX^3, stacked along z), and the slot's box corner,
//               spell colour and density, cast time and roll-out. The boxes
//               are written once, when a cloud is cast - each holding how far
//               the smoke travelled to reach it (core/smoke.js) - and how far
//               it has grown is worked out where it is read (smokeGrownAtTSL:
//               a box rolls in by its travel, as its cube does), so a cloud's
//               shadow grows with it, round corners as the smoke flows, and
//               no frame rewrites it.
//               Never a shader rebuild. Two copies of the boxes:
//                 hard  as they are, read nearest - the world's shadows, so a
//                       cube's shadow is a cube's;
//                 soft  blurred over each box's neighbours ([1 2 1] / 4 a
//                       side), read filtered - how the smoke shades itself and
//                       holds back the bounce: light falls off smoothly
//                       through a cloud instead of box by box.
// smokeTauTSL   the optical depth along a ray through every live cloud. The
//               shading multiplies its sun march and every light by
//               exp(-tau x strength) (gpu/shade.js, gpu/pointlights.js); the
//               cubes light themselves through the soft copy (gpu/smoke.js).
// smokeSampleTSL  the smoke at a point - extinction, and its colour by it: the
//               LPV's cells lose the bounce through it and the lit smoke in
//               them gives its own back (gpu/gi.js).
// smokeInscatterTSL  a ray through smoke: what it holds back, and the lit
//               smoke's light it gathers on the way - texel GI's rays
//               (gpu/texelgi.js), so a sunlit cloud brightens what is round it
//               (and one in shadow does not).
// Its own module so the light loop (pointlights.js) and the cubes (smoke.js),
// which use each other, both reach it.
import * as THREE from 'three/webgpu';
import { tableGroup, frameValues } from './uploads.js';
import { Fn, If, Loop, float, vec3, vec4, int, uniform, uniformArray, texture3D, exp, max, min, clamp,
         smoothstep, select } from 'three/tsl';
import { SMOKE_CUBE, SMOKE_BOX, SMOKE_GROW_IN, SMOKE_SUN_BOXES } from '../core/smoke.js';
import { layout } from './layout.js';
import { sunMapFarTSL } from './sunmap.js';
import { sunRoofTSL } from './sun.js';
import { skyOpenTSL } from './volume.js';

// Cloud spells live at once. Each is a slot of the atlas, and every shadow
// ray and light checks each slot's box: one more slot is one more box test a
// ray, wherever the clouds are.
export const MAX_SMOKE_CLOUDS = 8;

// Rows of the slot table a slot (see createSmokeShared).
const ROWS = 6;

export const SMOKE_REACH = 36;      // metres a shadow ray looks for smoke: C1

// Boxes along a ray through one cloud's box, at most: its diagonal.
const STEPS = Math.ceil(SMOKE_BOX * Math.sqrt(3));

// The binding the renderer shares (see above).
function createSmokeShared() {
  const N = MAX_SMOKE_CLOUDS, O = SMOKE_BOX, SLOT = O * O * O;
  const texture = (data, filter) => {
    const t = new THREE.Data3DTexture(data, O, O, O * N);
    t.format = THREE.RedFormat;
    t.type = THREE.UnsignedByteType;
    t.minFilter = t.magFilter = filter;
    t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
    t.unpackAlignment = 1;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    // Only the z layers a cast touched go up (WebGPU uploads a 3D texture a
    // layer a writeTexture - all 512 of them was the hitch on every cast).
    t.layerUpdates = new Set();
    t.clearLayerUpdates = () => t.layerUpdates.clear();
    return t;
  };
  const occData = new Uint8Array(SLOT * N), softData = new Uint8Array(SLOT * N);
  const occTex = texture(occData, THREE.NearestFilter), softTex = texture(softData, THREE.LinearFilter);
  const at = (x, y, z) => x + O * (y + O * z);
  const a = new Float32Array(SLOT), b = new Float32Array(SLOT);
  // Per slot, the z layers (its own, 0..O-1) it last filled, and the box
  // [lo, hi] it wrote: cleared on the next write. Only that box - a cloud
  // reshapes as bodies walk through it (core/smoke.js), a write each time,
  // and clearing a whole 64^3 slot three times over was most of one.
  const spans = new Array(N).fill(null), boxes = new Array(N).fill(null);
  const clearBox = (arr, off, lo, hi) => {
    for (let z = lo[2]; z <= hi[2]; z++) for (let y = lo[1]; y <= hi[1]; y++) {
      const r = off + at(lo[0], y, z);
      arr.fill(0, r, r + hi[0] - lo[0] + 1);
    }
  };
  // [1 2 1] / 4 along axis d, src -> dst, over [lo, hi] (the cloud's boxes and
  // a margin); a neighbour past the slot's edge is left out.
  const STRIDE = [1, O, O * O];
  const blurAxis = (src, dst, d, lo, hi) => {
    const s = STRIDE[d];
    for (let z = lo[2]; z <= hi[2]; z++) for (let y = lo[1]; y <= hi[1]; y++) for (let x = lo[0]; x <= hi[0]; x++) {
      const i = at(x, y, z), v = d === 0 ? x : d === 1 ? y : z;
      let sum = 2 * src[i], w = 2;
      if (v > 0) { sum += src[i - s]; w++; }
      if (v < O - 1) { sum += src[i + s]; w++; }
      dst[i] = sum / w;
    }
  };
  // One uniform array for all of it, four vec4 a slot - each uniformArray
  // is a uniform buffer, and a stage may bind only 12: four arrays took the
  // LPV's inject past it (an invalid pipeline).
  //   0  its box's min corner (world metres), 1 while live
  //   1  its colour (rgb) and density (/m)
  //   2  its emitter (world metres) and the second it was cast
  //   3  its radii across and up (metres), the farthest travel it took, the
  //      seconds it takes to roll out
  //   4  the min corner of what its smoke fills (world metres, a box of margin
  //      for the soft copy's blur), 5 its max - every ray is clipped to these,
  //      not to the slot's whole box: that is sized for a 9th-level cast (32 m),
  //      and marching all of it from everything near a small cloud doubled
  //      the frame
  const slots = uniformArray(new Array(N * ROWS).fill(0).map(() => new THREE.Vector4()), 'vec4').setName('bxbSmokeSlots').setGroup(tableGroup);
  return {
    occData, occTex, softData, softTex, slots,
    // Slot k's row r (above), as a node.
    row: (k, r) => slots.element(int(k).mul(int(ROWS)).add(int(r))),
    // How many slots are live: 0, and every smoke test is one branch.
    live: uniform(int(0)).setName('bxbSmokeLive'),
    // Seconds, on the same clock as the cast times (render/features/smoke.js).
    time: frameValues.smokeTime,
    // How dark the smoke is to light passing through it: a scale on the depth.
    strength: uniform(float(1)).setName('bxbSmokeStrength'),
    // Slot k: the cloud's boxes (core/smoke.js), its spell and its growth,
    // cast at `castTime`; null clears it.
    write(k, cloud, castTime = 0) {
      const ob = k * SLOT;
      const old = spans[k];
      if (boxes[k]) { clearBox(occData, ob, ...boxes[k]); clearBox(softData, ob, ...boxes[k]); }
      spans[k] = boxes[k] = null;
      if (cloud) {
        const lo = [O, O, O], hi = [-1, -1, -1];
        const inSlot = c => c[4] >= cloud.box[0] && c[5] >= cloud.box[1] && c[6] >= cloud.box[2] &&
          c[4] < cloud.box[0] + O && c[5] < cloud.box[1] + O && c[6] < cloud.box[2] + O;
        for (const c of cloud.cubes) {
          if (!inSlot(c)) continue;
          for (let d = 0; d < 3; d++) {
            const l = c[4 + d] - cloud.box[d];
            if (l < lo[d]) lo[d] = l;
            if (l > hi[d]) hi[d] = l;
          }
        }
        if (hi[0] >= 0) {
          // The blur only where the cloud is, and a box round it; it reads one
          // more each way, so a and b are clear over that too.
          const blo = lo.map(v => Math.max(0, v - 2)), bhi = hi.map(v => Math.min(O - 1, v + 2));
          const clo = blo.map(v => Math.max(0, v - 1)), chi = bhi.map(v => Math.min(O - 1, v + 1));
          clearBox(a, 0, clo, chi);
          clearBox(b, 0, clo, chi);
          for (const c of cloud.cubes) {
            if (!inSlot(c)) continue;
            const i = at(c[4] - cloud.box[0], c[5] - cloud.box[1], c[6] - cloud.box[2]);
            a[i] = 1;
            // Hard: its travel as a share of the farthest, 1..255 (0 is no smoke).
            occData[ob + i] = 1 + Math.round(254 * Math.min(1, c[3] / Math.max(cloud.gMax, 1e-6)));
          }
          // x, y, z in turn; the result in b.
          blurAxis(a, b, 0, blo, bhi); blurAxis(b, a, 1, blo, bhi); blurAxis(a, b, 2, blo, bhi);
          for (let z = blo[2]; z <= bhi[2]; z++) for (let y = blo[1]; y <= bhi[1]; y++) for (let x = blo[0]; x <= bhi[0]; x++) {
            const i = at(x, y, z);
            softData[ob + i] = Math.round(255 * b[i]);
          }
          spans[k] = [blo[2], bhi[2]];
          boxes[k] = [blo, bhi];
        }
        const { box, centre, spell, radius } = cloud;
        const r = slots.array;
        const R = ROWS * k;
        r[R].set((box[0] - 0.5) * SMOKE_CUBE, (box[1] - 0.5) * SMOKE_CUBE, (box[2] - 0.5) * SMOKE_CUBE, 1);
        r[R + 1].set(spell.colour[0], spell.colour[1], spell.colour[2], spell.density);
        r[R + 2].set(centre[0], centre[1], centre[2], castTime);
        r[R + 3].set(radius * SMOKE_CUBE, (radius + 0.5) * SMOKE_CUBE, cloud.gMax, cloud.growSeconds);
        // What it fills: its boxes, and one more each way for the blur.
        const corner = d => (box[d] + lo[d] - 1.5) * SMOKE_CUBE, far = d => (box[d] + hi[d] + 1.5) * SMOKE_CUBE;
        r[R + 4].set(corner(0), corner(1), corner(2), 0);
        r[R + 5].set(far(0), far(1), far(2), 0);
      } else {
        slots.array[ROWS * k].set(0, 0, 0, 0);
      }
      let dirty = false;
      for (const span of [old, spans[k]]) {
        if (!span) continue;
        for (let z = span[0]; z <= span[1]; z++) { occTex.layerUpdates.add(k * O + z); softTex.layerUpdates.add(k * O + z); }
        dirty = true;
      }
      if (dirty) { occTex.needsUpdate = true; softTex.needsUpdate = true; }
      let live = 0;
      for (let j = 0; j < N; j++) if (slots.array[ROWS * j].w > 0.5) live++;
      this.live.value = live;
    }
  };
}

export const smokeShared = createSmokeShared();

// How grown slot k's smoke is (0..1) where the smoke travelled `share` of
// its farthest to get: it rolls in over the slot's roll-out time by travel,
// each box over SMOKE_GROW_IN - as its cube grows in.
export function smokeGrowthTSL(shared, k, share) {
  const appear = share.mul(shared.row(k, 3).w);
  return smoothstep(appear, appear.add(float(SMOKE_GROW_IN)), shared.time.sub(shared.row(k, 2).w));
}

// How grown slot k's smoke is at p (uvw its atlas place): by the travel its
// hard box holds - and where the soft copy reaches past the boxes (its blur),
// once the cloud is fully out.
const grownAt = (shared, k, uvw) => {
  const r = texture3D(shared.occTex, uvw).level(float(0)).r;
  return smokeGrowthTSL(shared, k, select(r.greaterThan(float(0.002)), r.mul(float(255)).sub(float(1)).div(float(254)), float(1)));
};

// Where p falls in slot k's part of the atlas (z kept half a box inside it,
// so the filter never reads the next cloud).
const atlasUVW = (m, k, p) => {
  const O = SMOKE_BOX, SPAN = O * SMOKE_CUBE;
  const l = p.sub(m.xyz).div(float(SPAN)).toVar();
  const z = clamp(l.z, float(0.5 / O), float(1 - 0.5 / O));
  return vec3(clamp(l.x, float(0), float(1)), clamp(l.y, float(0), float(1)),
              float(k).add(z).div(float(MAX_SMOKE_CLOUDS)));
};

// What slot k's smoke fills, against the ray: { t0, t1 } (t1 <= t0: missed).
const boxSpan = (shared, k, origin, dir, reach) => {
  const inv = vec3(1, 1, 1).div(dir);
  const a = shared.row(k, 4).xyz.sub(origin).mul(inv), b = shared.row(k, 5).xyz.sub(origin).mul(inv);
  const lo = min(a, b), hi = max(a, b);
  return { t0: max(max(max(lo.x, lo.y), lo.z), float(0)).toVar(), t1: min(min(min(hi.x, hi.y), hi.z), reach).toVar() };
};

// Fn([origin, dir, reach]) -> the optical depth through every live cloud
// along the ray (unscaled by strength); soft: through the blurred copy. A
// layout function: the shading calls it at every sun march and every light.
const tauFns = new WeakMap();
export function smokeTauTSL(shared = smokeShared, soft = false, boxes = false) {
  if (!tauFns.has(shared)) tauFns.set(shared, {});
  const memo = tauFns.get(shared), key = boxes ? 'boxes' : soft ? 'soft' : 'hard';
  if (memo[key]) return memo[key];
  const step = SMOKE_CUBE, tex = soft ? shared.softTex : shared.occTex;
  const fn = Fn(([origin, dir, reach]) => {
    const tau = float(0).toVar();
    If(shared.live.greaterThan(int(0)), () => Loop({ start: int(0), end: int(MAX_SMOKE_CLOUDS), type: 'int', condition: '<', name: 'smk' }, ({ smk }) => {
      const m = shared.row(smk, 0).toVar();
      If(m.w.greaterThan(float(0.5)), () => {
        const { t0, t1 } = boxSpan(shared, smk, origin, dir, reach);
        If(t1.greaterThan(t0), () => {
          const density = boxes ? float(1 / SMOKE_CUBE) : shared.row(smk, 1).w.toVar();
          const t = t0.add(float(step * 0.5)).toVar();
          Loop({ start: int(0), end: int(STEPS), type: 'int', condition: '<', name: 'sms' }, () => {
            If(t.lessThan(t1), () => {
              const uvw = atlasUVW(m, smk, origin.add(dir.mul(t))).toVar();
              const raw = texture3D(tex, uvw).level(float(0)).r;
              // Hard: any travel is a whole box of smoke.
              const occ = soft ? raw : select(raw.greaterThan(float(0.002)), float(1), float(0));
              tau.addAssign(occ.mul(grownAt(shared, smk, uvw)).mul(density).mul(float(step)));
              t.addAssign(float(step));
            });
          });
        });
      });
    }));
    return tau;
  }).setLayout(layout(boxes ? 'smokeBoxes' : soft ? 'smokeTauSoft' : 'smokeTau', 'float', [
    { name: 'origin', type: 'vec3' }, { name: 'dir', type: 'vec3' }, { name: 'reach', type: 'float' }]));
  memo[key] = fn;
  return fn;
}

// The smoke's transmittance along a ray, as the shading takes it: exp(-tau x
// strength). reach: how far to look (metres); soft: the blurred copy.
export const smokeTransmitTSL = (shared, origin, dir, reach = float(SMOKE_REACH), soft = false) =>
  exp(smokeTauTSL(shared, soft)(origin, dir, reach).mul(shared.strength).negate());

// Thickness in half-metre box equivalents, independent of spell density.
// Eight boxes block the sun fully, like an opaque occluder; ambient remains.
export const smokeSunTransmitTSL = (shared, origin, dir) =>
  float(1).sub(clamp(smokeTauTSL(shared, false, true)(origin, dir, float(SMOKE_REACH)).div(float(SMOKE_SUN_BOXES)), float(0), float(1)));

// The smoke at p, every live cloud's (the soft copy, grown): vec4(colour x
// extinction, extinction) - extinction /m, times strength. Inline: one call
// site a kernel.
export function smokeSampleTSL(shared, p) {
  const out = vec4(0, 0, 0, 0).toVar();
  If(shared.live.greaterThan(int(0)), () => Loop({ start: int(0), end: int(MAX_SMOKE_CLOUDS), type: 'int', condition: '<', name: 'smp' }, ({ smp }) => {
    const m = shared.row(smp, 0).toVar();
    const lo = shared.row(smp, 4).xyz, hi = shared.row(smp, 5).xyz;
    const inside = m.w.greaterThan(float(0.5)).and(p.x.greaterThanEqual(lo.x)).and(p.y.greaterThanEqual(lo.y))
      .and(p.z.greaterThanEqual(lo.z)).and(p.x.lessThan(hi.x)).and(p.y.lessThan(hi.y)).and(p.z.lessThan(hi.z));
    If(inside, () => {
      const par = shared.row(smp, 1);
      const uvw = atlasUVW(m, smp, p).toVar();
      const sigma = texture3D(shared.softTex, uvw).level(float(0)).r
        .mul(grownAt(shared, smp, uvw)).mul(par.w).mul(shared.strength).toVar();
      out.addAssign(vec4(par.xyz.mul(sigma), sigma));
    });
  }));
  return out;
}

// Fn([origin, dir, reach, sunDir, sunLight, skyLight]) -> vec4(the light the
// lit smoke along the ray scatters into it, the optical depth x strength).
// Per cloud the ray crosses: its depth through the cloud (soft), the light
// the smoke there is lit by - the sun through the rest of the smoke from
// where it is thickest along the ray, and the sky - times its colour, and
// what it gives the ray: that light times (1 - its transmittance), dimmed by
// the clouds before it. Smoke in the world's shadow is not lit: at that point
// the sun takes the sun depth map (one hard tap) and the roof map toward the
// sun, and the sky the open-sky map (terrain: the terrain textures; null, no
// roof or sky test). Without them a cloud indoors, or in a building's shadow,
// lit what was round it as if it stood in the open sun (2026-10-30).
const inFns = new WeakMap();
export function smokeInscatterTSL(shared = smokeShared, terrain = null) {
  if (!inFns.has(shared)) inFns.set(shared, new Map());
  const cache = inFns.get(shared);
  if (cache.has(terrain)) return cache.get(terrain);
  const step = SMOKE_CUBE;
  const fn = Fn(([origin, dir, reach, sunDir, sunLight, skyLight]) => {
    const gathered = vec3(0, 0, 0).toVar(), tauAll = float(0).toVar();
    If(shared.live.greaterThan(int(0)), () => Loop({ start: int(0), end: int(MAX_SMOKE_CLOUDS), type: 'int', condition: '<', name: 'sik' }, ({ sik: k }) => {
      const m = shared.row(k, 0).toVar();
      If(m.w.greaterThan(float(0.5)), () => {
        const { t0, t1 } = boxSpan(shared, k, origin, dir, reach);
        If(t1.greaterThan(t0), () => {
          const par = shared.row(k, 1).toVar();
          const tau = float(0).toVar(), tw = float(0).toVar();
          const t = t0.add(float(step * 0.5)).toVar();
          Loop({ start: int(0), end: int(STEPS), type: 'int', condition: '<', name: 'sis' }, () => {
            If(t.lessThan(t1), () => {
              const uvw = atlasUVW(m, k, origin.add(dir.mul(t))).toVar();
              const s = texture3D(shared.softTex, uvw).level(float(0)).r
                .mul(grownAt(shared, k, uvw)).mul(par.w).mul(float(step)).toVar();
              tau.addAssign(s);
              tw.addAssign(s.mul(t));
              t.addAssign(float(step));
            });
          });
          If(tau.greaterThan(float(1e-4)), () => {
            const mid = origin.add(dir.mul(tw.div(tau))).toVar();
            const toSun = smokeSunTransmitTSL(shared, mid, sunDir)
              .mul(sunMapFarTSL(mid, float(1), float(0.5), float(0), float(0)))
              .mul(sunRoofTSL(terrain, mid, sunDir, float(12)));
            const open = terrain && terrain.skyTop ? skyOpenTSL(terrain, mid) : float(1);
            const lit = sunLight.mul(toSun).add(skyLight.mul(open)).mul(par.xyz);
            const T = exp(tau.mul(shared.strength).negate());
            gathered.addAssign(lit.mul(float(1).sub(T)).mul(exp(tauAll.negate())));
            tauAll.addAssign(tau.mul(shared.strength));
          });
        });
      });
    }));
    return vec4(gathered, tauAll);
  }).setLayout(layout('smokeInscatter', 'vec4', [
    { name: 'origin', type: 'vec3' }, { name: 'dir', type: 'vec3' }, { name: 'reach', type: 'float' },
    { name: 'sunDir', type: 'vec3' }, { name: 'sunLight', type: 'vec3' }, { name: 'skyLight', type: 'vec3' }]));
  cache.set(terrain, fn);
  return fn;
}

// A sun shadow (p, lift, skip) -> visibility, through the cloud spells too:
// the ground under a cloud is in its shadow. smoke null: the shadow as it is.
export const withSmokeTSL = (smoke, sunDir, shadow) => (smoke
  ? (p, lift, skip) => shadow(p, lift, skip).mul(smokeSunTransmitTSL(smoke, p, sunDir))
  : shadow);
