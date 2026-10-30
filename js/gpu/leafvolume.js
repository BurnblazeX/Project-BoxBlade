// --- The leaf volume: the light through every tree's leaves ---
//
// One binding (a module singleton, as gpu/smokeshadow.js smokeShared): a
// window of LEAF_DIMS BoxGrid boxes round the player (min box `origin`),
// moved when the player strays from its middle.
//   density  1 where a leaf cell is (core/leaves.js fillLeafDensity), read
//            nearest by the kernel.
//   index    per box, the drawn leaf cube there (gpu/leaves.js fillIndex): its
//            number + 1 in the low ID_BITS bits (0: none), then its species
//            (2 bits) and its height in the crown (8 bits).
//   light    per box, how much light reaches its centre through the leaves:
//            r the sun (a march toward it, a box a step), g the sky - five
//            marches, up and four at 45 degrees, cosine-weighted: most of the
//            sky gets round a crown, so under one is shaded, not black (one
//            march straight up left dark patches, 2026-10-30); each leaf box
//            crossed lets exp(-strength) through,
//            the box's own leaves left out (a leaf cube's faces are lit, not
//            shaded by itself). Read trilinear: soft, half-metre dapple. The
//            sky kernel reruns when the leaves change (into a buffer), the sun
//            kernel when they change or the sun moves - not every frame.
//            b: 1 where a drawn leaf cube is, a: its species x 64 + its
//            height (0 .. 63), / 255 - so a reflection finds the leaves in a
//            texture the shading binds anyway, not in another storage buffer
//            (splitFinal ran out of them: 9 of 8, 2026-10-30).
// leafLightTSL(p)  vec2(sun, sky) at world p; 1, 1 outside the window or with
//            no leaves.
// withLeavesTSL  a sun shadow (p, lift, skip) -> visibility, through the
//            leaves too: dappled shade under a tree (gpu/shade.js).
// leafDensityTSL(p)  the leaves at p, 0..1, filtered: the LPV's cells lose
//            the bounce through them, and the lit leaves give their own
//            (gpu/gi.js).
// leafTransmitTSL  a lamp's ray through the leaves: exp(-optical depth),
//            marched a box a step over the part of it inside the leaves'
//            bounds (u.lo, u.hi) - so a lamp lit through eight leaf cells is
//            dimmer than through one (gpu/pointlights.js).
// leafInscatterTSL  a texel GI ray through the leaves: what they hold back,
//            and the light the lit leaves give it - the sun only where the
//            world lets it reach them (the sun depth map and roof map at the
//            leaves, as spell smoke's since 2026-10-30), the sky where it is
//            open; times the leaves' bounce colour (u.albedo).
import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, vec3, vec4, int, uvec3, uniform, instanceIndex, texture3D, textureStore, storage,
         exp, floor, ceil, select, clamp, max, min } from 'three/tsl';
import { BOX_METRES } from '../core/boxgrid.js';
import { layout } from './layout.js';
import { sunMapFarTSL } from './sunmap.js';
import { sunRoofTSL } from './sun.js';
import { skyOpenTSL } from './volume.js';

export const LEAF_DIMS = [128, 64, 128];
export const LEAF_ID_BITS = 18;
// Boxes a march takes toward the sun (12 m), and each sky march (6 m).
const STEPS = 24, SKY_STEPS = 12;

function createLeafVolume() {
  const [X, Y, Z] = LEAF_DIMS;
  const data = new Uint8Array(X * Y * Z);
  const density = new THREE.Data3DTexture(data, X, Y, Z);
  density.format = THREE.RedFormat;
  density.type = THREE.UnsignedByteType;
  // Filtered: the kernel reads box centres (exact), the GI between them.
  density.minFilter = density.magFilter = THREE.LinearFilter;
  density.generateMipmaps = false;
  density.needsUpdate = true;
  const light = new THREE.Storage3DTexture(X, Y, Z);
  light.type = THREE.UnsignedByteType;
  light.format = THREE.RGBAFormat;
  light.generateMipmaps = false;
  light.minFilter = light.magFilter = THREE.LinearFilter;
  light.wrapS = light.wrapT = light.wrapR = THREE.ClampToEdgeWrapping;
  const u = {
    origin: uniform(new THREE.Vector3()).setName('bxbLeafOrigin'),   // the window's min box
    sunDir: uniform(new THREE.Vector3(0, 1, 0)),
    strength: uniform(float(0.36)),            // optical depth a leaf box: 0.7 through
    albedo: uniform(new THREE.Vector3(0.03, 0.2, 0)),   // the light the leaves bounce, linear (render/features/leaves.js)
    live: uniform(int(0)).setName('bxbLeafLive'),  // 1 once the light has been worked out
    // The leaves inside the window, world metres (min past max: none).
    lo: uniform(new THREE.Vector3(1, 1, 1)).setName('bxbLeafLo'),
    hi: uniform(new THREE.Vector3(0, 0, 0)).setName('bxbLeafHi')
  };
  const dims = vec3(X, Y, Z);
  const at = c => texture3D(density, c.add(0.5).div(dims)).level(0).r;
  const inside = c => c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.z.greaterThanEqual(0))
    .and(c.x.lessThan(float(X))).and(c.y.lessThan(float(Y))).and(c.z.lessThan(float(Z)));
  const march = (c, d, steps = STEPS) => {
    const tau = float(0).toVar();
    Loop({ start: int(1), end: int(steps + 1), type: 'int', condition: '<', name: 'lvs' }, ({ lvs }) => {
      const q = floor(c.add(0.5).add(d.mul(float(lvs)))).toVar();
      If(inside(q), () => { tau.addAssign(at(q)); });
    });
    return exp(tau.mul(u.strength).negate());
  };
  const indexAttr = new THREE.StorageBufferAttribute(new Int32Array(X * Y * Z), 1);
  const index = storage(indexAttr, 'int', X * Y * Z).toReadOnly().setName('bxbLeafIndex');
  const cellOf = i => vec3(float(i.mod(int(X))), float(i.div(int(X)).mod(int(Y))), float(i.div(int(X * Y))));
  const skyAttr = new THREE.StorageBufferAttribute(X * Y * Z, 1);
  const skyBuf = storage(skyAttr, 'float', X * Y * Z).setName('bxbLeafSky');
  const D = Math.SQRT1_2;
  const SKY_DIRS = [[0, 1, 0, 1], [D, D, 0, D], [-D, D, 0, D], [0, D, D, D], [0, D, -D, D]];
  const SKY_WEIGHT = SKY_DIRS.reduce((s, d) => s + d[3], 0);
  const skyKernel = Fn(() => {
    const i = int(instanceIndex);
    const c = cellOf(i).toVar();
    let sky = float(0);
    for (const [x, y, z, w] of SKY_DIRS) sky = sky.add(march(c, vec3(x, y, z), SKY_STEPS).mul(w / SKY_WEIGHT));
    skyBuf.element(i).assign(sky);
  })().compute(X * Y * Z).setName('leafVolumeSky');
  const kernel = Fn(() => {
    const i = int(instanceIndex);
    const c = cellOf(i).toVar();
    const sun = march(c, u.sunDir);
    const id = index.element(i).toVar();
    const here = select(id.bitAnd(int((1 << LEAF_ID_BITS) - 1)).greaterThan(int(0)), float(1), float(0));
    const code = float(id.shiftRight(int(LEAF_ID_BITS)).bitAnd(int(3)).mul(int(64))
      .add(id.shiftRight(int(LEAF_ID_BITS + 4)).bitAnd(int(63)))).div(255);
    textureStore(light, uvec3(c.x, c.y, c.z), vec4(sun, skyBuf.element(i), here, code));
  })().compute(X * Y * Z).setName('leafVolumeLight');
  return { density, data, light, u, kernel, skyKernel, index, indexAttr, dims: LEAF_DIMS };
}

export const leafVolume = createLeafVolume();

export function leafLightTSL(p, v = leafVolume) {
  const c = p.div(float(BOX_METRES)).sub(v.u.origin).add(0.5).toVar();
  const uvw = c.div(vec3(...v.dims));
  const l = texture3D(v.light, clamp(uvw, vec3(0), vec3(1))).level(0);
  const out = uvw.x.greaterThanEqual(0).and(uvw.y.greaterThanEqual(0)).and(uvw.z.greaterThanEqual(0))
    .and(uvw.x.lessThanEqual(1)).and(uvw.y.lessThanEqual(1)).and(uvw.z.lessThanEqual(1))
    .and(v.u.live.equal(int(1)));
  return select(out, l.xy, l.xy.mul(0).add(1));
}

// Inside the window, with the light worked out: 1, else 0.
const insideTSL = (uvw, v) => uvw.x.greaterThanEqual(0).and(uvw.y.greaterThanEqual(0)).and(uvw.z.greaterThanEqual(0))
  .and(uvw.x.lessThanEqual(1)).and(uvw.y.lessThanEqual(1)).and(uvw.z.lessThanEqual(1)).and(v.u.live.equal(int(1)));

export function leafDensityTSL(p, v = leafVolume) {
  const uvw = p.div(float(BOX_METRES)).sub(v.u.origin).add(0.5).div(vec3(...v.dims)).toVar();
  return select(insideTSL(uvw, v), texture3D(v.density, clamp(uvw, vec3(0), vec3(1))).level(0).r, float(0));
}

// Fn([origin, dir, reach, start]) -> how much light gets through the leaves
// along origin + dir t, start < t < reach: the part inside the leaves' bounds,
// a box a step (longer past LEAF_LAMP_STEPS), each step's filtered density
// times strength. start: one box for a point in a leaf cell (its own leaves
// left out, as the leaf volume's light does), 0 on a surface.
const LEAF_LAMP_STEPS = 24;
const transmitFns = new Map();
export function leafTransmitTSL(v = leafVolume) {
  if (transmitFns.has(v)) return transmitFns.get(v);
  const fn = Fn(([origin, dir, reach, start]) => {
    const T = float(1).toVar();
    If(v.u.live.equal(int(1)), () => {
      // The ray clipped to the bounds; an axis it runs along has 1/1e-7 - its
      // slab is all or nothing.
      const inv = float(1).div(dir.add(vec3(1e-7, 1e-7, 1e-7))).toVar();
      const ta = v.u.lo.sub(origin).mul(inv).toVar(), tb = v.u.hi.sub(origin).mul(inv).toVar();
      const near = min(ta, tb).toVar(), far = max(ta, tb).toVar();
      const t0 = max(max(max(near.x, near.y), near.z), start).toVar();
      const t1 = min(min(min(far.x, far.y), far.z), reach).toVar();
      If(t1.greaterThan(t0), () => {
        const n = clamp(ceil(t1.sub(t0).div(float(BOX_METRES))), float(1), float(LEAF_LAMP_STEPS)).toVar();
        const step = t1.sub(t0).div(n).toVar();
        const tau = float(0).toVar();
        Loop({ start: int(0), end: int(LEAF_LAMP_STEPS), type: 'int', condition: '<', name: 'ltr' }, ({ ltr }) => {
          If(float(ltr).lessThan(n), () => {
            tau.addAssign(leafDensityTSL(origin.add(dir.mul(t0.add(float(ltr).add(0.5).mul(step)))), v));
          });
        });
        T.assign(exp(tau.mul(step).div(float(BOX_METRES)).mul(v.u.strength).negate()));
      });
    });
    return T;
  }).setLayout(layout('leafTransmit', 'float', [
    { name: 'origin', type: 'vec3' }, { name: 'dir', type: 'vec3' },
    { name: 'reach', type: 'float' }, { name: 'start', type: 'float' }]));
  transmitFns.set(v, fn);
  return fn;
}

// Fn([origin, dir, reach, sunDir, sunLight, skyLight]) -> vec4(the light the
// lit leaves along the ray give it, their optical depth). A metre a step, at
// most LEAF_RAY_STEPS; from a mid point weighted by the leaves, the world's
// sun test once.
const LEAF_RAY_STEPS = 16;
const inFns = new Map();
export function leafInscatterTSL(terrain = null, v = leafVolume) {
  if (inFns.has(terrain)) return inFns.get(terrain);
  const fn = Fn(([origin, dir, reach, sunDir, sunLight, skyLight]) => {
    const tau = float(0).toVar(), tw = float(0).toVar(), gs = float(0).toVar(), gk = float(0).toVar();
    If(v.u.live.equal(int(1)), () => {
      const step = float(BOX_METRES * 2);
      Loop({ start: int(0), end: int(LEAF_RAY_STEPS), type: 'int', condition: '<', name: 'lis' }, ({ lis }) => {
        const t = float(lis).add(0.5).mul(step).toVar();
        If(t.lessThan(reach), () => {
          const q = origin.add(dir.mul(t)).toVar();
          const d = leafDensityTSL(q, v).mul(v.u.strength).mul(2).toVar();
          If(d.greaterThan(float(1e-4)), () => {
            const l = leafLightTSL(q, v).toVar();
            const share = exp(tau.negate()).mul(float(1).sub(exp(d.negate())));
            gs.addAssign(share.mul(l.x));
            gk.addAssign(share.mul(l.y));
            tw.addAssign(d.mul(t));
            tau.addAssign(d);
          });
        });
      });
    });
    const gathered = vec3(0, 0, 0).toVar();
    If(tau.greaterThan(float(1e-4)), () => {
      const mid = origin.add(dir.mul(tw.div(tau))).toVar();
      const world = sunMapFarTSL(mid, float(1), float(0.5), float(0), float(0)).mul(sunRoofTSL(terrain, mid, sunDir, float(12)));
      const open = terrain && terrain.skyTop ? skyOpenTSL(terrain, mid) : float(1);
      gathered.assign(sunLight.mul(gs.mul(world)).add(skyLight.mul(gk.mul(open))).mul(v.u.albedo));
    });
    return vec4(gathered, tau);
  }).setLayout(layout('leafInscatter', 'vec4', [
    { name: 'origin', type: 'vec3' }, { name: 'dir', type: 'vec3' }, { name: 'reach', type: 'float' },
    { name: 'sunDir', type: 'vec3' }, { name: 'sunLight', type: 'vec3' }, { name: 'skyLight', type: 'vec3' }]));
  inFns.set(terrain, fn);
  return fn;
}

export const withLeavesTSL = (leaves, shadow) => (leaves
  ? (p, lift, skip) => shadow(p, lift, skip).mul(leafLightTSL(p, leaves).x)
  : shadow);
