// --- Cloud spells on the GPU: cubes, lit once a cube, and shadows through them ---
//
// Each cloud (core/smoke.js - Fog Cloud, Stinking Cloud, ...) is a volume of
// cubes on the BoxGrid. Only its surface cubes are drawn and lit (the rest is
// only ever inside - it still shades, through gpu/smokeshadow.js). All clouds'
// cubes are one instanced box mesh, packed: SMOKE_CUBE_CAPACITY at most;
// a cube is (centre, how far the smoke travelled to it - when it rolls in), unused
// ones never appear.
//
// The smoke as the rest of the renderer sees it - its boxes, for shadows
// through it - is gpu/smokeshadow.js.
// light         per cube, every frame: the sun through the world (the shared
//               cone march, glass-tinted, roof-tested) and through all the
//               smoke (smokeTauTSL - one cloud shades another), the light
//               list (shadowed by the world and the smoke) and the sky (by the
//               roof map, less the smoke above). Light scattered more than
//               once inside the smoke shares the same eight-box cutoff.
//               Kept apart: the sun direct,
//               the sun scattered, the lamps and sky.
//               In basic mode (world: the box light's walks, gpu/boxlight.js)
//               the world's part comes from the BoxGrid instead of the field:
//               the sun walked to (glass, sprite cards), every lamp walked to,
//               and the sky straight up - in the basic material's units
//               (irradiance / pi); the smoke's part is the same.
// draw          a cube is its full grid cell at rest. It grows in when the
//               cloud rolls out to it (as its box does in the shadows), then breathes: it swells larger and back
//               with a wave travelling through the cloud. A soft noise on the
//               texel grid, drifting through the cloud in world space, is what
//               the scattered light shows; the direct sun takes the face's N.L;
//               the lamps and sky light it evenly; the colour is the spell's.
//               Each cube has a fixed sliver of depth bias, so cubes whose
//               faces cross flush do not flicker.
import * as THREE from 'three/webgpu';
import { Fn, If, float, vec3, vec4, int, uniform, instanceIndex, storage, select,
         exp, max, cos, fract, sin, varying, positionGeometry, normalGeometry,
         positionWorld, cameraProjectionMatrix, cameraViewMatrix } from 'three/tsl';
import { SMOKE_CUBE, SMOKE_GROW_IN, SMOKE_BOX } from '../core/smoke.js';
import { createConeTraceSunTSL } from './trace.js';
import { createPointLightsTSL } from './pointlights.js';
import { createSunUniforms, sunRoofTSL } from './sun.js';
import { glassRayTransmitTSL } from './glass.js';
import { NO_SKIP, createCardsTSL, CARD_SUN_MAX } from './cards.js';
import { smokeView, smokeColourTSL, SMOKE_CUBE_CAPACITY } from './smokeview.js';
export { SMOKE_CUBE_CAPACITY, DEFAULT_SMOKE } from './smokeview.js';
import { skyShTSL, sunColourUniform } from './sky.js';
import { skyOpenTSL } from './volume.js';
import { liftAtTSL } from './occlusion.js';
import { MAX_SMOKE_CLOUDS, smokeShared, smokeTauTSL, smokeGrowthTSL, smokeSunTransmitTSL } from './smokeshadow.js';

export { MAX_SMOKE_CLOUDS, smokeShared };

// Cubes drawn, all clouds together: only those that can be seen (core/smoke.js)
// - open to air, or pressed against solid and shown only while the interiors
// lift that solid away (shownTSL). The inside is never seen, and lights
// nothing. One packed list: the mesh draws exactly the cubes there are. (Per
// cloud slots of 12288 cut a cloud filling a room short - its smoke out of the
// door went missing - and drew every unused slot's instances anyway.)

// Seconds a cube stays after it is enclosed, while a cloud rolls out: the
// front thins out a layer behind itself.
export const SMOKE_ENCLOSED_DELAY = 0.25;

export const NEVER = 1e9;     // the measure of an unused cube: it never appears

// Depth bias between cubes, in clip depth per unit w (see draw, above): at
// 20 m it is a few millimetres.
const CUBE_DEPTH_BIAS = 2e-6;

const hash = v => fract(sin(v.dot(vec3(12.9898, 78.233, 37.719))).mul(float(43758.5453)));

// ctx as createLPV's: cascades, lights, biasUniform, fadeStartUniform,
// edgeFadeUniform, ambientUniform, sky (bindings), terrain (the terrain
// textures: the roof map), glass (the terrain textures in a world with glass);
// shared: smokeShared. world: the box light's walks (basic mode) - then no
// cascades, lights, sky or cards are needed.
export function createSmokeCubes({ cascades, lights, biasUniform, fadeStartUniform, edgeFadeUniform,
                                   ambientUniform, sky, terrain, glass = false, cards = null, lightCards = null,
                                   shared = smokeShared, world = null }) {
  const COUNT = SMOKE_CUBE_CAPACITY;
  // A cube's w: travel + AGAINST x the faces it presses on solid with (0 for
  // one open to air) + SLOT x its cloud's slot. Shown when the faces are 0, or
  // a block it presses on lifts.
  const AGAINST = 128, SLOT = 8192;
  const slotOf = w => int(w.div(float(SLOT)));
  const rest = w => w.sub(float(slotOf(w)).mul(float(SLOT)));
  const FACE_DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const against = w => int(rest(w).div(float(AGAINST)));
  const travelOf = w => rest(w).sub(float(against(w)).mul(float(AGAINST)));
  // Seen while the cloud rolls out until enclosed (and a little after), then
  // only if open to air or against a lifted block (below).
  const growingTSL = (w, share) => {
    const k = slotOf(w);
    const at = share.mul(shared.row(k, 3).w).add(float(SMOKE_GROW_IN + SMOKE_ENCLOSED_DELAY));
    return shared.time.sub(shared.row(k, 2).w).lessThan(at);
  };
  const shownTSL = (centre, w, share) => growingTSL(w, share).or(openTSL(centre, w));
  const openTSL = (centre, w) => {
    // Plain selects, no If: it is also built straight into the vertex position.
    const m = against(w);
    let lifted = float(0);
    FACE_DIRS.forEach(([x, y, z], f) => {
      const lift = liftAtTSL(centre.add(vec3(x, y, z).mul(float(SMOKE_CUBE))));
      lifted = max(lifted, select(m.shiftRight(int(f)).bitAnd(int(1)).equal(int(1)), lift, float(0)));
    });
    return m.equal(int(0)).or(lifted.greaterThan(float(0)));
  };
  // Per cube: centre (world metres), how far the smoke travelled to it (when
  // it rolls in: smokeGrowthTSL, as its box in the shadows).
  const cubeAttr = new THREE.StorageInstancedBufferAttribute(new Float32Array(COUNT * 4).fill(NEVER), 4);
  const cubes = storage(cubeAttr, 'vec4', COUNT).setName('bxbSmokeCubes');
  // Per cube: animated bounds, direct sun, scattered sun, then lamps and sky.
  const lightAttr = smokeView.attr;
  const light = storage(lightAttr, 'vec4', COUNT * 4).setName('bxbSmokeLight');
  const lightRead = smokeView.read;
  // Per cube: when it is enclosed, a share of the cloud's farthest travel
  // (NEVER: open to air).
  const enclosedAttr = new THREE.StorageInstancedBufferAttribute(new Float32Array(COUNT).fill(NEVER), 1);
  const enclosed = storage(enclosedAttr, 'float', COUNT).setName('bxbSmokeEnclosed');

  const u = {
    sunLight: uniform(float(1)),
    count: uniform(int(0)),          // cubes in the list
    ...smokeView.uniforms
  };
  const sun = createSunUniforms(new THREE.Vector3(0, 1, 0));
  const coneTrace = world ? null : createConeTraceSunTSL(cascades, !!glass);
  const spriteShadow = world ? null : createCardsTSL(cards);
  // The lamps through the smoke too - its soft copy, as the sun.
  const pointLights = world ? null : createPointLightsTSL({ cascades, lights, biasUniform, fadeStartUniform,
                                             edgeFadeUniform, cards: lightCards, glass, smoke: shared, smokeSoft: true });
  // Through the soft copy (gpu/smokeshadow.js): the light falls off smoothly
  // through the cloud. Through the hard one, each cube counted whole boxes
  // from where it happened to sit on the lattice, and lit cubes stood out in
  // shaded parts.
  const tau = smokeTauTSL(shared, true);
  const sunReach = float(12);
  const skyIso = () => skyShTSL(sky, vec3(0, 1, 0), true).add(skyShTSL(sky, vec3(0, -1, 0), true))
    .mul(float(0.5)).mul(ambientUniform).mul(sky.gain);

  // The world's light at p, before the smoke: the sun (times the sun on and
  // its gain), the lamps (through the smoke's soft copy) and the sky.
  const PI_INV = float(1 / Math.PI);
  const worldLight = world ? {
    sun: p => world.binding.sunColour.mul(PI_INV).mul(u.sunLight).mul(world.sun(p)).mul(u.sun),
    lamps: p => world.lamps(p).mul(world.binding.lampGain).mul(PI_INV),
    sky: p => world.binding.ambient.mul(PI_INV).mul(world.sky(p))
  } : {
    sun: p => {
      const lit = vec3(0, 0, 0).toVar();
      If(u.sunLight.greaterThan(float(0)), () => {
        const traced = coneTrace(p, sun.dir, sunReach, sun.coneRadius, fadeStartUniform, edgeFadeUniform).toVar();
        const through = glass ? glassRayTransmitTSL(glass, traced, p, sun.dir) : vec3(1, 1, 1);
        lit.assign(through.mul(traced.x.mul(sunRoofTSL(terrain, p, sun.dir, sunReach))));
      });
      if (spriteShadow) lit.mulAssign(spriteShadow(p, sun.dir, float(CARD_SUN_MAX), sun.coneRadius, NO_SKIP));
      return sunColourUniform.mul(u.sunLight).mul(lit).mul(u.sun);
    },
    lamps: p => pointLights.withSpecular(p, vec3(0, 0, 0), NO_SKIP, null),
    sky: p => skyIso().mul(terrain && terrain.skyTop ? skyOpenTSL(terrain, p) : float(1))
  };

  const lightKernel = Fn(() => {
    const i = int(instanceIndex);
    const c = cubes.element(i).toVar();
    const direct = vec3(0, 0, 0).toVar(), scattered = vec3(0, 0, 0).toVar(), iso = vec3(0, 0, 0).toVar();
    const k = slotOf(c.w);
    If(i.lessThan(u.count).and(shared.row(k, 0).w.greaterThan(float(0.5))).and(shownTSL(c.xyz, c.w, enclosed.element(i))), () => {
      const p = c.xyz.toVar();
      // Through the smoke toward the sun and up, from the cube's centre - the
      // same for every cube.
      const transmission = smokeSunTransmitTSL(shared, p, sun.dir).toVar();
      const tauUp = tau(p, vec3(0, 1, 0), float(8)).mul(shared.strength).toVar();
      const sunIn = worldLight.sun(p).toVar();
      // Sunlight falls linearly to zero after eight half-metre boxes.
      // Glow redistributes that transmitted light; it cannot bypass the cap.
      const glow = transmission.mul(float(1).sub(transmission)).mul(u.ms).toVar();
      direct.assign(sunIn.mul(transmission.sub(glow)));
      scattered.assign(sunIn.mul(glow));
      const lamps = worldLight.lamps(p);
      const skyL = worldLight.sky(p).mul(exp(tauUp.mul(float(-0.5))));
      iso.assign(lamps.mul(u.lamps).add(skyL.mul(u.sky)));
    });
    const b = i.mul(int(4));
    const halfSize = float(0).toVar();
    If(i.lessThan(u.count), () => {
      halfSize.assign(select(shownTSL(c.xyz, c.w, enclosed.element(i)),
        smokeGrowthTSL(shared, k, travelOf(c.w).div(max(shared.row(k, 3).z, float(1e-6)))), float(0))
        .mul(float(1).add(u.wave.mul(float(0.5).sub(cos(u.time.mul(u.speed).sub(c.xyz.dot(vec3(1.1, 1.7, 0.8)).mul(2))).mul(0.5)))))
        .mul(SMOKE_CUBE * 0.5));
    });
    light.element(b).assign(vec4(c.xyz, halfSize));
    light.element(b.add(int(1))).assign(vec4(direct, 0));
    light.element(b.add(int(2))).assign(vec4(scattered, 0));
    light.element(b.add(int(3))).assign(vec4(iso, 0));
  })().compute(COUNT).setName('smokeLight');

  // The cubes: their full cell at rest, swelling larger and back.
  const cube = cubes.element(instanceIndex);
  const slot = slotOf(cube.w);
  // Grown in as its box is in the shadows (the same roll from the emitter);
  // an unused cube never.
  const grown = select(shownTSL(cube.xyz, cube.w, enclosed.element(instanceIndex)),
    smokeGrowthTSL(shared, slot, travelOf(cube.w).div(max(shared.row(slot, 3).z, float(1e-6)))), float(0));
  const phase = cube.xyz.dot(vec3(1.1, 1.7, 0.8)).mul(float(2));
  const swell = float(0.5).sub(cos(u.time.mul(u.speed).sub(phase)).mul(float(0.5)));
  const scale = grown.mul(float(1).add(u.wave.mul(swell)));
  const mat = new THREE.MeshBasicNodeMaterial();
  const placed = cube.xyz.add(positionGeometry.mul(float(SMOKE_CUBE)).mul(scale));
  mat.positionNode = placed;
  const clip = cameraProjectionMatrix.mul(cameraViewMatrix.mul(vec4(placed, float(1))));
  mat.vertexNode = vec4(clip.xy, clip.z.sub(hash(cube.xyz).mul(float(CUBE_DEPTH_BIAS)).mul(clip.w)), clip.w);
  const at = int(instanceIndex).mul(int(4)).add(int(1));
  const vDirect = varying(lightRead.element(at).xyz, 'vSmokeDirect');
  const vScattered = varying(lightRead.element(at.add(int(1))).xyz, 'vSmokeScattered');
  const vIso = varying(lightRead.element(at.add(int(2))).xyz, 'vSmokeIso');
  const vColour = varying(shared.row(slot, 1).xyz, 'vSmokeColour');
  // Soft noise: value noise on the texel lattice, blended between texels,
  // under a coarser one - read in the world, so it runs on across
  // neighbouring cubes, and drifting, so it moves through the cloud as smoke.
  mat.colorNode = smokeColourTSL(positionWorld, normalGeometry, vDirect, vScattered, vIso, vColour, u, sun.dir);
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat, COUNT);
  mesh.frustumCulled = false;
  mesh.name = 'smokeCubes';

  // The cubes the last write listed: the next clears and uploads only those.
  let written = 0;
  const gridSpans = smokeView.spans;
  const LAYER = SMOKE_BOX * SMOKE_BOX;

  return {
    uniforms: u, sun, light: lightKernel, mesh, cubeAttr,
    // The cubes of every cloud: clouds[k] = { cloud, compact } in slot k (or
    // null). Each cloud's cubes open to air first, then those seen only
    // through lifted solid, then - while it still rolls out (not compact) -
    // the inside, shown only until enclosed. If the list is full, the last go.
    write(clouds) {
      const arr = cubeAttr.array, enc = enclosedAttr.array;
      let n = 0;
      const grid = smokeView.grid, idx = smokeView.indices;
      const old = gridSpans.slice();
      gridSpans.fill(null);
      for (let k = 0; k < old.length; k++) {
        if (old[k]) idx.fill(0, (k * SMOKE_BOX + old[k][0]) * LAYER, (k * SMOKE_BOX + old[k][1] + 1) * LAYER);
      }
      for (const pass of [0, 1, 2]) {
        clouds.forEach((entry, k) => {
          if (!entry || (pass === 2 && entry.compact)) return;
          const { cloud } = entry;
          for (const c of cloud.cubes) {
            if (n >= COUNT) return;
            const kind = c[7] ? 0 : c[8] ? 1 : 2;
            if (kind !== pass) continue;
            enc[n] = c[9] === Infinity ? NEVER : c[9] / Math.max(cloud.gMax, 1e-6);
            const x = c[4] - cloud.box[0], y = c[5] - cloud.box[1], z = c[6] - cloud.box[2];
            idx[x + SMOKE_BOX * (y + SMOKE_BOX * (z + SMOKE_BOX * k))] = n + 1;
            const span = gridSpans[k] || (gridSpans[k] = [z, z]);
            if (z < span[0]) span[0] = z;
            if (z > span[1]) span[1] = z;
            arr.set([c[0], c[1], c[2], c[3] + AGAINST * c[8] + SLOT * k], 4 * n++);
          }
        });
      }
      // Past the new count only what the last list held needs clearing, and
      // only that span goes up (the buffers are 2.5 MB whole).
      const end = Math.max(n, written);
      arr.fill(NEVER, 4 * n, 4 * end);
      enc.fill(NEVER, n, end);
      written = n;
      let dirty = false;
      for (const spans of [old, gridSpans]) {
        spans.forEach((span, k) => {
          if (!span) return;
          for (let z = span[0]; z <= span[1]; z++) grid.layerUpdates.add(k * SMOKE_BOX + z);
          dirty = true;
        });
      }
      if (dirty) grid.needsUpdate = true;
      if (end > 0) {
        cubeAttr.addUpdateRange(0, 4 * end);
        enclosedAttr.addUpdateRange(0, end);
        cubeAttr.needsUpdate = true;
        enclosedAttr.needsUpdate = true;
      }
      mesh.count = n;
      u.count.value = n;
      return n;
    },
    dispose() { mesh.geometry.dispose(); mat.dispose(); }
  };
}
