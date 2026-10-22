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
//               once inside the smoke - its subsurface light - is the octave
//               approximation (Wrenninge): octave i gives a^i as much and sees
//               b^i of the depth toward the sun. Kept apart: the sun direct,
//               the sun scattered, the lamps and sky.
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
import { SMOKE_CUBE, SMOKE_GROW_IN } from '../core/smoke.js';
import { createConeTraceSunTSL } from './trace.js';
import { createPointLightsTSL } from './pointlights.js';
import { createSunUniforms, sunRoofTSL } from './sun.js';
import { glassRayTransmitTSL } from './glass.js';
import { NO_SKIP } from './cards.js';
import { skyShTSL, sunColourUniform } from './sky.js';
import { skyOpenTSL } from './volume.js';
import { valueNoise } from './fog.js';
import { liftAtTSL } from './occlusion.js';
import { MAX_SMOKE_CLOUDS, SMOKE_REACH, smokeShared, smokeTauTSL, smokeGrowthTSL } from './smokeshadow.js';

export { MAX_SMOKE_CLOUDS, smokeShared };

// Cubes drawn, all clouds together: only those that can be seen (core/smoke.js)
// - open to air, or pressed against solid and shown only while the interiors
// lift that solid away (shownTSL). The inside is never seen, and lights
// nothing. One packed list: the mesh draws exactly the cubes there are. (Per
// cloud slots of 12288 cut a cloud filling a room short - its smoke out of the
// door went missing - and drew every unused slot's instances anyway.)
export const SMOKE_CUBE_CAPACITY = 131072;

// Seconds a cube stays after it is enclosed, while a cloud rolls out: the
// front thins out a layer behind itself.
export const SMOKE_ENCLOSED_DELAY = 0.25;

export const NEVER = 1e9;     // the measure of an unused cube: it never appears

// The noise's finest lattice: a texel. A coarser one at NOISE_COARSE times it.
const NOISE_TEXEL = 0.125;
const NOISE_COARSE = 4;

// Depth bias between cubes, in clip depth per unit w (see draw, above): at
// 20 m it is a few millimetres.
const CUBE_DEPTH_BIAS = 2e-6;

export const DEFAULT_SMOKE = {
  albedo: 1,         // brightness on the spell's colour
  wave: 0.22,        // how much larger than its cell a cube swells
  speed: 2.2,        // the wave's speed (radians a second)
  noise: 0.35,       // how much the soft noise thins and thickens the glow
  drift: 0.12,       // metres a second the noise drifts through the cloud
  ms: 0.8,           // the glow through: 0 single scattering, 1 the full octaves
  sun: 1,            // the sun's share
  lamps: 1,          // the light list's share
  sky: 0.6           // the sky's share
};

const MS_OCTAVES = 3;
const MS = { a: 0.5, b: 0.5 };

const hash = v => fract(sin(v.dot(vec3(12.9898, 78.233, 37.719))).mul(float(43758.5453)));

// ctx as createLPV's: cascades, lights, biasUniform, fadeStartUniform,
// edgeFadeUniform, ambientUniform, sky (bindings), terrain (the terrain
// textures: the roof map), glass (the terrain textures in a world with glass);
// shared: smokeShared.
export function createSmokeCubes({ cascades, lights, biasUniform, fadeStartUniform, edgeFadeUniform,
                                   ambientUniform, sky, terrain, glass = false, shared = smokeShared }) {
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
  // Per cube: the sun direct, the sun scattered, the lamps and sky.
  const lightAttr = new THREE.StorageBufferAttribute(COUNT * 3, 4);
  const light = storage(lightAttr, 'vec4', COUNT * 3).setName('bxbSmokeLight');
  const lightRead = storage(lightAttr, 'vec4', COUNT * 3).toReadOnly().setName('bxbSmokeLightRead');
  // Per cube: when it is enclosed, a share of the cloud's farthest travel
  // (NEVER: open to air).
  const enclosedAttr = new THREE.StorageInstancedBufferAttribute(new Float32Array(COUNT).fill(NEVER), 1);
  const enclosed = storage(enclosedAttr, 'float', COUNT).setName('bxbSmokeEnclosed');

  const u = {
    sunLight: uniform(float(1)),
    time: uniform(float(0)),
    count: uniform(int(0)),          // cubes in the list
    ...Object.fromEntries(Object.entries(DEFAULT_SMOKE).map(([k, v]) => [k, uniform(float(v))]))
  };
  const sun = createSunUniforms(new THREE.Vector3(0, 1, 0));
  const coneTrace = createConeTraceSunTSL(cascades, !!glass);
  // The lamps through the smoke too - its soft copy, as the sun.
  const pointLights = createPointLightsTSL({ cascades, lights, biasUniform, fadeStartUniform,
                                             edgeFadeUniform, cards: null, glass, smoke: shared, smokeSoft: true });
  // Through the soft copy (gpu/smokeshadow.js): the light falls off smoothly
  // through the cloud. Through the hard one, each cube counted whole boxes
  // from where it happened to sit on the lattice, and lit cubes stood out in
  // shaded parts.
  const tau = smokeTauTSL(shared, true);
  const sunReach = float(12);
  const skyIso = () => skyShTSL(sky, vec3(0, 1, 0), true).add(skyShTSL(sky, vec3(0, -1, 0), true))
    .mul(float(0.5)).mul(ambientUniform).mul(sky.gain);

  const lightKernel = Fn(() => {
    const i = int(instanceIndex);
    const c = cubes.element(i).toVar();
    const direct = vec3(0, 0, 0).toVar(), scattered = vec3(0, 0, 0).toVar(), iso = vec3(0, 0, 0).toVar();
    const k = slotOf(c.w);
    If(i.lessThan(u.count).and(shared.row(k, 0).w.greaterThan(float(0.5))).and(shownTSL(c.xyz, c.w, enclosed.element(i))), () => {
      const p = c.xyz.toVar();
      // Through the smoke toward the sun and up, from the cube's centre - the
      // same for every cube.
      const tauSun = tau(p, sun.dir, float(SMOKE_REACH)).mul(shared.strength).toVar();
      const tauUp = tau(p, vec3(0, 1, 0), float(8)).mul(shared.strength).toVar();
      const world = vec3(0, 0, 0).toVar();
      If(u.sunLight.greaterThan(float(0)), () => {
        const traced = coneTrace(p, sun.dir, sunReach, sun.coneRadius, fadeStartUniform, edgeFadeUniform).toVar();
        const through = glass ? glassRayTransmitTSL(glass, traced, p, sun.dir) : vec3(1, 1, 1);
        world.assign(through.mul(traced.x.mul(sunRoofTSL(terrain, p, sun.dir, sunReach))));
      });
      const sunIn = sunColourUniform.mul(u.sunLight).mul(world).mul(u.sun).toVar();
      direct.assign(sunIn.mul(exp(tauSun.negate())));
      const more = float(0).toVar();
      for (let o = 1; o < MS_OCTAVES; o++) more.addAssign(float(MS.a ** o).mul(exp(tauSun.mul(float(-(MS.b ** o))))));
      scattered.assign(sunIn.mul(more).mul(u.ms));
      const lamps = pointLights.withSpecular(p, vec3(0, 0, 0), NO_SKIP, null);
      const skyL = skyIso().mul(terrain && terrain.skyTop ? skyOpenTSL(terrain, p) : float(1))
        .mul(exp(tauUp.mul(float(-0.5))));
      iso.assign(lamps.mul(u.lamps).add(skyL.mul(u.sky)));
    });
    const b = i.mul(int(3));
    light.element(b).assign(vec4(direct, 0));
    light.element(b.add(int(1))).assign(vec4(scattered, 0));
    light.element(b.add(int(2))).assign(vec4(iso, 0));
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
  const world = cube.xyz.add(positionGeometry.mul(float(SMOKE_CUBE)).mul(scale));
  mat.positionNode = world;
  const clip = cameraProjectionMatrix.mul(cameraViewMatrix.mul(vec4(world, float(1))));
  mat.vertexNode = vec4(clip.xy, clip.z.sub(hash(cube.xyz).mul(float(CUBE_DEPTH_BIAS)).mul(clip.w)), clip.w);
  const at = int(instanceIndex).mul(int(3));
  const vDirect = varying(lightRead.element(at).xyz, 'vSmokeDirect');
  const vScattered = varying(lightRead.element(at.add(int(1))).xyz, 'vSmokeScattered');
  const vIso = varying(lightRead.element(at.add(int(2))).xyz, 'vSmokeIso');
  const vColour = varying(shared.row(slot, 1).xyz, 'vSmokeColour');
  // Soft noise: value noise on the texel lattice, blended between texels,
  // under a coarser one - read in the world, so it runs on across
  // neighbouring cubes, and drifting, so it moves through the cloud as smoke.
  mat.colorNode = Fn(() => {
    const n = normalGeometry;
    const q = positionWorld.add(vec3(u.drift.mul(u.time), u.drift.mul(u.time).mul(float(0.4)),
                                     u.drift.mul(u.time).mul(float(-0.7)))).div(float(NOISE_TEXEL)).toVar();
    const noise = valueNoise(q).mul(float(0.4)).add(valueNoise(q.div(float(NOISE_COARSE))).mul(float(0.6))).toVar();
    const thin = u.noise.mul(noise.sub(float(0.5))).mul(float(2)).toVar();
    const shade = vDirect.mul(float(0.45).add(float(0.55).mul(max(n.dot(sun.dir), float(0)))))
      .add(vScattered.mul(float(1).add(thin)))
      .add(vIso.mul(float(0.8).add(float(0.2).mul(n.y))).mul(float(1).add(thin.mul(float(0.25)))));
    return shade.mul(vColour).mul(u.albedo);
  })();
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat, COUNT);
  mesh.frustumCulled = false;
  mesh.name = 'smokeCubes';

  return {
    uniforms: u, sun, light: lightKernel, mesh, cubeAttr,
    // The cubes of every cloud: clouds[k] = { cloud, compact } in slot k (or
    // null). Each cloud's cubes open to air first, then those seen only
    // through lifted solid, then - while it still rolls out (not compact) -
    // the inside, shown only until enclosed. If the list is full, the last go.
    write(clouds) {
      const arr = cubeAttr.array, enc = enclosedAttr.array;
      let n = 0;
      for (const pass of [0, 1, 2]) {
        clouds.forEach((entry, k) => {
          if (!entry || (pass === 2 && entry.compact)) return;
          const { cloud } = entry;
          for (const c of cloud.cubes) {
            if (n >= COUNT) return;
            const kind = c[7] ? 0 : c[8] ? 1 : 2;
            if (kind !== pass) continue;
            enc[n] = c[9] === Infinity ? NEVER : c[9] / Math.max(cloud.gMax, 1e-6);
            arr.set([c[0], c[1], c[2], c[3] + AGAINST * c[8] + SLOT * k], 4 * n++);
          }
        });
      }
      arr.fill(NEVER, 4 * n);
      cubeAttr.needsUpdate = true;
      enclosedAttr.needsUpdate = true;
      mesh.count = n;
      u.count.value = n;
      return n;
    },
    dispose() { mesh.geometry.dispose(); mat.dispose(); }
  };
}
