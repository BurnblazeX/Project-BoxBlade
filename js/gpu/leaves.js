// --- Leaves on the GPU: cubes that sway, lit once a cube, textured by noise ---
//
// Every placed tree's leaf cells open to air (core/leaves.js) are one
// instanced box mesh, packed: LEAF_CUBE_CAPACITY at most, nearest the player
// first past that. Drawn as spell smoke's cubes are (gpu/smoke.js), lit by
// the same world light (gpu/cubelight.js).
// light   per cube, LEAF_LIGHT_SLICE cubes a frame round the list (the whole
//         list at once when it is written): the world's light at its centre -
//         the sun through the world and any spell smoke, the lamps, the sky.
//         Kept apart: the sun, and the lamps with the sky.
// draw    a cube is its full cell, and only its faces open to air are drawn
//         (a closed face collapses to the cube's centre): no face of one cube
//         ever lies on another's. The crown sways as one body: every corner
//         is moved by a smooth field of its REST position - a lean along the
//         wind (gpu/wind.js), a flutter across it and a little lift - so the
//         corners neighbouring cubes share move together and the crown stays
//         watertight. (Cubes swelling each on its own phase, as smoke's do,
//         crossed their neighbours' faces: z-fighting stripes on every leaf,
//         2026-10-30.) Its faces are cellular (Worley) noise on the texel
//         grid - one value a texel, so pixel-sized - read at the rest
//         position, so the pattern moves with the cube instead of swimming
//         over it, and continues across coplanar faces of neighbouring
//         cubes. Four levels, near a feature point lightest, coloured by the
//         species' ramp (darkest to lightest). The sun takes the face's N.L,
//         and a face turned from it the light that crossed the crown and the
//         cube (SSS - fewer leaves, more glow); the lamps and sky light it
//         evenly - both through the leaves (gpu/leafvolume.js), so the
//         crown shades itself. The albedo darkens toward the crown's bottom
//         (gradient: the share it loses there).
import * as THREE from 'three/webgpu';
import { Fn, If, float, vec2, vec3, vec4, int, uniform, uniformArray, instanceIndex, storage, select,
         max, min, sin, exp, fract, floor, round, abs, clamp, length, varying, positionGeometry, normalGeometry } from 'three/tsl';
import { createSunUniforms } from './sun.js';
import { createCubeWorldLight } from './cubelight.js';
import { smokeShared, smokeSunTransmitTSL } from './smokeshadow.js';
import { wind } from './wind.js';
import { leafVolume, leafLightTSL } from './leafvolume.js';
import { TREE_SPECIES } from '../core/treegen.js';
import { BOX_METRES } from '../core/boxgrid.js';
import { placedLeafCubes } from '../core/leaves.js';

export const LEAF_CUBE_CAPACITY = 131072;
// Cubes lit a frame: the whole list over a few frames.
export const LEAF_LIGHT_SLICE = 16384;
export const DEFAULT_LEAVES = {
  albedo: 1, sway: 0.06, speed: 1.6, waveLength: 6, clump: 3,
  sun: 1, lamps: 1, sky: 1, gradient: 0.35, sss: 0.8
};

const TEXEL = BOX_METRES / 4;
// A point in the unit square for lattice cell v of face plane s.
const hash2 = (v, s) => fract(sin(vec2(v.dot(vec2(127.1, 311.7)).add(s.mul(74.7)),
                                       v.dot(vec2(269.5, 183.3)).add(s.mul(246.1)))).mul(float(43758.5453)));

// The species' ramps, four colours each in species order (core/leaves.js
// LEAF_SPECIES), linear.
function rampRows() {
  return Object.values(TREE_SPECIES).flatMap(({ ramp }) => [0, 1, 2, 3].map(i => {
    const [r, g, b] = ramp[Math.round(i / 3 * (ramp.length - 1))];
    const c = new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
    return new THREE.Vector4(c.r, c.g, c.b, 1);
  }));
}

// A cube's w (core/leaves.js): open faces x 4 + species, + its sky share.
const speciesOf = w => int(floor(w)).mod(int(4));
const openOf = w => int(floor(w)).div(int(4));

// Where the sway moves a corner at rest position r (see draw, above): smooth
// in r, so shared corners move together.
function swayTSL(r, u) {
  const along = r.xz.dot(wind.dir).mul(float(2 * Math.PI)).div(u.waveLength).toVar();
  const across = r.xz.dot(vec2(wind.dir.y.negate(), wind.dir.x)).mul(float(2 * Math.PI)).div(u.waveLength.mul(1.7));
  const t = u.time.mul(u.speed);
  const a = u.sway.mul(wind.strength);
  const push = sin(t.sub(along)).mul(0.5).add(0.5);           // leans with the wind and back
  const flutter = sin(t.mul(1.9).sub(along.mul(1.3)).add(across)).mul(0.35);
  const lift = sin(t.mul(1.3).sub(along).add(r.y.mul(0.9))).mul(0.25);
  return vec3(wind.dir.x.mul(push).sub(wind.dir.y.mul(flutter)), lift,
              wind.dir.y.mul(push).add(wind.dir.x.mul(flutter))).mul(a);
}

// The leaf colour on a face with normal n at rest position r (see draw,
// above); w: the cube's.
function leafColourTSL(r, n, w, direct, iso, ramps, u, sunDir) {
  // Every input to the hash an exact integer, and small: the interpolated
  // normal is 1 give or take its last bit from pixel to pixel, and sin() at
  // ~1e5 turned that bit into a different point - texels near a level's edge
  // broke into shimmering per-pixel static (2026-10-30).
  const nr = round(n).toVar();
  const q = r.div(float(TEXEL)).toVar();
  const an = abs(nr);
  const uv = select(an.x.greaterThan(0.5), q.zy, select(an.y.greaterThan(0.5), q.xz, q.xy));
  // The face's plane: its own points, shared with coplanar faces - the
  // plane's place along its axis, and which way the face looks (0 .. 6).
  const plane = floor(q.dot(nr).add(0.5)).mod(256).add(nr.dot(vec3(1, 2, 3)).add(3).mul(256));
  const g = floor(uv).add(0.5).div(u.clump).toVar();
  const cell = floor(g).toVar(), f = fract(g).toVar();
  let d = float(4);
  for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
    const o = vec2(ox, oy);
    d = min(d, length(o.add(hash2(cell.add(o).mod(256), plane)).sub(f)));
  }
  const level = int(clamp(floor(float(1).sub(d).mul(4)), float(0), float(3)));
  const albedo = ramps.element(speciesOf(w).mul(int(4)).add(level)).xyz;
  // Darker toward the crown's bottom: its height share in w's fraction.
  const shade = float(1).sub(u.gradient.mul(float(1).sub(fract(w))));
  // The sun on a face: its N.L where it faces the sun; where it faces away,
  // what came through the crown to this cube, then through the cube itself
  // (one leaf box more), times the SSS gain - so it dims with every leaf box
  // the light crossed. (A flat 0.45 of the sun on every face lit a cube's
  // back the same behind one leaf box as behind eight, 2026-10-30.)
  const nl = n.dot(sunDir);
  const own = exp(leafVolume.u.strength.negate());
  const sunOn = max(nl, 0).add(max(nl.negate(), 0).mul(own).mul(u.sss));
  return albedo.mul(direct.mul(sunOn).add(iso.mul(float(0.8).add(n.y.mul(0.2))))).mul(shade).mul(u.albedo);
}

// ctx as createSmokeCubes's (gpu/smoke.js): cascades, lights, ..., or world
// (basic mode, the box light's walks).
export function createLeafCubes({ cascades, lights, biasUniform, fadeStartUniform, edgeFadeUniform,
                                  ambientUniform, sky, terrain, glass = false, cards = null, lightCards = null,
                                  shared = smokeShared, world = null }) {
  const COUNT = LEAF_CUBE_CAPACITY;
  // Per cube: centre (world metres), species + sky share (core/leaves.js).
  const cubeAttr = new THREE.StorageInstancedBufferAttribute(new Float32Array(COUNT * 4), 4);
  const cubes = storage(cubeAttr, 'vec4', COUNT).setName('bxbLeafCubes');
  // Per cube: the sun, then the lamps and sky.
  const lightAttr = new THREE.StorageBufferAttribute(COUNT * 2 * 4, 4);
  const light = storage(lightAttr, 'vec4', COUNT * 2).setName('bxbLeafLight');
  const lightRead = storage(lightAttr, 'vec4', COUNT * 2).toReadOnly().setName('bxbLeafLightRead');
  const ramps = uniformArray(rampRows(), 'vec4').setName('bxbLeafRamps');

  const u = {
    sunLight: uniform(float(1)),
    count: uniform(int(0)),          // cubes in the list
    offset: uniform(int(0)),         // the first cube this frame's light pass takes
    time: uniform(float(0)),
    ...Object.fromEntries(Object.entries(DEFAULT_LEAVES).map(([k, v]) => [k, uniform(v)]))
  };
  const sun = createSunUniforms(new THREE.Vector3(0, 1, 0));
  const worldLight = createCubeWorldLight({ cascades, lights, biasUniform, fadeStartUniform, edgeFadeUniform,
                                            ambientUniform, sky, terrain, glass, cards, lightCards, shared, world, sun, u });

  const lightKernel = Fn(() => {
    const i = int(instanceIndex).add(u.offset);
    If(i.lessThan(u.count), () => {
      const p = cubes.element(i).xyz.toVar();
      // Through the leaves (gpu/leafvolume.js): the crown shades itself.
      const through = leafLightTSL(p).toVar();
      const sunIn = worldLight.sun(p).mul(smokeSunTransmitTSL(shared, p, sun.dir)).mul(through.x);
      const iso = worldLight.lamps(p).mul(u.lamps).add(worldLight.sky(p).mul(u.sky).mul(through.y));
      light.element(i.mul(int(2))).assign(vec4(sunIn, 0));
      light.element(i.mul(int(2)).add(int(1))).assign(vec4(iso, 0));
    });
  })().compute(COUNT).setName('leafLight');

  // The cubes: their open faces, each corner swayed from its rest position.
  const cube = cubes.element(instanceIndex);
  const n = normalGeometry;
  const face = select(n.x.greaterThan(0.5), int(0), select(n.x.lessThan(-0.5), int(1),
    select(n.y.greaterThan(0.5), int(2), select(n.y.lessThan(-0.5), int(3), select(n.z.greaterThan(0.5), int(4), int(5))))));
  const open = openOf(cube.w).shiftRight(face).bitAnd(int(1)).equal(int(1));
  const rest = cube.xyz.add(positionGeometry.mul(float(BOX_METRES))).toVar();
  const mat = new THREE.MeshBasicNodeMaterial();
  mat.positionNode = select(open, rest.add(swayTSL(rest, u)), cube.xyz);
  const at = int(instanceIndex).mul(int(2));
  const vDirect = varying(lightRead.element(at).xyz, 'vLeafDirect');
  const vIso = varying(lightRead.element(at.add(int(1))).xyz, 'vLeafIso');
  const vRest = varying(rest, 'vLeafRest');
  const vW = varying(cube.w, 'vLeafW');
  mat.colorNode = leafColourTSL(vRest, normalGeometry, vW, vDirect, vIso, ramps, u, sun.dir);
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat, COUNT);
  mesh.frustumCulled = false;
  mesh.count = 0;
  mesh.name = 'leafCubes';

  return {
    uniforms: u, sun, light: lightKernel, mesh,
    // The leaves of the placed models (core/world.js WorldModels), nearest
    // `near` (metres) kept past the capacity. Returns { count, total }.
    write(models, near = null) {
      const r = placedLeafCubes(models, cubeAttr.array, near);
      if (r.count) {
        cubeAttr.addUpdateRange(0, 4 * r.count);
        cubeAttr.needsUpdate = true;
      }
      mesh.count = r.count;
      u.count.value = r.count;
      return r;
    },
    dispose() { mesh.geometry.dispose(); mat.dispose(); }
  };
}
