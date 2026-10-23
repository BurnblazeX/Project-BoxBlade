// --- Box light: basic mode's sun, sky and lamps, on the BoxGrid ---
//
// With the advanced effects off, the terrain wears this instead of the plain
// material: every face is lit by the BoxGrid boxes in front of it
// (core/boxlight.js works out which boxes, gpu/boxlight.js lights them), read
// filtered, so the light blends from texel to texel across the 0.5 m boxes.
//
//   ambient   the plain AmbientLight, darkened by the static voxel AO and by
//             the box's sky (its share of SKY_RAYS that get out; skyFloor
//             where none do)
//   sun       the plain light's colour x N.L, times the box's sun - rgb, so
//             stained glass throws its colours
//   lamps     the light list, each walked to, times form shading (top 1,
//             side 0.8, bottom 0.6 - a box is lit from every side, so a face
//             has no N.L of its own)
// Sprites shadow the sun and the lamps through their cards (gpu/cards.js), the
// ones the shaded path tests, hard; placed models through their boxes.
//
// One pass: three's lighting is off, and the colour node works the whole
// light out from one evaluation of the surface - the plain material ran it
// four times (colour, emission, normal, AO) plus three's lighting model. All
// in three's units: irradiance / pi on the albedo, the ambient with the 5%
// three's dielectric hands its specular (AMBIENT_DIFFUSE).
//
// Cost: two filtered reads a fragment (two more where a slope's plane needs
// them). The sky is walked per surface cell when the world changes, the sun
// when it turns by SUN_STEP, and the lamps and cards when a light row or a
// card's place, facing or frame changed.
//
// ctx:
//   renderer
//   shadowsOn()   whether the shaded material is on (then this stands aside)
//   lighting      the light list feature: bindings(), the rows it walks to
//   cards         the cards feature: options() - the sun's and the lights'
//                 card bindings, null until the atlas has loaded; ready and
//                 sunBindings for the sprites' contact AO
//   sunLight      the plain DirectionalLight: its position is the sun's
//                 direction, colour x intensity its light
//   ambientLight  the plain AmbientLight: colour x intensity
//   smoke         the cloud spells feature: version() and growing() - the
//                 terrain is lit through the smoke, so the lamps pass runs
//                 again when the clouds change and while one rolls out
import * as THREE from 'three/webgpu';
import { Fn, mix, vec3, float, max, select } from 'three/tsl';
import { boxLightRegion, buildBoxMasks, buildBoxCells } from '../../core/boxlight.js';
import { LIGHT_VEC4S } from '../../core/lights.js';
import { materialLayer } from '../../core/materials.js';
import { createBoxLightBinding, createBoxLightKernels, boxLightAtTSL } from '../../gpu/boxlight.js';
import { voxelAOAtTSL, createSpriteAOTSL } from '../../gpu/ambient.js';
import { terrainSurfaceTSL, terrainTexelTSL, terrainWhite } from '../../gpu/terrain.js';
import { outsideShadeTSL } from '../../gpu/occlusion.js';
import { terrainTextures, worldInstancedMesh, plainTerrainMeshMaterial } from '../render.js';
import { flip } from '../engine.js';

// Room past the surface cells and tints a rebuild found, so a few edits do
// not remake the buffers (and recompile the kernels).
const HEADROOM = 1.25;

// How far the sun turns before its walks run again: a hundredth of a radian
// moves a shadow's edge a box only 50 boxes from its caster. With the day
// cycle on, about every 4.6 s.
const SUN_STEP = Math.cos(0.01);

// Of three's ambient on a rough dielectric, what reaches the diffuse layer
// (its single + multi scattering takes the rest) - so the ambient matches
// the plain material's.
const AMBIENT_DIFFUSE = 0.95;

export function createBoxLight({ renderer, shadowsOn, lighting, cards, sunLight, ambientLight,
                                 smoke = { version: () => 0, growing: () => false } }) {
  const binding = createBoxLightBinding();
  let on = true;
  let gpu = null;
  let material = null;
  let materialCards;            // the card bindings its contact AO was made with
  let built = false;
  let builtWithTextures = false;
  let info = null;
  const dirty = { clear: true, static: true, sun: true, lamps: true };
  const sunNow = new THREE.Vector3();
  const lastSun = new THREE.Vector3(0, -2, 0);
  let lastSmoke = -1;
  // Light passes: how many, in the last second, for the stats.
  const passTimes = [];

  // The card bindings in use: null until the atlas is in, or with card
  // shadows off.
  const cardBindings = () => {
    const o = cards.options();
    return { sun: o.cards || null, light: o.lightCards || null };
  };
  const aoCards = () => (cards.ready ? cards.sunBindings : null);

  // Glass albedo, for the tints: top-down sRGB, once its texture has loaded.
  const glassPixels = id => terrainTextures.rgba[materialLayer(id)] || null;

  function build() {
    const t0 = performance.now();
    const region = boxLightRegion();
    const masks = buildBoxMasks(region, undefined, glassPixels);
    const { cells, top } = buildBoxCells(region, masks);
    const cb = cardBindings();
    const fits = gpu && gpu.region.dims.every((v, a) => v === region.dims[a]) &&
                 cells.length <= gpu.capacity && masks.tints.length <= gpu.tintCapacity &&
                 gpu.cards.sun === cb.sun && gpu.cards.light === cb.light;
    if (!fits) {
      if (gpu) gpu.dispose(renderer);
      gpu = createBoxLightKernels({
        binding, region, lights: lighting.bindings(), sunCards: cb.sun, lightCards: cb.light,
        capacity: Math.ceil(cells.length * HEADROOM) + 1024,
        tintCapacity: Math.ceil(masks.tints.length * HEADROOM) + 27 * 16
      });
      binding.setTextures(gpu.texA, gpu.texB);
      lists.clear();
    }
    gpu.region = region;
    gpu.write(masks, cells, top);
    binding.box0.value.set(...region.box0);
    binding.dims.value.set(...region.dims);
    if (region.clipped) console.warn(`[boxlight] the world is wider than ${region.size[0]} x ${region.size[2]} blocks: past it, no box light`);
    info = { region, cells: cells.length, glass: (masks.tints.length - 1) / 27, ms: performance.now() - t0 };
    dirty.clear = dirty.static = dirty.sun = dirty.lamps = true;
    built = true;
    builtWithTextures = !!terrainTextures.ready;
  }

  // The plain material's shape and look, lit in one pass. aoCardBindings: the
  // sun's cards for the sprites' contact AO, or null.
  function makeMaterial(aoCardBindings) {
    const m = plainTerrainMeshMaterial();
    m.lights = false;
    m.emissiveNode = null;
    m.normalNode = null;
    const spriteAO = createSpriteAOTSL(aoCardBindings);
    m.colorNode = Fn(() => {
      const surf = terrainSurfaceTSL();
      const texel = terrainTexelTSL(terrainTextures.albedo, surf.st, surf).rgb.toVar();
      const albedo = mix(texel, vec3(1, 1, 1), terrainWhite);
      const n = surf.n;
      const lit = boxLightAtTSL(binding, surf.p, n);
      const ao = voxelAOAtTSL(terrainTextures, surf, spriteAO);
      const ambient = binding.ambient.mul(ao.mul(float(AMBIENT_DIFFUSE)).mul(mix(binding.skyFloor, float(1), lit.sky)));
      const sun = binding.sunColour.mul(lit.sun).mul(max(n.dot(binding.sunDir), float(0)));
      const form = float(0.8).add(n.y.mul(float(0.2)));
      const light = ambient.add(sun).add(lit.lamp.mul(form).mul(binding.lampGain)).mul(float(1 / Math.PI));
      // A lightSource cube is all emission, as on the plain material.
      return select(surf.emissive, texel, albedo.mul(light)).mul(outsideShadeTSL());
    })();
    return m;
  }

  // The kernel lists, cached: three keys a compute pass's state on the array.
  const lists = new Map();
  function dispatch() {
    const key = `${dirty.clear}:${dirty.static}:${dirty.sun}`;
    let list = lists.get(key);
    if (!list) {
      list = [];
      if (dirty.clear) list.push(gpu.clear);
      if (dirty.static) list.push(gpu.sky);
      if (dirty.sun) list.push(gpu.sun);
      list.push(gpu.lamps);
      lists.set(key, list);
    }
    renderer.compute(list);
    dirty.clear = dirty.static = dirty.sun = dirty.lamps = false;
    const now = performance.now();
    passTimes.push(now);
    while (passTimes[0] < now - 1000) passTimes.shift();
  }

  // What the lamps pass reads, as numbers, compared with the last pass's: the
  // light list's first three rows a light (where, level, colour; its card
  // slice) and the cutoff, and each card's first three (where, facing, size;
  // its atlas frame) - not its fourth, whose LOD follows the camera and means
  // nothing here. Kept in one reused array.
  let seen = new Float32Array(1024), seenLength = -1;
  function changed() {
    const b = lighting.bindings();
    const groups = [[b.data.array, b.count.value, LIGHT_VEC4S, 3]];
    for (const c of [gpu.cards.sun, gpu.cards.light]) if (c) groups.push([c.data.array, c.count.value, 4, 3]);
    let n = 1;
    for (const [, count, , rows] of groups) n += 1 + count * rows * 4;
    if (seen.length < n) { seen = new Float32Array(n * 2); seenLength = -1; }
    let diff = seenLength !== n, o = 0;
    const put = v => { if (seen[o] !== v) { seen[o] = v; diff = true; } o++; };
    put(b.cutoff.value);
    for (const [arr, count, stride, rows] of groups) {
      put(count);
      for (let i = 0; i < count; i++) for (let r = 0; r < rows; r++) {
        const v = arr[i * stride + r];
        put(v.x); put(v.y); put(v.z); put(v.w);
      }
    }
    seenLength = n;
    return diff;
  }

  function active() { return on && !shadowsOn(); }

  // Each frame, before the main render.
  function update() {
    const mesh = worldInstancedMesh;
    if (!mesh || shadowsOn()) return;
    if (!mesh.userData.originalMaterial && mesh.material !== material) mesh.userData.originalMaterial = mesh.material;
    const plain = mesh.userData.originalMaterial;
    if (!on) {
      if (mesh.material === material) mesh.material = plain;
      return;
    }
    // Again once the glass textures are in (their tints), or the card
    // bindings came or went (the kernels test them).
    const cb = cardBindings();
    if (built && ((!builtWithTextures && terrainTextures.ready) ||
                  gpu.cards.sun !== cb.sun || gpu.cards.light !== cb.light)) built = false;
    if (!built) build();
    // Made again when the sprites' cards arrive (the contact AO reads them).
    if (!material || materialCards !== aoCards()) {
      const old = material;
      materialCards = aoCards();
      material = makeMaterial(materialCards);
      if (old) old.dispose();
    }
    if (mesh.material !== material) mesh.material = material;

    sunNow.copy(sunLight.position).normalize();
    if (sunNow.dot(lastSun) < SUN_STEP) {
      lastSun.copy(sunNow);
      binding.sunDir.value.copy(sunNow);
      dirty.sun = true;
    }
    const c = sunLight.color, k = sunLight.intensity;
    binding.sunColour.value.set(c.r * k, c.g * k, c.b * k);
    const a = ambientLight.color, ka = ambientLight.intensity;
    binding.ambient.value.set(a.r * ka, a.g * ka, a.b * ka);
    if (changed()) dirty.lamps = true;
    const sv = smoke.version();
    if (sv !== lastSmoke || smoke.growing()) { lastSmoke = sv; dirty.lamps = true; }
    if (dirty.clear || dirty.static || dirty.sun || dirty.lamps) dispatch();
  }

  const commands = {
    boxlight: {
      help: 'basic mode\'s light on the BoxGrid: sun and sprite shadows, glass colours, sky and lamps (advanced effects off)',
      usage: "bxb.boxlight()  on/off  |  bxb.boxlight('stats')",
      run: (a = null) => {
        if (a === 'stats') {
          if (!info) return 'box light not built yet (it builds on the first basic-mode frame)';
          const d = info.region.dims;
          const now = performance.now();
          const passes = passTimes.filter(t => t > now - 1000).length;
          return `box light: ${d.join(' x ')} boxes (${(d[0] * d[1] * d[2] * 12 / 1048576).toFixed(1)} MB), ` +
                 `${info.cells} surface cells (room for ${gpu.capacity}), ${info.glass} glass blocks, ` +
                 `sprite shadows ${gpu.cards.sun ? 'on' : 'off (no cards yet)'}, built in ${info.ms.toFixed(0)} ms; ` +
                 `${passes} light passes in the last second (0 when nothing moves)`;
        }
        on = typeof a === 'boolean' ? a : !on;
        return `box light ${on ? 'on' : 'off - three\'s unshadowed sun'}` + (shadowsOn() ? ' (seen with the advanced effects off)' : '');
      }
    }
  };

  return {
    name: 'boxlight',
    update,
    active,
    // The world changed: masks and surface cells again, next frame.
    rebuild: () => { built = false; },
    // The walks from any point (gpu/boxlight.js world), for the smoke's cubes:
    // null until built, or with the box light off. A new object when the
    // kernels are remade.
    world: () => (on && built && gpu ? gpu.world : null),
    // For the headless shader build (tools/lib/shaders.mjs): the material and
    // the kernels, made as the first basic-mode frame makes them.
    shaders() {
      if (!built) build();
      if (!material) material = makeMaterial(aoCards());
      return { material, kernels: [gpu.clear, gpu.sky, gpu.sun, gpu.lamps] };
    },
    commands,
    settings: { title: 'Box light (basic mode)', settings: [
      { key: 'boxlight', label: 'Box light', type: 'toggle',
        help: 'sun, sky and lamps worked out per BoxGrid box (0.5 m) - basic mode only',
        get: () => on, set: v => flip(on, v, () => commands.boxlight.run()) },
      { key: 'boxlamps', label: 'Lamp brightness', type: 'range', min: 0, max: 4, step: 0.05,
        help: 'a lamp\'s peak, in sun intensities (the plain sun is 1.2)',
        get: () => binding.lampGain.value, set: v => { binding.lampGain.value = v; } },
      { key: 'boxskyfloor', label: 'Sky floor', type: 'range', min: 0, max: 1, step: 0.01,
        help: 'the ambient left where a box sees no sky (inside a closed room); 1 is the old flat ambient',
        get: () => binding.skyFloor.value, set: v => { binding.skyFloor.value = v; } }
    ]}
  };
}
