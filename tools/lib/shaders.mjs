// Every shader the engine builds, generated headlessly as WGSL.
//
// three's node builder needs no GPU to turn a material or a compute node into
// WGSL - only a renderer object, here one over a stub canvas that is never
// initialised. The inputs mirror render/shadowsets.js obtainShadowSet and main.js shadowNodeOptions
// with the default settings, over the test area (glass, slopes, halves,
// stairs); they need not be realistic, only reach the same code. Used by
// shaders.test.mjs to prove a refactor leaves every shader byte-identical.
import * as THREE from 'three/webgpu';
import { uniform, mix, vec3 } from 'three/tsl';
import { createTestArea } from '../../js/core/world.js';
import { createBoxGridAt, gridOriginFor, CASCADE_COUNT } from '../../js/core/boxgrid.js';
import { POINT_CARD_CAPACITY } from '../../js/core/lights.js';
import { SUN_CARD_LOD_SHIFT, SUN_CARD_LOD_MAX } from '../../js/core/cards.js';
import { SUN_ANGULAR_SIZE } from '../../js/core/sun.js';
import { terrainTextures, initWorldRender } from '../../js/render/render.js';
import * as R from '../../js/render/render.js';
import { terrainTexelTSL, terrainWhite } from '../../js/gpu/terrain.js';
import { createSky } from '../../js/render/features/sky.js';
import { createLights } from '../../js/render/features/lights.js';
import { createMirrors } from '../../js/render/features/mirrors.js';
import { createGI } from '../../js/render/features/gi.js';
import { createDebugView, VIEWS } from '../../js/render/features/debugview.js';
import { createTexelCache, buildTerrainMaterials } from '../../js/render/features/texelcache.js';
import { SPLIT_VIEWS } from '../../js/gpu/debugview.js';
import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

// Every gpu/ module's exports under one name, so the inputs below read as
// main.js's do without tracking which module each comes from.
const gpuDir = fileURLToPath(new URL('../../js/gpu/', import.meta.url));
const G = Object.assign({}, ...await Promise.all(readdirSync(gpuDir).filter(f => f.endsWith('.js'))
  .map(f => import(new URL(`../../js/gpu/${f}`, import.meta.url).href))));

const canvas = { style: {}, width: 1, height: 1, addEventListener() {}, removeEventListener() {},
                 getContext() { return null; } };
const renderer = new THREE.WebGPURenderer({ canvas });
// What the builder asks of the device: WebGPU's default limits, no optional
// features - fixed, so the output never depends on the machine.
renderer.backend.device = {
  limits: new Proxy({ maxUniformBufferBindingSize: 65536, maxStorageBufferBindingSize: 134217728,
                      maxComputeWorkgroupSizeX: 256, maxComputeInvocationsPerWorkgroup: 256 },
                    { get: (t, k) => (k in t ? t[k] : 1 << 16) }),
  features: new Set()
};
renderer._initialized = true;   // only the builder runs; nothing is drawn

function wgslOf(object, material = null) {
  const b = new THREE.WGSLNodeBuilder(material ? object : null, renderer);
  if (!material) b.compute = object;   // as three's Nodes.getForCompute
  if (material) {
    b.scene = scene; b.camera = camera; b.material = material;
  }
  b.build();
  return material ? { vertex: b.vertexShader, fragment: b.fragmentShader } : { compute: b.computeShader };
}

// Every compute node in a value - itself, or inside arrays and objects -
// named by its path and its own name.
function* computeNodes(v, path = '', depth = 0) {
  if (!v || depth > 3) return;
  if (v.isComputeNode) { yield [`${path}${v.name ? ':' + v.name : ''}`, v]; return; }
  if (v.isNode || typeof v !== 'object') return;
  for (const [k, x] of Object.entries(v)) yield* computeNodes(x, path ? `${path}.${k}` : k, depth + 1);
}

// --- The world and its meshes ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 500);
camera.position.set(20, 15, 40);
camera.lookAt(20, 0, 20);
const layers = () => new THREE.DataArrayTexture(new Uint8Array(12 * 12 * 4 * 4), 12, 12, 4);
// Set before initWorldRender, which then skips loading the images.
terrainTextures.albedo = layers();
terrainTextures.normal = layers();
terrainTextures.specular = layers();
terrainTextures.plain = layers();
createTestArea(40, 40);
initWorldRender(scene);

// --- The shading inputs, as main.js makes them ---
const sunDirection = new THREE.Vector3(0.4, 0.8, 0.3).normalize();
const cascades = [];
for (let l = 0; l < CASCADE_COUNT; l++) {
  const o = gridOriginFor('explore', { x: 20, y: 0, z: 20 }, sunDirection, 0, l);
  const grid = createBoxGridAt(o.x, o.z, null, l);
  cascades.push(G.cascadeBindings(G.createDistanceTexture(grid), grid));
}
const sunCards = G.createCardBindings(G.MAX_CARDS, 'bxbSunCards');
sunCards.lodShift.value = SUN_CARD_LOD_SHIFT;
sunCards.lodMax.value = SUN_CARD_LOD_MAX;
const lightCards = G.createCardBindings(POINT_CARD_CAPACITY, 'bxbLightCards');
const viewCards = G.createCardBindings(G.MAX_CARDS, 'bxbViewCards');
// The render features, made as main.js makes them; their options are what
// the shading build is given.
const sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
sunLight.position.copy(sunDirection);
const lighting = createLights({
  scene, shadowsOn: () => true, mirrorRects: () => [],
  holder: { sprite: new THREE.Object3D(), facingRight: () => false },
  cards: { active: () => false }
});
const mirrorFeature = createMirrors({
  sunLight, sunOn: () => true, lighting, focus: new THREE.Vector3(), footprint: () => null, rebuild() {}
});
const features = [createSky({ scene, sunLight, moveSun() {} }), lighting, mirrorFeature];
const sky = features[0].bindings;
const lights = lighting.bindings();
const mirrors = mirrorFeature.bindings;
const sunU = {
  bayerTex: G.createBayerTexture(), stepsUniform: uniform(4), biasUniform: uniform(G.SURFACE_BIAS_VOXELS),
  fadeStartUniform: uniform(G.SHADOW_FADE_START), edgeFadeUniform: uniform(G.EDGE_FADE_VOXELS),
  ambientUniform: uniform(G.DEFAULT_AMBIENT)
};
// GI: its compute dispatches caught from a stand-in renderer, as lists.
const dispatched = [];
const giFeature = createGI({
  renderer: { compute: k => dispatched.push([k].flat()) },
  cascades: () => cascades, lighting, sky: features[0], mirrors: mirrorFeature,
  sun: { uniforms: () => sunU, on: () => true, angular: () => SUN_ANGULAR_SIZE, light: sunLight },
  sprites: { casters: () => [], types: [] }, terrainAlbedo: () => null,
  shadowsOn: () => true, rebuild() {}
});
features.push(giFeature);
const debugView = createDebugView({ shadowsOn: () => true, shadows() {}, rebuild() {} });
features.push(debugView);
// Split shading off in its options: each set below brings its own (over.split).
const texelFeature = createTexelCache({
  renderer, scene, camera, rig: { distance: 20 }, shadowsOn: () => true, activeSet: () => null,
  gi: giFeature, rebuild() {}
});
texelFeature.splitOn = false;
features.push(texelFeature);
const rt = (w, h) => {
  const t = new THREE.RenderTarget(w, h, { type: THREE.FloatType, depthBuffer: true });
  t.texture.minFilter = t.texture.magFilter = THREE.NearestFilter;
  t.texture.generateMipmaps = false;
  return t;
};
const texelCache = {
  rt: rt(480, 270), tex: null, idRT: rt(1920, 1080),
  scale: uniform(new THREE.Vector2(0.25, 0.25)), size: uniform(new THREE.Vector2(480, 270)),
  showMisses: false
};
texelCache.tex = texelCache.rt.texture;

function options(over = {}) {
  return {
    cascades, sunDirection,
    rays: 1, angular: SUN_ANGULAR_SIZE, cone: true, quantise: false, ...sunU,
    shadeMode: 'lambert', sun: true,
    cards: sunCards, lightCards, terrain: terrainTextures,
    albedo: (st, surf = null) => mix(terrainTexelTSL(terrainTextures.albedo, st, surf).rgb,
                                     vec3(1, 1, 1), terrainWhite),
    specular: true, reflections: true,
    viewCards,
    ao: true, aoDistanceUniform: uniform(G.AO_DISTANCE),
    glassBounces: G.GLASS_MAX_BOUNCES, glassLayers: G.GLASS_LAYERS,
    glassDispersion: 0, glassShadows: true, sunMarch: true,
    modelHits: true, hitCache: null, glass: !!R.glassInstancedMesh,
    ...Object.assign({}, ...features.map(f => f.options())),
    ...over
  };
}

// --- One shader set, as obtainShadowSet builds it ---
function shaderSet(name, over) {
  const out = {};
  const opts = options(over);
  if (opts.split) { opts.split.rt = texelCache.tex; opts.split.idRT = texelCache.idRT.texture; }
  const built = G.createShadowColorNode(opts);
  const world = R.worldInstancedMesh, glass = R.glassInstancedMesh;
  const m = buildTerrainMaterials({ built, opts, cache: texelCache, cached: !!(opts.split || over.texelCache) });
  const mats = { low: m.lowMat, terrain: m.terrainMat, miss: m.missMat,
                 glassLow: m.glassLow, glass: m.glassMat, glassMiss: m.glassMiss };
  for (const k of Object.keys(mats)) if (!mats[k]) delete mats[k];
  for (const [k, m] of Object.entries(mats)) {
    out[`${name}/${k}`] = wgslOf(k.startsWith('glass') ? glass : world, m);
  }
  // A sprite, lit as the game's are.
  const base = new THREE.MeshBasicMaterial({ map: new THREE.DataTexture(new Uint8Array(16 * 16 * 4), 16, 16),
                                             transparent: true, alphaTest: 0.5 });
  const sprite = new THREE.Mesh(new THREE.PlaneGeometry(1, 2), base);
  out[`${name}/sprite`] = wgslOf(sprite, G.createSpriteShadowMaterial(base, built.spriteLight, null, null));
  // The compute kernels the set carries.
  for (const [k, c] of computeNodes({ split: built.splitKernels, gi: built.giKernels,
                                      mirror: built.mirrorCache,
                                      room: built.splitKernels ? opts.split.room.ease : null })) {
    out[`${name}/${k}`] = wgslOf(c);
  }
  return out;
}

// The bxb.parity and bxb.cardparity kernels, caught from a stand-in renderer:
// their WGSL depends only on the ray count, not the rays.
async function diagnosticKernels() {
  const caught = [];
  const stand = { computeAsync: async k => { caught.push(k); },
                  getArrayBufferAsync: async () => new ArrayBuffer(4) };
  const grid = createBoxGridAt(0, 0, null, 0);
  const rays = Array.from({ length: 256 }, () => ({ origin: { x: 1, y: 1, z: 1 },
                                                    dir: { x: 1, y: -1, z: 0.5 }, maxDist: 12 }));
  await G.runGPUMarch(stand, G.createDistanceTexture(grid), grid, rays);
  const samples = Array.from({ length: 128 }, () => ({ from: { x: 1, y: 1, z: 1 },
                                                       dir: { x: 0, y: 1, z: 0 }, maxDist: 12 }));
  await G.runGPUCards(stand, sunCards, samples, 0.01);
  return { 'diag/parity': wgslOf(caught[0]), 'diag/cardparity': wgslOf(caught[1]) };
}

export async function buildAllShaders() {
  const hitCache = G.createHitCache();
  const splitShading = () => {
    const s = G.createSplitShading(1 << 18, 1 << 17, 1 << 21, true);
    s.resolve = G.createSplitResolve(s, texelCache);
    return s;
  };
  // Each set with GI as main.js would have it: the LPV, texel GI, the LPV, off.
  const giSet = (mode, on = true) => {
    const settings = giFeature.settings.settings;
    settings.find(x => x.key === 'gimode').set(mode);
    settings.find(x => x.key === 'gi').set(on);
  };
  const sets = {};
  giSet('lpv');
  Object.assign(sets, shaderSet('default', { split: splitShading() }));
  giSet('texel');
  Object.assign(sets, shaderSet('texelgi', { split: splitShading() }));
  giSet('lpv');
  Object.assign(sets, shaderSet('cache', { texelCache: true, hitCache }));
  giSet('lpv', false);
  Object.assign(sets, shaderSet('plain', {}));
  // The LPV's own kernels: what one frame with both volumes resetting
  // dispatches (the far volume every frame, for this).
  giSet('lpv', true);
  giFeature.commands.gifar.run();
  giFeature.update();
  dispatched.forEach((list, i) => {
    [...new Set(list)].forEach((c, j) => { sets[`lpv${i}/${j}:${c.name}`] = wgslOf(c); });
  });
  Object.assign(sets, await diagnosticKernels());
  return sets;
}

// Every render debug view's set (render/features/debugview.js), built as
// main.js builds it - not snapshotted, only proved to build. The bounce views
// with texel GI under split shading, which is where they are traced; the rest
// per fragment, with the LPV.
export function buildViewShaders() {
  const giSet = mode => giFeature.settings.settings.find(x => x.key === 'gimode').set(mode);
  const out = {};
  for (const v of VIEWS.filter(x => x.view || x.glassView)) {
    debugView.commands.view.run(v.id);
    giSet('lpv');
    out[v.id] = shaderSet(`view-${v.id}`, {});
    if (SPLIT_VIEWS.has(v.view)) {
      giSet('texel');
      const split = G.createSplitShading(1 << 18, 1 << 17, 1 << 21, true);
      split.resolve = G.createSplitResolve(split, texelCache);
      Object.assign(out[v.id], shaderSet(`view-${v.id}-split`, { split }));
    }
  }
  debugView.commands.view.run('final');
  giSet('lpv');
  return out;
}

// bxb.gpuprofile's kernel flags turned off, each alone - built, not snapshotted,
// so a profiling run never meets a shader that fails to compile.
export function buildProfilingShaders() {
  return { 'sun march off': shaderSet('profile-sunmarch', { sunMarch: false }),
           'AO off': shaderSet('profile-ao', { ao: false }) };
}

// The basic mode's terrain (render.js plainTerrainMaterial, as initWorldRender
// dresses it) - built, not snapshotted.
export function buildPlainShaders() {
  return { 'plain terrain': wgslOf(R.worldInstancedMesh, R.worldInstancedMesh.material) };
}

// The exposure meter's kernels (gpu/exposure.js), over each source it reads -
// built, not snapshotted.
export function buildExposureShaders() {
  const meter = G.createExposureMeter();
  const split = G.createSplitShading(1 << 18, 1 << 17, 1 << 21, true);
  const out = {};
  for (const [name, source] of [['split', { split }], ['target', { cache: texelCache }]]) {
    meter.kernels(source).forEach((k, i) => { out[`exposure/${name}/${i}`] = wgslOf(k); });
  }
  // Our AgX (gpu/tonemap.js), on a plain material: proves the function builds.
  // Built twice, as the output pass is when the tone mapping is switched off
  // and on again: the second must still declare the look's uniforms.
  const agx = G.createAgXLook();
  for (const k of ['first', 'second']) {
    const mat = new THREE.MeshBasicNodeMaterial();
    mat.colorNode = agx.fn(vec3(0.5, 0.25, 0.1), uniform(1));
    out[`tonemap/agx-${k}`] = { compute: wgslOf(new THREE.Mesh(new THREE.PlaneGeometry(1, 1), mat), mat).fragment };
  }
  return out;
}
