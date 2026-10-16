// Every shader the engine builds, generated headlessly as WGSL.
//
// three's node builder needs no GPU to turn a material or a compute node into
// WGSL - only a renderer object, here one over a stub canvas that is never
// initialised. The inputs mirror main.js obtainShadowSet / shadowNodeOptions
// with the default settings, over the test area (glass, slopes, halves,
// stairs); they need not be realistic, only reach the same code. Used by
// shaders.test.mjs to prove a refactor leaves every shader byte-identical.
import * as THREE from 'three/webgpu';
import { uniform, mix, vec3 } from 'three/tsl';
import { createTestArea } from '../../js/core/world.js';
import { createBoxGridAt, gridOriginFor, CASCADE_COUNT } from '../../js/core/boxgrid.js';
import { MAX_LIGHTS, DEFAULT_LIGHT_CUTOFF, POINT_CARD_CAPACITY } from '../../js/core/lights.js';
import { MAX_MIRRORS } from '../../js/core/mirrors.js';
import { SUN_CARD_LOD_SHIFT, SUN_CARD_LOD_MAX } from '../../js/core/cards.js';
import { LPV_LEVELS, GI_MAX_BOUNCES } from '../../js/core/lpv.js';
import { SUN_ANGULAR_SIZE } from '../../js/core/sun.js';
import { terrainTextures, initWorldRender } from '../../js/render/render.js';
import * as R from '../../js/render/render.js';
import { terrainKeepTSL, terrainTexelTSL, terrainWhite } from '../../js/gpu/terrain.js';
import { createLPV } from '../../js/gpu/gi.js';
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
const lights = G.createLightBindings(MAX_LIGHTS);
lights.cutoff.value = DEFAULT_LIGHT_CUTOFF;
const sunCards = G.createCardBindings(G.MAX_CARDS, 'bxbSunCards');
sunCards.lodShift.value = SUN_CARD_LOD_SHIFT;
sunCards.lodMax.value = SUN_CARD_LOD_MAX;
const lightCards = G.createCardBindings(POINT_CARD_CAPACITY, 'bxbLightCards');
const viewCards = G.createCardBindings(G.MAX_CARDS, 'bxbViewCards');
const sky = G.createSkyBindings();
const mirrors = G.createMirrorBindings(MAX_MIRRORS);
const sunU = {
  bayerTex: G.createBayerTexture(), stepsUniform: uniform(4), biasUniform: uniform(G.SURFACE_BIAS_VOXELS),
  fadeStartUniform: uniform(G.SHADOW_FADE_START), edgeFadeUniform: uniform(G.EDGE_FADE_VOXELS),
  ambientUniform: uniform(G.DEFAULT_AMBIENT)
};
const giStrength = uniform(1);
const lpvs = [];
for (let level = 0; level < LPV_LEVELS; level++) {
  lpvs.push(createLPV({ cascades, lights, level, res: 1, orders: GI_MAX_BOUNCES,
                        biasUniform: sunU.biasUniform, fadeStartUniform: sunU.fadeStartUniform,
                        edgeFadeUniform: sunU.edgeFadeUniform, sky, ambientUniform: sunU.ambientUniform,
                        blocks: terrainTextures.blocks, glass: terrainTextures,
                        terrain: terrainTextures, mirrors }));
}
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
    shadeMode: 'lambert', shadowOnly: false, sun: true,
    lights, cards: sunCards, lightCards, terrain: terrainTextures, specularOnly: false,
    albedo: (st, surf = null) => mix(terrainTexelTSL(terrainTextures.albedo, st, surf).rgb,
                                     vec3(1, 1, 1), terrainWhite),
    specular: true, reflections: true, sky,
    mirrors, mirrorOnly: false, viewCards,
    ao: true, aoDistanceUniform: uniform(G.AO_DISTANCE), aoOnly: false,
    gi: null, giOnly: false, texelGI: null,
    glassView: null, glassBounces: G.GLASS_MAX_BOUNCES, glassLayers: G.GLASS_LAYERS,
    glassDispersion: 0, glassShadows: true,
    mirrorCache: true, modelHits: true, hitCache: null, split: null,
    splitResolve: true, splitLegacy: false, glass: !!R.glassInstancedMesh,
    ...over
  };
}

// --- One shader set, as obtainShadowSet builds it ---
function shaderSet(name, over) {
  const out = {};
  const opts = options(over);
  if (opts.split) { opts.split.rt = texelCache.tex; opts.split.idRT = texelCache.idRT.texture; }
  const built = G.createShadowColorNode(opts);
  const node = G.terrainEmissiveTSL(built.node, terrainTextures);
  const world = R.worldInstancedMesh, glass = R.glassInstancedMesh;
  const glassNode = glass ? built.glassNode : null;
  const mats = {};
  const split = built.splitKernels ? opts.split : null;
  if (opts.split || over.texelCache) {
    mats.low = new THREE.MeshBasicNodeMaterial();
    mats.low.outputNode = split ? G.createTexelGBufferNode(false, opts.splitLegacy)
      : G.createTexelCacheWriteNode(G.terrainEmissiveTSL(built.nodeCaching, terrainTextures));
    const resolved = split && opts.splitResolve && !opts.splitLegacy && (built.splitGlass || !glassNode);
    if (resolved) {
      mats.terrain = G.createShadowMaterial(world, G.createSplitReadNode(texelCache, split));
    } else {
      mats.terrain = G.createShadowMaterial(world, G.createTexelCacheLookupNode(texelCache, split));
      mats.miss = G.createShadowMaterial(world, G.createTexelMissNode(texelCache, node));
    }
    if (glassNode && resolved) {
      mats.glassLow = new THREE.MeshBasicNodeMaterial();
      mats.glassLow.outputNode = G.createTexelGBufferNode(true, opts.splitLegacy);
      mats.glass = G.createShadowMaterial(glass, G.createSplitReadNode(texelCache, split));
    } else if (glassNode) {
      mats.glassLow = new THREE.MeshBasicNodeMaterial();
      mats.glassLow.outputNode = built.splitGlass ? G.createTexelGBufferNode(true)
        : G.createTexelCacheWriteNode(glassNode, !!split);
      mats.glass = G.createShadowMaterial(glass, G.createTexelCacheLookupNode(texelCache, split));
      mats.glassMiss = G.createShadowMaterial(glass, G.createTexelMissNode(texelCache, glassNode));
    }
  } else {
    mats.terrain = G.createShadowMaterial(world, node);
    if (glassNode) mats.glass = G.createShadowMaterial(glass, glassNode);
  }
  for (const k of ['terrain', 'low', 'miss']) if (mats[k]) mats[k].positionNode = terrainKeepTSL(false);
  for (const k of ['glass', 'glassLow', 'glassMiss']) if (mats[k]) mats[k].positionNode = terrainKeepTSL(true);
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
                                      mirror: built.mirrorCache })) {
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
  const sets = {
    ...shaderSet('default', { split: splitShading(), gi: { levels: lpvs.map(v => v.binding), strength: giStrength } }),
    ...shaderSet('texelgi', { split: splitShading(), texelGI: G.createTexelGI() }),
    ...shaderSet('cache', { texelCache: true, hitCache, gi: { levels: lpvs.map(v => v.binding), strength: giStrength } }),
    ...shaderSet('plain', {})
  };
  // The LPV's own kernels: what one reset frame dispatches, caught from a
  // stand-in renderer.
  for (const [i, v] of lpvs.entries()) {
    const list = [];
    v.update({ compute: k => list.push(...[k].flat()) }, { shift: 'reset', iterations: 2 });
    [...new Set(list)].forEach((c, j) => { sets[`lpv${i}/${j}:${c.name}`] = wgslOf(c); });
  }
  Object.assign(sets, await diagnosticKernels());
  return sets;
}
