import * as THREE from 'three/webgpu';
import { attribute, texture, uv, int, mix, vec3, uniform } from 'three/tsl';
import { World, BLOCK_METRES } from './world.js';
import { MATERIALS, BLOCK_TEXELS, DEFAULT_SPECULAR, materialLayer } from './materials.js';
import { terrainNormals } from './normals.js';

// Every block texture, by URL, so Vite bundles them - albedo, _n and _s alike.
const textureUrls = import.meta.glob('../assets/textures/*.png',
                                     { eager: true, query: '?url', import: 'default' });
function textureUrl(name) {
  const hit = Object.keys(textureUrls).find(k => k.endsWith('/' + name + '.png'));
  return hit ? textureUrls[hit] : null;
}

export const VOXEL_SIZE = BLOCK_METRES; // single source of truth is world.js

export let worldInstancedMesh = null;
export const voxelIndexMap = new Map(); // grid key -> instance ID
const instanceKeyByIndex = [];          // instance ID -> grid key (raycast picking)

export function keyForInstance(instanceId) {
  return instanceKeyByIndex[instanceId];
}

// --- Terrain textures, one array layer per material ---
//
// Three DataArrayTextures, stacked in materials.js order: albedo (sRGB), normal
// and specular (both LabPBR, both data). One layer per material, every layer
// BLOCK_TEXELS square, so a block picks its layer by the per-instance
// blockLayer attribute and the uv needs no remapping - its derivatives, which
// the texel lock and the normal-map frame both take, stay those of one face.
//
// They exist from the start, filled with placeholders (mid grey, flat, rough),
// and each layer is overwritten as its images arrive. So a material binds them
// once and never waits on a load, and a slow texture is a grey block for a
// moment rather than a rebuild.
//
// Rows are stored BOTTOM-UP (row 0 is v = 0), the orientation normals.js writes
// and a loaded image ends up in on the GPU.
export const terrainTextures = {
  albedo: null, normal: null, specular: null,
  // Source (sRGB) RGBA per layer, top-down, once loaded - what GI averages.
  rgba: [],
  // The same for each layer's _s, or undefined where it has none.
  specRgba: [],
  // Which material each BLOCK is, as a 3D texture over the world's block
  // bounds: layer + 1, 0 for air. What a reflection ray reads to find the
  // texture of the block it hit. { tex, origin, size } - origin and size in
  // blocks, as vec3 uniforms. Built with the terrain mesh.
  blocks: null,
  ready: false
};

// 1 draws the terrain plain white (whiteworld) - a uniform, so both the plain
// and the shadowed materials follow it with no rebuild.
export const terrainWhite = uniform(0);

// The block's material layer, in the fragment: a per-instance attribute.
export const terrainLayerTSL = () => int(attribute('blockLayer', 'float').add(0.5));
// A terrain array sampled at this fragment's own layer - at st when given
// (the shaded path passes its texel's centre), else the mesh uv.
export const terrainSampleTSL = (tex, st = null) =>
  texture(tex, st || uv()).depth(terrainLayerTSL());

function makeArray(fill, colorSpace) {
  const n = BLOCK_TEXELS, layers = MATERIALS.length;
  const data = new Uint8Array(n * n * 4 * layers);
  for (let i = 0; i < n * n * layers; i++) data.set(fill, i * 4);
  const tex = new THREE.DataArrayTexture(data, n, n, layers);
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = false;
  tex.colorSpace = colorSpace;
  tex.needsUpdate = true;
  return tex;
}

// An image's pixels, top-down, at BLOCK_TEXELS square. A texture authored at
// another size is resampled nearest and warned about - the §4 lock is one
// texel per voxel, so it is a content error, but not one worth a black block.
function imagePixels(img, name) {
  const n = BLOCK_TEXELS;
  if (img.width !== n || img.height !== n) {
    console.warn(`[terrain] ${name} is ${img.width}x${img.height}, not ${n}x${n} - resampled`);
  }
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = n;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0, n, n);
  return ctx.getImageData(0, 0, n, n).data;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

// Top-down pixels into one layer of an array, flipped to bottom-up.
function writeLayer(tex, layer, topDown) {
  const n = BLOCK_TEXELS, row = n * 4;
  const base = layer * n * row;
  for (let y = 0; y < n; y++) {
    tex.image.data.set(topDown.subarray(y * row, (y + 1) * row), base + (n - 1 - y) * row);
  }
  tex.needsUpdate = true;
}

// One byte per block over the World's bounding box. Tiny - the test map is
// 36 x 4 x 36 - and rebuilt whenever the terrain mesh is.
function buildBlockVolume() {
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const blocks = [];
  for (const [key, block] of World.entries()) {
    const c = key.split(',').map(Number);
    blocks.push([c, block]);
    for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a], c[a]); hi[a] = Math.max(hi[a], c[a]); }
  }
  const size = lo.map((l, a) => hi[a] - l + 1);
  const data = new Uint8Array(size[0] * size[1] * size[2]);
  for (const [c, block] of blocks) {
    const i = ((c[2] - lo[2]) * size[1] + (c[1] - lo[1])) * size[0] + (c[0] - lo[0]);
    data[i] = materialLayer(block.materialId) + 1;
  }
  const v = terrainTextures.blocks;
  if (v && v.tex.image.width === size[0] && v.tex.image.height === size[1] &&
      v.tex.image.depth === size[2]) {
    v.tex.image.data.set(data);
    v.tex.needsUpdate = true;
    v.origin.value.set(...lo);
    return;
  }
  // A new size means a new texture, and any material built on the old one
  // has to rebuild - the terrain mesh rebuilding does that anyway.
  const tex = new THREE.Data3DTexture(data, size[0], size[1], size[2]);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.generateMipmaps = false;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  terrainTextures.blocks = {
    tex,
    origin: uniform(new THREE.Vector3(...lo)),
    size: uniform(new THREE.Vector3(...size))
  };
}

function initTerrainTextures() {
  if (terrainTextures.albedo) return;
  terrainTextures.albedo = makeArray([128, 128, 128, 255], THREE.SRGBColorSpace);
  terrainTextures.normal = makeArray([128, 128, 255, 255], THREE.NoColorSpace);
  terrainTextures.specular = makeArray(DEFAULT_SPECULAR, THREE.NoColorSpace);
  const loads = MATERIALS.map(async (m, layer) => {
    const albedoUrl = textureUrl(m.texture);
    if (!albedoUrl) { console.warn(`[terrain] no texture ${m.texture}.png`); return; }
    const rgba = imagePixels(await loadImage(albedoUrl), m.texture);
    terrainTextures.rgba[layer] = rgba;
    writeLayer(terrainTextures.albedo, layer, rgba);
    // Normal: authored _n, else generated from the albedo (already bottom-up).
    const nUrl = textureUrl(m.texture + '_n');
    if (nUrl) writeLayer(terrainTextures.normal, layer,
                         imagePixels(await loadImage(nUrl), m.texture + '_n'));
    else {
      const n = BLOCK_TEXELS;
      terrainTextures.normal.image.data.set(terrainNormals(rgba, n, n), layer * n * n * 4);
      terrainTextures.normal.needsUpdate = true;
    }
    // Specular: authored _s, else the placeholder already is DEFAULT_SPECULAR.
    const sUrl = textureUrl(m.texture + '_s');
    if (sUrl) {
      const spec = imagePixels(await loadImage(sUrl), m.texture + '_s');
      terrainTextures.specRgba[layer] = spec;
      writeLayer(terrainTextures.specular, layer, spec);
    }
  });
  Promise.allSettled(loads).then(r => {
    for (const x of r) if (x.status === 'rejected') console.warn('[terrain] texture load failed', x.reason);
    terrainTextures.ready = true;
  });
}

export function initWorldRender(scene) {
  voxelIndexMap.clear();
  instanceKeyByIndex.length = 0;
  // Every SOLID block renders, not just standable ones: walls, pillars and the
  // underground layers are all real geometry that the lighting work needs to
  // occlude with. Standability is a derived query in world.js, not a render gate.
  const renderCount = World.size;
  if (renderCount === 0) return;

  initTerrainTextures();
  buildBlockVolume();

  const geometry = new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
  // Which material each block draws with - its layer in the texture arrays.
  const layers = new THREE.InstancedBufferAttribute(new Float32Array(renderCount), 1);
  geometry.setAttribute('blockLayer', layers);
  // The unshadowed path. Standard lighting, but the colour is the block's own
  // layer - a plain `map` would put layer 0 on everything.
  const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.9, metalness: 0.0 });
  material.colorNode = mix(terrainSampleTSL(terrainTextures.albedo).rgb, vec3(1, 1, 1), terrainWhite);

  worldInstancedMesh = new THREE.InstancedMesh(geometry, material, renderCount);
  worldInstancedMesh.receiveShadow = true;
  worldInstancedMesh.castShadow = true;

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  let instanceIndex = 0;

  for (const key of World.keys()) {
    const [gx, gy, gz] = key.split(',').map(Number);
    position.set(gx * VOXEL_SIZE, gy * VOXEL_SIZE, gz * VOXEL_SIZE);
    matrix.setPosition(position);
    worldInstancedMesh.setMatrixAt(instanceIndex, matrix);

    // Both directions are needed: key -> index for tinting and visibility,
    // index -> key so a raycast hit (which reports only an instanceId) can be
    // resolved back to the block that was clicked.
    voxelIndexMap.set(key, instanceIndex);
    instanceKeyByIndex[instanceIndex] = key;
    worldInstancedMesh.setColorAt(instanceIndex, new THREE.Color(0xffffff));
    layers.setX(instanceIndex, materialLayer(World.get(key).materialId));
    instanceIndex++;
  }
  worldInstancedMesh.instanceMatrix.needsUpdate = true;
  if (worldInstancedMesh.instanceColor) worldInstancedMesh.instanceColor.needsUpdate = true;
  scene.add(worldInstancedMesh);
}

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _scaleHidden = new THREE.Vector3(0, 0, 0);
const _scaleVisible = new THREE.Vector3(1, 1, 1);
const _rotation = new THREE.Quaternion();

export function updateVoxelVisibility(arenaMap, isBattle) {
  if (!worldInstancedMesh) return;
  
  const _matrix = new THREE.Matrix4();
  const _position = new THREE.Vector3();
  const _scale = new THREE.Vector3();
  const _quaternion = new THREE.Quaternion(); // Identity (0 rotation)

  for (const [key, idx] of voxelIndexMap.entries()) {
    // Calculate exact position from scratch based on grid coordinates
    const [gx, gy, gz] = key.split(',').map(Number);
    _position.set(gx * VOXEL_SIZE, gy * VOXEL_SIZE, gz * VOXEL_SIZE);
    
    if (isBattle && (!arenaMap || !arenaMap.has(key))) {
      _scale.set(0, 0, 0); // Hide
    } else {
      _scale.set(1, 1, 1); // Show
    }
    
    _matrix.compose(_position, _quaternion, _scale);
    worldInstancedMesh.setMatrixAt(idx, _matrix);
  }
  worldInstancedMesh.instanceMatrix.needsUpdate = true;
}

export function updateVoxelTints(arenaMap, reachableMap, isBattle) {
  if (!worldInstancedMesh) return;
  const colorWhite = new THREE.Color(0xffffff);
  const colorGrey = new THREE.Color(0x333333); 
  
  for (const [key, idx] of voxelIndexMap.entries()) {
    if (!isBattle) {
      worldInstancedMesh.setColorAt(idx, colorWhite);
    } else if (arenaMap.has(key)) {
      if (reachableMap && reachableMap.has(key)) {
        worldInstancedMesh.setColorAt(idx, colorWhite);
      } else {
        worldInstancedMesh.setColorAt(idx, colorGrey);
      }
    }
  }
  worldInstancedMesh.instanceColor.needsUpdate = true;
}