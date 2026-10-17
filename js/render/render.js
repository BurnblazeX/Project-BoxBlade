import * as THREE from 'three/webgpu';
import { attribute, texture, uv, mix, vec3, vec4, uniform, select, Fn, floor, normalize, cameraViewMatrix, storage } from 'three/tsl';
import { terrainWhite, terrainKeepTSL, terrainShapeTSL, SLOPE_ROWS, HALF_SLOPE_ROWS, terrainSurfaceTSL, MODEL_ATTRS, modelTexelTSL, terrainTexelTSL } from '../gpu/terrain.js';
import { applyOcclusionMask } from '../gpu/occlusion.js';
import { World, BLOCK_METRES, blockBoxes, blockSpan, isSlope, blockShape, blockFacing, getVoxelKey, shapeCode,
         isDual, blockParts,
         WorldModels, Y_MIN, CHUNK_HEIGHT, exposedFaceMask, opaqueBoxes } from '../core/world.js';
import { MODELS, TERRAIN_LAYERS } from '../core/models.js';
import { lightColourOf, modelInstanceData, buildModelHits, modelFaceMasks, MODEL_HIT_VEC4S, MODEL_HIT_RECORD } from '../core/bbmodel.js';
import { VOXEL_METRES } from '../core/boxgrid.js';
import { MATERIALS, BLOCK_TEXELS, DEFAULT_SPECULAR, materialLayer, isGlassMaterial,
         MATERIAL_LAYERS, SLOPE_SUFFIXES } from '../core/materials.js';
import { terrainNormals } from '../core/normals.js';

// Every block texture, by URL, so Vite bundles them - albedo, _n and _s alike.
// Only under Vite; in node (the shader snapshot test) there are none.
let textureUrls = {};
try {
  textureUrls = import.meta.glob('../../assets/textures/*.png',
                                 { eager: true, query: '?url', import: 'default' });
} catch { /* not under Vite */ }
export function textureUrl(name) {
  const hit = Object.keys(textureUrls).find(k => k.endsWith('/' + name + '.png'));
  return hit ? textureUrls[hit] : null;
}

export const VOXEL_SIZE = BLOCK_METRES; // single source of truth is world.js

export let worldInstancedMesh = null;
// Glass blocks, drawn by a mesh of their own: it shares the terrain's geometry
// and instance buffers (so picking, arena hiding and tinting need nothing
// new), and each material keeps only its own blocks - see terrainKeepTSL.
// null when the world holds no glass, so a map without any pays nothing.
export let glassInstancedMesh = null;
export const voxelIndexMap = new Map(); // grid key -> its instance IDs (a slope has several)
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
  // The shaded path's albedo (materials.js, _a): the same layout as albedo's
  // data, filled only at flatLayers; flatRgba as rgba, per material with one.
  // setShadedAlbedo swaps them in.
  flat: null, flatLayers: new Set(), flatRgba: [],
  // What each BLOCK is, as a 3D texture over a fixed box of blocks: what a
  // ray that hit something reads to find its texture (buildBlockVolume).
  // { tex, origin, size } - origin and size in blocks, as vec3 uniforms.
  blocks: null,
  // Placed models' cubes, by cell, for the rays (buildBlockVolume): a
  // read-only storage buffer of vec4. { attr, node }.
  modelHits: null,
  ready: false
};

// --- Slopes, as drawn ---
//
// A slope is SEEN as a plane (world.js BLOCK_SHAPES; the field holds its
// steps). It is one instance of the terrain box over the slope's span, with
// the top face's downhill edge dropped by the rise, so the top face IS the
// plane; the downhill face collapses (or, on a high half slope, is left at
// half height) and the sides become its profile. Per instance, blockSlope:
// 0 for any other block, else kind * 4 + facing - kind 1 slope, 2 half
// slope, 3 high half slope; facing 0 N, 1 E, 2 S, 3 W (uphill).
const SLOPE_KIND = { slope: 1, halfSlope: 2, halfSlopeHigh: 3 };
const FACING_INDEX = { N: 0, E: 1, S: 2, W: 3 };
export function slopeCode(block) {
  if (!isSlope(block)) return 0;
  return SLOPE_KIND[blockShape(block)] * 4 + FACING_INDEX[blockFacing(block)];
}

// The instances a block draws: its boxes (one, unless it is a slope), or for
// a slope one box over its span, which terrainShapeTSL makes the plane.
export function drawnBoxes(block) {
  if (!isSlope(block)) return blockBoxes(block);
  const [lo, hi] = blockSpan(block);
  return [[0, 1, lo, hi, 0, 1]];
}
// Per material part (world.js blockParts): a dual block draws each half with
// its own material.
function drawnParts(block) {
  if (!isDual(block)) return [{ materialId: block.materialId, boxes: drawnBoxes(block) }];
  return blockParts(block);
}


// The instance matrix of one box a block draws (drawnBoxes, cell fractions
// [x0, x1, y0, y1, z0, z1]): the unit box scaled and moved onto it. A full
// block is the whole cell, a half block half of it, a slope its span.
// Axis-aligned scales only. Scale 0 hides it (battle).
const _pos = new THREE.Vector3(), _scl = new THREE.Vector3(), _rot = new THREE.Quaternion();
export function boxMatrix(out, key, box, visible = true) {
  const [gx, gy, gz] = key.split(',').map(Number);
  const [x0, x1, y0, y1, z0, z1] = box;
  _pos.set((gx + (x0 + x1) / 2 - 0.5) * VOXEL_SIZE, (gy + (y0 + y1) / 2 - 0.5) * VOXEL_SIZE,
           (gz + (z0 + z1) / 2 - 0.5) * VOXEL_SIZE);
  if (visible) _scl.set(x1 - x0, y1 - y0, z1 - z0); else _scl.set(0, 0, 0);
  return out.compose(_pos, _rot, _scl);
}

function makeArray(fill, colorSpace, h = BLOCK_TEXELS, layers = TERRAIN_LAYERS) {
  const n = BLOCK_TEXELS;
  const data = new Uint8Array(n * h * 4 * layers);
  for (let i = 0; i < n * h * layers; i++) data.set(fill, i * 4);
  const tex = new THREE.DataArrayTexture(data, n, h, layers);
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
function imagePixels(img, name, h = BLOCK_TEXELS) {
  const n = BLOCK_TEXELS;
  if (img.width !== n || img.height !== h) {
    console.warn(`[terrain] ${name} is ${img.width}x${img.height}, not ${n}x${h} - resampled`);
  }
  const canvas = document.createElement('canvas');
  canvas.width = n;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0, n, h);
  return ctx.getImageData(0, 0, n, h).data;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

// Top-down pixels into one layer of an array, flipped to bottom-up. rows:
// the image's height; the layer's own is the array's.
function writeLayer(tex, layer, topDown, rows = BLOCK_TEXELS) {
  const row = BLOCK_TEXELS * 4;
  const base = layer * tex.image.height * row;
  for (let y = 0; y < rows; y++) {
    tex.image.data.set(topDown.subarray(y * row, (y + 1) * row), base + (rows - 1 - y) * row);
  }
  tex.needsUpdate = true;
}

// Top-down image pixels (12 wide, `rows` tall) flipped to bottom-up.
function bottomUp(topDown, rows) {
  const row = BLOCK_TEXELS * 4, out = new Uint8ClampedArray(rows * row);
  for (let y = 0; y < rows; y++) out.set(topDown.subarray(y * row, (y + 1) * row), (rows - 1 - y) * row);
  return out;
}

// A block layer (bottom-up, 12 x 12) stretched nearest onto `rows` rows,
// bottom-up - a missing slope map's fallback: the foot at row 0.
function stretched(src, srcLayer, rows) {
  const row = BLOCK_TEXELS * 4, from = srcLayer * BLOCK_TEXELS * row;
  const out = new Uint8ClampedArray(rows * row);
  for (let y = 0; y < rows; y++) {
    const sy = Math.floor(y * BLOCK_TEXELS / rows);
    out.set(src.image.data.subarray(from + sy * row, from + (sy + 1) * row), y * row);
  }
  return out;
}

// Bottom-up rows of a slope texture into its two layers: rows 0-11 at
// `first`, the rest from row 0 of first + 1 (terrainSurfaceTSL reads it so).
function writeSplit(tex, first, data, rows) {
  const row = BLOCK_TEXELS * 4, layerBytes = BLOCK_TEXELS * row;
  for (let y = 0; y < rows; y++) {
    const to = (first + Math.floor(y / BLOCK_TEXELS)) * layerBytes + (y % BLOCK_TEXELS) * row;
    tex.image.data.set(data.subarray(y * row, (y + 1) * row), to);
  }
  tex.needsUpdate = true;
}

// One material's slope planes that are authored (materials.js
// MATERIAL_LAYERS), once its block textures are in: each map authored, or -
// for a normal map, generated from an authored albedo - else the side's own
// stretched. A plane with none of the three is not allocated; the shader
// stretches the side itself.
async function loadSlopeLayers(m, layer) {
  const t = terrainTextures, flatSide = t.flatLayers.has(layer);
  const rowsOf = [SLOPE_ROWS, HALF_SLOPE_ROWS];
  const firsts = MATERIAL_LAYERS.faces[layer].slope;
  for (let kind = 0; kind < 2; kind++) {
    const first = firsts[kind], rows = rowsOf[kind];
    if (first < 0) continue;
    const name = m.texture + SLOPE_SUFFIXES[kind];
    const read = async suffix => {
      const url = textureUrl(name + suffix);
      return url ? bottomUp(imagePixels(await loadImage(url), name + suffix, rows), rows) : null;
    };
    const aUrl = textureUrl(name);
    const albedoTD = aUrl ? imagePixels(await loadImage(aUrl), name, rows) : null;
    writeSplit(t.albedo, first, albedoTD ? bottomUp(albedoTD, rows) : stretched(t.albedo, layer, rows), rows);
    const normal = await read('_n');
    writeSplit(t.normal, first, normal || (albedoTD ? terrainNormals(albedoTD, BLOCK_TEXELS, rows)
                                                    : stretched(t.normal, layer, rows)), rows);
    writeSplit(t.specular, first, (await read('_s')) || stretched(t.specular, layer, rows), rows);
    // Shaded albedo: its own _a, else - a plane drawn with the side's texture
    // stretched - the side's _a stretched.
    const flat = m.glass ? null : (await read('_a')) || (!albedoTD && flatSide ? stretched(t.flat, layer, rows) : null);
    if (flat) { writeSplit(t.flat, first, flat, rows); t.flatLayers.add(first).add(first + 1); }
  }
}

// A material's top and bottom faces, when it has them (materials.js
// MATERIAL_LAYERS): albedo, then _n (else generated from it) and _s (else
// the side's), as a block's.
async function loadFaceLayers(m, layer) {
  const t = terrainTextures, n = BLOCK_TEXELS;
  const f = MATERIAL_LAYERS.faces[layer];
  for (const [suffix, dst] of [['_top', f.top], ['_bottom', f.bottom]]) {
    if (dst === layer || (suffix === '_bottom' && dst === f.top)) continue;
    const name = m.texture + suffix;
    const aUrl = textureUrl(name), nUrl = textureUrl(name + '_n'), sUrl = textureUrl(name + '_s');
    const rgba = aUrl ? imagePixels(await loadImage(aUrl), name) : null;
    if (rgba) writeLayer(t.albedo, dst, rgba);
    else writeSplit(t.albedo, dst, stretched(t.albedo, layer, n), n);
    if (nUrl) writeLayer(t.normal, dst, imagePixels(await loadImage(nUrl), name + '_n'));
    else writeSplit(t.normal, dst, rgba ? terrainNormals(rgba, n, n) : stretched(t.normal, layer, n), n);
    if (sUrl) writeLayer(t.specular, dst, imagePixels(await loadImage(sUrl), name + '_s'));
    else writeSplit(t.specular, dst, stretched(t.specular, layer, n), n);
    // Shaded albedo: its own _a, else the side's _a when it took the side's.
    const fUrl = m.glass ? null : textureUrl(name + '_a');
    if (fUrl) writeLayer(t.flat, dst, imagePixels(await loadImage(fUrl), name + '_a'));
    else if (rgba || m.glass || !t.flatLayers.has(layer)) continue;
    else writeSplit(t.flat, dst, stretched(t.flat, layer, n), n);
    t.flatLayers.add(dst);
  }
}

// What each block is, for the rays, over a FIXED volume: an edit (the
// editor) rewrites it in place, where a volume sized to the world's bounds
// changed size - and every material built on the old texture had to be
// rebuilt. 256 x 12 x 256 blocks around the origin, RGBA8, 3 MB; a block
// outside it reads as air to the rays that ask (reflection and glass hits,
// the GI). Per block:
//   a block       R layer + 1 (its side), G its top layer, B its bottom, A its
//                 shape code (world.js shapeCode: a slope's is slopeCode's,
//                 so the split kernels read it as before; a half's or a
//                 pane's gives the box the glass walk steps to);
//   a model cell  R 0 (air to the walk through glass, which ignores models),
//                 G + 256 B its cell in modelHits, A the model's GI entry + 1
//                 (models.js giIndex);
//   air           all 0.
// A dual block (world.js isDual) adds 128 to A, and its other half's R, G, B
// sit BLOCK_VOLUME_SIZE[2] further along z, in the second half of the
// texture (dims) - one binding for both (gpu/volume.js volumeAtTSL).
// modelHits: the model cubes by cell (bbmodel.js buildModelHits) - what
// gpu/volume.js hitSurfaceTSL needs to find a hit's texel as the cube's own draw does.
export const BLOCK_VOLUME_ORIGIN = [-96, Y_MIN, -96];
export const BLOCK_VOLUME_SIZE = [256, CHUNK_HEIGHT, 256];
function buildBlockVolume() {
  const lo = BLOCK_VOLUME_ORIGIN, size = BLOCK_VOLUME_SIZE;
  let v = terrainTextures.blocks;
  if (!v) {
    const tex = new THREE.Data3DTexture(new Uint8Array(size[0] * size[1] * size[2] * 8), size[0], size[1], size[2] * 2);
    tex.format = THREE.RGBAFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.generateMipmaps = false;
    tex.unpackAlignment = 1;
    v = terrainTextures.blocks = {
      tex,
      origin: uniform(new THREE.Vector3(...lo)),
      size: uniform(new THREE.Vector3(...size)),
      dims: uniform(new THREE.Vector3(size[0], size[1], size[2] * 2))   // the texture's own
    };
    const attr = new THREE.StorageBufferAttribute(new Float32Array(MODEL_HIT_VEC4S * 4), 4);
    terrainTextures.modelHits = {
      attr, node: storage(attr, 'vec4', MODEL_HIT_VEC4S).toReadOnly().setName('bxbModelHits')
    };
  }
  const data = v.tex.image.data;
  data.fill(0);
  const at = c => {
    const r = c.map((x, a) => x - lo[a]);
    return r.some((x, a) => x < 0 || x >= size[a]) ? -1 : ((r[2] * size[1] + r[1]) * size[0] + r[0]) * 4;
  };
  for (const [key, block] of World.entries()) {
    const o = at(key.split(',').map(Number));
    if (o < 0) continue;
    const layer = materialLayer(block.materialId), f = MATERIAL_LAYERS.faces[layer];
    data[o] = layer + 1;
    data[o + 1] = f ? f.top : layer;
    data[o + 2] = f ? f.bottom : layer;
    data[o + 3] = shapeCode(block);   // its shape and facing (world.js shapeCode)
    if (isDual(block)) {
      data[o + 3] += 128;
      const o2 = o + size[0] * size[1] * size[2] * 4;
      const layer2 = materialLayer(block.second), f2 = MATERIAL_LAYERS.faces[layer2];
      data[o2] = layer2 + 1;
      data[o2 + 1] = f2 ? f2.top : layer2;
      data[o2 + 2] = f2 ? f2.bottom : layer2;
    }
  }
  const hits = buildModelHits(WorldModels, key => World.has(key));
  if (hits.overflow) console.warn(`[terrain] model hit table full (${MODEL_HIT_VEC4S} vec4): some model cubes reflect as air`);
  for (const [key, { header, giIndex }] of hits.index) {
    const o = at(key.split(',').map(Number));
    if (o < 0 || header > 65535) continue;
    data[o + 1] = header & 255;
    data[o + 2] = header >> 8;
    data[o + 3] = giIndex + 1;
  }
  const mh = terrainTextures.modelHits;
  mh.attr.array.set(hits.data);
  mh.attr.needsUpdate = true;
  v.tex.needsUpdate = true;
}

function initTerrainTextures() {
  if (terrainTextures.albedo) return;
  terrainTextures.albedo = makeArray([128, 128, 128, 255], THREE.SRGBColorSpace);
  terrainTextures.normal = makeArray([128, 128, 255, 255], THREE.NoColorSpace);
  terrainTextures.specular = makeArray(DEFAULT_SPECULAR, THREE.NoColorSpace);
  // CPU only, never bound: the writers above need just image.data and height.
  terrainTextures.flat = { image: { data: new Uint8Array(terrainTextures.albedo.image.data.length),
                                    height: BLOCK_TEXELS } };
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
    const aUrl = m.glass ? null : textureUrl(m.texture + '_a');
    if (aUrl) {
      const flat = imagePixels(await loadImage(aUrl), m.texture + '_a');
      terrainTextures.flatRgba[layer] = flat;
      writeLayer(terrainTextures.flat, layer, flat);
      terrainTextures.flatLayers.add(layer);
    }
    await loadFaceLayers(m, layer);
    await loadSlopeLayers(m, layer);
  });
  for (const model of MODELS.values()) loads.push(loadModelTiles(model));
  Promise.allSettled(loads).then(r => {
    for (const x of r) if (x.status === 'rejected') console.warn('[terrain] texture load failed', x.reason);
    terrainTextures.ready = true;
    applyAlbedoMode();
  });
}

// --- Shaded albedo ---
//
// With shadows on, every layer with an _a (materials.js) draws it in place of
// its main texture - per texel, so a metal texel (_s G >= 230) keeps its own.
// Swapped in the albedo array's data, not chosen in the shaders: every reader
// of the array (the view, reflections, glass, the split kernels) follows with
// no rebuild. Before the loads are in it is only remembered; they apply it.
let shadedAlbedo = false, mainAlbedo = null;
export function setShadedAlbedo(on) {
  if (shadedAlbedo === on) return;
  shadedAlbedo = on;
  if (terrainTextures.ready) applyAlbedoMode();
}
function applyAlbedoMode() {
  const t = terrainTextures, d = t.albedo.image.data, flat = t.flat.image.data;
  const spec = t.specular.image.data, size = BLOCK_TEXELS * BLOCK_TEXELS * 4;
  if (!t.flatLayers.size) return;
  // The main textures, kept the first time - the array holds them until now.
  mainAlbedo ??= new Map([...t.flatLayers].map(l => [l, d.slice(l * size, (l + 1) * size)]));
  for (const [l, main] of mainAlbedo) {
    const o = l * size;
    d.set(main, o);
    if (!shadedAlbedo) continue;
    for (let i = 0; i < size; i += 4) {
      if (spec[o + i + 1] < 230) d.set(flat.subarray(o + i, o + i + 4), o + i);
    }
  }
  t.albedo.needsUpdate = true;
}

// A model's textures (models.js), cut into 12 x 12 tiles in its layers of the
// terrain arrays - row by row, tilesX across. A map it has no companion for
// keeps the arrays' placeholder: flat for normals, DEFAULT_SPECULAR.
async function loadModelTiles(model) {
  // The emissive texture goes in the ALBEDO array, at its own layers: a
  // lightSource cube reads those in place of its albedo (modelInstanceData).
  const maps = [[model.textures.albedo, terrainTextures.albedo, model.modelLayer, 'albedo'],
                [model.textures.normal, terrainTextures.normal, model.modelLayer, 'normal'],
                [model.textures.specular, terrainTextures.specular, model.modelLayer, 'specular'],
                [model.textures.emissive, terrainTextures.albedo, model.emissiveLayer, 'emissive']];
  const n = BLOCK_TEXELS;
  const pixels = {};
  for (const [url, tex, first, kind] of maps) {
    if (!url) continue;
    const img = await loadImage(url);
    const w = model.tilesX * n, h = model.tilesY * n;
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const all = ctx.getImageData(0, 0, w, h).data;
    pixels[kind] = { all, w };
    for (let ty = 0; ty < model.tilesY; ty++) for (let tx = 0; tx < model.tilesX; tx++) {
      const tile = new Uint8ClampedArray(n * n * 4);
      for (let y = 0; y < n; y++) {
        const from = ((ty * n + y) * w + tx * n) * 4;
        tile.set(all.subarray(from, from + n * 4), y * n * 4);
      }
      writeLayer(tex, first + ty * model.tilesX + tx, tile);
    }
  }
  // What a probe landing on the model bounces (main.js resolveLayerAlbedos).
  if (pixels.albedo) model.albedoPixels = pixels.albedo.all;
  const src = pixels.emissive || pixels.albedo;
  if (src) model.lightColour = lightColourOf(model, src.all, src.w);
}



// What each instance is drawn over: { box } (a block's box, cell fractions)
// or { texels } (a model cube, world texels) - for rebuilding its matrix.
const instanceSource = [];
const TEXEL_METRES = BLOCK_METRES / BLOCK_TEXELS;
function instanceMatrix(out, idx, key, visible = true) {
  const src = instanceSource[idx];
  if (src.box) return boxMatrix(out, key, src.box, visible);
  const { min, max } = src.texels;
  _pos.set((min[0] + max[0]) / 2 * TEXEL_METRES, (min[1] + max[1]) / 2 * TEXEL_METRES,
           (min[2] + max[2]) / 2 * TEXEL_METRES);
  if (visible) _scl.set((max[0] - min[0]) / BLOCK_TEXELS, (max[1] - min[1]) / BLOCK_TEXELS,
                        (max[2] - min[2]) / BLOCK_TEXELS);
  else _scl.set(0, 0, 0);
  return out.compose(_pos, _rot, _scl);
}

// The terrain mesh is made ONCE, with room to spare, and refilled in place
// when the world changes (the editor): the texel cache's twins, the glass mesh
// and every compiled shader set hold this mesh and its buffers, and a new mesh
// would strand them all. Past its capacity an edit cannot be shown until a
// reload (rebuildWorldInstances says so).
export const INSTANCE_HEADROOM = 16384;
let terrain = null;   // { geometry, instanceData, attrs: {...}, capacity }

function instanceCount() {
  let n = 0;
  for (const block of World.values()) for (const p of drawnParts(block)) n += p.boxes.length;
  for (const m of WorldModels) n += m.cubes.length;
  return n;
}

// The plain terrain's look on a fresh material: also the interiors' fade
// (render/features/occlusion.js), which must match it. params: the
// material's own (transparent, depthWrite...).
export function plainTerrainMaterial(params = {}) {
  const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.9, metalness: 0.0, ...params });
  // At the surface's texel (terrainSurfaceTSL): the face uv from the world
  // position, not the mesh uv, and on a slope its own texture. Each node gets
  // a surface of its OWN, built inside its own Fn: the normal is built in a
  // separate pass (three's NORMAL sub-build), and a surface shared with it
  // left the colour reading variables declared there - zero, so every
  // fragment sampled texel (0, 0).
  material.colorNode = Fn(() => {
    const surf = terrainSurfaceTSL();
    const c = mix(terrainTexelTSL(terrainTextures.albedo, surf.st, surf).rgb, vec3(1, 1, 1), terrainWhite);
    return select(surf.emissive, vec3(0, 0, 0), c);
  })();
  // A lightSource cube is unlit: all of its colour is emission.
  material.emissiveNode = Fn(() => {
    const surf = terrainSurfaceTSL();
    return select(surf.emissive, terrainTexelTSL(terrainTextures.albedo, surf.st, surf).rgb, vec3(0, 0, 0));
  })();
  // Lit by the plane's own normal: the mesh's is still the box top's.
  material.normalNode = Fn(() =>
    normalize(cameraViewMatrix.mul(vec4(terrainSurfaceTSL().n, 0)).xyz))();
  return material;
}

export function initWorldRender(scene) {
  const needed = instanceCount();
  if (needed === 0) return;
  initTerrainTextures();
  buildBlockVolume();
  const capacity = needed + INSTANCE_HEADROOM;

  const geometry = new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
  // Every per-instance value in ONE interleaved buffer: WebGPU allows 8
  // vertex buffers a pipeline, and position, normal, uv, the instance matrix
  // and colour take five. One attribute each was 11 - no pipeline at all.
  // Per instance, 20 floats:
  //   0 blockLayer  which material it draws with - its layer in the arrays
  //   1 blockGlass  1 for glass: terrainKeepTSL sends each to its own mesh
  //   2 blockSlope  a slope's kind and facing (slopeCode), 0 otherwise
  //   4.. modelInfo, modelUV0..2 - a model cube's mapping (modelTexelTSL),
  //   zeros for a block.
  const STRIDE = 20;
  const instanceData = new THREE.InstancedInterleavedBuffer(new Float32Array(capacity * STRIDE), STRIDE);
  const field = (name, size, offset) => {
    const a = new THREE.InterleavedBufferAttribute(instanceData, size, offset);
    geometry.setAttribute(name, a);
    return a;
  };
  terrain = {
    geometry, instanceData, capacity,
    layers: field('blockLayer', 1, 0),
    glassFlags: field('blockGlass', 1, 1),
    slopes: field('blockSlope', 1, 2),
    modelAttrs: MODEL_ATTRS.map((name, k) => field(name, 4, 4 + 4 * k))
  };
  // The unshadowed path. Standard lighting, but the colour is the block's own
  // layer - a plain `map` would put layer 0 on everything.
  const material = plainTerrainMaterial();
  material.positionNode = terrainKeepTSL(false);
  applyOcclusionMask(material);      // the occlusion cut-out (gpu/occlusion.js)

  worldInstancedMesh = new THREE.InstancedMesh(geometry, material, capacity);
  worldInstancedMesh.receiveShadow = true;
  worldInstancedMesh.castShadow = true;
  worldInstancedMesh.setColorAt(0, new THREE.Color(0xffffff));   // allocates the colour buffer
  const glassCount = fillInstances();
  scene.add(worldInstancedMesh);

  glassInstancedMesh = null;
  if (glassCount > 0) {
    // The unshadowed path's glass: plainly see-through, blended - there is no
    // field to trace without shadows. With them on, main.js swaps in the
    // traced material (gpu/shade.js glassNode).
    // Rough, as the plain terrain is (0.9): the unshadowed path has no specular
    // anywhere - marble and iron show none there either - so glass does not
    // get three's highlight from the directional light.
    const glassMat = new THREE.MeshStandardNodeMaterial({
      roughness: 0.9, metalness: 0.0, transparent: true, opacity: 0.35, depthWrite: false
    });
    // From the world position, as the terrain's: the mesh uv squashed the whole
    // texture onto a pane's top or a half's side.
    glassMat.colorNode = Fn(() => {
      const surf = terrainSurfaceTSL();
      return mix(terrainTexelTSL(terrainTextures.albedo, surf.st, surf).rgb, vec3(1, 1, 1), terrainWhite);
    })();
    glassMat.positionNode = terrainKeepTSL(true);
    const g = new THREE.InstancedMesh(geometry, glassMat, capacity);
    g.instanceMatrix = worldInstancedMesh.instanceMatrix;
    g.instanceColor = worldInstancedMesh.instanceColor;
    g.count = worldInstancedMesh.count;
    g.frustumCulled = false;
    g.raycast = () => {};          // picking stays on the terrain mesh, glass included
    g.name = 'glass';
    scene.add(g);
    glassInstancedMesh = g;
  }
}

// Writes every block's and placed model's instances, from index 0, and sets
// the mesh's count. Returns how many blocks are glass.
function fillInstances() {
  const mesh = worldInstancedMesh, t = terrain;
  voxelIndexMap.clear();
  instanceKeyByIndex.length = 0;
  instanceSource.length = 0;
  const matrix = new THREE.Matrix4();
  const white = new THREE.Color(0xffffff);
  const zero4 = [0, 0, 0, 0];
  // What hides a face, per cell (world.js opaqueBoxes), looked up once.
  const hiding = new Map();
  const boxesAt = (x, y, z) => {
    const key = getVoxelKey(x, y, z);
    let b = hiding.get(key);
    if (!b) hiding.set(key, b = opaqueBoxes(World.get(key)));
    return b;
  };
  const slopeKind = { slope: 0, halfSlope: 1, halfSlopeHigh: 1 };
  let i = 0, glassCount = 0;
  // Every SOLID block renders, not just standable ones: walls, pillars and the
  // underground layers are all real geometry that the lighting work needs to
  // occlude with. Standability is a derived query in world.js, not a render gate.
  for (const [key, block] of World) {
    // Both directions are needed: key -> indices for tinting and visibility,
    // index -> key so a raycast hit (which reports only an instanceId) can be
    // resolved back to the block that was clicked.
    const ids = [];
    const slope = slopeCode(block);
    const [bx, by, bz] = key.split(',').map(Number);
    let anyGlass = false;
    drawnParts(block).forEach(({ materialId, boxes }, part) => {
      const layer = materialLayer(materialId);
      const glass = isGlassMaterial(materialId);
      anyGlass = anyGlass || glass;
      for (const box of boxes) {
        instanceSource[i] = { box, glass, slope: !!slope, cell: [bx, by, bz], part };
        mesh.setMatrixAt(i, instanceMatrix(matrix, i, key));
        ids.push(i);
        instanceKeyByIndex[i] = key;
        mesh.setColorAt(i, white);
        t.layers.setX(i, layer);
        t.glassFlags.setX(i, (glass ? 1 : 0) + 2 * exposedFaceMask(bx, by, bz, box, !!slope, boxesAt));
        t.slopes.setX(i, slope);
        // A block's modelInfo (w 0): its slope plane's first layer + 1 (0: the
        // side stretched), its top and bottom layers (terrainSurfaceTSL).
        const f = MATERIAL_LAYERS.faces[layer] || { top: layer, bottom: layer, slope: [-1, -1] };
        const sk = slope ? slopeKind[blockShape(block)] : null;
        t.modelAttrs[0].setXYZW(i, sk == null ? 0 : f.slope[sk] + 1, f.top, f.bottom, 0);
        for (let k = 1; k < 4; k++) t.modelAttrs[k].setXYZW(i, ...zero4);
        i++;
      }
    });
    voxelIndexMap.set(key, ids);
    if (anyGlass) glassCount++;
  }
  // Placed models: each cube an instance, filed under the block the model
  // stands on (or hangs from) - picking, battle hiding and tint treat it as
  // that tile's.
  // Their buried faces collapse as a block's do (bbmodel.js modelFaceMasks).
  const masks = modelFaceMasks(WorldModels.flatMap(m => m.cubes), boxesAt);
  let mi = 0;
  for (const m of WorldModels) {
    const key = getVoxelKey(m.x, m.y, m.z);
    const ids = voxelIndexMap.get(key) || [];
    for (const cube of m.cubes) {
      instanceSource[i] = { texels: cube, model: m };
      mesh.setMatrixAt(i, instanceMatrix(matrix, i, key));
      ids.push(i);
      instanceKeyByIndex[i] = key;
      mesh.setColorAt(i, white);
      t.layers.setX(i, 0);
      t.glassFlags.setX(i, 2 * masks[mi++]);
      t.slopes.setX(i, 0);
      const d = modelInstanceData(cube, m.model);
      t.modelAttrs[0].setXYZW(i, ...d.info);
      for (let k = 0; k < 3; k++) t.modelAttrs[k + 1].setXYZW(i, ...d.uv.slice(k * 4, k * 4 + 4));
      i++;
    }
    voxelIndexMap.set(key, ids);
  }
  mesh.count = i;
  if (glassInstancedMesh) glassInstancedMesh.count = i;
  t.instanceData.needsUpdate = true;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  // Picking and frustum culling bound the instances; they have moved.
  mesh.boundingSphere = null;
  mesh.boundingBox = null;
  return glassCount;
}

// After the world changes (World, WorldModels): the instances, refilled in
// place, and the block-material volume. Returns { ok, count, glassNeeded }:
// ok false when the world has outgrown the mesh (a reload is needed to show
// it); glassNeeded when glass has appeared in a world that was built without
// any (its mesh and shaders do not exist - also a reload).
export function rebuildWorldInstances() {
  if (!worldInstancedMesh) return { ok: false, count: 0, glassNeeded: false };
  const needed = instanceCount();
  if (needed > terrain.capacity) return { ok: false, count: needed, glassNeeded: false };
  const glassCount = fillInstances();
  buildBlockVolume();
  return { ok: true, count: needed, glassNeeded: glassCount > 0 && !glassInstancedMesh };
}

// What an instance is: its block's key, and the placed model it is a cube of
// (null for a block) - for the editor's picking.
export function instanceInfo(id) {
  const src = instanceSource[id];
  return src ? { key: instanceKeyByIndex[id], model: src.model || null, part: src.part || 0 } : null;
}

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _scaleHidden = new THREE.Vector3(0, 0, 0);
const _scaleVisible = new THREE.Vector3(1, 1, 1);
const _rotation = new THREE.Quaternion();

export function updateVoxelVisibility(arenaMap, isBattle) {
  if (!worldInstancedMesh) return;
  
  const _matrix = new THREE.Matrix4();
  for (const [key, ids] of voxelIndexMap.entries()) {
    const visible = !(isBattle && (!arenaMap || !arenaMap.has(key)));
    for (const idx of ids) worldInstancedMesh.setMatrixAt(idx, instanceMatrix(_matrix, idx, key, visible));
  }
  worldInstancedMesh.instanceMatrix.needsUpdate = true;
  // A hidden neighbour buries nothing: the arena's edge shows its sides.
  const shown = key => !(isBattle && (!arenaMap || !arenaMap.has(key)));
  const boxesAt = (x, y, z) => {
    const key = getVoxelKey(x, y, z);
    return shown(key) ? opaqueBoxes(World.get(key)) : [];
  };
  instanceSource.forEach((src, i) => {
    if (!src.box) return;
    const [x, y, z] = src.cell;
    terrain.glassFlags.setX(i, (src.glass ? 1 : 0) + 2 * exposedFaceMask(x, y, z, src.box, src.slope, boxesAt));
  });
  // Model cubes the same way (a model at the edge against a hidden block).
  const cubeIds = [], cubes = [];
  instanceSource.forEach((src, i) => { if (src.texels) { cubeIds.push(i); cubes.push(src.texels); } });
  modelFaceMasks(cubes, boxesAt).forEach((m, k) => terrain.glassFlags.setX(cubeIds[k], 2 * m));
  terrain.instanceData.needsUpdate = true;
}

export function updateVoxelTints(arenaMap, reachableMap, isBattle) {
  if (!worldInstancedMesh) return;
  const colorWhite = new THREE.Color(0xffffff);
  const colorGrey = new THREE.Color(0x333333); 
  
  for (const [key, ids] of voxelIndexMap.entries()) {
    let colour = null;
    if (!isBattle) colour = colorWhite;
    else if (arenaMap.has(key)) colour = reachableMap && reachableMap.has(key) ? colorWhite : colorGrey;
    if (colour) for (const idx of ids) worldInstancedMesh.setColorAt(idx, colour);
  }
  worldInstancedMesh.instanceColor.needsUpdate = true;
}