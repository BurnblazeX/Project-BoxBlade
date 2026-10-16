// --- The model registry ---
//
// Every .bbmodel in assets/objects, read at load (bbmodel.js parseBBModel) and
// kept by name. Each model's texture is cut into BLOCK_TEXELS-square tiles that
// ride in the terrain's own texture arrays, after the materials' layers
// (render.js): a model adds no texture binding to the terrain shaders, which
// are near WebGPU's limit of 16. modelLayer: its first layer; tilesX: tiles
// across; emissiveLayer: its _e texture's, when it has one. Its warnings are
// printed once, with its name.
import { parseBBModel } from './bbmodel.js';
import { MATERIALS, MATERIAL_LAYERS, BLOCK_TEXELS } from './materials.js';

// Only under Vite; in node (the tests) there are no models (as materials.js).
let sources = {};
try {
  sources = import.meta.glob('../../assets/objects/*.bbmodel',
                             { eager: true, query: '?raw', import: 'default' });
} catch { /* not under Vite */ }

export const MODELS = new Map();
let nextLayer = MATERIAL_LAYERS.end;
// Each model's entry in the GI's per-layer tables (gi.js), after the
// materials': what a probe landing on one of its cubes bounces.
let nextGi = MATERIALS.length;
for (const [path, text] of Object.entries(sources)) {
  const file = path.split('/').pop().replace(/\.bbmodel$/, '');
  let model;
  try { model = parseBBModel(text, file); }
  catch (err) { console.warn(`[models] ${file}: not a readable .bbmodel`, err); continue; }
  for (const w of model.warnings) console.warn(`[models] ${file}: ${w}`);
  model.tilesX = Math.max(1, Math.ceil(model.width / BLOCK_TEXELS));
  model.tilesY = Math.max(1, Math.ceil(model.height / BLOCK_TEXELS));
  model.modelLayer = nextLayer;
  nextLayer += model.tilesX * model.tilesY;
  // The emissive texture's tiles, after the albedo's: a lightSource cube reads
  // them in place of its albedo (render.js).
  model.emissiveLayer = model.textures.emissive ? nextLayer : null;
  if (model.textures.emissive) nextLayer += model.tilesX * model.tilesY;
  // The light's colour, once its pixels are decoded (render.js loadModelTiles).
  model.lightColour = null;
  // Its diffuse albedo for the GI (render.js loadModelTiles), at giIndex.
  model.giIndex = nextGi++;
  model.giAlbedo = null;
  MODELS.set(file, model);
}

// Layers the terrain arrays need: the materials' (with their faces and
// slopes, materials.js MATERIAL_LAYERS), then every model's tiles.
export const TERRAIN_LAYERS = nextLayer;
// The GI's per-layer tables' length: materials, then models.
export const GI_LAYERS = nextGi;
