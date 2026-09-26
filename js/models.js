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
import { MATERIALS, BLOCK_TEXELS } from './materials.js';

const sources = import.meta.glob('../assets/objects/*.bbmodel',
                                 { eager: true, query: '?raw', import: 'default' });

export const MODELS = new Map();
let nextLayer = MATERIALS.length;
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
  MODELS.set(file, model);
}

// Layers the terrain arrays need: the materials', then every model's tiles.
export const TERRAIN_LAYERS = nextLayer;
