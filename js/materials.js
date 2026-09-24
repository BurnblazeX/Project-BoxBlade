// --- Block materials ---
//
// The table a block's materialId indexes. Deliberately small: this is the
// rendering half only - which textures a block draws with - until the §6.4
// material system (gameplay properties, derived PBR) exists to own it.
//
// A material's layer is its index here, and that index is what the terrain's
// texture arrays are stacked by (render.js), so the order is load-bearing:
// append, never reorder. Layer 0 is the fallback for a block with no id.
//
// Textures live in assets/textures as <texture>.png (albedo), with optional
// LabPBR companions beside it: <texture>_n.png (normal) and <texture>_s.png
// (specular). A missing _n is generated from the albedo (normals.js); a missing
// _s is DEFAULT_SPECULAR. Every block texture is BLOCK_TEXELS square - one
// texel per boxGrid voxel, the §4 lock.
export const BLOCK_TEXELS = 12;

export const MATERIALS = [
  { id: 'grass',     texture: 'terrain_grass' },
  // Polished black marble: a dielectric mirror-ish finish (see its _s).
  { id: 'marble',    texture: 'terrain_marble' },
  // Polished iron plate: LabPBR metal 230.
  { id: 'ironPlate', texture: 'terrain_ironPlate' },
  // Clear glass (tools/make-glass-textures.py). glass: true puts it in the
  // field's glass channel and draws it with the glass mesh, whose shading traces
  // through it; its albedo is the tint what passes through takes on.
  { id: 'glass',     texture: 'terrain_glass', glass: true },
  // Stained glass: four coloured panes in lead. Its texels colour the light
  // through it - in the view and in the shadows it casts.
  { id: 'stainedGlass', texture: 'terrain_stainedGlass', glass: true },
  // Frosted glass: rough (smoothness 155, alpha 0.15), so what shows through it
  // is jittered per texel into a stable grain.
  { id: 'frostedGlass', texture: 'terrain_frostedGlass', glass: true }
];

const layerById = new Map(MATERIALS.map((m, i) => [m.id, i]));

// What the two faces of a glass slab let through, of light meeting the first at
// |cos| cosI: each reflects F (Schlick, from its F0), so (1 - F)^2. The CPU
// mirror of gpu.js glassSurfaceTransmitTSL, which a shadow ray through glass
// takes - head-on it is (1 - 0.04)^2 = 0.92 for glass, and at grazing, nothing.
export function glassSurfaceTransmit(f0, cosI) {
  const c = Math.min(1, Math.max(0, cosI));
  const F = f0 + (1 - f0) * Math.pow(1 - c, 5);
  return (1 - F) * (1 - F);
}

// Glass: a material with glass: true. Baked into the field's glass distance
// instead of its opaque one (boxgrid.js FIELD_CHANNELS).
export function isGlassMaterial(id) {
  const i = id == null ? -1 : layerById.get(id);
  return i != null && i >= 0 && MATERIALS[i].glass === true;
}

export function materialLayer(id) {
  return id == null ? 0 : (layerById.get(id) ?? 0);
}

// --- LabPBR specular ---
//
// The layout, per channel (labpbr.github.io, "specular texture"):
//   R  perceptual smoothness. roughness = (1 - R/255)^2
//   G  0-229: F0, linear, G/255.  230-237: a predefined metal (below).
//      238-255: some other metal - its albedo is its F0.
//   B  0-64: porosity (0-1 over 0-64).  65-255: subsurface scattering
//      (0-1 over 65-255).  Always 0 for metals.
//   A  emission, 0-254 over 0-1. 255 is NOT full - it is "no emission", so a
//      texture authored without alpha does not glow.
// A surface with no _s texture: rough (0), F0 0.04, nothing else.
export const DEFAULT_SPECULAR = [0, 10, 0, 255];

// LabPBR's predefined metals, 230-237, as their complex index of refraction
// (n, k per RGB) from the standard's own table. F0 is derived below rather than
// stored, so the numbers stay checkable against the source.
export const LABPBR_METALS = [
  { id: 230, name: 'iron',
    n: [2.9114, 2.9497, 2.5845], k: [3.0893, 2.9318, 2.7670] },
  { id: 231, name: 'gold',
    n: [0.18299, 0.42108, 1.3734], k: [3.4242, 2.34590, 1.7704] },
  { id: 232, name: 'aluminum',
    n: [1.3456, 0.96521, 0.61722], k: [7.4746, 6.3995, 5.3031] },
  { id: 233, name: 'chrome',
    n: [3.1071, 3.1812, 2.3230], k: [3.3314, 3.3291, 3.1350] },
  { id: 234, name: 'copper',
    n: [0.27105, 0.67693, 1.31640], k: [3.60920, 2.62480, 2.29210] },
  { id: 235, name: 'lead',
    n: [1.9100, 1.8300, 1.4400], k: [3.5100, 3.4000, 3.1800] },
  { id: 236, name: 'platinum',
    n: [2.3757, 2.0847, 1.8453], k: [4.2655, 3.7153, 3.1365] },
  { id: 237, name: 'silver',
    n: [0.15943, 0.14512, 0.13547], k: [3.9291, 3.1900, 2.3808] }
];

// Fresnel at normal incidence for a conductor: ((n-1)^2 + k^2) / ((n+1)^2 + k^2).
export function conductorF0(n, k) {
  return n.map((ni, i) => ((ni - 1) ** 2 + k[i] ** 2) / ((ni + 1) ** 2 + k[i] ** 2));
}
export const METAL_F0 = LABPBR_METALS.map(m => conductorF0(m.n, m.k));

// One specular texel, decoded. albedo (linear RGB, 0-1) is only consulted for
// the 238-255 "albedo is F0" metals. f0 is always RGB.
export function decodeSpecular([r, g, b, a], albedo = [1, 1, 1]) {
  const smoothness = r / 255;
  const roughness = (1 - smoothness) ** 2;
  const metal = g >= 230;
  let f0;
  if (!metal) f0 = [g / 255, g / 255, g / 255];
  else if (g <= 237) f0 = METAL_F0[g - 230].slice();
  else f0 = albedo.slice();
  return {
    smoothness, roughness, metal, f0,
    metalName: metal && g <= 237 ? LABPBR_METALS[g - 230].name : (metal ? 'albedo' : null),
    porosity: !metal && b <= 64 ? b / 64 : 0,
    sss: !metal && b > 64 ? (b - 65) / 190 : 0,
    emission: a === 255 ? 0 : a / 254
  };
}
