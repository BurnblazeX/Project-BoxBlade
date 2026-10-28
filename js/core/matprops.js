// --- Materials: the gameplay layer (Gameplay §4.1, World plan P3) ---
//
// What a material DOES, per material id, beside materials.js's table of how it
// looks. All 3-bit (0..7) but conductive, which is one bit. Temperature is not
// here: it is a per-cell state (a chunk layer, core/chunks.js), not a property
// of the material. Read by the simulations as they come - fire (flammability),
// water (porosity soaks, conductive carries a shock), movement (friction),
// breaking (hardness); nothing reads them yet.
//
// A material with no row here - a texture dropped into assets/ that the asset
// pass has not given properties yet - takes DEFAULT_PROPS: middling, inert.

export const PROP_KEYS = ['flammability', 'porosity', 'friction', 'conductive', 'hardness'];

export const DEFAULT_PROPS = Object.freeze({ flammability: 0, porosity: 2, friction: 4, conductive: 0, hardness: 4 });

const PROPS = {
  // Turf: catches when dry, drinks water, grips.
  grass:        { flammability: 3, porosity: 5, friction: 5, conductive: 0, hardness: 1 },
  // Polished stone: sealed, slick, hard.
  marble:       { flammability: 0, porosity: 1, friction: 2, conductive: 0, hardness: 6 },
  ironPlate:    { flammability: 0, porosity: 0, friction: 3, conductive: 1, hardness: 7 },
  glass:        { flammability: 0, porosity: 0, friction: 2, conductive: 0, hardness: 2 },
  stainedGlass: { flammability: 0, porosity: 0, friction: 2, conductive: 0, hardness: 2 },
  frostedGlass: { flammability: 0, porosity: 0, friction: 3, conductive: 0, hardness: 2 }
};

for (const row of Object.values(PROPS)) Object.freeze(row);

// A material's gameplay properties.
export function materialProps(id) {
  return PROPS[id] || DEFAULT_PROPS;
}

// The ids that have their own row.
export const PROP_MATERIALS = Object.keys(PROPS);
