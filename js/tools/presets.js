// --- Graphics presets ---
//
// Five tiers over the settings panel's own keys, so a preset is nothing but a
// list of settings: the panel and the console stay the only owners of state.
// `effects` says whether the advanced effects (shadows, GI, reflections) are on;
// `values` are set in order, so a select that rebuilds (the GI method) comes
// before the knobs that only mean something under it.
//
// Every advanced preset names the same keys, so going from one to another puts
// each knob back - except the texel GI's own, which only Ultra uses (they are
// inert under the LPV, and reading them builds the texel GI tables).
//
// Measured in the Chromium pane at 1920x1080 on the dev box (RTX 5060 Ti), the
// spawn view, uncapped (bxb.pacing('uncapped')), 2026-10-27: Potato 1714,
// Low 779, Original 589, High 407, Ultra 414 fps; inside the church
// (bxb.tp(10, 40, 1)) Potato 1024, Low 586, Original 457, High 280, Ultra 279.
// Low then lost reflections and two shadowed lights: at least 777 fps over the
// church and spawn at all four rotations. Cost moves with the spot and the
// camera's rotation, so measure several of each and judge by the worst.
// The rendering doc §0 has the rest.

const BASE = {
  gimode: 'lpv', gi: true, gires: 1, gibounces: 2, giiter: 8, gislices: 4, gifar: true,
  soften: 'cone', sunfar: 36, aodist: 1,
  // Long shadows (core/sunmap.js): the soft march (the cone's penumbra, the
  // wanted look) runs this far, the sun depth map takes over past it -
  // softened inside C2 by a grid of bilinear taps, hard past it.
  sunreach: 'c1', farshadow: 4,
  reflect: true, reflectbounces: 2, spec: true,
  mirrors: true, glassglare: 1, caustics: 1,
  glasslayers: 2, glassbounces: 4, dispersion: false,
  budget: 16, cardbudget: 16, hitbudget: 4, mirrorbudget: 4,
  smokeshadows: true
};

export const PRESETS = {
  potato: {
    label: 'Potato', target: '1000+ fps',
    help: 'basic mode: box light, voxel AO, no marched shadows',
    effects: false, values: { boxlight: true }
  },
  low: {
    label: 'Low', target: '600+ fps',
    help: 'advanced, cheaper: one LPV bounce at half rate, no reflections or mirror light, ' +
          'two shadowed lights',
    effects: true,
    values: { ...BASE, gibounces: 1, giiter: 4, gislices: 8,
              reflect: false, reflectbounces: 1, mirrors: false, glassglare: 0, caustics: 0,
              glasslayers: 1, glassbounces: 2,
              budget: 2, cardbudget: 1, hitbudget: 1, mirrorbudget: 0, sunfar: 24, aodist: 0.75,
              sunreach: '12m', farshadow: 'hard' }
  },
  original: {
    label: 'Original', target: '400+ fps',
    help: 'the default advanced effects: LPV GI (2 cells a block near, 1 far), two bounces, ' +
          'soft sun march through C1, longer shadows from a smoothed sun map',
    effects: true, values: { ...BASE }
  },
  high: {
    label: 'High', target: '200-240 fps',
    help: 'finer LPV (3 cells a block near), three bounces, far volume every frame, longer AO, ' +
          'more reflection and glass bounces, dispersion',
    effects: true,
    values: { ...BASE, gires: 1.5, gibounces: 3, gifar: false, aodist: 1.25,
              reflectbounces: 3, glasslayers: 3, glassbounces: 6, dispersion: true, sunfar: 54,
              hitbudget: 8, mirrorbudget: 8, farshadow: 6 }
  },
  ultra: {
    label: 'Ultra', target: '120+ fps',
    help: 'per-texel ray-traced GI, longest reaches, every light in reflections and mirrors',
    effects: true,
    values: { ...BASE, gimode: 'texel', gibounces: 3, gifar: false, aodist: 1.5,
              reflectbounces: 4, glasslayers: 4, glassbounces: 8, dispersion: true, sunfar: 72,
              hitbudget: 16, mirrorbudget: 16, farshadow: 8,
              gireach: 16, gidirectevery: 2 }
  }
};

export const PRESET_NAMES = Object.keys(PRESETS);

// A setting's value as the panel would store it; a def that is missing (a
// feature not built in this build) is skipped rather than failing the preset.
function same(a, b) {
  return typeof a === 'number' || typeof b === 'number' ? Math.abs(Number(a) - Number(b)) < 1e-6
                                                        : String(a) === String(b);
}

// Which preset the current state is, or null (custom). Stops at the first
// mismatch, and the GI method is checked first, so a read never builds the
// texel GI tables unless texel GI is already the method.
export function matchPreset(defs, effectsOn) {
  for (const name of PRESET_NAMES) {
    const p = PRESETS[name];
    if (p.effects !== effectsOn) continue;
    let ok = true;
    for (const [key, v] of Object.entries(p.values)) {
      const def = defs.get(key);
      if (!def) continue;
      let cur;
      try { cur = def.get(); } catch { ok = false; break; }
      if (!same(cur, v)) { ok = false; break; }
    }
    if (ok) return name;
  }
  return null;
}

// Applies a preset: effects off first, every value set while nothing is built,
// then effects on - so a switch compiles one shader set, not one a setting.
// setEffects(on) must resolve once the effects are (or are not) built.
export async function applyPreset(name, { defs, effectsOn, setEffects }) {
  const p = PRESETS[name];
  if (!p) throw new Error(`no preset '${name}' - use ${PRESET_NAMES.join(', ')}`);
  if (effectsOn()) await setEffects(false);
  const skipped = [];
  for (const [key, v] of Object.entries(p.values)) {
    const def = defs.get(key);
    if (!def) { skipped.push(key); continue; }
    try {
      if (!same(def.get(), v)) await def.set(v);
    } catch { skipped.push(key); }
  }
  if (p.effects) await setEffects(true);
  return `${p.label} (${p.target})` + (skipped.length ? ` - skipped ${skipped.join(', ')}` : '');
}
