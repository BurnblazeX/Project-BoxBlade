// --- Render features, and what they share ---
//
// Each effect is a feature: one module under render/features/, made by
// create<Name>(ctx), owning its own state, and returning
//   name           its key here
//   options()      its slice of the shading build's options
//                  (gpu/shade.js createShadowColorNode)
//   update(frame)  optional: its per-frame CPU work. The frame loop calls the
//                  features in a fixed, written-out order - the order matters
//                  (a light's cards after the torch moved, the mirrors after the
//                  lights) - not in the order they were added.
//   commands       optional: its bxb console commands ({ name: { help, run } })
//   settings       optional: its settings-panel section ({ title, settings })
//
// A feature reaches another only through what it is handed in ctx, never by
// importing main.js or reaching for another feature's internals.

// A settings toggle: runs fn (a toggle) only when the value asked for differs
// from the current one.
export const flip = (now, want, fn) => (!!now === !!want ? undefined : fn());

export function createEngine() {
  const features = new Map();
  return {
    add(feature) {
      if (features.has(feature.name)) throw new Error(`render feature ${feature.name} added twice`);
      features.set(feature.name, feature);
      return feature;
    },
    get: name => features.get(name),
    // Every feature's slice of the shading options, as one object.
    options() {
      const out = {};
      for (const f of features.values()) if (f.options) Object.assign(out, f.options());
      return out;
    }
  };
}
