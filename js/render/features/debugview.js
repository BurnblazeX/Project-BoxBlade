// --- The render debugger: one view at a time, in place of the shaded colour ---
//
// Every debug display that replaces what the frame shows is an entry here:
// the surface (albedo, normals, the raw LabPBR map, the texel lock), the field
// (the SDF march, the cascades), each lighting term raw (gpu/shade.js picks it
// out where it is built), glass's own views and split shading's reflection
// hits. A view is compiled in (gpu/debugview.js), so picking one rebuilds the
// shading - and needs the marched shading on, which picking one turns on.
// The reflection-hit views are a uniform (gpu/shade.js splitDebug): no rebuild.
//
// ctx:
//   shadowsOn()   whether the marched shading is on
//   shadows()     turns it on (built with the current view)
//   rebuild()     the shading rebuilt, when the view changed
import { splitDebug } from '../../gpu/shade.js';

// id: the bxb.view name. view: gpu/shade.js's view; glassView: glass's own
// (gpu/shade.js glassNode); splitDebug: the reflection-hit view.
export const VIEWS = [
  { id: 'final', label: 'Final composite' },
  { id: 'albedo', label: 'Material albedo', view: 'albedo' },
  { id: 'normal', label: 'Surface normals (normal-mapped)', view: 'normal',
    help: 'the shading normal, xyz as rgb: +x red, +y green, +z blue' },
  { id: 'facenormal', label: 'Face normals (geometry)', view: 'facenormal' },
  { id: 'labpbr', label: 'Specular map (LabPBR)', view: 'labpbr',
    help: 'the _s map raw: red smoothness, green F0 (230+ is metal), blue porosity/SSS' },
  { id: 'texels', label: 'Texel lock', view: 'texels',
    help: 'one colour per shaded texel: every light is worked out at its centre' },
  { id: 'sdf', label: 'Distance field (SDF)', view: 'sdf',
    help: 'the camera ray marched through the field: hue is steps (blue few, red 96+), banded every metre. ' +
          'Magenta: the field has no surface here. White: the field has one in front of what is drawn' },
  { id: 'cascades', label: 'Field cascades', view: 'cascades',
    help: 'which cascade holds the surface - green C0, yellow C1, orange C2, red none - checkered by its voxels' },
  { id: 'direct', label: 'Direct lighting', view: 'direct', help: 'the sun and the light list, N·L times shadow' },
  { id: 'shadow', label: 'Shadow visibility', view: 'shadow',
    help: 'the raw sun visibility: white lit, black occluded. Speckle or stripes on open ground is acne' },
  { id: 'indirect', label: 'Indirect illumination', view: 'indirect',
    help: 'the sky ambient (with AO), the GI bounce and the mirror light' },
  { id: 'ao', label: 'Ambient occlusion', view: 'ao' },
  { id: 'gi', label: 'GI bounce', view: 'gi',
    help: 'with texel GI and its sky on: the sky and the bounce together, as the patches hold them' },
  { id: 'skyopen', label: 'Sky open (roof map)', view: 'skyopen',
    help: 'green where nothing but glass is above, red under a roof - a GI ray that misses brings the sky only from green' },
  { id: 'mirror', label: 'Mirror light', view: 'mirror', help: 'sunlight and lamplight thrown by polished surfaces' },
  { id: 'reflection', label: 'Reflection contribution', view: 'reflection',
    help: 'what reflections add: Fresnel times the reflected (or ambient) light' },
  { id: 'highlights', label: 'Specular highlights', view: 'highlights' },
  { id: 'glass-thickness', label: 'Glass: thickness', glassView: 'thickness',
    help: 'path length inside the glass, white at two blocks' },
  { id: 'glass-through', label: 'Glass: transmitted only', glassView: 'through' },
  { id: 'glass-path', label: 'Glass: ray path', glassView: 'path',
    help: 'red opaque inside, green out, blue out after an internal reflection, white trapped' },
  { id: 'refl-colour', label: 'Reflection hits: colour, unlit', splitDebug: 1,
    help: 'split shading: what each reflection ray landed on, unlit' },
  { id: 'refl-distance', label: 'Reflection hits: distance', splitDebug: 2,
    help: 'split shading: white at 16 m, red a miss' },
  { id: 'refl-none', label: 'Reflection hits: none', splitDebug: 3 },
  { id: 'refl-normal', label: 'Reflection hits: face normal', splitDebug: 4 },
  { id: 'refl-block', label: 'Reflection hits: block', splitDebug: 5,
    help: 'grey by layer, magenta air, black a miss' }
];

// What a view id sets, or null for an unknown one.
export function resolveView(id) {
  const v = VIEWS.find(x => x.id === id);
  return v ? { view: v.view || null, glassView: v.glassView || null, splitDebug: v.splitDebug || 0 } : null;
}

export function createDebugView({ shadowsOn, shadows, rebuild }) {
  let current = 'final';
  let resolved = resolveView(current);

  function pick(id) {
    const next = resolveView(id);
    if (!next) return false;
    const recompile = next.view !== resolved.view || next.glassView !== resolved.glassView;
    current = id;
    resolved = next;
    splitDebug.value = next.splitDebug;
    if (id !== 'final' && !shadowsOn()) shadows();
    else if (recompile && shadowsOn()) rebuild();
    return true;
  }

  const commands = {
    view: {
      help: 'render debugger: show one term in place of the final colour. No argument lists the views; ' +
            "'next' / 'prev' step through them",
      usage: "bxb.view('sdf')  |  bxb.view('final')  |  bxb.view('next')  |  bxb.view()",
      run: (id = null) => {
        if (id === null) {
          return VIEWS.map(v => `${v.id === current ? '>' : ' '} ${v.id.padEnd(16)} ${v.label}`).join('\n');
        }
        if (id === 'next' || id === 'prev') {
          const i = VIEWS.findIndex(v => v.id === current);
          id = VIEWS[(i + (id === 'next' ? 1 : VIEWS.length - 1)) % VIEWS.length].id;
        }
        if (!pick(id)) return `unknown view '${id}' - bxb.view() lists them`;
        const v = VIEWS.find(x => x.id === id);
        return `view: ${v.label}${v.help ? ` - ${v.help}` : ''}`;
      }
    }
  };

  return {
    name: 'debugView',
    options: () => ({ view: resolved.view, glassView: resolved.glassView }),
    get current() { return current; },
    commands,
    // Alt+B: the next view, Alt+Shift+B the previous. True when it was this key.
    key(e) {
      if (!e.altKey || e.code !== 'KeyB') return false;
      e.preventDefault();
      console.log(commands.view.run(e.shiftKey ? 'prev' : 'next'));
      return true;
    },
    // For the Debug section.
    setting: { key: 'view', label: 'View (Alt+B)', type: 'select',
      options: VIEWS.map(v => ({ value: v.id, label: v.label })),
      help: 'one term of the frame in place of the final colour (bxb.view() for what each shows)',
      get: () => current, set: v => commands.view.run(v) }
  };
}
