// --- Tone mapping and auto-exposure ---
//
// three draws the frame into a half-float target and converts it to the
// screen in one output pass; the tone mapping and the exposure live in that
// pass (renderer.toneMapping, renderer.toneMappingExposure), so switching
// either rebuilds no material. AgX by default: bright saturated light - a
// torch, a glowing cube, the sun on snow - fades toward white rather than
// clipping or shifting hue. Our own AgX (gpu/tonemap.js), for its look: base
// AgX is flat on this art, so Punchy (more contrast and saturation) is the
// default. Off is the look from before it.
//
// Auto-exposure meters the terrain every METER_EVERY frames (gpu/exposure.js),
// reads the result back a few frames late, and eases toward the exposure that
// brings the frame's geometric mean to middle grey - within a range, so a dark
// room is brightened but night still reads as night. No meter source (the
// marched shading off) holds the last exposure. Only with a tone mapping on:
// three's NoToneMapping ignores the exposure.
//
// ctx:
//   renderer
//   effectsOn()    whether the advanced effects (the marched shading) are on:
//                  the basic mode is never tone mapped - it keeps its plain look
//   meterSource()  where the terrain's lit colour is this frame
//                  (render/features/texelcache.js meterSource), or null
import * as THREE from 'three/webgpu';
import { createExposureMeter } from '../../gpu/exposure.js';
import { createAgXLook, AGX_LOOKS } from '../../gpu/tonemap.js';
import { meteredLog2, targetEV, adaptEV, DEFAULT_EXPOSURE } from '../../core/exposure.js';
import { flip } from '../engine.js';

const METER_EVERY = 8;

const MODES = { agx: THREE.CustomToneMapping, off: THREE.NoToneMapping };

export function createToneMap({ renderer, effectsOn, meterSource }) {
  let mode = 'agx';
  let auto = true;
  let manualEV = 0;
  const params = { ...DEFAULT_EXPOSURE };
  let ev = 0;            // the exposure in use
  let target = 0;        // where auto is heading
  let metered = null;    // the last meter's mean log2 luminance
  const meter = createExposureMeter();
  let frame = 0, pending = false, lastT = 0;

  const agx = createAgXLook();
  renderer.library.addToneMapping(agx.fn, THREE.CustomToneMapping);
  // The look's name: a preset when its numbers match one, else 'custom'.
  const lookName = () => Object.keys(AGX_LOOKS).find(k => AGX_LOOKS[k].power === agx.power.value &&
    AGX_LOOKS[k].saturation === agx.saturation.value) || 'custom';
  // The mode with the effects on; none in the basic mode.
  function applyMode() { renderer.toneMapping = effectsOn() ? MODES[mode] : THREE.NoToneMapping; }
  applyMode();

  function readBack() {
    pending = true;
    renderer.getArrayBufferAsync(meter.attr).then(buf => {
      const w = new Uint32Array(buf);
      const m = meteredLog2(w[0], w[1]);
      if (m !== null) { metered = m; target = targetEV(m, params); }
    }).catch(() => {}).finally(() => { pending = false; });
  }

  // Each frame, after the terrain's low-res pass and before the main render.
  function update(now = performance.now()) {
    applyMode();
    const dt = lastT ? (now - lastT) / 1000 : 0;
    lastT = now;
    if (mode !== 'off' && auto) {
      const source = meterSource();
      if (source && ++frame % METER_EVERY === 0 && !pending) {
        renderer.compute(meter.kernels(source));
        readBack();
      }
      ev = adaptEV(ev, target, dt, params);
    } else {
      ev = manualEV;
    }
    renderer.toneMappingExposure = 2 ** ev;
  }

  const describe = () => (mode === 'off' ? 'tone mapping off (exposure has no effect)'
    : !effectsOn() ? `AgX ${lookName()} - with the effects on (the basic mode is never tone mapped)`
    : `AgX ${lookName()}, exposure ${auto ? 'auto' : 'fixed'} ${ev >= 0 ? '+' : ''}${ev.toFixed(2)} EV` +
      (auto && metered !== null ? ` (scene mean ${(2 ** metered).toFixed(3)}, ` +
        `range ${params.minEV}..${params.maxEV}, bias ${params.bias})` : ''));

  const commands = {
    tonemap: {
      help: "tone mapping: 'agx' (default) or 'off' (the look before it)",
      usage: "bxb.tonemap()  toggles  |  bxb.tonemap('agx')  |  bxb.tonemap('off')",
      run: (m = null) => {
        mode = m === null ? (mode === 'off' ? 'agx' : 'off') : (MODES[m] !== undefined ? m : mode);
        applyMode();
        return describe();
      }
    },
    agxlook: {
      help: "AgX's look: 'punchy' (default: contrast 1.1, saturation 1.05), 'base' (three's AgX, flat), or a contrast power and a saturation",
      usage: "bxb.agxlook('punchy')  |  bxb.agxlook('base')  |  bxb.agxlook(1.2, 1.25)",
      run: (a = 'punchy', sat = null) => {
        const preset = AGX_LOOKS[a];
        agx.power.value = preset ? preset.power : Math.max(0.5, Math.min(2.5, Number(a) || 1));
        agx.saturation.value = preset ? preset.saturation : Math.max(0, Math.min(2.5, sat === null ? 1 : Number(sat)));
        return describe();
      }
    },
    exposure: {
      help: "exposure: 'auto' meters the terrain and adapts; a number fixes it in EV",
      usage: "bxb.exposure('auto')  |  bxb.exposure(1.5)  |  bxb.exposure()  reports",
      run: (v = null) => {
        if (v === 'auto') auto = true;
        else if (v !== null) { auto = false; manualEV = Number(v) || 0; ev = manualEV; }
        return describe();
      }
    }
  };

  const range = (key, label, min, max, step, help) => ({
    key: 'exp' + key, label, type: 'range', min, max, step, help,
    get: () => params[key], set: v => { params[key] = v; if (metered !== null) target = targetEV(metered, params); }
  });

  return {
    name: 'tonemap',
    update,
    get mode() { return mode; },
    get ev() { return ev; },
    commands,
    settings: { title: 'Tone mapping', settings: [
      { key: 'tonemap', label: 'Tone mapping', type: 'select', options: Object.keys(MODES),
        help: 'AgX: bright light fades to white without clipping or hue shifts. Off: the old look',
        get: () => mode, set: v => commands.tonemap.run(v) },
      { key: 'agxlook', label: 'AgX look', type: 'select', options: [...Object.keys(AGX_LOOKS), 'custom'],
        help: "punchy: more contrast and colour; base: three's AgX, flat. Custom: the two sliders",
        get: lookName, set: v => { if (v !== 'custom') commands.agxlook.run(v); } },
      { key: 'agxpower', label: 'AgX contrast', type: 'range', min: 0.5, max: 2.5, step: 0.05,
        get: () => agx.power.value, set: v => commands.agxlook.run(v, agx.saturation.value) },
      { key: 'agxsat', label: 'AgX saturation', type: 'range', min: 0, max: 2.5, step: 0.05,
        get: () => agx.saturation.value, set: v => commands.agxlook.run(agx.power.value, v) },
      { key: 'autoexposure', label: 'Auto exposure', type: 'toggle',
        help: 'meters the terrain and adapts, like an eye; off holds the fixed exposure below',
        get: () => auto, set: v => flip(auto, v, () => commands.exposure.run(v ? 'auto' : manualEV)) },
      { key: 'exposure', label: 'Fixed exposure (EV)', type: 'range', min: -4, max: 6, step: 0.1,
        get: () => manualEV, set: v => commands.exposure.run(v) },
      range('bias', 'Auto: bias (EV)', -3, 3, 0.1, 'brighter or darker than middle grey'),
      range('minEV', 'Auto: darkest (EV)', -6, 0, 0.5, 'how far a bright scene is darkened'),
      range('maxEV', 'Auto: brightest (EV)', 0, 8, 0.5, 'how far a dark scene is brightened'),
      range('brighten', 'Auto: into the dark (s)', 0, 5, 0.1, 'seconds to adapt toward brighter'),
      range('darken', 'Auto: into the light (s)', 0, 5, 0.1, 'seconds to adapt toward darker')
    ]}
  };
}
