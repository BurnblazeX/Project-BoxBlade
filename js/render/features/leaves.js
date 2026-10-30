// --- Leaves: every placed tree's canopy as swaying cubes (core/leaves.js
// gathers them, gpu/leaves.js draws and lights them) ---
//
// A tree's wood is model cubes like any model's; its leaves are cells drawn
// here. The list is rewritten when the placed models change (invalidate():
// a world edit, a region loaded); its light goes round the list a slice a
// frame, and the whole of it at once after a rewrite.
//
// ctx as createSmoke's (render/features/smoke.js): renderer, scene, camera,
// cascades(), lighting, sky, sun, cards, shadowsOn(), boxLight(); and
// models() (core/world.js WorldModels), focus() (the player, metres: past
// the capacity the nearest leaves are kept).
//
// The leaf volume (gpu/leafvolume.js) round the player holds every leaf
// cell; its light - the sun and sky through the leaves - is worked out again
// when the leaves change, the window moves or the sun turns. The leaf cubes
// and the shading's sun shadow (options: leaves) read it.
//
// The leaves are also in the sky under them (gpu/shade.js ambient), basic
// mode's box light, and the bounce: the LPV and texel GI rays lose light
// through them and gain the lit leaves' own - world-shadowed (gpu/gi.js,
// gpu/texelgi.js). Not yet: reflections, battle hiding.
import { createLeafCubes, DEFAULT_LEAVES, LEAF_LIGHT_SLICE, leafView } from '../../gpu/leaves.js';
import { wind, setWindHeading } from '../../gpu/wind.js';
import { writeSunUniforms } from '../../gpu/sun.js';
import { terrainTextures, glassInstancedMesh } from '../render.js';
import { leafVolume, LEAF_DIMS } from '../../gpu/leafvolume.js';
import { placedLeafBoxes, fillLeafDensity } from '../../core/leaves.js';
import { boxIndex, BOX_METRES } from '../../core/boxgrid.js';
import * as THREE from 'three/webgpu';
import { TREE_SPECIES } from '../../core/treegen.js';

// Metres above the player's feet the camera's view of them is aimed at.
const PLAYER_CHEST = 0.9;
// The window's min box from the player's: the player a quarter up it.
const windowAt = f => [boxIndex(f.x) - LEAF_DIMS[0] / 2, boxIndex(f.y) - LEAF_DIMS[1] / 4, boxIndex(f.z) - LEAF_DIMS[2] / 2];
// Boxes the player may stray from the window's middle before it moves.
const STRAY = 16;
// The sun's turn (1 - cos) that works the light out again.
const SUN_TURN = 1e-5;

// The light the leaves bounce (gpu/leafvolume.js u.albedo): the first
// species' ramp, its mean, linear. One colour for all leaves for now.
{
  const ramp = Object.values(TREE_SPECIES)[0].ramp;
  const sum = new THREE.Color(0, 0, 0);
  for (const [r, g, b] of ramp) sum.add(new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace));
  leafVolume.u.albedo.value.set(sum.r / ramp.length, sum.g / ramp.length, sum.b / ramp.length);
}

export function createLeaves({ renderer, scene, camera, cascades, lighting, sky, sun, cards = null, shadowsOn,
                               boxLight = () => null, models = () => [], focus = () => null }) {
  const params = { ...DEFAULT_LEAVES };
  let heading = 0;            // the wind's, degrees from +x toward +z
  let on = true;
  let gpu = null, gpuFor = null, gpuCards = null, gpuLightCards = null, gpuBasic = false;
  let warmed = null;
  let dirty = true;           // the list to rewrite
  let lightAll = false;       // the next light pass takes the whole list
  let offset = 0;
  let last = { count: 0, total: 0 };
  let boxes = [];             // every leaf cell's box (core/leaves.js placedLeafBoxes)
  let volumeDirty = true;     // the density to refill
  let windowOrigin = null;
  let lit = null;             // the sun direction the light was worked out for
  const t0 = performance.now();
  const now = () => (performance.now() - t0) / 1000;

  const write = () => {
    const f = focus();
    last = gpu.write(models(), f ? [f.x, f.y, f.z] : null);
    boxes = placedLeafBoxes(models());
    dirty = false;
    lightAll = true;
    volumeDirty = true;
  };

  // The leaf volume: moved with the player, refilled, its light worked out
  // again when anything it depends on changed. The leaf cubes are lit after;
  // their index (for the corners' light) is refilled with the window.
  function updateVolume(sunDir) {
    const v = leafVolume;
    const f = focus();
    if (f) {
      const want = windowAt(f);
      if (!windowOrigin || Math.abs(want[0] - windowOrigin[0]) > STRAY || Math.abs(want[2] - windowOrigin[2]) > STRAY ||
          Math.abs(want[1] - windowOrigin[1]) > STRAY / 2) {
        windowOrigin = want;
        volumeDirty = true;
      }
    }
    if (!windowOrigin) return;
    let relight = false;
    if (volumeDirty) {
      const b = [];
      const any = fillLeafDensity(boxes, windowOrigin, LEAF_DIMS, v.data, b);
      v.density.needsUpdate = true;
      // The leaves' bounds in metres, a box's margin round them (the lamps'
      // march, gpu/leafvolume.js leafTransmitTSL; the density is filtered);
      // none, min past max.
      if (any) {
        v.u.lo.value.set(b[0] - 1, b[1] - 1, b[2] - 1).multiplyScalar(BOX_METRES);
        v.u.hi.value.set(b[3] + 1, b[4] + 1, b[5] + 1).multiplyScalar(BOX_METRES);
      } else { v.u.lo.value.set(1, 1, 1); v.u.hi.value.set(0, 0, 0); }
      gpu.fillIndex(windowOrigin);
      v.u.origin.value.set(...windowOrigin);
      volumeDirty = false;
      relight = true;
      renderer.compute(v.skyKernel);
    }
    if (!lit || 1 - lit.dot(sunDir) > SUN_TURN) relight = true;
    if (!relight) return;
    v.u.sunDir.value.copy(sunDir);
    lit = sunDir.clone();
    renderer.compute(v.kernel);
    v.u.live.value = 1;
    lightAll = true;
  }

  // Made for the cascades in use, or in basic mode for the box light's walks
  // (as the smoke's cubes are).
  function ensure() {
    let made = null;
    if (!shadowsOn()) {
      const box = boxLight();
      const w = box && box.world();
      if (!w) return null;
      if (gpu && gpuBasic && gpuFor === w) return gpu;
      made = () => createLeafCubes({ terrain: terrainTextures, world: w });
      gpuFor = w; gpuBasic = true; gpuCards = gpuLightCards = null;
    } else {
      const c = cascades();
      if (!c || !c.length || !lighting.bindings()) return null;
      const cardOptions = cards ? cards.options() : {};
      if (gpu && !gpuBasic && gpuFor === c && gpuCards === cardOptions.cards && gpuLightCards === cardOptions.lightCards) return gpu;
      const s = sun.uniforms();
      made = () => createLeafCubes({ cascades: c, lights: lighting.bindings(), biasUniform: s.biasUniform,
                                     fadeStartUniform: s.fadeStartUniform, edgeFadeUniform: s.edgeFadeUniform,
                                     ambientUniform: s.ambientUniform, sky: sky.bindings, terrain: terrainTextures,
                                     cards: cardOptions.cards, lightCards: cardOptions.lightCards,
                                     glass: glassInstancedMesh ? terrainTextures : false });
      gpuFor = c; gpuBasic = false; gpuCards = cardOptions.cards; gpuLightCards = cardOptions.lightCards;
    }
    if (gpu) { scene.remove(gpu.mesh); gpu.dispose(); }
    gpu = made();
    scene.add(gpu.mesh);
    for (const [k, v] of Object.entries(params)) gpu.uniforms[k].value = v;
    dirty = true;
    return gpu;
  }

  // Compiled in the background as soon as made (as the smoke's cubes are).
  async function warm(g) {
    try {
      await renderer.compileComputeAsync(g.light);
      await renderer.compileComputeAsync(g.corners);
      const shown = g.mesh.visible;
      g.mesh.visible = true;
      let pending;
      try { pending = renderer.compileAsync(g.mesh, camera, scene); }
      finally { g.mesh.visible = shown; }
      await pending;
    } catch (err) {
      console.warn('[leaves] background compile failed - the first frame compiles instead', err);
    }
  }

  // Each frame, before the main render.
  function update() {
    const made = ensure();
    if (!made) return;
    if (made !== warmed) { warmed = made; warm(made); }
    if (dirty) write();
    const u = gpu.uniforms;
    gpu.mesh.visible = on && u.count.value > 0;
    if (!gpu.mesh.visible) { leafVolume.u.live.value = 0; lit = null; return; }
    writeSunUniforms(gpu.sun, sun.light.position, sun.angular());
    // Where the crown thins (gpu/leaves.js hideCutTSL): the player's chest.
    const f = focus();
    if (f) leafView.focus.value.set(f.x, f.y + PLAYER_CHEST, f.z);
    updateVolume(gpu.sun.dir.value);
    const ambient = sun.uniforms().ambientUniform.value;
    u.sunLight.value = sun.on() ? (gpuBasic ? 1 : 1 - ambient) : 0;
    u.time.value = now();
    const n = u.count.value;
    // The light, then its corners (gpu/leaves.js), the same cubes.
    const pass = count => { renderer.compute(gpu.light, count); renderer.compute(gpu.corners, count * 8); };
    if (lightAll) {
      u.offset.value = 0;
      pass(n);
      lightAll = false;
      offset = 0;
      return;
    }
    if (offset >= n) offset = 0;
    u.offset.value = offset;
    pass(Math.min(LEAF_LIGHT_SLICE, n - offset));
    offset += LEAF_LIGHT_SLICE;
  }

  const set = (k, v) => { params[k] = v; if (gpu) gpu.uniforms[k].value = v; };
  const range = (key, label, k, min, max, step, help) => ({ key, label, type: 'range', min, max, step, help,
    get: () => params[k], set: v => set(k, v) });

  const commands = {
    leaves: {
      help: 'the trees\' leaves: cubes that sway in the wind, textured by noise (core/leaves.js, gpu/leaves.js)',
      usage: "bxb.leaves()  how many  |  bxb.leaves('off')  |  bxb.leaves('on')  |  bxb.leaves('wind', 90)  heading in degrees",
      run: (a = null, b = null) => {
        if (a === 'off' || a === 'on') { on = a === 'on'; return `leaves ${a}`; }
        if (a === 'wind') { heading = Number(b) || 0; setWindHeading(heading); return `wind from ${heading} degrees`; }
        return `${last.count} leaf cubes drawn of ${last.total}`;
      }
    }
  };

  return {
    name: 'leaves',
    // The shading's sun shadow passes through the leaves (compiled in once).
    options: () => ({ leaves: leafVolume, leafView: true }),
    update,
    // The placed models changed: the list is rewritten next frame.
    invalidate() { dirty = true; },
    commands,
    settings: { title: 'Leaves', settings: [
      { key: 'leaveson', label: 'Leaves', type: 'toggle', get: () => on, set: v => { on = v; } },
      range('leafalbedo', 'Brightness', 'albedo', 0, 2, 0.01, 'on the species\' ramp'),
      range('leafsway', 'Sway (m)', 'sway', 0, 0.3, 0.005, 'how far the crown leans with the wind, and flutters across it'),
      range('leafspeed', 'Sway speed', 'speed', 0, 6, 0.1, 'how fast the wave runs along the wind'),
      range('leafwavelength', 'Gust length (m)', 'waveLength', 1, 30, 0.5, 'the distance from one swell to the next along the wind'),
      { key: 'leafwindstrength', label: 'Wind strength', type: 'range', min: 0, max: 3, step: 0.05,
        help: 'scales the sway (gpu/wind.js - one wind for everything that sways)',
        get: () => wind.strength.value, set: v => { wind.strength.value = v; } },
      { key: 'leafwindheading', label: 'Wind heading (degrees)', type: 'range', min: 0, max: 360, step: 5,
        help: 'from +x toward +z', get: () => heading, set: v => { heading = v; setWindHeading(v); } },
      range('leafclump', 'Clump size (texels)', 'clump', 1, 8, 0.5, 'the cellular noise\'s cell: the size of a leaf clump'),
      range('leafsss', 'Translucency', 'sss', 0, 1, 0.05, 'the share of the sun and lamps a face turned from them still takes (as smoke takes 0.45)'),
      range('leafms', 'Glow', 'ms', 0, 2, 0.05, 'sun scattered inside the crown, lighting every face alike - most where half of it gets through'),
      range('leafhide', 'See-through by player', 'hide', 0, 1, 0.05, 'how far the cut-outs open round the camera\'s view of the player'),
      range('leafhideradius', 'See-through radius (m)', 'hideRadius', 0.5, 6, 0.25, 'how wide the opening round that view is'),
      range('leafcutout', 'Cut-outs', 'cutout', 0, 0.75, 0.05, 'how much of the leaves is gaps - the eye sees through them into the crown and its branches'),
      range('leafgradient', 'Crown gradient', 'gradient', 0, 1, 0.05, 'how much darker the leaves are at the crown\'s bottom than at its top'),
      { key: 'leafdensity', label: 'Leaf density', type: 'range', min: 0, max: 2, step: 0.02,
        help: 'optical depth a leaf cell: how much sun and sky one cell of leaves holds back (gpu/leafvolume.js)',
        get: () => leafVolume.u.strength.value, set: v => { leafVolume.u.strength.value = v; volumeDirty = true; } },
      range('leafsun', 'Sun', 'sun', 0, 2, 0.05, 'the sun on the leaves'),
      range('leaflamps', 'Lamps', 'lamps', 0, 4, 0.05, 'the light list on the leaves'),
      range('leafsky', 'Sky', 'sky', 0, 2, 0.05, 'the sky on the leaves')
    ]}
  };
}
