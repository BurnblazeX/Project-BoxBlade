// --- Cloud spells: smoke of cubes that the world sees (core/smoke.js forms
// them, gpu/smoke.js draws them, gpu/smokeshadow.js shadows through them) ---
//
// A cloud spell (core/smoke.js CLOUD_SPELLS: Fog Cloud, Stinking Cloud,
// Cloudkill, Darkness, Incendiary Cloud) cast on a block is a sphere of cubes
// (three a block) centred on the middle of the block's top face - the part
// above ground shows. It rolls out over a second and a half, its cubes then
// breathe in a wave, and it stays until cleared. Debug: Alt+F casts the
// chosen spell on the block under the pointer, Alt+Shift+F clears (main.js);
// bxb.smoke(). Upcast by the cast level (Fog Cloud only, as in BG3). Up to MAX_SMOKE_CLOUDS (8) at once -
// a new one past that replaces the oldest: each is a slot every shadow ray
// and light checks (gpu/smokeshadow.js).
//
// It is in the world's light both ways: the shading's sun march and every
// light pass through it (options: smoke, compiled in with the shading - a
// cloud cast or cleared is a texture and uniform write, no rebuild), so the
// ground and walls under a cloud are in its shadow; and each cube is lit by
// the sun through the world and all the smoke, and by the lamps.
//
// ctx:
//   renderer, scene
//   cascades()     the shadow cascades
//   lighting       the light list feature: bindings()
//   sky            the sky feature: bindings
//   sun            { light (its position is the sun's direction), on(),
//                    angular(), uniforms() - the shading's march uniforms }
//   shadowsOn()    whether the marched shading is on
//   rebuild()      the shading rebuilt (smoke shadows switched)
//   boxLight()     basic mode's box light feature (render/features/boxlight.js):
//                  with the advanced effects off the cubes light themselves
//                  by its walks (world()), and it shadows the terrain through
//                  the smoke - version() and growing() tell it when to look
//                  again.
import { createSmokeCloud, smokeRadius, CLOUD_SPELLS, MAX_CAST_LEVEL, SMOKE_GROW_IN } from '../../core/smoke.js';
import { createSmokeCubes, MAX_SMOKE_CLOUDS, DEFAULT_SMOKE, SMOKE_ENCLOSED_DELAY, smokeShared } from '../../gpu/smoke.js';
import { smokeView } from '../../gpu/smokeview.js';
import { writeSunUniforms } from '../../gpu/sun.js';
import { terrainTextures, glassInstancedMesh } from '../render.js';
import { flip } from '../engine.js';

export function createSmoke({ renderer, scene, camera, cascades, lighting, sky, sun, cards = null, shadowsOn,
                              rebuild = () => {}, boxLight = () => null }) {
  const clouds = new Array(MAX_SMOKE_CLOUDS).fill(null);
  let nextSlot = 0;
  let spell = 'fogCloud';
  let castLevel = 1;         // the level Alt+F casts at - a spell's own if that is higher
  let shadows = true;        // the smoke in the world's light (a rebuild to switch)
  const params = { ...DEFAULT_SMOKE };
  let gpu = null, gpuFor = null, gpuCards = null, gpuLightCards = null;
  // The cubes whose pipelines are compiled (or compiling): see warm().
  let warmed = null;
  // Basic mode's cubes: made over the box light's walks (gpuFor is them).
  let gpuBasic = false;
  // Bumped whenever the clouds change (a cast, a clear, a roll-out done).
  let version = 0;
  const t0 = performance.now();
  const now = () => (performance.now() - t0) / 1000;

  // Slot k to the GPU, once a cast: its boxes and growth for the shadows and
  // the bounce, its surface cubes to draw. How far it has rolled out is worked
  // out on the GPU from the cast time (gpu/smokeshadow.js smokeGrowthTSL).
  const write = k => {
    const c = clouds[k];
    smokeShared.write(k, c ? c.cloud : null, c ? c.at : 0);
    if (gpu) gpu.write(clouds);
    version++;
  };

  // Made for the cascades in use (a shadows rebuild makes new ones) - or in
  // basic mode for the box light's walks (remade with its kernels).
  function ensure() {
    if (!shadowsOn()) {
      const box = boxLight();
      const w = box && box.world();
      if (!w) return null;
      if (gpu && gpuBasic && gpuFor === w) return gpu;
      if (gpu) { scene.remove(gpu.mesh); gpu.dispose(); }
      gpu = createSmokeCubes({ terrain: terrainTextures, world: w });
      gpuFor = w; gpuBasic = true; gpuCards = gpuLightCards = null;
      scene.add(gpu.mesh);
      for (const [k, v] of Object.entries(params)) gpu.uniforms[k].value = v;
      for (let k = 0; k < MAX_SMOKE_CLOUDS; k++) write(k);
      return gpu;
    }
    const c = cascades();
    if (!c || !c.length || !lighting.bindings()) return null;
    const cardOptions = cards ? cards.options() : {};
    if (gpu && !gpuBasic && gpuFor === c && gpuCards === cardOptions.cards && gpuLightCards === cardOptions.lightCards) return gpu;
    if (gpu) { scene.remove(gpu.mesh); gpu.dispose(); }
    gpuBasic = false;
    const s = sun.uniforms();
    gpu = createSmokeCubes({ cascades: c, lights: lighting.bindings(), biasUniform: s.biasUniform,
                             fadeStartUniform: s.fadeStartUniform, edgeFadeUniform: s.edgeFadeUniform,
                             ambientUniform: s.ambientUniform, sky: sky.bindings, terrain: terrainTextures,
                             cards: cardOptions.cards, lightCards: cardOptions.lightCards,
                             glass: glassInstancedMesh ? terrainTextures : false });
    gpuFor = c; gpuCards = cardOptions.cards; gpuLightCards = cardOptions.lightCards;
    scene.add(gpu.mesh);
    for (const [k, v] of Object.entries(params)) gpu.uniforms[k].value = v;
    for (let k = 0; k < MAX_SMOKE_CLOUDS; k++) write(k);
    return gpu;
  }

  // The cubes' light kernel and material compiled in the background as soon
  // as they are made, before any cloud: built on the first cast instead, the
  // compile was a ~200 ms frame. Shown only for the synchronous part of the
  // call, where the render object is captured (as render/shadowsets.js does).
  async function warm(g) {
    try {
      await renderer.compileComputeAsync(g.light);
      const shown = g.mesh.visible;
      g.mesh.visible = true;
      let pending;
      try { pending = renderer.compileAsync(g.mesh, camera, scene); }
      finally { g.mesh.visible = shown; }
      await pending;
    } catch (err) {
      console.warn('[smoke] background compile failed - the first cast compiles instead', err);
    }
  }

  // The chosen cloud spell on the block whose top is at tile { gx, gy, gz },
  // at `level` (upcast past its own: core/smoke.js smokeRadius).
  function summonAt(tile, which = spell, level = castLevel) {
    const def = CLOUD_SPELLS[which];
    if (!def) return `no cloud spell '${which}' - ${Object.keys(CLOUD_SPELLS).join(', ')}`;
    const cloud = createSmokeCloud(tile, undefined, def, Math.max(level, def.level));
    if (!cloud.cubes.length) return `no air round block ${tile.gx}, ${tile.gy}, ${tile.gz}`;
    const k = nextSlot;
    clouds[k] = { cloud, at: now() };
    nextSlot = (nextSlot + 1) % MAX_SMOKE_CLOUDS;
    write(k);
    return `${def.name} (level ${cloud.level}, radius ${cloud.radius} cubes) on block ${tile.gx}, ${tile.gy}, ${tile.gz}: ` +
           `${cloud.cubes.length} cubes, ${cloud.cubes.filter(c => c[7]).length} open to air`;
  }

  function clear() {
    clouds.fill(null);
    nextSlot = 0;
    for (let k = 0; k < MAX_SMOKE_CLOUDS; k++) write(k);
    return 'smoke cleared';
  }

  // Each frame, before the main render.
  function update() {
    // The clock the clouds roll out by (their cast times are on it).
    smokeShared.time.value = now();
    // A cloud rolled out: its inside cubes are enclosed now - drop them.
    let compacted = false;
    clouds.forEach(c => {
      if (c && !c.compact && now() - c.at > c.cloud.growSeconds + SMOKE_GROW_IN + SMOKE_ENCLOSED_DELAY + 0.1) {
        c.compact = compacted = true;
      }
    });
    if (compacted) { version++; if (gpu) gpu.write(clouds); }
    const made = ensure();
    if (made && made !== warmed) { warmed = made; warm(made); }
    const live = clouds.some(Boolean) && made;
    if (gpu) gpu.mesh.visible = !!live;
    if (!live) return;
    const u = gpu.uniforms;
    writeSunUniforms(gpu.sun, sun.light.position, sun.angular());
    smokeView.sunDir.value.copy(gpu.sun.dir.value);
    const ambient = sun.uniforms().ambientUniform.value;
    // Basic mode's sun is the plain light, whole; the shaded path's is what
    // its ambient leaves.
    u.sunLight.value = sun.on() ? (gpuBasic ? 1 : 1 - ambient) : 0;
    u.time.value = now();
    // Over the cubes in the list, not the whole capacity: past the count a
    // thread only writes a light nothing reads (128k of them, 8 MB a frame).
    renderer.compute(gpu.light, Math.max(1, u.count.value));
  }

  const set = (k, v) => { params[k] = v; if (gpu) gpu.uniforms[k].value = v; };
  const range = (key, label, k, min, max, step, help) => ({ key, label, type: 'range', min, max, step, help,
    get: () => params[k], set: v => set(k, v) });

  const commands = {
    smoke: {
      help: 'cloud spells: smoke of cubes on a block - Alt+F on the block under the pointer, Alt+Shift+F clears',
      usage: "bxb.smoke(x, y, z)  the chosen spell on that block  |  bxb.smoke('stinkingCloud')  choose  |  " +
             "bxb.smoke('level', 5)  upcast  |  " +
             "bxb.smoke('clear')  |  bxb.smoke('shadows')  toggles the smoke in the world's light",
      run: (a = null, b = null, c = null) => {
        if (a === 'clear') return clear();
        if (a === 'shadows') {
          shadows = !shadows;
          if (shadowsOn()) rebuild();
          return `smoke ${shadows ? 'casts shadows and dims lights' : 'casts no shadows'}`;
        }
        if (a === 'level') {
          castLevel = Math.max(1, Math.min(MAX_CAST_LEVEL, Math.round(Number(b))));
          const def = CLOUD_SPELLS[spell];
          return `Alt+F casts at level ${castLevel}: ${def.name} radius ${smokeRadius(def, castLevel)} cubes`;
        }
        if (typeof a === 'string') {
          if (!CLOUD_SPELLS[a]) return `cloud spells: ${Object.keys(CLOUD_SPELLS).join(', ')}`;
          spell = a;
          return `Alt+F casts ${CLOUD_SPELLS[a].name}`;
        }
        if (a !== null && b !== null && c !== null) return summonAt({ gx: a, gy: b, gz: c });
        return `bxb.smoke(x, y, z) or Alt+F casts ${CLOUD_SPELLS[spell].name}`;
      }
    }
  };

  return {
    name: 'smoke',
    // The shading reads the smoke: its sun march and lights pass through it.
    options: () => ({ smoke: shadows ? smokeShared : null, smokeView: true }),
    update,
    summonAt,
    clear,
    // For the box light: when the clouds last changed, and whether one is
    // still rolling out (its shadow growing - a little past, so the last look
    // sees it whole).
    version: () => version,
    growing: () => clouds.some(c => c && now() - c.at < c.cloud.growSeconds + SMOKE_GROW_IN + 0.3),
    // The live clouds (core/smoke.js), for gameplay - what a line of sight crosses.
    clouds: () => clouds.filter(Boolean).map(c => c.cloud),
    commands,
    settings: { title: 'Smoke (cloud spells)', settings: [
      { key: 'smokespell', label: 'Spell (Alt+F)', type: 'select',
        options: Object.entries(CLOUD_SPELLS).map(([value, d]) => ({ value, label: d.name })),
        get: () => spell, set: v => { spell = v; } },
      { key: 'smokelevel', label: 'Cast level (Alt+F)', type: 'range', min: 1, max: MAX_CAST_LEVEL, step: 1,
        help: 'upcasting: Fog Cloud grows by BG3\'s radii (4.5 m at 1st to 12 m at 6th, then +0.5 m a level); the rest keep their size',
        get: () => castLevel, set: v => { castLevel = Math.round(v); } },
      { key: 'smokeshadows', label: 'Shadows through smoke', type: 'toggle',
        help: 'the sun and every light pass through the smoke: the ground under a cloud is in its shadow (rebuilds)',
        get: () => shadows, set: v => flip(shadows, v, () => commands.smoke.run('shadows')) },
      { key: 'smokestrength', label: 'Lamp / bounce density', type: 'range', min: 0, max: 3, step: 0.05,
        help: 'density for lamp shadows and indirect light; sunlight is blocked after eight smoke boxes',
        get: () => smokeShared.strength.value, set: v => { smokeShared.strength.value = v; } },
      range('smokealbedo', 'Brightness', 'albedo', 0, 2, 0.01, 'on the spell\'s own colour'),
      range('smokewave', 'Breathing', 'wave', 0, 0.6, 0.01, 'how much larger than its cell a cube swells before shrinking back'),
      range('smokespeed', 'Breathing speed', 'speed', 0, 8, 0.1, 'how fast the wave moves through the cloud'),
      range('smokenoise', 'Noise', 'noise', 0, 1, 0.05, 'how much the soft texel noise thins and thickens the glow through'),
      range('smokedrift', 'Noise drift (m/s)', 'drift', 0, 1, 0.01, 'how fast the noise moves through the cloud'),
      range('smokems', 'Translucent glow', 'ms', 0, 1, 0.05, 'glow from transmitted sunlight; eight smoke boxes fully block sunlight'),
      range('smokesun', 'Sun', 'sun', 0, 2, 0.05, 'the sun on the smoke'),
      range('smokelamps', 'Lamps', 'lamps', 0, 4, 0.05, 'the light list on the smoke'),
      range('smokesky', 'Sky', 'sky', 0, 2, 0.05, 'the sky on the smoke')
    ]}
  };
}
