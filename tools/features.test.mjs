import * as THREE from 'three/webgpu';
import { file, section, ok, near } from './lib/harness.mjs';
import { createEngine } from '../js/render/engine.js';
import { createSky } from '../js/render/features/sky.js';
import { sunForHour, DEFAULT_SKY_HOUR } from '../js/core/sky.js';
import { sunColourUniform } from '../js/gpu/sky.js';
import { createWorldClock, HOUR_TICKS } from '../js/core/clock.js';

file('features.test.mjs - render features, each on its own');

section('engine');
{
  const engine = createEngine();
  engine.add({ name: 'a', options: () => ({ x: 1 }) });
  engine.add({ name: 'b', options: () => ({ y: 2 }) });
  engine.add({ name: 'c' });
  ok('options merge every feature\'s slice', JSON.stringify(engine.options()), '{"x":1,"y":2}');
  ok('a feature by name', engine.get('b').name, 'b');
  let threw = false;
  try { engine.add({ name: 'a' }); } catch { threw = true; }
  ok('a name added twice is an error', threw, true);
}

section('sky');
{
  const scene = new THREE.Scene();
  const light = new THREE.DirectionalLight();
  light.position.set(...sunForHour(DEFAULT_SKY_HOUR));
  const moved = [];
  const sky = createSky({ scene, sunLight: light, moveSun: (x, y, z) => { moved.push([x, y, z]); light.position.set(x, y, z); } });
  ok('the background is the sky', !!scene.backgroundNode, true);
  ok('starts at the default hour', sky.hour, DEFAULT_SKY_HOUR);
  ok('its options are its bindings', sky.options().sky, sky.bindings);
  const first = sky.bindings.sh.array[0].clone();
  ok('projected at once', first.lengthSq() > 0, true);

  ok('bxb.time says what it did', sky.commands.time.run(18), 'time 18.00h, sun moved');
  ok('the sun moved along its arc', JSON.stringify(moved[0]), JSON.stringify(sunForHour(18)));
  ok('the sky re-projected', sky.bindings.sh.array[0].equals(first), false);
  ok('the plain light takes the sun\'s colour', light.color.equals(new THREE.Color().setRGB(
    sunColourUniform.value.r, sunColourUniform.value.g, sunColourUniform.value.b)), true);
  sky.commands.time.run(-1, false);
  near('the clock wraps', sky.hour, 23, 1e-9);
  ok('without moving the sun', moved.length, 1);

  // At noon, where the sun's glow is in the sky.
  sky.commands.time.run(12, false);
  const sh = () => sky.bindings.sh.array.map(v => v.toArray().join()).join('|');
  const before = sh();
  sky.update();
  ok('unchanged hour and sun: no re-projection', sh(), before);
  light.position.set(0.8, 0.3, -0.5);
  sky.update();
  ok('the sun moved elsewhere: re-projected', sh() === before, false);

  sky.settings.settings[0].set(6);
  ok('the setting runs the command', sky.settings.settings[0].get(), 6);
}

section('sky on a running world clock');
{
  const scene = new THREE.Scene();
  const light = new THREE.DirectionalLight();
  const clock = createWorldClock(9);
  const moved = [];
  const sky = createSky({ scene, sunLight: light, clock,
                          moveSun: (x, y, z) => { moved.push([x, y, z]); light.position.set(x, y, z); } });
  ok('reads the clock\'s hour', sky.hour, 9);
  for (let i = 0; i < HOUR_TICKS; i++) clock.tick();
  sky.update();
  ok('an hour of ticks later, an hour later', sky.hour, 10);
  ok('the sun followed it', JSON.stringify(moved.at(-1)), JSON.stringify(sunForHour(10)));
  const n = moved.length;
  clock.tick();
  sky.update();
  ok('a tick is below the hour\'s resolution: the sun stays', moved.length, n);
  ok('bxb.daycycle stops it', sky.commands.daycycle.run(false), 'day cycle off, 10.00h');
  for (let i = 0; i < HOUR_TICKS; i++) clock.tick();
  sky.update();
  ok('stopped: the hour holds', sky.hour.toFixed(3), '10.000');
  sky.commands.time.run(15);
  ok('bxb.time sets the world clock', clock.hour(), 15);
}

section('lights');
{
  const { createLights } = await import('../js/render/features/lights.js');
  const { MAX_LIGHTS, TORCH_HEIGHT } = await import('../js/core/lights.js');
  const scene = new THREE.Scene();
  const bob = new THREE.Object3D();
  bob.position.set(3, 1, 3);
  let right = false;
  const packed = [];
  const lighting = createLights({
    scene, shadowsOn: () => true, mirrorRects: () => [],
    holder: { sprite: bob, facingRight: () => right },
    cards: { active: () => true, pack: (l, used, room) => { packed.push(l.name); return 1; }, commit() {} }
  });
  ok('its options are the light list', lighting.options().lights, lighting.bindings());
  lighting.update();
  ok('nothing lit, nothing listed', lighting.bindings().count.value, 0);

  ok('the torch lights', lighting.commands.torch.run(12).startsWith('torch on'), true);
  ok('and says so', lighting.torchOn, true);
  lighting.moveTorch();
  lighting.update();
  ok('one row: the torch', lighting.bindings().count.value, 1);
  near('held at torch height', lighting.bindings().data.array[0].y, 1 + TORCH_HEIGHT, 1e-6);
  ok('its sprite cards packed', packed.join(), 'torch');
  ok('one mirror-row flag per row', lighting.mirrorRows.length, 1);

  const x0 = lighting.bindings().data.array[0].x, z0 = lighting.bindings().data.array[0].z;
  right = true;
  lighting.moveTorch();
  lighting.update();
  const moved = lighting.bindings().data.array[0];
  ok('the torch swaps hands as Bob turns', moved.x !== x0 || moved.z !== z0, true);

  const before = scene.children.length;
  lighting.commands.lamp.run(9, '#ff0000', 5, 2, 5);
  ok('a lamp has a marker', scene.children.length, before + 1);
  lighting.update();
  ok('two rows', lighting.bindings().count.value, 2);
  ok('listed', lighting.commands.lamps.run().includes('1: lamp L9 #ff0000'), true);
  lighting.commands.lamps.run('pop');
  ok('pop removes it and its marker', scene.children.length, before);

  ok('budgets are clamped', lighting.commands.shadowbudget.run(99).startsWith(`shadow budget ${MAX_LIGHTS}`), true);
  lighting.commands.lightcutoff.run(0.02);
  near('the cutoff is a uniform write', lighting.bindings().cutoff.value, 0.02, 1e-9);
  ok('perf settings read it back', lighting.perfSettings.find(s => s.key === 'cutoff').get(), lighting.bindings().cutoff.value);

  lighting.settings.settings.find(s => s.key === 'torch').set(false);
  ok('the panel toggle turns it off', lighting.torchOn, false);
  lighting.update();
  ok('and it leaves the list', lighting.bindings().count.value, 0);
}

section('mirrors');
{
  const { createLights } = await import('../js/render/features/lights.js');
  const { createMirrors } = await import('../js/render/features/mirrors.js');
  const { terrainTextures } = await import('../js/render/render.js');
  const { MATERIALS } = await import('../js/core/materials.js');
  const { createTestArea, World } = await import('../js/core/world.js');
  const { glassBackGlare } = await import('../js/gpu/glass.js');
  createTestArea(40, 40);
  const sunLight = new THREE.DirectionalLight();
  sunLight.position.set(0.3, 0.8, 0.4);
  const lighting = createLights({ scene: new THREE.Scene(), shadowsOn: () => true, mirrorRects: () => [],
                                  holder: { sprite: new THREE.Object3D(), facingRight: () => false },
                                  cards: { active: () => false } });
  let rebuilds = 0;
  const focus = new THREE.Vector3(20, 0, 20);
  const mirrors = createMirrors({
    sunLight, sunOn: () => true, lighting, focus,
    footprint: () => ({ origin: { x: -100, z: -100 }, level: 4 }),
    rebuild: () => { rebuilds++; }
  });
  ok('options: on, the bindings', mirrors.options().mirrors, mirrors.bindings);

  mirrors.update();
  ok('no rectangles before the textures are in', mirrors.rects, null);

  // The polished material's _s, as the texture load would leave it.
  const saved = { ready: terrainTextures.ready, spec: terrainTextures.specRgba };
  terrainTextures.ready = true;
  terrainTextures.specRgba = MATERIALS.map(m => (m.id === 'ironPlate' ? new Uint8Array([245, 230, 0, 255]) : null));
  mirrors.update();
  ok('then built from the world', mirrors.rects.length > 0, true);
  ok('and packed for the shader', mirrors.bindings.count.value > 0, true);

  ok('off is a rebuild', mirrors.commands.mirrors.run().startsWith('mirror light off'), true);
  ok('rebuilt once', rebuilds, 1);
  ok('options: no mirrors', mirrors.options().mirrors, null);
  mirrors.commands.mirrors.run();
  ok('on again', mirrors.on, true);

  const rects = mirrors.rects;
  mirrors.commands.glassglare.run(2);
  ok('back-face glare: a uniform', glassBackGlare.value, 1);
  ok('and the rectangles are built again', mirrors.rects, null);
  mirrors.update();
  ok('on the next update', mirrors.rects !== null && mirrors.rects !== rects, true);
  mirrors.glareSetting.set(1);
  ok('the setting runs the command', mirrors.glare, 1);

  mirrors.invalidate();
  ok('a world edit drops them', mirrors.rects, null);
  terrainTextures.ready = saved.ready;
  terrainTextures.specRgba = saved.spec;
  World.clear();
}

section('gi');
{
  const { createLights } = await import('../js/render/features/lights.js');
  const { createMirrors } = await import('../js/render/features/mirrors.js');
  const { createGI } = await import('../js/render/features/gi.js');
  const { uniform } = await import('three/tsl');
  const scene = new THREE.Scene();
  const sunLight = new THREE.DirectionalLight();
  sunLight.position.set(0.3, 0.8, 0.4);
  const lighting = createLights({ scene, shadowsOn: () => true, mirrorRects: () => [],
                                  holder: { sprite: new THREE.Object3D(), facingRight: () => false },
                                  cards: { active: () => false } });
  const sky = createSky({ scene, sunLight, moveSun() {} });
  const mirrors = createMirrors({ sunLight, sunOn: () => true, lighting, focus: new THREE.Vector3(),
                                  footprint: () => null, rebuild() {} });
  let cascades = [];
  let rebuilds = 0;
  const dispatched = [];
  const gi = createGI({
    renderer: { compute: k => dispatched.push(k) }, cascades: () => cascades, lighting, sky, mirrors,
    sun: { uniforms: () => ({ biasUniform: uniform(1), fadeStartUniform: uniform(1), edgeFadeUniform: uniform(1),
                              ambientUniform: uniform(0.2) }),
           on: () => true, angular: () => 0.01, light: sunLight },
    sprites: { casters: () => [], types: [] }, terrainAlbedo: () => null,
    shadowsOn: () => true, rebuild: () => { rebuilds++; }
  });
  ok('no cascades yet: no volumes, no GI binding', gi.options().gi, null);

  // Three cascades, as the field makes them (only their origins are read here).
  const { createFieldAt } = await import('../js/core/field.js');
  const { createDistanceTexture, cascadeBindings } = await import('../js/gpu/field.js');
  cascades = [0, 1].map(l => { const g = createFieldAt(0, 0, null, l); return cascadeBindings(createDistanceTexture(g), g); });
  ok('two cascades: no volumes yet (the far one lies over C2)', gi.options().gi, null);
  { const g = createFieldAt(0, 0, null, 2); cascades.push(cascadeBindings(createDistanceTexture(g), g)); }
  const binding = gi.options().gi;
  ok('then the LPV: one level per volume', binding.levels.length, 2);
  ok('2 x 2 a block over C1, 1 x 1 over C2', binding.levels.map(l => l.cellMetres).join(','), '0.75,1.5');
  ok('texel GI is off in LPV mode', gi.options().texelGI, null);

  gi.update();
  ok('a frame dispatches the near volume', dispatched.length, 1);
  gi.update();
  ok('and the far one every other frame', dispatched.length, 3);
  gi.commands.gifar.run();
  gi.update();
  ok('every frame with gifar off', dispatched.length, 5);

  const mode = gi.settings.settings.find(x => x.key === 'gimode');
  mode.set('texel');
  ok('texel mode: a rebuild', rebuilds, 1);
  ok('and the texel tables instead', gi.options().gi === null && !!gi.options().texelGI, true);
  ok('with the strength uniform on them', !!gi.options().texelGI.strength, true);
  mode.set('lpv');

  ok('back to the LPV: rebuilt', rebuilds, 2);
  ok('off', gi.commands.gi.run(false), 'GI off');
  ok('off leaves the frame alone', (gi.update(), dispatched.length), 5);
  gi.commands.gi.run(true, 2);
  ok('strength is a uniform write', gi.settings.settings.find(x => x.key === 'gistrength').get(), 2);
}

section('debug view');
{
  const { createDebugView, VIEWS, resolveView } = await import('../js/render/features/debugview.js');
  const { splitDebug } = await import('../js/gpu/shade.js');
  let on = false, turnedOn = 0, rebuilds = 0;
  const dv = createDebugView({ shadowsOn: () => on, shadows: () => { turnedOn++; on = true; },
                               rebuild: () => { rebuilds++; } });
  ok('starts on the final composite', dv.current, 'final');
  ok('which compiles no view', JSON.stringify(dv.options()), '{"view":null,"glassView":null}');
  ok('ids are unique', new Set(VIEWS.map(v => v.id)).size, VIEWS.length);
  ok('each sets exactly one thing (or none: final)',
     VIEWS.every(v => [v.view, v.glassView, v.splitDebug].filter(Boolean).length === (v.id === 'final' ? 0 : 1)), true);
  ok('an unknown id resolves to nothing', resolveView('nope'), null);

  dv.commands.view.run('sdf');
  ok('a view with the shading off turns it on', turnedOn, 1);
  ok('and is in the options', dv.options().view, 'sdf');
  dv.commands.view.run('glass-path');
  ok('a glass view: glassView, no view', `${dv.options().view}/${dv.options().glassView}`, 'null/path');
  ok('rebuilt', rebuilds, 1);
  dv.commands.view.run('refl-distance');
  ok('a reflection-hit view is the uniform', splitDebug.value, 2);
  ok('and leaves the kernel view off', dv.options().glassView, null);
  dv.commands.view.run('refl-block');
  ok('between two of them: no rebuild', rebuilds, 2);
  dv.commands.view.run('next');
  ok("'next' wraps round to final", dv.current, 'final');
  ok('which clears the uniform', splitDebug.value, 0);
  dv.commands.view.run('prev');
  ok("'prev' steps back", dv.current, VIEWS[VIEWS.length - 1].id);
  ok('an unknown view is refused', dv.commands.view.run('nope').startsWith('unknown view'), true);
  ok('the setting lists every view', dv.setting.options.length, VIEWS.length);
  dv.setting.set('final');
  ok('and sets through the command', dv.current, 'final');
  ok('the listing marks the current one', dv.commands.view.run().split('\n')[0].startsWith('>'), true);
  const press = (code, altKey, shiftKey = false) => {
    const e = { code, altKey, shiftKey, prevented: false, preventDefault() { this.prevented = true; } };
    return [dv.key(e), e.prevented];
  };
  const log = console.log;
  console.log = () => {};
  ok('Alt+B is taken', press('KeyB', true).join(), 'true,true');
  ok('and steps to the next view', dv.current, VIEWS[1].id);
  press('KeyB', true, true);
  ok('Alt+Shift+B steps back', dv.current, 'final');
  console.log = log;
  ok('plain B is left alone', press('KeyB', false).join(), 'false,false');
  ok('as is another Alt key', press('KeyX', true).join(), 'false,false');
}

section('texel cache');
{
  const { createTexelCache } = await import('../js/render/features/texelcache.js');
  let on = false, rebuilds = 0;
  const tc = createTexelCache({
    renderer: { getDrawingBufferSize: v => v.set(1920, 1080) }, scene: new THREE.Scene(),
    camera: new THREE.PerspectiveCamera(50, 16 / 9, 0.1, 500), rig: { distance: 20 },
    shadowsOn: () => on, activeSet: () => null, gi: {}, rebuild: () => { rebuilds++; }
  });
  tc.splitOn = false;
  ok('split off: no split in the options', JSON.stringify(tc.options()), '{"split":null,"splitResolve":true,"splitLegacy":false}');
  ok('on by default', tc.on, true);
  ok('a fixed spacing sizes the target', tc.commands.texelcache.run(4), 'texel cache on, 1/4 per axis');
  ok('480 x 270 at 1/4', `${tc.cache.rt.width}x${tc.cache.rt.height}`, '480x270');
  ok('shading off: nothing rebuilt', rebuilds, 0);
  tc.missesSetting.set(true);
  ok('the misses view is the cache flag', tc.cache.showMisses, true);
  on = true;
  ok('off', tc.commands.texelcache.run(), 'texel cache off');
  ok('rebuilt with the shading on', rebuilds, 1);
  ok('off: the key says so', JSON.stringify(tc.keyParts()), '[false,null]');
  ok('the settings run the command', (tc.settings[0].set(true), tc.on), true);
  ok('nothing to draw without a set', (tc.render(), rebuilds), 2);
}

section('sprite atlas');
{
  const { createSpriteAtlas } = await import('../js/render/features/spriteatlas.js');
  const { addSpriteTangent } = await import('../js/render/sprites.js');
  const art = { image: { width: 0, height: 32 } };
  const mesh = new THREE.Mesh(addSpriteTangent(new THREE.PlaneGeometry(1, 2)), new THREE.MeshBasicMaterial({ map: art }));
  let on = false, rebuilds = 0;
  const sa = createSpriteAtlas({ renderer: {}, scene: new THREE.Scene(), camera: new THREE.PerspectiveCamera(),
                                 meshes: () => [mesh], shadowsOn: () => on, activeSet: () => null,
                                 rebuild: () => { rebuilds++; } });
  ok('not ready while the art loads', sa.ready(), false);
  art.image.width = 16;
  ok('ready once it has', sa.ready(), true);
  sa.ensure();
  ok('the sprite gets its twin', !!mesh.userData.spriteAtlasTwin, true);
  ok('the target is made', sa.bindings.rt.width, 512);
  on = true;
  ok('off', sa.commands.spriteatlas.run(), 'sprite atlas off: every pixel shades');
  ok('rebuilt', rebuilds, 1);
  ok('off: not ready', sa.ready(), false);
}

section('cards');
{
  const { createCards } = await import('../js/render/features/cards.js');
  const sprite = () => { const m = new THREE.Object3D(); m.visible = true; return m; };
  const player = sprite(), enemy = sprite();
  let on = false, rebuilds = 0, ready = 0;
  const light = new THREE.DirectionalLight();
  light.position.set(0.4, 0.8, 0.3);
  const cards = createCards({
    renderer: {}, camera: new THREE.PerspectiveCamera(), pivot: new THREE.Object3D(), sunLight: light,
    sunAngular: () => 0.01,
    sprites: { player: { mesh: player, texture: {} }, enemy: { mesh: enemy, texture: {} },
               trees: () => [], treeTexture: {}, mirrored: () => false },
    normalMap: () => null, grids: () => [], textures: () => [], shadowsOn: () => on,
    rebuild: () => { rebuilds++; }, onReady: () => { ready++; }
  });
  ok('no atlas yet: no cards in the options', JSON.stringify(cards.options()),
     '{"cards":null,"lightCards":null,"viewCards":null}');
  cards.update();
  ok('textures not loaded: still waiting', `${cards.ready}/${ready}/${cards.casters()}`, 'false/0/null');
  ok('the point lights pack nothing', cards.lightSlices.active(), false);
  cards.updateOccluders();
  ok('proxies on, shading off', cards.commands.proxies.run(), 'proxies on - run bxb.shadows() to see them');
  cards.updateOccluders();
  ok('and off again', cards.commands.proxies.run(), 'proxies off - sprites cast nothing, terrain still does');
  ok('cardface cycles to x', cards.commands.cardface.run().startsWith('cutouts LOCKED to the XY'), true);
  ok('and back to the sun', cards.commands.cardface.run('sun').startsWith('cutouts face the SUN'), true);
  on = true;
  ok('cards off rebuilds', (cards.commands.cards.run(), rebuilds), 1);
  ok('on again says why there is no atlas',
     cards.commands.cards.run().startsWith('cards on - no atlas. pending 3'), true);
  ok('parity waits for cards', await cards.commands.cardparity.run(), 'no cards yet - bxb.shadows() first, then retry');
  ok('the Sprites section', cards.settings.settings.map(x => x.key).join(), 'cards,cardface,proxies');
  ok('no card yet: no normal map', cards.normalFor(player), null);
}

section('sun');
{
  const { createSun } = await import('../js/render/features/sun.js');
  const light = new THREE.DirectionalLight();
  light.position.set(0.4, 0.8, 0.3);
  let on = false, rebuilds = 0, reorigins = 0;
  const sun = createSun({ sunLight: light, shadowsOn: () => on, rebuild: () => { rebuilds++; },
                          reorigin: () => { reorigins++; } });
  ok('a cone, one ray, on', `${sun.cone}/${sun.rays}/${sun.on}`, 'true/1/true');
  ok('no uniforms until asked', sun.options().biasUniform, null);
  ok('the GI asks: made', !!sun.uniforms().biasUniform, true);
  ok('in the options from then on', sun.options().biasUniform, sun.uniforms().biasUniform);
  ok('bxb.sun moves the light', sun.commands.sun.run(90, 30), 'sun toward 0.87,0.50,0.00');
  ok('no lean: no re-origin', reorigins, 0);
  sun.commands.gridbias.run(2);
  ok('a lean re-origins', reorigins, 1);
  sun.move(0, 1, 0);
  ok('and so does moving the sun with a lean', reorigins, 2);
  await sun.commands.soften.run(16);
  ok('sixteen sampled rays', `${sun.cone}/${sun.rays}`, 'false/16');
  ok('shading off: nothing rebuilt', rebuilds, 0);
  on = true;
  sun.commands.daylight.run(false);
  ok('daylight off rebuilds', `${sun.on}/${rebuilds}`, 'false/1');
  ok('ambient is a uniform write', (sun.commands.ambient.run(0.3), rebuilds), 1);
  ok('the profiler flag', (sun.march = false, sun.options().sunMarch), false);
  ok('the settings', [...sun.shadowSettings, ...sun.settings.settings, ...sun.shadingSettings].length, 14);
}

section('shadow sets');
{
  const { createShadowSets } = await import('../js/render/shadowsets.js');
  let removed = 0;
  const sets = createShadowSets({
    renderer: {}, scene: new THREE.Scene(), camera: new THREE.PerspectiveCamera(), options: () => ({ x: 1 }),
    texel: { remove: () => { removed++; }, keyParts: () => [false, null] },
    spriteAtlas: { ready: () => false }, sun: { ensureUniforms() {} },
    sprites: { meshes: () => [], clipping: () => [], normalFor: () => null }
  });
  ok('nothing showing at first', sets.active, null);
  ok('nothing kept: a build would compile', sets.wouldCompile(), true);
  sets.remove();
  ok('remove puts the plain materials back', removed, 1);
  ok('keeping off', sets.commands.shadowcache.run(), 'shadow sets disposed when replaced');
  ok('the setting reads it', sets.setting.get(), false);
  ok('on again', sets.commands.shadowcache.run(), 'shadow sets kept (newest 2)');
}

section('tone map');
{
  const { createToneMap } = await import('../js/render/features/tonemap.js');
  const { METER_LOG_MIN, METER_LOG_SCALE } = await import('../js/core/exposure.js');
  // The meter's words for a frame of uniform luminance L.
  const words = L => new Uint32Array([Math.floor((Math.log2(L) - METER_LOG_MIN) * METER_LOG_SCALE) * 100, 100]).buffer;
  let lum = 0.18 / 8, computes = 0, source = null;
  const registered = new Map();
  const renderer = { toneMapping: null, toneMappingExposure: 1, library: { addToneMapping: (fn, k) => registered.set(k, fn) },
                     compute: () => { computes++; }, getArrayBufferAsync: async () => words(lum) };
  let effects = false;
  const tm = createToneMap({ renderer, effectsOn: () => effects, meterSource: () => source });
  ok('the basic mode is never tone mapped', renderer.toneMapping, THREE.NoToneMapping);
  ok('and says so', tm.commands.exposure.run().endsWith('(the basic mode is never tone mapped)'), true);
  effects = true;
  tm.update(990);
  ok('our AgX with the effects on', renderer.toneMapping, THREE.CustomToneMapping);
  ok('registered as the custom tone mapping', registered.has(THREE.CustomToneMapping), true);
  tm.update(1000);
  ok('no source: nothing metered, exposure 1', `${computes}/${renderer.toneMappingExposure}`, '0/1');
  source = { split: {} };
  // A stand-in source: the kernels are only built into WGSL by the renderer (shaders.test).
  tm.update(1016);
  ok('meters every eighth frame', computes, 0);
  for (let i = 0; i < 7; i++) tm.update(1032 + i * 16);
  ok('the eighth frame meters', computes, 1);
  await new Promise(r => setTimeout(r, 0));
  for (let t = 1200; t < 12000; t += 16) tm.update(t);
  near('a frame 8x too dark adapts to +3 EV, less the default half stop', tm.ev, 2.5, 0.01);
  near('the output pass gets 2^EV', renderer.toneMappingExposure, 2 ** 2.5, 0.1);
  ok('fixed exposure', tm.commands.exposure.run(-1).startsWith('AgX punchy, exposure fixed -1.00 EV'), true);
  ok('the base look', tm.commands.agxlook.run('base').startsWith('AgX base,'), true);
  ok('numbers are a custom look', tm.commands.agxlook.run(1.2, 1.25).startsWith('AgX custom,'), true);
  ok('the look select reads custom', tm.settings.settings.find(x => x.key === 'agxlook').get(), 'custom');
  tm.settings.settings.find(x => x.key === 'agxlook').set('punchy');
  ok('and sets a preset', tm.settings.settings.find(x => x.key === 'agxlook').get(), 'punchy');
  tm.update(12016);
  ok('applied at once', renderer.toneMappingExposure, 0.5);
  ok('off', tm.commands.tonemap.run('off'), 'tone mapping off (exposure has no effect)');
  ok('the renderer is told', renderer.toneMapping, THREE.NoToneMapping);
  ok('toggles back to AgX', (tm.commands.tonemap.run(), tm.mode), 'agx');
  ok('an unknown mode is ignored', (tm.commands.tonemap.run('aces'), tm.mode), 'agx');
  ok('the section', tm.settings.settings.length, 11);
}
