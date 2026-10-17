import * as THREE from 'three/webgpu';
import { file, section, ok, near } from './lib/harness.mjs';
import { createEngine } from '../js/render/engine.js';
import { createSky } from '../js/render/features/sky.js';
import { sunForHour, DEFAULT_SKY_HOUR } from '../js/core/sky.js';
import { sunColourUniform } from '../js/gpu/sky.js';

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
  mirrors.commands.mirrors.run('only');
  ok("'only' turns it back on", mirrors.on && mirrors.options().mirrorOnly, true);

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
  let white = 0, rebuilds = 0;
  const dispatched = [];
  const gi = createGI({
    renderer: { compute: k => dispatched.push(k) }, cascades: () => cascades, lighting, sky, mirrors,
    sun: { uniforms: () => ({ biasUniform: uniform(1), fadeStartUniform: uniform(1), edgeFadeUniform: uniform(1),
                              ambientUniform: uniform(0.2) }),
           on: () => true, angular: () => 0.01, light: sunLight },
    sprites: { casters: () => [], types: [] }, terrainAlbedo: () => null,
    syncWhite: () => { white++; }, shadowsOn: () => true, rebuild: () => { rebuilds++; }
  });
  ok('no cascades yet: no volumes, no GI binding', gi.options().gi, null);

  // Two cascades, as the field makes them (only their origins are read here).
  const { createBoxGridAt } = await import('../js/core/boxgrid.js');
  const { createDistanceTexture, cascadeBindings } = await import('../js/gpu/field.js');
  cascades = [0, 1].map(l => { const g = createBoxGridAt(0, 0, null, l); return cascadeBindings(createDistanceTexture(g), g); });
  const binding = gi.options().gi;
  ok('then the LPV: one level per volume', binding.levels.length, 2);
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

  gi.commands.gi.run('only');
  ok("'only' views the bounce alone, on whiteworld", gi.only && white === 1, true);
  ok('a kernel flag: rebuilt', rebuilds, 3);
  ok('off', gi.commands.gi.run(false), 'GI off');
  ok('off leaves the frame alone', (gi.update(), dispatched.length), 5);
  gi.commands.gi.run(true, 2);
  ok('strength is a uniform write', gi.settings.settings.find(x => x.key === 'gistrength').get(), 2);
}
