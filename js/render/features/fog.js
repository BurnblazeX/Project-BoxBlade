// --- Fog: froxels lit by the LPV (gpu/fog.js) ---
//
// Off by default, a test. Needs the marched shading, the LPV as the bounce
// (the fog's light is the LPV volumes': gpu/gi.js fog kernel) and split
// shading's full-res G-buffer (the depth it fogs each pixel by). Laid over the
// frame by one fullscreen quad, drawn last on the default layer - the texel
// cache and sprite atlas passes, which draw other layers, never see it - so
// turning it on compiles one pipeline and touches no material.
//
// ctx:
//   renderer, scene, camera
//   texel          the texel cache feature: gbuffer()
//   gi             the GI feature: lpvBinding()
//   sky            the sky feature: bindings
//   sun            { light (its position is the sun's direction), on(),
//                    uniforms() - the shading's ambientUniform }
//   shadowsOn()    whether the marched shading is on
import { createFroxelFog, DEFAULT_FOG } from '../../gpu/fog.js';

export function createFog({ renderer, scene, camera, texel, gi, sky, sun, shadowsOn }) {
  let on = false;
  let fog = null, fogKey = null;
  let quad = null, quadKey = null;
  const params = { ...DEFAULT_FOG };
  const t0 = performance.now();

  // The froxels, made for the LPV's current volumes (a rebuilt LPV has new
  // textures, so the fog is made again).
  function ensure() {
    const b = gi.lpvBinding();
    if (!b) return null;
    const key = b.levels.map(l => l.fogSunTex.uuid).join();
    if (fog && fogKey === key) return fog;
    if (fog) fog.dispose();
    dropQuad();
    fog = createFroxelFog({ gi: b, sky: sky.bindings, ambient: sun.uniforms().ambientUniform });
    fogKey = key;
    for (const [k, v] of Object.entries(params)) fog.uniforms[k].value = v;
    return fog;
  }

  function dropQuad() {
    if (!quad) return;
    scene.remove(quad);
    quad.material.dispose();
    quad = quadKey = null;
  }

  const active = () => on && shadowsOn();

  // Before the main render: the froxels for this frame's camera, and the quad
  // over the frame - shown only when there is a G-buffer to read.
  function update() {
    const gb = active() && ensure() ? texel.gbuffer() : null;
    if (gb && (!quad || quadKey !== gb.id.uuid + gb.depth.uuid)) {
      dropQuad();
      quad = fog.makeQuad(gb);
      quadKey = gb.id.uuid + gb.depth.uuid;
      scene.add(quad);
    }
    if (quad) quad.visible = !!gb;
    if (!gb) return;
    const u = fog.uniforms;
    camera.updateMatrixWorld();
    u.projInv.value.copy(camera.projectionMatrixInverse);
    u.camWorld.value.copy(camera.matrixWorld);
    u.camPos.value.setFromMatrixPosition(camera.matrixWorld);
    camera.getWorldDirection(u.camFwd.value);
    u.sunDir.value.copy(sun.light.position).normalize();
    const ambient = sun.uniforms().ambientUniform.value;
    u.sunLight.value = sun.on() ? 1 - ambient : 0;
    u.time.value = (performance.now() - t0) / 1000;
    renderer.compute(fog.kernels);
  }

  const set = (k, v) => { params[k] = v; if (fog) fog.uniforms[k].value = v; };
  const range = (key, label, k, min, max, step, help) => ({ key, label, type: 'range', min, max, step, help,
    get: () => params[k], set: v => set(k, v) });

  return {
    name: 'fog',
    update,
    // The LPV lights its air while this is on.
    on: active,
    commands: {
      fog: {
        help: 'froxel fog lit by the LPV (a test): on/off, and the density per metre',
        usage: 'bxb.fog()  toggles  |  bxb.fog(true, 0.08)',
        run: (a = null, density = null) => {
          on = a === null ? !on : !!a;
          if (density !== null) set('density', Math.max(0, Number(density)));
          if (!on) return 'fog off';
          if (!shadowsOn()) return 'fog on - but it needs the advanced effects (R)';
          if (!gi.lpvBinding()) return 'fog on - but it needs the LPV as the GI method';
          if (!texel.gbuffer()) return 'fog on - but it needs the full-res G-buffer (split shading, texel cache)';
          return `fog on - density ${params.density}/m, base ${params.base} m, falloff ${params.falloff} m`;
        }
      }
    },
    settings: { title: 'Fog', settings: [
      { key: 'fog', label: 'Fog (froxels, LPV-lit)', type: 'toggle',
        help: 'a test: needs the advanced effects and the LPV as the GI method; turning it on recompiles once',
        get: () => on, set: v => { on = !!v; } },
      range('fogdensity', 'Density (/m)', 'density', 0, 0.3, 0.005, 'extinction per metre below the base height'),
      range('fogbase', 'Base height (m)', 'base', -5, 20, 0.25, 'world height below which the fog is full'),
      range('fogfalloff', 'Height falloff (m)', 'falloff', 0.25, 20, 0.25, 'metres above the base for the density to fall by e'),
      range('fognoise', 'Noise', 'noise', 0, 1, 0.05, 'how much the soft texel noise thins and thickens it'),
      range('fognoisesize', 'Noise size (m)', 'noiseSize', 0.0625, 1, 0.0625, 'the finest noise cell - 0.125 is a texel'),
      range('fogwind', 'Drift (m/s)', 'wind', 0, 2, 0.05, 'how fast the noise moves'),
      range('fogalbedo', 'Albedo', 'albedo', 0, 1, 0.05, 'the share of the light it stops that it scatters'),
      range('fogg', 'Sun forward scatter (g)', 'g', 0, 0.95, 0.05, 'how bright it glows looking toward the sun'),
      range('fogms', 'Multiple scattering', 'ms', 0, 1, 0.05, 'light scattered more than once: 0 single scattering, 1 full'),
      range('foglamps', 'Lamps', 'lamps', 0, 4, 0.05, 'the light list in the fog'),
      range('fogsky', 'Sky and bounce', 'sky', 0, 4, 0.05, 'the sky and the LPV bounce in the fog')
    ]}
  };
}
