// --- The sun: a marched cone ---
//
// The CSM is gone; see gpu/sun.js for why. The sun is now the same sphere-traced
// ray bxb.shadows always cast, widened into a cone over the solar disc, so
// softening and shadowing are one technique rather than a shadow plus a filter.
//
// ctx:
//   sunLight     the DirectionalLight: its position is the sun's direction
//   shadowsOn()  whether the marched shading is on
//   rebuild()    the shading rebuilt, when a kernel flag changed
//   reorigin()   the cascades re-originate next frame (the footprint's lean moved)
import { uniform } from 'three/tsl';
import { SUN_BIAS_BLOCKS } from '../../core/boxgrid.js';
import { CHUNK_SIZE } from '../../core/world.js';
import { SUN_ANGULAR_SIZE, penumbraTexels } from '../../core/sun.js';
import { createBayerTexture, writeSunUniforms, DEFAULT_AMBIENT } from '../../gpu/sun.js';
import { SURFACE_BIAS_VOXELS, SHADOW_FADE_START, EDGE_FADE_VOXELS } from '../../gpu/field.js';

export function createSun({ sunLight, shadowsOn, rebuild, reorigin }) {
  // sunRays is the sample count and is a COMPILE-TIME property of the kernel: 1 is
  // the hard reference, and anything above it is soft. Tracked alongside the other
  // pass-shape flags because reusing a pass built at a different count is silent -
  // everything runs and nothing on screen changes, which is exactly what toggling
  // the old bxb.csm() off and on used to do.
  let sunRays = 1;
  // Which way the penumbra is obtained. false samples the solar disc with sunRays
  // rays; true gets the whole thing from a single cone trace against the widened
  // field. Both are the same march and the same field - see gpu/sun.js for the one
  // thing the cone gives up, which is occluder shape. The cone is the default:
  // it is what R turns on, and so what the load-time compile builds.
  let sunCone = true;
  let sunAngular = SUN_ANGULAR_SIZE;
  let sunBayer = null;
  let sunSteps = null;    // quantisation levels for the soft tail
  let sunBias = null;     // ray origin lift off the face, in voxels
  // Where a shadow starts dissolving as its ray runs out of reach. Uniforms rather
  // than constants because this is the knob that decides whether the march's limits
  // are visible, and it has to be found by eye.
  let sunFadeStart = null, sunEdgeFade = null;
  // How N.L is applied; 'lambert' (raw N.L) by default. 'ground' normalises against what a horizontal surface
  // receives, so the sun's ANGLE sets shadow direction and length rather than the
  // whole scene's brightness - see sunShadeTSL. A kernel branch, so changing it
  // rebuilds; ambient beside it is a plain uniform.
  let shadeMode = 'lambert';
  // How far the explore footprint leans toward the sun, in blocks. A plain number
  // rather than a uniform: it changes WHERE the grid is baked, so it takes effect
  // on the next re-origin, not on the next shade.
  let sunGridBias = SUN_BIAS_BLOCKS;
  let sunAmbient = null;
  // The sun, off. Not a zero direction or a black colour - a KERNEL flag, so the
  // material has no sun march in it at all and a torch-only scene pays for one
  // light instead of two. This is what a cave or a night is.
  let sunOn = true;
  // Quantisation is a KERNEL flag, not a uniform, so changing it rebuilds. Off by
  // default now: it was inherited from the sampled path, where a fixed sample
  // pattern genuinely does band, and applied to the cone trace, whose output is
  // analytic and already smooth. Dithering a smooth gradient only adds crosshatch.
  let sunQuantise = false;
  // The sun's shadow march. Off only inside bxb.gpuprofile, to price it.
  let sunMarch = true;
  // The showing set's sun uniforms, so moving the sun can rewrite them.
  let shadowSun = null;

  function ensureSunUniforms() {
    sunBayer = sunBayer || createBayerTexture();
    sunSteps = sunSteps || uniform(4);
    sunBias = sunBias || uniform(SURFACE_BIAS_VOXELS);
    sunFadeStart = sunFadeStart || uniform(SHADOW_FADE_START);
    sunEdgeFade = sunEdgeFade || uniform(EDGE_FADE_VOXELS);
    sunAmbient = sunAmbient || uniform(DEFAULT_AMBIENT);
  }

  function bxbLight(x, y, z) {
    sunLight.position.set(x, y, z);
    // The cone's basis has to be rewritten WITH the direction. A basis left over
    // from the previous sun tilts the penumbra off to one side, which reads as a
    // softening bug rather than as a stale uniform - which is why this goes
    // through writeSunUniforms instead of setting the direction alone.
    if (shadowSun) writeSunUniforms(shadowSun, sunLight.position, sunAngular);
    // The footprint leans toward the sun, so moving the sun moves WHERE the field
    // should be baked, not just how it is lit. Dropping the remembered origin makes
    // the next frame notice and re-origin; without this the grid would keep the
    // previous sun's lean until the player happened to step a block. With no
    // lean (the default) the sun does not move the footprint at all, and the
    // forced re-origin would only rebake it for nothing.
    if (sunGridBias) reorigin();
    return 'sun toward ' + x.toFixed(2) + ',' + y.toFixed(2) + ',' + z.toFixed(2);
  }

  const commands = {
    soften: {
      help: 'distance-based soft sun. "cone" is one ray, a number is that many over the disc',
      usage: 'bxb.soften("cone")  |  bxb.soften(16)  |  bxb.soften(1) for the hard reference',
      run: async (mode = 'cone', a = sunAngular, steps = 0) => {
        // Two techniques, one seam. Both march the same field from the same origin
        // and both return visibility, so they can be swapped and diffed - which is
        // the whole reason the sampled one was built first even though the cone is
        // sixteen times cheaper.
        if (mode === 'cone') {
          sunCone = true;
          sunRays = 1;
        } else {
          sunCone = false;
          sunRays = Math.max(1, Math.min(64, Math.round(Number(mode) || 16)));
        }
        sunAngular = a;
        ensureSunUniforms();
        // steps = 0 means do not quantise at all, and it is the default. Passing a
        // count opts back into the stepped-and-dithered look.
        sunQuantise = steps > 0;
        if (sunQuantise) sunSteps.value = steps;
        if (shadowSun) writeSunUniforms(shadowSun, sunLight.position, sunAngular);

        // The technique and the ray count are both baked into the kernel, so this
        // is a material swap rather than a uniform write.
        if (shadowsOn()) { rebuild(); }

        if (!sunCone && sunRays === 1) {
          return 'hard sun - one ray down the cone axis, which is exactly what ' +
                 'bxb.shadows always cast. This is the reference every soft ' +
                 'result is judged against.';
        }
        // What to expect, in texels, so the result can be checked rather than
        // admired: a penumbra that does not grow with separation means the cone
        // basis is wrong, not that the angular size is too small.
        const at = s => penumbraTexels(s, a).toFixed(1);
        const expect = `Expect a penumbra of ${at(1)} texels at 1 m separation, ` +
          `${at(4)} at 4 m, ${at(8)} at 8 m. Contact stays hard. If it does not ` +
          `widen with separation the cone basis is wrong, not the angular size.`;
        if (sunCone) {
          return `soft sun by CONE TRACE - one ray, ${a} rad disc, ` +
                 `${sunQuantise ? steps + ' quantised steps' : 'smooth (no dither)'}. ` +
                 `Analytic rather than sampled, so it is smooth by construction, ` +
                 `and ~16x cheaper than sampling. What it gives up is occluder ` +
                 `SHAPE: it sees the nearest surface, not how much of the disc that ` +
                 `surface covers, so two thin occluders read as one and a shadow ` +
                 `through a narrow gap comes out slightly too dark. ` +
                 `bxb.soften(16) to diff against the sampled reference. ` + expect;
        }
        return `soft sun by SAMPLING - ${sunRays} rays over a ${a} rad disc, ` +
               `${sunQuantise ? steps + ' quantised steps' : 'no quantisation'}. ` +
               `This is the reference: it resolves ` +
               `occluder shape, at ${sunRays}x the rays. ` + expect;
      }
    },
    ambient: {
      help: 'toggle the ambient floor - off means unlit surfaces are black',
      usage: 'bxb.ambient()  |  bxb.ambient(false)  |  bxb.ambient(0.2) for a level',
      run: (on = null) => {
        ensureSunUniforms();
        const v = on === null ? (sunAmbient.value > 0 ? 0 : DEFAULT_AMBIENT)
                : (on === true ? DEFAULT_AMBIENT : (on === false ? 0 : Number(on)));
        // A uniform, so this is a write rather than a rebuild - it changes how
        // much light a surface gets, not whether the kernel computes one.
        sunAmbient.value = Math.max(0, Math.min(1, v));
        return sunAmbient.value > 0
          ? `ambient ${sunAmbient.value.toFixed(2)}`
          : 'ambient off - anything no light reaches is black. With bxb.daylight(false) ' +
            'too, the torch is the only thing you can see by.';
      }
    },
    daylight: {
      help: 'turn the sun off, leaving ambient plus whatever dynamic lights are lit',
      usage: 'bxb.daylight()  |  bxb.daylight(false)',
      run: (on = !sunOn) => {
        sunOn = !!on;
        if (shadowsOn()) { rebuild(); }
        return sunOn
          ? 'sun on'
          : 'sun off - ambient only, plus any dynamic light. The sun march is ' +
            'gone from the kernel entirely, so a torch-lit scene pays for one ' +
            'light rather than two. bxb.torch() if it is too dark to see.';
      }
    },
    gridbias: {
      help: 'how far the shading footprint leans toward the sun, in blocks',
      usage: 'bxb.gridbias(blocks)  |  bxb.gridbias(0) to centre it on the player',
      run: (blocks = SUN_BIAS_BLOCKS) => {
        sunGridBias = blocks;
        // Takes effect on the next re-origin, which dropping the remembered origin
        // forces on the next frame.
        reorigin();
        const half = Math.floor(CHUNK_SIZE / 2);
        const up = (half + blocks) * 1.5, down = (half - blocks) * 1.5;
        return `footprint leans ${blocks} blocks up-sun: ${up.toFixed(1)} m of ` +
               `caster range toward the sun, ${down.toFixed(1)} m away from it. ` +
               `Shadow rays travel TOWARD the sun, so range up-sun is what stops a ` +
               `long shadow being cut off; range down-sun holds nothing that can ` +
               `cast onto anything visible. The cost is that ground down-sun of you ` +
               `falls out of the job list and keeps its last shaded value instead ` +
               `of updating. bxb.gridbias(0) restores the centred footprint.`;
      }
    },
    shade: {
      help: 'how N.L is applied, and the ambient floor - both paths share one formula',
      usage: "bxb.shade('ground'|'lambert'|'flat', ambient?)",
      run: async (mode = 'lambert', ambient = DEFAULT_AMBIENT) => {
        const modes = ['ground', 'lambert', 'flat'];
        if (!modes.includes(mode)) return `mode must be one of ${modes.join(', ')}`;
        shadeMode = mode;
        ensureSunUniforms();
        sunAmbient.value = ambient;
        // Ambient is a uniform, the mode is a kernel branch - so only the latter
        // costs a rebuild.
        if (shadowsOn()) { rebuild(); }
        if (!shadowsOn()) return 'run bxb.shadows() first';
        const el = Math.asin(Math.max(-1, Math.min(1,
          sunLight.position.clone().normalize().y))) * 180 / Math.PI;
        const explain = {
          ground: 'flat ground under open sky reads FULLY LIT at any sun angle, ' +
                  'because N.L is normalised against what a horizontal surface ' +
                  'receives. The sun angle then sets shadow direction and length ' +
                  'rather than overall brightness. Faces turned away still fall to ' +
                  'ambient, so nothing goes flat.',
          lambert: 'raw N.L - physically right and DIM at a low sun, because every ' +
                   `up-facing surface is multiplied by sin(${el.toFixed(0)}) = ` +
                   `${Math.sin(el * Math.PI / 180).toFixed(2)}. This is what made ` +
                   'the shaded world look like it was in permanent partial shade.',
          flat: 'no N.L at all - what bxb.shadows() always did. Bright, but an ' +
                'unshadowed wall facing away from the sun reads exactly as bright ' +
                'as one facing it.'
        }[mode];
        return `shade mode '${mode}', ambient ${ambient}. ${explain}`;
      }
    },
    fade: {
      help: 'how a shadow dissolves where its ray runs out of reach, instead of stopping',
      usage: 'bxb.fade(startFraction, edgeVoxels)  |  bxb.fade(1, 0) to see the hard cut',
      run: async (start = SHADOW_FADE_START, edge = EDGE_FADE_VOXELS) => {
        ensureSunUniforms();
        sunFadeStart.value = start;
        sunEdgeFade.value = edge;
        // Both are uniforms, so a sweep is a re-shade rather than a rebuild.
        if (!shadowsOn()) return 'run bxb.shadows() first';
        return `shadows fade over the last ${((1 - start) * 100).toFixed(0)}% of the ` +
               `12 m march, and within ${edge} voxels (${(edge * 0.125).toFixed(2)} m) ` +
               `of the grid boundary. This is what stops a long shadow ending on a ` +
               `straight line: past those limits the ray does not know what is out ` +
               `there, and fading is the honest way to say so. bxb.fade(1, 0) turns ` +
               `it off and puts the hard cut back, to confirm that is what you were ` +
               `seeing. Lower start = longer, softer dissolve.`;
      }
    },
    sunbias: {
      help: 'how far a shadow ray starts off the surface, along the face normal, in voxels',
      usage: 'bxb.sunbias(voxels)',
      run: async (v = SURFACE_BIAS_VOXELS) => {
        ensureSunUniforms();
        sunBias.value = v;
        // A uniform, not a rebuild: this is the one knob that has to be found by
        // eye, so a sweep costs a re-shade.
        if (!shadowsOn()) return 'run bxb.shadows() first';
        return `ray origin lifted ${v} voxels (${(v * 0.125 * 100).toFixed(1)} cm) ` +
               `along the face normal. Raise until speckle clears, then stop - too ` +
               `much detaches contact shadows from their casters. Along the NORMAL ` +
               `and not the ray, so this is independent of sun angle.`;
      }
    },
    light: {
      help: 'set the sun direction (a vector pointing toward the sun)',
      usage: 'bxb.light(x, y, z)',
      // Goes through bxbLight so the cone basis is rewritten with the direction -
      // see there for why writing the direction alone is not enough now.
      run: (x, y, z) => bxbLight(x, y, z) +
        (shadowsOn() ? '' : ' (run bxb.shadows() to see it)')
    },
    sun: {
      help: 'set the sun by angle - elevation 15-25 gives long readable shadows',
      usage: 'bxb.sun(azimuthDeg, elevationDeg)',
      run: (az = 45, el = 20) => {
        const a = az * Math.PI / 180, e = el * Math.PI / 180;
        const x = Math.cos(e) * Math.sin(a), y = Math.sin(e), z = Math.cos(e) * Math.cos(a);
        return bxbLight(x, y, z);
      }
    }
  };

  function sunAngles() {
    const d = sunLight.position.clone().normalize();
    const az = (Math.atan2(d.x, d.z) * 180 / Math.PI + 360) % 360;
    const el = Math.asin(Math.max(-1, Math.min(1, d.y))) * 180 / Math.PI;
    return { az, el };
  }
  const softSteps = () => (sunQuantise && sunSteps ? sunSteps.value : 0);

  return {
    name: 'sun',
    options: () => ({
      sunDirection: sunLight.position,
      rays: sunRays, angular: sunAngular, cone: sunCone, quantise: sunQuantise,
      bayerTex: sunBayer, stepsUniform: sunSteps, biasUniform: sunBias,
      fadeStartUniform: sunFadeStart, edgeFadeUniform: sunEdgeFade,
      ambientUniform: sunAmbient, shadeMode, sun: sunOn, sunMarch
    }),
    // Its uniforms, made if need be - before a build reads options().
    ensureUniforms: ensureSunUniforms,
    // The march's uniforms, for the GI's own sun march.
    uniforms() {
      ensureSunUniforms();
      return { biasUniform: sunBias, fadeStartUniform: sunFadeStart, edgeFadeUniform: sunEdgeFade,
               ambientUniform: sunAmbient };
    },
    // A set put on screen: its sun uniforms are the live ones from now on.
    apply(set) {
      shadowSun = set.sun;
      writeSunUniforms(shadowSun, sunLight.position, sunAngular);
    },
    move: bxbLight,
    get on() { return sunOn; },
    get angular() { return sunAngular; },
    get cone() { return sunCone; },
    get rays() { return sunRays; },
    get gridBias() { return sunGridBias; },
    get march() { return sunMarch; },
    set march(v) { sunMarch = !!v; },
    commands,
    // For the Shadows section, after its own on/off.
    shadowSettings: [
      { key: 'daylight', label: 'Daylight (sun)', type: 'toggle',
        get: () => sunOn, set: v => commands.daylight.run(v) },
      { key: 'soften', label: 'Soft sun', type: 'select',
        options: [{ value: 'cone', label: 'cone trace' }, { value: 1, label: 'hard (1 ray)' },
                  { value: 4, label: '4 rays' }, { value: 16, label: '16 rays' },
                  { value: 32, label: '32 rays' }],
        help: 'cone is one analytic ray; a count samples that many over the disc',
        get: () => (sunCone ? 'cone' : sunRays),
        set: v => commands.soften.run(v, sunAngular, softSteps()) },
      { key: 'disc', label: 'Sun disc (rad)', type: 'range', min: 0, max: 0.2, step: 0.005,
        live: false, get: () => sunAngular,
        set: v => commands.soften.run(sunCone ? 'cone' : sunRays, v, softSteps()) },
      { key: 'steps', label: 'Quantise steps', type: 'range', min: 0, max: 8, step: 1,
        live: false, format: v => (v ? String(v) : 'off'), get: softSteps,
        set: v => commands.soften.run(sunCone ? 'cone' : sunRays, sunAngular, v) },
      { key: 'sunbias', label: 'Ray bias (voxels)', type: 'range', min: 0, max: 4, step: 0.05,
        get: () => (sunBias ? sunBias.value : SURFACE_BIAS_VOXELS), set: v => commands.sunbias.run(v) },
      { key: 'fadestart', label: 'Fade start', type: 'range', min: 0.5, max: 1, step: 0.01,
        get: () => (sunFadeStart ? sunFadeStart.value : SHADOW_FADE_START),
        set: v => commands.fade.run(v, sunEdgeFade ? sunEdgeFade.value : EDGE_FADE_VOXELS) },
      { key: 'edgefade', label: 'Edge fade (voxels)', type: 'range', min: 0, max: 24, step: 1,
        get: () => (sunEdgeFade ? sunEdgeFade.value : EDGE_FADE_VOXELS),
        set: v => commands.fade.run(sunFadeStart ? sunFadeStart.value : SHADOW_FADE_START, v) },
      { key: 'gridbias', label: 'Grid lean (blocks)', type: 'range', min: -6, max: 6, step: 1,
        live: false, get: () => sunGridBias, set: v => commands.gridbias.run(v) }
    ],
    settings: { title: 'Sun', settings: [
      { key: 'az', label: 'Azimuth', type: 'range', min: 0, max: 359, step: 1, live: false,
        format: v => v.toFixed(0) + '°',
        get: () => sunAngles().az, set: v => commands.sun.run(v, sunAngles().el) },
      { key: 'el', label: 'Elevation', type: 'range', min: 1, max: 89, step: 1, live: false,
        format: v => v.toFixed(0) + '°',
        get: () => sunAngles().el, set: v => commands.sun.run(sunAngles().az, v) }
    ]},
    // For the Shading section.
    shadingSettings: [
      { key: 'shade', label: 'N·L mode', type: 'select', options: ['ground', 'lambert', 'flat'],
        get: () => shadeMode,
        set: v => commands.shade.run(v, sunAmbient ? sunAmbient.value : DEFAULT_AMBIENT) },
      { key: 'ambient', label: 'Ambient', type: 'range', min: 0, max: 1, step: 0.01,
        get: () => (sunAmbient ? sunAmbient.value : DEFAULT_AMBIENT), set: v => commands.ambient.run(v) }
    ]
  };
}
