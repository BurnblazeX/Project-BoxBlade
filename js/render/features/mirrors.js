// --- Mirrors (core/mirrors.js): reflected sunlight ---
//
// The reflective surfaces as rectangles, built once the specular textures are
// in (which materials reflect is read off their _s), packed nearest the player
// first. The shader gathers the sun off them onto whatever faces the mirrored
// sun. A kernel flag, so toggling rebuilds; the rectangles are uniforms.
//
// ctx:
//   sunLight     the DirectionalLight: its position is the sun's direction
//   sunOn()      whether the sun is in the shading at all
//   lighting     the lights feature: its light list, and which rows throw
//                mirror light (its mirror-light budget)
//   focus        where the player is (a Vector3): nearest mirrors first
//   footprint()  the cascade whose footprint bounds the mirrors packed (C1)
//   rebuild()    the shading rebuilt, when a kernel flag changed
import { buildMirrors, packMirrors, isReflective, MAX_MIRRORS, mirrorCacheStride,
         MIRROR_CACHE_VEC4S } from '../../core/mirrors.js';
import { MATERIALS, isGlassMaterial } from '../../core/materials.js';
import { World } from '../../core/world.js';
import { cascadeExtentMetres } from '../../core/boxgrid.js';
import { LIGHT_VEC4S } from '../../core/lights.js';
import { createMirrorBindings, writeMirrorBindings } from '../../gpu/mirrorlight.js';
import { glassBackGlare } from '../../gpu/glass.js';
import { sunColourUniform } from '../../gpu/sky.js';
import { terrainTextures } from '../render.js';
import { flip } from '../engine.js';

export function createMirrors({ sunLight, sunOn, lighting, focus, footprint, rebuild }) {
  let mirrorsOn = true;
  let mirrorOnly = false;   // the view: mirror light alone
  let mirrorRects = null;
  // The mirror texel cache (mirrors.js): each mirror texel's own terms, worked
  // out once a frame by a compute pass instead of by every receiver. Off
  // (bxb.mirrorcache()) goes back to per receiver, for A/B.
  let mirrorCacheOn = true;
  let mirrorCacheTexels = 0;
  // Glare off glass: 0 off, 1 its front surface (glass joins the mirror light's
  // rectangles), 2 its back face too (gpu/glass.js glassBackGlare). Only while the
  // mirror light itself is on. Front-surface glare by default.
  let glassGlare = 1;
  const bindings = createMirrorBindings(MAX_MIRRORS);
  const pack = new Float32Array(MAX_MIRRORS * 16);

  function update() {
    if (!mirrorRects) {
      if (!terrainTextures.ready) return;
      // Glass only with its glare on (bxb.glassglare): otherwise its faces would
      // take rectangles from the real mirrors' cap for a 4% reflection. Each
      // rectangle carries its own texels' F0, so glass glares as glass does.
      const reflective = new Set(MATERIALS.filter((m, l) => (!m.glass || glassGlare > 0) &&
                                                     isReflective(terrainTextures.specRgba[l]))
                                          .map(m => m.id));
      // Glass faces split by the slab behind them only for the back-face glare.
      mirrorRects = buildMirrors(World, reflective, isGlassMaterial, { backFaces: glassGlare >= 2 });
    }
    // Culled to what each mirror can reflect THIS frame: the sun if it faces
    // it, and the lights that reach it - read back from the light list the
    // shader sees. See mirrors.js mirrorReach.
    const d = sunLight.position.clone().normalize();
    const sunLive = sunOn() && (sunColourUniform.value.r + sunColourUniform.value.g
                                + sunColourUniform.value.b) > 0;
    const lights = [];
    const lb = lighting.bindings();
    for (let i = 0; i < lb.count.value; i++) {
      const A = lb.data.array[i * LIGHT_VEC4S];
      // Past the mirror-light budget a light reaches no mirror (radius 0).
      lights.push({ x: A.x, y: A.y, z: A.z, radius: lighting.mirrorRows[i] === false ? 0 : A.w });
    }
    // Only mirrors inside C1's footprint. The cascades follow the player, so a
    // player (or free cam's subject) far from every mirror packs none, and the
    // shader's mirror loop runs zero times.
    const g1 = footprint();
    let area = null;
    if (g1) {
      const side = cascadeExtentMetres(g1.level ?? 1);
      area = { minX: g1.origin.x, maxX: g1.origin.x + side,
               minZ: g1.origin.z, maxZ: g1.origin.z + side };
    }
    const cache = { stride: mirrorCacheStride(lights.length), capacity: MIRROR_CACHE_VEC4S, texels: 0 };
    const n = area ? packMirrors(mirrorRects, focus, pack, MAX_MIRRORS,
                                 { dir: [d.x, d.y, d.z], on: sunLive }, lights, area, cache) : 0;
    writeMirrorBindings(bindings, pack, n);
    mirrorCacheTexels = n ? cache.texels : 0;
  }

  const commands = {
    glassglare: {
      help: 'glare off glass: 0 off, 1 its front surface, 2 its back face too. Needs the mirror light on',
      usage: 'bxb.glassglare(1)  |  bxb.glassglare(2)  |  bxb.glassglare(0)',
      run: (level = null) => {
        const next = level === null ? (glassGlare + 1) % 3 : Math.max(0, Math.min(2, Math.round(level)));
        if ((next > 0) !== (glassGlare > 0)) mirrorRects = null;   // glass joins or leaves
        if ((next >= 2) !== (glassGlare >= 2)) mirrorRects = null;  // faces split by slab, or merge
        glassGlare = next;
        glassBackGlare.value = glassGlare >= 2 ? 1 : 0;
        const what = ['off', 'front surface', 'front and back face'][glassGlare];
        return `glass glare: ${what}${mirrorsOn || !glassGlare ? '' : ' - but the mirror light is off (bxb.mirrors())'}`;
      }
    },
    mirrorcache: {
      help: "the mirror texel cache: each mirror texel's own sun shadow, light visibility and surface " +
            'worked out once a frame, not per receiver. Off computes them per receiver, as before',
      run: () => {
        mirrorCacheOn = !mirrorCacheOn;
        rebuild();
        return `mirror texel cache ${mirrorCacheOn ? 'on' : 'off'}`;
      }
    },
    mirrors: {
      help: 'reflected sunlight: polished surfaces throw the sun onto what faces them. ' +
            '"only" views it alone',
      usage: "bxb.mirrors()  toggles  |  bxb.mirrors('only')",
      run: (a = null) => {
        if (a === 'only') {
          mirrorOnly = !mirrorOnly;
          if (mirrorOnly) mirrorsOn = true;
        } else mirrorsOn = !mirrorsOn;
        rebuild();
        return `mirror light ${mirrorsOn ? 'on' : 'off'}` +
               (mirrorRects ? ` - ${mirrorRects.length} mirror rectangles` : ' - textures still loading');
      }
    }
  };

  return {
    name: 'mirrors',
    options: () => ({ mirrors: mirrorsOn ? bindings : null, mirrorOnly, mirrorCache: mirrorCacheOn }),
    bindings,
    update,
    get on() { return mirrorsOn; },
    get glare() { return glassGlare; },
    // The rectangles this frame (null until the textures are in).
    get rects() { return mirrorRects; },
    // Mirror texels the cache pass works out this frame (0: none).
    get cacheTexels() { return mirrorCacheTexels; },
    // The world changed: the rectangles are built again on the next update.
    invalidate() { mirrorRects = null; },
    commands,
    // For the Specular section.
    settings: [
      { key: 'mirrors', label: 'Mirror light', type: 'toggle',
        help: 'polished surfaces throw reflected sunlight onto what faces them',
        get: () => mirrorsOn, set: v => flip(mirrorsOn, v, () => commands.mirrors.run()) },
      { key: 'mirroronly', label: 'View: mirror light only', type: 'toggle',
        help: 'the reflected sunlight alone, raw - not part of the GI bounce view',
        get: () => mirrorOnly, set: v => flip(mirrorOnly, v, () => commands.mirrors.run('only')) }
    ],
    // For the Glass section.
    glareSetting: { key: 'glassglare', label: 'Glare', type: 'range', min: 0, max: 2, step: 1,
      help: '0 off, 1 front surface, 2 plus the back face (fainter, tinted, offset). Needs the mirror light',
      get: () => glassGlare, set: v => commands.glassglare.run(v) }
  };
}
