// --- GI: the bounce - the LPV (core/lpv.js, gpu/gi.js), or texel GI ---
//
// On/off is compiled into the shading kernel, like AO (its view is the render
// debugger's: render/features/debugview.js).
// Strength and spread are uniforms; iterations and injection slices are CPU
// numbers read each frame. The volume is built once, on the first shadows-on
// frame that has cascades and a light list, and follows its cascade from then on.
//
// ctx:
//   renderer
//   cascades()     the shadow cascades' bindings (the volumes follow C1, C2)
//   sun            { uniforms() the sun's march uniforms, made if need be:
//                    { biasUniform, fadeStartUniform, edgeFadeUniform,
//                    ambientUniform }; on(); angular(); light (its position
//                    is the sun's direction) }
//   lighting, sky, mirrors   those features
//   sprites        { casters() this frame's card casters; types the card
//                    types they index }: the sprites the LPV bounces
//   terrainAlbedo() the terrain's mean albedo, once its textures are in
//   shadowsOn()    whether the marched shading is on
//   rebuild()      the shading rebuilt, when a kernel flag changed
//   fogOn()        whether there is fog: the volumes then light their air for
//                  it too (gpu/gi.js fog kernel; render/features/fog.js reads it)
import { uniform, int } from 'three/tsl';
import { diffuseAlbedo, specularAlbedo, MAX_GI_SPRITES, SPRITE_GI_FILL, averageAlbedo, lpvShift,
         DEFAULT_LPV_SPREAD, DEFAULT_LPV_ITERATIONS, GI_MAX_BOUNCES, DEFAULT_GI_BOUNCES,
         DEFAULT_LPV_SLICES, DEFAULT_GI_STRENGTH, LPV_VOLUMES, lpvVolumeRes, LPV_RES_OPTIONS,
         DEFAULT_LPV_SKY_SPREAD } from '../../core/lpv.js';
import { MATERIALS } from '../../core/materials.js';
import { MODELS } from '../../core/models.js';
import { createLPV, lpvStorage } from '../../gpu/gi.js';
import { createTexelGI, GI_LIST_CAP } from '../../gpu/texelgi.js';
import { writeSunUniforms } from '../../gpu/sun.js';
import { reflectEscapeUniform } from '../../gpu/specular.js';
import { terrainTextures, glassInstancedMesh } from '../render.js';
import { flip } from '../engine.js';

export function createGI({ renderer, cascades, sun, lighting, sky, mirrors, sprites, terrainAlbedo,
                           shadowsOn, rebuild, fogOn = () => false }) {
  let giOn = true;
  // Which bounce: 'texel' - a radiance cache on the block faces, ray-traced
  // (gpu/texelgi.js createTexelGI; needs split shading, which finds the patches) - or
  // 'lpv', the light propagation volume.
  let giMode = 'lpv';
  let texelGI = null;
  // Texel GI response time (s): the patches' shown light eases toward each new
  // sum with this time constant, in real time (gpu/texelgi.js createTexelGI blend).
  let giResponse = 0.15;
  let giLastT = 0;
  function ensureTexelGI() {
    if (!texelGI) {
      texelGI = createTexelGI();
      texelGI.strength = giStrength;
    }
    return texelGI;
  }
  // One volume per lpv.js LPV_VOLUMES, finest first: C1 at 0.75 m cells, C2 at 1.5 m. They
  // share every setting; the shading pass blends them (lpvSampleTSL).
  let lpvs = null;
  let lpvOrigins = [];        // cascade origin each volume was last laid over
  let lpvIterations = DEFAULT_LPV_ITERATIONS;
  let lpvSpread = DEFAULT_LPV_SPREAD;
  let lpvBounces = DEFAULT_GI_BOUNCES;
  // GI resolution (lpv.js LPV_RES_OPTIONS): 1 is 2 x 2 cells a block face.
  // Finer cells keep the reach in metres - the spread per cell is the 1/res
  // power, the iterations res times - so only the detail changes, and the cost.
  let lpvRes = 1;
  let lpvSlices = DEFAULT_LPV_SLICES;
  // The LPV's sky visibility (lpv.js propagateSkyStep): 1 the sky ambient is
  // multiplied by it, so roofs shade it as texel GI's rays do - and it stands
  // in for the injection's sky cone; 0 the sky everywhere, as before. A
  // uniform, read by the volumes and the shading alike.
  const lpvSky = uniform(int(1));
  let lpvSkySpread = DEFAULT_LPV_SKY_SPREAD;
  const giStrength = uniform(DEFAULT_GI_STRENGTH);
  let lastGIms = 0;

  // --- Performance ---
  //
  // giFarHalf: the far GI volume (C2, 1.5 m cells) updates every
  // other frame. Its contents are coarse and slow-changing; the lag is two
  // frames instead of one, which nothing that far away shows.
  let giFarHalf = true;
  // giBatch: each GI volume's kernels go to the GPU as one compute pass and one
  // submit, not eleven. Same kernels, same order, same result; off submits them
  // one by one as before, for A/B.
  let giBatch = true;
  let giFrame = 0;

  // Mirror bounce: light the mirrors throw bounces from where it lands (the GI's
  // injection runs the mirror light at its probe hits). Only with the mirror
  // light on; without it, every surface bounces its whole specular share itself.
  let mirrorBounce = true;

  function ensureLPV() {
    const c = cascades();
    if (lpvs || LPV_VOLUMES.some(v => v.level >= c.length)) return lpvs;
    const u = sun.uniforms();
    lpvs = [];
    for (const spec of LPV_VOLUMES) {
      const res = lpvVolumeRes(spec) * lpvRes;
      // Every bounce order fits in a 24^3 volume (~3.5 MB each); a bigger one
      // holds only the ones asked for - 16 orders of 48^3 are 113 MB.
      lpvs.push(createLPV({ cascades: c, lights: lighting.bindings(), level: spec.level, res,
                            orders: res === 1 ? GI_MAX_BOUNCES : lpvBounces,
                            biasUniform: u.biasUniform, fadeStartUniform: u.fadeStartUniform,
                            edgeFadeUniform: u.edgeFadeUniform,
                            sky: sky.bindings, ambientUniform: u.ambientUniform,
                            blocks: terrainTextures.blocks,
                            glass: glassInstancedMesh ? terrainTextures : false,
                            // The mirror light at each probe hit, so what the
                            // mirrors throw bounces from where it lands.
                            terrain: terrainTextures, mirrors: mirrors.bindings, skyAB: lpvSky }));
    }
    return lpvs;
  }

  // Throws the volumes away, to be made again at the current resolution and
  // bounce count; the shaders that sample them rebuild with them.
  function rebuildLPV() {
    if (!lpvs) return;
    for (const v of lpvs) v.dispose(renderer);
    lpvs = null;
    lpvOrigins = [];
    rebuild();
  }

  function giBinding() {
    return { levels: lpvs.map(v => v.binding), strength: giStrength, sky: lpvSky };
  }

  // Each material's diffuse albedo, once its textures are in (lpv.js
  // diffuseAlbedo) - what a probe landing on that block bounces. null until then.
  let layerAlbedos = null;
  let layerSpecAll = null, layerSpecRough = null, layerTints = null;
  function resolveLayerAlbedos() {
    if (layerAlbedos || !terrainTextures.ready) return;
    const spec = l => terrainTextures.specRgba[l];
    // The GI runs with shadows on only: what it bounces is the _a, if any.
    layerAlbedos = terrainTextures.rgba.map((rgba, l) =>
      (rgba ? diffuseAlbedo(terrainTextures.flatRgba[l] || rgba, spec(l)) : { r: 0.5, g: 0.5, b: 0.5 }));
    // The specular share each material bounces (lpv.js specularAlbedo): all of
    // it, and the part too rough for the mirror light, which carries the rest.
    const grey = { r: 0.04, g: 0.04, b: 0.04 };
    layerSpecAll = terrainTextures.rgba.map((rgba, l) => (rgba ? specularAlbedo(rgba, spec(l)) : grey));
    layerSpecRough = terrainTextures.rgba.map((rgba, l) =>
      (rgba ? specularAlbedo(rgba, spec(l), { mirrored: true }) : grey));
    // Glass's tint through a block, for the light the LPV carries through it.
    layerTints = MATERIALS.map((m, l) => (m.glass && terrainTextures.rgba[l]
      ? averageAlbedo(terrainTextures.rgba[l]) : { r: 1, g: 1, b: 1 }));
    // Then each model's entry (models.js giIndex): its texture's diffuse
    // albedo, a dielectric's specular share, no tint.
    for (const m of MODELS.values()) {
      layerAlbedos[m.giIndex] = m.albedoPixels ? diffuseAlbedo(m.albedoPixels) : { r: 0.5, g: 0.5, b: 0.5 };
      layerSpecAll[m.giIndex] = layerSpecRough[m.giIndex] = grey;
      layerTints[m.giIndex] = { r: 1, g: 1, b: 1 };
    }
  }

  // The sprites the LPV sees this frame (lpv.js, "Sprites in the LPV"): every
  // card caster, as a lit box - its quad's width square and its height tall -
  // with its texture's mean albedo and a fill of its opaque fraction. The card
  // casters are the sprites that cast shadows, so the same set bounces light.
  function packGISprites(u) {
    const list = sprites.casters() || [];
    const n = Math.min(list.length, MAX_GI_SPRITES);
    for (let i = 0; i < n; i++) {
      const c = list[i], t = sprites.types[c.typeIndex];
      if (!t.albedo) t.albedo = t.card.rgba ? averageAlbedo(t.card.rgba) : { r: 0.5, g: 0.5, b: 0.5 };
      u.sprites.array[i * 2].set(c.centre.x, c.centre.y, c.centre.z, t.widthMetres / 2);
      u.sprites.array[i * 2 + 1].set(t.albedo.r, t.albedo.g, t.albedo.b, t.heightMetres / 2);
      u.spriteFill.array[i] = (t.coverage ?? 0.5) * SPRITE_GI_FILL;
    }
    u.spriteCount.value = n;
  }

  // The same sprites for texel GI's rays (gpu/texelgi.js createTexelGI
  // sprites): three vec4 each, the third its opaque fraction.
  function packTexelSprites(t) {
    const list = sprites.casters() || [];
    const n = Math.min(list.length, MAX_GI_SPRITES);
    for (let i = 0; i < n; i++) {
      const c = list[i], ty = sprites.types[c.typeIndex];
      if (!ty.albedo) ty.albedo = ty.card.rgba ? averageAlbedo(ty.card.rgba) : { r: 0.5, g: 0.5, b: 0.5 };
      t.sprites.array[i * 3].set(c.centre.x, c.centre.y, c.centre.z, ty.widthMetres / 2);
      t.sprites.array[i * 3 + 1].set(ty.albedo.r, ty.albedo.g, ty.albedo.b, ty.heightMetres / 2);
      t.sprites.array[i * 3 + 2].set(ty.coverage ?? 0.5, 0, 0, 0);
    }
    t.spriteCount.value = n;
  }

  // One frame of GI, per volume: follow its cascade, then solid / inject /
  // propagate / resolve.
  function update() {
    if (!giOn || !lpvs || giMode !== 'lpv') return;
    const t0 = performance.now();
    resolveLayerAlbedos();
    giFrame++;
    const c = cascades(), albedo = terrainAlbedo();
    for (const v of lpvs) {
      // lpvOrigins is only written when a volume updates, so a skipped frame
      // cannot lose a re-origin: the next update sees the move.
      if (giFarHalf && v !== lpvs[0] && (giFrame & 1)) continue;
      const o = c[v.level].origin.value;
      const at = { x: o.x, y: o.y, z: o.z };
      const was = lpvOrigins[v.level];
      let shift = null;
      if (!was) shift = 'reset';
      else if (at.x !== was.x || at.y !== was.y || at.z !== was.z) {
        shift = lpvShift(was, at, v.cellMetres, v.dim) || 'reset';
      }
      lpvOrigins[v.level] = at;
      const u = v.uniforms;
      u.origin.value.set(at.x, at.y, at.z);
      if (albedo) u.albedo.value.set(albedo.r, albedo.g, albedo.b);
      if (layerAlbedos) {
        layerAlbedos.forEach((a, l) => u.layerAlbedo.array[l].set(a.r, a.g, a.b, 0));
        layerSpecAll.forEach((a, l) => u.layerSpecAll.array[l].set(a.r, a.g, a.b, 0));
        layerSpecRough.forEach((a, l) => u.layerSpecRough.array[l].set(a.r, a.g, a.b, 0));
        layerTints.forEach((a, l) => u.layerTint.array[l].set(a.r, a.g, a.b, 0));
      }
      u.mirrorGain.value = mirrors.on && mirrorBounce ? 1 : 0;
      u.skyGain.value = sky.gain;
      packGISprites(u);
      u.sunGain.value = sun.on() ? 1 : 0;
      u.spread.value = Math.pow(lpvSpread, 1 / v.res);
      u.skySpread.value = Math.pow(lpvSkySpread, 1 / v.res);
      u.bounces.value = Math.min(lpvBounces, v.orders);
      v.slices = lpvSlices;
      writeSunUniforms(v.sun, sun.light.position, sun.angular());
      v.update(renderer, { shift, iterations: Math.round(lpvIterations * v.res), batch: giBatch, fog: fogOn() });
    }
    lastGIms = performance.now() - t0;
  }

  // Texel GI, inside split shading's submit: each listed patch traces this
  // frame's slice of its directions. kernels: the shading set's GI kernels.
  function stepTexel(kernels, list) {
    const now = performance.now();
    const dt = giLastT ? Math.min((now - giLastT) / 1000, 0.25) : 1 / 60;
    giLastT = now;
    texelGI.blend.value = giResponse > 0 ? 1 - Math.exp(-dt / giResponse) : 1;
    packTexelSprites(texelGI);
    list.push(kernels.patches, kernels.sums, kernels.filter, kernels.direct);
    texelGI.frame.value = (texelGI.frame.value + 1) >>> 0;
  }

  const commands = {
    mirrorbounce: {
      help: 'light thrown by the mirrors bounces (GI) from where it lands - the mirror light run at each GI probe hit',
      run: () => {
        mirrorBounce = !mirrorBounce;
        return `mirror bounce ${mirrorBounce ? 'on' : 'off'}${mirrors.on ? '' : ' - but the mirror light is off'}`;
      }
    },
    gibatch: {
      help: "submit each GI volume's kernels as one compute pass (off: one submit per kernel)",
      run: () => {
        giBatch = !giBatch;
        return `GI kernels ${giBatch ? 'batched: one submit per volume' : 'one submit per kernel'}`;
      }
    },
    lpvhalf: {
      help: 'LPV propagation in half floats (packed): half the bytes for the largest GI kernel. ' +
            'Off: 32-bit floats, as before (rebuilds the volumes)',
      run: () => {
        lpvStorage.half = !lpvStorage.half;
        rebuildLPV();
        return `LPV propagation in ${lpvStorage.half ? 'half floats' : '32-bit floats'}`;
      }
    },
    gifar: {
      help: 'update the far (C2) GI volume every other frame',
      run: () => {
        giFarHalf = !giFarHalf;
        return `far GI volume ${giFarHalf ? 'every other frame' : 'every frame'}`;
      }
    },
    gi: {
      help: "global illumination: the LPV bounce. On/off and strength (bxb.view('gi') shows it alone)",
      usage: 'bxb.gi()  toggles  |  bxb.gi(true, 1.5) strength',
      run: (a = null, strength = null) => {
        const was = giOn;
        if (a !== null || strength === null) giOn = a === null ? !giOn : !!a;
        if (strength !== null) giStrength.value = Math.max(0, Number(strength));
        if (!giOn) lpvOrigins = [];   // stale while off; start over when it returns
        if (was !== giOn) rebuild();
        if (!shadowsOn()) return 'run bxb.shadows() first';
        if (!giOn) return 'GI off';
        return `GI on - ${lpvs ? lpvs.map(v => `${v.dim}^3 over C${v.level} (${v.cellMetres} m cells)`).join(' and ') : 'no volumes yet'}, ` +
               `strength ${giStrength.value.toFixed(2)}, spread ${lpvSpread.toFixed(2)}, ` +
               `${lpvIterations} iterations, 1/${lpvSlices} injected a frame, ` +
               `${lastGIms.toFixed(2)} ms CPU to encode`;
      }
    },
    gistats: {
      help: "texel GI: last frame's patches on screen against the list's room",
      usage: 'await bxb.gistats()',
      run: async () => {
        if (!texelGI || giMode !== 'texel') return 'texel GI is off';
        let q;
        try {
          q = new Uint32Array(await renderer.getArrayBufferAsync(texelGI.queueAttr));
        } catch { return 'no patches listed yet (split shading off, or the kernels have not run)'; }
        const n = q[0], claims = q[texelGI.claimsAt];
        // Claims: patches that took a slot (new, or evicted) - near n every frame
        // while standing still means history is being lost.
        return `${n} patches listed (room ${GI_LIST_CAP}${n > GI_LIST_CAP ? ' - OVERFLOWING: the rest go stale' : ''}), ` +
               `${claims} took a slot this frame`;
      }
    }
  };

  // The texel GI A/B switches: each a uniform on its tables.
  const ab = (key, label, name) => ({ key, label, type: 'toggle',
    get: () => ensureTexelGI().ab[name].value === 1, set: v => { ensureTexelGI().ab[name].value = v ? 1 : 0; } });

  return {
    name: 'gi',
    options: () => ({
      gi: giOn && giMode === 'lpv' && ensureLPV() ? giBinding() : null,
      texelGI: giOn && giMode === 'texel' ? ensureTexelGI() : null
    }),
    update,
    stepTexel,
    // The LPV's bindings for the fog (gpu/fog.js), while the LPV is the bounce.
    lpvBinding: () => (giOn && giMode === 'lpv' && lpvs ? giBinding() : null),
    commands,
    // For the Perf section.
    perfSettings: [
      { key: 'gibatch', label: 'Batch GI kernels', type: 'toggle',
        help: 'one compute pass and submit per GI volume instead of one per kernel',
        get: () => giBatch, set: v => flip(giBatch, v, () => commands.gibatch.run()) },
      { key: 'gifar', label: 'Far GI every 2nd frame', type: 'toggle',
        help: 'the C2 LPV volume updates at half rate',
        get: () => giFarHalf, set: v => flip(giFarHalf, v, () => commands.gifar.run()) }
    ],
    settings: { title: 'GI', settings: [
      { key: 'mirrorbounce', label: 'Mirror bounce', type: 'toggle',
        help: 'light the mirrors throw bounces from where it lands (needs the mirror light)',
        get: () => mirrorBounce, set: v => flip(mirrorBounce, v, () => commands.mirrorbounce.run()) },
      { key: 'gi', label: 'Bounce light (LPV)', type: 'toggle',
        get: () => giOn, set: v => flip(giOn, v, () => commands.gi.run()) },
      { key: 'gimode', label: 'Method', type: 'select',
        options: [{ value: 'texel', label: 'per-texel rays' }, { value: 'lpv', label: 'LPV (volume)' }],
        help: 'per-texel rays need split shading (Perf); the LPV settings below are its own (rebuilds)',
        get: () => giMode,
        set: v => { if (v !== giMode) { giMode = v; rebuild(); } } },
      ab('giabshadow', 'A/B: lamps shadowed at GI hits', 'hitShadows'),
      ab('giabphys', 'A/B: physical bounce (Fresnel, metal)', 'physical'),
      ab('giabfresh', 'A/B: shading reads fresh patches only', 'fresh'),
      ab('giabanchor', 'A/B: rays from anchor texel', 'anchor'),
      ab('giabsky', 'A/B: sky from GI rays (roofs block it)', 'sky'),
      ab('giabdirect', 'A/B: GI hits read the patch direct-light cache', 'direct'),
      ab('giabrotate', 'A/B: each patch turns its GI directions', 'rotate'),
      ab('giabfilter', 'A/B: patches blurred with their neighbours (3x3)', 'filter'),
      ab('giabnear', 'A/B: AO cones split the patch light near/far (with sky)', 'near'),
      { key: 'gidirectevery', label: 'Texel GI direct refresh (frames)', type: 'range', min: 1, max: 16, step: 1,
        help: 'how often a patch works out its sun and lamp light for the GI rays that land on it - ' +
              'higher is cheaper, and a moving torch shows in the bounce later',
        get: () => ensureTexelGI().directEvery.value,
        set: v => { ensureTexelGI().directEvery.value = Math.max(1, Math.round(v)); } },
      { key: 'giabreflsky', label: 'A/B: reflection misses: sky outdoors only', type: 'toggle',
        help: 'a reflection ray that met nothing shows the sky only if it ended above the roof of its column',
        get: () => reflectEscapeUniform.value === 1, set: v => { reflectEscapeUniform.value = v ? 1 : 0; } },
      { key: 'giresponse', label: 'Texel GI response (s)', type: 'range', min: 0, max: 1, step: 0.01,
        help: 'time the bounce takes to follow a change - smooths flicker from anything that moves. 0: none',
        get: () => giResponse, set: v => { giResponse = v; } },
      { key: 'gireach', label: 'Texel GI reach (m)', type: 'range', min: 1, max: 24, step: 0.5,
        help: 'how far a GI ray marches before it gives up (a miss brings the sky only if it ends outdoors)',
        get: () => ensureTexelGI().reach.value, set: v => { ensureTexelGI().reach.value = v; } },
      { key: 'gistrength', label: 'Strength', type: 'range', min: 0, max: 8, step: 0.05,
        get: () => giStrength.value, set: v => commands.gi.run(true, v) },
      { key: 'gispread', label: 'Spread', type: 'range', min: 0.5, max: 0.98, step: 0.01,
        help: 'how far light carries cell to cell - brightness is normalised, only reach changes',
        get: () => lpvSpread, set: v => { lpvSpread = v; } },
      { key: 'giabskylpv', label: 'A/B: LPV sky visibility (roofs block the sky)', type: 'toggle',
        help: 'the sky ambient times how much sky the LPV says each place sees - off: the sky everywhere',
        get: () => lpvSky.value === 1, set: v => { lpvSky.value = v ? 1 : 0; } },
      { key: 'giskyspread', label: 'LPV sky reach indoors', type: 'range', min: 0.5, max: 0.98, step: 0.01,
        help: 'how much of its neighbours\' sky a roofed cell keeps - higher lets daylight further in through doors and windows',
        get: () => lpvSkySpread, set: v => { lpvSkySpread = v; } },
      { key: 'gibounces', label: 'Bounces', type: 'range', min: 1, max: GI_MAX_BOUNCES, step: 1,
        help: 'orders of bounce light: 1 is light off a surface once, each more lands and leaves again',
        get: () => lpvBounces,
        set: v => {
          lpvBounces = Math.max(1, Math.min(GI_MAX_BOUNCES, Math.round(v)));
          if (lpvs && lpvs[0].orders < lpvBounces) rebuildLPV();   // finer res: more orders to hold
        } },
      { key: 'gires', label: 'Resolution', type: 'select',
        // The near volume's cells a block face, by its own cells a block (LPV_VOLUMES).
        options: LPV_RES_OPTIONS.map(r => {
          const n = LPV_VOLUMES[0].blockCells * r;
          return { value: r, label: `${n}x${n} a block face (${+(1.5 / n).toFixed(3)} m)` };
        }),
        help: 'GI cells a block face. Finer is sharper bounce and dearer: ~3.4x, 8x, 27x the cells (rebuilds)',
        get: () => lpvRes,
        set: v => { const r = Number(v); if (r !== lpvRes) { lpvRes = r; rebuildLPV(); } } },
      { key: 'giiter', label: 'Iterations / frame', type: 'range', min: 2, max: 32, step: 2,
        get: () => lpvIterations, set: v => { lpvIterations = v; } },
      { key: 'gislices', label: 'Inject 1/N a frame', type: 'range', min: 1, max: 64, step: 1,
        help: 'injection re-marches every light per surface; 1/N of the cells each frame',
        get: () => lpvSlices, set: v => { lpvSlices = v; } }
    ]},
    // For the Debug section: an overlay on the bounce, not a view of its own.
    missingSetting: { key: 'gimissing', label: 'Texel GI: show missing patches', type: 'toggle',
      help: 'the bounce reads bright red where a texel finds no patch in the table',
      get: () => ensureTexelGI().debug.value === 1, set: v => { ensureTexelGI().debug.value = v ? 1 : 0; } }
  };
}
