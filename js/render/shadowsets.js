// --- Shadow material sets, kept compiled (core/shadowsets.js) ---
//
// A set is everything one build makes: the terrain's material (and the texel
// cache's low-res and miss materials), the sprites' lit materials, and the sun
// uniforms they read. Kept after shadows go off, and after a rebuild replaces
// it, so turning shadows back on - or flipping a setting back - is a swap, not
// a compile. The newest SHADOW_SETS_KEPT are kept. bxb.shadowcache() turns
// keeping off: each set is then disposed as soon as it is replaced, as before.
//
// ctx:
//   renderer, scene, camera
//   options()    the shading build's options (gpu/shade.js createShadowColorNode):
//                the features' slices and main.js's own
//   texel, spriteAtlas, sun   those features: the terrain's passes, the
//                sprites' atlas, the sun uniforms a set carries
//   sprites      { meshes() drawn now, clipping() the characters (an on-top
//                variant each), normalFor(mesh) }
import { createShadowColorNode } from '../gpu/shade.js';
import { createSpriteShadowMaterial, createSpriteAtlasMaterial } from '../gpu/sprite.js';
import { keyPart, createSetCache, diffKeyParts } from '../core/shadowsets.js';
import { worldInstancedMesh, glassInstancedMesh, terrainTextures } from './render.js';
import { syncCutout, wantsTransparent } from './cutout.js';
import { flip } from './engine.js';

const SHADOW_SETS_KEPT = 2;

// One material per frame: building the node graph into WGSL is synchronous
// JavaScript, and doing all four in one frame is a hitch of its own.
export const nextFrame = () => new Promise(r => requestAnimationFrame(() => r()));

export function createShadowSets({ renderer, scene, camera, options, texel, spriteAtlas, sun, sprites }) {
  const shadowSets = createSetCache(SHADOW_SETS_KEPT);
  let shadowCacheOn = true;
  let activeShadowSet = null;
  // Builds started; one finishing after a newer started is dropped.
  let shadowBuild = 0;
  // Resolves once the latest build is compiled and on screen (or superseded).
  let shadowBuildDone = Promise.resolve();

  // Sprites receive the same lights as the terrain. One lit material per ORIGINAL
  // material, so the trees' planes and all eight trees still share one. The lit
  // materials belong to their shadow set, which disposes them.
  function restoreSpriteMaterials() {
    for (const m of sprites.meshes()) {
      if (!m.userData.originalMaterial) continue;
      m.material = m.userData.originalMaterial;
      if (m.userData.depthVariants) {
        m.userData.depthVariants = null;
        // The plain material missed any clip transitions while it was parked.
        const top = !!m.userData.drawingOnTop;
        if (m.material.depthTest === top) {
          m.material.depthTest = m.material.depthWrite = !top;
          m.material.needsUpdate = true;
        }
      }
      // Its transparent flag too - the clip state, or bxb.cutouts(), may have
      // moved on while it was parked.
      syncCutout(m.material);
    }
  }
  // The characters also get an always-on-top variant for the clip fix in
  // sprites.js - compiled here with the rest, so leaning into a wall never
  // compiles anything. A sprite's normal map is null until its card has
  // resolved; the sprite is shaded flat until the rebuild that follows.
  // atlas: shade into the sprite texel atlas and read it back (spriteAtlas.ensure
  // has made the twins); false shades every pixel.
  function buildSpriteShadows(spriteLight, atlas = false) {
    const made = new Map();
    const pairs = [];
    const variants = new Map();
    const atlasMats = atlas ? new Map() : null;
    const reads = atlas ? spriteAtlas.bindings : null;
    for (const m of sprites.meshes()) {
      const base = m.userData.originalMaterial || m.material;
      m.userData.originalMaterial = base;
      if (atlas && !atlasMats.has(base)) {
        atlasMats.set(base, createSpriteAtlasMaterial(base, spriteLight, sprites.normalFor(m), spriteAtlas.bindings));
      }
      if (atlas) pairs.push([m.userData.spriteAtlasTwin, atlasMats.get(base), spriteAtlas.bindings.rt]);
      if (!made.has(base)) {
        const mat = createSpriteShadowMaterial(base, spriteLight, sprites.normalFor(m), reads);
        mat.depthTest = mat.depthWrite = true;
        // Opaque when its texture's alpha is binary (js/cutout.js). Set from the
        // texture, not copied from base: a character parked mid-clip has a
        // transparent base, and this is the variant for NOT clipping.
        mat.transparent = wantsTransparent(base);
        made.set(base, mat);
      }
      pairs.push([m, made.get(base)]);
    }
    const mats = [...made.values()];
    for (const m of sprites.clipping()) {
      const top = createSpriteShadowMaterial(m.userData.originalMaterial, spriteLight,
                                             sprites.normalFor(m), reads);
      top.depthTest = top.depthWrite = false;
      // Always transparent: drawn last, over everything, is the transparent pass.
      top.transparent = true;
      variants.set(m, { normal: made.get(m.userData.originalMaterial), onTop: top });
      pairs.push([m, top]);
      mats.push(top);
    }
    if (atlasMats) mats.push(...atlasMats.values());
    return { pairs, mats, variants, made, atlasMats };
  }

  // Compile the new materials before anything visible uses them, so the swap
  // costs nothing and the previous materials keep drawing meanwhile.
  //
  // Against the REAL mesh in the REAL scene: compileAsync keys its cache on the
  // scene it is handed, so compiling stand-ins in a scratch scene produced
  // pipelines the actual render never looked up. The new material is put on the
  // mesh only for the synchronous part of the call - which is where the render
  // object is captured - and taken off again before any frame can draw it. Culling
  // is lifted for the same span, or an off-screen sprite would be skipped and
  // compile on first sight instead.

  // Compile everything in the scene that has not been drawn yet, with culling
  // lifted - otherwise a sprite first seen on walking into new ground compiles
  // its pipeline right then, which is the hitch. Anything already compiled is a
  // cache hit, so this is cheap to repeat.
  async function warmScene() {
    const lifted = [];
    scene.traverse(o => { if (o.frustumCulled) { lifted.push(o); o.frustumCulled = false; } });
    let pending;
    try { pending = renderer.compileAsync(scene, camera); }
    finally { for (const o of lifted) o.frustumCulled = true; }
    await pending;
  }
  async function compileOffscreen(pairs) {
    const seen = new Set();
    for (const [mesh, mat, target = null] of pairs) {
      if (seen.has(mat)) continue;   // meshes sharing a material share its pipeline
      seen.add(mat);
      const old = mesh.material, culled = mesh.frustumCulled;
      const oldTarget = renderer.getRenderTarget();
      const mask = camera.layers.mask;
      mesh.material = mat;
      mesh.frustumCulled = false;
      // compileAsync skips what the camera's layers leave out - a sprite atlas
      // twin is on its own layer only.
      camera.layers.mask |= mesh.layers.mask;
      let pending;
      // A pipeline is per target format: compile against the one it draws into.
      if (target) renderer.setRenderTarget(target);
      try { pending = renderer.compileAsync(mesh, camera, scene); }
      finally {
        mesh.material = old; mesh.frustumCulled = culled;
        camera.layers.mask = mask;
        if (target) renderer.setRenderTarget(oldTarget);
      }
      await pending;
      await nextFrame();
    }
  }

  // Everything a set's shaders depend on: the node options, the terrain mesh and
  // the texel cache, and per sprite its plain material, its normal map and
  // whether it draws opaque.
  function shadowSetKeyParts(opts) {
    const parts = {};
    for (const [k, v] of Object.entries(opts)) parts[k] = keyPart(v);
    // The terrain bag also carries load progress (ready, the per-layer pixels the
    // GI averages), which no shader reads - keying on it would miss the cache
    // whenever textures finished loading between two builds. Only the textures
    // the shaders bind.
    const t = terrainTextures;
    parts.terrain = keyPart([t.albedo, t.normal, t.specular, t.blocks]);
    parts.mesh = keyPart(worldInstancedMesh);
    parts.glass = keyPart(glassInstancedMesh);
    parts.texelCache = keyPart(texel.keyParts());
    parts.spriteAtlas = keyPart(spriteAtlas.ready());
    parts.sprites = keyPart(sprites.meshes().map(m => {
      const base = m.userData.originalMaterial || m.material;
      return [base, sprites.normalFor(m), wantsTransparent(base)];
    }));
    return parts;
  }

  function disposeShadowSet(set) {
    for (const m of set.mats) m.dispose();
  }

  // The set for the current settings: kept, or built and compiling. set.ready
  // resolves once it is compiled, either way.
  function obtainShadowSet() {
    texel.prepare();
    const opts = options();
    const parts = shadowSetKeyParts(opts);
    const key = keyPart(parts);
    const kept = shadowCacheOn ? shadowSets.get(key) : null;
    if (kept) return kept;
    // A compile is the hitch this cache exists to avoid, so say what forced it:
    // which inputs differ from the newest kept set.
    const newest = shadowSets.values().pop();
    if (newest) {
      console.log(`[shadows] compiling a new shader set - differs from the kept one in: ` +
                  diffKeyParts(parts, newest.parts).join(', '));
    }

    if (opts.split) texel.prepareSplit(opts);
    const built = createShadowColorNode(opts);
    const { spriteLight, sun: sunUniforms } = built;
    // The terrain's and glass's materials: the texel cache's passes, or the shading itself.
    const { terrainMat, lowMat, missMat, glassMat, glassLow, glassMiss } = texel.materials(built, opts);
    const atlas = spriteAtlas.ready();
    if (atlas) spriteAtlas.ensure();
    const spriteSet = buildSpriteShadows(spriteLight, atlas);
    const pairs = [[worldInstancedMesh, terrainMat], ...spriteSet.pairs,
                   ...texel.compilePairs({ lowMat, missMat, glassMat, glassLow, glassMiss })];
    const set = {
      key, parts, sun: sunUniforms, terrainMat, lowMat, missMat, glassMat, glassLow, glassMiss,
      made: spriteSet.made, variants: spriteSet.variants, atlasMats: spriteSet.atlasMats,
      mirrorCache: built.mirrorCache,
      splitKernels: lowMat && built.splitKernels ? built.splitKernels : null,
      giKernels: lowMat && built.giKernels ? built.giKernels : null,
      resolved: !!(lowMat && built.splitKernels && !missMat),
      mats: [terrainMat, ...[lowMat, missMat].filter(Boolean),
             ...[glassMat, glassLow, glassMiss].filter(Boolean), ...spriteSet.mats],
      compiled: false
    };
    set.ready = compileOffscreen(pairs).catch(err => {
      console.warn('[shadows] background compile failed, swapping anyway', err);
    }).then(() => {
      set.compiled = true;
      for (const old of shadowSets.evict(activeShadowSet)) disposeShadowSet(old);
    });
    if (shadowCacheOn) shadowSets.set(key, set);
    return set;
  }

  // Puts a compiled set on the meshes.
  function applyShadowSet(set) {
    const prev = activeShadowSet;
    texel.apply(set);
    restoreSpriteMaterials();
    if (set.atlasMats) spriteAtlas.ensure(set.atlasMats);
    for (const m of sprites.meshes()) {
      const v = set.variants.get(m);
      if (v) {
        m.userData.depthVariants = v;
        m.material = m.userData.drawingOnTop ? v.onTop : v.normal;
      } else {
        const base = m.userData.originalMaterial || m.material;
        const lit = set.made.get(base);
        if (lit) { m.userData.originalMaterial = base; m.material = lit; }
      }
    }
    // A kept set's sun uniforms were last written when it was last shown.
    sun.apply(set);
    activeShadowSet = set;
    // Replaced and not kept: disposed now, as before the cache.
    if (prev && prev !== set && !shadowSets.has(prev.key)) disposeShadowSet(prev);
    warmScene();
    // And the plain materials, so turning shadows off swaps without compiling.
    // Cheap while they are still compiled; if not, compiled now, while this
    // turn-on is already the moment that paid.
    warmPlainMaterials();
  }

  async function warmPlainMaterials() {
    const pairs = [];
    const base = worldInstancedMesh.userData.originalMaterial;
    if (base) pairs.push([worldInstancedMesh, base]);
    const glassBase = glassInstancedMesh && glassInstancedMesh.userData.originalMaterial;
    if (glassBase) pairs.push([glassInstancedMesh, glassBase]);
    for (const m of sprites.meshes()) {
      if (m.userData.originalMaterial) pairs.push([m, m.userData.originalMaterial]);
    }
    await compileOffscreen(pairs);
  }

  // Back to the plain materials. The set is kept (or disposed, cache off).
  function removeShadowSet() {
    const set = activeShadowSet;
    texel.remove();
    restoreSpriteMaterials();
    activeShadowSet = null;
    if (set && !shadowSets.has(set.key)) disposeShadowSet(set);
  }

  // bxb.shadowcache(): off disposes every kept set but the one showing, and
  // from then on each set is disposed as soon as it is replaced.
  function setShadowCache(on) {
    shadowCacheOn = !!on;
    if (shadowCacheOn) return;
    for (const s of shadowSets.drain(activeShadowSet)) {
      if (s.compiled) disposeShadowSet(s);
    }
    if (activeShadowSet) shadowSets.delete(activeShadowSet.key);
  }

  const commands = {
    shadowcache: {
      help: 'keep compiled shadow material sets, so toggling shadows (or a setting) back is a swap. Off: dispose on replace, as before',
      run: () => {
        setShadowCache(!shadowCacheOn);
        return `shadow sets ${shadowCacheOn ? `kept (newest ${SHADOW_SETS_KEPT})` : 'disposed when replaced'}`;
      }
    }
  };

  return {
    // The set showing (null with shadows off).
    get active() { return activeShadowSet; },
    // Resolves once the latest build is compiled and on screen (or superseded).
    get done() { return shadowBuildDone; },
    // The set for the settings as they stand, put on screen once compiled -
    // unless a newer build or remove() came first.
    build() {
      const set = obtainShadowSet();
      const id = ++shadowBuild;
      shadowBuildDone = set.ready.then(() => {
        // A newer build, or shadows turned off, while this one compiled. A kept
        // set stays in the cache; one nothing holds is disposed.
        if (id !== shadowBuild) {
          if (!shadowSets.has(set.key) && set !== activeShadowSet) disposeShadowSet(set);
          return;
        }
        applyShadowSet(set);
      });
      return set;
    },
    // Back to the plain materials, dropping any build still compiling.
    remove() {
      shadowBuild++;
      removeShadowSet();
    },
    // Whether a build now would compile: no kept, compiled set for the settings
    // as they stand.
    wouldCompile() {
      if (!shadowCacheOn) return true;
      sun.ensureUniforms();
      const kept = shadowSets.peek(keyPart(shadowSetKeyParts(options())));
      return !kept || !kept.compiled;
    },
    commands,
    setting: { key: 'shadowcache', label: 'Keep shadow materials', type: 'toggle',
      help: 'compiled shadow sets are kept, so toggling back is a swap',
      get: () => shadowCacheOn, set: v => flip(shadowCacheOn, v, () => commands.shadowcache.run()) }
  };
}
