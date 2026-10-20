// --- Texel-rate shading (gpu/texel.js) ---
//
// The terrain is shaded in a low-res pass - 1/scale per axis - into a float
// target of vec4(colour, texel id); the full-res pass then reuses the colour
// of a nearby sample from the same texel, and shades only pixels that have
// none. Exact, since a texel's colour is the same everywhere in it.
// The low-res pass draws only layer TEXEL_LAYER, which only the terrain is on.
//
// Split shading (gpu/texel.js createSplitShading) rides on it: the low-res
// pass writes samples, compute kernels light them a feature at a time.
//
// ctx:
//   renderer, scene, camera
//   rig          the camera rig: its distance sizes the texel on screen
//   shadowsOn()  whether the marched shading is on
//   activeSet()  the shader set showing (its low-res kernels, resolved or not)
//   gi           the GI feature: stepTexel, inside the split shading's submit
//   rebuild()    the shading rebuilt, when a kernel flag changed
import * as THREE from 'three/webgpu';
import { uniform, Fn, If, mix, vec3 } from 'three/tsl';
import { texelPixels, texelCacheSpacingFor, splitCapacityFor } from '../../core/texelcache.js';
import { createTexelCacheWriteNode, createTexelCacheLookupNode, createTexelMissNode,
         createSplitShading, createTexelGBufferNode, createSplitResolve, createSplitReadNode,
         createShadowMaterial } from '../../gpu/texel.js';
import { terrainKeepTSL, terrainShapeTSL, terrainClipTSL, applyTerrainVertex, terrainRevealedTSL,
         terrainSurfaceTSL, terrainTexelTSL, terrainWhite } from '../../gpu/terrain.js';
import { terrainEmissiveTSL } from '../../gpu/shade.js';
import { applyOcclusionMask, outsideShadeTSL } from '../../gpu/occlusion.js';
import { worldInstancedMesh, glassInstancedMesh, terrainTextures } from '../render.js';
import { flip } from '../engine.js';

export const TEXEL_LAYER = 1;

// The terrain's and glass's materials for one shader set (gpu/shade.js
// createShadowColorNode's result `built`, built from `opts`): with the cache
// (`cached`), the low-res pass, the lookup and - unless the kernels resolve
// the misses - the stencil-gated miss pass; without, the shading itself.
// cache: the texel cache's target bindings. Also what the headless shader
// fixture builds (tools/lib/shaders.mjs).
export function buildTerrainMaterials({ built, opts, cache, cached }) {
  // A model's lightSource cube draws unlit, in every terrain pass.
  const node = terrainEmissiveTSL(built.node, terrainTextures);
  // The node applies the albedo itself - specular adds after it, not under it.
  // With the texel cache, the terrain's own material looks its colour up and
  // a second, low-res material does the shading - see render().
  let lowMat = null, missMat = null;
  let terrainMat;
  // Glass, when the world has any and the node could be built (it needs the
  // reflection trace): its own materials, drawing only glass blocks.
  const glassNode = glassInstancedMesh ? built.glassNode : null;
  let glassMat = null, glassLow = null, glassMiss = null;
  if (cached) {
    lowMat = new THREE.MeshBasicNodeMaterial();
    // outputNode, not colorNode: the stored colour must not carry the
    // per-instance tint, which the lookup material applies once itself.
    // The low-res pass alone also fills the hit lighting cache (gpu/shade.js makeNode).
    const split = built.splitKernels ? opts.split : null;
    lowMat.outputNode = split ? createTexelGBufferNode(false, opts.splitLegacy)
      : createTexelCacheWriteNode(terrainEmissiveTSL(built.nodeCaching, terrainTextures));
    // Resolved: misses shaded by the kernels too, read per pixel - only when
    // glass (if any) is split as well, as it shares the passes.
    // Legacy sends slopes and glowing cubes to the miss pass, so not resolved.
    const resolved = split && opts.splitResolve && !opts.splitLegacy && (built.splitGlass || !glassNode);
    if (resolved) {
      terrainMat = createShadowMaterial(worldInstancedMesh, createSplitReadNode(cache, split));
    } else {
    terrainMat = createShadowMaterial(worldInstancedMesh, createTexelCacheLookupNode(cache, split));
    terrainMat.stencilWrite = true;
    terrainMat.stencilFunc = THREE.AlwaysStencilFunc;
    terrainMat.stencilRef = 1;
    terrainMat.stencilZPass = THREE.ReplaceStencilOp;
    missMat = createShadowMaterial(worldInstancedMesh, createTexelMissNode(cache, node));
    missMat.stencilWrite = true;           // enables the test; nothing is written
    missMat.stencilFunc = THREE.EqualStencilFunc;
    missMat.stencilRef = 0;
    missMat.stencilZPass = THREE.KeepStencilOp;
    }
    if (glassNode && resolved) {
      glassLow = new THREE.MeshBasicNodeMaterial();
      glassLow.outputNode = createTexelGBufferNode(true, opts.splitLegacy);
      glassMat = createShadowMaterial(glassInstancedMesh, createSplitReadNode(cache, split));
    } else if (glassNode) {
      glassLow = new THREE.MeshBasicNodeMaterial();
      glassLow.outputNode = built.splitGlass ? createTexelGBufferNode(true)
        : createTexelCacheWriteNode(glassNode, !!split);
      glassMat = createShadowMaterial(glassInstancedMesh, createTexelCacheLookupNode(cache, split));
      glassMat.stencilWrite = true;
      glassMat.stencilFunc = THREE.AlwaysStencilFunc;
      glassMat.stencilRef = 1;
      glassMat.stencilZPass = THREE.ReplaceStencilOp;
      glassMiss = createShadowMaterial(glassInstancedMesh, createTexelMissNode(cache, glassNode));
      glassMiss.stencilWrite = true;
      glassMiss.stencilFunc = THREE.EqualStencilFunc;
      glassMiss.stencilRef = 0;
      glassMiss.stencilZPass = THREE.KeepStencilOp;
    }
  } else {
    terrainMat = createShadowMaterial(worldInstancedMesh, node);
    if (glassNode) glassMat = createShadowMaterial(glassInstancedMesh, glassNode);
  }
  // Each draws only its own blocks - terrainKeepTSL in render.js.
  // All with the same clip position (terrainClipTSL: a lifted roof shrinks
  // away), so every pass agrees on depth - glass too, which the prepass draws.
  for (const m of [terrainMat, lowMat, missMat]) if (m) applyTerrainVertex(m);
  // A lifted roof is not drawn (gpu/occlusion.js occlusionHiddenTSL), but
  // still casts into any shadow map.
  if (terrainMat) terrainMat.castShadowPositionNode = terrainKeepTSL(false, true, false);
  for (const m of [glassMat, glassLow, glassMiss]) if (m) {
    m.positionNode = terrainKeepTSL(true);
    m.vertexNode = terrainClipTSL();
  }
  // The interiors' mask, in every opaque pass alike (gpu/occlusion.js) -
  // never glass. The miss pass writes no depth - the prepass already holds
  // it - so its discard leaves the stencil test early, before the heavy
  // shader starts.
  for (const m of [terrainMat, lowMat, missMat]) if (m) applyOcclusionMask(m);
  if (missMat) missMat.depthWrite = false;
  // Not under a debug view: it shows a term alone, and should show it there too.
  if (!opts.view) for (const m of [terrainMat, missMat]) if (m) litWhereRevealed(m, built.revealedLight);
  return { terrainMat, lowMat, missMat, glassMat, glassLow, glassMiss };
}

// A revealed face (gpu/terrain.js terrainRevealedTSL) - a wall's top under a
// lifted roof - is buried in what the lighting sees, so its own shading comes
// from inside a solid: black. Drawn its main texture (terrainTextures.plain -
// never the _a, which is made for the normals and specular this light has
// none of) in the room's light instead
// (gpu/shade.js revealedLight), whiteworld and the darkening outside a
// building still applied. The light only in the branch the revealed pixels
// take - every read in it at an explicit level; the albedo before it, where
// the texel lock's derivatives are defined.
function litWhereRevealed(mat, revealedLight) {
  const shaded = mat.colorNode;
  mat.colorNode = Fn(() => {
    const c = vec3(shaded).toVar();
    const surf = terrainSurfaceTSL();
    const albedo = mix(terrainTexelTSL(terrainTextures.plain, surf.st, surf).rgb, vec3(1, 1, 1), terrainWhite).toVar();
    If(terrainRevealedTSL(), () => {
      c.assign(albedo.mul(revealedLight(surf.p, surf.n)).mul(outsideShadeTSL()));
    });
    return c;
  })();
}

export function createTexelCache({ renderer, scene, camera, rig, shadowsOn, activeSet, gi, rebuild }) {
  let texelCacheOn = true;
  // The sample spacing in pixels: 0 follows the texel's size on screen
  // (core/texelcache.js texelCacheSpacingFor), a number fixes it.
  // texelCacheScale is the spacing in use.
  let texelCacheSpacing = 0;
  let texelCacheScale = 4;
  const texelCache = {
    rt: null,
    tex: null,
    scale: uniform(new THREE.Vector2(1, 1)),   // low-res / full-res, per axis
    size: uniform(new THREE.Vector2(1, 1)),    // low-res pixels
    showMisses: false                          // debug view, compiled in
  };
  let terrainLowMat = null;      // the low-res pass's material, when the cache is on
  let terrainLookupMat = null;   // the terrain's material it belongs with
  let glassLowMat = null;        // the same pair for the glass mesh
  let glassLookupMat = null;

  // The full-res terrain is three draws of the same instances, in this order:
  //
  //   prepass  depth only - so each pixel's visible terrain fragment, and only
  //            it, passes the depth tests of the two after it
  //   lookup   the terrain mesh itself: the cached colour, stencil set to 1 on
  //            every pixel it fills; a pixel with no matching sample discards
  //   miss     the full shading, stencil-tested against 0: a pixel the lookup
  //            filled is rejected BEFORE its shader runs, so the heavy shader
  //            only ever starts on the misses
  //
  // The twins share the terrain's geometry and instance buffers, so arena
  // hiding and tinting reach them with no extra writes. Identical vertex code in
  // all three, so the visible fragment's depth matches exactly.
  const terrainTwins = { prepass: null, miss: null };
  function terrainTwin(material, renderOrder) {
    const src = worldInstancedMesh;
    const m = new THREE.InstancedMesh(src.geometry, material, src.count);
    m.instanceMatrix = src.instanceMatrix;
    m.instanceColor = src.instanceColor;
    m.frustumCulled = false;
    m.raycast = () => {};          // picking stays on the terrain mesh itself
    m.renderOrder = renderOrder;
    m.visible = false;
    scene.add(m);
    return m;
  }
  function ensureTerrainTwins() {
    if (terrainTwins.prepass) return terrainTwins;
    const depthOnly = new THREE.MeshBasicNodeMaterial({ colorWrite: false });
    // The same vertex code as the shaded draws: a slope's plane, not its box.
    depthOnly.positionNode = terrainShapeTSL();
    depthOnly.vertexNode = terrainClipTSL();
    applyOcclusionMask(depthOnly, true);   // its glass instances never cut
    // The prepass keeps every block, glass included: glass is the front surface
    // wherever it is, so the terrain behind it fails the depth test and is never
    // shaded.
    terrainTwins.prepass = terrainTwin(depthOnly, -3);
    terrainTwins.miss = terrainTwin(depthOnly, -1);   // real material set per build
    if (glassInstancedMesh) terrainTwins.glassMiss = terrainTwin(depthOnly, -1);
    return terrainTwins;
  }
  const _drawSize = new THREE.Vector2();
  function sizeTexelCache() {
    renderer.getDrawingBufferSize(_drawSize);
    texelCacheScale = texelCacheSpacing ||
      texelCacheSpacingFor(texelPixels(_drawSize.y, camera.fov, rig.distance), texelCacheScale);
    const w = Math.max(1, Math.ceil(_drawSize.x / texelCacheScale));
    const h = Math.max(1, Math.ceil(_drawSize.y / texelCacheScale));
    if (!texelCache.rt) {
      texelCache.rt = new THREE.RenderTarget(w, h, { type: THREE.FloatType, depthBuffer: true });
      texelCache.tex = texelCache.rt.texture;
      texelCache.tex.minFilter = texelCache.tex.magFilter = THREE.NearestFilter;
      texelCache.tex.generateMipmaps = false;
    } else if (texelCache.rt.width !== w || texelCache.rt.height !== h) {
      texelCache.rt.setSize(w, h);
    }
    texelCache.scale.value.set(w / _drawSize.x, h / _drawSize.y);
    texelCache.size.value.set(w, h);
    // The full-res G-buffer for split shading's resolve.
    const fw = Math.max(1, _drawSize.x), fh = Math.max(1, _drawSize.y);
    if (!texelCache.idRT) {
      texelCache.idRT = new THREE.RenderTarget(fw, fh, { type: THREE.FloatType, depthBuffer: true });
      texelCache.idRT.texture.minFilter = texelCache.idRT.texture.magFilter = THREE.NearestFilter;
      texelCache.idRT.texture.generateMipmaps = false;
    } else if (texelCache.idRT.width !== fw || texelCache.idRT.height !== fh) {
      texelCache.idRT.setSize(fw, fh);
    }
  }

  // Split shading: a rebuild to switch (bxb.splitshading()); off by itself if
  // the GPU rejects it.
  let splitOn = true;
  let splitShading = null;
  // The ambient / highlights between kernels at half precision (bxb.splithalf()).
  let splitHalf = true;
  // Misses resolved in the kernels (no miss pass) - bxb.splitresolve().
  let splitResolveOn = true;
  // Debug: the kernels' A08//00a surface path (bxb.splitlegacy()).
  let splitLegacy = false;
  let splitFailed = false;
  function ensureSplitShading() {
    if (splitShading) return splitShading;
    // Include the actual viewport: an emulated 4K canvas can exceed the screen.
    const dpr = window.devicePixelRatio || 1;
    const { full, low, miss } = splitCapacityFor(Math.max(screen.width * dpr, window.innerWidth),
                                               Math.max(screen.height * dpr, window.innerHeight));
    splitShading = createSplitShading(low, miss, full, splitHalf);
    splitShading.resolve = createSplitResolve(splitShading, texelCache);
    return splitShading;
  }
  // A validation error soon after split shading goes on: fall back to the
  // one-shader path for the session rather than draw nothing.
  let splitSince = 0;
  let watching = false;
  function watchSplitErrors() {
    const device = renderer.backend && renderer.backend.device;
    if (!device || watching) return;
    watching = true;
    device.addEventListener('uncapturederror', e => {
      if (!splitOn || splitFailed || performance.now() - splitSince > 20000) return;
      splitFailed = true;
      splitOn = false;
      console.warn('[split] the GPU rejected split shading - back to one shader. First error:',
                   e.error && e.error.message);
      rebuild();
    });
  }

  // Split shading's kernels over this frame's low-res samples, in one submit.
  function runSplitShading(resolved) {
    const set = activeSet();
    const ks = set && set.splitKernels;
    if (!ks || !splitShading) return;
    const s = splitShading, rt = texelCache.rt;
    s.width.value = rt.width;
    s.count.value = Math.min(rt.width * rt.height, s.lowCap);
    s.viewPos.value.copy(camera.position);
    const list = [];
    // Texel GI: the patch list starts empty; the base kernel fills it.
    const gk = set.giKernels;
    if (gk) list.push(gk.reset);
    if (resolved) {
      const full = texelCache.idRT;
      s.fullWidth.value = full.width;
      s.fullCount.value = Math.min(full.width * full.height, s.fullCap);
      s.resolve[1].count = s.fullCount.value;
      list.push(...s.resolve);
    }
    // This frame's low-res samples, then (resolved) up to missCap misses.
    const n = s.count.value + (resolved ? s.missCap : 0);
    for (const k of ks) { k.count = n; list.push(k); }
    // Texel GI: each listed patch traces this frame's slice of its directions.
    if (gk) gi.stepTexel(gk, list);
    renderer.compute(list);
  }

  // The low-res pass, before the main render. Only while the terrain wears the
  // material this cache was built for - mid-rebuild it may not.
  function render() {
    const mesh = worldInstancedMesh;
    const set = activeSet();
    const active = texelCacheOn && shadowsOn() && terrainLowMat && mesh
      && mesh.material === terrainLookupMat;
    // Glass joins the same cache: its own low-res, lookup and miss materials,
    // the same target and ids (a glass texel is a world texel), one prepass.
    const glass = glassInstancedMesh;
    const glassActive = !!active && glass && glassLowMat && glass.material === glassLookupMat;
    // The twins follow the terrain's own visibility (the grid overlay hides it).
    const resolved = !!(set && set.resolved);
    if (terrainTwins.prepass) {
      terrainTwins.prepass.visible = terrainTwins.miss.visible = !!active && mesh.visible && !resolved;
    }
    if (terrainTwins.glassMiss) {
      terrainTwins.glassMiss.visible = !!glassActive && mesh.visible && glass.visible && !resolved;
    }
    if (!active) return;
    sizeTexelCache();
    const mask = camera.layers.mask;
    mesh.material = terrainLowMat;
    if (glassActive) glass.material = glassLowMat;
    camera.layers.set(TEXEL_LAYER);
    renderer.setRenderTarget(texelCache.rt);
    renderer.render(scene, camera);
    // Resolved: the same G-buffer draws at full res, so every pixel knows its texel.
    if (resolved) {
      renderer.setRenderTarget(texelCache.idRT);
      renderer.render(scene, camera);
    }
    renderer.setRenderTarget(null);
    camera.layers.mask = mask;
    runSplitShading(resolved);
    mesh.material = terrainLookupMat;
    if (glassActive) glass.material = glassLookupMat;
  }

  const commands = {
    texelcache: {
      help: 'texel-rate shading: shade the terrain in a low-res pass and reuse each texel\'s colour. ' +
            "A number fixes the sample spacing in pixels; 'auto' (default) follows the texel's size on screen",
      usage: "bxb.texelcache()  toggles  |  bxb.texelcache(4)  |  bxb.texelcache('auto')  |  bxb.texelcache('misses')",
      run: (scale = null) => {
        if (scale === 'misses') texelCache.showMisses = !texelCache.showMisses;
        else if (scale === 'auto' || scale === 0) texelCacheSpacing = 0;
        else if (scale !== null) texelCacheSpacing = Math.max(1, Math.min(8, Math.round(Number(scale))));
        else texelCacheOn = !texelCacheOn;
        if (texelCacheOn) sizeTexelCache();
        if (shadowsOn()) rebuild();
        return `texel cache ${texelCacheOn ? `on, 1/${texelCacheScale} per axis` +
               (texelCacheSpacing ? '' : ' (auto)') : 'off'}`;
      }
    },
    splithalf: {
      help: 'split shading: half or full precision for the ambient / highlights between kernels (rebuilds)',
      usage: 'bxb.splithalf()',
      run: () => {
        splitHalf = !splitHalf;
        splitShading = null;          // reallocated at the new precision
        rebuild();
        return `split shading buffers at ${splitHalf ? 'half' : 'full'} precision`;
      }
    },
    splitresolve: {
      help: 'split shading: misses shaded by the kernels (on) or by the old miss pass (off) (rebuilds)',
      usage: 'bxb.splitresolve()',
      run: () => {
        splitResolveOn = !splitResolveOn;
        rebuild();
        return `misses ${splitResolveOn ? 'resolved in the kernels' : 'shaded by the miss pass'}`;
      }
    },
    splitlegacy: {
      help: "debug: split shading's kernels on A08//00a's surface path (rebuilds)",
      usage: 'bxb.splitlegacy()',
      run: () => {
        splitLegacy = !splitLegacy;
        rebuild();
        return `split kernels on the ${splitLegacy ? 'A08//00a' : 'current'} surface path`;
      }
    },
    splitstats: {
      help: "split shading: last frame's misses against the list's room",
      usage: 'await bxb.splitstats()',
      run: async () => {
        if (!splitShading) return 'split shading is off';
        let misses;
        try {
          misses = new Uint32Array(await renderer.getArrayBufferAsync(splitShading.missCount.attr))[0];
        } catch { return 'no misses resolved yet (resolve off, or the kernels have not run)'; }
        const s = splitShading;
        return `${misses} misses (room ${s.missCap}${misses > s.missCap ? ' - OVERFLOWING' : ''}), ` +
               `${s.count.value} low-res samples, ${s.fullCount.value} pixels`;
      }
    },
    splitshading: {
      help: 'light the low-res samples in compute, a feature per kernel (rebuilds)',
      usage: 'bxb.splitshading()',
      run: () => {
        splitOn = !splitOn;
        splitFailed = false;
        rebuild();
        return `split shading ${splitOn ? 'on' : 'off - one shader shades each low-res sample'}`;
      }
    }
  };

  return {
    name: 'texelcache',
    options: () => ({
      split: splitOn && texelCacheOn ? ensureSplitShading() : null,
      splitResolve: splitResolveOn,
      splitLegacy
    }),
    // What a shader set's key must hold beyond the options: the target it draws into.
    keyParts: () => [texelCacheOn, texelCacheOn ? texelCache.rt : null],
    get on() { return texelCacheOn; },
    get splitOn() { return splitOn; },
    set splitOn(v) { splitOn = !!v; },
    // Where the terrain's lit colour is this frame, for the exposure meter
    // (gpu/exposure.js): split shading's result buffer, or the low-res target
    // (under split shading that holds positions, not colour). null with no
    // cached set showing.
    meterSource() {
      const set = activeSet();
      if (!texelCacheOn || !shadowsOn() || !set || !set.lowMat) return null;
      if (set.splitKernels) return splitShading ? { split: splitShading } : null;
      return texelCache.tex ? { cache: texelCache } : null;
    },
    // The target bindings, and the meshes drawing the full-res passes.
    cache: texelCache,
    twins: terrainTwins,
    // Before a set is built: the target sized to the screen, the twins made.
    prepare() {
      if (texelCacheOn) { sizeTexelCache(); ensureTerrainTwins(); }
    },
    // The build's options carry split shading: its targets, and the watch for
    // a GPU that rejects it.
    prepareSplit(opts) {
      sizeTexelCache();
      opts.split.rt = texelCache.tex;
      opts.split.idRT = texelCache.idRT.texture;
      splitSince = performance.now();
      watchSplitErrors();
    },
    materials: (built, opts) => buildTerrainMaterials({ built, opts, cache: texelCache, cached: texelCacheOn }),
    // A set's materials to compile, each against the target it draws into.
    compilePairs(set) {
      const pairs = [];
      if (set.lowMat) pairs.push([worldInstancedMesh, set.lowMat, texelCache.rt]);
      if (set.missMat) pairs.push([terrainTwins.miss, set.missMat], [terrainTwins.prepass, terrainTwins.prepass.material]);
      if (set.glassMat) pairs.push([glassInstancedMesh, set.glassMat]);
      if (set.glassLow) pairs.push([glassInstancedMesh, set.glassLow, texelCache.rt]);
      if (set.glassMiss) pairs.push([terrainTwins.glassMiss, set.glassMiss]);
      return pairs;
    },
    // A set put on the terrain and glass meshes.
    apply(set) {
      const mesh = worldInstancedMesh;
      if (set.lowMat) {
        mesh.layers.enable(TEXEL_LAYER);
        mesh.renderOrder = -2;
      }
      mesh.material = set.terrainMat;
      terrainLowMat = set.lowMat;
      terrainLookupMat = set.terrainMat;
      if (terrainTwins.miss && set.missMat) terrainTwins.miss.material = set.missMat;
      // Glass: its traced material, drawn after the terrain's lookup when cached.
      // Without one (reflections off) it stays the plain, blended glass.
      // Every field set either way, so a set without the cache (or without glass
      // shading) never inherits the last one's layer or order.
      const glass = glassInstancedMesh;
      if (glass) {
        if (set.glassLow) glass.layers.enable(TEXEL_LAYER);
        else glass.layers.disable(TEXEL_LAYER);
        glass.renderOrder = set.glassLow ? -2 : 0;
        glass.material = set.glassMat || glass.userData.originalMaterial || glass.material;
        if (terrainTwins.glassMiss && set.glassMiss) terrainTwins.glassMiss.material = set.glassMiss;
      }
      glassLowMat = set.glassLow || null;
      glassLookupMat = set.glassLow ? set.glassMat : null;
    },
    // Back to the plain materials.
    remove() {
      const mesh = worldInstancedMesh;
      if (mesh.userData.originalMaterial) mesh.material = mesh.userData.originalMaterial;
      const glass = glassInstancedMesh;
      if (glass && glass.userData.originalMaterial) {
        glass.material = glass.userData.originalMaterial;
        glass.renderOrder = 0;
      }
      glassLowMat = glassLookupMat = null;
    },
    // The terrain's instances changed in number (an edit): the twins follow.
    syncCount() {
      for (const t of Object.values(terrainTwins)) if (t) t.count = worldInstancedMesh.count;
    },
    render,
    commands,
    // For the Debug views section.
    missesSetting: { key: 'texelmisses', label: 'Texel cache: show misses', type: 'toggle',
      help: 'red where the low-res pass left a texel for the per-pixel miss pass',
      get: () => texelCache.showMisses,
      set: v => flip(texelCache.showMisses, v, () => commands.texelcache.run('misses')) },
    // For the Perf section.
    settings: [
      { key: 'texelcache', label: 'Texel-rate shading', type: 'toggle',
        help: 'terrain shaded once per texel in a low-res pass, reused at full res',
        get: () => texelCacheOn, set: v => flip(texelCacheOn, v, () => commands.texelcache.run()) },
      { key: 'texelscale', label: 'Texel cache spacing (px, 0 auto)', type: 'range', min: 0, max: 8, step: 1,
        help: 'low-res sample spacing; texels smaller than this shade themselves. 0 follows the texel size on screen',
        get: () => texelCacheSpacing, set: v => commands.texelcache.run(v) }
    ]
  };
}
