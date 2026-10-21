import { Fn, If, Loop, Break, float, vec3, uniform, instanceIndex, positionWorld, normalWorld, floor, max, texture, uv, abs, vec2, vec4, ivec2, min, normalize, clamp, sin, smoothstep, length, int, select, textureSize, mix, step, modelWorldMatrix, frontFacing, sign, sqrt, fract, cameraPosition, pow, round, reflect, refract, textureLoad, uint, bitOr, shiftLeft, atomicMax, atomicMin, atomicStore, atomicAdd, packHalf2x16, unpackHalf2x16, uvec4 } from 'three/tsl';
import { GRID_DIM, VOXEL_METRES, GLASS_WALK_NUDGE } from '../core/boxgrid.js';
import { SUN_ANGULAR_SIZE } from '../core/sun.js';
import { BLOCK_METRES } from '../core/world.js';
import { terrainSampleTSL, terrainWhite, axisNormalTSL, blockFaceTSL as boxFaceTSL, texelLockTSL, terrainSurfaceTSL, surfaceBumpTSL, terrainTexelTSL, slopeFromCodeTSL, slopePlaneTSL } from './terrain.js';
import { layout } from './layout.js';
import { SURFACE_BIAS_VOXELS } from './field.js';
import { GLASS_LAYERS, GLASS_MAX_BOUNCES, GLASS_WALK_STEPS } from './glass.js';
import { createReflectTraceTSL } from './trace.js';
import { blockFaceTSL, blockLayerTSL, blockVolumeTSL, hitSurfaceTSL, hitTexelTSL, inHitBoundsTSL, isGlassCodeTSL, partAtTSL, skyOpenTSL } from './volume.js';
import { boxBumpedNormalTSL, labNormalTSL } from './normals.js';
import { NO_SKIP, createCardHitTSL } from './cards.js';
import { skyShTSL, sunColourUniform } from './sky.js';
import { AO_DISTANCE, createAOTSL, createSpriteAOTSL, createRevealedLightTSL, lpvAtTSL, lpvHalfCell } from './ambient.js';
import { spriteShadeNormalTSL, spriteTexelLockTSL } from './sprite.js';
import { REFLECT_MAX_DISTANCE, REFLECT_ROUGH_END, REFLECT_ROUGH_START, ROUGH_SPEC, SPEC_MAX, decodeSpecularTSL, fresnelRoughTSL, reflectBouncesUniform, reflectEscapeUniform, specularTSL } from './specular.js';
import { createMirrorLightTSL } from './mirrorlight.js';
import { HIT_FAR_METRES, hitCacheReadTSL, hitCacheWriteTSL } from './hitcache.js';
import { GI_ANCHORS, GI_CHECK, GI_LAST, GI_LIST_CAP, GI_STALE_FRAMES, GI_STAMPS, GI_STRIDE, faceFrameTSL, faceIndexTSL, patchKeyTSL, patchSideTSL, texelGILevelTSL, createTexelGIKernels, createTexelGIReadTSL, createPatchRegisterTSL } from './texelgi.js';
import { createPointLightsTSL } from './pointlights.js';
import { RAW_VIEWS, SPLIT_VIEWS, surfaceViewTSL } from './debugview.js';
import { TEXEL_ID_OFFSET, slopeLayerTable } from './texel.js';
import { DEFAULT_AMBIENT, GROUND_REFERENCE_FLOOR, createSoftSunTSL, createSunUniforms, sunShadeTSL } from './sun.js';

// ---------------------------------------------------------------------------
// Shadow shading node
//
// Phase B shades per screen pixel, not into an atlas - that is Phase C. One
// light, one shadow ray, marched through occupancy. Enough to prove traversal
// works against real geometry before the atlas is built on top of it.
// ---------------------------------------------------------------------------
// The sun is DIRECTIONAL: every surface marches along one shared direction for
// a capped distance, not toward a point. Treating it as a point light at
// (10,20,10) meant each pixel marched ~160 voxels to reach it, which is what
// killed the framerate, and put the light nearly overhead so every shadow hid
// underneath the block casting it.
//
// Capping the march is the perf lever. At 12.5cm voxels a 12m cap is ~96 steps
// worst case, and anything further away contributes a shadow too faint to
// matter at this art scale.
// Runs the SAME cone as the atlas, per fragment instead of per texel, and that
// is its whole remaining value: two passes over the same geometry in different
// spaces, sharing one marcher, so a disagreement between them is a bug in the
// addressing rather than in the shadow. It is the more expensive of the two by
// the ratio of screen pixels to world texels, and it pays that cost every frame
// where the atlas pays it per move, so the atlas is the shipping path.
// --- THE TEXEL LOCK ---
//
// The whole idea, and the only thing the atlas was ever really buying: a shadow
// is a property of the SURFACE, not of the screen. Shade a fragment at its own
// position and the shadow is resolved per pixel, so its edge slides smoothly
// under the camera and reads as a soft 3D shadow laid over flat-shaded pixel
// art - two different resolutions in one image, which is what made it look
// wrong however good the penumbra was.
//
// So snap the shading position to the centre of the world texel it falls in
// before marching. Every fragment inside one 12.5 cm texel then marches the
// IDENTICAL ray and gets the IDENTICAL answer, so the shadow comes out in whole
// texels, locked to the world grid, at exactly the density of the sprite art.
// Zoom, pan, rotate: the shadow stays painted on the ground. Walk, and its edge
// advances a texel at a time rather than sliding.
//
// This is what the atlas did, minus the atlas. No pages, no job list, no
// footprint to fall outside of, no cached answer to go stale, no second
// addressing scheme to disagree with the first - and the same result, because
// the quantisation was always the point and the storage never was. It costs a
// march per fragment instead of per texel, which is the trade the atlas existed
// to avoid; at this world's screen coverage that is the cheaper problem to have.
//
// SNAPPED ACROSS THE FACE ONLY. The component along the normal is left exactly
// as it is: quantising that would move the origin off the surface - inside it on
// one side of the texel and into the air on the other - and the bias that lifts
// the ray off the face is measured from the real surface, not from a rounded
// one. abs(n) is 1 on the face's own axis and 0 on the other two, and the terrain
// is axis-aligned boxes, so this is a select written as a lerp.

// A model's lightSource cube is drawn unlit - its own texel, the _e texture's
// where it has one (render.js modelInstanceData). Lit, every face of it faces
// away from the light at its centre and would draw dark. Wraps the terrain's
// shading, so the low-res, lookup and miss passes all see it.
export function terrainEmissiveTSL(shade, terrain) {
  return Fn(() => {
    const surf = terrainSurfaceTSL();
    return select(surf.emissive, terrainTexelTSL(terrain.albedo, surf.st, surf).rgb, shade);
  })();
}

export function createShadowColorNode({ cascades, sunDirection, maxDistance = 12,
                                        ambientUniform = null, shadeMode = 'ground',
                                        rays = 1, view = null, lights = null, sun: sunOn = true,
                                        angular = SUN_ANGULAR_SIZE,
                                        bayerTex = null, stepsUniform = null,
                                        biasUniform = null,
                                        cone = false, quantise = true,
                                  fadeStartUniform = null, edgeFadeUniform = null,
                                  cards = null, lightCards = null,
                                  terrain = null,
                                  albedo = null, specular: specOn = true,
                                  reflections: reflectOn = true, sky = null,
                                  mirrors = null,
                                  viewCards = null,
                                  ao: aoOn = true, aoDistanceUniform = null,
                                  gi = null,
                                  glassView = null, glass = false, glassShadows = true,
                                  sunMarch = true,
                                  glassBounces = GLASS_MAX_BOUNCES,
                                  glassLayers = GLASS_LAYERS, glassDispersion = 0,
                                  mirrorCache = true, modelHits = true, hitCache = null,
                                  split = null, splitLegacy = false, texelGI = null }) {
  // texelGI: createTexelGI's tables - the bounce from per-texel rays, in place
  // of the LPV's (gi). The split base kernel traces them; everything that
  // reads the bounce reads its table.
  // mirrorCache: the mirror light reads the mirror texel cache (mirrors.js);
  // the returned mirrorCache is the pass that fills it, run before rendering.
  // glass: the world has glass. Shadow rays then tint through it, and the sun's
  // visibility is a colour rather than a number (see createConeTraceSunTSL).
  // The markers below take the terrain textures, for the entry texel's tint.
  // glassShadows false (profiling only): shadow rays pass glass as if it were
  // air - the no-glass march, to measure what the tinting costs.
  const glassT = glass && glassShadows && terrain ? terrain : false;
  const sunVisZero = () => (glassT ? vec3(0, 0, 0) : float(0)).toVar();
  const asColour = v => (glassT ? v : vec3(v, v, v));
  // Held as uniforms whose .value is live, so moving the sun is a uniform write
  // rather than a material rebuild.
  const sun = createSunUniforms(sunDirection, angular);
  const maxDist = uniform(float(maxDistance));
  const ambient = ambientUniform || uniform(float(DEFAULT_AMBIENT));
  // sunMarch false (profiling only): the sun is unshadowed everywhere - to
  // measure what its march costs. The GI's own sun march is left alone.
  const sunShadow = sunMarch ? createSoftSunTSL({
    cascades, sunDir: sun.dir, sunTangent: sun.tangent,
    sunBitangent: sun.bitangent, maxDist, coneRadiusUniform: sun.coneRadius,
    rayCount: rays, bayerTex, stepsUniform, biasUniform, cone, quantise,
    fadeStartUniform, edgeFadeUniform, cards, glass: glassT, terrain
  }) : () => asColour(float(1));

  const aoDistance = aoDistanceUniform || uniform(float(AO_DISTANCE));
  // The cones are built for AO, for GI, or both - GI gathers along them - so
  // AO off with GI on still marches them and just ignores the occlusion.
  const cones = aoOn || gi
    ? createAOTSL({ cascades, distance: aoDistance, biasUniform, lpv: gi }) : null;
  const spriteAO = aoOn ? createSpriteAOTSL(cards) : null;
  // { ao, gi }: terrain cones times the sprite discs times the texture's own
  // AO, and the gathered bounce (null without GI). The bounce is scaled by the
  // texture AO too - a crevice in the art receives less of it.
  // bent: the cones' bent normal, or n without cones.
  // single: one AO cone instead of six (see createAOTSL) - what sprites use.
  // Measured (1080p): the six cones on the trees' foliage cost 0.34 ms of
  // 2.35, because an alpha-tested crossed-plane tree shades many layers deep;
  // on the terrain they cost 0.05. A sprite's AO comes mostly from its contact
  // disc and its normal map's own AO, so one cone along its normal is plenty.
  // (A cascade LOD - cheaper shading past C0 - was tried and removed: it was
  // no faster, and it dropped reflections at distance.)
  const giLevel = texelGI ? texelGILevelTSL(cascades) : null;
  // The texel GI reads (gpu/texelgi.js createTexelGIReadTSL).
  const { giRead, texelAmbient, abSky, abNear } = texelGI ? createTexelGIReadTSL(texelGI, giLevel) : {};
  // At a hit (p, n its face): the bounce and what is left of the sky ambient -
  // texel GI's patches, the LPV's light and sky visibility, or neither (null).
  const hitGI = (p, n) => (texelAmbient ? texelAmbient(p, n)
    : gi ? lpvAtTSL(gi, p.add(n.mul(float(lpvHalfCell(gi))))) : null);
  // Set inside the split kernels' builds to list the patches their rays land
  // on (reflections, the view through glass) - surfaces never seen directly
  // would have none. null in every other build, as cacheMode is set per build.
  let hitRegister = null;
  // nS: the shading normal the texel GI is seen by (the face's without one).
  // giAt: { p, n } - where the texel GI is read instead of p, n (a sprite
  // reads the ground under it).
  const occlusion = (p, n, lift, skip, texAO, single = false, nS = null, giAt = null) => {
    // giAt reads any patch, not only fresh ones: the ground under a sprite is
    // often hidden behind the sprite itself, and so not traced.
    // sky: what is left of the sky ambient (ambientLight's keep).
    // The cones first: a texel's own visibility splits its patch light.
    const r = cones ? cones(p, n, lift, skip, single || null) : null;
    const coneVis = r && aoOn ? r.vis : float(1);
    const near = r && aoOn && !giAt ? { vis: coneVis } : null;
    const t = texelAmbient
      ? texelAmbient(giAt ? giAt.p : p, giAt ? giAt.n : n, nS || n, !giAt, !!giAt, near) : null;
    const tg = t ? t.light.mul(texAO).toVar() : null;
    const sky = t ? t.keep.toVar() : null;
    if (!cones) return { ao: texAO, gi: tg, bent: n, sky };
    const sprite = spriteAO ? spriteAO(p, skip).toVar() : null;
    const vis = sprite ? coneVis.mul(sprite) : coneVis;
    // Holding the sky, the patch light takes the cones as its near
    // occlusion: split by them (ab.near, where the read had them), else
    // times them. The sprites' contact discs darken it either way.
    const giVis = near ? mix(mix(float(1), vis, abSky), sprite || float(1), abNear) : mix(float(1), vis, abSky);
    return {
      ao: vis.mul(texAO).toVar(),
      gi: tg ? tg.mul(giVis).toVar() : (gi ? r.gi.mul(gi.strength).mul(texAO).toVar() : null),
      bent: aoOn ? r.bent : n,
      // The LPV's sky visibility, gathered where the cones ended.
      sky: sky || (gi ? mix(float(1), r.sky, float(gi.sky)).toVar() : null)
    };
  };

  // --- Ambient: the sky, read along the bent normal ---
  //
  // Without a sky it is the flat ambient uniform, as it always was. With one,
  // the same uniform scales the sky's irradiance (normalised so noon, straight
  // up, is exactly the uniform), so the ambient takes the sky's colour and
  // direction: a wall facing the horizon at dusk goes orange, one facing up
  // goes blue, and an overhang's underside - whose bent normal leans out
  // toward the open side - is lit from that side. One SH evaluation.
  //
  // Deliberately NOT quantised. The ambient is the smoothest light there is;
  // banding it cut steps into a gradient that has no edges, and the dither
  // read as crosshatch. Tried and removed.
  const skyIrradiance = d => (sky
    ? skyShTSL(sky, d, true).mul(ambient).mul(sky.gain)
    : vec3(ambient, ambient, ambient));
  const ambientLight = (bent, ao, keep = null) => skyIrradiance(bent).mul(keep ? ao.mul(keep) : ao);

  // A revealed face's light: the room's (gpu/ambient.js createRevealedLightTSL).
  const revealedLight = createRevealedLightTSL(skyIrradiance, gi, split && split.room);
  // The sun's DIRECT term alone - sunShadeTSL with its ambient zeroed, so the
  // ambient can come from the sky instead. Scalar.
  const sunDirect = (visible, n, mode) =>
    sunShadeTSL({ visible, n, sunDir: sun.dir, ambientUniform: ambient, mode, ao: float(0) });

  // Every point light, in one loop over the list. Compiled in whenever there is
  // a list at all: an empty list is a loop of zero iterations, so turning lights
  // on and off is a uniform write rather than a rebuild.
  const pointLight = lights
    ? createPointLightsTSL({ cascades, lights, biasUniform,
                             fadeStartUniform, edgeFadeUniform, cards: lightCards, glass: glassT })
    : null;

  // The specular path needs the terrain's arrays (its _s texels) and its albedo,
  // which it now owns: diffuse and specular are no longer one multiply apart.
  const spec = specOn && terrain && albedo;
  const reflectTrace = spec && reflectOn ? createReflectTraceTSL(cascades) : null;
  // Normal maps on hits reach this far (hitSurfaceTSL): C1's footprint.
  const nearCascade = cascades.length > 1 ? cascades[1] : null;
  const reflectMax = float(REFLECT_MAX_DISTANCE);
  // What a reflection ray that met nothing sees: the sky - with
  // reflectEscapeUniform, only if it ended (at its full reach) above its
  // column's roof (volume.js skyOpenTSL). Indoors, it ran out of reach.
  const skyMiss = (o, d) => {
    if (!sky) return vec3(0, 0, 0);
    if (!terrain || !terrain.skyTop) return skyShTSL(sky, d);
    const open = skyOpenTSL(terrain, o.add(d.mul(reflectMax)));
    return skyShTSL(sky, d).mul(mix(float(1), open, float(reflectEscapeUniform)));
  };
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  // The sun's light, in the units the diffuse term uses. sunShadeTSL's 'ground'
  // mode divides N.L by a horizontal surface's so the ground reads fully lit
  // at any elevation; the highlight takes the same scale or it would dim as the
  // sun sets while the ground beside it did not.
  const sunLight = () => {
    const e = float(1).sub(ambient);
    return shadeMode === 'ground'
      ? e.div(max(sun.dir.y, float(GROUND_REFERENCE_FLOOR))) : e;
  };

  // Sprites, for the reflection ray: the view cards (see createCardHitTSL).
  const spriteHit = reflectTrace ? createCardHitTSL(viewCards, viewCards && viewCards.colour)
                                 : null;

  // The light at a reflection hit: the sun (with its shadow march - lifted
  // along `lift`, leaving `skip`'s card out), the light list WITH its shadows,
  // and the bounce. `mode` as the surface's own shading.
  //
  // Shadowed on purpose. It was unshadowed to spare a reflective fragment a
  // second loop of marches, and the result was a reflection that disagreed
  // with the thing reflected: a block in a torch's shadow, or round a corner
  // from a lamp, came back lit in the mirror, and no torch or lamp shadow ever
  // appeared in one. The loop is the same one the surface itself runs - same
  // shadow budget, same sprite cards - so the mirror shows what is there.
  // nS: the shading normal (normal-mapped) for every N.L and the sky; n, the
  // geometric one, still decides which faces the sun can reach at all, as it
  // does for a surface seen directly. ao: the texture's own, on the sky.
  // lambert: a flag node that picks 'lambert' over `mode` at run time, so one
  // call can light either a block or a sprite (shadeTraced).
  const hitLight = (p, n, lift, skip, mode, nS = n, ao = null, lambert = null) => {
    const tb = hitGI(p, n);
    const light = skyIrradiance(nS).toVar();
    if (ao) light.mulAssign(ao);
    if (tb) light.mulAssign(tb.keep);
    if (sunOn) {
      const vis = sunVisZero();
      If(n.dot(sun.dir).greaterThan(float(0)), () => {
        vis.assign(sunShadow(p, lift, skip));
      });
      const shade = lambert
        ? select(lambert.greaterThan(float(0.5)), sunDirect(vis, nS, 'lambert'), sunDirect(vis, nS, mode))
        : sunDirect(vis, nS, mode);
      light.addAssign(sunColourUniform.mul(shade));
    }
    // Lifted along `lift` as the sun's ray is; N.L against the hit's normal.
    if (pointLight) light.addAssign(pointLight(p, lift, skip, nS));
    if (tb) light.addAssign(tb.light);
    return light;
  };

  // Reflected light from the mirrors (createMirrorLightTSL) - the sun and the
  // light list, thrown here by every polished rectangle that can reach.
  const mirrorLight = createMirrorLightTSL({
    cascades, terrain, mirrors, lights, sun, sunShadow, sunLight, sunOn, bias,
    fadeStartUniform, edgeFadeUniform, cards, lightCards,
    cache: mirrorCache && !!mirrors, sunColoured: !!glassT });
  const mirrorTrace = mirrorLight;

  // What a reflection ray sees: the nearer of the terrain it marched to and
  // any sprite card it crosses first. Terrain shows its block's texture; a
  // sprite its own pixels, lit as the sprites are (Lambert, facing back along
  // the ray, since the card was turned to be seen). Past the field, the sky.
  //
  // In three parts, so that everything a shader lights can go through ONE sun
  // march and ONE light loop (lightBatch): the D3D12 shader compiler (DXC,
  // behind both Firefox and Chrome) inlines every function at every call site,
  // so each place that lit something was another full copy of the lighting to
  // compile. The values are the same - the same functions, with what differed
  // between the call sites chosen before them.
  //
  // traceRecord: what a ray from origin along dir met, given its trace (hit) -
  // into the vars of `rec` (newHitRecord), so it can be filled inside a branch.
  // direct: a flag node - the hit is a known surface at origin, normal hit.xyz
  // (an opaque block met inside glass): no sprite in front, no arena test,
  // as glass's own walk has already judged both.
  const newHitRecord = () => ({
    col: vec3(0, 0, 0).toVar(), P: vec3(0, 0, 0).toVar(), N: vec3(0, 1, 0).toVar(),
    lift: vec3(0, 1, 0).toVar(), nS: vec3(0, 1, 0).toVar(), alb: vec3(0, 0, 0).toVar(),
    ao: float(1).toVar(), skip: vec2(NO_SKIP).toVar(), lambert: float(0).toVar(),
    met: float(0).toVar(), far: float(0).toVar(),
    // For a block hit that may reflect in turn (nestedReflect): the ray that
    // met it, its _s texel, and 1 once the surface is read.
    dir: vec3(0, 0, 1).toVar(), spec: vec4(...ROUGH_SPEC).toVar(), nest: float(0).toVar()
  });
  // How the shader being built reaches the hit lighting cache: 'atomic' (the
  // low-res terrain, which writes it), 'read' (the other terrain passes) or
  // 'none' (glass: it shares the low-res pass with the writer and the main
  // pass with the readers, and a pass may bind the buffer only one way). Set
  // at the top of each node's Fn, which builds its traces synchronously.
  let cacheMode = 'none';
  const traceRecord = (origin, dir, hit, direct = null, rec = newHitRecord()) => {
    // A miss sees the sky - the same SH the background draws.
    rec.col.assign(skyMiss(origin, dir));
    const sp = spriteHit
      ? spriteHit(origin, dir, select(hit.w.greaterThanEqual(float(0)), hit.w, reflectMax))
      : null;
    // A hit on a block outside the battle arena is not on screen: a miss.
    const hitAt = origin.add(dir.mul(max(hit.w, float(0)))).toVar();
    const inArena = inHitBoundsTSL(hitAt.sub(hit.xyz.mul(float(VOXEL_METRES * 0.5))));
    const traced = sp ? hit.w.greaterThanEqual(float(0)).and(inArena).and(sp.t.lessThan(float(0)))
                      : hit.w.greaterThanEqual(float(0)).and(inArena);
    const terrainWins = direct ? direct.greaterThan(float(0.5)).or(traced) : traced;
    // A block or a sprite, never both (terrainWins needs no sprite in front).
    If(terrainWins, () => {
      const ht = hitTexelTSL(terrain, hitAt, hit.xyz, dir);
      const nh = ht.n, hpL = ht.p.toVar();
      const surface = () => {
        // modelHits false (profiling only): models reflect as layer 0, to
        // price the lookup's weight in the shader.
        const hs = hitSurfaceTSL(modelHits ? terrain : { ...terrain, modelHits: null }, hpL, nh, nearCascade);
        rec.P.assign(hpL); rec.N.assign(nh); rec.lift.assign(nh); rec.nS.assign(hs.n);
        rec.ao.assign(hs.ao); rec.alb.assign(hs.albedo);
        rec.dir.assign(dir); rec.spec.assign(hs.spec); rec.nest.assign(float(1));
        if (hitRegister) hitRegister(hpL, nh);
        // Far along the ray: small on screen - point lights light it unshadowed.
        rec.far.assign(select(hit.w.greaterThan(float(HIT_FAR_METRES)), float(1), float(0)));
      };
      if (!hitCache || cacheMode === 'none') { surface(); rec.met.assign(float(1)); return; }
      // Lit on screen last frame: its colour as is, and no light batch (met 0).
      // Its surface is still read when it may reflect in turn.
      const cached = hitCacheReadTSL(hitCache, hpL, nh, cacheMode === 'atomic');
      If(cached.ok.not().or(reflectBouncesUniform.greaterThan(int(1))), surface);
      If(cached.ok, () => {
        rec.col.assign(cached.colour); rec.met.assign(float(0));
      }).Else(() => { rec.met.assign(float(1)); });
    });
    if (sp) {
      const spriteWins = direct ? sp.t.greaterThanEqual(float(0)).and(direct.lessThan(float(0.5)))
                                : sp.t.greaterThanEqual(float(0));
      If(spriteWins, () => {
        const back = vec3(dir.x.negate(), float(0), dir.z.negate());
        const ns = select(length(back).greaterThan(float(1e-4)), normalize(back),
                          vec3(0, 1, 0)).toVar();
        // Lifted straight up, as the sprites' own shading is - see spriteLight.
        rec.P.assign(origin.add(dir.mul(sp.t))); rec.N.assign(ns); rec.lift.assign(vec3(0, 1, 0));
        rec.nS.assign(ns); rec.alb.assign(sp.col); rec.skip.assign(sp.skip);
        rec.lambert.assign(float(1)); rec.met.assign(float(1));
      });
    }
    return rec;
  };
  // --- Reflections in reflections ---
  //
  // A block hit that is itself smooth shows its own reflection, and that one
  // its own, up to reflectBouncesUniform surfaces in all. Each surface keeps
  // the diffuse share of its colour (1 - F, none for a metal) and passes on F
  // of the next one, faded out over the rough band as the first reflection is.
  // The surfaces past the first are lit cheaply: the sky, the sun with its
  // shadow, and the bounce - no light list (the torch reaches them only through
  // the bounce). Not the hit lighting cache: a layout function's body is built
  // once per shader, and which way the cache is bound (cacheMode) is not fixed
  // then. A layout function, so the loop is compiled once a shader, however
  // many places compose a hit.
  // base: the first surface's lit colour; p, n, nS, alb, spec: that surface;
  // d: the ray that met it. Returns the colour it shows, reflections and all.
  const nestedReflect = reflectTrace ? Fn(([base0, p0, n0, nS0, d0, alb0, spec0]) => {
    const out = vec3(0, 0, 0).toVar(), thr = vec3(1, 1, 1).toVar();
    const base = base0.toVar(), p = p0.toVar(), n = n0.toVar(), nS = nS0.toVar();
    const d = d0.toVar(), alb = alb0.toVar(), spec = spec0.toVar();
    Loop({ start: int(1), end: reflectBouncesUniform, type: 'int', condition: '<', name: 'rb' }, () => {
      const sd = decodeSpecularTSL(spec, alb);
      const w = float(1).sub(smoothstep(float(REFLECT_ROUGH_START), float(REFLECT_ROUGH_END), sd.alpha)).toVar();
      If(w.lessThanEqual(float(0)), () => { Break(); });
      const F = fresnelRoughTSL(sd.f0, max(nS.dot(d.negate()), float(1e-4)), sd.alpha).toVar();
      const kd = vec3(1, 1, 1).sub(F).mul(float(1).sub(sd.metal));
      out.addAssign(thr.mul(base).mul(mix(vec3(1, 1, 1), kd, w)));
      thr.mulAssign(F.mul(w));
      const r0 = reflect(d, nS).toVar();
      const r = select(r0.dot(n).greaterThan(float(0.02)), r0, reflect(d, n)).toVar();
      const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();
      const hit = reflectTrace(origin, r, reflectMax).toVar();
      const hitAt = origin.add(r.mul(max(hit.w, float(0)))).toVar();
      // A miss, until a block says otherwise: the sky, and nothing further.
      base.assign(skyMiss(origin, r));
      spec.assign(vec4(...ROUGH_SPEC));
      d.assign(r);
      If(hit.w.greaterThanEqual(float(0)).and(inHitBoundsTSL(hitAt.sub(hit.xyz.mul(float(VOXEL_METRES * 0.5))))), () => {
        const ht = hitTexelTSL(terrain, hitAt, hit.xyz, r);
        const hp = ht.p.toVar(), nh = ht.n.toVar();
        const hs = hitSurfaceTSL(terrain, hp, nh, nearCascade);
        p.assign(hp); n.assign(nh); nS.assign(hs.n); alb.assign(hs.albedo); spec.assign(hs.spec);
        const lightIt = () => {
          const tb = hitGI(hp, nh);
          const light = skyIrradiance(hs.n).mul(hs.ao).toVar();
          if (tb) light.mulAssign(tb.keep);
          if (sunOn) {
            const vis = sunVisZero();
            If(nh.dot(sun.dir).greaterThan(float(0)), () => { vis.assign(sunShadow(hp, nh, NO_SKIP)); });
            light.addAssign(sunColourUniform.mul(sunDirect(vis, hs.n, shadeMode)));
          }
          if (tb) light.addAssign(tb.light);
          base.assign(hs.albedo.mul(light));
        };
        lightIt();
      });
    });
    out.addAssign(thr.mul(base));
    return out;
  }).setLayout(layout('nestedReflect', 'vec3', [
    { name: 'base0', type: 'vec3' }, { name: 'p0', type: 'vec3' }, { name: 'n0', type: 'vec3' },
    { name: 'nS0', type: 'vec3' }, { name: 'd0', type: 'vec3' }, { name: 'alb0', type: 'vec3' },
    { name: 'spec0', type: 'vec4' }])) : null;

  // A record as a lighting request (lightBatch).
  const hitRequest = rec => ({ active: rec.met, P: rec.P, N: rec.N, lift: rec.lift,
                               skip: rec.skip, nS: rec.nS, secondary: true, far: rec.far });
  // The colour a record shows, from its request's answer (vis, diff): what
  // hitLight computes, term for term and in its order.
  const composeHit = (rec, vis, diff) => {
    const out = rec.col.toVar();
    If(rec.met.greaterThan(float(0.5)), () => {
      const tb = hitGI(rec.P, rec.N);
      const light = skyIrradiance(rec.nS).toVar();
      light.mulAssign(tb ? rec.ao.mul(tb.keep) : rec.ao);
      if (sunOn) {
        light.addAssign(sunColourUniform.mul(select(rec.lambert.greaterThan(float(0.5)),
          sunDirect(vis, rec.nS, 'lambert'), sunDirect(vis, rec.nS, shadeMode))));
      }
      if (pointLight) light.addAssign(diff);
      if (tb) light.addAssign(tb.light);
      out.assign(rec.alb.mul(light));
    });
    if (!nestedReflect) return out;
    // A smooth block hit shows its own reflections too.
    const shown = out.toVar();
    If(rec.nest.greaterThan(float(0.5)).and(reflectBouncesUniform.greaterThan(int(1))), () => {
      shown.assign(nestedReflect(out, rec.P, rec.N, rec.nS, rec.dir, rec.alb, rec.spec));
    });
    return shown;
  };

  // Light for several points through ONE sun march and ONE light loop, in a
  // loop over the requests: { active (a flag node), P, N (the geometric
  // normal: which faces the sun can reach), lift (the ray's lift), skip, nS
  // (N.L), brdf: null or { v, alpha, f0 } for highlights }. Answers, per
  // request, { vis: the sun's visibility (0 facing away), diff: the light
  // list's diffuse, spec: its highlights }. An inactive request is skipped.
  const lightBatch = reqs => {
    const outs = reqs.map(() => ({ vis: sunVisZero(), diff: vec3(0, 0, 0).toVar(),
                                   spec: vec3(0, 0, 0).toVar() }));
    const anySpec = reqs.some(r => r.brdf);
    // One request needs no loop: the body runs as it stands.
    const over = reqs.length > 1
      ? fn => Loop({ start: int(0), end: int(reqs.length), type: 'int', condition: '<', name: 'lr' },
                   ({ lr }) => fn(lr))
      : fn => fn(null);
    over(lr => {
      // Request lr's value of f (the last one where no other matches).
      const pick = f => (lr ? reqs.slice(0, -1).reduceRight(
        (acc, r, i) => select(lr.equal(int(i)), f(r), acc), f(reqs[reqs.length - 1])) : f(reqs[0]));
      If(pick(r => float(r.active)).greaterThan(float(0.5)), () => {
        const P = pick(r => r.P).toVar(), N = pick(r => r.N).toVar();
        const lift = pick(r => r.lift).toVar(), skip = pick(r => r.skip).toVar();
        const nS = pick(r => r.nS).toVar();
        // 0 primary, 1 a secondary hit, 2 a far one (no point-light marches).
        const second = reqs.some(r => r.secondary)
          ? pick(r => (r.secondary ? float(1).add(r.far || float(0)) : float(0))).toVar() : null;
        const vis = sunVisZero();
        if (sunOn) {
          If(N.dot(sun.dir).greaterThan(float(0)), () => { vis.assign(sunShadow(P, lift, skip)); });
        }
        const diff = vec3(0, 0, 0).toVar(), spec = vec3(0, 0, 0).toVar();
        if (pointLight) {
          if (anySpec) {
            // Requests without highlights take the first one's inputs; their
            // spec is not read.
            const b = r => r.brdf || reqs.find(q => q.brdf).brdf;
            const brdf = specularTSL(nS, pick(r => b(r).v), pick(r => b(r).alpha), pick(r => b(r).f0));
            const pl = pointLight.withSpecular(P, lift, skip, nS, brdf, second);
            diff.assign(pl.diffuse); spec.assign(pl.spec);
          } else {
            diff.assign(pointLight.withSpecular(P, lift, skip, nS, null, second));
          }
        }
        reqs.forEach((r, i) => {
          const store = () => {
            outs[i].vis.assign(vis); outs[i].diff.assign(diff);
            if (r.brdf) outs[i].spec.assign(spec);
          };
          if (lr) If(lr.equal(int(i)), store); else store();
        });
      });
    });
    return outs;
  };
  // One ray's colour, lit on its own.
  const shadeTraced = (origin, dir, hit, direct = null) => {
    const rec = traceRecord(origin, dir, hit, direct);
    const [o] = lightBatch([hitRequest(rec)]);
    return composeHit(rec, o.vis, o.diff);
  };
  const reflectedTSL = (origin, dir, maxDist = reflectMax) =>
    shadeTraced(origin, dir, reflectTrace(origin, dir, maxDist).toVar());

  // --- Glass: a surface whose shading traces through it ---
  //
  // Glass is not blended over what is behind it. It is shaded like any other
  // texel - locked, one answer per texel - and its answer includes what it
  // shows: the Fresnel share of a reflection ray, and the rest of a ray sent on
  // THROUGH it. Glass is only in the field's glass channel (boxgrid.js), and the
  // trace reads the opaque one, so that ray walks straight through the glass -
  // and any other glass - to the first opaque surface, and shades it with the
  // same hit shading reflections use: its texture, the sun with its shadow, the
  // light list with theirs, the bounce, and sprites through the view cards.
  // So nothing is sorted, nothing is blended, and glass takes the texel cache.
  //
  // Stage 3: the through-ray refracts in, walks through the glass block by
  // block (below), and refracts out - or, past the critical angle, reflects
  // inside and walks on, up to glassBounces times (GLASS_MAX_BOUNCES unless
  // set - Alt+V -> Glass). Glass
  // casting light and shadow is stage 5; until then it casts none.
  //
  // The glass's albedo is not a diffuse colour - glass has none - but its
  // tint through one block, absorbed by thickness (Beer-Lambert).
  // glassView: 'thickness' shows the path length inside the glass (white at
  // two blocks), 'through' the transmitted light alone, 'path' how each
  // texel's ray ended (see below).
  // Everything a glass texel lights goes through one lightBatch in glassNode.
  const skyAlong = d => (sky ? skyShTSL(sky, d) : vec3(0, 0, 0));
  // The ambient at q, facing dirq: sky irradiance plus the bounce. What glass
  // too rough to see through shows, and where a ray's budget ran out.
  const ambientAt = (q, dirq) => {
    const a = skyIrradiance(dirq).toVar();
    if (gi) {
      const l = lpvAtTSL(gi, q);
      a.assign(a.mul(l.keep).add(l.light));
    }
    return a;
  };

  // A ray leaving glass: the trace that also stops at glass. Its hit, as
  // reflectTrace answers; w <= -2 means more glass, -2 - w away. What it saw is
  // shaded after the walk, with the walk's other endings (glassThroughRec).
  // Only built for a world with glass and more than one layer.
  const glassSeekTrace = reflectTrace && glass && glassLayers > 1
    ? createReflectTraceTSL(cascades, true) : null;
  const seenOrGlassFn = glassSeekTrace;

  // --- What shows through glass, at one index of refraction ---
  //
  // Dispersion calls it three times - once per colour, at slightly different
  // indices. In the debug views it answers the view's colour instead.
  // smoothW, not smooth: `smooth` is a reserved word in WGSL (and the
  // compile fails on it) - tools/wgslnames.test.mjs checks every name.
  //
  // The ray refracts in, walks the glass block by block (below), refracts out -
  // and, while glassLayers allows, if what it meets next is more glass, it
  // refracts into that pane and walks on: glass seen through glass. Each pane
  // absorbs by its own length and the tint of the texel the ray entered it by.
  // Returns what the walk ended on rather than its colour - { rec, need,
  // pre, sO, sD, exitW, smoothW, absorb } - so glassNode lights it in one
  // batch with the surface's own reflection and highlights, and
  // glassThroughColour finishes it. In the debug views, { debug: colour }.
  // Inline, not a shader function: its answer is many values. Dispersion
  // calls it three times, so that (debug) build holds three walks.
  const glassThroughRec = reflectTrace ? (p, n, nS, v, ior, f0, alpha, tint0, smoothW) => {
    const lift = bias.mul(float(VOXEL_METRES));
    const f0v = vec3(f0, f0, f0);
    // In: bent toward the normal. Entering a denser medium cannot reflect
    // totally, so refract always answers.
    const dIn = refract(v.negate(), nS, float(1).div(ior)).toVar();
    // What the ray came to: 0 still walking, 1 an opaque block inside the
    // glass (what shows through, e.g. the ground under the cube), 2 out of the
    // glass - trace what it sees, 3 trapped (internal reflections spent), 4 too
    // rough to trace, 5 out, and what it sees already found by the seek.
    const kind = int(0).toVar();
    const lenAll = float(0).toVar();
    const absorb = vec3(1, 1, 1).toVar();
    const tint = tint0.toVar();
    const hitP = vec3(0, 0, 0).toVar(), hitN = vec3(0, 1, 0).toVar();
    const outO = vec3(0, 0, 0).toVar(), outD = dIn.toVar();
    const exitW = float(1).toVar();
    const bounces = int(0).toVar();          // internal reflections taken
    const seenHit = vec4(0, 0, 0, -1).toVar();   // kind 5: the seek's hit
    const layers = int(0).toVar();           // panes entered

    // --- The walk through the glass, block by block ---
    //
    // Glass is whole blocks, so the path inside it is walked on the block grid
    // itself (Amanatides-Woo), not marched through the distance field: the
    // field's resolution is the cascade's, and at C1's 25 cm voxels it could not
    // tell an edge texel's start from the outside, while the grid is exact at
    // any distance. Each step crosses into the next block along the ray: glass,
    // keep going; air, the ray leaves through that face - refracting out, or
    // past the critical angle reflecting back in and walking on; anything else,
    // it has met an opaque block, and that face is what the glass shows.
    //
    // A steep ray in a slab reflects off its sides again and again on the way
    // down, like light in a light pipe, so the budget is several reflections,
    // not one - each is a step or two, and a block is 1.5 m.
    const pos = p.toVar();
    const dir = dIn.toVar();
    // The glass box the ray is in (partAtTSL): a whole block's is its cell,
    // a half's or a pane's its own. Faces are crossed a nudge deep to ask
    // what is past them (boxgrid.js GLASS_WALK_NUDGE).
    const nudge = float(GLASS_WALK_NUDGE);
    const part0 = partAtTSL(terrain, p.sub(n.mul(nudge)));
    const boxLo = part0.lo.toVar(), boxHi = part0.hi.toVar();
    // Only when the glass is smoothW enough to see through at all.
    If(smoothW.greaterThan(float(0)), () => {
    Loop({ start: int(0), end: int(glassLayers), type: 'int', condition: '<', name: 'gl' }, ({ gl }) => {
      layers.addAssign(int(1));
      const paneLen = float(0).toVar();
      kind.assign(int(0));
      Loop({ start: int(0), end: int(GLASS_WALK_STEPS), type: 'int', condition: '<', name: 'gw' }, () => {
        const st3 = sign(dir).toVar();
        // Distance along the ray to the box's next face on each axis. Per
        // component by mix / step: select's condition is ONE bool (three's
        // ConditionalNode), so a bvec3 there picks a whole vector.
        const bound = mix(boxLo, boxHi, step(vec3(0, 0, 0), dir));
        const flat = step(abs(dir), vec3(1e-6, 1e-6, 1e-6)).toVar();   // 1 where the ray runs parallel
        const safe = mix(dir, vec3(1e-6, 1e-6, 1e-6), flat);
        const tm = mix(bound.sub(pos).div(safe), vec3(1e9, 1e9, 1e9), flat).toVar();
        const ax = tm.x.lessThanEqual(tm.y).and(tm.x.lessThanEqual(tm.z));
        const ay = tm.y.lessThan(tm.x).and(tm.y.lessThanEqual(tm.z));
        const e = select(ax, vec3(1, 0, 0), select(ay, vec3(0, 1, 0), vec3(0, 0, 1))).toVar();
        const tA = max(select(ax, tm.x, select(ay, tm.y, tm.z)), float(0)).toVar();
        const npos = pos.add(dir.mul(tA)).toVar();
        const nx = e.mul(st3).toVar();             // the face crossed, pointing along the ray
        // What is past it. Outside the battle arena a block is hidden, so it
        // is air here too.
        const next = partAtTSL(terrain, npos.add(nx.mul(nudge)));
        const code = next.code.toVar();
        paneLen.addAssign(tA);
        If(isGlassCodeTSL(code), () => {
          pos.assign(npos);
          boxLo.assign(next.lo);
          boxHi.assign(next.hi);
        }).ElseIf(code.equal(float(0)), () => {
          // Air: leave through nx, or reflect back in.
          const outR = refract(dir, nx.negate(), ior).toVar();
          If(length(outR).greaterThan(float(0.5)), () => {
            const od = normalize(outR).toVar();
            const ce = max(od.dot(nx), float(1e-4));
            exitW.mulAssign(float(1).sub(fresnelRoughTSL(f0v, ce, alpha).x));
            outO.assign(npos.add(nx.mul(lift)));
            outD.assign(od);
            kind.assign(int(2));
            Break();
          }).Else(() => {
            bounces.addAssign(int(1));
            outO.assign(npos);
            outD.assign(dir);
            If(bounces.greaterThan(int(glassBounces)), () => {
              kind.assign(int(3));
              Break();
            });
            dir.assign(reflect(dir, nx));
            pos.assign(npos);
          });
        }).Else(() => {
          // Opaque: its face toward the ray.
          hitP.assign(npos);
          hitN.assign(nx.negate());
          kind.assign(int(1));
          Break();
        });
      });
      // Beer-Lambert per pane: its albedo is the tint through one block (1.5 m),
      // so over its length it is tint^(len / 1.5). A ray that clips an edge
      // stays nearly clear; one across the cube's middle takes the full tint.
      absorb.mulAssign(pow(max(tint, vec3(1e-3, 1e-3, 1e-3)),
                           vec3(paneLen, paneLen, paneLen).div(float(BLOCK_METRES))));
      lenAll.addAssign(paneLen);
      If(kind.notEqual(int(2)), () => { Break(); });
      if (!seenOrGlassFn) {
        Break();
      } else {
        // Out of this pane. With layers left, look for more glass on the way.
        If(gl.lessThan(int(glassLayers - 1)), () => {
          const hit = seenOrGlassFn(outO, outD, reflectMax).toVar();
          If(hit.w.lessThanEqual(float(-2)), () => {
            // Into the next pane: its face from the block grid, its tint from
            // the texel the ray enters by, less the Fresnel it reflects away.
            const entry = outO.add(outD.mul(float(-2).sub(hit.w))).toVar();
            const nf = blockFaceTSL(entry, outD, false, terrain).toVar();
            const pL = texelLockTSL(entry, nf).toVar();
            const f = boxFaceTSL(pL, nf);
            tint.assign(texture(terrain.albedo, f.st).depth(int(blockLayerTSL(terrain, f.block, nf, pL)))
                          .level(int(0)).rgb);
            const ce = max(outD.dot(nf.negate()), float(1e-4));
            exitW.mulAssign(float(1).sub(fresnelRoughTSL(f0v, ce, alpha).x));
            dir.assign(refract(outD, nf, float(1).div(ior)));
            pos.assign(pL);
            const entered = partAtTSL(terrain, pL.sub(nf.mul(nudge)));
            boxLo.assign(entered.lo);
            boxHi.assign(entered.hi);
          }).Else(() => {
            seenHit.assign(hit);
            kind.assign(int(5));
            Break();
          });
        }).Else(() => { Break(); });
      }
    });
    }).Else(() => {
      // Fully rough: nothing is traced; the ambient beyond the glass, tinted as
      // through one block of it.
      kind.assign(int(4));
      lenAll.assign(float(BLOCK_METRES));
      absorb.assign(tint);
      outO.assign(p.sub(n.mul(lift)));
      outD.assign(v.negate());
    });

    // Which way each texel's ray ended, as a colour: red an opaque surface
    // inside the glass, green out through a face, blue out after an internal
    // reflection, cyan out after crossing more than one pane, white trapped,
    // grey too rough to trace.
    if (glassView === 'path') {
      const out2 = kind.equal(int(2)).or(kind.equal(int(5)));
      return { debug: select(kind.equal(int(1)), vec3(1, 0.2, 0.2),
             select(out2,
                    select(layers.greaterThan(int(1)), vec3(0.2, 1, 1),
                           select(bounces.greaterThan(int(0)), vec3(0.2, 0.4, 1), vec3(0.2, 1, 0.2))),
                    select(kind.equal(int(4)), vec3(0.5, 0.5, 0.5), vec3(1, 1, 1)))) };
    }
    if (glassView === 'thickness') {
      const k = lenAll.div(float(BLOCK_METRES * 2));
      return { debug: vec3(k, k, k) };
    }
    // Whatever the walk ended on - an opaque block inside the glass (1), the
    // world beyond a face (2, traced now), or what the seek already met (5) -
    // is recorded here and lit with the rest of the texel's lighting in one
    // batch (glassNode). The other endings need no lighting: `pre`.
    const pre = vec3(0, 0, 0).toVar();
    const isOpaque = kind.equal(int(1));
    const need = isOpaque.or(kind.equal(int(2))).or(kind.equal(int(5)));
    const rec = newHitRecord();
    const sO = select(isOpaque, hitP, outO).toVar();
    const sD = select(isOpaque, hitN, outD).toVar();
    If(need, () => {
      const sHit = seenHit.toVar();
      If(isOpaque, () => { sHit.assign(vec4(hitN, 0)); })
        .ElseIf(kind.equal(int(2)), () => { sHit.assign(reflectTrace(outO, outD, reflectMax)); });
      traceRecord(sO, sD, sHit, select(isOpaque, float(1), float(0)), rec);
    })
      .ElseIf(kind.equal(int(4)), () => { pre.assign(ambientAt(outO, outD)); })
      .Else(() => {
        // Trapped: the budget of internal reflections is spent. Not the sky's
        // radiance - that is a bright light the ray never reached - but the
        // ambient around it: sky irradiance along the last direction, plus the
        // bounce where it stopped.
        pre.assign(ambientAt(outO, outD));
      });
    return { rec, need, pre, sO, sD, exitW, smoothW, absorb };
  } : null;
  // The colour through the glass, once its record is lit (lightBatch's answer).
  // Rougher glass shows less of what is beyond and more of the ambient.
  const glassThroughColour = (t, lit) => {
    if (t.debug) return t.debug;
    const seen = t.pre.toVar();
    If(t.need, () => {
      seen.assign(mix(ambientAt(t.sO, t.sD), composeHit(t.rec, lit.vis, lit.diff), t.smoothW)
                    .mul(t.exitW));
    });
    return seen.mul(t.absorb);
  };

  const glassNode = reflectTrace ? Fn(() => {
    cacheMode = 'none';
    hitRegister = null;
    const n = axisNormalTSL(normalWorld).toVar();
    const p = texelLockTSL(positionWorld, n).toVar();
    const st = boxFaceTSL(p, n).st.toVar();
    const sample = tex => terrainSampleTSL(tex, st);
    const lab = labNormalTSL(sample(terrain.normal)).toVar();
    const nS = boxBumpedNormalTSL(n, lab.xyz).toVar();
    const tint = albedo(st).toVar();
    const d = decodeSpecularTSL(sample(terrain.specular), tint);
    const v = normalize(cameraPosition.sub(p)).toVar();
    const nv = max(nS.dot(v), float(1e-4));
    const F = fresnelRoughTSL(d.f0, nv, d.alpha).toVar();
    const lift = bias.mul(float(VOXEL_METRES));
    // Frosted glass is its normal map: noise at texel resolution (see
    // tools/make-glass-textures.py), read at the locked texel like every other
    // input, so both rays bend off it and the grain is stable, authored, and
    // free. Its roughness still spreads the highlights, and past the smooth
    // band - the one the terrain's reflections use - the glass is too rough to
    // show anything through or in it: both rays give way to the ambient, and
    // are not traced at all once the ambient is all that is left.
    const smooth = float(1).sub(smoothstep(float(REFLECT_ROUGH_START),
                                           float(REFLECT_ROUGH_END), d.alpha)).toVar();
    // Reflected: off the bumped normal, kept on the outside of the face.
    const r0 = reflect(v.negate(), nS).toVar();
    const rdir = select(r0.dot(n).greaterThan(float(0.02)), r0, reflect(v.negate(), n)).toVar();
    const reflOrigin = p.add(n.mul(lift)).toVar();
    const reflRec = newHitRecord();
    If(smooth.greaterThan(float(0)), () => {
      traceRecord(reflOrigin, rdir, reflectTrace(reflOrigin, rdir, reflectMax).toVar(), null, reflRec);
    });

    // --- Transmitted ---
    //
    // The index of refraction from the texel's own F0 (Schlick inverted):
    // F0 = ((n - 1) / (n + 1))^2, so n = (1 + sqrt F0) / (1 - sqrt F0). The
    // LabPBR glass texel's F0 of 0.04 is n = 1.5.
    const sf = sqrt(clamp(d.f0.x, float(0), float(0.5)));
    const ior = float(1).add(sf).div(float(1).sub(sf)).toVar();
    // Dispersion (off unless glassDispersion): red and blue take their own
    // index - three traces instead of one - so edges split into colour.
    const iors = glassDispersion > 0
      ? [ior.mul(float(1 - glassDispersion)), ior, ior.mul(float(1 + glassDispersion))] : [ior];
    const throughs = iors.map(x => glassThroughRec(p, n, nS, v, x, d.f0.x, d.alpha, tint, smooth));
    const traced = throughs.filter(t => !t.debug);
    // Everything this texel lights - its own highlights, what its reflection
    // meets, what shows through it - in ONE batch: one sun march, one light
    // loop (lightBatch).
    const lit = lightBatch([
      { active: float(1), P: p, N: n, lift: n, skip: NO_SKIP, nS,
        brdf: { v, alpha: d.alpha, f0: d.f0 } },
      hitRequest(reflRec),
      ...traced.map(t => hitRequest(t.rec))]);
    const colours = throughs.map(t => glassThroughColour(t, t.debug ? null : lit[2 + traced.indexOf(t)]));
    const seenThrough = (glassDispersion > 0
      ? vec3(colours[0].x, colours[1].y, colours[2].z) : colours[0]).toVar();
    if (glassView) return seenThrough;
    const reflected = ambientAt(reflOrigin, nS).toVar();
    If(smooth.greaterThan(float(0)), () => {
      reflected.assign(mix(reflected, composeHit(reflRec, lit[1].vis, lit[1].diff), smooth));
    });
    // Highlights: the sun, and every light, from the same visibility the
    // surfaces use.
    const brdf = specularTSL(nS, v, d.alpha, d.f0);
    const direct = vec3(0, 0, 0).toVar();
    if (sunOn) {
      If(n.dot(sun.dir).greaterThan(float(0)), () => {
        direct.addAssign(brdf(sun.dir).mul(lit[0].vis)
                           .mul(sunLight()).mul(sunColourUniform));
      });
    }
    if (pointLight) direct.addAssign(lit[0].spec);
    return F.mul(reflected)
      .add(vec3(1, 1, 1).sub(F).mul(seenThrough))
      .add(min(direct, vec3(SPEC_MAX, SPEC_MAX, SPEC_MAX)));
  })() : null;

  // Stable per-texel randoms, for the rough-surface jitter below. Hashed from
  // the world texel, so the grain is painted on and does not crawl.
  const texelHash = (wt, k) =>
    fract(sin(wt.dot(vec3(12.9898 + k, 78.233 - k, 37.719 + 2 * k))).mul(float(43758.5453)));

  // writeCache: this copy also fills the hit lighting cache. Only the texel
  // cache's LOW-RES pass takes it: a fragment shader that writes storage
  // loses early depth and stencil tests, and the miss pass relies on its
  // stencil test to shade only what the lookup could not fill (measured:
  // +4 ms with the write in every pass).
  const makeNode = (writeCache = false) => Fn(() => {
    cacheMode = hitCache ? (writeCache ? 'atomic' : 'read') : 'none';
    hitRegister = null;
    // normalWorld rather than a derivative reconstruction: the terrain is boxes,
    // so the interpolated normal IS the exact face normal, and the marcher needs
    // it to lift the ray origin off the face along the surface rather than along
    // the ray - see SURFACE_BIAS_VOXELS for why that distinction matters at low
    // sun angles.
    // (A slope's plane gives its own exact normal - terrainSurfaceTSL.)
    const face = terrainSurfaceTSL(!!terrain);
    const n = face.n;
    // Locked once, used by every light. Two lights that disagreed about where a
    // surface IS would put their shadows on different texel grids, and the
    // mismatch would read as the torch's shadow crawling against the sun's.
    const p = face.p;
    // Every texture read at THIS texel's centre, from the locked position -
    // not at the fragment's own uv, which at a texel's edge can read the
    // neighbour's normal or specular while p is still this texel's.
    const st = terrain ? face.st : null;
    const sample = tex => terrainTexelTSL(tex, st, face);
    // Shading normal: the face normal bent by the block texture's normal map.
    // The geometric n still lifts rays and decides which faces the sun can
    // reach at all; only N.L sees the bumps.
    // terrain: { normal, specular, albedo, blocks }, from render.js.
    const lab = terrain ? labNormalTSL(sample(terrain.normal)).toVar() : null;
    // The face's exact frame, not a derivative one - see boxBumpedNormalTSL.
    const nS = lab ? surfaceBumpTSL(face, lab.xyz).toVar() : n;
    const texAO = lab ? lab.w : float(1);
    // albedo: a function of the face uv (main.js), so it is read here too.
    const base = albedo && st ? albedo(st, face).toVar() : vec3(1, 1, 1);
    // The debug views that need no light; the light-term views below show the
    // term raw, not times the albedo.
    const shown = view && surfaceViewTSL(view, { p, n, nS, base, terrain,
      specular: terrain && (() => sample(terrain.specular).rgb) }, cascades);
    if (shown) return shown;
    // Traced AO times the texture's own. The hemisphere is the geometric face,
    // not the bumped normal - the rays test real geometry.
    const occ = occlusion(p, n, n, NO_SKIP, texAO, false, nS);
    const ao = occ.ao;
    if (view === 'ao') return vec3(ao, ao, ao);
    if (view === 'gi') return occ.gi || vec3(0, 0, 0);
    // The bounce adds under everything else - it is light, like the others.
    const plusGI = col => (occ.gi ? col.add(occ.gi) : col);
    // The sky ambient, along the bent normal - bent further by the normal
    // map, so the bumps catch the sky as they catch the sun.
    const amb = ambientLight(normalize(occ.bent.add(nS).sub(n)), ao, occ.sky).toVar();

    // The mirror light alone - not in the GI view, as it never goes through the LPV.
    if (view === 'mirror') return mirrorTrace ? mirrorLight(p, n, nS, n) : vec3(0, 0, 0);
    // Light thrown here by the mirrors - the iron floor's glare on the wall,
    // from the sun and every lamp. Added under the sun-on and sun-off paths.
    const mirrored = mirrorTrace ? mirrorLight(p, n, nS, n).toVar() : null;
    const plusMirror = col => (mirrored ? col.add(mirrored) : col);
    if (view === 'indirect') return plusGI(plusMirror(amb));

    // The surface's specular response, when it has one. V from the locked
    // position, so the whole texel shares one view direction.
    let surf = null;
    if (spec) {
      const d = decodeSpecularTSL(sample(terrain.specular), base);
      const v = normalize(cameraPosition.sub(p)).toVar();
      surf = { ...d, v, brdf: specularTSL(nS, v, d.alpha, d.f0) };
    }

    // light: the diffuse light (x albedo). direct: the highlights, summed.
    const finish = (light, direct) => {
      // This texel's diffuse light, for reflections that land on it next
      // frame (the hit lighting cache). Box faces only - a slope's plane is
      // never what a ray hits (the field holds its steps).
      if (hitCache && writeCache) {
        const write = () => hitCacheWriteTSL(hitCache, p, n, base.mul(light));
        if (face.on) If(face.on.not(), write); else write();
      }
      if (!surf) return view ? vec3(0, 0, 0) : base.mul(light);
      const highlights = min(direct, vec3(SPEC_MAX, SPEC_MAX, SPEC_MAX));
      if (view === 'highlights') return highlights;
      const nv = max(nS.dot(surf.v), float(1e-4));
      const F = fresnelRoughTSL(surf.f0, nv, surf.alpha).toVar();
      // What a rough surface reflects: the ambient light around it, the same
      // estimate the diffuse ambient uses. Smooth surfaces replace it with the
      // ray's answer across the rough band.
      const env = amb.toVar();
      if (occ.gi) env.addAssign(occ.gi);
      if (reflectTrace) {
        const w = float(1).sub(smoothstep(float(REFLECT_ROUGH_START),
                                          float(REFLECT_ROUGH_END), surf.alpha)).toVar();
        If(w.greaterThan(float(0)), () => {
          // The mirror direction, jittered per texel by the roughness - a
          // glossy surface reflects as a stable grain rather than a blur.
          // Half a texel inside, so the along-normal floor is not taken on
          // the face plane itself, where it could round either way.
          const wt = p.sub(n.mul(float(VOXEL_METRES * 0.5))).div(float(VOXEL_METRES))
                      .floor().toVar();
          const jit = vec3(texelHash(wt, 0), texelHash(wt, 1), texelHash(wt, 2))
                        .sub(float(0.5)).mul(surf.alpha.mul(float(2)));
          const r0 = reflect(surf.v.negate(), nS).toVar();
          const rj = normalize(r0.add(jit)).toVar();
          const dir = select(rj.dot(n).greaterThan(float(0.02)), rj, r0).toVar();
          const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();
          // Through the blur band the reach shrinks to a quarter: a rough
          // reflection of something far is a smear of the ambient anyway.
          const reach = reflectMax.mul(float(1).sub(roughReach.mul(float(1).sub(w))));
          env.assign(mix(env, reflectedTSL(origin, dir, reach), w));
        });
      }
      if (view === 'reflection') return F.mul(env).mul(texAO);
      const kd = vec3(1, 1, 1).sub(F).mul(float(1).sub(surf.metal));
      return base.mul(light).mul(kd).add(highlights).add(F.mul(env).mul(texAO));
    };
    const noSpec = vec3(0, 0, 0);
    // The light list: diffuse, and the highlights from the same visibility.
    const points = () => {
      if (!surf) return { diffuse: pointLight(p, n, NO_SKIP, nS), spec: noSpec };
      return pointLight.withSpecular(p, n, NO_SKIP, nS, surf.brdf);
    };

    // Sun off: no march, no shading term, just the ambient floor for anything a
    // dynamic light does not reach.
    if (!sunOn) {
      const dark = plusMirror(amb);
      if (view === 'shadow' || (view === 'direct' && !pointLight)) return vec3(0, 0, 0);
      if (!pointLight) return finish(plusGI(dark), noSpec);
      const pl = points();
      if (view === 'direct') return pl.diffuse;
      return finish(plusGI(dark.add(pl.diffuse)), pl.spec);
    }
    // Faces turned from the sun get N.L = 0, so the march result is multiplied
    // away - skip it. Not in the shadow view, which shows the raw term everywhere.
    let visible;
    if (view === 'shadow') visible = sunShadow(p, n, NO_SKIP);
    else {
      visible = sunVisZero();
      If(n.dot(sun.dir).greaterThan(float(0)), () => {
        visible.assign(sunShadow(p, n, NO_SKIP));
      });
    }
    // The shadow view writes the raw visibility term. Ambient and N.L both
    // compress the shadowed range toward the middle, so acne that is plain
    // here is invisible in the shaded result.
    if (view === 'shadow') return asColour(visible);
    const sunTerm = sunColourUniform.mul(sunDirect(visible, nS, shadeMode));
    const sunLit = plusMirror(amb.add(sunTerm));
    const sunSpec = surf
      ? surf.brdf(sun.dir).mul(visible).mul(sunLight()).mul(sunColourUniform) : noSpec;
    if (view === 'direct' && !pointLight) return sunTerm;
    if (!pointLight) return finish(plusGI(sunLit), sunSpec);
    const pl = points();
    if (view === 'direct') return sunTerm.add(pl.diffuse);
    return finish(plusGI(sunLit.add(pl.diffuse)), sunSpec.add(pl.spec));
  })();
  const node = makeNode(false);
  const nodeCaching = hitCache ? makeNode(true) : node;

  // Split shading's kernels (createSplitShading): the node above, a feature
  // per kernel, over the low-res samples. The same functions, the same
  // inputs - the surface is rebuilt from the sample's position and face the
  // way a reflection hit rebuilds one (hitSurfaceTSL), which for a box face
  // is the drawn face's own texel. Not under the debug views, but the bounce ones.
  let splitKernels = null;
  // Glass splits too, unless a glass debug view is on (glassNode keeps those).
  const splitGlass = !!(split && glass && reflectTrace && !glassView);
  if (split && terrain && (!view || (texelGI && SPLIT_VIEWS.has(view)))) {
    const faceNormal = id => {
      const f = floor(id.sub(float(TEXEL_ID_OFFSET)).div(float(2097152)));
      return select(f.lessThan(float(0.5)), vec3(1, 0, 0), select(f.lessThan(float(1.5)), vec3(-1, 0, 0),
             select(f.lessThan(float(2.5)), vec3(0, 1, 0), select(f.lessThan(float(3.5)), vec3(0, -1, 0),
             select(f.lessThan(float(4.5)), vec3(0, 0, 1), vec3(0, 0, -1))))));
    };
    // A sample's geometry: its locked position and normal - a box face's
    // from its id, a slope plane's (face 6) from its block's code.
    const geoAt = s => {
      if (splitLegacy) {
        // A08//00a: box faces only (slopes and glowing cubes are misses).
        return { p: s.xyz.toVar(), n: faceNormal(abs(s.w)).toVar(), slope: null, legacy: true };
      }
      const p = s.xyz.toVar();
      const slope = floor(abs(s.w).sub(float(TEXEL_ID_OFFSET)).div(float(2097152))).greaterThan(float(5.5)).toVar();
      const vol = blockVolumeTSL(terrain, round(p.div(float(BLOCK_METRES)))).toVar();
      const sl = slopeFromCodeTSL(vol.w);
      const n = select(slope, sl.n, faceNormal(abs(s.w))).toVar();
      return { p, n, slope, sl, vol };
    };
    // The surface there - a box face as a reflection hit rebuilds one, a
    // slope plane as the terrain draws it (slopePlaneTSL).
    const surfaceAt = geo => {
      const { p, n } = geo;
      if (geo.legacy) {
        const hs = hitSurfaceTSL(terrain, p, n, null);
        const S = { base: hs.albedo.toVar(), nS: hs.n.toVar(), texAO: hs.ao.toVar(),
                    emissive: float(0), kd: vec3(1, 1, 1) };
        if (spec) {
          S.d = decodeSpecularTSL(texture(terrain.specular, hs.st).depth(int(hs.layer)).level(int(0)), S.base);
          S.v = normalize(split.viewPos.sub(p)).toVar();
          S.brdf = specularTSL(S.nS, S.v, S.d.alpha, S.d.f0);
          S.F = fresnelRoughTSL(S.d.f0, max(S.nS.dot(S.v), float(1e-4)), S.d.alpha).toVar();
          S.kd = vec3(1, 1, 1).sub(S.F).mul(float(1).sub(S.d.metal)).toVar();
        }
        return S;
      }
      const S = { base: vec3(0, 0, 0).toVar(), nS: n.toVar(), texAO: float(1).toVar(),
                  emissive: float(0).toVar(), kd: vec3(1, 1, 1) };
      const layer = int(0).toVar(), st = vec2(0, 0).toVar();
      If(geo.slope, () => {
        const side = int(max(geo.vol.x.sub(float(1)), float(0))).toVar();
        const row = slopeLayerTable.element(side);
        const first1 = select(geo.sl.kind.lessThan(float(1.5)), row.x, row.y);
        const sp = slopePlaneTSL(geo.sl, p, p, side, first1);
        layer.assign(sp.layer); st.assign(sp.st);
        S.base.assign(mix(texture(terrain.albedo, sp.st).depth(int(sp.layer)).level(int(0)).rgb,
                          vec3(1, 1, 1), terrainWhite));
        const lab = labNormalTSL(texture(terrain.normal, sp.st).depth(int(sp.layer)).level(int(0))).toVar();
        S.nS.assign(surfaceBumpTSL(sp, lab.xyz));
        S.texAO.assign(lab.w);
      }).Else(() => {
        const hs = hitSurfaceTSL(terrain, p, n, null);
        layer.assign(hs.layer); st.assign(hs.st);
        S.base.assign(hs.albedo); S.nS.assign(hs.n); S.texAO.assign(hs.ao); S.emissive.assign(hs.emissive);
      });
      if (spec) {
        S.d = decodeSpecularTSL(texture(terrain.specular, st).depth(int(layer)).level(int(0)), S.base);
        S.v = normalize(split.viewPos.sub(p)).toVar();
        S.brdf = specularTSL(S.nS, S.v, S.d.alpha, S.d.f0);
        S.F = fresnelRoughTSL(S.d.f0, max(S.nS.dot(S.v), float(1e-4)), S.d.alpha).toVar();
        S.kd = vec3(1, 1, 1).sub(S.F).mul(float(1).sub(S.d.metal)).toVar();
      }
      return S;
    };
    // body(idx, s, geo): a terrain sample only; glass and empty ones are the
    // base kernel's alone. Threads: this frame's low-res samples, then the
    // misses; idx is where each one's buffers are (misses from lowCap on).
    const kernel = (name, body, base = null) => Fn(() => {
      cacheMode = 'none';
      hitRegister = texelGI ? registerPatch : null;
      if (splitLegacy) {
        // A08//00a exactly: low-res samples only, no miss list.
        const idx = int(instanceIndex).toVar();
        If(idx.lessThan(split.count), () => {
          const s = textureLoad(split.rt, ivec2(idx.mod(split.width), idx.div(split.width))).toVar();
          const terrainSample = () => body(idx, s, geoAt(s));
          if (base) base(idx, s, terrainSample);
          else If(s.w.greaterThan(float(1.5)), terrainSample);
        });
        return;
      }
      const t = int(instanceIndex).toVar();
      const misses = min(int(split.missCount.ro.element(int(0))), int(split.missCap));
      const low = t.lessThan(split.count);
      const slot = t.sub(split.count).toVar();
      If(low.or(slot.lessThan(misses)), () => {
        const idx = select(low, t, int(split.lowCap).add(slot)).toVar();
        const s = select(low,
          textureLoad(split.rt, ivec2(t.mod(split.width), t.div(split.width))),
          split.missSamples.ro.element(max(slot, int(0)))).toVar();
        const terrainSample = () => body(idx, s, geoAt(s));
        if (base) base(idx, s, terrainSample);
        else If(s.w.greaterThan(float(1.5)), terrainSample);
      });
    })().compute(split.capacity).setName(name);
    const addResult = (idx, c) => {
      const r = split.result.rw.element(idx).toVar();
      split.result.rw.element(idx).assign(vec4(r.xyz.add(c), r.w));
    };
    // The packed ambient / highlights (createSplitShading: light).
    const readLight = idx => {
      if (!split.half) {
        return { env: split.env.rw.element(idx).xyz.toVar(), spec: split.spec.rw.element(idx).xyz.toVar() };
      }
      const w = split.light.rw.element(idx).toVar();
      const a = unpackHalf2x16(w.x), b = unpackHalf2x16(w.y), c = unpackHalf2x16(w.z);
      return { env: vec3(a.x, a.y, b.x), spec: vec3(b.y, c.x, c.y) };
    };
    const writeLight = (idx, env, spec) => {
      if (!split.half) {
        split.env.rw.element(idx).assign(vec4(env, float(0)));
        split.spec.rw.element(idx).assign(vec4(spec, float(0)));
        return;
      }
      split.light.rw.element(idx).assign(uvec4(packHalf2x16(vec2(env.x, env.y)),
        packHalf2x16(vec2(env.z, spec.x)), packHalf2x16(vec2(spec.y, spec.z)), uint(0)));
    };
    const addSpec = (idx, c) => {
      const l = readLight(idx);
      writeLight(idx, l.env, l.spec.add(c));
    };
    // Texel GI: list this sample's patch for the patch kernel (gpu/texelgi.js
    // createPatchRegisterTSL) - p locked on a box face, n its axis normal.
    const registerPatch = texelGI ? createPatchRegisterTSL(texelGI, giLevel) : null;
    splitKernels = [];
    // The room meter's tap (gpu/roommeter.js): each kernel's diffuse light.
    const roomTap = split.room ? split.room.tap(terrain) : null;
    // Glowing cubes are a kernel case unless legacy (then they are misses).
    const unlessGlowing = (S, fn) => (splitLegacy ? fn() : If(S.emissive.lessThan(float(0.5)), fn));
    splitKernels.push(kernel('splitBase', (idx, s, geo) => {
      const { p, n } = geo;
      const S = surfaceAt(geo);
      If(S.emissive.greaterThan(float(0.5)), () => {
        // A glowing model cube: its own texel, unlit (terrainEmissiveTSL).
        split.result.rw.element(idx).assign(vec4(S.base, s.w));
        writeLight(idx, vec3(0, 0, 0), vec3(0, 0, 0));
      }).Else(() => {
      // Box faces only: a slope's plane is never what a ray lands on.
      if (texelGI) {
        if (geo.slope) If(geo.slope.not(), () => registerPatch(p, n));
        else registerPatch(p, n);
      }
      const occ = occlusion(p, n, n, NO_SKIP, S.texAO, false, S.nS);
      const amb = ambientLight(normalize(occ.bent.add(S.nS).sub(n)), occ.ao, occ.sky);
      if (view) {
        // The bounce views: the split kernels stay on for them, so it is traced.
        const bounce = occ.gi || vec3(0, 0, 0);
        split.result.rw.element(idx).assign(vec4(view === 'gi' ? bounce
          : amb.add(bounce).add(mirrorTrace ? mirrorLight(p, n, S.nS, n) : vec3(0, 0, 0)), s.w));
        writeLight(idx, vec3(0, 0, 0), vec3(0, 0, 0));
        return;
      }
      const env = (occ.gi ? amb.add(occ.gi) : amb).toVar();
      const light = env.toVar();
      const sunSpec = vec3(0, 0, 0).toVar();
      if (sunOn) {
        const vis = sunVisZero();
        If(n.dot(sun.dir).greaterThan(float(0)), () => { vis.assign(sunShadow(p, n, NO_SKIP)); });
        light.addAssign(sunColourUniform.mul(sunDirect(vis, S.nS, shadeMode)));
        if (spec) sunSpec.assign(S.brdf(sun.dir).mul(vis).mul(sunLight()).mul(sunColourUniform));
      }
      split.result.rw.element(idx).assign(vec4(S.base.mul(light).mul(S.kd), s.w));
      writeLight(idx, env, sunSpec);
      if (roomTap) roomTap(idx, p, n, light, true);
      });
    }, (idx, s, terrainSample) => {
      // Glass (negative id): its own kernels below, or - without them - its
      // own low-res shading, passed through.
      If(s.w.lessThan(float(-0.5)), () => {
        if (!splitGlass) split.result.rw.element(idx).assign(vec4(s.xyz, s.w.negate()));
      }).ElseIf(s.w.greaterThan(float(1.5)), terrainSample).Else(() => {
        split.result.rw.element(idx).assign(vec4(0, 0, 0, 0));
      });
    }));
    if (pointLight && !view) {
      splitKernels.push(kernel('splitPoints', (idx, s, geo) => {
        const { p, n } = geo;
        const S = surfaceAt(geo);
        unlessGlowing(S, () => {
        const pl = spec ? pointLight.withSpecular(p, n, NO_SKIP, S.nS, S.brdf)
                        : { diffuse: pointLight(p, n, NO_SKIP, S.nS).toVar(), spec: null };
        addResult(idx, spec ? S.base.mul(pl.diffuse).mul(S.kd) : S.base.mul(pl.diffuse));
        if (pl.spec) addSpec(idx, pl.spec);
        if (roomTap) roomTap(idx, p, n, pl.diffuse);
        });
      }));
    }
    if (mirrorTrace && !view) {
      splitKernels.push(kernel('splitMirror', (idx, s, geo) => {
        const { p, n } = geo;
        const S = surfaceAt(geo);
        unlessGlowing(S, () => {
          const m = mirrorLight(p, n, S.nS, n).toVar();
          addResult(idx, S.base.mul(m).mul(S.kd));
          if (roomTap) roomTap(idx, p, n, m);
        });
      }));
    }
    if (spec && !view) {
      splitKernels.push(kernel('splitFinal', (idx, s, geo) => {
        const { p, n } = geo;
        const S = surfaceAt(geo);
        unlessGlowing(S, () => {
        const stored = readLight(idx);
        const env = stored.env.toVar();
        if (reflectTrace) {
          const w = float(1).sub(smoothstep(float(REFLECT_ROUGH_START),
                                            float(REFLECT_ROUGH_END), S.d.alpha)).toVar();
          If(w.greaterThan(float(0)), () => {
            const wt = p.sub(n.mul(float(VOXEL_METRES * 0.5))).div(float(VOXEL_METRES)).floor().toVar();
            const jit = vec3(texelHash(wt, 0), texelHash(wt, 1), texelHash(wt, 2))
                          .sub(float(0.5)).mul(S.d.alpha.mul(float(2)));
            const r0 = reflect(S.v.negate(), S.nS).toVar();
            const rj = normalize(r0.add(jit)).toVar();
            const dir = select(rj.dot(n).greaterThan(float(0.02)), rj, r0).toVar();
            const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();
            const reach = reflectMax.mul(float(1).sub(roughReach.mul(float(1).sub(w))));
            if (splitLegacy) {
              env.assign(mix(env, reflectedTSL(origin, dir, reach), w));
              return;
            }
            // shadeTraced, spelled out so the reflection-hit debug views (bxb.view) can show its parts.
            const hit = reflectTrace(origin, dir, reach).toVar();
            const rec = traceRecord(origin, dir, hit);
            const [o] = lightBatch([hitRequest(rec)]);
            const refl = composeHit(rec, o.vis, o.diff).toVar();
            // 1 the hit's own colour, unlit; 2 its distance (white = 16 m, red = a miss).
            If(splitDebug.equal(int(1)), () => { refl.assign(select(rec.met.greaterThan(float(0.5)), rec.alb, rec.col)); });
            If(splitDebug.equal(int(2)), () => {
              refl.assign(select(hit.w.lessThan(float(0)), vec3(1, 0, 0), vec3(hit.w.div(float(16)))));
            });
            // 4 the hit's normal; 5 its block's code (magenta: air - layer 0).
            If(splitDebug.greaterThanEqual(int(4)), () => {
              const ht = hitTexelTSL(terrain, origin.add(dir.mul(max(hit.w, float(0)))), hit.xyz, dir);
              const nh = ht.n, hpL = ht.p.toVar();
              const code = blockVolumeTSL(terrain, boxFaceTSL(hpL, nh).block).toVar();
              refl.assign(select(splitDebug.equal(int(4)), nh.mul(float(0.5)).add(float(0.5)),
                select(hit.w.lessThan(float(0)), vec3(0, 0, 0),
                select(code.x.lessThan(float(0.5)), vec3(1, 0, 1), vec3(code.x.div(float(16)))))));
              // Raw, not through the Fresnel blend.
              const r = split.result.rw.element(idx).toVar();
              split.result.rw.element(idx).assign(vec4(refl, r.w));
            });
            env.assign(mix(env, refl, select(splitDebug.equal(int(3)), float(0), w)));
          });
        }
        const sp = stored.spec;
        If(splitDebug.lessThan(int(4)), () => {
          addResult(idx, min(sp, vec3(SPEC_MAX, SPEC_MAX, SPEC_MAX)).add(S.F.mul(env).mul(S.texAO)));
        });
        });
      }));
    }
    if (splitGlass) {
      // glassNode, a part per kernel, over glass samples (negative id):
      // highlights -> spec, reflection -> env, the view through -> result,
      // then the Fresnel blend. The surface as glassNode builds it.
      const glassAt = (p, n) => {
        const hs = hitSurfaceTSL(terrain, p, n, null);
        const G = { nS: hs.n.toVar(), tint: hs.albedo.toVar() };
        G.d = decodeSpecularTSL(texture(terrain.specular, hs.st).depth(int(hs.layer)).level(int(0)), G.tint);
        G.v = normalize(split.viewPos.sub(p)).toVar();
        G.F = fresnelRoughTSL(G.d.f0, max(G.nS.dot(G.v), float(1e-4)), G.d.alpha).toVar();
        G.smooth = float(1).sub(smoothstep(float(REFLECT_ROUGH_START),
                                           float(REFLECT_ROUGH_END), G.d.alpha)).toVar();
        return G;
      };
      const glassKernel = (name, body) => kernel(name, body, (idx, s, run) => {
        If(s.w.lessThan(float(-1.5)), run);
      });
      splitKernels.push(glassKernel('splitGlassSurface', (idx, s, geo) => {
        const { p, n } = geo;
        const G = glassAt(p, n);
        const [o] = lightBatch([{ active: float(1), P: p, N: n, lift: n, skip: NO_SKIP, nS: G.nS,
                                  brdf: { v: G.v, alpha: G.d.alpha, f0: G.d.f0 } }]);
        const direct = vec3(0, 0, 0).toVar();
        if (sunOn) {
          const brdf = specularTSL(G.nS, G.v, G.d.alpha, G.d.f0);
          If(n.dot(sun.dir).greaterThan(float(0)), () => {
            direct.addAssign(brdf(sun.dir).mul(o.vis).mul(sunLight()).mul(sunColourUniform));
          });
        }
        if (pointLight) direct.addAssign(o.spec);
        writeLight(idx, readLight(idx).env, direct);
      }));
      splitKernels.push(glassKernel('splitGlassReflect', (idx, s, geo) => {
        const { p, n } = geo;
        const G = glassAt(p, n);
        const r0 = reflect(G.v.negate(), G.nS).toVar();
        const rdir = select(r0.dot(n).greaterThan(float(0.02)), r0, reflect(G.v.negate(), n)).toVar();
        const reflOrigin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();
        const reflRec = newHitRecord();
        If(G.smooth.greaterThan(float(0)), () => {
          traceRecord(reflOrigin, rdir, reflectTrace(reflOrigin, rdir, reflectMax).toVar(), null, reflRec);
        });
        const [o] = lightBatch([hitRequest(reflRec)]);
        const reflected = ambientAt(reflOrigin, G.nS).toVar();
        If(G.smooth.greaterThan(float(0)), () => {
          reflected.assign(mix(reflected, composeHit(reflRec, o.vis, o.diff), G.smooth));
        });
        writeLight(idx, reflected, readLight(idx).spec);
      }));
      splitKernels.push(glassKernel('splitGlassThrough', (idx, s, geo) => {
        const { p, n } = geo;
        const G = glassAt(p, n);
        const sf = sqrt(clamp(G.d.f0.x, float(0), float(0.5)));
        const ior = float(1).add(sf).div(float(1).sub(sf)).toVar();
        const iors = glassDispersion > 0
          ? [ior.mul(float(1 - glassDispersion)), ior, ior.mul(float(1 + glassDispersion))] : [ior];
        const throughs = iors.map(x => glassThroughRec(p, n, G.nS, G.v, x, G.d.f0.x, G.d.alpha, G.tint, G.smooth));
        const traced = throughs.filter(t => !t.debug);
        const lit = traced.length ? lightBatch(traced.map(t => hitRequest(t.rec))) : [];
        const colours = throughs.map(t => glassThroughColour(t, t.debug ? null : lit[traced.indexOf(t)]));
        const seen = glassDispersion > 0 ? vec3(colours[0].x, colours[1].y, colours[2].z) : colours[0];
        split.result.rw.element(idx).assign(vec4(seen, s.w.negate()));
      }));
      splitKernels.push(glassKernel('splitGlassCompose', (idx, s, geo) => {
        const { p, n } = geo;
        const G = glassAt(p, n);
        const r = split.result.rw.element(idx).toVar();
        const stored = readLight(idx);
        const refl = stored.env, direct = stored.spec;
        split.result.rw.element(idx).assign(vec4(
          G.F.mul(refl).add(vec3(1, 1, 1).sub(G.F).mul(r.xyz))
            .add(min(direct, vec3(SPEC_MAX, SPEC_MAX, SPEC_MAX))), r.w));
      }));
    }
  }

  // Sprites: the same lights, shaded at the sprite's own texel centres and
  // lifted straight up. Up rather than the quad's normal: the quad turns with the
  // camera, and a shadow that moved with the camera would crawl. Built per map,
  // because the texel grid is that texture's.
  //
  // `drawn` gates the marches: a fragment alphaTest is about to discard still
  // ran both of them, and the trees' crossed planes are mostly transparent.
  // front: which side is being shaded - the rasterised side (frontFacing) by
  // default; the sprite atlas pass says which explicitly. size: the art's size
  // in texels, when the caller has it as a uniform (the atlas pass) - else read
  // off the texture.
  const spriteLight = (map, drawn, normalTex = null, front = frontFacing, size = null) => Fn(() => {
    const n = vec3(0, 1, 0);
    // Before the branch - the snap takes screen derivatives.
    const p = spriteTexelLockTSL(positionWorld, uv(),
                                 size || vec2(textureSize(texture(map), int(0)))).toVar();
    const skip = modelWorldMatrix.mul(vec4(0, 0, 0, 1)).xz.toVar();
    const shade = spriteShadeNormalTSL(map, normalTex, front);
    const nS = shade.n.toVar();
    const texAO = shade.ao.toVar();
    const out = vec3(0, 0, 0).toVar();
    If(drawn, () => {
      const shown = view && surfaceViewTSL(view, { p, n, nS, base: vec3(1, 1, 1), specular: null, terrain }, cascades);
      if (shown) { out.assign(shown); return; }
      if (view === 'reflection' || view === 'highlights') return;   // a sprite has no specular
      // Over the shading normal's hemisphere - a billboard has no geometric
      // face in the field, so the way its texels face is the best there is.
      // Texel GI: a sprite is in no face's patches, so each of its texels
      // reads the ground's straight under it - at the sprite's feet, the top
      // of the block it stands on - seen by the texel's own normal.
      const feet = modelWorldMatrix.mul(vec4(0, 0, 0, 1)).y;
      const occ = occlusion(p, nS, n, skip, texAO, true, nS,
                            texelGI ? { p: vec3(p.x, feet, p.z), n: vec3(0, 1, 0) } : null);
      const ao = occ.ao;
      if (view === 'ao') { out.assign(vec3(ao, ao, ao)); return; }
      if (view === 'gi') { if (occ.gi) out.assign(occ.gi); return; }
      const plusGI = col => (occ.gi ? col.add(occ.gi) : col);
      if (view === 'mirror') { if (mirrorTrace) out.assign(mirrorLight(p, n, nS, n, skip)); return; }
      // A sprite by a mirror catches its glare too, sun or lamp. Lifted
      // straight up, as the sprite's own rays are.
      const mirrored = mirrorTrace ? mirrorLight(p, n, nS, n, skip) : null;
      const amb0 = ambientLight(occ.bent, ao, occ.sky);
      const amb = mirrored ? amb0.add(mirrored) : amb0;
      if (view === 'indirect') { out.assign(plusGI(amb)); return; }
      if (!sunOn) {
        if (view === 'direct') { if (pointLight) out.assign(pointLight(p, n, skip, nS)); return; }
        if (view !== 'shadow') out.assign(plusGI(pointLight ? amb.add(pointLight(p, n, skip, nS)) : amb));
        return;
      }
      const visible = sunShadow(p, n, skip);
      if (view === 'shadow') { out.assign(asColour(visible)); return; }
      // Plain Lambert on the normal-mapped normal: a sprite is lit by which way
      // each of its texels faces, so turning the camera round to the sun's far
      // side shows its dark side.
      const sunTerm = sunColourUniform.mul(sunDirect(visible, nS, 'lambert'));
      if (view === 'direct') { out.assign(pointLight ? sunTerm.add(pointLight(p, n, skip, nS)) : sunTerm); return; }
      const lit = amb.add(sunTerm);
      out.assign(plusGI(pointLight ? lit.add(pointLight(p, n, skip, nS)) : lit));
    });
    return out;
  })();

  // Under a debug view glass is shaded as any block is: the view shows that
  // term at its surface, not glass's own look. glassView is glass's own views.
  const terrainView = !!view;
  // Texel GI's patch kernels (gpu/texelgi.js createTexelGIKernels), lit as
  // the shading lights a reflection hit.
  const giKernels = texelGI && splitKernels ? createTexelGIKernels({
    texelGI, cascades, terrain, trace: reflectTrace, glass: glassT, bias, sky, ambient, shadeMode, sun,
    sunOn, sunShadow, sunVisZero, sunDirect, skyIrradiance, pointLight, mirrorTrace, giRead,
    // The near band and V_p need the cones' AO.
    cones: cones && aoOn ? cones : null, nearReach: cones && aoOn ? aoDistance : null }) : null;

  return { node, nodeCaching, splitKernels, giKernels, splitGlass: !!splitKernels && splitGlass,
           glassNode: terrainView ? node : glassNode, sun, cascades, lights,
           // A raw view: the sprite draws the term itself, not times its art (gpu/sprite.js).
           spriteLight: Object.assign(spriteLight, { raw: RAW_VIEWS.has(view) }),
           mirrorCache: (mirrorLight && mirrorLight.cacheKernel) || null,
           maxDistUniform: maxDist, ambientUniform: ambient, revealedLight };
}

// How far through the rough band a reflection's reach shrinks: 0.75 = to a
// quarter; 0 = full reach everywhere (bxb.roughreach()). A uniform: no rebuild.
export const roughReach = uniform(float(0.75));

// Split shading's reflection-hit views (bxb.view, render/features/debugview.js):
// 0 off, 1 hit colour unlit, 2 hit distance, 3 no reflection. A uniform: no rebuild.
export const splitDebug = uniform(int(0));
