import * as THREE from 'three/webgpu';
import { tableGroup } from './uploads.js';
import {
  Fn, If, Loop, Break, float, vec2, vec3, vec4, int, uvec2, uvec3, uniform, instanceIndex,
  packHalf2x16, unpackHalf2x16,
  storage, texture3D, textureStore, max, min, select, normalize, round, uniformArray,
  length, pow, mix, exp
} from 'three/tsl';
import { HIT_EPS, MIN_STEP, cascadeVoxelMetres, VOXEL_METRES } from '../core/field.js';
import { decodeFieldTSL, ringUV, SURFACE_BIAS_VOXELS, SHADOW_FADE_START, EDGE_FADE_VOXELS } from './field.js';
import { glassRayTransmitTSL } from './glass.js';
import { createMirrorLightTSL } from './mirrorlight.js';
import { createPointLightsTSL } from './pointlights.js';
import { smokeShared, smokeSampleTSL, smokeSunTransmitTSL } from './smokeshadow.js';
import { sunColourUniform, skyShTSL } from './sky.js';
import { createSunUniforms, sunRoofTSL } from './sun.js';
import { sunMapFarTSL } from './sunmap.js';
import { createConeTraceSunTSL } from './trace.js';
import { volumeAtTSL, skyOpenTSL } from './volume.js';
import { BLOCK_METRES } from '../core/world.js';
import { GI_LAYERS } from '../core/models.js';
// One entry per material, then per model (models.js giIndex).
const GI_TABLE = Array.from({ length: GI_LAYERS });
import {
  LPV_DIM, LPV_CELL_VOXELS, FACE_DIRS, SOLID_THRESHOLD,
  DEFAULT_LPV_SPREAD, DEFAULT_LPV_SLICES, MAX_GI_SPRITES, GI_MAX_BOUNCES, DEFAULT_GI_BOUNCES,
  DEFAULT_LPV_SKY_SPREAD
} from '../core/lpv.js';

// --- The LPV on the GPU: solid, inject, propagate, resolve ---
//
// lpv.js has the why and the CPU reference; this is the kernels. Four storage
// buffers and one 3D texture:
//
//   solid   float   how much of each cell is geometry        every frame
//   E       vec4    injected light (rgb); w 1 if the cell is  one slice a frame
//                   air open to the sky (core/skytop.js)
//   W       vec4    what a cell's surfaces bounce (albedo)   one slice a frame
//   A, B    vec4    propagated light, ping-ponged            N iterations a frame
//                   - GI_MAX_BOUNCES orders of it, order k at k * CELLS;
//                   order 0's w the cell's sky visibility (lpv.js
//                   propagateSkyStep)
//   tex     RGBA16F what the shading pass samples, trilinear  every frame
//                   - rgb the light, a the sky visibility
//
// The texture is written by a resolve pass that DILATES into solid cells: a
// solid cell takes the mean of its air neighbours. The surfaces the shading pass
// samples near are exactly where trilinear filtering would otherwise blend in
// the zeros inside the ground and darken every contact.

const NO_SKIP = vec2(1e9, 1e9);

// instanceIndex to cell coordinates, as ints, in a volume D cells a side.
function cellOf(index, D) {
  const i = int(index).toVar();
  return { i, x: i.mod(int(D)), y: i.div(int(D)).mod(int(D)), z: i.div(int(D * D)) };
}

// The neighbour one step along (dx, dy, dz), or this cell when that is off the
// volume - the CPU reference's edge rule.
function neighbour(c, dx, dy, dz, D) {
  const nx = c.x.add(int(dx)), ny = c.y.add(int(dy)), nz = c.z.add(int(dz));
  const inside = nx.greaterThanEqual(int(0)).and(ny.greaterThanEqual(int(0)))
    .and(nz.greaterThanEqual(int(0))).and(nx.lessThan(int(D)))
    .and(ny.lessThan(int(D))).and(nz.lessThan(int(D)));
  const n = nx.add(ny.mul(int(D))).add(nz.mul(int(D * D)));
  return { inside, index: select(inside, n, c.i) };
}

// level: which cascade the volume lies over. At res 1, 24^3 cells of six of
// its voxels: C0 0.75 m cells over 18 m, C1 1.5 m over 36 m, C2 3 m over 72 m
// - each cascade is the same 144^3 at a doubled voxel size. lpv.js
// LPV_VOLUMES picks the levels and res (C1 and C2, at res 2).
// sky: the sky SH bindings (sky.js createSkyBindings), with ambientUniform its
// scale. blocks: render.js's block-material volume, so a probe hit bounces its
// own block's albedo. Either may be null, and the old behaviour stands.
// res: a block face's cells a side, halved (lpv.js LPV_RES_OPTIONS) - 1 is
// 0.75 m over C0, the original. The volume keeps its extent, so it is
// 24 * res cells a side and res^3 times the cells. orders: how many
// bounce orders its buffers hold (at most GI_MAX_BOUNCES; the bounces uniform
// must not exceed it).
// The propagated light (A, B) as half floats, packed two to a u32: half the
// bytes a cell for the propagation, which reads six neighbours an order and
// runs ~24 times a frame - the largest kernel. The resolved texture the
// shading reads was half float already. bxb.lpvhalf() flips it (rebuilds).
export const lpvStorage = { half: true };

export function createLPV({ cascades, lights, level = 0, biasUniform = null, res = 1,
                            orders: orderCount = GI_MAX_BOUNCES,
                            fadeStartUniform = null, edgeFadeUniform = null,
                            sky = null, ambientUniform = null, blocks = null,
                            glass = false, terrain = null, mirrors = null, skyAB = null }) {
  // skyAB: an int uniform - 1, the sky visibility (lpv.js propagateSkyStep)
  // also stands in for the injection's sky cone; 0, the cone as before. The
  // visibility itself is always propagated: it is a lane of math the kernels
  // run anyway, and the shading decides whether to use it.
  // terrain: the terrain textures, for the mirror light's texels (with mirrors,
  // the mirror bindings) - see the injection below. Either null: no mirror bounce.
  const cascade = cascades[level];
  const D = LPV_DIM * res, CELLS = D * D * D;
  const cellMetres = LPV_CELL_VOXELS * cascadeVoxelMetres(level) / res;
  // ONE shader for both volumes: nothing that differs between C0 and C1 is
  // written into the code. The cell size is a uniform, not a literal; and the
  // volume's own cascade is read through uniforms and a texture binding of its
  // own (c0), not the cascade's - three merges texture bindings by texture, so
  // sampling cascade[level] directly landed on whichever of the three cascade
  // bindings the sun trace declares, a different one per volume. Firefox
  // compiles every pipeline from scratch; this inject is ~0.8 s of it.
  // Same values: the uniforms follow the cascade's own (the vectors are the
  // cascade's objects; the numbers are re-read each frame), and the texture is
  // the same texture, bound twice.
  const cellU = uniform(float(cellMetres));
  // A sixth of a cell - a voxel at res 1: the solid kernel's sample spacing.
  const voxU = uniform(float(cellMetres / LPV_CELL_VOXELS));
  const blockShareU = uniform(float(cellMetres / BLOCK_METRES));
  // A fresh node per sample (sample() would clone away the override), all
  // sharing the one binding by their hash.
  const ownField = uvw => {
    const t = texture3D(cascade.tex, uvw);
    t.getUniformHash = () => 'bxbLpvOwnField';
    return t;
  };
  const c0 = {
    tex: cascade.tex,
    origin: uniform(cascade.origin.value),
    ring: uniform(cascade.ring.value),
    voxel: uniform(float(cascade.voxel.value)).onFrameUpdate(() => cascade.voxel.value),
    range: uniform(float(cascade.range.value)).onFrameUpdate(() => cascade.range.value)
  };
  const buf = (itemSize) => new THREE.StorageBufferAttribute(CELLS, itemSize);
  const ORDERS = CELLS * orderCount;
  const solidBuf = buf(1), eBuf = buf(4), wBuf = buf(4);
  // Per cell, what the cloud spells' smoke in it lets through: exp(-sigma x
  // cell) (gpu/smokeshadow.js) - 1 with no smoke. The bounce and the sky
  // visibility are both cut by it in propagate.
  const smokeBuf = buf(1);
  const smokeK = storage(smokeBuf, 'float', CELLS).setName(`bxbLpvSmoke`);
  const half = lpvStorage.half;
  const lightBuf = () => (half ? new THREE.StorageBufferAttribute(new Uint32Array(ORDERS * 2), 2)
                                : new THREE.StorageBufferAttribute(ORDERS, 4));
  const aBuf = lightBuf(), bBuf = lightBuf();
  // Named: an unnamed buffer is NodeBuffer_<global node id> in the WGSL, and
  // that id depends on how many nodes existed before - so the shader text, and
  // with it the browser's pipeline cache key, changed from run to run.
  // The same names in both volumes: a kernel whose code is otherwise the same
  // for C0 and C1 (propagate, resolve, clear, shift, copy) is then the same
  // text, and three compiles one pipeline for both - Firefox compiles every
  // pipeline from scratch.
  const solid = storage(solidBuf, 'float', CELLS).setName(`bxbLpvSolid`);
  const E = storage(eBuf, 'vec4', CELLS).setName(`bxbLpvE`);
  // Per cell, what its light is multiplied by: glass tints it (lpv.js
  // glassCellTransmit). Only in a world with glass.
  const tBuf = glass ? buf(4) : null;
  const T = glass ? storage(tBuf, 'vec4', CELLS).setName(`bxbLpvT`) : null;
  const W = storage(wBuf, 'vec4', CELLS).setName(`bxbLpvW`);
  const A = storage(aBuf, half ? 'uvec2' : 'vec4', ORDERS).setName(`bxbLpvA`);
  const B = storage(bBuf, half ? 'uvec2' : 'vec4', ORDERS).setName(`bxbLpvB`);
  // Light at index i of A or B, as vec4 / its w alone, and a write.
  const light = (buf, i) => {
    if (!half) return buf.element(i);
    const v = buf.element(i).toVar();
    return vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y));
  };
  const lightW = (buf, i) => (half ? unpackHalf2x16(buf.element(i).y).y : buf.element(i).w);
  const setLight = (buf, i, v) => (half
    ? buf.element(i).assign(uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw)))
    : buf.element(i).assign(v));
  const noLight = () => (half ? uvec2(0, 0) : vec4(0));

  const tex = new THREE.Storage3DTexture(D, D, D);
  tex.type = THREE.HalfFloatType;
  tex.format = THREE.RGBAFormat;
  tex.generateMipmaps = false;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;

  // The fog's light in this volume's air (gpu/fog.js reads it): per cell, the
  // sun's transmittance to the cell centre (rgb - stained glass tints it), and
  // the light list's light there, shadowed, from every side. Written by the
  // fog kernel only while there is fog.
  const fogTex = (() => {
    const t = new THREE.Storage3DTexture(D, D, D);
    t.type = THREE.HalfFloatType;
    t.format = THREE.RGBAFormat;
    t.generateMipmaps = false;
    t.minFilter = t.magFilter = THREE.LinearFilter;
    t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
    return t;
  });
  const fogSunTex = fogTex(), fogLampTex = fogTex();

  const u = {
    // World-space min corner of the volume: C0's origin, tracked by the CPU.
    origin: uniform(new THREE.Vector3()),
    spread: uniform(float(DEFAULT_LPV_SPREAD)),
    albedo: uniform(new THREE.Vector3(0.3, 0.4, 0.2)),
    sunGain: uniform(float(1)),
    slice: uniform(int(0)),
    slices: uniform(int(DEFAULT_LPV_SLICES)),
    // How many orders of bounce light (1..GI_MAX_BOUNCES): 1 is the injection
    // alone, as it always was.
    bounces: uniform(int(Math.min(DEFAULT_GI_BOUNCES, orderCount))),
    shift: uniform(new THREE.Vector3()),
    // Per material layer: the DIFFUSE albedo it bounces (lpv.js diffuseAlbedo).
    layerAlbedo: uniformArray(GI_TABLE.map(() => new THREE.Vector4(0.3, 0.4, 0.2, 0)), 'vec4')
      .setName(`bxbLpvAlbedo`).setGroup(tableGroup),
    // ... the SPECULAR share it bounces (lpv.js specularAlbedo): all of it, and
    // only its rough part - the smooth part the mirror light carries instead.
    layerSpecAll: uniformArray(GI_TABLE.map(() => new THREE.Vector4(0.04, 0.04, 0.04, 0)), 'vec4')
      .setName(`bxbLpvSpecAll`).setGroup(tableGroup),
    layerSpecRough: uniformArray(GI_TABLE.map(() => new THREE.Vector4(0.04, 0.04, 0.04, 0)), 'vec4')
      .setName(`bxbLpvSpecRough`).setGroup(tableGroup),
    // ... and for glass, its tint through one block (1 for everything else).
    layerTint: uniformArray(GI_TABLE.map(() => new THREE.Vector4(1, 1, 1, 0)), 'vec4')
      .setName(`bxbLpvTint`).setGroup(tableGroup),
    // 1: light thrown by the mirrors bounces from where it lands (and the
    // smooth texels' own share is left to it). 0: no mirror light - every
    // texel bounces its whole specular share where it is.
    mirrorGain: uniform(float(0)),
    // Sprites (lpv.js MAX_GI_SPRITES), two vec4 each:
    //   0  centre xyz, half width        1  albedo rgb (linear), half height
    // and a fill each (opaque fraction x lpv.js SPRITE_GI_FILL) in spriteFill.
    sprites: uniformArray(new Array(MAX_GI_SPRITES * 2).fill(0).map(() => new THREE.Vector4()), 'vec4')
      .setName(`bxbLpvSprites`).setGroup(tableGroup),
    spriteFill: uniformArray(new Array(MAX_GI_SPRITES).fill(0), 'float')
      .setName(`bxbLpvSpriteFill`).setGroup(tableGroup),
    spriteCount: uniform(int(0)),
    skyGain: uniform(float(1)),
    // The share of its neighbours' sky a roofed cell keeps (propagateSkyStep).
    skySpread: uniform(float(DEFAULT_LPV_SKY_SPREAD))
  };
  const skyOn = skyAB ? skyAB.equal(int(1)) : null;
  // Air open to the sky - every air cell without the sky-top map.
  const skyOpenAt = p => (terrain && terrain.skyTop ? skyOpenTSL(terrain, p) : float(1));
  const ambient = ambientUniform || uniform(float(0.17));
  const sun = createSunUniforms(new THREE.Vector3(0, 1, 0));
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const fadeStart = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const edgeFade = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));

  const fieldAt = p => {
    const vp = p.sub(c0.origin).div(c0.voxel);
    return decodeFieldTSL(ownField(ringUV(vp, c0.ring)).r, c0.range);
  };
  const glassAt = p => {
    const vp = p.sub(c0.origin).div(c0.voxel);
    return decodeFieldTSL(ownField(ringUV(vp, c0.ring)).g, c0.range);
  };
  // A block's entry in the per-layer tables, from the block volume: its
  // material's (R, layer + 1), or in a model's cell the model's (A, giIndex
  // + 1; render.js buildBlockVolume). 0 when there is neither.
  // q: the world point it is for - a dual block's half is the one holding it.
  const layerOf = (bi, q) => {
    const v = volumeAtTSL({ blocks }, bi, q);
    const code = select(v.r.greaterThan(float(0.5)), v.r, v.a);
    return int(max(code.sub(float(1)), float(0)));
  };
  // Centre of a cell, in world metres.
  const centreOf = c => u.origin.add(
    vec3(float(c.x), float(c.y), float(c.z)).add(float(0.5)).mul(cellU));

  // How much of cell c a sprite's box covers, as a fraction of the cell - the
  // mirror of lpv.js boxCellOverlap.
  const spriteOverlap = (cellMin, A, B) => {
    const half = vec3(A.w, B.w, A.w);
    const lo = max(cellMin, A.xyz.sub(half));
    const hi = min(cellMin.add(cellU), A.xyz.add(half));
    const e = max(hi.sub(lo), vec3(0, 0, 0)).div(cellU);
    return e.x.mul(e.y).mul(e.z);
  };
  const minCorner = c => u.origin.add(vec3(float(c.x), float(c.y), float(c.z))
                                        .mul(cellU));

  // --- solid: 27 field samples a cell, at voxel centres spread through it ---
  const solidKernel = Fn(() => {
    const c = cellOf(instanceIndex, D);
    const base = u.origin.add(vec3(float(c.x), float(c.y), float(c.z))
                                .mul(cellU)).toVar();
    const count = float(0).toVar();
    const glassCount = float(0).toVar();
    for (const oz of [1, 3, 5]) for (const oy of [1, 3, 5]) for (const ox of [1, 3, 5]) {
      const p = base.add(vec3(ox + 0.5, oy + 0.5, oz + 0.5).mul(voxU));
      count.addAssign(select(fieldAt(p).lessThan(float(0)), float(1), float(0)));
      if (glass) glassCount.addAssign(select(glassAt(p).lessThan(float(0)), float(1), float(0)));
    }
    if (glass) {
      // The cell's light, tinted by the glass in it: its block's tint through a
      // block, to the power of the cell's share of a block, blended by how much
      // of the cell is glass (lpv.js glassCellTransmit).
      const frac = glassCount.div(float(27));
      const centre = base.add(cellU.mul(float(0.5)));
      const tint = blocks ? u.layerTint.element(layerOf(round(centre.div(float(BLOCK_METRES))), centre)).xyz
                          : vec3(1, 1, 1);
      const through = pow(max(tint, vec3(1e-3, 1e-3, 1e-3)),
                          vec3(blockShareU, blockShareU, blockShareU));
      T.element(c.i).assign(vec4(mix(vec3(1, 1, 1), through, frac), 0));
    }
    // Plus the sprites standing in it - partly, since a sprite is a flat card.
    const sprite = float(0).toVar();
    Loop({ start: int(0), end: u.spriteCount, type: 'int', condition: '<', name: 'sp' },
         ({ sp }) => {
      const A = u.sprites.element(sp.mul(int(2))).toVar();
      const B = u.sprites.element(sp.mul(int(2)).add(int(1))).toVar();
      sprite.addAssign(spriteOverlap(base, A, B).mul(u.spriteFill.element(sp)));
    });
    solid.element(c.i).assign(min(count.div(float(27)).add(sprite), float(1)));
    smokeK.element(c.i).assign(exp(smokeSampleTSL(smokeShared, base.add(cellU.mul(float(0.5)))).w.mul(cellU).negate()));
  })().compute(CELLS).setName(`LPV.Solid`);

  // --- inject: light leaving the surfaces around each air cell ---
  //
  // Six probes along the axes, each up to one cell out. Where one lands on a
  // surface, that surface's direct light - the sun's cone trace and the whole
  // light list, the same functions the shading pass calls - times its albedo is
  // light this cell receives from that side. The terrain is boxes, so a probe
  // along an axis meets a face square on and the surface normal is -dir.
  // glass: the terrain textures when the world has glass (false otherwise):
  // the sun and the lights reach a probe through glass, tinted by the texel
  // they entered through - so a floor behind a window is lit, and bounces,
  // the window's colour.
  const coneTrace = createConeTraceSunTSL(cascades, !!glass);
  const pointLights = createPointLightsTSL({ cascades, lights, biasUniform: bias,
                                             fadeStartUniform: fadeStart,
                                             edgeFadeUniform: edgeFade, cards: null, glass });
  const maxProbe = cellU;
  const sunReach = float(12);
  // Sky visibility at a probe hit: one wide cone (45 degree half-angle) up
  // and out from the surface, over SKY_REACH. Coarse - it is the 0.75 m LPV
  // it feeds, not a texel - but it keeps an overhang's underside or a room's
  // interior from bouncing sky it cannot see.
  const SKY_REACH = float(4);
  const SKY_CONE = float(1);
  // The albedo of the block a probe landed on: which block (the block volume),
  // then that material's diffuse albedo.
  const blockLayer = (p, n) => {
    const q = p.sub(n.mul(c0.voxel.mul(float(0.5)))).toVar();
    return layerOf(round(q.div(float(BLOCK_METRES))), q);
  };
  const blockAlbedo = (p, n) => (blocks ? u.layerAlbedo.element(blockLayer(p, n)).xyz : u.albedo);
  // What the surface bounces in all: its diffuse albedo, plus its specular
  // share (lpv.js specularAlbedo) - the rough part only while the mirror light
  // carries the smooth part, all of it otherwise.
  const blockBounce = (p, n) => {
    if (!blocks) return u.albedo;
    const l = blockLayer(p, n).toVar();
    const spec = mix(u.layerSpecAll.element(l).xyz, u.layerSpecRough.element(l).xyz, u.mirrorGain);
    return u.layerAlbedo.element(l).xyz.add(spec);
  };

  // The mirror light at a probe hit: light a mirror throws there bounces from
  // there, times the surface's own diffuse albedo - where it lands, not at the
  // mirror. The same function the shading pass calls (createMirrorLightTSL),
  // lit by this volume's sun and its shadow; cards are left out, as the
  // injection's lights leave them out.
  const giSunShadow = (p, n) => {
    const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();
    const traced = coneTrace(origin, sun.dir, sunReach, sun.coneRadius, fadeStart, edgeFade).toVar();
    // The far map, hard: the bounce under a long shadow is dark too.
    const vis = traced.x.mul(sunRoofTSL(terrain, origin, sun.dir, sunReach))
      .mul(sunMapFarTSL(origin, sunReach, fadeStart, sun.coneRadius, float(0)));
    return glass ? glassRayTransmitTSL(glass, traced, origin, sun.dir).mul(vis) : vis;
  };
  const mirrorLight = terrain && mirrors
    ? createMirrorLightTSL({ cascades, terrain, mirrors, lights, sun, sunShadow: giSunShadow,
                             sunLight: () => u.sunGain, sunOn: true, bias,
                             fadeStartUniform: fadeStart, edgeFadeUniform: edgeFade })
    : null;

  const injectKernel = Fn(() => {
    const c = cellOf(instanceIndex, D);
    If(c.i.mod(u.slices).equal(u.slice), () => {
      const out = vec3(0, 0, 0).toVar();
      // What the surfaces round this cell bounce, summed as the light is: the
      // later orders' light leaves them by this (propagate).
      const bounceW = vec3(0, 0, 0).toVar();
      const centre = centreOf(c).toVar();
      const air = solid.element(c.i).lessThan(float(SOLID_THRESHOLD))
                    .and(fieldAt(centre).greaterThan(float(0)));
      If(air, () => {
        // A loop, not six unrolled copies: each face calls the sun trace and the
        // whole light loop, and unrolling would put six of those in one kernel.
        // Faces in FACE_DIRS order: +x -x +y -y +z -z.
        Loop({ start: int(0), end: int(6), type: 'int', condition: '<', name: 'face' },
             ({ face }) => {
          const axis = face.div(int(2));
          const sgn = select(face.mod(int(2)).equal(int(0)), float(1), float(-1));
          const dir = vec3(select(axis.equal(int(0)), sgn, float(0)),
                           select(axis.equal(int(1)), sgn, float(0)),
                           select(axis.equal(int(2)), sgn, float(0))).toVar();
          const t = float(0).toVar();
          const hit = float(0).toVar();
          Loop({ start: 0, end: 24, type: 'int', condition: '<' }, () => {
            const d = fieldAt(centre.add(dir.mul(t)));
            If(d.lessThan(float(HIT_EPS)), () => { hit.assign(float(1)); Break(); });
            t.addAssign(max(d, float(MIN_STEP)).mul(c0.voxel));
            If(t.greaterThan(maxProbe), () => { Break(); });
          });
          If(hit.greaterThan(float(0)), () => {
            const p = centre.add(dir.mul(t)).toVar();
            const n = dir.negate().toVar();
            const ndotl = max(n.dot(sun.dir), float(0)).toVar();
            const sunLit = vec3(0, 0, 0).toVar();
            If(ndotl.greaterThan(float(0)).and(u.sunGain.greaterThan(float(0))), () => {
              const origin = p.add(n.mul(bias.mul(c0.voxel))).toVar();
              const traced = coneTrace(origin, sun.dir, sunReach, sun.coneRadius,
                                       fadeStart, edgeFade).toVar();
              const through = glass ? glassRayTransmitTSL(glass, traced, origin, sun.dir)
                                    : vec3(1, 1, 1);
              sunLit.assign(through.mul(traced.x.mul(sunRoofTSL(terrain, origin, sun.dir, sunReach))
                                               .mul(sunMapFarTSL(origin, sunReach, fadeStart, sun.coneRadius,
                                                                 float(0)))
                                               .mul(ndotl).mul(u.sunGain)));
            });
            const lit = sunColourUniform.mul(sunLit).add(pointLights(p, n, NO_SKIP, n)).toVar();
            // Sky light the surface reflects: its sky irradiance, as the
            // shading pass computes it, times how much sky it can see. Not the
            // sky itself - that is the per-fragment ambient already, and
            // injecting it would count it twice.
            // With skyAB, the visibility is this cell's own (last frame's),
            // and no cone is marched.
            if (sky) {
              If(n.y.greaterThan(float(-0.5)), () => {
                const vis = float(0).toVar();
                const cone = () => {
                  const origin = p.add(n.mul(bias.mul(c0.voxel))).toVar();
                  const up = normalize(n.add(vec3(0, 1, 0)));
                  vis.assign(coneTrace(origin, up, SKY_REACH, SKY_CONE, fadeStart, edgeFade).x);
                };
                if (skyOn) If(skyOn, () => { vis.assign(lightW(A, c.i)); }).Else(cone);
                else cone();
                lit.addAssign(skyShTSL(sky, n, true).mul(ambient).mul(u.skyGain).mul(vis));
              });
            }
            const bb = blockBounce(p, n).toVar();
            out.addAssign(lit.mul(bb));
            bounceW.addAssign(bb);
            if (mirrorLight) {
              If(u.mirrorGain.greaterThan(float(0)), () => {
                out.addAssign(mirrorLight(p, n, n, n).mul(blockAlbedo(p, n)).mul(u.mirrorGain));
              });
            }
          });
        });
      });
      // Sprites in this cell: each box that overlaps it injects its bounce -
      // its albedo lit by the sun (a shadow march from its centre) and the
      // sky - in proportion to how much of the cell it fills.
      const spriteE = vec3(0, 0, 0).toVar();
      const cmin = minCorner(c).toVar();
      Loop({ start: int(0), end: u.spriteCount, type: 'int', condition: '<', name: 'sp' },
           ({ sp }) => {
        const A = u.sprites.element(sp.mul(int(2))).toVar();
        const B = u.sprites.element(sp.mul(int(2)).add(int(1))).toVar();
        const w = spriteOverlap(cmin, A, B).mul(u.spriteFill.element(sp)).toVar();
        If(w.greaterThan(float(0)), () => {
          // A card faces its viewer, so its Lambert against the sun is taken
          // as the sun's horizontal share - bright at a low sun, dim overhead.
          const facing = length(vec2(sun.dir.x, sun.dir.z));
          const vis = coneTrace(A.xyz, sun.dir, sunReach, sun.coneRadius,
                                fadeStart, edgeFade).x.mul(sunRoofTSL(terrain, A.xyz, sun.dir, sunReach));
          const light = sunColourUniform.mul(vis.mul(facing).mul(u.sunGain)).toVar();
          if (sky) light.addAssign(skyShTSL(sky, vec3(0, 1, 0), true).mul(ambient)
                                     .mul(u.skyGain).mul(float(0.5)));
          spriteE.addAssign(B.xyz.mul(light).mul(w));
        });
      });
      const open = select(air, skyOpenAt(centre), float(0));
      // Cloud spells' smoke in this cell gives back the light it is lit by -
      // the sun, through the world and through the smoke toward it, and the
      // sky - times its colour and the share of light it scatters here.
      const smokeE = vec3(0, 0, 0).toVar();
      const smk = smokeSampleTSL(smokeShared, centre).toVar();
      If(smk.w.greaterThan(float(1e-4)), () => {
        const lit = vec3(0, 0, 0).toVar();
        If(u.sunGain.greaterThan(float(0)), () => {
          const traced = coneTrace(centre, sun.dir, sunReach, sun.coneRadius, fadeStart, edgeFade).toVar();
          const through = glass ? glassRayTransmitTSL(glass, traced, centre, sun.dir) : vec3(1, 1, 1);
          lit.assign(sunColourUniform.mul(through).mul(traced.x.mul(sunRoofTSL(terrain, centre, sun.dir, sunReach)))
            .mul(u.sunGain).mul(smokeSunTransmitTSL(smokeShared, centre, sun.dir)));
        });
        if (sky) lit.addAssign(skyShTSL(sky, vec3(0, 1, 0), true).mul(ambient).mul(u.skyGain).mul(skyOpenAt(centre)));
        const colour = smk.xyz.div(smk.w);
        smokeE.assign(lit.mul(colour).mul(float(1).sub(exp(smk.w.mul(cellU).negate()))));
      });
      E.element(c.i).assign(vec4(out.div(float(6)).add(spriteE).add(smokeE), open));
      W.element(c.i).assign(vec4(bounceW.div(float(6)), 0));
    });
  })().compute(CELLS).setName(`LPV.Inject`);

  // --- fog: the light at each air cell's centre, for the froxels ---
  //
  // The same sun march and light loop the injection runs at its probe hits,
  // but at the cell centre and with no surface: the sun's visibility with its
  // glass tint and the roof test, and every light shadowed with no N.L (a
  // point in the air is lit from all sides). Sliced as the injection is.
  const fogKernel = Fn(() => {
    const c = cellOf(instanceIndex, D);
    If(c.i.mod(u.slices).equal(u.slice), () => {
      const centre = centreOf(c).toVar();
      const sunT = vec3(0, 0, 0).toVar(), lamp = vec3(0, 0, 0).toVar();
      If(fieldAt(centre).greaterThan(float(0)), () => {
        If(u.sunGain.greaterThan(float(0)), () => {
          const traced = coneTrace(centre, sun.dir, sunReach, sun.coneRadius, fadeStart, edgeFade).toVar();
          const through = glass ? glassRayTransmitTSL(glass, traced, centre, sun.dir) : vec3(1, 1, 1);
          sunT.assign(through.mul(traced.x.mul(sunRoofTSL(terrain, centre, sun.dir, sunReach)).mul(u.sunGain)));
        });
        lamp.assign(pointLights.withSpecular(centre, vec3(0, 0, 0), NO_SKIP, null));
      });
      textureStore(fogSunTex, uvec3(c.x, c.y, c.z), vec4(sunT, 0));
      textureStore(fogLampTex, uvec3(c.x, c.y, c.z), vec4(lamp, 0));
    });
  })().compute(CELLS).setName(`LPV.Fog`);

  // --- propagate: one iteration, src -> dst (lpv.js propagateStep) ---
  // Every order in one loop. Order 0 is lit by the injection (E); order k by
  // order k - 1 in this cell, as the shading pass would read it (times
  // 1 - spread, see resolve), leaving the surfaces here times what they
  // bounce (W). From the last iteration's answer, so the chain settles over
  // frames as the propagation itself does.
  const propagate = (src, dst) => Fn(() => {
    const c = cellOf(instanceIndex, D);
    // Smoke in the cell holds the light back as glass tints it.
    const open = float(1).sub(solid.element(c.i)).mul(smokeK.element(c.i)).toVar();
    // Glass tints the light in its cells (lpv.js propagateStep's trans).
    const keep = (T ? T.element(c.i).xyz.mul(open) : vec3(open, open, open)).toVar();
    const nbs = FACE_DIRS.map(([dx, dy, dz]) => {
      const nb = neighbour(c, dx, dy, dz, D);
      return { index: nb.index.toVar(),
               w: select(nb.inside, float(1).sub(solid.element(nb.index)), open).toVar() };
    });
    const rebounce = W.element(c.i).xyz.mul(float(1).sub(u.spread)).toVar();
    // Sky visibility, order 0's w (lpv.js propagateSkyStep): 1 open to the
    // sky, else the mean of its air neighbours' less a share; glass passes
    // its tint's mean.
    const vsum = float(0).toVar(), vw = float(0).toVar();
    for (const nb of nbs) { vsum.addAssign(lightW(src, nb.index).mul(nb.w)); vw.addAssign(nb.w); }
    const seen = select(E.element(c.i).w.greaterThan(float(0.5)), float(1),
                        u.skySpread.mul(vsum).div(max(vw, float(1e-4))));
    const vis = seen.mul(T ? keep.dot(vec3(1 / 3, 1 / 3, 1 / 3)) : open).toVar();
    Loop({ start: int(0), end: u.bounces, type: 'int', condition: '<', name: 'order' }, ({ order }) => {
      const base = order.mul(int(CELLS)).toVar();
      const sum = vec3(0, 0, 0).toVar();
      for (const nb of nbs) sum.addAssign(light(src, base.add(nb.index)).xyz.mul(nb.w));
      const prev = max(base.sub(int(CELLS)), int(0)).add(c.i);
      const emit = select(order.equal(int(0)), E.element(c.i).xyz, light(src, prev).xyz.mul(rebounce));
      setLight(dst, base.add(c.i), vec4(emit.add(sum.mul(u.spread.div(float(6)))).mul(keep),
                                        select(order.equal(int(0)), vis, float(0))));
    });
  })().compute(CELLS).setName(`LPV.Propagate`);
  const aToB = propagate(A, B), bToA = propagate(B, A);

  // --- resolve: A into the texture, normalised, dilated into solid cells ---
  // Every order's light, summed: what the cell holds in all.
  const totalAt = i => {
    const s = vec3(0, 0, 0).toVar();
    Loop({ start: int(0), end: u.bounces, type: 'int', condition: '<', name: 'order' }, ({ order }) => {
      s.addAssign(light(A, order.mul(int(CELLS)).add(i)).xyz);
    });
    return s;
  };
  const resolveKernel = Fn(() => {
    const c = cellOf(instanceIndex, D);
    const own = totalAt(c.i).toVar();
    const isSolid = solid.element(c.i).greaterThanEqual(float(SOLID_THRESHOLD));
    const outc = own.toVar();
    const outv = lightW(A, c.i).toVar();
    If(isSolid, () => {
      const sum = vec3(0, 0, 0).toVar();
      const vsum = float(0).toVar();
      const wsum = float(0).toVar();
      for (const [dx, dy, dz] of FACE_DIRS) {
        const nb = neighbour(c, dx, dy, dz, D);
        const w = select(nb.inside, float(1).sub(solid.element(nb.index)), float(0));
        If(w.greaterThan(float(0)), () => {
          sum.addAssign(totalAt(nb.index).mul(w));
          vsum.addAssign(lightW(A, nb.index).mul(w));
        });
        wsum.addAssign(w);
      }
      outc.assign(select(wsum.greaterThan(float(0)), sum.div(max(wsum, float(1e-6))), own));
      outv.assign(select(wsum.greaterThan(float(0)), vsum.div(max(wsum, float(1e-6))), outv));
    });
    // (1 - spread): see DEFAULT_LPV_SPREAD - brightness independent of reach.
    // The sky visibility as it is: it is a share, not light.
    textureStore(tex, uvec3(c.x, c.y, c.z),
                 vec4(outc.mul(float(1).sub(u.spread)), outv));
  })().compute(CELLS).setName(`LPV.Resolve`);

  // --- shift: when C0 re-origins, move A by whole cells (lpv.js shiftVolume) ---
  // Over every order: thread k * CELLS + i is cell i of order k.
  const shiftKernel = Fn(() => {
    const at = int(instanceIndex).toVar();
    const base = at.sub(at.mod(int(CELLS))).toVar();
    const c = cellOf(at.mod(int(CELLS)), D);
    const sx = c.x.add(int(u.shift.x)), sy = c.y.add(int(u.shift.y)), sz = c.z.add(int(u.shift.z));
    const inside = sx.greaterThanEqual(int(0)).and(sy.greaterThanEqual(int(0)))
      .and(sz.greaterThanEqual(int(0))).and(sx.lessThan(int(D)))
      .and(sy.lessThan(int(D))).and(sz.lessThan(int(D)));
    const s = sx.add(sy.mul(int(D))).add(sz.mul(int(D * D)));
    B.element(at).assign(select(inside, A.element(base.add(select(inside, s, int(0)))), noLight()));
  })().compute(ORDERS).setName(`LPV.Shift`);
  const copyBA = Fn(() => {
    A.element(instanceIndex).assign(B.element(instanceIndex));
  })().compute(ORDERS).setName(`LPV.Copy`);
  const clearA = Fn(() => {
    A.element(instanceIndex).assign(noLight());
    If(int(instanceIndex).lessThan(int(CELLS)), () => {
      E.element(instanceIndex).assign(vec4(0));
      W.element(instanceIndex).assign(vec4(0));
    });
  })().compute(ORDERS).setName(`LPV.Clear`);

  let frame = 0;
  // Kept on the CPU: the uniform is overwritten with 1 on a full inject.
  let sliceCount = DEFAULT_LPV_SLICES;

  // The frame's kernels, in order, as one list. renderer.compute(list) records
  // them all into ONE compute pass and submits once; WebGPU orders the
  // dispatches within a pass and makes each one's storage writes visible to the
  // next, so the ping-pong is exactly what separate calls computed. Separate
  // calls were a pass and a submit each - 11 a volume, and each submit costs
  // ~140 us of GPU-process time whatever it holds (doc §0, profiles).
  // Every uniform the kernels read is written before the list is submitted,
  // and none changes partway through it, so batching cannot change a value.
  // Cached, not rebuilt per frame: three keys the pass's state on the array.
  const lists = new Map();
  const kernelList = (prefix, iterations, fog, inject = true) => {
    const key = `${prefix}:${iterations}:${fog}:${inject}`;
    let list = lists.get(key);
    if (!list) {
      list = prefix === 'reset' ? [clearA] : prefix === 'shift' ? [shiftKernel, copyBA] : [];
      if (inject) list.push(solidKernel, injectKernel);
      if (fog) list.push(fogKernel);
      for (let k = 0; k < iterations; k += 2) list.push(aToB, bToA);
      list.push(resolveKernel);
      lists.set(key, list);
    }
    return list;
  };

  return {
    tex, uniforms: u, sun, level, cellMetres, dim: D, res, orders: orderCount,
    // What the shading pass needs to sample it: see lpvSampleTSL in ambient.js.
    binding: { tex, origin: u.origin, extent: D * cellMetres, dim: D, cellMetres, fogSunTex, fogLampTex },

    // Injection slices: 1/N of the cells re-injected each frame.
    get slices() { return sliceCount; },
    set slices(n) { sliceCount = Math.max(1, Math.round(n)); },

    // One frame. shift is lpv.js lpvShift's answer, 'reset' for a jump the
    // volume cannot follow, or null when C0 did not move.
    // batch false submits each kernel on its own, as before - the same list in
    // the same order, so the two can be A/B'd (bxb.gibatch()).
    // fog: also light the air for the froxels (the fog kernel).
    // inject false: propagate only - the solid and inject kernels (and the
    // slice they would have done) wait for an update that injects. A shift
    // or a reset always injects.
    update(renderer, { shift = null, iterations = 8, fullInject = false, batch = true, fog = false,
                       inject = true } = {}) {
      let prefix = '';
      if (shift === 'reset') {
        prefix = 'reset';
        fullInject = true;
      } else if (shift) {
        u.shift.value.set(shift.x, shift.y, shift.z);
        prefix = 'shift';
        fullInject = true;
      }
      if (fullInject) inject = true;
      if (inject) {
        const slices = Math.max(1, sliceCount);
        // A full inject is one dispatch with a single slice covering every cell:
        // the old injection is in the wrong cells after a shift.
        u.slices.value = fullInject ? 1 : slices;
        u.slice.value = fullInject ? 0 : frame % slices;
        frame++;
      }
      // Even, so the answer always ends in A, where resolve reads it.
      const n = Math.max(2, iterations + (iterations % 2));
      const list = kernelList(prefix, n, fog, inject);
      if (batch) renderer.compute(list);
      else for (const k of list) renderer.compute(k);
    },

    // renderer: frees the storage buffers too - at res 4 they are most of
    // a hundred MB.
    dispose(renderer = null) {
      tex.dispose();
      fogSunTex.dispose();
      fogLampTex.dispose();
      const attrs = renderer && renderer._attributes;
      if (attrs) for (const b of [solidBuf, smokeBuf, eBuf, wBuf, aBuf, bBuf, tBuf]) if (b) attrs.delete(b);
    }
  };
}
