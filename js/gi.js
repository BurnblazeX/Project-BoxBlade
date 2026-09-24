import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, float, vec2, vec3, vec4, int, uvec3, uniform, instanceIndex,
  storage, texture3D, textureStore, max, min, select, normalize, round, uniformArray,
  length, pow, mix
} from 'three/tsl';
import { HIT_EPS, MIN_STEP, cascadeVoxelMetres, VOXEL_METRES } from './boxgrid.js';
import {
  decodeFieldTSL, ringUV, createConeTraceSunTSL, createPointLightsTSL,
  createSunUniforms, SURFACE_BIAS_VOXELS, SHADOW_FADE_START, EDGE_FADE_VOXELS,
  sunColourUniform, skyShTSL, glassRayTransmitTSL, createMirrorLightTSL
} from './gpu.js';
import { BLOCK_METRES } from './world.js';
import { MATERIALS } from './materials.js';
import {
  LPV_DIM, LPV_CELLS, LPV_CELL_VOXELS, FACE_DIRS, SOLID_THRESHOLD,
  DEFAULT_LPV_SPREAD, DEFAULT_LPV_SLICES, MAX_GI_SPRITES
} from './lpv.js';

// --- The LPV on the GPU: solid, inject, propagate, resolve ---
//
// lpv.js has the why and the CPU reference; this is the kernels. Four storage
// buffers and one 3D texture:
//
//   solid   float   how much of each cell is geometry        every frame
//   E       vec4    injected light (rgb)                     one slice a frame
//   A, B    vec4    propagated light, ping-ponged            N iterations a frame
//   tex     RGBA16F what the shading pass samples, trilinear  every frame
//
// The texture is written by a resolve pass that DILATES into solid cells: a
// solid cell takes the mean of its air neighbours. The surfaces the shading pass
// samples near are exactly where trilinear filtering would otherwise blend in
// the zeros inside the ground and darken every contact.

const NO_SKIP = vec2(1e9, 1e9);
const D = LPV_DIM;

// instanceIndex to cell coordinates, as ints.
function cellOf(index) {
  const i = int(index).toVar();
  return { i, x: i.mod(int(D)), y: i.div(int(D)).mod(int(D)), z: i.div(int(D * D)) };
}

// The neighbour one step along (dx, dy, dz), or this cell when that is off the
// volume - the CPU reference's edge rule.
function neighbour(c, dx, dy, dz) {
  const nx = c.x.add(int(dx)), ny = c.y.add(int(dy)), nz = c.z.add(int(dz));
  const inside = nx.greaterThanEqual(int(0)).and(ny.greaterThanEqual(int(0)))
    .and(nz.greaterThanEqual(int(0))).and(nx.lessThan(int(D)))
    .and(ny.lessThan(int(D))).and(nz.lessThan(int(D)));
  const n = nx.add(ny.mul(int(D))).add(nz.mul(int(D * D)));
  return { inside, index: select(inside, n, c.i) };
}

// level: which cascade the volume lies over. C0 gives 0.75 m cells over 18 m,
// C1 1.5 m cells over 36 m - the same 24^3 and six voxels a cell either way,
// because each cascade is the same 144^3 at a doubled voxel size.
// sky: the sky SH bindings (gpu.js createSkyBindings), with ambientUniform its
// scale. blocks: render.js's block-material volume, so a probe hit bounces its
// own block's albedo. Either may be null, and the old behaviour stands.
export function createLPV({ cascades, lights, level = 0, biasUniform = null,
                            fadeStartUniform = null, edgeFadeUniform = null,
                            sky = null, ambientUniform = null, blocks = null,
                            glass = false, terrain = null, mirrors = null }) {
  // terrain: the terrain textures, for the mirror light's texels (with mirrors,
  // the mirror bindings) - see the injection below. Either null: no mirror bounce.
  const c0 = cascades[level];
  const cellMetres = LPV_CELL_VOXELS * cascadeVoxelMetres(level);
  const buf = (itemSize) => new THREE.StorageBufferAttribute(LPV_CELLS, itemSize);
  const solidBuf = buf(1), eBuf = buf(4), aBuf = buf(4), bBuf = buf(4);
  // Named: an unnamed buffer is NodeBuffer_<global node id> in the WGSL, and
  // that id depends on how many nodes existed before - so the shader text, and
  // with it the browser's pipeline cache key, changed from run to run.
  const solid = storage(solidBuf, 'float', LPV_CELLS).setName(`bxbLpv${level}Solid`);
  const E = storage(eBuf, 'vec4', LPV_CELLS).setName(`bxbLpv${level}E`);
  // Per cell, what its light is multiplied by: glass tints it (lpv.js
  // glassCellTransmit). Only in a world with glass.
  const tBuf = glass ? buf(4) : null;
  const T = glass ? storage(tBuf, 'vec4', LPV_CELLS).setName(`bxbLpv${level}T`) : null;
  const A = storage(aBuf, 'vec4', LPV_CELLS).setName(`bxbLpv${level}A`);
  const B = storage(bBuf, 'vec4', LPV_CELLS).setName(`bxbLpv${level}B`);

  const tex = new THREE.Storage3DTexture(D, D, D);
  tex.type = THREE.HalfFloatType;
  tex.format = THREE.RGBAFormat;
  tex.generateMipmaps = false;
  tex.minFilter = tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;

  const u = {
    // World-space min corner of the volume: C0's origin, tracked by the CPU.
    origin: uniform(new THREE.Vector3()),
    spread: uniform(float(DEFAULT_LPV_SPREAD)),
    albedo: uniform(new THREE.Vector3(0.3, 0.4, 0.2)),
    sunGain: uniform(float(1)),
    slice: uniform(int(0)),
    slices: uniform(int(DEFAULT_LPV_SLICES)),
    shift: uniform(new THREE.Vector3()),
    // Per material layer: the DIFFUSE albedo it bounces (lpv.js diffuseAlbedo).
    layerAlbedo: uniformArray(MATERIALS.map(() => new THREE.Vector4(0.3, 0.4, 0.2, 0)), 'vec4')
      .setName(`bxbLpv${level}Albedo`),
    // ... the SPECULAR share it bounces (lpv.js specularAlbedo): all of it, and
    // only its rough part - the smooth part the mirror light carries instead.
    layerSpecAll: uniformArray(MATERIALS.map(() => new THREE.Vector4(0.04, 0.04, 0.04, 0)), 'vec4')
      .setName(`bxbLpv${level}SpecAll`),
    layerSpecRough: uniformArray(MATERIALS.map(() => new THREE.Vector4(0.04, 0.04, 0.04, 0)), 'vec4')
      .setName(`bxbLpv${level}SpecRough`),
    // ... and for glass, its tint through one block (1 for everything else).
    layerTint: uniformArray(MATERIALS.map(() => new THREE.Vector4(1, 1, 1, 0)), 'vec4')
      .setName(`bxbLpv${level}Tint`),
    // 1: light thrown by the mirrors bounces from where it lands (and the
    // smooth texels' own share is left to it). 0: no mirror light - every
    // texel bounces its whole specular share where it is.
    mirrorGain: uniform(float(0)),
    // Sprites (lpv.js MAX_GI_SPRITES), two vec4 each:
    //   0  centre xyz, half width        1  albedo rgb (linear), half height
    // and a fill each (opaque fraction x lpv.js SPRITE_GI_FILL) in spriteFill.
    sprites: uniformArray(new Array(MAX_GI_SPRITES * 2).fill(0).map(() => new THREE.Vector4()), 'vec4')
      .setName(`bxbLpv${level}Sprites`),
    spriteFill: uniformArray(new Array(MAX_GI_SPRITES).fill(0), 'float')
      .setName(`bxbLpv${level}SpriteFill`),
    spriteCount: uniform(int(0)),
    skyGain: uniform(float(1))
  };
  const ambient = ambientUniform || uniform(float(0.17));
  const sun = createSunUniforms(new THREE.Vector3(0, 1, 0));
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const fadeStart = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const edgeFade = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));

  const fieldAt = p => {
    const vp = p.sub(c0.origin).div(c0.voxel);
    return decodeFieldTSL(texture3D(c0.tex, ringUV(vp, c0.ring)).r, c0.range);
  };
  const glassAt = p => {
    const vp = p.sub(c0.origin).div(c0.voxel);
    return decodeFieldTSL(texture3D(c0.tex, ringUV(vp, c0.ring)).g, c0.range);
  };
  // A block's material layer, from the block volume (0 when there is none).
  const layerOf = bi => {
    const uvw = bi.sub(blocks.origin).add(float(0.5)).div(blocks.size);
    const code = round(texture3D(blocks.tex, uvw).level(int(0)).r.mul(float(255)));
    return int(max(code.sub(float(1)), float(0)));
  };
  // Centre of a cell, in world metres.
  const centreOf = c => u.origin.add(
    vec3(float(c.x), float(c.y), float(c.z)).add(float(0.5)).mul(float(cellMetres)));

  // How much of cell c a sprite's box covers, as a fraction of the cell - the
  // mirror of lpv.js boxCellOverlap.
  const spriteOverlap = (cellMin, A, B) => {
    const half = vec3(A.w, B.w, A.w);
    const lo = max(cellMin, A.xyz.sub(half));
    const hi = min(cellMin.add(float(cellMetres)), A.xyz.add(half));
    const e = max(hi.sub(lo), vec3(0, 0, 0)).div(float(cellMetres));
    return e.x.mul(e.y).mul(e.z);
  };
  const minCorner = c => u.origin.add(vec3(float(c.x), float(c.y), float(c.z))
                                        .mul(float(cellMetres)));

  // --- solid: 27 field samples a cell, at voxel centres spread through it ---
  const solidKernel = Fn(() => {
    const c = cellOf(instanceIndex);
    const base = u.origin.add(vec3(float(c.x), float(c.y), float(c.z))
                                .mul(float(cellMetres))).toVar();
    const count = float(0).toVar();
    const glassCount = float(0).toVar();
    const vox = cellMetres / LPV_CELL_VOXELS;
    for (const oz of [1, 3, 5]) for (const oy of [1, 3, 5]) for (const ox of [1, 3, 5]) {
      const p = base.add(vec3((ox + 0.5) * vox, (oy + 0.5) * vox, (oz + 0.5) * vox));
      count.addAssign(select(fieldAt(p).lessThan(float(0)), float(1), float(0)));
      if (glass) glassCount.addAssign(select(glassAt(p).lessThan(float(0)), float(1), float(0)));
    }
    if (glass) {
      // The cell's light, tinted by the glass in it: its block's tint through a
      // block, to the power of the cell's share of a block, blended by how much
      // of the cell is glass (lpv.js glassCellTransmit).
      const frac = glassCount.div(float(27));
      const centre = base.add(float(cellMetres * 0.5));
      const tint = blocks ? u.layerTint.element(layerOf(round(centre.div(float(BLOCK_METRES))))).xyz
                          : vec3(1, 1, 1);
      const through = pow(max(tint, vec3(1e-3, 1e-3, 1e-3)),
                          vec3(cellMetres / BLOCK_METRES, cellMetres / BLOCK_METRES,
                               cellMetres / BLOCK_METRES));
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
  })().compute(LPV_CELLS).setName(`LPV${level}.Solid`);

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
  const maxProbe = float(cellMetres);
  const sunReach = float(12);
  // Sky visibility at a probe hit: one wide cone (45 degree half-angle) up
  // and out from the surface, over SKY_REACH. Coarse - it is the 0.75 m LPV
  // it feeds, not a texel - but it keeps an overhang's underside or a room's
  // interior from bouncing sky it cannot see.
  const SKY_REACH = float(4);
  const SKY_CONE = float(1);
  // The albedo of the block a probe landed on: which block (the block volume),
  // then that material's diffuse albedo.
  const blockLayer = (p, n) =>
    layerOf(round(p.sub(n.mul(c0.voxel.mul(float(0.5)))).div(float(BLOCK_METRES))));
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
    return glass ? glassRayTransmitTSL(glass, traced, origin, sun.dir).mul(traced.x) : traced.x;
  };
  const mirrorLight = terrain && mirrors
    ? createMirrorLightTSL({ cascades, terrain, mirrors, lights, sun, sunShadow: giSunShadow,
                             sunLight: () => u.sunGain, sunOn: true, bias,
                             fadeStartUniform: fadeStart, edgeFadeUniform: edgeFade })
    : null;

  const injectKernel = Fn(() => {
    const c = cellOf(instanceIndex);
    If(c.i.mod(u.slices).equal(u.slice), () => {
      const out = vec3(0, 0, 0).toVar();
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
              sunLit.assign(through.mul(traced.x.mul(ndotl).mul(u.sunGain)));
            });
            const lit = sunColourUniform.mul(sunLit).add(pointLights(p, n, NO_SKIP, n)).toVar();
            // Sky light the surface reflects: its sky irradiance, as the
            // shading pass computes it, times how much sky it can see. Not the
            // sky itself - that is the per-fragment ambient already, and
            // injecting it would count it twice.
            if (sky) {
              If(n.y.greaterThan(float(-0.5)), () => {
                const origin = p.add(n.mul(bias.mul(c0.voxel))).toVar();
                const up = normalize(n.add(vec3(0, 1, 0)));
                const vis = coneTrace(origin, up, SKY_REACH, SKY_CONE, fadeStart, edgeFade).x;
                lit.addAssign(skyShTSL(sky, n, true).mul(ambient).mul(u.skyGain).mul(vis));
              });
            }
            out.addAssign(lit.mul(blockBounce(p, n)));
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
                                fadeStart, edgeFade).x;
          const light = sunColourUniform.mul(vis.mul(facing).mul(u.sunGain)).toVar();
          if (sky) light.addAssign(skyShTSL(sky, vec3(0, 1, 0), true).mul(ambient)
                                     .mul(u.skyGain).mul(float(0.5)));
          spriteE.addAssign(B.xyz.mul(light).mul(w));
        });
      });
      E.element(c.i).assign(vec4(out.div(float(6)).add(spriteE), 0));
    });
  })().compute(LPV_CELLS).setName(`LPV${level}.Inject`);

  // --- propagate: one iteration, src -> dst (lpv.js propagateStep) ---
  const propagate = (src, dst) => Fn(() => {
    const c = cellOf(instanceIndex);
    const open = float(1).sub(solid.element(c.i)).toVar();
    const sum = vec3(0, 0, 0).toVar();
    for (const [dx, dy, dz] of FACE_DIRS) {
      const nb = neighbour(c, dx, dy, dz);
      const w = select(nb.inside, float(1).sub(solid.element(nb.index)), open);
      sum.addAssign(src.element(nb.index).xyz.mul(w));
    }
    const next = E.element(c.i).xyz.add(sum.mul(u.spread.div(float(6)))).mul(open).toVar();
    // Glass tints the light in its cells (lpv.js propagateStep's trans).
    if (T) next.mulAssign(T.element(c.i).xyz);
    dst.element(c.i).assign(vec4(next, 0));
  })().compute(LPV_CELLS).setName(`LPV${level}.Propagate`);
  const aToB = propagate(A, B), bToA = propagate(B, A);

  // --- resolve: A into the texture, normalised, dilated into solid cells ---
  const resolveKernel = Fn(() => {
    const c = cellOf(instanceIndex);
    const own = A.element(c.i).xyz.toVar();
    const isSolid = solid.element(c.i).greaterThanEqual(float(SOLID_THRESHOLD));
    const outc = own.toVar();
    If(isSolid, () => {
      const sum = vec3(0, 0, 0).toVar();
      const wsum = float(0).toVar();
      for (const [dx, dy, dz] of FACE_DIRS) {
        const nb = neighbour(c, dx, dy, dz);
        const w = select(nb.inside, float(1).sub(solid.element(nb.index)), float(0));
        sum.addAssign(A.element(nb.index).xyz.mul(w));
        wsum.addAssign(w);
      }
      outc.assign(select(wsum.greaterThan(float(0)), sum.div(max(wsum, float(1e-6))), own));
    });
    // (1 - spread): see DEFAULT_LPV_SPREAD - brightness independent of reach.
    textureStore(tex, uvec3(c.x, c.y, c.z),
                 vec4(outc.mul(float(1).sub(u.spread)), 1));
  })().compute(LPV_CELLS).setName(`LPV${level}.Resolve`);

  // --- shift: when C0 re-origins, move A by whole cells (lpv.js shiftVolume) ---
  const shiftKernel = Fn(() => {
    const c = cellOf(instanceIndex);
    const sx = c.x.add(int(u.shift.x)), sy = c.y.add(int(u.shift.y)), sz = c.z.add(int(u.shift.z));
    const inside = sx.greaterThanEqual(int(0)).and(sy.greaterThanEqual(int(0)))
      .and(sz.greaterThanEqual(int(0))).and(sx.lessThan(int(D)))
      .and(sy.lessThan(int(D))).and(sz.lessThan(int(D)));
    const s = sx.add(sy.mul(int(D))).add(sz.mul(int(D * D)));
    B.element(c.i).assign(select(inside, A.element(select(inside, s, int(0))), vec4(0)));
  })().compute(LPV_CELLS).setName(`LPV${level}.Shift`);
  const copyBA = Fn(() => {
    A.element(instanceIndex).assign(B.element(instanceIndex));
  })().compute(LPV_CELLS).setName(`LPV${level}.Copy`);
  const clearA = Fn(() => {
    A.element(instanceIndex).assign(vec4(0));
    E.element(instanceIndex).assign(vec4(0));
  })().compute(LPV_CELLS).setName(`LPV${level}.Clear`);

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
  const kernelList = (prefix, iterations) => {
    const key = `${prefix}:${iterations}`;
    let list = lists.get(key);
    if (!list) {
      list = prefix === 'reset' ? [clearA] : prefix === 'shift' ? [shiftKernel, copyBA] : [];
      list.push(solidKernel, injectKernel);
      for (let k = 0; k < iterations; k += 2) list.push(aToB, bToA);
      list.push(resolveKernel);
      lists.set(key, list);
    }
    return list;
  };

  return {
    tex, uniforms: u, sun, level, cellMetres,
    // What the shading pass needs to sample it: see lpvSampleTSL in gpu.js.
    binding: { tex, origin: u.origin, extent: LPV_DIM * cellMetres },

    // Injection slices: 1/N of the cells re-injected each frame.
    get slices() { return sliceCount; },
    set slices(n) { sliceCount = Math.max(1, Math.round(n)); },

    // One frame. shift is lpv.js lpvShift's answer, 'reset' for a jump the
    // volume cannot follow, or null when C0 did not move.
    // batch false submits each kernel on its own, as before - the same list in
    // the same order, so the two can be A/B'd (bxb.gibatch()).
    update(renderer, { shift = null, iterations = 8, fullInject = false, batch = true } = {}) {
      let prefix = '';
      if (shift === 'reset') {
        prefix = 'reset';
        fullInject = true;
      } else if (shift) {
        u.shift.value.set(shift.x, shift.y, shift.z);
        prefix = 'shift';
        fullInject = true;
      }
      const slices = Math.max(1, sliceCount);
      // A full inject is one dispatch with a single slice covering every cell:
      // the old injection is in the wrong cells after a shift.
      u.slices.value = fullInject ? 1 : slices;
      u.slice.value = fullInject ? 0 : frame % slices;
      frame++;
      // Even, so the answer always ends in A, where resolve reads it.
      const n = Math.max(2, iterations + (iterations % 2));
      const list = kernelList(prefix, n);
      if (batch) renderer.compute(list);
      else for (const k of list) renderer.compute(k);
    },

    dispose() {
      tex.dispose();
    }
  };
}
