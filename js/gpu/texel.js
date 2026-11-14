import * as THREE from 'three/webgpu';
import { tableGroup, frameValues } from './uploads.js';
import { Fn, If, float, vec3, uniform, instanceIndex, storage, floor, abs, vec2, vec4, ivec2, clamp, uniformArray, int, select, textureLoad, screenCoordinate, Discard, uint, atomicStore, atomicAdd,
         atomicMax, atomicOr, atomicLoad, floatBitsToUint, uintBitsToFloat, bitAnd, bitOr, bitXor, shiftRight,
         uvec4, packHalf2x16, unpackHalf2x16 } from 'three/tsl';
import { VOXEL_METRES } from '../core/field.js';
import { terrainSurfaceTSL } from './terrain.js';
import { outsideShadeTSL } from './occlusion.js';
import { MATERIAL_LAYERS } from '../core/materials.js';
import { createRoomMeter } from './roommeter.js';
import { mix32TSL } from './hitcache.js';
import { createHitStore, HIT_STORE_SLOTS, HIT_STORE_STRIDE } from './hitstore.js';
import { TABLE_PROBES } from '../core/texelcache.js';

// ---------------------------------------------------------------------------
// Texel-rate shading: the terrain shaded once per texel, not once per pixel
//
// Everything the terrain shades is texel-locked - one texel, one answer - so
// every pixel inside a texel runs the identical marches and gets the identical
// colour. At play zoom a 12.5 cm texel covers many pixels; profiling put the
// torch, the mirrors and the reflection at ~3 ms of the frame, almost all of
// it repeats.
//
// So the terrain is drawn twice:
//
//   1  a LOW-RES pass (1/scale per axis) with the full shading, writing
//      vec4(colour, texel id) to a float target. Each low-res pixel is one
//      sample of whatever texel lies under it.
//   2  the normal full-res pass, which for each pixel computes only its own
//      texel id (cheap: the lock and a hash), and takes the colour of the
//      first sample in the 3x3 around it with the same id. Only a pixel with
//      no matching sample - a texel smaller than the sample spacing, at a far
//      zoom or a grazing angle - shades itself, in full.
//
// Nothing is approximated: a match means the sample's texel IS this pixel's
// texel, and a texel's colour is the same everywhere in it, bit for bit. No
// atlas, no paging, no invalidation - the cache is rebuilt every frame, so it
// is always the current frame's answer.
//
// The id is the world texel coordinate, each axis mod 128, plus the face (6):
// 7 + 7 + 7 + 3 bits, exact as a float. It only has to tell apart the handful
// of texels in a 3x3 window of samples, which are never 16 m apart.
// Offset by 2 so no id is 0 (cleared) or 1 (the sky's alpha).
export const TEXEL_ID_OFFSET = 2;

export const texelIdTSL = (p, n) => {
  // Half a texel inside the face, so the along-normal floor is not taken on
  // the plane itself.
  const t = p.sub(n.mul(float(VOXEL_METRES * 0.5))).div(float(VOXEL_METRES)).floor();
  const wrap = v => v.sub(floor(v.div(float(128))).mul(float(128)));
  const face = select(abs(n.x).greaterThan(float(0.5)), select(n.x.greaterThan(float(0)), float(0), float(1)),
               select(abs(n.y).greaterThan(float(0.5)), select(n.y.greaterThan(float(0)), float(2), float(3)),
                      select(n.z.greaterThan(float(0)), float(4), float(5))));
  return wrap(t.x).add(wrap(t.y).mul(float(128))).add(wrap(t.z).mul(float(16384)))
    .add(face.mul(float(2097152))).add(float(TEXEL_ID_OFFSET));
};

// A glass kernel under the table: dispatched for the glass list's count
// (times its lanes).
export const tableGlassKernelTSL = (split, k, lanes = 1) => {
  if (split.table) k.dispatchSize = (lanes > 1 ? split.glassArgs3 : split.glassArgs).attr;
  return k;
};

// A box face's axis normal from its texel id (abs, the offset still on).
export const faceNormalTSL = id => {
  const f = floor(id.sub(float(TEXEL_ID_OFFSET)).div(float(2097152)));
  return select(f.lessThan(float(0.5)), vec3(1, 0, 0), select(f.lessThan(float(1.5)), vec3(-1, 0, 0),
         select(f.lessThan(float(2.5)), vec3(0, 1, 0), select(f.lessThan(float(3.5)), vec3(0, -1, 0),
         select(f.lessThan(float(4.5)), vec3(0, 0, 1), vec3(0, 0, -1))))));
};

// This terrain fragment's texel id - the same lock the shading takes.
export const terrainTexelIdTSL = () => {
  const surf = terrainSurfaceTSL();
  return select(surf.on, slopeTexelIdTSL(surf), texelIdTSL(surf.p, surf.n));
};

// A slope texel's id: its rows are finer than the voxels, so two can share a
// voxel - it is named by its block and indices instead, face 6. Only the
// 3 x 3 window has to tell texels apart, so each block coordinate keeps its
// low bits: x (2) with the row (5), z (3) with the column (4), y (7).
function slopeTexelIdTSL(surf) {
  const wrap = (v, m) => v.sub(floor(v.div(float(m))).mul(float(m)));
  const b = surf.block;
  const fx = wrap(b.x, 4).mul(float(32)).add(surf.slopeTexel.y);
  const fz = wrap(b.z, 8).mul(float(16)).add(surf.slopeTexel.x);
  return fx.add(wrap(b.y, 128).mul(float(128))).add(fz.mul(float(16384)))
    .add(float(6 * 2097152)).add(float(TEXEL_ID_OFFSET));
}

// The low-res pass's output: the full shading and the texel it belongs to.
// negate: glass under split shading - its id negative, so the split kernels
// pass it through instead of shading it as terrain.
export function createTexelCacheWriteNode(shade, negate = false) {
  const id = terrainTexelIdTSL();
  return vec4(shade, negate ? id.negate() : id);
}

// --- Split shading ---
//
// With it on, the low-res pass shades nothing: it writes each sample's
// texel-locked position and texel id (createTexelGBufferNode), and compute
// kernels light the samples a feature at a time (createShadowColorNode,
// splitKernels): base (surface, AO, GI, sun), point lights, mirror light,
// and a final kernel (reflections, the clamped highlights). Each kernel is a
// small program, so it runs with the registers it needs - one shader holding
// every feature ran every texel at the occupancy of its worst path. The
// lookup reads their result. A slope's plane or a glowing model cube writes
// no sample (id 0): its pixels fall to the miss pass, which keeps the full
// shader. capacity: samples (vec4 each) per buffer.
//
// The misses too (2026-10-14): a full-res pass of the same G-buffer materials
// (idRT) gives every pixel its texel; a resolve kernel finds each pixel's
// low-res sample, or appends the pixel to a miss list, and records which
// sample colours it (pixelSample). The kernels light low-res samples and
// misses as one index range - lowCap samples, then missCap misses - and the
// terrain's own draw reads its pixel's sample (createSplitReadNode). No
// prepass, no stencil, no miss pass: the full shaders are never compiled.
//
// The visible-texel table (table > 0, 2026-11-11): no low-res pass. The
// resolve hashes every pixel's texel into a table of `table` slots and lists
// each texel once (createTableResolve); the kernels go one thread a listed
// texel, by an indirect dispatch, and keep its buffers at its slot. So each
// texel on screen is shaded once a frame, where the low-res samples (half a
// texel apart) shaded it ~4 times. The lookup is the same pixelSample.
export function createSplitShading(lowCap, missCap, fullCap, half = true, table = 0) {
  const capacity = table || lowCap + missCap;
  const buffer = (name, n = capacity, type = 'vec4', item = 4) => {
    const attr = new THREE.StorageBufferAttribute(type === 'vec4' ? n
      : type === 'uvec4' ? new Uint32Array(n * 4) : new Int32Array(n), item);
    return { attr, rw: storage(attr, type, n).setName(name),
             ro: storage(attr, type, n).toReadOnly().setName(name + 'Read') };
  };
  const countAttr = table ? null : new THREE.StorageBufferAttribute(new Uint32Array(1), 1);
  // Table mode: one atomic buffer, as texel GI's queue (tableWords) - the
  // kernels read it read-only.
  const words = table ? tableWords(table) : null;
  const queueAttr = table ? new THREE.StorageBufferAttribute(new Uint32Array(words.total), 1) : null;
  const args = () => {
    const attr = new THREE.IndirectStorageBufferAttribute(new Uint32Array([1, 1, 1]), 1);
    return { attr, rw: storage(attr, 'uint', 3).setName('bxbSplitArgs') };
  };
  const queue = table ? storage(queueAttr, 'uint', words.total).toAtomic().setName('bxbSplitQueue') : null;
  return {
    capacity, lowCap, missCap, fullCap, table, words,
    // The direct light at the rays' hits, per world texel (gpu/hitstore.js),
    // at the queue's end. Table mode only.
    hitStore: table ? createHitStore(queue, words.hits) : null,
    result: buffer('bxbSplitResult'),   // colour so far, texel id (0: none)
    // The light on the inside's surfaces, for revealed faces (gpu/roommeter.js).
    room: createRoomMeter(),
    // Between kernels: the ambient a reflection blends with, and highlights
    // so far (clamped at the end) - six half floats in one uvec4. fp16 is
    // ~0.05%, a tenth of an 8-bit step before Fresnel scales it down; texel-
    // locked, so flat per texel - nothing to band.
    // half false: full precision, two vec4 buffers (bxb.splithalf()).
    half,
    light: half ? buffer('bxbSplitLight', capacity, 'uvec4') : null,
    env: half ? null : buffer('bxbSplitEnv'),
    spec: half ? null : buffer('bxbSplitSpec'),
    missSamples: table ? null : buffer('bxbSplitMisses', missCap),   // a miss's position, id
    pixelSample: buffer('bxbSplitPixel', fullCap, 'int', 1),    // per pixel; -1 none
    missCount: table ? null : { attr: countAttr,
                 rw: storage(countAttr, 'uint', 1).toAtomic().setName('bxbSplitMissCount'),
                 ro: storage(countAttr, 'uint', 1).toReadOnly().setName('bxbSplitMissCountRead') },
    // Atomic in every kernel that binds it - the hit-light store in it is
    // written by the reflection and GI kernels, which read the list too.
    queue: table ? { attr: queueAttr, rw: queue } : null,
    // The kernels' workgroups (createTableResolve's args kernel writes them),
    // and the sun store's refresh kernel's.
    args: table ? args() : null,
    refreshArgs: table ? args() : null,
    // The glass kernels' workgroups: one thread a glass texel (the glass
    // list) - and three, a lane a colour, for the view through with dispersion.
    glassArgs: table ? args() : null,
    glassArgs3: table ? args() : null,
    // The sun store (createSunRefreshKernel): per slot, held across frames -
    // the texel's key, the generation it was marched in, and its static sun
    // visibility as three halves. sunGen: this frame's generation (bumped
    // whenever anything the march reads changes); storeOn 0 refreshes every
    // texel every frame (bxb.sunstore(), the A/B).
    store: table ? buffer('bxbSunStore', table, 'uvec4') : null,

    sunGen: uniform(1, 'uint'),
    storeOn: uniform(1, 'int'),
    width: uniform(int(1)),             // low-res
    count: uniform(int(0)),             // low-res samples this frame
    fullWidth: uniform(int(1)),
    fullCount: uniform(int(0)),
    viewPos: frameValues.viewPos,
    rt: null,                           // the low-res target's texture, set by main.js
    idRT: null                          // the full-res one's
  };
}

// Table mode's queue (m slots), in words: count, the list (a slot per listed
// texel), the keys (a word a slot), the samples (a listed texel's position
// and id at its slot, four words, bitcast), the sun refresh count and list,
// the glass count and list, the hit-light store (gpu/hitstore.js).
export const tableWords = m => ({ list: 1, keys: 1 + m, samples: 1 + 2 * m,
                                  refreshCount: 1 + 6 * m, refresh: 2 + 6 * m,
                                  glassCount: 2 + 7 * m, glass: 3 + 7 * m, hits: 3 + 8 * m,
                                  total: 3 + 8 * m + HIT_STORE_SLOTS * HIT_STORE_STRIDE });

// A queue word in a split kernel - through the atomic binding, the only one.
export const tableWordTSL = (split, i) => uint(int(atomicLoad(split.queue.rw.element(i))));

// A listed texel's sample, read in a split kernel.
export const tableSampleTSL = (split, slot) => {
  const at = k => uintBitsToFloat(tableWordTSL(split, int(split.words.samples + k).add(slot.mul(int(4)))));
  return vec4(at(0), at(1), at(2), at(3));
};

// A split kernel's per-sample buffers: addResult adds to the colour so far;
// readLight / writeLight / addSpec the packed ambient and highlights
// (createSplitShading: light).
export function splitLightTSL(split) {
  const addResult = (idx, c) => {
    const r = split.result.rw.element(idx).toVar();
    split.result.rw.element(idx).assign(vec4(r.xyz.add(c), r.w));
  };
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
  return { addResult, readLight, writeLight, addSpec };
}

// Per material, its slope planes' first layers + 1 (0: the side stretched):
// x high, y low, z medium - render.js fillInstances' modelInfo.x, as a table.
export const slopeLayerTable = uniformArray(MATERIAL_LAYERS.faces.map(f =>
  new THREE.Vector4(f.slope[0] + 1, f.slope[1] + 1, f.slope[2] + 1, 0)), 'vec4').setName('bxbSlopeLayers').setGroup(tableGroup);

// The resolve, before the kernels: [clear the miss count, per pixel find
// its sample]. cache: texelCache (scale, size).
export function createSplitResolve(split, cache) {
  if (split.table) return createTableResolve(split);
  const clear = Fn(() => { atomicStore(split.missCount.rw.element(int(0)), uint(0)); })()
    .compute(1).setName('splitMissClear');
  const resolve = Fn(() => {
    const idx = int(instanceIndex).toVar();
    If(idx.lessThan(split.fullCount), () => {
      const px = vec2(float(idx.mod(split.fullWidth)), float(idx.div(split.fullWidth))).toVar();
      const s = textureLoad(split.idRT, ivec2(px)).toVar();
      const out = int(-1).toVar();
      If(abs(s.w).greaterThan(float(1.5)), () => {
        // The lookup's own 3x3, on the low-res ids (glass ones negative too).
        const lc = floor(px.add(float(0.5)).mul(cache.scale)).toVar();
        const hi = cache.size.sub(float(1));
        for (const [dx, dy] of TEXEL_CACHE_TAPS) {
          If(out.lessThan(int(0)), () => {
            const c = clamp(lc.add(vec2(dx, dy)), vec2(0, 0), hi);
            const i = int(c.y).mul(split.width).add(int(c.x)).toVar();
            If(i.lessThan(split.count), () => {
              If(textureLoad(split.rt, ivec2(c)).w.equal(s.w), () => { out.assign(i); });
            });
          });
        }
        If(out.lessThan(int(0)), () => {
          const slot = int(atomicAdd(split.missCount.rw.element(int(0)), uint(1))).toVar();
          If(slot.lessThan(int(split.missCap)), () => {
            split.missSamples.rw.element(slot).assign(s);
            out.assign(int(split.lowCap).add(slot));
          }).Else(() => {
            // The list is full: the nearest low-res sample, not nothing.
            const c = clamp(lc, vec2(0, 0), hi);
            out.assign(int(c.y).mul(split.width).add(int(c.x)));
          });
        });
      });
      split.pixelSample.rw.element(idx).assign(out);
    });
  })().compute(split.fullCap).setName('splitResolve');
  return [clear, resolve];
}

// --- The visible-texel table's resolve ---
//
// A texel's key is a hash of its G-buffer sample's bits - the locked position
// and the id. texelLockTSL gives every fragment of a texel the identical
// floats, so every pixel of it makes the same key; -0 is folded to +0 first.
// Two independent hashes: one picks the home slot, the other is the key kept
// in it (31 bits, never 0), so two texels are one only if both collide and
// their homes lie within a probe run - never seen at ~10^5 texels a frame.
// A pixel that finds no slot (a full table, or a key carried past its run)
// is dropped: the lookup borrows a neighbour's colour.
//
// The kernels, in one submit:
//   clear   the keys and the count
//   insert  per pixel: its key into the table by ordered linear probing - an
//           atomicMax at each slot; the smaller key carries on to the next -
//           so the table ends the same whatever order the threads ran in
//   claim   per pixel: the slot holding its key. The first pixel there (the
//           key's top bit, taken by atomicOr) lists the slot and writes the
//           sample; every pixel records its slot in pixelSample
//   args    the kernels' workgroups for the listed count (an indirect dispatch)
//   bind    per listed terrain texel: its sun store entry - another key, an
//           older generation or the store off lists it for refresh. A glass
//           texel goes on the glass list instead
//   refreshArgs, glassArgs  the refresh and glass kernels' workgroups
// Threads per workgroup of the split kernels (three's default).
export const SPLIT_TABLE_WORKGROUP = 64;
const CLAIMED = 0x80000000;

function texelKeyTSL(s, slots) {
  const bits = ['x', 'y', 'z', 'w'].map(c => floatBitsToUint(select(s[c].equal(float(0)), float(0), s[c])));
  const chain = seed => {
    let h = mix32TSL(bitXor(bits[3], uint(seed)));
    for (const b of [bits[2], bits[1], bits[0]]) h = mix32TSL(bitXor(b, h));
    return h;
  };
  return { home: int(bitAnd(chain(0x9e3779b9), uint(slots - 1))).toVar(),
           key: bitOr(shiftRight(chain(0x85ebca6b), uint(1)), uint(1)).toVar() };
}

function createTableResolve(split) {
  const q = split.queue.rw, W = split.words;
  const m = split.table, mask = int(m - 1);
  const keyAt = j => q.element(int(1 + m).add(j));
  const pixel = idx => textureLoad(split.idRT, ivec2(idx.mod(split.fullWidth), idx.div(split.fullWidth)));
  const clear = Fn(() => {
    const i = int(instanceIndex);
    atomicStore(keyAt(i), uint(0));
    If(i.equal(int(0)), () => {
      atomicStore(q.element(int(0)), uint(0));
      atomicStore(q.element(int(W.refreshCount)), uint(0));
      atomicStore(q.element(int(W.glassCount)), uint(0));
    });
  })().compute(m).setName('splitTableClear');
  const insert = Fn(() => {
    const idx = int(instanceIndex).toVar();
    If(idx.lessThan(split.fullCount), () => {
      const s = pixel(idx).toVar();
      If(abs(s.w).greaterThan(float(1.5)), () => {
        const k = texelKeyTSL(s, m);
        const carry = k.key.toVar(), j = k.home.toVar();
        for (let i = 0; i < TABLE_PROBES; i++) {
          If(carry.notEqual(uint(0)), () => {
            // Through int(): a uint atomic's old value is lost in uint() (texelgi.js).
            const old = uint(int(atomicMax(keyAt(j), carry))).toVar();
            // Empty, or this key already: done. A smaller key: it was put
            // out, and carries on. A larger: this key carries on.
            If(old.equal(uint(0)).or(old.equal(carry)), () => { carry.assign(uint(0)); })
              .ElseIf(old.lessThan(carry), () => { carry.assign(old); });
            j.assign(bitAnd(j.add(int(1)), mask));
          });
        }
      });
    });
  })().compute(split.fullCap).setName('splitTableInsert');
  const claim = Fn(() => {
    const idx = int(instanceIndex).toVar();
    If(idx.lessThan(split.fullCount), () => {
      const s = pixel(idx).toVar();
      const out = int(-1).toVar();
      If(abs(s.w).greaterThan(float(1.5)), () => {
        const k = texelKeyTSL(s, m);
        const j = k.home.toVar();
        for (let i = 0; i < TABLE_PROBES; i++) {
          If(out.lessThan(int(0)), () => {
            const held = bitAnd(uint(int(atomicLoad(keyAt(j)))), uint(CLAIMED - 1));
            If(held.equal(k.key), () => { out.assign(j); });
            j.assign(bitAnd(j.add(int(1)), mask));
          });
        }
        If(out.greaterThanEqual(int(0)), () => {
          // Negative as an int: the top bit was set - claimed already.
          const was = int(atomicOr(keyAt(out), uint(CLAIMED))).toVar();
          If(was.greaterThanEqual(int(0)), () => {
            const at = int(atomicAdd(q.element(int(0)), uint(1))).toVar();
            atomicStore(q.element(int(1).add(at)), uint(out));
            ['x', 'y', 'z', 'w'].forEach((c, k) => {
              atomicStore(q.element(int(W.samples + k).add(out.mul(int(4)))), floatBitsToUint(s[c]));
            });
          });
        });
      });
      split.pixelSample.rw.element(idx).assign(out);
    });
  })().compute(split.fullCap).setName('splitTableClaim');
  const argsFor = (at, out, name, lanes = 1) => Fn(() => {
    const raw = uint(int(atomicLoad(q.element(int(at)))));
    const n = (lanes > 1 ? raw.mul(uint(lanes)) : raw).toVar();
    out.rw.element(int(0)).assign(n.add(uint(SPLIT_TABLE_WORKGROUP - 1)).div(uint(SPLIT_TABLE_WORKGROUP)));
    out.rw.element(int(1)).assign(uint(1));
    out.rw.element(int(2)).assign(uint(1));
  })().compute(1).setName(name);
  const bind = Fn(() => {
    const t = int(instanceIndex).toVar();
    If(t.lessThan(int(atomicLoad(q.element(int(0))))), () => {
      const slot = int(atomicLoad(q.element(int(1).add(t)))).toVar();
      const key = bitAnd(uint(int(atomicLoad(keyAt(slot)))), uint(CLAIMED - 1)).toVar();
      const id = uintBitsToFloat(uint(int(atomicLoad(q.element(int(W.samples + 3).add(slot.mul(int(4))))))));
      const held = split.store.ro.element(slot).toVar();
      const stale = split.storeOn.equal(int(0)).or(held.x.notEqual(key)).or(held.y.notEqual(split.sunGen));
      // Terrain only: glass is lit live, by its own kernels.
      If(id.greaterThan(float(1.5)).and(stale), () => {
        const at = int(atomicAdd(q.element(int(W.refreshCount)), uint(1))).toVar();
        atomicStore(q.element(int(W.refresh).add(at)), uint(slot));
      });
      // Glass, listed apart: the glass kernels' heavy paths then run in warps
      // of glass alone, not a few lanes of each warp of the whole list.
      If(id.lessThan(float(-1.5)), () => {
        const at = int(atomicAdd(q.element(int(W.glassCount)), uint(1))).toVar();
        atomicStore(q.element(int(W.glass).add(at)), uint(slot));
      });
    });
  })().compute(m).setName('splitSunBind');
  bind.dispatchSize = split.args.attr;
  return [clear, insert, claim, argsFor(0, split.args, 'splitTableArgs'), bind,
          argsFor(W.refreshCount, split.refreshArgs, 'splitSunRefreshArgs'),
          argsFor(W.glassCount, split.glassArgs, 'splitGlassArgs'),
          argsFor(W.glassCount, split.glassArgs3, 'splitGlassArgs3', 3)];
}

// The sun store's refresh (one thread a texel bind listed): the static part
// of its sun visibility (gpu/sun.js parts.static - the field march, the far
// march and the sun depth map; cards, smoke and leaves stay live), written
// with its key and this generation. A texel facing away from the sun holds 0.
// geoAt: shade.js's sample geometry; glass: the visibility is a colour.
// The first of the split kernels, so the base kernel reads this frame's.
export function createSunRefreshKernel({ split, geoAt, sunStatic, sunDir, glass }) {
  const W = split.words, q = i => tableWordTSL(split, i);
  const k = Fn(() => {
    const t = int(instanceIndex).toVar();
    If(t.lessThan(int(q(int(W.refreshCount)))), () => {
      const slot = int(q(int(W.refresh).add(t))).toVar();
      const geo = geoAt(tableSampleTSL(split, slot).toVar());
      const v = vec3(0, 0, 0).toVar();
      If(geo.n.dot(sunDir).greaterThan(float(0)), () => {
        const st = sunStatic(geo.p, geo.n).toVar();
        v.assign(glass ? st.xyz.mul(st.w) : vec3(st, st, st));
      });
      const key = bitAnd(q(int(W.keys).add(slot)), uint(CLAIMED - 1));
      split.store.rw.element(slot).assign(uvec4(key, split.sunGen, packHalf2x16(vec2(v.x, v.y)),
                                                packHalf2x16(vec2(v.z, float(0)))));
    });
  })().compute(split.table).setName('splitSunRefresh');
  k.dispatchSize = split.refreshArgs.attr;
  return k;
}

// A texel's stored static sun visibility (createSunRefreshKernel), in the
// type the sun shadow answers in.
export function sunStoreReadTSL(split, slot, glass) {
  const held = split.store.ro.element(slot).toVar();
  const a = unpackHalf2x16(held.z), b = unpackHalf2x16(held.w);
  return glass ? vec3(a.x, a.y, b.x) : a.x;
}

// A split kernel's thread over the table (gpu/shade.js): a listed texel,
// its buffers at its slot - run(idx, sample, lane). glass: over the glass
// list. lanes: threads a texel, run with lane 0..lanes-1 (1: lane null).
export function splitTableThreadTSL(split, run, glass = false, lanes = 1) {
  const [count, list] = glass ? [split.words.glassCount, split.words.glass] : [0, 1];
  const t = int(instanceIndex).toVar();
  const n = int(tableWordTSL(split, int(count)));
  If(t.lessThan(lanes > 1 ? n.mul(int(lanes)) : n), () => {
    const idx = int(tableWordTSL(split, int(list).add(lanes > 1 ? t.div(int(lanes)) : t))).toVar();
    run(idx, tableSampleTSL(split, idx).toVar(), lanes > 1 ? t.mod(int(lanes)).toVar() : null);
  });
}

// The terrain's (and glass's) draw under split shading: its pixel's sample -
// or, where this fragment is another texel than the pixel's centre (an MSAA
// edge), its own in the 3x3 low-res window; failing both, the pixel's.
export function createSplitReadNode(cache, split) {
  if (split.table) return createTableReadNode(split);
  return Fn(() => {
    const id = terrainTexelIdTSL().toVar();
    const px = floor(screenCoordinate.xy).toVar();
    const si = split.pixelSample.ro.element(int(px.y).mul(split.fullWidth).add(int(px.x))).toVar();
    const out = vec3(0, 0, 0).toVar();
    const found = float(0).toVar();
    If(si.greaterThanEqual(int(0)), () => {
      const r = split.result.ro.element(si).toVar();
      out.assign(r.xyz);
      If(r.w.equal(id), () => { found.assign(float(1)); });
    });
    const lc = floor(screenCoordinate.xy.mul(cache.scale)).toVar();
    const hi = cache.size.sub(float(1));
    for (const [dx, dy] of TEXEL_CACHE_TAPS) {
      If(found.equal(float(0)), () => {
        const c = clamp(lc.add(vec2(dx, dy)), vec2(0, 0), hi);
        const s = split.result.ro.element(int(c.y).mul(split.width).add(int(c.x))).toVar();
        If(s.w.equal(id), () => { out.assign(s.xyz); found.assign(float(1)); });
      });
    }
    return out;
  })();
}

// Table mode: the same, with the 3x3 window over the neighbouring pixels'
// slots. Failing a match, the pixel's own colour; dropped as well (no slot),
// the first neighbour that has one.
function createTableReadNode(split) {
  return Fn(() => {
    const id = terrainTexelIdTSL().toVar();
    const px = ivec2(floor(screenCoordinate.xy)).toVar();
    const hi = ivec2(split.fullWidth.sub(int(1)), split.fullCount.div(split.fullWidth).sub(int(1)));
    const out = vec3(0, 0, 0).toVar();
    const found = float(0).toVar();
    const have = float(0).toVar();
    for (const [dx, dy] of TEXEL_CACHE_TAPS) {
      If(found.equal(float(0)), () => {
        const c = clamp(px.add(ivec2(dx, dy)), ivec2(0, 0), hi);
        const si = split.pixelSample.ro.element(c.y.mul(split.fullWidth).add(c.x)).toVar();
        If(si.greaterThanEqual(int(0)), () => {
          const r = split.result.ro.element(si).toVar();
          If(r.w.equal(id), () => { out.assign(r.xyz); found.assign(float(1)); })
            .ElseIf(have.equal(float(0)), () => { out.assign(r.xyz); have.assign(float(1)); });
        });
      });
    }
    return out;
  })();
}

// The low-res pass's output under split shading: position and texel id -
// negative for glass, which has kernels of its own.
export function createTexelGBufferNode(glass = false, legacy = false) {
  return Fn(() => {
    const surf = terrainSurfaceTSL();
    if (legacy) {
      // A08//00a: slopes and glowing cubes write nothing (they are misses).
      const id = texelIdTSL(surf.p, surf.n);
      return select(surf.on.or(surf.emissive), vec4(0, 0, 0, 0), vec4(surf.p, glass ? id.negate() : id));
    }
    const id = select(surf.on, slopeTexelIdTSL(surf), texelIdTSL(surf.p, surf.n));
    return vec4(surf.p, glass ? id.negate() : id);
  })();
}

// The full-res lookup. cache: { tex, scale (vec2 low/full), size (vec2 low) }.
//
// A pixel with no matching sample is DISCARDED here, not shaded: the full
// shading lives in a separate miss pass (createTexelMissNode), drawn after,
// where a stencil test keeps it off every pixel this pass filled. It cannot
// live in a branch of this shader - measured, a never-taken branch holding the
// full shading cost 2.5 ms of 3.5, because a GPU sizes a shader's registers for
// its worst path, and every pixel then runs with that fewer in flight.
const TEXEL_CACHE_TAPS = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                          [1, 1], [-1, 1], [1, -1], [-1, -1]];

export function createTexelCacheLookupNode(cache, split = null) {
  // Under split shading the colours are in the kernels' result buffer.
  const load = c => (split
    ? split.result.ro.element(int(c.y).mul(split.width).add(int(c.x)))
    : textureLoad(cache.tex, ivec2(c)));
  return Fn(() => {
    const id = terrainTexelIdTSL().toVar();
    const lc = floor(screenCoordinate.xy.mul(cache.scale)).toVar();
    const hi = cache.size.sub(float(1));
    const out = vec3(0, 0, 0).toVar();
    const found = float(0).toVar();
    // Unrolled on purpose: nine loads, the centre first, each skipped once a
    // match is in hand.
    for (const [dx, dy] of TEXEL_CACHE_TAPS) {
      If(found.equal(float(0)), () => {
        const c = clamp(lc.add(vec2(dx, dy)), vec2(0, 0), hi);
        const s = load(c).toVar();
        If(s.w.equal(id), () => { out.assign(s.xyz); found.assign(float(1)); });
      });
    }
    If(found.equal(float(0)), () => { Discard(); });
    return out;
  })();
}

// The miss pass's colour: the full shading - or, in the miss view, red, so
// which pixels shade themselves can be seen.
export function createTexelMissNode(cache, shade) {
  return cache.showMisses ? vec3(1, 0, 0) : shade;
}

// Swaps the terrain to a node material whose colour is its own albedo modulated
// by the marched shadow term, keeping the original so it can be restored.
export function applyShadowMaterial(mesh, shadowNode, albedo = null) {
  const mat = createShadowMaterial(mesh, shadowNode, albedo);
  if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
  mesh.material = mat;
  return mat;
}

// The same material, built but not assigned - so it can be compiled off-screen
// while the current one keeps drawing.
// albedo: the surface colour node - for the terrain, its per-block layer with
// whiteworld mixed in (render.js). White when null.
export function createShadowMaterial(mesh, shadowNode, albedo = null) {
  if (!mesh.userData.originalMaterial) mesh.userData.originalMaterial = mesh.material;
  const mat = new THREE.MeshBasicNodeMaterial();
  // .rgb, because the shadow node is a vec3 now that a coloured light adds to it.
  // Per-instance tint (arena greying) lives in instanceColor, so it has to be
  // carried across or battle mode loses its reachable-tile shading.
  // Darkened outside a building the player is in - the view only, after the
  // shading (gpu/occlusion.js outsideShadeTSL).
  mat.colorNode = (albedo || vec3(1, 1, 1)).mul(shadowNode).mul(outsideShadeTSL());
  return mat;
}

export function restoreOriginalMaterial(mesh) {
  if (!mesh.userData.originalMaterial) return false;
  if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
  mesh.material = mesh.userData.originalMaterial;
  return true;
}
