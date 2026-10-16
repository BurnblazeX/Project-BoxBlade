import * as THREE from 'three/webgpu';
import { Fn, float, vec3, uniform, instanceIndex, storage, floor, abs, clamp, int, select, sqrt, uint, ivec3, bitXor, shiftRight, bitAnd, bitOr, shiftLeft, atomicMax, atomicLoad, atomicStore } from 'three/tsl';
import { VOXEL_METRES } from '../core/boxgrid.js';

// --- The hit lighting cache ---
//
// A reflection (or a look through glass) that lands on a terrain texel needs
// that texel's DIFFUSE light, which does not depend on where it is seen from.
// The terrain shading works it out for every texel on screen anyway, so it
// writes it here, keyed by world texel and face; a hit reads last frame's and,
// on a match, skips its whole light batch. Two halves: one written this frame
// (cleared first), the other read - so what a hit sees is fixed for the frame.
// Off-screen texels (behind the camera, under things) are never written, and
// are shaded as before. Lag: one frame on moving lights.
//
// Outside compute, three makes only ATOMIC storage writable, and there is no
// compare-and-swap - so a slot is two u32 words, each taken by atomicMax:
//   0  the texel's check (a second hash of its key, never 0): of texels that
//      collide on a slot, the highest check owns it;
//   1  that check's top 8 bits, then its colour as 8:8:8, each sqrt(c / 4).
// A hit takes the slot only if both match its own check - a collision is a
// miss (shaded), not a wrong colour.

export const HIT_CACHE_SLOTS = 1 << 18;

// Past this along a reflection or through-glass ray, a hit is lit by point
// lights without their shadow march (the sun keeps its own).
export const HIT_FAR_METRES = 12;

const HIT_CACHE_HALF = HIT_CACHE_SLOTS * 2;   // words

const HIT_CACHE_RANGE = 4;                     // colours above it clamp

export function createHitCache() {
  const attr = new THREE.StorageBufferAttribute(new Uint32Array(HIT_CACHE_HALF * 2), 1);
  const node = storage(attr, 'uint', HIT_CACHE_HALF * 2).toAtomic().setName('bxbHitCache');
  // The same buffer, read-only, for every shader that only reads it: an
  // atomic (read-write) binding counts as a side effect and costs the shader
  // its early depth and stencil tests - measured +3.7 ms on the miss pass.
  const read = storage(attr, 'uint', HIT_CACHE_HALF * 2).toReadOnly().setName('bxbHitCacheRead');
  const writeHalf = uniform(int(0));
  const clear = Fn(() => {
    atomicStore(node.element(int(instanceIndex).add(writeHalf.mul(int(HIT_CACHE_HALF)))), uint(0));
  })().compute(HIT_CACHE_HALF).setName('hitCacheClear');
  return { attr, node, read, writeHalf, clear };
}

// A 32-bit integer mix (the hash's finaliser), for keys and per-texel seeds.
export const mix32TSL = h => {
  const a = bitXor(h, shiftRight(h, uint(16))).mul(uint(0x7feb352d)).toVar();
  const b = bitXor(a, shiftRight(a, uint(15))).mul(uint(0x846ca68b)).toVar();
  return bitXor(b, shiftRight(b, uint(16)));
};

// p on a box face (texel-locked), n axis-exact: { slot (word index in a
// half), index (the slot itself), check (u32, never 0), tag (its top 8 bits,
// shifted into place) }. slots: the table's size, a power of two.
function hitCacheKeyTSL(p, n, slots = HIT_CACHE_SLOTS) {
  const q = float(VOXEL_METRES);
  const t = ivec3(floor(p.sub(n.mul(q.mul(float(0.5)))).div(q))).toVar();
  const face = select(abs(n.x).greaterThan(float(0.5)), select(n.x.greaterThan(float(0)), uint(0), uint(1)),
               select(abs(n.y).greaterThan(float(0.5)), select(n.y.greaterThan(float(0)), uint(2), uint(3)),
                      select(n.z.greaterThan(float(0)), uint(4), uint(5))));
  const mix32 = mix32TSL;
  const h = mix32(bitXor(bitXor(uint(t.x).mul(uint(73856093)), uint(t.y).mul(uint(19349663))),
                         bitXor(uint(t.z).mul(uint(83492791)), face.mul(uint(2654435761))))).toVar();
  const check = bitOr(mix32(bitXor(h, uint(0x9e3779b9))), uint(1)).toVar();
  const index = int(bitAnd(h, uint(slots - 1))).toVar();
  return { slot: index.mul(int(2)), index,
           check, tag: shiftLeft(shiftRight(check, uint(24)), uint(24)) };
}

const hitCacheEncodeTSL = c => {
  const e = sqrt(clamp(c.div(float(HIT_CACHE_RANGE)), vec3(0, 0, 0), vec3(1, 1, 1))).mul(float(255)).add(float(0.5));
  return bitOr(bitOr(shiftLeft(uint(e.x), uint(16)), shiftLeft(uint(e.y), uint(8))), uint(e.z));
};

const hitCacheDecodeTSL = w => {
  const ch = k => float(bitAnd(shiftRight(w, uint(k)), uint(255))).div(float(255));
  const v = vec3(ch(16), ch(8), ch(0));
  return v.mul(v).mul(float(HIT_CACHE_RANGE));
};

// This texel's diffuse colour into the half being written.
export function hitCacheWriteTSL(cache, p, n, colour) {
  const key = hitCacheKeyTSL(p, n);
  const base = key.slot.add(cache.writeHalf.mul(int(HIT_CACHE_HALF))).toVar();
  atomicMax(cache.node.element(base), key.check);
  atomicMax(cache.node.element(base.add(int(1))), bitOr(key.tag, hitCacheEncodeTSL(colour)));
}

// Last frame's colour for the texel at p, n: { ok, colour }. atomic: through
// the read-write binding (a shader that also writes the cache - a pass may
// not bind the buffer both ways), else the read-only one.
export function hitCacheReadTSL(cache, p, n, atomic) {
  const key = hitCacheKeyTSL(p, n);
  const base = key.slot.add(int(HIT_CACHE_HALF).sub(cache.writeHalf.mul(int(HIT_CACHE_HALF)))).toVar();
  const load = i => (atomic ? atomicLoad(cache.node.element(i)) : cache.read.element(i));
  const w0 = load(base).toVar();
  const w1 = load(base.add(int(1))).toVar();
  const ok = w0.equal(key.check).and(bitAnd(w1, uint(0xff000000)).equal(key.tag));
  return { ok, colour: hitCacheDecodeTSL(w1) };
}
