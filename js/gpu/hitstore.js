import { If, float, vec3, int, uint, ivec3, floor, abs, select, uniform, bitAnd, bitOr, bitXor,
         shiftLeft, shiftRight, packHalf2x16, unpackHalf2x16, vec2, atomicLoad, atomicStore } from 'three/tsl';
import { VOXEL_METRES } from '../core/field.js';
import { LIGHT_VEC4S } from '../core/lights.js';
import { mix32TSL } from './hitcache.js';
import { sunInputs } from './uploads.js';

// --- The hit-light store: one texel, one answer, for the rays ---
//
// A reflection ray or a texel GI ray that lands on a block face needs the
// direct light there - the sun's shadow march and every lamp's - and many
// rays land on the same texel: neighbouring texels' reflections, every patch
// whose rays reach it, the patch refresh. Each of them lit it again, every
// frame. The store keeps the answer per world texel: the first ray to land
// there lights it and writes it; every later one, this frame and the next,
// reads it. It is held until something it depends on changes (gen: the sun,
// the world, the cascades, the lamps - stepped on the CPU by a stamp, as the
// sun store's) and, for what no stamp sees - a sprite's shadow, smoke, the
// leaves - for at most `every` frames: each texel's entry expires on its own
// frame of that cycle, so the refresh is spread evenly and never a spike.
//
// Keyed by the texel (its integer coordinates and face) and a variant - the
// reflections and texel GI light a hit differently (normal-mapped N.L and
// separate sun and lamps; the face normal and one sum) - so they never share
// an entry. A slot is STRIDE u32 words: the check (a hash of the key, the
// generation and the texel's age bucket; never 0), then six values, each a
// half float with the check's low 16 bits as a tag. Two texels writing one
// slot at once can interleave word by word; a reader takes the entry only if
// the check and every tag are its own, so a torn entry is a miss, never
// another texel's light.
//
// It lives at the end of the visible-texel table's queue buffer (gpu/texel.js
// tableWords), which the split kernels bind anyway: a buffer of its own was
// one storage binding too many for the reflection kernel with smoke and
// leaves on.

export const HIT_STORE_SLOTS = 1 << 18;
export const HIT_STORE_STRIDE = 8;         // the check, six values, one spare
const STRIDE = HIT_STORE_STRIDE;
export const HIT_STORE_EVERY = 8;          // frames an entry lives at most (bxb.hitstore)
// Variants: 1 a reflection hit (sun visibility, lamps), 2 a texel GI hit (the sum).
export const HIT_REFLECT = 1, HIT_GI = 2;

// node: the atomic buffer it lives in, from word `offset` on.
export function createHitStore(node, offset, slots = HIT_STORE_SLOTS) {
  return {
    slots, offset, node,
    gen: uniform(1, 'uint'),
    frame: uniform(0, 'uint'),
    every: uniform(HIT_STORE_EVERY, 'uint'),
    on: uniform(1, 'int')
  };
}

// p on a block face (texel-locked), n its axis normal: { base (word index),
// check, tag }.
function keyTSL(store, p, n, variant) {
  const q = float(VOXEL_METRES);
  const t = ivec3(floor(p.sub(n.mul(q.mul(float(0.5)))).div(q))).toVar();
  const face = select(abs(n.x).greaterThan(float(0.5)), select(n.x.greaterThan(float(0)), uint(0), uint(1)),
               select(abs(n.y).greaterThan(float(0.5)), select(n.y.greaterThan(float(0)), uint(2), uint(3)),
                      select(n.z.greaterThan(float(0)), uint(4), uint(5))));
  const h = mix32TSL(bitXor(bitXor(uint(t.x).mul(uint(73856093)), uint(t.y).mul(uint(19349663))),
                            bitXor(uint(t.z).mul(uint(83492791)),
                                   face.mul(uint(8)).add(uint(variant)).mul(uint(2654435761))))).toVar();
  // The texel's age bucket: its own phase in the cycle, from bits the slot
  // does not use, so neighbours expire on different frames.
  const bucket = store.frame.add(shiftRight(h, uint(20))).div(store.every);
  const check = bitOr(mix32TSL(bitXor(h, mix32TSL(bitXor(store.gen, bucket.mul(uint(0x9e3779b9)))))), uint(1)).toVar();
  return { base: int(bitAnd(h, uint(store.slots - 1))).mul(int(STRIDE)).add(int(store.offset)).toVar(), check,
           tag: shiftLeft(bitAnd(check, uint(0xffff)), uint(16)).toVar() };
}

// { ok, a, b } - two vec3 - for the texel at (p, n); ok 0 a miss.
export function hitStoreReadTSL(store, p, n, variant) {
  const k = keyTSL(store, p, n, variant);
  const ok = int(0).toVar();
  const a = vec3(0, 0, 0).toVar(), b = vec3(0, 0, 0).toVar();
  If(store.on.equal(int(1)).and(uint(int(atomicLoad(store.node.element(k.base)))).equal(k.check)), () => {
    const w = [1, 2, 3, 4, 5, 6].map(i => uint(int(atomicLoad(store.node.element(k.base.add(int(i)))))).toVar());
    const tagged = w.map(x => bitAnd(x, uint(0xffff0000)).equal(k.tag)).reduce((x, y) => x.and(y));
    If(tagged, () => {
      const v = w.map(x => unpackHalf2x16(bitAnd(x, uint(0xffff))).x);
      a.assign(vec3(v[0], v[1], v[2]));
      b.assign(vec3(v[3], v[4], v[5]));
      ok.assign(int(1));
    });
  });
  return { ok, a, b };
}

export function hitStoreWriteTSL(store, p, n, variant, a, b) {
  If(store.on.equal(int(1)), () => {
    const k = keyTSL(store, p, n, variant);
    atomicStore(store.node.element(k.base), k.check);
    [a.x, a.y, a.z, b.x, b.y, b.z].forEach((c, i) => {
      atomicStore(store.node.element(k.base.add(int(i + 1))),
                  bitOr(bitAnd(packHalf2x16(vec2(c, float(0))), uint(0xffff)), k.tag));
    });
  });
}

// One vec3 for the texel at (p, n), from the store or - on a miss - from
// compute(), which is then stored. store null: compute() itself.
export function hitStoredTSL(store, p, n, variant, compute) {
  if (!store) return compute();
  const look = hitStoreReadTSL(store, p, n, variant);
  const out = look.a.toVar();
  If(look.ok.equal(int(0)), () => {
    const v = vec3(compute()).toVar();
    out.assign(v);
    hitStoreWriteTSL(store, p, n, variant, v, vec3(0, 0, 0));
  });
  return out;
}

// The store's stamp (createShadowColorNode's hitStamp): what its entries were
// lit by - the sun's static inputs (gpu/sun.js parts.stamp, which carries the
// world's epoch), the light list, texel GI's hit-shadow switch.
export const hitStampFor = ({ sunParts, lights, texelGI }) => () =>
  `${sunParts.stamp ? sunParts.stamp() : sunInputs.epoch}|${lightsStamp(lights)}|` +
  `${texelGI ? texelGI.ab.hitShadows.value : ''}`;

// The light list as a number, for the store's stamp: any lamp moved, turned,
// recoloured, switched or re-ranked changes it.
export function lightsStamp(b) {
  if (!b) return 0;
  let h = 2166136261 ^ b.count.value;
  const v = b.data.array, n = Math.min(b.count.value, b.capacity) * LIGHT_VEC4S;
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  for (let i = 0; i < n; i++) {
    for (const c of [v[i].x, v[i].y, v[i].z, v[i].w]) { f[0] = c; h = Math.imul(h ^ u[0], 16777619); }
  }
  f[0] = b.cutoff.value;
  return (Math.imul(h ^ u[0], 16777619) >>> 0);
}
