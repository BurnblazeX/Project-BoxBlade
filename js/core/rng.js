// --- Seeded randomness: the same answer on every machine ---
//
// Everything the world generates from a seed - terrain, which road tile goes
// where, where each tree stands - must come out identical in every browser,
// because co-op clients and a reloaded save rebuild it from the seed rather
// than being sent it (World plan §0.4, P0). So this module, and every
// generator built on it, keeps to arithmetic that is exact everywhere:
//   - 32-bit integer maths (Math.imul, the bit operators), and
//   - + - * / and Math.floor/sqrt on doubles, which IEEE 754 rounds the same
//     in every engine.
// Never Math.random, and never Math.sin, cos, exp, log, pow or ** - the spec
// leaves their last bits to the engine, and V8 and SpiderMonkey differ
// (Gameplay §9). tools/rng.test.mjs scans the generator modules for them.
//
// Seeds are pure functions of what they are for, never of what ran before:
// chunkSeed(world, region, cx, cz, layer) gives a chunk the same stream
// whichever order the chunks are generated in, so a region rebuilt on load,
// the backdrop seen from the next region and a co-op client all agree.
//
// RNG_VERSION is saved beside the seed. Changing any output here moves every
// generated tree in every save, so it is a version bump, and the fixed
// outputs in the test say when it happened.

export const RNG_VERSION = 1;

const TWO_32 = 4294967296;

// --- Hashing ---

// Chris Wellons' lowbias32: a 32-bit integer finaliser, every input bit
// reaching every output bit. The same constants as gpu/hitcache.js mix32TSL,
// so a kernel can reproduce a CPU seed.
export function mix32(h) {
  h = (h ^ (h >>> 16)) >>> 0;
  h = Math.imul(h, 0x7feb352d) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0;
  h = Math.imul(h, 0x846ca68b) >>> 0;
  return (h ^ (h >>> 16)) >>> 0;
}

// FNV-1a over the string's UTF-16 code units: region and layer names.
export function hashString(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h;
}

// Any number of parts - integers in the 32-bit range (negative is fine) or
// strings - to one uint32. Order matters: hash32(1, 2) is not hash32(2, 1).
// A fraction throws rather than being rounded: a seed built from a position
// in metres would drift the moment the position did. A string takes one more
// mixing round than a number, so 'village' and the integer hashString('village')
// are different parts.
export function hash32(...parts) {
  let h = mix32(0x2545f491 ^ parts.length);
  for (const p of parts) {
    let word;
    if (typeof p === 'string') { h = mix32((h ^ 0x9e3779b9) >>> 0); word = hashString(p); }
    else if (Number.isInteger(p)) word = p >>> 0;
    else throw new Error(`hash32: ${p} is not an integer or a string`);
    h = mix32((h ^ word) >>> 0);
  }
  return h;
}

// A chunk's seed for one generator layer ('terrain', 'scatter', 'tiles', ...).
// worldSeed is the world's key (seedKey below), not the text a player typed.
export function chunkSeed(worldSeed, region, cx, cz, layer) {
  return hash32(worldSeed, region, cx, cz, layer);
}

// --- World seeds: the player's side ---
//
// As Minecraft's: a world is made from a seed a player can read, type and
// share, and the same seed gives the same world. A seed is text:
//   - a whole number ("-4172", " 007") is read as one, as Minecraft reads a
//     long: signs and leading zeros do not matter, so "007" and "7" are one
//     world. Past 64 bits it is plain text.
//   - anything else is text, exactly as typed but for the spaces around it,
//     and case counts ("Highport" is not "highport").
// Blank means "pick one" (randomSeed). The canonical text is what a save
// stores and what the game shows to copy.
//
// The same seed gives the same world only from the same generators and the
// same authored content: a save keeps RNG_VERSION and the generator versions
// beside the seed, as a Minecraft seed means one world per game version.
// In this game the seed moves the details - which variant each road tile is,
// where each tree and bush stands, the footpaths - and never the map itself
// (World plan §0.4).

const INT64_MIN = -(1n << 63n), INT64_MAX = (1n << 63n) - 1n;

// The canonical text of a seed, or null for a blank one.
export function normaliseSeed(text) {
  const t = String(text ?? '').trim();
  if (t === '') return null;
  if (/^[+-]?\d+$/.test(t)) {
    const n = BigInt(t);
    if (n >= INT64_MIN && n <= INT64_MAX) return n.toString();
  }
  return t;
}

// The world's key - the uint32 every generator's seeds start from. Text and
// number seeds both go through their canonical text, so "007" and 7 agree.
export function seedKey(seed) {
  const canonical = normaliseSeed(seed);
  if (canonical === null) throw new Error('seedKey: a blank seed has no key - pick one with randomSeed');
  return hash32('world', canonical);
}

// A fresh seed, as Minecraft offers one: a signed 64-bit number, as text.
// From the platform's cryptographic source, not Math.random - this is the one
// draw that is meant to differ, and it happens before any generator runs.
export function randomSeed() {
  const w = crypto.getRandomValues(new Uint32Array(2));
  return BigInt.asIntN(64, (BigInt(w[0]) << 32n) | BigInt(w[1])).toString();
}

// --- The generator ---
//
// sfc32 (Chris Doty-Humphrey's small fast counting generator): 128 bits of
// state, a counter that rules out short cycles, and it passes PractRand. Its
// four words are filled from the seed by splitmix32, so nearby seeds start
// far apart. The state is four plain numbers - it saves, and crosses to a
// worker or a co-op client, as data.

function splitmix32(s) {
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z = Math.imul(z ^ (z >>> 16), 0x85ebca6b) >>> 0;
    z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35) >>> 0;
    return (z ^ (z >>> 16)) >>> 0;
  };
}

// createRng(seed) or createRng(...parts) - the parts hashed as hash32 does.
export function createRng(...parts) {
  const seed = parts.length === 1 && Number.isInteger(parts[0]) ? parts[0] >>> 0 : hash32(...parts);
  const next = splitmix32(seed);
  const rng = rngFromState([next(), next(), next(), next()]);
  // A few rounds so the first outputs carry no trace of the seeding.
  for (let i = 0; i < 12; i++) rng.u32();
  return rng;
}

// A generator resumed exactly where state() left it.
export function rngFromState([sa, sb, sc, sd]) {
  let a = sa >>> 0, b = sb >>> 0, c = sc >>> 0, d = sd >>> 0;

  const u32 = () => {
    const t = (((a + b) >>> 0) + d) >>> 0;
    d = (d + 1) >>> 0;
    a = (b ^ (b >>> 9)) >>> 0;
    b = (c + (c << 3)) >>> 0;
    c = ((c << 21) | (c >>> 11)) >>> 0;
    c = (c + t) >>> 0;
    return t;
  };

  // [0, 1), a multiple of 2^-32: exact in a double, so the same everywhere.
  const float = () => u32() / TWO_32;

  // 0 .. n-1, unbiased: draws past the last whole multiple of n are redrawn.
  const int = n => {
    if (!Number.isInteger(n) || n < 1 || n > TWO_32) throw new Error(`rng.int: bad range ${n}`);
    const limit = TWO_32 - (TWO_32 % n);
    let x = u32();
    while (x >= limit) x = u32();
    return x % n;
  };

  const rng = {
    u32,
    float,
    int,
    // lo .. hi, both included.
    range: (lo, hi) => lo + int(hi - lo + 1),
    chance: p => float() < p,
    pick: list => list[int(list.length)],
    // One item, each with the share weightOf gives it (0 is never picked).
    weighted(list, weightOf = item => item.weight) {
      let total = 0;
      for (const item of list) total += weightOf(item);
      let r = float() * total;
      for (const item of list) {
        const w = weightOf(item);
        if (r < w) return item;
        r -= w;
      }
      // Rounding can leave r a hair past the last weight: the last real item.
      for (let i = list.length - 1; i >= 0; i--) if (weightOf(list[i]) > 0) return list[i];
      return undefined;
    },
    // Fisher-Yates, in place.
    shuffle(list) {
      for (let i = list.length - 1; i > 0; i--) {
        const j = int(i + 1);
        const t = list[i]; list[i] = list[j]; list[j] = t;
      }
      return list;
    },
    state: () => [a, b, c, d]
  };
  return rng;
}

// --- Noise ---
//
// Value noise: a hashed value at every integer lattice point, eased between
// them by the quintic fade 6t^5 - 15t^4 + 10t^3 - a polynomial, so exact
// everywhere, where gradient noise's usual normalisations reach for sqrt of
// sums or trigonometry. Lattice points are cells of whatever unit the caller
// scales into (pass x / 8 for features eight blocks across).

const fade = t => t * t * t * (t * (t * 6 - 15) + 10);
const lattice = (seed, ix, iz) => hash32(seed, ix, iz) / TWO_32;

// [0, 1). At a lattice point it is that point's value; between, it is smooth.
export function valueNoise2(seed, x, z) {
  const fx = Math.floor(x), fz = Math.floor(z);
  const ix = fx | 0, iz = fz | 0;
  const u = fade(x - fx), v = fade(z - fz);
  const a = lattice(seed, ix, iz), b = lattice(seed, ix + 1, iz);
  const c = lattice(seed, ix, iz + 1), d = lattice(seed, ix + 1, iz + 1);
  const top = a + (b - a) * u, bottom = c + (d - c) * u;
  return top + (bottom - top) * v;
}

// Octaves of value noise, each at twice the frequency and half the weight
// (by default), renormalised to [0, 1). Each octave has its own seed, so they
// do not line up at the origin.
export function fbm2(seed, x, z, octaves = 4, lacunarity = 2, gain = 0.5) {
  let sum = 0, weight = 1, total = 0, scale = 1;
  for (let o = 0; o < octaves; o++) {
    sum += valueNoise2(hash32(seed, o), x * scale, z * scale) * weight;
    total += weight;
    weight *= gain;
    scale *= lacunarity;
  }
  return sum / total;
}
