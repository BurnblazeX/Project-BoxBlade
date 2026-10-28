import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { file, section, ok, inRange, throws } from './lib/harness.mjs';
import { RNG_VERSION, mix32, hashString, hash32, chunkSeed, createRng, rngFromState,
         valueNoise2, fbm2, normaliseSeed, seedKey, randomSeed } from '../js/core/rng.js';

file('rng.test.mjs - seeded randomness, the same on every machine');

// The modules every seeded generator lives in. A generator added in P9 goes
// on this list, and from then on it may not reach for the functions whose
// last bits differ between browsers.
const GENERATOR_MODULES = ['core/rng.js'];
const UNSAFE = /\bMath\.(random|sin|cos|tan|asin|acos|atan2?|sinh|cosh|tanh|asinh|acosh|atanh|exp|expm1|log|log1p|log2|log10|pow|cbrt|hypot)\b|\*\*/;

section('generators use only exact arithmetic');
{
  const js = join(dirname(fileURLToPath(import.meta.url)), '..', 'js');
  for (const f of GENERATOR_MODULES) {
    const code = readFileSync(join(js, f), 'utf8')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .replace(/\/\/.*$/gm, '');
    const hit = code.match(UNSAFE);
    ok(`${f}: no Math.random, no transcendentals, no **`, hit ? hit[0] : '', '');
  }
}

// Fixed outputs. If one of these changes, every generated world changes with
// it: bump RNG_VERSION and update them together, on purpose.
section(`fixed outputs (RNG_VERSION ${RNG_VERSION})`);
{
  ok('version', RNG_VERSION, 1);
  const r = createRng(1);
  ok('seed 1: the first six draws', Array.from({ length: 6 }, () => r.u32()).join(),
     '1828152527,3394835397,2967886022,2251045104,4148684523,4055680838');
  ok('mix32', [mix32(0), mix32(1), mix32(0xdeadbeef)].join(), '0,1753845952,3861431939');
  ok('hash32', [hash32(1, 2), hash32(2, 1), hash32('village', -3, 7)].join(), '1035792493,2052497589,1730010372');
  ok('chunkSeed', chunkSeed(12345, 'village', 2, -1, 'scatter'), 2594038256);
  const s = createRng('village', 4);
  ok('seeded by parts', [s.u32(), s.int(6), s.range(10, 20), s.float()].join(), '3370752306,4,15,0.6098617669194937');
  ok('value noise', valueNoise2(7, 0.5, 0.25), 0.5676057248917914);
  ok('fbm', fbm2(7, 3.3, -1.7), 0.622112096817629);
  ok('world keys', [seedKey('BoxBlade'), seedKey('7'), seedKey('-4172')].join(), '574730399,2438517659,2113417');
}

section('world seeds, as Minecraft\'s');
{
  ok('a number ignores its sign and leading zeros', normaliseSeed(' +007 '), '7');
  ok('minus zero is zero', normaliseSeed('-0'), '0');
  ok('negative numbers stay negative', normaliseSeed('-4172'), '-4172');
  ok('the largest long is a number', normaliseSeed('9223372036854775807'), '9223372036854775807');
  ok('one past it is text', normaliseSeed('9223372036854775808'), '9223372036854775808');
  ok('text keeps its case and inner spaces', normaliseSeed('  Over the River '), 'Over the River');
  ok('blank is no seed', normaliseSeed('   '), null);
  ok('a number typed as text or as a number: one world', seedKey(' 007'), seedKey(7));
  ok('case counts', seedKey('Highport') !== seedKey('highport'), true);
  ok('a number and its text spelled out differ', seedKey('7') !== seedKey('seven'), true);
  throws('a blank seed has no key', () => seedKey(''));
  const fresh = randomSeed();
  ok('a fresh seed is a whole number', /^-?\d+$/.test(fresh), true);
  ok('... already in canonical form', normaliseSeed(fresh), fresh);
  ok('two fresh seeds differ', randomSeed() !== randomSeed(), true);
  // Same seed, same world: every chunk's stream follows from the key alone.
  const world = seed => Array.from({ length: 9 }, (_, i) =>
    createRng(chunkSeed(seedKey(seed), 'village', i % 3, Math.floor(i / 3), 'scatter')).u32()).join();
  ok('the same seed typed twice builds the same chunks', world('Over the River'), world(' Over the River'));
  ok('another seed builds other chunks', world('Over the River') !== world('Over the river'), true);
}

section('hashing');
{
  ok('order matters', hash32(1, 2) !== hash32(2, 1), true);
  ok('a part count matters', hash32(0) !== hash32(0, 0), true);
  ok('a string and its hash differ as parts', hash32('a') === hash32(hashString('a')), false);
  ok('negative coordinates are fine', hash32(-1) !== hash32(1), true);
  throws('a fraction is refused', () => hash32(1.5));
  throws('a missing part is refused', () => hash32(undefined));
  // 64 x 64 chunks of one layer: every seed different.
  const seeds = new Set();
  for (let cx = -32; cx < 32; cx++) for (let cz = -32; cz < 32; cz++) seeds.add(chunkSeed(1, 'road', cx, cz, 'scatter'));
  ok('4096 neighbouring chunks, 4096 seeds', seeds.size, 4096);
  ok('layers do not share a seed', chunkSeed(1, 'road', 0, 0, 'scatter') !== chunkSeed(1, 'road', 0, 0, 'terrain'), true);
}

section('streams are pure functions of their seed');
{
  const take = (rng, n) => Array.from({ length: n }, () => rng.u32()).join();
  ok('same seed, same stream', take(createRng(42), 50), take(createRng(42), 50));
  ok('next seed, another stream', take(createRng(42), 5) !== take(createRng(43), 5), true);
  // Chunks generated in two orders: each chunk's stream is its own.
  const draw = (cx, cz) => createRng(chunkSeed(9, 'village', cx, cz, 'scatter')).u32();
  const forward = [], backward = new Map();
  for (let i = 0; i < 25; i++) forward.push(draw(i % 5, Math.floor(i / 5)));
  for (let i = 24; i >= 0; i--) backward.set(i, draw(i % 5, Math.floor(i / 5)));
  ok('chunk order does not matter', forward.join(), forward.map((_, i) => backward.get(i)).join());
  // The state is plain data: save it, carry on, restore, get the same draws.
  const r = createRng(7);
  for (let i = 0; i < 10; i++) r.u32();
  const saved = JSON.parse(JSON.stringify(r.state()));
  const after = take(r, 5);
  ok('a restored state resumes exactly', take(rngFromState(saved), 5), after);
}

section('distributions');
{
  const r = createRng(2026);
  const N = 60000, counts = new Array(6).fill(0);
  let lo = 1, hi = 0, sum = 0;
  for (let i = 0; i < N; i++) {
    counts[r.int(6)]++;
    const f = r.float();
    lo = Math.min(lo, f); hi = Math.max(hi, f); sum += f;
  }
  ok('int(6): each face within 3% of 10000', counts.every(c => Math.abs(c - N / 6) < N / 6 * 0.03), true);
  ok('float stays in [0, 1)', lo >= 0 && hi < 1, true);
  inRange('float mean', sum / N, 0.495, 0.505);
  let inBounds = true;
  for (let i = 0; i < 1000; i++) { const v = r.range(-3, 3); if (v < -3 || v > 3) inBounds = false; }
  ok('range includes both ends and nothing past them', inBounds, true);
  ok('int(1) is always 0', Array.from({ length: 20 }, () => r.int(1)).every(v => v === 0), true);
  ok('int(2^32) is a whole uint32', Number.isInteger(r.int(4294967296)), true);
  throws('int(0) is refused', () => r.int(0));

  const items = [{ name: 'oak', weight: 6 }, { name: 'birch', weight: 3 }, { name: 'none', weight: 0 }, { name: 'pine', weight: 1 }];
  const picks = { oak: 0, birch: 0, none: 0, pine: 0 };
  for (let i = 0; i < 20000; i++) picks[r.weighted(items).name]++;
  inRange('weighted: oak about 60%', picks.oak / 20000, 0.58, 0.62);
  inRange('weighted: pine about 10%', picks.pine / 20000, 0.09, 0.11);
  ok('weighted: a weight of 0 is never picked', picks.none, 0);

  const deck = r.shuffle([...Array(20).keys()]);
  ok('shuffle keeps every item', [...deck].sort((a, b) => a - b).join(), [...Array(20).keys()].join());
  ok('pick takes from the list', [1, 2, 3].includes(r.pick([1, 2, 3])), true);
}

section('noise');
{
  let lo = 1, hi = 0, smooth = true;
  for (let i = 0; i < 2000; i++) {
    const x = i * 0.137 - 100, z = i * 0.071 - 50;
    const n = valueNoise2(3, x, z);
    lo = Math.min(lo, n); hi = Math.max(hi, n);
    if (Math.abs(valueNoise2(3, x + 1e-4, z) - n) > 1e-3) smooth = false;
  }
  ok('value noise stays in [0, 1)', lo >= 0 && hi < 1, true);
  ok('value noise is continuous', smooth, true);
  ok('at a lattice point it is that point\'s value', valueNoise2(3, 4, -2), hash32(3, 4, -2) / 4294967296);
  ok('another seed, another field', valueNoise2(3, 0.5, 0.5) !== valueNoise2(4, 0.5, 0.5), true);
  let fLo = 1, fHi = 0;
  for (let i = 0; i < 2000; i++) { const f = fbm2(5, i * 0.31, i * -0.17, 5); fLo = Math.min(fLo, f); fHi = Math.max(fHi, f); }
  ok('fbm stays in [0, 1)', fLo >= 0 && fHi < 1, true);
}
