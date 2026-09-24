// --- Compiled shadow material sets, kept ---
//
// A shadows-on material set (terrain, texel-cache pair, sprites) costs two big
// pipeline compiles, and a compile blocks the frame in both browsers - the
// driver work runs on the same GPU-process thread as frame submission, so
// compileAsync hides nothing (doc §0, profiles). Disposing the set when shadows
// go off, and rebuilding it when they come back, paid that every toggle.
//
// So sets are kept, keyed by everything that goes into their shaders. A key
// hit is a material swap. A setting that changes any input changes the key, so
// a stale set can never be picked up by mistake: the key is built generically
// from the build's own options, not from a hand-kept list that could miss one.
//
// Pure - no three, no DOM - so the key and the eviction order test headless.

const ids = new WeakMap();
let nextId = 1;
const idOf = o => {
  let i = ids.get(o);
  if (!i) ids.set(o, i = nextId++);
  return i;
};

// Primitives by value; plain objects and arrays by their contents (a few
// levels, which is how deep the build's option bags go); anything else - a
// uniform, a texture, a material, a binding the renderer keeps - by identity.
// Functions are skipped: the one the build passes (the albedo sampler) is a
// fresh closure over the same, identity-keyed inputs every time.
export function keyPart(v, depth = 0) {
  if (v === null || v === undefined) return String(v);
  const t = typeof v;
  if (t === 'number' || t === 'boolean' || t === 'string') return JSON.stringify(v);
  if (t === 'function') return 'fn';
  if (depth < 4 && Array.isArray(v)) {
    return '[' + v.map(x => keyPart(x, depth + 1)).join(',') + ']';
  }
  if (depth < 4 && Object.getPrototypeOf(v) === Object.prototype) {
    return '{' + Object.keys(v).sort()
      .map(k => k + ':' + keyPart(v[k], depth + 1)).join(',') + '}';
  }
  return '#' + idOf(v);
}

// --- Is the shader text the same from run to run? ---
//
// The browser's pipeline cache is keyed by the WGSL text, so text that differs
// between runs recompiles every run. The known culprit is three naming an
// unnamed buffer NodeBuffer_<global node id>: that id counts every node made
// before it, which depends on timing. Our buffers are all named; this finds any
// that are not (three's own instancing buffer is one: an InstancedMesh of at
// most 1024 instances keeps its matrices in such a buffer).
export function unstableNames(wgsl) {
  return [...new Set(wgsl.match(/NodeBuffer_\d+/g) || [])];
}

// FNV-1a, 32 bit, hex: short enough to compare by eye across two runs.
export function hashText(text) {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h.toString(16).padStart(8, '0');
}

// Which named parts of two keys differ - for saying WHY a turn-on had to
// compile, rather than guessing. parts: { name: keyPart(...) }.
export function diffKeyParts(a, b) {
  const names = new Set([...Object.keys(a || {}), ...Object.keys(b || {})]);
  return [...names].filter(k => (a || {})[k] !== (b || {})[k]).sort();
}

// Keeps the newest `keep` sets. The active set is never evicted, nor the newest
// (the one about to be shown), nor one still compiling (disposing a material mid-compile is asking for trouble);
// evict() returns what it dropped, for the caller to dispose.
export function createSetCache(keep = 2) {
  const sets = new Map();   // insertion order = age; get() refreshes it
  return {
    get size() { return sets.size; },
    has: key => sets.has(key),
    get(key) {
      const s = sets.get(key);
      if (s) { sets.delete(key); sets.set(key, s); }
      return s;
    },
    peek: key => sets.get(key),
    set(key, s) { sets.delete(key); sets.set(key, s); },
    delete: key => sets.delete(key),
    evict(active) {
      const out = [];
      const newest = [...sets.values()].pop();
      for (const [key, s] of sets) {
        if (sets.size <= keep) break;
        if (s === active || s === newest || !s.compiled) continue;
        sets.delete(key);
        out.push(s);
      }
      return out;
    },
    // Everything but the active set, for turning the cache off.
    drain(active) {
      const out = [];
      for (const [key, s] of sets) {
        if (s === active) continue;
        sets.delete(key);
        out.push(s);
      }
      return out;
    },
    values: () => [...sets.values()]
  };
}
