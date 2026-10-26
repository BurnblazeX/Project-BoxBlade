// --- GPU time by pass and by kernel (bxb.gpupasses) ---
//
// bxb.gpu says how long the frame's render and compute take; this says which
// pass or kernel it is. For the measurement only, every compute list is
// dispatched a kernel at a time (three times a whole call, not a dispatch),
// and each render call is named by where in the game it came from. Timestamps
// are three's own (trackTimestamp), so they come in whole resolves, rounded to
// whatever the browser allows (Chrome: 0.066 ms) - averaged over a few
// seconds, that rounding evens out.
//
// The split dispatches cost more API calls than the game makes, so read the
// GPU times here, not the frame rate.

// The caller outside three, from a stack - "texelcache.js:343" and the like.
function callerOf(stack) {
  for (const line of stack.split('\n').slice(2)) {
    if (line.includes('three.webgpu') || line.includes('passtimes.js') || line.includes('uploads.js')) continue;
    const m = line.match(/([\w.-]+\.js):(\d+)/);
    if (m) return `${m[1]}:${m[2]}`;
  }
  return '?';
}

export async function measurePasses(renderer, seconds = 5) {
  const backend = renderer.backend;
  const pool = backend.timestampQueryPool;
  if (!pool || !pool.render) return 'no timestamp queries on this device (trackTimestamp)';
  const labels = new Map();   // uid without its frame -> name
  const base = uid => uid.replace(/:f\d+$/, '');
  const realCompute = renderer.compute, realRender = renderer.render;
  const realBeginRender = backend.beginRender, realBeginCompute = backend.beginCompute;
  let renderLabel = '?';
  backend.beginRender = function (ctx) {
    const out = realBeginRender.call(this, ctx);
    const uid = this.get(ctx).timestampUID;
    if (uid) labels.set(base(uid), `render ${renderLabel}`);
    return out;
  };
  backend.beginCompute = function (group) {
    const out = realBeginCompute.call(this, group);
    const uid = this.get(group).timestampUID;
    if (uid) labels.set(base(uid), [group].flat().map(k => k.name || `kernel #${k.id}`).join('+'));
    return out;
  };
  renderer.compute = (k, d) => { for (const x of [k].flat()) realCompute(x, d); };
  renderer.render = (scene, camera) => { renderLabel = callerOf(new Error().stack); return realRender(scene, camera); };
  const seen = new Set(), sums = new Map(), frames = new Set();
  try {
    await new Promise(r => setTimeout(r, 500));   // let the split dispatches settle in
    const t0 = performance.now();
    while (performance.now() - t0 < seconds * 1000) {
      for (const type of ['render', 'compute']) {
        for (const [key, ms] of pool[type].timestamps) {
          if (seen.has(key)) continue;
          seen.add(key);
          const m = key.match(/^(.*):f(\d+)$/);
          if (!m) continue;
          frames.add(m[2]);
          const name = labels.get(m[1]) || m[1];
          sums.set(name, (sums.get(name) || 0) + ms);
        }
      }
      await new Promise(r => setTimeout(r, 15));
    }
  } finally {
    renderer.compute = realCompute;
    renderer.render = realRender;
    backend.beginRender = realBeginRender;
    backend.beginCompute = realBeginCompute;
  }
  const n = Math.max(1, frames.size);
  const rows = [...sums].map(([pass, s]) => ({ pass, ms: +(s / n).toFixed(4) })).sort((a, b) => b.ms - a.ms);
  const total = +rows.reduce((s, r) => s + r.ms, 0).toFixed(3);
  console.table(rows);
  return { frames: n, total, rows };
}
