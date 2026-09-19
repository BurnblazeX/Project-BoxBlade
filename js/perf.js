// --- Frame time profiling ---
//
// Split in two on purpose: createFrameStats is a pure ring buffer with no DOM,
// so the statistics can be tested headlessly, and createPerfOverlay is the
// canvas drawing on top of it. The graph is a 2D canvas overlay rather than
// anything in the scene, so profiling never perturbs what it is measuring.

const DEFAULT_CAPACITY = 180; // ~3 seconds at 60fps

export function createFrameStats(capacity = DEFAULT_CAPACITY) {
  const samples = new Float32Array(capacity);
  let count = 0;
  let head = 0;

  return {
    capacity,
    get count() { return count; },

    push(ms) {
      samples[head] = ms;
      head = (head + 1) % capacity;
      if (count < capacity) count++;
    },

    // Oldest first, so the graph can draw left-to-right without reordering.
    toArray() {
      const out = new Array(count);
      const start = count < capacity ? 0 : head;
      for (let i = 0; i < count; i++) out[i] = samples[(start + i) % capacity];
      return out;
    },

    reset() { count = 0; head = 0; },

    stats() {
      if (count === 0) return { last: 0, avg: 0, min: 0, max: 0, low1: 0, p95: 0, fps: 0, count: 0 };
      const arr = this.toArray();
      let sum = 0, min = Infinity, max = -Infinity;
      for (const v of arr) { sum += v; if (v < min) min = v; if (v > max) max = v; }
      const avg = sum / arr.length;
      // "1% low" in the usual sense: the frame time at the 99th percentile,
      // i.e. how bad the worst 1% of frames are. Reported as an fps figure
      // because that is how it is normally quoted.
      const sorted = [...arr].sort((a, b) => a - b);
      const at = q => sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))];
      const p99 = at(0.99);
      return {
        last: arr[arr.length - 1],
        avg, min, max,
        low1: p99 > 0 ? 1000 / p99 : 0,
        p95: at(0.95),
        fps: avg > 0 ? 1000 / avg : 0,
        count: arr.length
      };
    }
  };
}

const WIDTH = 240;
const GRAPH_H = 64;
const BUDGET_60 = 1000 / 60;  // 16.67ms
const BUDGET_30 = 1000 / 30;  // 33.33ms

function barColour(ms) {
  if (ms <= BUDGET_60) return '#4ade80'; // within 60fps
  if (ms <= BUDGET_30) return '#facc15'; // within 30fps
  return '#f87171';
}

export function createPerfOverlay(doc = document) {
  const stats = createFrameStats();

  const canvas = doc.createElement('canvas');
  canvas.width = WIDTH;
  canvas.height = GRAPH_H + 34;
  // The project stylesheet has a blanket `canvas { width: 100%; height: 100% }`
  // for the renderer's canvas, which would stretch this overlay across the whole
  // window. Inline styles outrank it, so the CSS size is pinned to the backing
  // buffer size explicitly.
  Object.assign(canvas.style, {
    position: 'fixed', top: '8px', right: '8px', zIndex: '9999',
    width: WIDTH + 'px', height: (GRAPH_H + 34) + 'px',
    background: 'rgba(0,0,0,0.65)', borderRadius: '4px',
    pointerEvents: 'none', display: 'none',
    imageRendering: 'pixelated'
  });
  doc.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  let visible = false;
  let skipNext = false;
  let renderer = null;

  function draw() {
    const s = stats.stats();
    const arr = stats.toArray();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Scaled by the 95th percentile rather than the absolute max: a single
    // multi-second hitch (a chunk rebuild, a shader compile) would otherwise
    // flatten every normal frame into an invisible sliver. Never tighter than
    // the 30fps budget, so a smooth run doesn't magnify noise into apparent
    // spikes. Bars past the top are clamped; the true max is in the readout.
    const scaleMax = Math.max(BUDGET_30, s.p95 || 0);
    const top = 30;

    for (const budget of [BUDGET_60, BUDGET_30]) {
      const y = top + GRAPH_H - (budget / scaleMax) * GRAPH_H;
      ctx.strokeStyle = 'rgba(255,255,255,0.22)';
      ctx.beginPath();
      ctx.moveTo(0, y); ctx.lineTo(WIDTH, y); ctx.stroke();
    }

    const barW = WIDTH / stats.capacity;
    for (let i = 0; i < arr.length; i++) {
      const h = Math.min(GRAPH_H, (arr[i] / scaleMax) * GRAPH_H);
      ctx.fillStyle = barColour(arr[i]);
      ctx.fillRect(i * barW, top + GRAPH_H - h, Math.max(1, barW), h);
    }

    ctx.fillStyle = '#e5e7eb';
    ctx.font = '11px monospace';
    ctx.fillText(`${s.fps.toFixed(0)} fps   ${s.last.toFixed(1)}ms`, 6, 13);
    ctx.fillText(`1% low ${s.low1.toFixed(0)}   max ${s.max.toFixed(1)}ms`, 6, 25);

    if (renderer && renderer.info && renderer.info.render) {
      const r = renderer.info.render;
      ctx.fillStyle = '#9ca3af';
      ctx.fillText(`draws ${r.drawCalls ?? '-'}  tris ${(r.triangles ?? 0).toLocaleString()}`,
                   6, top + GRAPH_H + 16);
    }
  }

  return {
    // dt arrives in seconds from the existing clock.
    update(dt) {
      if (!visible) return;
      // The frame that turns the graph on also contains the work that turned it
      // on (a console call, a chunk rebuild), so its dt is not a real frame time
      // and would sit in the buffer skewing the average for seconds.
      if (skipNext) { skipNext = false; return; }
      stats.push(dt * 1000);
      draw();
    },
    toggle() {
      visible = !visible;
      canvas.style.display = visible ? 'block' : 'none';
      if (visible) skipNext = true; else stats.reset();
      return visible;
    },
    get visible() { return visible; },
    attachRenderer(r) { renderer = r; },
    stats: () => stats.stats()
  };
}
