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

// The frame split into where the CPU spends it, stacked bottom-up in this
// order, plus the GPU's own time for the frame drawn as a line over the stack.
// A frame whose total bar is tall but whose stack is short was waiting on
// something else - the GPU, or presentation.
export const SECTIONS = [
  ['logic', '#60a5fa'],    // input, movement, camera, AI
  ['grids', '#f472b6'],    // cascade follow and worker hand-offs
  ['cards', '#a78bfa'],    // cards, occluders, torch
  ['submit', '#fbbf24']    // renderer.render: encoding and submission
];
const GPU_COLOUR = '#f8fafc';
const BREAK_TOP = 30 + GRAPH_H + 10;
const HEIGHT = BREAK_TOP + GRAPH_H + 44;

export function createPerfOverlay(doc = document) {
  const stats = createFrameStats();
  const parts = Object.fromEntries(SECTIONS.map(([k]) => [k, createFrameStats()]));
  const gpu = createFrameStats();
  // GPU time of the compute passes (the LPV), resolved separately from render.
  const gpuCompute = createFrameStats();

  const canvas = doc.createElement('canvas');
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  // The project stylesheet has a blanket `canvas { width: 100%; height: 100% }`
  // for the renderer's canvas, which would stretch this overlay across the whole
  // window. Inline styles outrank it, so the CSS size is pinned to the backing
  // buffer size explicitly.
  Object.assign(canvas.style, {
    position: 'fixed', top: '8px', right: '8px', zIndex: '9999',
    width: WIDTH + 'px', height: HEIGHT + 'px',
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

    drawBreakdown(s);
  }

  // Scaled to the frame's own time rather than to the 30 fps budget above:
  // at high refresh rates the whole frame is a few milliseconds, and on that
  // scale the stages would be slivers.
  function drawBreakdown(s) {
    const cols = SECTIONS.map(([k]) => parts[k].toArray());
    const g = gpu.toArray();
    const scale = Math.max(1, s.p95 || 0, ...g) * 1.1;
    const y0 = BREAK_TOP + GRAPH_H;
    const barW = WIDTH / stats.capacity;
    const n = cols[0].length;
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let k = 0; k < SECTIONS.length; k++) {
        const v = cols[k][i] || 0;
        const h = (v / scale) * GRAPH_H;
        ctx.fillStyle = SECTIONS[k][1];
        ctx.fillRect(i * barW, y0 - acc - h, Math.max(1, barW), h);
        acc += h;
      }
    }
    // GPU time arrives a frame or two late and at its own cadence, so it is its
    // own series, right-aligned with the newest frame.
    if (g.length > 1) {
      ctx.strokeStyle = GPU_COLOUR;
      ctx.beginPath();
      const off = stats.capacity - g.length;
      g.forEach((v, i) => {
        const x = (off + i + 0.5) * barW, y = y0 - Math.min(GRAPH_H, (v / scale) * GRAPH_H);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
    ctx.fillStyle = '#9ca3af';
    ctx.fillText(`scale ${scale.toFixed(1)}ms`, WIDTH - 84, BREAK_TOP + 9);

    // Legend: averages over the window, so a stage that spikes once does not
    // read as its steady cost.
    const avg = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0;
    ctx.font = '10px monospace';
    let x = 6, y = y0 + 13;
    SECTIONS.forEach(([k, colour], i) => {
      ctx.fillStyle = colour;
      ctx.fillText(`${k} ${avg(cols[i]).toFixed(2)}`, x, y);
      x += 58;
    });
    ctx.fillStyle = GPU_COLOUR;
    ctx.fillText(g.length ? `gpu ${avg(g).toFixed(2)}ms` : 'gpu n/a (no timestamp-query)', 6, y + 13);
    if (renderer && renderer.info && renderer.info.render) {
      const r = renderer.info.render;
      ctx.fillStyle = '#9ca3af';
      ctx.fillText(`draws ${r.drawCalls ?? '-'}  tris ${(r.triangles ?? 0).toLocaleString()}`,
                   6, y + 26);
    }
    ctx.font = '11px monospace';
  }

  return {
    // dt arrives in seconds from the existing clock.
    // sections: ms per SECTIONS key for this frame's CPU work.
    update(dt, sections = null) {
      if (!visible) return;
      // The frame that turns the graph on also contains the work that turned it
      // on (a console call, a chunk rebuild), so its dt is not a real frame time
      // and would sit in the buffer skewing the average for seconds.
      if (skipNext) { skipNext = false; return; }
      stats.push(dt * 1000);
      for (const [k] of SECTIONS) parts[k].push(sections ? sections[k] || 0 : 0);
      draw();
    },
    // GPU time for a frame, in ms, whenever a timestamp resolve lands.
    gpu(ms) { if (visible) gpu.push(ms); },
    gpuCompute(ms) { if (visible) gpuCompute.push(ms); },
    // Averages for scripted profiling (bxb.gpu): render and compute, ms.
    gpuStats: () => ({ render: gpu.stats(), compute: gpuCompute.stats() }),
    resetGPU() { gpu.reset(); gpuCompute.reset(); },
    toggle() {
      visible = !visible;
      canvas.style.display = visible ? 'block' : 'none';
      if (visible) skipNext = true;
      else { stats.reset(); gpu.reset(); gpuCompute.reset(); for (const [k] of SECTIONS) parts[k].reset(); }
      return visible;
    },
    get visible() { return visible; },
    attachRenderer(r) { renderer = r; },
    stats: () => stats.stats()
  };
}
