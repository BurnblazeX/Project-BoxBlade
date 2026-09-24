// --- Frame time profiling ---
//
// Split in two on purpose: createFrameStats is a pure ring buffer with no DOM,
// so the statistics can be tested headlessly, and createPerfOverlay is the
// canvas drawing on top of it. The graph is a 2D canvas overlay rather than
// anything in the scene, so profiling never perturbs what it is measuring.

const DEFAULT_CAPACITY = 180;
// The overlay's window is a fixed span of TIME, not a count of frames: at 240 fps
// a 180-frame window was 0.75 s and scrolled four times faster than at 60. The
// capacity only has to hold the busiest window - 3 s at over 2,700 fps.
export const WINDOW_MS = 3000;
const OVERLAY_CAPACITY = 8192;

// Samples carry the time they were taken. With a window, only samples within
// windowMs of the NEWEST one count - anchored to the data rather than the clock,
// so a stall freezes the picture instead of emptying it. Without one (the
// default), it is a plain ring buffer of the last `capacity` samples.
export function createFrameStats(capacity = DEFAULT_CAPACITY, windowMs = Infinity) {
  const samples = new Float32Array(capacity);
  const times = new Float64Array(capacity);
  let count = 0;
  let head = 0;
  let clock = 0;   // stands in for a timestamp when none is given

  // How many of the newest samples lie inside the window.
  function inWindow() {
    if (count === 0 || !Number.isFinite(windowMs)) return count;
    const newest = times[(head - 1 + capacity) % capacity];
    let n = 0;
    while (n < count && newest - times[(head - 1 - n + capacity * 2) % capacity] <= windowMs) n++;
    return n;
  }

  return {
    capacity,
    windowMs,
    get count() { return inWindow(); },

    push(ms, t = ++clock) {
      samples[head] = ms;
      times[head] = t;
      head = (head + 1) % capacity;
      if (count < capacity) count++;
    },

    // Oldest first, so the graph can draw left-to-right without reordering.
    toArray() { return this.series().v; },

    // Values and their times, oldest first, inside the window.
    series() {
      const n = inWindow();
      const v = new Array(n), t = new Array(n);
      const start = (head - n + capacity) % capacity;
      for (let i = 0; i < n; i++) {
        const j = (start + i) % capacity;
        v[i] = samples[j]; t[i] = times[j];
      }
      return { v, t };
    },

    // Time of the newest sample, or null when empty.
    get newest() { return count ? times[(head - 1 + capacity) % capacity] : null; },

    reset() { count = 0; head = 0; },

    // last: only the newest `last` samples (bxb.gpu's "over N frames").
    stats(last = Infinity) {
      const all = this.toArray();
      const arr = all.length > last ? all.slice(-last) : all;
      if (arr.length === 0) return { last: 0, avg: 0, min: 0, max: 0, low1: 0, p95: 0, fps: 0, count: 0 };
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
export const DEFAULT_DRAW_HZ = 60;
// Frame times jitter around vsync: at exactly 60 fps a frame can land a hair
// before the 16.67 ms mark, and a strict test would then skip every other one
// and repaint at 30. A millisecond of slack keeps 60 fps at 60 repaints.
const REDRAW_SLACK_MS = 1;

// The repaint schedule: at most hz a second, and never more than once a frame
// (it is only asked once a frame), so a game below hz repaints every frame.
// due is when the next repaint is owed; it advances by whole intervals, so at
// 144 fps the repaints average 60 a second rather than rounding down to 48.
// Returns { draw, due }. hz 0 repaints every frame.
export function nextRedraw(now, due, hz) {
  if (!(hz > 0)) return { draw: true, due: now };
  if (now < due - REDRAW_SLACK_MS) return { draw: false, due };
  const interval = 1000 / hz;
  let next = due + interval;
  // Fallen behind (a slow frame, a hitch, the first repaint): restart from now
  // rather than repainting on every frame to catch up.
  if (next <= now) next = now + interval;
  return { draw: true, due: next };
}
// Per pixel column, the index of the worst frame whose span overlaps it, or -1
// where none does. A frame spans (t - ms, t]: it fills every column its duration
// touched, so a 60 fps run leaves no gaps between 12.5 ms columns and a hitch is
// drawn as wide as it lasted. The worst, not the average, so a spike among
// fast frames still shows.
export function worstPerColumn(v, t, end, windowMs, columns) {
  const out = new Int32Array(columns).fill(-1);
  const start = end - windowMs, colMs = windowMs / columns;
  for (let i = 0; i < v.length; i++) {
    const c0 = Math.max(0, Math.floor((t[i] - v[i] - start) / colMs));
    const c1 = Math.min(columns - 1, Math.floor((t[i] - start) / colMs));
    for (let c = c0; c <= c1; c++) if (out[c] < 0 || v[i] > v[out[c]]) out[c] = i;
  }
  return out;
}

const BREAK_TOP = 30 + GRAPH_H + 10;
const HEIGHT = BREAK_TOP + GRAPH_H + 44;

export function createPerfOverlay(doc = document) {
  // All on one time axis. The sections are pushed with the frame, at the same
  // time, so their series line up index for index with the frame's.
  const series = () => createFrameStats(OVERLAY_CAPACITY, WINDOW_MS);
  const stats = series();
  const parts = Object.fromEntries(SECTIONS.map(([k]) => [k, series()]));
  // The GPU series are counted, not windowed: bxb.gpu waits for N resolves, and
  // resolves land less often than frames - on a slow device a 3 s window might
  // never hold N. They are cut to the window only when drawn.
  // Large enough for the whole window at any rate a resolve can land.
  const gpu = createFrameStats(OVERLAY_CAPACITY);
  // GPU time of the compute passes (the LPV), resolved separately from render.
  const gpuCompute = createFrameStats(OVERLAY_CAPACITY);

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
  // Repaints a second, at most. Every frame is still SAMPLED - the stats,
  // percentiles and averages are exactly what they were - only the picture
  // refreshes less at high frame rates. Drawing every frame cost ~11% of
  // Firefox's WebGPU thread (doc §0) and lowered the fps it reported.
  // 0 repaints every frame.
  let drawHz = DEFAULT_DRAW_HZ;
  let drawDue = -Infinity;

  function draw() {
    const s = stats.stats();
    const { v: arr, t: times } = stats.series();
    const end = stats.newest;
    const cols = worstPerColumn(arr, times, end, WINDOW_MS, WIDTH);
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

    // One path and one fill per colour, not a fillRect per bar: in Firefox 2D
    // canvas is remoted to the same GPU-process thread as WebGPU, and there the
    // cost is per call. Same rectangles, same pixels.
    // One column a pixel, WINDOW_MS across the width.
    const bars = new Map();
    for (let c = 0; c < WIDTH; c++) {
      const i = cols[c];
      if (i < 0) continue;
      const h = Math.min(GRAPH_H, (arr[i] / scaleMax) * GRAPH_H);
      const colour = barColour(arr[i]);
      let path = bars.get(colour);
      if (!path) bars.set(colour, path = new Path2D());
      path.rect(c, top + GRAPH_H - h, 1, h);
    }
    for (const [colour, path] of bars) { ctx.fillStyle = colour; ctx.fill(path); }

    ctx.fillStyle = '#e5e7eb';
    ctx.font = '11px monospace';
    ctx.fillText(`${s.fps.toFixed(0)} fps   ${s.last.toFixed(1)}ms`, 6, 13);
    ctx.fillText(`1% low ${s.low1.toFixed(0)}   max ${s.max.toFixed(1)}ms`, 6, 25);

    // The resolution actually rendered: the canvas's backing size, which is CSS
    // pixels times the renderer's pixel ratio - not the window's physical size.
    // dpr is the display's, so a gap between the two is upscaling.
    if (renderer && renderer.domElement) {
      const c = renderer.domElement;
      const dpr = (doc.defaultView && doc.defaultView.devicePixelRatio) || 1;
      ctx.fillStyle = '#9ca3af';
      ctx.textAlign = 'right';
      ctx.fillText(`${c.width}×${c.height}`, WIDTH - 6, 13);
      ctx.fillText(`dpr ${+dpr.toFixed(2)}`, WIDTH - 6, 25);
      ctx.textAlign = 'left';
    }

    drawBreakdown(s, cols, end);
  }

  // Scaled to the frame's own time rather than to the 30 fps budget above:
  // at high refresh rates the whole frame is a few milliseconds, and on that
  // scale the stages would be slivers.
  // Each column stacks the sections of the frame the graph above shows there -
  // the worst one - so the two graphs describe the same frame.
  function drawBreakdown(s, frameCols, end) {
    const cols = SECTIONS.map(([k]) => parts[k].toArray());
    const all = gpu.series();
    const keep = end === null ? [] : all.t.map((t, i) => i).filter(i => all.t[i] >= end - WINDOW_MS);
    const g = keep.map(i => all.v[i]), gt = keep.map(i => all.t[i]);
    const scale = Math.max(1, s.p95 || 0, ...g) * 1.1;
    const y0 = BREAK_TOP + GRAPH_H;
    const paths = SECTIONS.map(() => new Path2D());
    for (let c = 0; c < WIDTH; c++) {
      const i = frameCols[c];
      if (i < 0) continue;
      let acc = 0;
      for (let k = 0; k < SECTIONS.length; k++) {
        const v = cols[k][i] || 0;
        const h = (v / scale) * GRAPH_H;
        paths[k].rect(c, y0 - acc - h, 1, h);
        acc += h;
      }
    }
    SECTIONS.forEach(([, colour], k) => { ctx.fillStyle = colour; ctx.fill(paths[k]); });
    // GPU time arrives a frame or two late and at its own cadence, so it is its
    // own series, placed by when it landed on the same time axis.
    if (g.length > 1 && end !== null) {
      ctx.strokeStyle = GPU_COLOUR;
      ctx.beginPath();
      const start = end - WINDOW_MS;
      let first = true;
      g.forEach((v, i) => {
        const x = Math.min(WIDTH, ((gt[i] - start) / WINDOW_MS) * WIDTH);
        if (x < 0) return;
        const y = y0 - Math.min(GRAPH_H, (v / scale) * GRAPH_H);
        if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
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
      const now = performance.now();
      stats.push(dt * 1000, now);
      for (const [k] of SECTIONS) parts[k].push(sections ? sections[k] || 0 : 0, now);
      const r = nextRedraw(now, drawDue, drawHz);
      drawDue = r.due;
      if (!r.draw) return;
      draw();
    },
    get drawHz() { return drawHz; },
    set drawHz(hz) { drawHz = Math.max(0, Number(hz) || 0); drawDue = -Infinity; },
    // GPU time for a frame, in ms, whenever a timestamp resolve lands.
    gpu(ms) { if (visible) gpu.push(ms, performance.now()); },
    gpuCompute(ms) { if (visible) gpuCompute.push(ms, performance.now()); },
    // Averages for scripted profiling (bxb.gpu): render and compute, ms.
    // Over the newest n resolves - the buffer itself holds far more.
    gpuStats: (n = DEFAULT_CAPACITY) => ({ render: gpu.stats(n), compute: gpuCompute.stats(n) }),
    resetGPU() { gpu.reset(); gpuCompute.reset(); },
    toggle() {
      visible = !visible;
      canvas.style.display = visible ? 'block' : 'none';
      if (visible) { skipNext = true; drawDue = -Infinity; }
      else { stats.reset(); gpu.reset(); gpuCompute.reset(); for (const [k] of SECTIONS) parts[k].reset(); }
      return visible;
    },
    get visible() { return visible; },
    attachRenderer(r) { renderer = r; },
    stats: () => stats.stats()
  };
}
