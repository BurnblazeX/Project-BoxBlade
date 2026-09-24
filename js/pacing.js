// --- Frame pacing ---
//
// A page cannot turn vsync off or choose double or triple buffering:
// requestAnimationFrame fires once per display refresh, WebGPU's canvas has no
// present mode, and the compositor presents on vblank whatever the page does.
// What a page CAN choose is which refreshes it renders on:
//
//   'vsync'     every refresh - the browser default.
//   30/40/60    a cap: render on the first refresh at or after each 1/cap
//               interval and skip the rest. The average is exact; a cap that does
//               not divide the refresh rate alternates frame lengths (40 on a
//               60 Hz display is 16.7, 33.3, 16.7...). A cap at or above the
//               refresh rate is the same as 'vsync'.
//   'uncapped'  frames back to back from a MessageChannel loop instead of rAF.
//               The compositor still shows at most one per refresh and drops the
//               rest, so there is no tearing and little latency won, at full
//               power draw. For measuring throughput, not for playing.
//               At most `depth` frames are queued on the GPU: unbounded,
//               a saturated GPU (or GPU-process thread) let JS race ahead until
//               the browser blocked on the backlog, and the graph read as bursts
//               of short frames and then one long stall.
//
// paceTick is pure; createFramePacer needs a renderer.

import { nextRedraw } from './perf.js';

export const PACING_MODES = ['vsync', 30, 40, 60, 'uncapped'];
export const DEFAULT_PACING = 'vsync';
// Frames the uncapped loop lets the GPU hold. Two halved throughput: the
// completion signal reaches the page late (a round trip to the GPU process, and
// Chrome only checks for it periodically), so waiting on the previous frame
// left the GPU idle while the news travelled. Deeper keeps it fed; still
// bounded, so no backlog. 0 is unbounded - the old behaviour, for comparison.
// 4 matched unbounded throughput without its stalls on the dev machine (5950X,
// RTX 5060 Ti, Chrome). Untested elsewhere; the setting is there to re-tune.
export const DEFAULT_IN_FLIGHT = 4;
export const MAX_IN_FLIGHT = 8;

// '30' and 30 are the same mode; anything unknown is null.
export function normalisePacing(mode) {
  const m = typeof mode === 'string' && mode.trim() !== '' && !isNaN(mode) ? Number(mode) : mode;
  return PACING_MODES.includes(m) ? m : null;
}

// Whether the refresh at `now` renders. The cap uses the frame graph's repaint
// schedule: due advances by whole intervals, so the average is the cap even when
// refreshes do not line up with it, and a millisecond of slack keeps a 60 cap on
// a 60 Hz display from skipping a refresh that lands a hair early.
// Returns { draw, due }; due is carried to the next call.
export function paceTick(now, due, mode) {
  if (typeof mode !== 'number') return { draw: true, due: now };
  return nextRedraw(now, due, mode);
}

// frame(time) is the game's per-frame function. Hand onRefresh to
// renderer.setAnimationLoop in its place.
export function createFramePacer(renderer, frame) {
  let mode = DEFAULT_PACING;
  let due = -Infinity;
  let pumping = false;
  const channel = new MessageChannel();
  const next = () => channel.port2.postMessage(0);
  // Completion promises of the frames still on the GPU, oldest first.
  const inFlight = [];
  let depth = DEFAULT_IN_FLIGHT;
  // The loop blocked on the oldest frame. A wait can be released early (the
  // depth was raised), so each carries an id and only the current one resumes.
  let waitId = 0, waiting = false;
  function waitFor(done) {
    const id = ++waitId;
    waiting = true;
    done.then(() => { if (id !== waitId) return; waiting = false; next(); });
  }
  function release() { if (!waiting) return; waitId++; waiting = false; next(); }

  // Outside rAF, three's animation loop does not run for us, so do what it does
  // before each frame: reset the per-frame counters and advance the node frame
  // (the time uniforms and once-per-frame node updates). Its rAF loop keeps
  // ticking meanwhile with nothing to call.
  function pump() {
    if (mode !== 'uncapped') { pumping = false; inFlight.length = 0; return; }
    // A hidden tab gets no rAF; the uncapped loop should not keep burning either.
    if (typeof document !== 'undefined' && document.hidden) { setTimeout(pump, 250); return; }
    try {
      const info = renderer.info;
      if (info.autoReset) info.reset();
      const nodeFrame = renderer._nodes && renderer._nodes.nodeFrame;
      if (nodeFrame) { nodeFrame.update(); info.frame = nodeFrame.frameId; }
      frame(performance.now());
    } finally {
      // Continued even after a throw, or one bad frame would stop the loop for
      // good. With the GPU's queue full, the next frame waits for the oldest to
      // finish; a failed wait (a lost device) must not stall the loop either.
      const queue = renderer.backend && renderer.backend.device && renderer.backend.device.queue;
      // Caught as it is queued: one not yet waited on would otherwise surface as
      // an unhandled rejection every frame.
      if (depth > 0 && queue && queue.onSubmittedWorkDone) {
        inFlight.push(queue.onSubmittedWorkDone().catch(() => {}));
      }
      // Past the depth (it was just lowered), the oldest are dropped, not waited
      // on - otherwise one in, one out would hold the queue at its old length.
      while (inFlight.length > depth) inFlight.shift();
      if (depth > 0 && inFlight.length === depth) waitFor(inFlight.shift());
      else next();
    }
  }
  channel.port1.onmessage = pump;

  return {
    onRefresh(time, xrFrame) {
      if (mode === 'uncapped') return;
      const r = paceTick(time, due, mode);
      due = r.due;
      if (r.draw) frame(time, xrFrame);
    },
    get mode() { return mode; },
    // Uncapped queue depth, 0 (unbounded) to MAX_IN_FLIGHT.
    get depth() { return depth; },
    set depth(n) {
      const old = depth;
      depth = Math.max(0, Math.min(MAX_IN_FLIGHT, Math.round(Number(n)) || 0));
      // Raised, or unbounded: a loop blocked under the old depth goes on now.
      // Lowered: the next frames drop the excess.
      if (depth === 0 || depth > old) release();
    },
    // Unknown modes are ignored; returns whether it took.
    setMode(next) {
      const m = normalisePacing(next);
      if (m === null) return false;
      mode = m;
      due = -Infinity;
      // Still true if a switch away and back lands before the loop noticed: the
      // chain already running carries on rather than a second one starting.
      if (m === 'uncapped' && !pumping) { pumping = true; channel.port2.postMessage(0); }
      return true;
    }
  };
}

export function describePacing(mode) {
  if (mode === 'vsync') return 'every refresh (vsync)';
  if (mode === 'uncapped') return 'uncapped - benchmark only, extra frames are never shown';
  return `capped at ${mode} fps`;
}
