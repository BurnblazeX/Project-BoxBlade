// --- Game time: the 60 Hz tick and the world clock ---
//
// The simulation steps in fixed ticks of 1/60 s, whatever the frame rate. Each
// frame hands advance() the real time that passed and it runs as many whole
// ticks as that covers - two at 30 fps, none or one at 144 - so game speed
// never depends on the frame rate. The remainder is alpha, the fraction of a
// tick the frame sits past the last one; the view draws moving things between
// their last two tick positions by it (createTickSmoothing).
//
// A frame runs at most MAX_TICKS_PER_FRAME (0.1 s, the old dt clamp): after a
// stall - a tab switch, a compile - the backlog is dropped, not caught up,
// or a slow frame would owe more ticks than the next could run.
//
// Slower systems run every n ticks, n dividing into 60 (2, 3, 4, 5, 6, ...),
// and are handed dt = n / 60. Each gets the phase within its n that collides
// least with the slower systems already there, so a half-, a quarter- and a
// sixth-rate system do not all land on tick 0.
//
// The world clock is game time: what the hour is, how long a spell lasts. It
// is counted in ticks as well, but it is not the tick - in explore it advances
// one per tick, in battle it holds still and each round adds ROUND_TICKS.
// Pure: headless, tested in node.

export const TICK_RATE = 60;
export const TICK_SECONDS = 1 / TICK_RATE;
export const MAX_TICKS_PER_FRAME = 6;

// A 48-minute day: 2 s a game minute, 30x real time. Any multiple of 24 real
// minutes keeps a game second a whole number of ticks; 48 makes it 2.
export const DAY_REAL_MINUTES = 48;
export const DAY_TICKS = DAY_REAL_MINUTES * 60 * TICK_RATE;   // 172,800
export const HOUR_TICKS = DAY_TICKS / 24;                     // 7,200
export const GAME_SECONDS_PER_TICK = 86400 / DAY_TICKS;       // 0.5
export const ticksForGameSeconds = s => Math.round(s / GAME_SECONDS_PER_TICK);
// A D&D round is 6 s of game time.
export const ROUND_TICKS = ticksForGameSeconds(6);            // 12

const gcd = (a, b) => b ? gcd(b, a % b) : a;

// The phase in 0..n-1 that the systems already scheduled share least. Two
// systems meet on some tick when their phases agree modulo gcd(n, m); each
// meeting is weighted by how often the other runs. Every-tick systems meet
// every phase alike and are left out.
export function quietestPhase(n, systems) {
  let best = 0, bestLoad = Infinity;
  for (let p = 0; p < n; p++) {
    let load = 0;
    for (const s of systems) {
      if (s.n === 1) continue;
      if ((p - s.phase) % gcd(n, s.n) === 0) load += 1 / s.n;
    }
    if (load < bestLoad) { best = p; bestLoad = load; }
  }
  return best;
}

export function createTickClock() {
  let tick = 0, acc = 0;
  const systems = [];
  let timers = [];

  // fn(dt, tick) every n ticks. Returns a function that removes it.
  function every(n, fn, phase = quietestPhase(n, systems)) {
    const s = { n, phase: phase % n, fn };
    systems.push(s);
    return () => { const i = systems.indexOf(s); if (i >= 0) systems.splice(i, 1); };
  }

  // fn() once, `ticks` ticks from now (at least the next). Returns a cancel.
  function after(ticks, fn) {
    const t = { at: tick + Math.max(1, Math.round(ticks)), fn };
    timers.push(t);
    return () => { t.fn = null; };
  }

  function step() {
    tick++;
    for (const s of systems.slice()) if (tick % s.n === s.phase) s.fn(s.n * TICK_SECONDS, tick);
    if (timers.length) {
      // Taken out before they run: one may schedule another.
      const due = timers.filter(t => t.at <= tick);
      if (due.length) {
        timers = timers.filter(t => t.at > tick);
        for (const t of due) if (t.fn) t.fn();
      }
    }
  }

  // Real seconds since the last frame; returns how many ticks ran.
  function advance(seconds) {
    acc += Math.max(0, seconds);
    let n = 0;
    while (acc >= TICK_SECONDS && n < MAX_TICKS_PER_FRAME) { step(); acc -= TICK_SECONDS; n++; }
    // Past the cap: the whole ticks still owed are dropped, the fraction kept.
    if (acc >= TICK_SECONDS) acc %= TICK_SECONDS;
    return n;
  }

  return {
    get tick() { return tick; },
    // 0..1: how far this frame is past the last tick.
    get alpha() { return acc / TICK_SECONDS; },
    every, after, step, advance
  };
}

// Game time in ticks. Hours run 0-24.
//   live     advances with the tick (explore), or only by advance() (a
//            battle's rounds)
//   running  whether game time passes at all; stopped, neither moves it
//            (the day cycle off - bxb.daycycle)
export function createWorldClock(startHour = 0) {
  let ticks = Math.round(startHour * HOUR_TICKS);
  return {
    live: true,
    running: true,
    get ticks() { return ticks; },
    get day() { return Math.floor(ticks / DAY_TICKS); },
    // alpha: the tick clock's, so a running clock reads smoothly between ticks.
    hour(alpha = 0) {
      const t = ticks + (this.live && this.running ? alpha : 0);
      return (t - Math.floor(t / DAY_TICKS) * DAY_TICKS) / HOUR_TICKS;
    },
    tick() { if (this.live && this.running) ticks++; },
    advance(n) { if (this.running) ticks += n; },
    // To that hour of the current day.
    setHour(h) {
      const hh = ((Number(h) % 24) + 24) % 24;
      ticks = Math.floor(ticks / DAY_TICKS) * DAY_TICKS + Math.round(hh * HOUR_TICKS);
    }
  };
}

// Positions moved by the ticks, drawn between their last two tick values.
// objects: anything with a .position {x, y, z} (sprites). Per frame:
//   restore()     before the ticks: each back to where the last tick left it
//   capture()     at the start of each tick: that tick's starting point
//   show(alpha)   after the ticks: drawn alpha of the way from the last
//                 tick's start to its end
// Something else may move an object between frames (a battle snapping it to
// its tile, a teleport): restore() sees it is not where show() put it and
// takes the new place as both ends, so it jumps there rather than sliding.
export function createTickSmoothing(objects) {
  const v = () => ({ x: 0, y: 0, z: 0 });
  const set = (a, b) => { a.x = b.x; a.y = b.y; a.z = b.z; };
  const same = (a, b) => a.x === b.x && a.y === b.y && a.z === b.z;
  const items = objects.map(o => {
    const it = { o, prev: v(), cur: v(), shown: v() };
    set(it.prev, o.position); set(it.cur, o.position); set(it.shown, o.position);
    return it;
  });
  return {
    restore() {
      for (const it of items) {
        if (same(it.o.position, it.shown)) set(it.o.position, it.cur);
        else { set(it.cur, it.o.position); set(it.prev, it.o.position); }
        set(it.shown, it.cur);   // so a second restore is harmless
      }
    },
    capture() {
      for (const it of items) set(it.prev, it.o.position);
    },
    show(alpha) {
      for (const it of items) {
        const p = it.o.position;
        set(it.cur, p);
        p.x = it.prev.x + (it.cur.x - it.prev.x) * alpha;
        p.y = it.prev.y + (it.cur.y - it.prev.y) * alpha;
        p.z = it.prev.z + (it.cur.z - it.prev.z) * alpha;
        set(it.shown, p);
      }
    }
  };
}
