import { file, section, ok } from './lib/harness.mjs';
import { createTickClock, createWorldClock, createTickSmoothing, quietestPhase, TICK_SECONDS,
         MAX_TICKS_PER_FRAME, DAY_TICKS, HOUR_TICKS, ROUND_TICKS, ticksForGameSeconds } from '../js/core/clock.js';

file('clock.test.mjs - the 60 Hz tick, slower systems, the world clock');

section('ticks follow real time, not frames');
{
  const run = fps => { const c = createTickClock(); for (let i = 0; i < fps; i++) c.advance(1 / fps); return c.tick; };
  ok('30 fps: 60 ticks a second', run(30), 60);
  ok('60 fps: 60', Math.abs(run(60) - 60) <= 1, true);
  ok('144 fps: 60', Math.abs(run(144) - 60) <= 1, true);
  const c = createTickClock();
  c.advance(TICK_SECONDS * 0.25);
  ok('a quarter tick: none run', c.tick, 0);
  ok('... and alpha says so', Math.abs(c.alpha - 0.25) < 1e-9, true);
  ok('a stall runs only the cap', c.advance(1), MAX_TICKS_PER_FRAME);
  ok('... and drops the rest', c.advance(0), 0);
}

section('slower systems');
{
  ok('the first half-rate system takes phase 0', quietestPhase(2, []), 0);
  ok('a second avoids it', quietestPhase(2, [{ n: 2, phase: 0 }]), 1);
  ok('a quarter avoids the half', quietestPhase(4, [{ n: 2, phase: 0 }]) % 2, 1);
  const c = createTickClock();
  const runs = { half: [], quarter: [], sixth: [] }, dts = {};
  c.every(2, (dt, t) => { runs.half.push(t); dts.half = dt; });
  c.every(4, (dt, t) => { runs.quarter.push(t); dts.quarter = dt; });
  c.every(6, (dt, t) => { runs.sixth.push(t); dts.sixth = dt; });
  c.every(1, () => {});
  for (let i = 0; i < 120; i++) c.step();
  ok('half rate: 30 a second', runs.half.length, 60);
  ok('quarter: 15', runs.quarter.length, 30);
  ok('sixth: 10', runs.sixth.length, 20);
  ok('each is handed its own dt', [dts.half, dts.quarter, dts.sixth].map(d => Math.round(d * 60)).join(), '2,4,6');
  const all = [...runs.half, ...runs.quarter, ...runs.sixth];
  const most = Math.max(...all.map(t => all.filter(u => u === t).length));
  ok('never all three on one tick', most < 3, true);
  const stop = c.every(3, () => runs.extra = (runs.extra || 0) + 1);
  c.step(); c.step(); c.step(); stop(); c.step(); c.step(); c.step();
  ok('a removed system stops', runs.extra, 1);
}

section('after');
{
  const c = createTickClock();
  const seen = [];
  c.after(3, () => seen.push(c.tick));
  const cancel = c.after(2, () => seen.push('cancelled'));
  c.after(1, () => c.after(1, () => seen.push('chained ' + c.tick)));
  cancel();
  for (let i = 0; i < 5; i++) c.step();
  ok('runs on its tick, chains, cancels', seen.join(), 'chained 2,3');
}

section('the world clock');
{
  ok('a game second is 2 ticks', ticksForGameSeconds(1), 2);
  ok('a game minute is 2 real seconds', ticksForGameSeconds(60) * TICK_SECONDS, 2);
  ok('a round is 12 ticks', ROUND_TICKS, 12);
  ok('a 1-minute spell is 10 rounds', ticksForGameSeconds(60) / ROUND_TICKS, 10);
  ok('a day is 48 real minutes', DAY_TICKS * TICK_SECONDS / 60, 48);
  const w = createWorldClock(9);
  ok('starts at its hour', w.hour(), 9);
  for (let i = 0; i < HOUR_TICKS; i++) w.tick();
  ok('an hour of ticks is an hour', w.hour(), 10);
  ok('reads between ticks', w.hour(0.5), 10 + 0.5 / HOUR_TICKS);
  w.live = false;
  w.tick();
  ok('not live: ticks leave it', w.hour(), 10);
  ok('... and alpha too', w.hour(0.5), 10);
  w.advance(ROUND_TICKS);
  ok('a round moves it 6 s', w.ticks - 10 * HOUR_TICKS, 12);
  w.live = true;
  w.advance(DAY_TICKS);
  ok('past midnight: the next day', w.day, 1);
  w.setHour(23.5);
  ok('setHour keeps the day', w.day === 1 && w.hour() === 23.5, true);
  w.running = false;
  w.tick(); w.advance(ROUND_TICKS);
  ok('stopped: nothing moves it', w.hour(0.5), 23.5);
}

section('smoothing between ticks');
{
  const s = { position: { x: 0, y: 0, z: 0 } };
  const sm = createTickSmoothing([s]);
  const frame = (dx, ticks, alpha) => {
    sm.restore();
    for (let i = 0; i < ticks; i++) { sm.capture(); s.position.x += dx; }
    sm.show(alpha);
    return s.position.x;
  };
  ok('drawn between the last tick\'s ends', frame(1, 1, 0.25), 0.25);
  ok('restored before the next ticks', (sm.restore(), s.position.x), 1);
  ok('no tick this frame: further along the same one', frame(1, 0, 0.75), 0.75);
  ok('two ticks: between the last two', frame(1, 2, 0.5), 2.5);
  s.position.x = 10;   // moved outside the ticks
  ok('a jump outside the ticks is not slid across', frame(0, 1, 0.5), 10);
}
