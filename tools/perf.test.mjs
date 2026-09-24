import { file, section, ok, near, truthy, falsy } from './lib/harness.mjs';
import { createFrameStats, nextRedraw, DEFAULT_DRAW_HZ, WINDOW_MS, worstPerColumn } from "../js/perf.js";
import { createConsole, installConsole } from '../js/console.js';

file('perf.test.mjs - frame stats ring buffer and bxb console');

section('ring buffer');
const s = createFrameStats(4);
ok('starts empty', s.count, 0);
ok('empty stats do not divide by zero', s.stats().fps, 0);

s.push(10); s.push(20); s.push(30);
ok('counts pushes', s.count, 3);
ok('keeps insertion order, oldest first', s.toArray().join(','), '10,20,30');

s.push(40); s.push(50);
ok('caps at capacity', s.count, 4);
ok('evicts the oldest sample', s.toArray().join(','), '20,30,40,50');
ok('wraps without reordering', s.toArray()[3], 50);

section('statistics');
const t = createFrameStats(100);
for (const ms of [16, 16, 16, 16, 32]) t.push(ms);
const st = t.stats();
ok('last is the newest sample', st.last, 32);
near('min', st.min, 16);
near('max', st.max, 32);
near('average', st.avg, (16 * 4 + 32) / 5);
near('fps derives from the average', st.fps, 1000 / st.avg, 1e-6);
truthy('1% low is worse than the average fps', st.low1 <= st.fps);
truthy('p95 is reported for graph scaling', st.p95 >= st.avg);

section('a single hitch must not flatten the graph');
const h = createFrameStats(100);
for (let i = 0; i < 99; i++) h.push(16);
h.push(4560); // a chunk rebuild, not a frame
const hs = h.stats();
near('max still reports the hitch', hs.max, 4560);
truthy('but p95, which sets the graph scale, ignores it', hs.p95 < 100);

section('reset');
t.reset();
ok('reset clears the buffer', t.count, 0);
ok('and the derived stats', t.stats().max, 0);

section('bxb console');
let toggled = 0;
const api = createConsole({
  perf:  { help: 'toggle the frame time graph', run: () => { toggled++; return 'ok'; } },
  tp:    { help: 'teleport', usage: 'bxb.tp(x, z, y?)', run: (x, z) => `${x},${z}` }
});
truthy('commands become callable functions', typeof api.perf === 'function');
ok('a command runs and returns', api.perf(), 'ok');
ok('and actually fired', toggled, 1);
ok('arguments pass through', api.tp(3, 7), '3,7');
ok('help text is attached to each command', api.perf.help, 'toggle the frame time graph');
ok('custom usage is kept', api.tp.usage, 'bxb.tp(x, z, y?)');
ok('default usage is derived from the name', api.perf.usage, 'bxb.perf()');

const helpText = api.help();
truthy('help lists every command', helpText.includes('bxb.perf()') && helpText.includes('bxb.tp(x, z, y?)'));
truthy('help includes the descriptions', helpText.includes('teleport'));

section('installation');
const target = {};
installConsole(api, target);
truthy('installs under the bxb namespace', !!target.bxb);
ok('and exposes the commands', target.bxb.tp(1, 2), '1,2');
falsy('does not leak other globals', Object.keys(target).length !== 1);

section('overlay repaint throttle');
ok('capped at 60 a second', DEFAULT_DRAW_HZ, 60);
truthy('first repaint is immediate', nextRedraw(0, -Infinity, 60).draw);
// Repaints over one second of frames at a given fps, with a little jitter.
const repaints = (fps, jitter = 0) => {
  let due = -Infinity, n = 0;
  for (let i = 1; i <= fps; i++) {
    const now = i * 1000 / fps + (i % 2 ? jitter : -jitter);
    const r = nextRedraw(now, due, 60);
    due = r.due;
    if (r.draw) n++;
  }
  return n;
};
ok('240 fps: 60 repaints', repaints(240), 60);
ok('144 fps: 60 repaints, not 48', repaints(144), 60);
ok('60 fps with vsync jitter: every frame', repaints(60, 0.3), 60);
ok('45 fps: every frame (the game rate)', repaints(45), 45);
ok('20 fps: every frame', repaints(20), 20);
truthy('0 Hz repaints every frame', nextRedraw(1000.1, 5000, 0).draw);

section('fixed-length window');
// One second of frames at a given fps, then the same window's contents.
const windowed = fps => {
  const w = createFrameStats(8192, WINDOW_MS);
  for (let i = 1; i <= fps * 5; i++) w.push(1000 / fps, i * 1000 / fps);
  return w;
};
ok('60 fps: 3 s holds 181 frames (both ends)', windowed(60).count, 181);
ok('240 fps: 3 s holds 721 frames, not 180', windowed(240).count, 721);
near('fps over the window', windowed(240).stats().fps, 240, 1e-3);
const stall = createFrameStats(8192, WINDOW_MS);
for (let i = 1; i <= 600; i++) stall.push(16.7, i * 16.7);
ok('no new frames (a stall): the picture freezes, not empties', stall.count, 180);
ok('newest is the last push', stall.newest, 600 * 16.7);
const hitch = createFrameStats(8192, WINDOW_MS);
hitch.push(16, 1000); hitch.push(4000, 5000); hitch.push(16, 5016);
ok('a 4 s hitch pushes older frames out of the window', hitch.count, 2);
ok('without a window it is a plain ring buffer', createFrameStats(3).capacity, 3);

section('columns by time');
const W = 240;
// 3 s of 60 fps frames with one 50 ms spike in the middle.
const cv = [], ct = [];
for (let i = 1; i <= 180; i++) { cv.push(i === 90 ? 50 : 16.7); ct.push(i * 16.7); }
const end = ct[ct.length - 1];
const cols = worstPerColumn(cv, ct, end, WINDOW_MS, W);
truthy('60 fps leaves no empty columns', [...cols].slice(1).every(i => i >= 0));
truthy('the spike is drawn', [...cols].some(i => cv[i] === 50));
truthy('a 50 ms spike is 4-5 columns of 12.5 ms wide', [4, 5].includes([...cols].filter(i => cv[i] === 50).length));
ok('the newest frame lands in the last column', cols[W - 1], 179);
const fast = [], ft = [];
for (let i = 1; i <= 720; i++) { fast.push(i === 400 ? 30 : 4.17); ft.push(i * 4.17); }
const fcols = worstPerColumn(fast, ft, ft[ft.length - 1], WINDOW_MS, W);
truthy('240 fps: a spike among fast frames survives the column', [...fcols].some(i => fast[i] === 30));
const long = worstPerColumn([4000], [5000], 5000, WINDOW_MS, W);
truthy('a hitch longer than the window fills every column', [...long].every(i => i === 0));

section('stats over the newest n');
const gs = createFrameStats(8192);
for (let i = 0; i < 1000; i++) gs.push(i < 900 ? 10 : 2);
ok('n caps the count', gs.stats(100).count, 100);
near('and the average covers only those', gs.stats(100).avg, 2);
ok('fewer samples than n: all of them', createFrameStats(8192).stats(120).count, 0);
ok('no n: everything held', gs.stats().count, 1000);
