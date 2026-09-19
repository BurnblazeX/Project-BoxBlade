import { file, section, ok, near, truthy, falsy } from './lib/harness.mjs';
import { createFrameStats } from '../js/perf.js';
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
