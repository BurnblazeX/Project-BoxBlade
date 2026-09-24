import { file, section, ok, truthy, falsy } from './lib/harness.mjs';
import { paceTick, normalisePacing, createFramePacer, PACING_MODES, DEFAULT_PACING, DEFAULT_IN_FLIGHT, MAX_IN_FLIGHT }
  from '../js/pacing.js';

file('pacing.test.mjs - frame pacing caps and modes');

// Frames rendered over one second of refreshes at `hz`, under `mode`.
const rendered = (hz, mode, jitter = 0) => {
  let due = -Infinity, n = 0;
  for (let i = 1; i <= hz; i++) {
    const now = i * 1000 / hz + (i % 2 ? jitter : -jitter);
    const r = paceTick(now, due, mode);
    due = r.due;
    if (r.draw) n++;
  }
  return n;
};

section('caps');
ok('vsync renders every refresh at 60 Hz', rendered(60, 'vsync'), 60);
ok('vsync renders every refresh at 144 Hz', rendered(144, 'vsync'), 144);
ok('cap 30 on 60 Hz: every other refresh', rendered(60, 30), 30);
ok('cap 60 on 60 Hz with vsync jitter: every refresh', rendered(60, 60, 0.3), 60);
ok('cap 60 on 144 Hz averages 60', rendered(144, 60), 60);
ok('cap 30 on 144 Hz averages 30', rendered(144, 30), 30);
ok('cap 40 on 60 Hz averages 40', rendered(60, 40), 40);
ok('cap 60 on a 40 Hz device: every refresh', rendered(40, 60), 40);

section('modes');
ok('default is vsync', DEFAULT_PACING, 'vsync');
ok('the five modes', PACING_MODES.join(','), 'vsync,30,40,60,uncapped');
ok("'30' from a <select> is 30", normalisePacing('30'), 30);
ok('unknown cap rejected', normalisePacing(45), null);
ok('unknown name rejected', normalisePacing('triple'), null);
ok('empty string rejected', normalisePacing(''), null);

section('pacer');
// A stand-in renderer: the pacer only touches info and the node frame.
const renderer = { info: { autoReset: true, reset() {}, frame: 0 },
                   _nodes: { nodeFrame: { frameId: 0, update() { this.frameId++; } } } };
let frames = 0;
const pacer = createFramePacer(renderer, () => { frames++; });
pacer.onRefresh(16.7);
ok('vsync: a refresh renders', frames, 1);
truthy('set a cap', pacer.setMode(30));
frames = 0;
for (let i = 1; i <= 60; i++) pacer.onRefresh(1000 + i * 1000 / 60);
ok('cap 30: 30 of 60 refreshes render', frames, 30);
falsy('unknown mode refused', pacer.setMode('double'));
ok('refused mode leaves the cap', pacer.mode, 30);

truthy('uncapped accepted', pacer.setMode('uncapped'));
frames = 0;
pacer.onRefresh(5000);
ok('uncapped: refreshes do not render', frames, 0);
// The loop runs on MessageChannel tasks: let a few go by, then stop it.
await new Promise(r => setTimeout(r, 30));
truthy('uncapped: the loop renders on its own', frames > 0);
ok('uncapped: the node frame advances per frame', renderer._nodes.nodeFrame.frameId, frames);
pacer.setMode('vsync');
await new Promise(r => setTimeout(r, 10));
const after = frames;
await new Promise(r => setTimeout(r, 20));
ok('back to vsync: the loop stops', frames, after);

section('uncapped: frames in flight');
// A GPU that finishes only when told to.
const pending = [];
const gpuRenderer = {
  info: { autoReset: true, reset() {}, frame: 0 },
  _nodes: { nodeFrame: { frameId: 0, update() { this.frameId++; } } },
  backend: { device: { queue: {
    onSubmittedWorkDone: () => new Promise(r => pending.push(r))
  } } }
};
let drawn = 0;
const held = createFramePacer(gpuRenderer, () => { drawn++; });
held.setMode('uncapped');
await new Promise(r => setTimeout(r, 30));
ok(`a stalled GPU holds the loop at ${DEFAULT_IN_FLIGHT} frames`, drawn, DEFAULT_IN_FLIGHT);
pending.shift()();
await new Promise(r => setTimeout(r, 30));
ok('one frame finishing lets exactly one more through', drawn, DEFAULT_IN_FLIGHT + 1);
held.setMode('vsync');
while (pending.length) pending.shift()();
await new Promise(r => setTimeout(r, 30));
ok('leaving uncapped while blocked stops the loop', drawn, DEFAULT_IN_FLIGHT + 1);
// A lost device rejects its promises; the loop must not die with it.
const lost = { ...gpuRenderer, backend: { device: { queue: {
  onSubmittedWorkDone: () => Promise.reject(new Error('device lost')) } } } };
let lostDrawn = 0;
const survivor = createFramePacer(lost, () => { lostDrawn++; });
survivor.setMode('uncapped');
await new Promise(r => setTimeout(r, 30));
truthy('a rejected wait does not stop the loop', lostDrawn > DEFAULT_IN_FLIGHT);
survivor.setMode('vsync');
await new Promise(r => setTimeout(r, 10));

section('uncapped: queue depth');
const q = [];
const stub = { ...gpuRenderer, backend: { device: { queue: {
  onSubmittedWorkDone: () => new Promise(r => q.push(r)) } } } };
let n = 0;
const deep = createFramePacer(stub, () => { n++; });
ok('default depth', deep.depth, DEFAULT_IN_FLIGHT);
deep.depth = 99;
ok('depth clamps to the maximum', deep.depth, MAX_IN_FLIGHT);
deep.depth = 5;
deep.setMode('uncapped');
await new Promise(r => setTimeout(r, 30));
ok('a stalled GPU holds the loop at the set depth', n, 5);
deep.depth = 2;
q.shift()();
await new Promise(r => setTimeout(r, 30));
const lowered = n;
q.splice(0).forEach(r => r());
await new Promise(r => setTimeout(r, 30));
truthy('lowering the depth while queued does not wedge the loop', n > lowered);
deep.depth = 0;
const before0 = n;
await new Promise(r => setTimeout(r, 30));
truthy('depth 0 is unbounded: runs with the GPU never finishing', n - before0 > 10);
deep.setMode('vsync');
await new Promise(r => setTimeout(r, 10));
