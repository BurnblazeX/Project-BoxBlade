import { file, section, ok } from './lib/harness.mjs';
import { installUploads, uploadFlags, tableGroup } from '../js/gpu/uploads.js';

file('uploads.test.mjs - fewer writeBuffer calls for the same bytes');

// A stand-in renderer: three's bindingUtils.updateBinding as r186 has it (one
// write per run of touching ranges, the whole buffer with no ranges), and a
// queue that records every write.
function rig() {
  const writes = [];
  const gpuBuffers = new Map();
  const backend = {
    device: { queue: { writeBuffer: (buf, off, data, dOff = 0, size) =>
      writes.push({ buf, off, words: Array.from(data.slice(dOff, size === undefined ? undefined : dOff + size)) }) } },
    get: b => { if (!gpuBuffers.has(b)) gpuBuffers.set(b, { buffer: { id: gpuBuffers.size } }); return gpuBuffers.get(b); }
  };
  backend.bindingUtils = {
    updateBinding(binding) {
      const a = binding.buffer, buffer = backend.get(binding).buffer, ranges = binding.updateRanges;
      if (ranges.length === 0) { backend.device.queue.writeBuffer(buffer, 0, a, 0); return; }
      let start = ranges[0].start;
      for (let i = 0; i < ranges.length; i++) {
        const r = ranges[i], next = ranges[i + 1], end = r.start + r.count;
        if (next !== undefined && next.start === end) continue;
        backend.device.queue.writeBuffer(buffer, start * 4, a, start, end - start);
        if (next !== undefined) start = next.start;
      }
    }
  };
  const renderer = { backend, render() {}, compute() {} };
  installUploads(renderer);
  return { renderer, writes, update: b => backend.bindingUtils.updateBinding(b) };
}

section('coalesce: one write per changed buffer');
{
  uploadFlags.coalesce = true; uploadFlags.unchanged = true;
  const { writes, update } = rig();
  // Three vec3s, each padded to four floats: three ranges that never touch.
  const group = { buffer: new Float32Array(12), updateRanges: [] };
  group.buffer.set([1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0]);
  group.updateRanges.push({ start: 0, count: 3 }, { start: 4, count: 3 }, { start: 8, count: 3 });
  update(group);
  ok('one call', writes.length, 1);
  ok('from the first changed value to the last', writes[0].off, 0);
  ok('... the same bytes three would have written, and the padding between', writes[0].words.join(), '1,2,3,0,4,5,6,0,7,8,9');
  writes.length = 0;
  group.updateRanges = [{ start: 4, count: 3 }, { start: 8, count: 1 }];
  update(group);
  ok('a later span starts at its first change', writes.length === 1 && writes[0].off === 16, true);
  uploadFlags.coalesce = false;
  writes.length = 0;
  group.updateRanges = [{ start: 0, count: 3 }, { start: 4, count: 3 }];
  update(group);
  ok('off: three r186 as it was (a call per range)', writes.length, 2);
  uploadFlags.coalesce = true;
}

section('unchanged: a table goes up only when its bytes change');
{
  const { renderer, writes, update } = rig();
  const table = { buffer: new Float32Array(8).fill(1), updateRanges: [] };
  update(table);
  ok('the first upload happens', writes.length, 1);
  renderer.render();
  update(table);
  ok('the same bytes next call: skipped', writes.length, 1);
  renderer.compute();
  table.buffer[5] = 2;
  update(table);
  ok('one value moved: uploaded', writes.length, 2);
  update(table);
  ok('asked again in the same call (a shared table, another object): checked once', writes.length, 2);
  renderer.render();
  const ints = new Uint32Array(table.buffer.buffer);
  ints[0] = 0x7fc00001;   // a NaN pattern: compared as bits, not as floats
  update(table);
  ok('an int written into the float view counts as a change', writes.length, 3);
  renderer.render();
  update(table);
  ok('... and is then the same again', writes.length, 3);
  uploadFlags.unchanged = false;
  renderer.render();
  update(table);
  ok('off: every call uploads, as before', writes.length, 4);
  uploadFlags.unchanged = true;
}

section('the shared table group');
ok('shared, so three binds one copy per pipeline, not one per object', tableGroup.shared, true);
ok('asked every object (updateBinding answers once a call)', tableGroup.updateType, 'object');
ok('after the per-object group', tableGroup.order > 1, true);

section('the shared tables are visible to every stage');
{
  const { NodeBuilder } = await import('three/webgpu');
  // One context for both, as for a material compileAsync builds outside a render.
  const renderer = {};
  const make = (material, vis) => NodeBuilder.prototype._getBindGroup.call(
    { material, renderer }, 'bxbTables',
    [{ groupNode: tableGroup, nodeUniform: { id: 1 }, visibility: vis }]);
  const kernelFirst = make(null, 4);
  ok('built for a kernel: vertex, fragment and compute', kernelFirst.bindings[0].visibility, 7);
  ok('a render pipeline handed the same group can read it in its fragment stage',
     make({}, 2).bindings[0].visibility & 2, 2);
}
