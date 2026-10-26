import * as THREE from 'three/webgpu';
import { sharedUniformGroup, uniform, int, uint, float } from 'three/tsl';

// --- Getting uniforms to the GPU in fewer calls ---
//
// Firefox makes a new D3D12 staging buffer and a shared-memory section for
// every queue.writeBuffer, so its WebGPU thread pays per call, not per byte:
// writeBuffer was 39% of CanvasRenderer with the effects on (profile
// 2026-10-26, ~1.9 ms of 4.8 a frame). three r186 makes more calls than the
// data needs, in three ways, each answered here:
// - One call per changed uniform. updateBinding merges ranges only where they
//   touch, and a vec3's padding breaks every run, so a 42-field object struct
//   can cost a call per field. coalesce: one call over the changed span.
// - A uniformArray is rewritten whole every time it is bound
//   (UniformBuffer.update is always true). unchanged: the bytes are compared
//   with the last ones written to that buffer, and an identical upload is
//   skipped.
// - A uniform group that is not shared is cloned per render object, so each
//   sprite and twin carried its own copy of the light list, the cards, the
//   mirrors and the sky. The tables are in tableGroup, a shared group: one
//   copy per pipeline (per render target), checked once per render or compute
//   call.
// All of it writes the same bytes to the same places, so the image cannot
// change. bxb.uploads() turns coalesce / unchanged off for A/B; the shared
// group is part of the shaders (it moves the tables to their own bind group).

// The uniformArray tables: bound once per pipeline, not per object. OBJECT
// update type, so three asks every object - updateBinding below answers once
// per call.
export const tableGroup = sharedUniformGroup('bxbTables', 2, THREE.NodeUpdateType.OBJECT);
export const sharedTable = node => node.setGroup(tableGroup);

// A shared bind group is cached by its uniforms alone - per render context,
// and a material that compileAsync builds outside a render shares the
// renderer's own context with the kernels - so one BindGroup reaches render
// and compute pipelines alike, carrying the stage flags of whichever shader
// made it. A table first built for a kernel then failed in a fragment shader
// ("Visibility flags don't include the shader stage"). So the tables are
// visible to every stage, as three's own uniform groups always are.
const ALL_STAGES = 1 | 2 | 4;   // GPUShaderStage VERTEX | FRAGMENT | COMPUTE
const baseGetBindGroup = THREE.NodeBuilder.prototype._getBindGroup;
THREE.NodeBuilder.prototype._getBindGroup = function (groupName, bindings) {
  if (bindings[0] && bindings[0].groupNode === tableGroup) {
    for (const b of bindings) b.visibility = ALL_STAGES;
  }
  return baseGetBindGroup.call(this, groupName, bindings);
};

// --- The per-frame values: one buffer for every shader ---
//
// A few values change every frame however still the view is: the hit cache's
// half, texel GI's frame and blend, the room meter's blend, the smoke clock,
// the eye. Each sat in the per-object struct of every material and kernel
// reading it, so each of those structs was rewritten every frame - ~23 calls
// a frame, the largest share left (bxb.gpucalls, 2026-10-26). A shared group
// alone does not fix that: three shares a group's buffer only between shaders
// with the SAME set of its uniforms, and every kernel reads a different few.
// So a shader that reads any of them declares them all (getUniformFromNode
// below): one struct, one buffer per render target, written once a frame.
// The values are the same uniforms - only where they live moves.
export const frameGroup = sharedUniformGroup('bxbFrame', 3, THREE.NodeUpdateType.OBJECT);
const frameUniforms = [];
function frameUniform(node) {
  node.setGroup(frameGroup);
  frameUniforms.push(node);
  return node;
}
// The values themselves, made here once so the set (and the struct every shader
// declares) never depends on what else was built first. Their owners take them
// from here: gpu/hitcache.js, roommeter.js, smokeshadow.js, smokeview.js,
// texel.js (split shading's eye), texelgi.js.
export const frameValues = {
  hitHalf: frameUniform(uniform(int(0))),          // the hit cache's write half
  roomBlend: frameUniform(uniform(float(1))),      // the room meter's ease
  smokeTime: frameUniform(uniform(float(0)).setName('bxbSmokeTime')),
  smokeViewTime: frameUniform(uniform(float(0))),  // the cubes' clock in secondary rays
  viewPos: frameUniform(uniform(new THREE.Vector3())),
  giFrame: frameUniform(uniform(uint(0))),         // texel GI's frame counter
  giBlend: frameUniform(uniform(float(1)))         // texel GI's temporal blend
};
const baseGetUniform = THREE.WGSLNodeBuilder.prototype.getUniformFromNode;
THREE.WGSLNodeBuilder.prototype.getUniformFromNode = function (node, type, shaderStage, name = null) {
  const out = baseGetUniform.call(this, node, type, shaderStage, name);
  if (node.groupNode === frameGroup && !this.bxbFrameDeclared) {
    this.bxbFrameDeclared = true;
    // As UniformNode.generate would ask for each.
    for (const u of frameUniforms) {
      const shared = u.getSharedNode(this);
      if (shared !== node) baseGetUniform.call(this, shared, shared.getInputType(this), shaderStage, u.name || null);
    }
  }
  return out;
};

export const uploadFlags = { coalesce: true, unchanged: true, batch: true };

// The last bytes written to each whole-buffer binding, by binding: the GPU
// buffer they went to, a copy, and the call that last checked it.
const written = new WeakMap();
let call = 0;

const words = a => new Uint32Array(a.buffer, a.byteOffset, a.byteLength >> 2);
function sameWords(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Before the first frame (the backend's binding utilities exist from the
// renderer's constructor on).
export function installUploads(renderer) {
  const backend = renderer.backend, utils = backend.bindingUtils;
  const base = utils.updateBinding.bind(utils);
  // Every render and compute call is a new chance for a table to change.
  for (const m of ['render', 'compute']) {
    const f = renderer[m].bind(renderer);
    renderer[m] = (...args) => { call++; return f(...args); };
  }
  utils.updateBinding = binding => {
    const array = binding.buffer;
    if (!ArrayBuffer.isView(array) || (array.byteLength & 3)) return base(binding);
    const ranges = binding.updateRanges;
    const gpu = backend.get(binding).buffer;
    // A whole-buffer upload: a table (or a group's first write).
    if (ranges.length === 0) {
      if (!uploadFlags.unchanged) return base(binding);
      const w = words(array);
      let last = written.get(binding);
      if (last && last.gpu === gpu) {
        if (last.call === call) return;   // a shared table, already checked this call
        last.call = call;
        if (sameWords(last.copy, w)) return;
        last.copy.set(w);
      } else {
        written.set(binding, last = { gpu, copy: w.slice(), call });
      }
      return base(binding);
    }
    // Some uniforms changed: one write from the first to the last of them.
    // The rest of the span already holds what the GPU has.
    if (!uploadFlags.coalesce || ranges.length === 1) return base(binding);
    let lo = Infinity, hi = 0;
    for (const r of ranges) {
      if (r.start < lo) lo = r.start;
      if (r.start + r.count > hi) hi = r.start + r.count;
    }
    backend.device.queue.writeBuffer(gpu, lo * array.BYTES_PER_ELEMENT, array, lo, hi - lo);
    // A coalesced write leaves the whole-buffer copy (if any) stale.
    written.delete(binding);
  };
}

// --- Command buffers sent together ---
//
// three submits each render() and compute() call on its own: ~6 a frame with
// the effects on, each with a fixed cost on the browser's WebGPU thread
// (Firefox: queue_submit's pending submission and maintain, ~40-65 us apiece).
// Here a submit is held and sent with the ones after it in ONE
// queue.submit([...]) - at the end of the task (a microtask), or sooner,
// before anything the queue must see after it: a write (writeBuffer,
// writeTexture, copyExternalImageToTexture), a readback (mapAsync),
// onSubmittedWorkDone, or a destroy. Queue order is kept, so the GPU runs the
// same work in the same order: 6.25 submits a frame became 3.1 (Chrome,
// 2026-10-26). One difference: a batch with an invalid command buffer fails
// whole, where alone only its pass would have gone missing.
// bxb.uploads('batch') off: each submit sent as it comes.
export const submitStats = { submits: 0, buffers: 0, installed: false };
export function installSubmitBatching(device) {
  if (!device) return;
  const queue = device.queue;
  const realSubmit = queue.submit.bind(queue);
  const pending = [];
  const flush = () => {
    if (!pending.length) return;
    submitStats.submits++;
    realSubmit(pending.splice(0));
  };
  queue.submit = list => {
    submitStats.buffers += list.length;
    if (!uploadFlags.batch) { flush(); submitStats.submits++; realSubmit(list); return; }
    if (!pending.length) queueMicrotask(flush);
    pending.push(...list);
  };
  for (const name of ['writeBuffer', 'writeTexture', 'copyExternalImageToTexture', 'onSubmittedWorkDone']) {
    const f = queue[name].bind(queue);
    queue[name] = (...args) => { flush(); return f(...args); };
  }
  for (const [proto, name] of [[GPUBuffer.prototype, 'mapAsync'], [GPUBuffer.prototype, 'destroy'],
                               [GPUTexture.prototype, 'destroy']]) {
    const f = proto[name];
    proto[name] = function (...args) { flush(); return f.apply(this, args); };
  }
  submitStats.installed = true;
}

// bxb.uploads(name): flip one of the upload fixes, for A/B.
export function toggleUpload(name) {
  if (!(name in uploadFlags)) return `bxb.uploads('coalesce' | 'unchanged' | 'batch') - now ${JSON.stringify(uploadFlags)}`;
  uploadFlags[name] = !uploadFlags[name];
  return `${name} ${uploadFlags[name] ? 'on' : 'off'}`;
}

// bxb.gpucalls(frames): WebGPU queue calls a frame, measured over that many
// frames - writeBuffer calls and bytes, submits, and the buffers written most
// (three labels a uniform buffer bindingBuffer<id>_<name>_<stages>).
export async function countGPUCalls(renderer, frames = 120) {
  const queue = renderer.backend.device.queue;
  const byName = new Map();
  let writes = 0, bytes = 0, submits = 0;
  // Own properties (the batching wrappers) are put back as they were.
  const ownWb = Object.prototype.hasOwnProperty.call(queue, 'writeBuffer');
  const ownSb = Object.prototype.hasOwnProperty.call(queue, 'submit');
  const wb = queue.writeBuffer, sb = queue.submit;
  const s0 = submitStats.submits, b0 = submitStats.buffers;
  queue.writeBuffer = function (buffer, offset, data, dataOffset = 0, size) {
    const per = data.BYTES_PER_ELEMENT || 1;
    writes++;
    bytes += size !== undefined ? size * per : data.byteLength - dataOffset * per;
    const name = (buffer.label || '(unlabelled)').replace(/^bindingBuffer(\d+|undefined)_(UniformBuffer_)?/, '');
    byName.set(name, (byName.get(name) || 0) + 1);
    return wb.apply(this, arguments);
  };
  queue.submit = function () { submits++; return sb.apply(this, arguments); };
  const f0 = renderer.info.frame;
  try {
    while (renderer.info.frame - f0 < frames) await new Promise(r => setTimeout(r, 50));
  } finally {
    if (ownWb) queue.writeBuffer = wb; else delete queue.writeBuffer;
    if (ownSb) queue.submit = sb; else delete queue.submit;
  }
  // Batched: the submits that reached the queue, and the command buffers in them.
  if (submitStats.installed) submits = submitStats.submits - s0;
  const buffers = submitStats.installed ? submitStats.buffers - b0 : submits;
  const n = Math.max(1, renderer.info.frame - f0);
  const r1 = x => +(x / n).toFixed(1);
  // The whole list printed too: the console folds arrays in the returned object.
  console.table([...byName].sort((a, b) => b[1] - a[1]).map(([buffer, v]) => ({ buffer, perFrame: r1(v) })));
  return {
    frames: n, writeBuffer: r1(writes), kB: +(bytes / n / 1024).toFixed(1), submits: r1(submits),
    commandBuffers: r1(buffers),
    flags: { ...uploadFlags },
    top: [...byName].sort((a, b) => b[1] - a[1]).slice(0, 15).map(([k, v]) => `${r1(v)}  ${k}`)
  };
}
