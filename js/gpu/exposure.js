import * as THREE from 'three/webgpu';
import { Fn, If, float, vec3, int, uint, ivec2, instanceIndex, storage, textureLoad, log2, max, clamp,
         atomicAdd, atomicStore } from 'three/tsl';
import { METER_SAMPLES, METER_GRID, METER_LOG_MIN, METER_LOG_SCALE, METER_LOG_MAX_Q } from '../core/exposure.js';

// --- The exposure meter (core/exposure.js) ---
//
// Two words: the fixed-point sum of log2 luminance, and how many samples
// were terrain. Cleared and filled by one submit of [clear, meter]; read back
// by the tone map feature (render/features/tonemap.js) a few frames late.
//
// What it meters is the terrain's lit colour before exposure and tone mapping,
// wherever the showing shader set keeps it: split shading's result buffer
// (colour, texel id - id 0 is no terrain), or the texel cache's low-res target
// (the same layout). The sky and sprites are not metered: the terrain is most
// of any view, and it is what decides whether a room reads as dark.

const LUMA = vec3(0.2126, 0.7152, 0.0722);

export function createExposureMeter() {
  const attr = new THREE.StorageBufferAttribute(new Uint32Array(2), 1);
  const acc = storage(attr, 'uint', 2).toAtomic().setName('bxbExposureMeter');
  const clear = Fn(() => {
    atomicStore(acc.element(int(0)), uint(0));
    atomicStore(acc.element(int(1)), uint(0));
  })().compute(1).setName('exposureClear');
  // One sample's colour and id into the sum.
  const add = s => {
    If(s.w.greaterThan(float(0)), () => {
      const l = log2(max(s.xyz.dot(LUMA), float(2 ** METER_LOG_MIN)));
      const q = clamp(l.sub(float(METER_LOG_MIN)).mul(float(METER_LOG_SCALE)), float(0), float(METER_LOG_MAX_Q));
      atomicAdd(acc.element(int(0)), uint(q));
      atomicAdd(acc.element(int(1)), uint(1));
    });
  };
  // split: createSplitShading's buffers - samples spread evenly over this
  // frame's count. In float: count * i overflows an int at full screen sizes.
  const splitKernel = split => Fn(() => {
    const i = float(int(instanceIndex));
    const idx = int(i.add(float(0.5)).mul(float(split.count)).div(float(METER_SAMPLES))).toVar();
    If(idx.lessThan(split.count), () => { add(split.result.ro.element(idx)); });
  })().compute(METER_SAMPLES).setName('exposureMeterSplit');
  // cache: the texel cache's target bindings - a grid over its size.
  const targetKernel = cache => Fn(() => {
    const i = int(instanceIndex);
    const gx = float(i.mod(int(METER_GRID))).add(float(0.5)).div(float(METER_GRID));
    const gy = float(i.div(int(METER_GRID))).add(float(0.5)).div(float(METER_GRID));
    add(textureLoad(cache.tex, ivec2(int(gx.mul(cache.size.x)), int(gy.mul(cache.size.y)))));
  })().compute(METER_SAMPLES).setName('exposureMeterTarget');
  const made = new Map();
  return {
    attr,
    // [clear, meter] for a source ({ split } or { cache }), built once per source.
    kernels(source) {
      const key = source.split || source.cache;
      if (!made.has(key)) {
        made.set(key, [clear, source.split ? splitKernel(source.split) : targetKernel(source.cache)]);
      }
      return made.get(key);
    }
  };
}
