import * as THREE from 'three/webgpu';
import { Fn, float, uniform, instanceIndex, storage } from 'three/tsl';
import { traceDistanceTSL } from './field.js';
import { NO_SKIP, createCardsTSL } from './cards.js';

// ---------------------------------------------------------------------------
// Compute smoke test
//
// The smallest thing that proves the whole compute path works end to end:
// dispatch a kernel, have it write a storage buffer, read it back, check the
// numbers. Run this before anything depends on compute, because a failure here
// is an environment problem, not a bug in the lighting.
// ---------------------------------------------------------------------------
export async function runComputeSmokeTest(renderer, count = 64) {
  const buffer = new THREE.StorageBufferAttribute(count, 1);
  const target = storage(buffer, 'float', count).setName('bxbSmoke');

  const kernel = Fn(() => {
    // Deliberately not a constant: index-dependent output catches a dispatch
    // that runs once, or runs with the wrong workgroup count.
    target.element(instanceIndex).assign(instanceIndex.toFloat().mul(2).add(1));
  })().compute(count);

  await renderer.computeAsync(kernel);
  const read = new Float32Array(await renderer.getArrayBufferAsync(buffer));

  let mismatches = 0;
  let firstBad = null;
  for (let i = 0; i < count; i++) {
    const want = i * 2 + 1;
    if (read[i] !== want) {
      mismatches++;
      if (firstBad === null) firstBad = { index: i, got: read[i], want };
    }
  }

  const result = { ok: mismatches === 0, count, mismatches, firstBad, sample: [...read.slice(0, 8)] };
  console.log(result.ok
    ? `[gpu] compute smoke test PASSED - ${count} values written and read back`
    : `[gpu] compute smoke test FAILED - ${mismatches}/${count} wrong, first: ${JSON.stringify(firstBad)}`);
  return result;
}

// ---------------------------------------------------------------------------
// CPU/GPU parity
//
// Runs the same rays through the GPU kernel and returns the results, so they
// can be diffed against marchOccupancy() from core/field.js. The whole reason the
// CPU DDA was written first was to have something trustworthy to check this
// against - a shader that silently disagrees is very hard to notice by eye.
// ---------------------------------------------------------------------------
export async function runGPUMarch(renderer, occTex, grid, rays) {
  const n = rays.length;
  const originBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);
  const dirBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);
  const outBuf = new THREE.StorageBufferAttribute(n, 1);

  for (let i = 0; i < n; i++) {
    const r = rays[i];
    const len = Math.hypot(r.dir.x, r.dir.y, r.dir.z) || 1;
    originBuf.array.set([r.origin.x, r.origin.y, r.origin.z, 0], i * 4);
    dirBuf.array.set([r.dir.x / len, r.dir.y / len, r.dir.z / len, r.maxDist], i * 4);
  }

  const origins = storage(originBuf, 'vec4', n).setName('bxbParityOrigins');
  const dirs = storage(dirBuf, 'vec4', n).setName('bxbParityDirs');
  const out = storage(outBuf, 'float', n).setName('bxbParityOut');
  const gridOrigin = uniform(new THREE.Vector3(grid.origin.x, grid.origin.y, grid.origin.z));
  const ring = uniform(new THREE.Vector3(grid.ringX || 0, 0, grid.ringZ || 0));

  const kernel = Fn(() => {
    const o = origins.element(instanceIndex);
    const d = dirs.element(instanceIndex);
    out.element(instanceIndex).assign(
      traceDistanceTSL(occTex, gridOrigin, o.xyz, d.xyz, d.w, ring)
    );
  })().compute(n);

  await renderer.computeAsync(kernel);
  return new Float32Array(await renderer.getArrayBufferAsync(outBuf));
}

// ---------------------------------------------------------------------------
// Card parity: the same rays through createCardsTSL and through cardVisibility
//
// The card path is the one place a silent disagreement is most likely, because
// it samples a texture inside a dynamic loop - and a shader that quietly returns
// the wrong thing there looks like a rendering artifact rather than like a bug.
// Compute rather than raster, which also means it runs without the render loop.
// ---------------------------------------------------------------------------
export async function runGPUCards(renderer, bindings, samples, coneSlope) {
  const n = samples.length;
  const fromBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);
  const dirBuf = new THREE.StorageBufferAttribute(new Float32Array(n * 4), 4);
  const outBuf = new THREE.StorageBufferAttribute(n, 1);

  for (let i = 0; i < n; i++) {
    const s = samples[i];
    fromBuf.array.set([s.from.x, s.from.y, s.from.z, 0], i * 4);
    dirBuf.array.set([s.dir.x, s.dir.y, s.dir.z, s.maxDist], i * 4);
  }

  const froms = storage(fromBuf, 'vec4', n).setName('bxbConeFroms');
  const dirs = storage(dirBuf, 'vec4', n).setName('bxbConeDirs');
  const out = storage(outBuf, 'float', n).setName('bxbConeOut');
  const slope = uniform(float(coneSlope));
  const cardsFn = createCardsTSL(bindings);

  const kernel = Fn(() => {
    const f = froms.element(instanceIndex);
    const d = dirs.element(instanceIndex);
    out.element(instanceIndex).assign(
      cardsFn ? cardsFn(f.xyz, d.xyz, d.w, slope, NO_SKIP) : float(1)
    );
  })().compute(n);

  await renderer.computeAsync(kernel);
  return new Float32Array(await renderer.getArrayBufferAsync(outBuf));
}
