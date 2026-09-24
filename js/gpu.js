
import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, float, vec3, uniform, instanceIndex, storage, texture3D,
  positionWorld, normalWorld, floor, max, texture, uv, abs,
  attribute, varying, vec2, vec4, ivec2, textureStore, min,
  normalize, cross, clamp, cos, sin, smoothstep, length, log2,
  uniformArray, int, dot, select, dFdx, dFdy, textureSize, mix,
  modelWorldMatrix, materialColor, inverseSqrt, frontFacing, sign, sqrt, fract, exp2,
  cameraPosition, pow, round, reflect, textureLoad, screenCoordinate, Discard
} from 'three/tsl';
import { GRID_DIM, VOXEL_METRES, createBoxGridAt, DISTANCE_RANGE,
         gridIndex, ringSegments, ringSegmentsZ, CASCADE_COUNT,
         HIT_EPS, MIN_STEP, MAX_TRACE_STEPS } from './boxgrid.js';
import { BAYER4, SUN_ANGULAR_SIZE, coneOffsets, coneRadius,
         sunBasis } from './sun.js';
import { MAX_LIGHT_LEVEL, MAX_LIGHTS, LIGHT_VEC4S, DEFAULT_LIGHT_CUTOFF } from './lights.js';
import { CARD_RANGE, CARD_PAD, CARD_FADE_START,
         CARD_SUN_REACH } from './cards.js';
import { BLOCK_METRES } from './world.js';
import { LPV_DIM, LPV_CELL_METRES } from './lpv.js';
import { terrainSampleTSL, terrainWhite, terrainLayerTSL } from './render.js';
import { METAL_F0 } from './materials.js';
import { MIRROR_VEC4S, MIRROR_SUN_BIT } from './mirrors.js';
import { VOXEL_AO_MIN } from './voxelao.js';

// --- WebGPU compute plumbing and the sphere trace on the GPU ---

// The step ceiling, the hit epsilon and the minimum step all come from
// boxgrid.js rather than being restated here. They are the difference between
// the CPU reference and the shader agreeing and not, and bxb.parity() exists to
// catch exactly that - so there must be one definition, not two.

// ---------------------------------------------------------------------------
// 1. Compute smoke test
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
// 2. The distance field as a 3D texture
//
// A Data3DTexture rather than a storage buffer, because the passes that consume
// it want a sampler: the field is read at continuous positions along a ray, not
// at integer cells.
// ---------------------------------------------------------------------------
export function createDistanceTexture(grid) {
  const tex = new THREE.Data3DTexture(grid.data, GRID_DIM, GRID_DIM, GRID_DIM);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  // LINEAR, and this is the one place in the renderer where that is right.
  // Interpolating occupancy was meaningless - it smeared solid into empty. A
  // distance field interpolates exactly: between a solid voxel centre at -0.5
  // and its air neighbour at +0.5, the zero-crossing lands on the shared face,
  // which is where the surface is. It is also what gives a slope's zero-crossing
  // a sub-voxel position. Radiance stays nearest; only occlusion is smooth.
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  // X and Z repeat because the field is a ring in those axes (see gridIndex in
  // boxgrid.js): filtering across the physical seam then reads the logical
  // neighbour. Y is not ringed and clamps as before.
  tex.wrapS = tex.wrapR = THREE.RepeatWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

export function updateDistanceTexture(tex) {
  tex.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// 3. Sphere tracing in TSL
//
// The port of sphereTrace() in boxgrid.js, and the technique the design doc
// specifies: step by the distance stored at the current point, which is the
// radius of a sphere guaranteed to be empty, so the step can never overshoot a
// surface. The binary Amanatides-Woo DDA it replaced is still in boxgrid.js as
// an independent cross-check - two unrelated traversals agreeing on the same
// geometry is worth more than either one being carefully reviewed.
//
// Why this is what slopes need: the DDA could only ever answer "is this VOXEL
// solid", so an occluder's edge was pinned to a voxel boundary. Sampling the
// field linearly puts the zero-crossing wherever the bake put it, including
// partway through a voxel, which is the whole mechanism behind slopes and
// stairs arriving without sub-blocks.
//
// Returns 1.0 when the ray reached a surface before maxDist, 0.0 otherwise.
// ---------------------------------------------------------------------------

// The decode, and it MUST stay the exact mirror of decodeDistance() in
// boxgrid.js - the CPU reference and this shader are only worth having as a pair
// if they read the same bytes the same way, which is what bxb.parity() checks.
//
// An R8 texture samples normalised to [0,1], the field stores c over [-1,+1], and
// the mapping is squared so that precision concentrates at the surface: distance
// = c * |c| * DISTANCE_RANGE, in voxels. See boxgrid.js for the precision table
// and for why a linear map over the same range would have been a bad trade.
//
// The normalisation that silently broke the binary version - where an occupied
// voxel arrived as 1/255 and never cleared a 0.5 threshold - cannot recur, because
// the full byte range is meaningful rather than two values in it.
export const decodeFieldTSL = Fn(([sample, range]) => {
  const c = sample.mul(float(2)).sub(float(1));
  return c.mul(abs(c)).mul(range);
});
// A cascade's bindings as one value. Passing a level around as four loose
// arguments is how the sun basis went wrong earlier - a stale one paired with a
// fresh one and nothing complains.
export function cascadeBindings(tex, grid) {
  return {
    tex,
    origin: uniform(new THREE.Vector3(grid.origin.x, grid.origin.y, grid.origin.z)),
    ring: uniform(new THREE.Vector3(grid.ringX || 0, 0, grid.ringZ || 0)),
    voxel: uniform(float(grid.voxelSize)),
    range: uniform(float(grid.range))
  };
}

export function writeCascadeBindings(b, grid) {
  b.origin.value.set(grid.origin.x, grid.origin.y, grid.origin.z);
  b.ring.value.set(grid.ringX || 0, 0, grid.ringZ || 0);
  b.voxel.value = grid.voxelSize;
  b.range.value = grid.range;
  return b;
}

// Logical voxel position to texture coordinate. Clamped to the outer texel
// centres FIRST: with repeat wrapping, a sample within half a voxel of the
// footprint edge would otherwise filter in the far side of the ring - real data,
// but from 18 m away - which sparkled in penumbrae as terrain crossed the edge.
// This is exactly what clamp-to-edge did before the ring.
// --- Real shader functions, not inlined copies ---
//
// A TSL Fn WITHOUT a layout is pasted inline at every call site. The march
// helpers are called from many places - the sun, every light, reflection
// hits, the mirrors, sprites - so the terrain shader held about ten copies of
// the cascade walk, each three step loops, and compiling it froze the game for
// tens of seconds. With setLayout each is ONE WGSL function, emitted once per
// shader and called. Names must be unique per shader, hence the counter; and
// the per-cascade ones are memoised, so every caller shares one function.
//
// PER SHADER, NOT PER BACKEND. three caches a layout function's generated
// WGSL per backend, keyed by the function node, and hands that same text to
// every shader that calls it. That is right for its own layout functions,
// which are pure maths; ours read uniforms and textures (the cascades, the
// cards, the light list), and the text names them as the FIRST shader to build
// it did - so every other shader got "no definition for nodeUniform2". The
// body is built by whichever builder calls it (flowShaderNode), registering
// its uniforms there correctly, so the fix is only the cache: functions whose
// layout carries perShader are cached per builder instead.
const perBuilderFns = new WeakMap();
// Per builder too: how many functions of each base name it has named. A name is
// base_n, n counting that base's functions in THIS shader in the order they are
// first used - so the same graph gives the same WGSL every run. (A global
// serial was unique too, but it counted every function made before, in any
// build, so the text - and the browser's pipeline cache key - varied between
// runs with what had been built earlier.)
const perBuilderNames = new WeakMap();
const baseBuildFunctionNode = THREE.NodeBuilder.prototype.buildFunctionNode;
THREE.NodeBuilder.prototype.buildFunctionNode = function (shaderNode) {
  if (!(shaderNode.layout && shaderNode.layout.perShader)) {
    return baseBuildFunctionNode.call(this, shaderNode);
  }
  let fns = perBuilderFns.get(this);
  if (!fns) perBuilderFns.set(this, fns = new Map());
  let fn = fns.get(shaderNode);
  if (!fn) {
    fn = new THREE.FunctionNode();
    let counts = perBuilderNames.get(this);
    if (!counts) perBuilderNames.set(this, counts = new Map());
    const layout = shaderNode.layout;
    const n = counts.get(layout.baseName) || 0;
    counts.set(layout.baseName, n + 1);
    // Only read while the code is generated (the call sites take the name from
    // the code), so it is set for that and put back.
    const resting = layout.name;
    layout.name = `${layout.baseName}_${n}`;
    const previous = this.currentFunctionNode;
    this.currentFunctionNode = fn;
    try {
      fn.code = this.buildFunctionCode(shaderNode);
    } finally {
      this.currentFunctionNode = previous;
      layout.name = resting;
    }
    fns.set(shaderNode, fn);
  }
  return fn;
};

let fnSerial = 0;
const fnName = base => `${base}_${fnSerial++}`;
// Every layout here goes through this: a unique name, and per-shader caching.
const layout = (base, type, inputs) => ({ name: fnName(base), baseName: base, type, inputs,
                                         perShader: true });
const perCascades = new WeakMap();
function memoCascades(cascades, key, make) {
  if (!perCascades.has(cascades)) perCascades.set(cascades, {});
  const m = perCascades.get(cascades);
  return m[key] || (m[key] = make());
}

export const ringUV = Fn(([vp, ring]) => {
  const dim = float(GRID_DIM);
  return clamp(vp, vec3(0.5, 0.5, 0.5), vec3(dim.sub(0.5))).add(ring).div(dim);
});

// ring: the grid's x/z ring offset in voxels - see gridIndex in boxgrid.js.
export const traceDistanceTSL = Fn(([sdfTex, gridOrigin, origin, dir, maxDist, ring]) => {
  const dim = float(GRID_DIM);
  const vs = float(VOXEL_METRES);
  const t = float(0).toVar();
  const hit = float(0).toVar();

  Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
    // Continuous voxel-space position of the current point. No cell index and
    // no per-axis boundary bookkeeping - that was all the DDA needed.
    const vp = origin.add(dir.mul(t)).sub(gridOrigin).div(vs);

    // Leaving the grid means open sky. For the sun - the one light with no
    // termination distance - this is also where a long shadow truncates, and it
    // is a RANGE limit of the field's footprint, not of the traversal.
    If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
       .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
       .or(vp.z.greaterThanEqual(dim)), () => {
      Break();
    });

    const d = decodeFieldTSL(texture3D(sdfTex, ringUV(vp, ring)).r, float(DISTANCE_RANGE));
    If(d.lessThan(float(HIT_EPS)), () => {
      hit.assign(float(1));
      Break();
    });

    // MIN_STEP is the floor that guarantees progress. A ray running parallel to a
    // wall sits at a small constant distance and would otherwise inch forward
    // until the step cap. Running out of steps reports no-hit, which resolves a
    // grazing ray as lit - the conservative answer, and the one nobody can see.
    t.addAssign(max(d, float(MIN_STEP)).mul(vs));
    If(t.greaterThan(maxDist), () => { Break(); });
  });

  return hit;
});

// ---------------------------------------------------------------------------
// 3b. The cone trace: a soft shadow from ONE ray
//
// The same march, but instead of returning hit/miss it returns how close the ray
// came to a surface, relative to how wide the sun's cone is at that distance. The
// field already stores the distance to the nearest surface, so the width of the
// penumbra is information the traversal is throwing away - this keeps it.
//
//   visibility = min over t of clamp( d(t) / (R * t) )
//
// R = tan(half-angle), so R*t is the cone's radius in metres at distance t along
// the ray. A surface the cone fully clears contributes 1; one it fully contains
// contributes 0; one grazing its edge contributes the fraction. Because R*t grows
// with t, an occluder far from the receiver softens and one in contact stays hard,
// which is distance-based softening arriving from the geometry with no blocker
// search, no filter kernel and no bias.
//
// WHY THIS NEEDED THE WIDENED FIELD. At DISTANCE_RANGE = 1 the stored d saturated
// at 12.5 cm, so open air 3 m from anything computed 0.125/(0.06*3) = 0.69 and the
// whole world dimmed by 31%. At 8 voxels the expression only saturates beyond
// 1/0.06 = 16.7 m, which is past the 12 m march cap - so open sky reads as exactly
// lit everywhere a ray can reach. That threshold is the entire reason the range is
// 8 and not 4.
//
// Sixteen times cheaper than sampling the disc with 16 rays, and smooth rather
// than quantised by the sample count. What it gives up is occluder SHAPE: it sees
// the nearest surface, not how much of the disc that surface covers, so two thin
// occluders read as one and a shadow through a narrow gap is slightly too dark.
// The sampled path stays available to diff against for exactly that reason.
// ---------------------------------------------------------------------------
// --- Why a shadow needs to FADE at the limits, not stop ---
//
// A march ends for one of three reasons, and only one of them is geometry:
//
//   1. it hit something                  - a real shadow
//   2. it ran past maxDist               - the 12 m cap
//   3. it left the grid                  - the 18 m footprint
//
// Cases 2 and 3 are answers about the FIELD, not about the world, and both used
// to resolve to "fully lit". That put a step discontinuity in the middle of open
// ground: the texel whose ray reaches a wall at 11.9 m came out fully black, and
// the texel beside it whose ray would have reached it at 12.1 m came out lit.
// Neighbouring texels, opposite answers, so the shadow ends on a straight line
// across the ground with no penumbra at all - which is exactly the hard edge at
// the far end of a long shadow.
//
// No bias or sample count fixes that, because it is not a sampling error. The
// ray genuinely does not know what is past its limit. So the honest thing is to
// stop pretending to know: weight what a sample is allowed to darken by how close
// it is to the limit, so occlusion decays to nothing as the march runs out of
// road. A shadow then dissolves over the last stretch instead of being cut off,
// and the boundary stops being visible at all.
//
// Two fades, multiplied:
//
//   distance - smoothstep down to zero over the last of the march. Smoothstep
//              rather than linear because its derivative vanishes at both ends,
//              so neither the start of the fade nor the cap itself shows as a
//              crease.
//   edge     - linear in the sample's distance to the nearest grid face. This one
//              is spatial rather than temporal: a ray can leave the grid at any t,
//              including immediately, so fading by t alone would not catch it.
//
// This is a range limitation being made invisible, NOT repaired. The shadow is
// still absent past the footprint; it now ends by fading out, the way a shadow
// too faint to see would.
export const fadeWeightTSL = Fn(([vp, dim, t, maxDist, fadeStart, edgeFade]) => {
  // Distance to the nearest of the six grid faces, in voxels.
  const lo = min(min(vp.x, vp.y), vp.z);
  const hi = min(min(dim.sub(vp.x), dim.sub(vp.y)), dim.sub(vp.z));
  // max() on the divisor rather than a branch: a width of 0 means "no edge fade",
  // and dividing by it gives 0/0 = NaN exactly at the grid corner, which shows up
  // as an unexplainable speck rather than as a division error.
  const edge = clamp(min(lo, hi).div(max(edgeFade, float(1e-6))), float(0), float(1));
  // fadeStart is a FRACTION of maxDist, so the fade tracks the cap rather than
  // needing to be retuned whenever the cap moves.
  const far = float(1).sub(smoothstep(fadeStart.mul(maxDist), maxDist, t));
  return edge.mul(far);
});

// Both shadow traces walk the cascades in order, finest first, carrying t across
// the seam. A ray leaves a level exactly once - the footprints are nested boxes
// around the receiver, so once it is outside C0 it cannot re-enter - which is why
// this is a sequence of loops rather than a per-step branch on which level to
// read. One texture fetch per step, not two.
//
// THE EDGE FADE ONLY APPLIES TO THE OUTERMOST LEVEL. Leaving C0 used to mean
// "nothing is known past here"; now it means "continue in C1", so fading there
// would dim a shadow at an invisible boundary in the middle of the scene. Inner
// levels are passed a zero fade width, which the guard in fadeWeightTSL reads as
// off.
function eachCascade(cascades, edgeFade) {
  return cascades.map((c, i) => ({
    c,
    // Only the last level's boundary is the end of knowledge.
    edge: i === cascades.length - 1 ? edgeFade : float(0)
  }));
}

// --- Smoothing the C0 -> C1 seam ---
//
// A receiver inside C0 marches its first metres at C0's resolution; one just
// outside starts in C1 at half of it. Where the footprint ends, neighbouring
// texels get differently-resolved answers and the boundary shows as a line on
// the ground - one that steps along with the player as the grid re-origins.
//
// So over the last SEAM_BAND_VOXELS of C0, march twice - once starting in C0,
// once starting in C1 as if C0 were not there - and crossfade by how deep the
// receiver sits. Everywhere else it is one pass, as before; only the band pays
// for the second.
export const SEAM_BAND_VOXELS = 16;

function c0WeightTSL(origin, c0) {
  const dim = float(GRID_DIM);
  const vp = origin.sub(c0.origin).div(c0.voxel);
  const lo = min(min(vp.x, vp.y), vp.z);
  const hi = min(min(dim.sub(vp.x), dim.sub(vp.y)), dim.sub(vp.z));
  return smoothstep(float(0), float(SEAM_BAND_VOXELS), min(lo, hi));
}

// march(skipC0) builds one pass and returns its vec2; skipC0 is a bool node, or
// null when there is only one level and nothing to blend.
function seamBlend(cascades, origin, march) {
  if (cascades.length < 2) return march(null);
  const w0 = c0WeightTSL(origin, cascades[0]).toVar();
  const both = w0.greaterThan(float(0)).and(w0.lessThan(float(1))).toVar();
  const passes = select(both, int(2), int(1)).toVar();
  const out = vec2(0, 0).toVar();
  Loop({ start: int(0), end: passes, type: 'int', condition: '<', name: 'seamPass' },
       ({ seamPass }) => {
    const skip = seamPass.equal(int(1)).toVar();
    const r = march(skip);
    // Pass 0 alone takes everything; paired, it takes w0 and pass 1 the rest.
    const wt = select(skip, float(1).sub(w0), select(both, w0, float(1)));
    out.addAssign(r.mul(wt));
  });
  return out;
}

// The C0 loop runs unless this pass is the one pretending C0 is absent.
const levelLive = (i, skipC0, done) =>
  i === 0 && skipC0 ? done.equal(float(0)).and(skipC0.not()) : done.equal(float(0));

export function createConeTraceSunTSL(cascades) {
  return memoCascades(cascades, 'cone', () => Fn(([origin0, dir0, maxDist, coneR, fadeStart, edgeFade]) => {
    // Pinned: both are reused across passes and levels - see the cone path in
    // createSoftSunTSL for what an unpinned shared node does here.
    const origin = origin0.toVar(), dir = dir0.toVar();
    return seamBlend(cascades, origin, skipC0 => {
    const dim = float(GRID_DIM);
    // Started one voxel out rather than at zero: at t = 0 the cone has no radius,
    // so d/(R*t) divides by zero, and the surface the ray is leaving is the
    // nearest thing to it by construction.
    const t = float(VOXEL_METRES).toVar();
    const res = float(1).toVar();
    const done = float(0).toVar();

    for (const [i, { c, edge }] of eachCascade(cascades, edgeFade).entries()) {
      If(levelLive(i, skipC0, done), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel);
          // Leaving this level is not a result: t carries into the next one, and
          // only the outermost level's exit means open sky.
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => {
            Break();
          });

          // How much this sample may darken anything: 1 well inside the march's
          // limits, falling to 0 as it approaches them, so a shadow dissolves
          // rather than stopping on a line. See fadeWeightTSL.
          const w = fadeWeightTSL(vp, dim, t, maxDist, fadeStart, edge);

          const d = decodeFieldTSL(texture3D(c.tex, ringUV(vp, c.ring)).r, c.range);
          // A real hit occludes fully, but only in proportion to w - a hit found
          // at the very end of the march is one the next texel along will miss.
          // Nothing later can lower the minimum, so the whole trace ends here.
          If(d.lessThan(float(HIT_EPS)), () => {
            res.assign(min(res, float(1).sub(w)));
            done.assign(float(1));
            Break();
          });

          // d is in voxels of THIS level, t and the cone radius in metres. The
          // occlusion implied is 1 - ratio; weighting that rather than the
          // visibility is what makes w = 0 mean "contributes nothing" rather
          // than "is fully lit".
          const ratio = d.mul(c.voxel).div(coneR.mul(t));
          const occ = float(1).sub(clamp(ratio, float(0), float(1))).mul(w);
          res.assign(min(res, float(1).sub(occ)));

          t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }

    // x is the visibility; y is whether this march CONCLUDED - it hit something,
    // or it spent its whole distance budget - as opposed to running off the edge
    // of the outermost cascade, where it learned nothing at all. The caller uses
    // that to decide whether the answer is worth storing. See the atlas kernel.
    return vec2(clamp(res, float(0), float(1)), done);
    });
  }).setLayout(layout('coneTrace', 'vec2', [
    { name: 'origin0', type: 'vec3' }, { name: 'dir0', type: 'vec3' },
    { name: 'maxDist', type: 'float' }, { name: 'coneR', type: 'float' },
    { name: 'fadeStart', type: 'float' }, { name: 'edgeFade', type: 'float' }])));
}

// The sampled path's trace. Identical in structure, but returns how much this ray
// should darken - faded at the march's limits - rather than a binary hit.
//
// Separate from traceDistanceTSL because that one is what bxb.parity() diffs
// against the CPU sphere trace, and a hit test that sometimes returns 0.6 is not
// a hit test any more.
export function createShadowTraceTSL(cascades) {
  return memoCascades(cascades, 'shadow', () => Fn(([origin0, dir0, maxDist, fadeStart, edgeFade]) => {
    const origin = origin0.toVar(), dir = dir0.toVar();
    return seamBlend(cascades, origin, skipC0 => {
    const dim = float(GRID_DIM);
    const t = float(0).toVar();
    const occ = float(0).toVar();
    const done = float(0).toVar();

    for (const [i, { c, edge }] of eachCascade(cascades, edgeFade).entries()) {
      If(levelLive(i, skipC0, done), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel);
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => {
            Break();
          });

          const d = decodeFieldTSL(texture3D(c.tex, ringUV(vp, c.ring)).r, c.range);
          If(d.lessThan(float(HIT_EPS)), () => {
            occ.assign(fadeWeightTSL(vp, dim, t, maxDist, fadeStart, edge));
            done.assign(float(1));
            Break();
          });

          t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }

    return vec2(occ, done);
    });
  }).setLayout(layout('shadowTrace', 'vec2', [
    { name: 'origin0', type: 'vec3' }, { name: 'dir0', type: 'vec3' },
    { name: 'maxDist', type: 'float' }, { name: 'fadeStart', type: 'float' },
    { name: 'edgeFade', type: 'float' }])));
}

// ---------------------------------------------------------------------------
// 4. CPU/GPU parity
//
// Runs the same rays through the GPU kernel and returns the results, so they
// can be diffed against marchOccupancy() from boxgrid.js. The whole reason the
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
// 5. Shadow shading node
//
// Phase B shades per screen pixel, not into an atlas - that is Phase C. One
// light, one shadow ray, marched through occupancy. Enough to prove traversal
// works against real geometry before the atlas is built on top of it.
// ---------------------------------------------------------------------------
// The sun is DIRECTIONAL: every surface marches along one shared direction for
// a capped distance, not toward a point. Treating it as a point light at
// (10,20,10) meant each pixel marched ~160 voxels to reach it, which is what
// killed the framerate, and put the light nearly overhead so every shadow hid
// underneath the block casting it.
//
// Capping the march is the perf lever. At 12.5cm voxels a 12m cap is ~96 steps
// worst case, and anything further away contributes a shadow too faint to
// matter at this art scale.
// Runs the SAME cone as the atlas, per fragment instead of per texel, and that
// is its whole remaining value: two passes over the same geometry in different
// spaces, sharing one marcher, so a disagreement between them is a bug in the
// addressing rather than in the shadow. It is the more expensive of the two by
// the ratio of screen pixels to world texels, and it pays that cost every frame
// where the atlas pays it per move, so the atlas is the shipping path.
// --- THE TEXEL LOCK ---
//
// The whole idea, and the only thing the atlas was ever really buying: a shadow
// is a property of the SURFACE, not of the screen. Shade a fragment at its own
// position and the shadow is resolved per pixel, so its edge slides smoothly
// under the camera and reads as a soft 3D shadow laid over flat-shaded pixel
// art - two different resolutions in one image, which is what made it look
// wrong however good the penumbra was.
//
// So snap the shading position to the centre of the world texel it falls in
// before marching. Every fragment inside one 12.5 cm texel then marches the
// IDENTICAL ray and gets the IDENTICAL answer, so the shadow comes out in whole
// texels, locked to the world grid, at exactly the density of the sprite art.
// Zoom, pan, rotate: the shadow stays painted on the ground. Walk, and its edge
// advances a texel at a time rather than sliding.
//
// This is what the atlas did, minus the atlas. No pages, no job list, no
// footprint to fall outside of, no cached answer to go stale, no second
// addressing scheme to disagree with the first - and the same result, because
// the quantisation was always the point and the storage never was. It costs a
// march per fragment instead of per texel, which is the trade the atlas existed
// to avoid; at this world's screen coverage that is the cheaper problem to have.
//
// SNAPPED ACROSS THE FACE ONLY. The component along the normal is left exactly
// as it is: quantising that would move the origin off the surface - inside it on
// one side of the texel and into the air on the other - and the bias that lifts
// the ray off the face is measured from the real surface, not from a rounded
// one. abs(n) is 1 on the face's own axis and 0 on the other two, and the terrain
// is axis-aligned boxes, so this is a select written as a lerp.

// ---------------------------------------------------------------------------
// 3c. Analytic sprite cards - the GPU port of cards.js
//
// Sprites are NOT in the distance field. The field is geometry shared by every
// light, so a card baked into it points one way and every other light gets the
// wrong silhouette; the fix is to keep sprites out of it and intersect each
// shadow ray with a card turned to face THAT light.
//
// This is the mirror of cardVisibility() in cards.js and has to stay one, the
// way traceDistanceTSL mirrors sphereTrace - bxb.cardparity() diffs them.
//
// Cheap for a reason worth being explicit about: a card is a PLANE, so this is
// one intersection per card per RAY. Marching a voxelised sprite costs a test
// per card per STEP, tens of times more, and buys nothing the plane does not
// already give exactly. Being analytic it also has no voxel size, so the
// cascade-scale seam a baked silhouette suffers cannot arise at all.
//
// Layout, four vec4 per card, packed by packCardInstances():
//   0  centre xyz, flip (-1 or +1)
//   1  facing ux, uz, half width, half height   (metres)
//   2  atlas rect x, y, w, h                    (pixels)
//   3  metres per pixel, cols, rows, cascade lod (see the LOD block below)
// ---------------------------------------------------------------------------

export const MAX_CARDS = 64;

// The sun's cards reach far past the march cap - see CARD_SUN_REACH.
export const CARD_SUN_MAX = CARD_SUN_REACH;

// name: the buffer's name in WGSL, unique per shader - see bufferName.
export function createCardBindings(capacity = MAX_CARDS, name = 'bxbCards') {
  return {
    capacity,
    // Four vec4 per card, flat. A uniform array rather than a storage buffer:
    // 64 cards is 4 KB, far inside the uniform limit, and a uniform read is the
    // cheaper of the two in a per-fragment loop.
    data: uniformArray(new Array(capacity * 4).fill(0).map(() => new THREE.Vector4()),
                       'vec4').setName(name),
    count: uniform(int(0)),
    // How many levels finer than its cascade this light draws a card - see the
    // LOD block in createCardsTSL. The sun uses 1, point lights 0.
    lodShift: uniform(float(0)),
    lodMax: uniform(float(CASCADE_COUNT - 1)),   // coarsest quality level
    atlas: null,
    atlasSize: uniform(new THREE.Vector2(1, 1))
  };
}

export function createCardAtlasTexture(atlas) {
  const tex = new THREE.DataTexture(atlas.data, atlas.width, atlas.height,
                                    THREE.RedFormat, THREE.UnsignedByteType);
  // LINEAR for the same reason the 3D field is: interpolating a DISTANCE is
  // exact, where interpolating occupancy would smear solid into empty. It is
  // also what lets one texel of silhouette resolve a smooth penumbra edge.
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

// Push a packed Float32Array into the uniform array.
export function writeCardBindings(b, packed, count) {
  const v = b.data.array;
  for (let i = 0; i < count * 4; i++) {
    v[i].set(packed[i * 4], packed[i * 4 + 1], packed[i * 4 + 2], packed[i * 4 + 3]);
  }
  b.count.value = count;
  return b;
}

// How much of one light the cards leave visible along a ray.
//
// Multiplied rather than min'd, matching cardsVisibility(): two sprites
// overlapping a ray each take their own bite, where min would let the nearer
// one hide the further.
//
// ranged: the Fn takes two more arguments, the first card and one past the last,
// for a buffer shared between lights where each owns a slice - see the point
// light loop. Unranged it walks the whole buffer, as the sun's does.
export function createCardsTSL(b, { ranged = false } = {}) {
  // No atlas means no cards in this pass at all, and the caller drops the
  // multiply rather than compiling a loop that can never run.
  if (!b || !b.atlas) return null;
  // skip is the XZ of the card to leave out: a sprite shading itself must not
  // be occluded by its own card. Terrain passes a point far off the map.
  const body = (from, dir, maxDist, coneSlope, skip, start, end) => {
    const vis = float(1).toVar();

    Loop({ start, end, type: 'int', condition: '<' }, ({ i }) => {
      const base = i.mul(int(4));
      const A = b.data.element(base);
      const B = b.data.element(base.add(int(1)));
      const C = b.data.element(base.add(int(2)));
      const D = b.data.element(base.add(int(3)));

      const centre = A.xyz;
      const flip = A.w;
      const ux = B.x, uz = B.y;
      // The card's normal is its width turned 90 degrees, same as the CPU side.
      const nx = uz.negate(), nz = ux;

      const denom = dir.x.mul(nx).add(dir.z.mul(nz));
      // Edge-on. A card built to face this light cannot be, but a stale facing
      // can, and a near-zero divisor throws the hit point to infinity.
      If(abs(denom).greaterThan(float(1e-6))
           .and(length(centre.xz.sub(skip)).greaterThan(float(0.05))), () => {
        const t = centre.x.sub(from.x).mul(nx)
                  .add(centre.z.sub(from.z).mul(nz)).div(denom);
        // Between the receiver and the light, or it is not an occluder. The
        // maxDist cap is what stops a sprite BEYOND a torch from shadowing it.
        If(t.greaterThan(float(0)).and(t.lessThan(maxDist)), () => {
          const h = from.add(dir.mul(t)).sub(centre);
          const a = h.x.mul(ux).add(h.z.mul(uz)).mul(flip);
          const bH = h.y;

          const mpp = D.x, cols = D.y, rows = D.z, lod = D.w;
          const rx = C.x, ry = C.y, rw = C.z, rh = C.w;
          const hw = B.z, hh = B.w;

          // EARLY OUT, exact. The card's plane is infinite, so nearly every
          // ray crosses it somewhere - usually metres off the card, where the
          // answer is "fully lit" - and every such crossing used to read the
          // silhouette texture anyway. That was the single largest cost in the
          // frame (1.7 ms of 3.7 with the torch on, measured). The distance
          // field is never less than the distance to the card's rectangle, less
          // the largest cascade snap below; past the cone's radius plus that,
          // the visibility is exactly 1. Skip it.
          const outside = length(vec2(max(abs(a).sub(hw), float(0)),
                                      max(abs(bH).sub(hh), float(0))));
          const maxSnap = float(VOXEL_METRES).mul(exp2(b.lodMax));
          If(outside.lessThanEqual(coneSlope.mul(t).add(maxSnap)), () => {

            // Signed distance to the silhouette at card-local (sa, sb), metres.
            const distAt = (sa, sb) => {
              // Card-local metres to atlas pixels, then to UV. CARD_PAD is the
              // margin the falloff grows into; past it the border value already
              // reads as fully clear.
              const pu = sa.div(mpp).add(cols.mul(float(0.5))).add(float(CARD_PAD));
              const pv = rows.mul(float(0.5)).sub(sb.div(mpp)).add(float(CARD_PAD));
              // Inset by half a texel on every side. Bilinear filtering samples the
              // four texels AROUND the coordinate, so clamping to the rect edge
              // still reaches one texel outside it - into the neighbouring card, or
              // into the unwritten rows below a short one. That bleed is what put
              // spurious solid bands across the ground.
              const su = clamp(pu, float(0.5), rw.sub(float(0.5)));
              const sv = clamp(pv, float(0.5), rh.sub(float(0.5)));
              const uvp = vec2(rx.add(su).div(b.atlasSize.x),
                               ry.add(sv).div(b.atlasSize.y));
              // Same decode as the boxGrid's, because it is the same encoding - one
              // distance format in the renderer, not two that can drift.
              const dPix = decodeFieldTSL(texture(b.atlas, uvp).r, float(CARD_RANGE));
              // The transform saturates at CARD_PAD pixels, so far from the card it
              // under-reports the distance and a wide cone reads shadow everywhere.
              // The silhouette sits inside its own rectangle, so that rectangle's
              // distance is a valid floor - exact far away, negative and ignored
              // inside. See cardVisibility for the same step and why it is needed.
              // Outside only: the rectangle distance is zero inside, and maxing
              // against zero there would erase the negative distances that are the
              // silhouette itself.
              const dRect = length(vec2(max(abs(sa).sub(hw), float(0)),
                                        max(abs(sb).sub(hh), float(0))));
              const dtM = dPix.mul(mpp);
              return select(dRect.greaterThan(float(0)), max(dtM, dRect), dtM);
            };

            // CASCADE LOD - the card's resolution follows the level it stands in,
            // as the terrain's does. lod is set per card on the CPU: its integer
            // part is the level, its fraction how far into the band toward the
            // next one. Past C0 the hit point is snapped to that level's voxel
            // grid, so a distant sprite's shadow is as coarse as the ground's
            // around it; across the band the two resolutions crossfade, and past
            // the outermost level the card fades out as terrain shadows do.
            const l0 = floor(lod), f = lod.sub(l0);
            // Quality level: the cascade minus this light's shift, capped at its
            // lodMax - see SUN_CARD_LOD_SHIFT in cards.js.
            const quantum = l => {
              const ql = clamp(l.sub(b.lodShift), float(0), b.lodMax);
              return select(ql.lessThan(float(0.5)), float(0), float(VOXEL_METRES).mul(exp2(ql)));
            };
            const snap = (v, half, q) => select(q.greaterThan(float(0)),
              v.add(half).div(q).floor().add(float(0.5)).mul(q).sub(half), v);
            const q0 = quantum(l0), q1 = quantum(l0.add(float(1)));
            const d = distAt(snap(a, hw, q0), snap(bH, hh, q0)).toVar();
            const lastLevel = float(CASCADE_COUNT - 1);
            // Only where the two resolutions actually differ - under the sun
            // the C0/C1 band is full resolution on both sides.
            If(f.greaterThan(float(0)).and(l0.lessThan(lastLevel)).and(q1.notEqual(q0)), () => {
              d.assign(mix(d, distAt(snap(a, hw, q1), snap(bH, hh, q1)), f));
            });
            const lodFade = select(l0.greaterThanEqual(lastLevel), float(1).sub(f), float(1));

            const r = coneSlope.mul(t);
            // Hard when the cone has no width, otherwise d / (R*t) - the very
            // expression createConeTraceSunTSL uses, so a card's penumbra and a
            // wall's come from one formula rather than two that look alike.
            const soft = clamp(d.div(max(r, float(1e-6))), float(0), float(1));
            const hard = select(d.lessThan(float(0)), float(0), float(1));
            const raw = select(r.greaterThan(float(0)), soft, hard);

            // FADE, do not cut - the mirror of cardVisibility's tail, and the same
            // reasoning as fadeWeightTSL. A card faces the light, so t is very
            // nearly the distance from receiver to caster: the maxDist cap lands as
            // a straight line across the ground and a tree's shadow simply stops
            // along it, which reads as a diagonal slice cut out of the shadow.
            const fade = float(1).sub(
              smoothstep(maxDist.mul(float(CARD_FADE_START)), maxDist, t));
            vis.mulAssign(float(1).sub(float(1).sub(raw).mul(fade).mul(lodFade)));
          });
        });
      });
    });

    return vis;
  };
  const inputs = [
    { name: 'rayFrom', type: 'vec3' }, { name: 'dir', type: 'vec3' },
    { name: 'maxDist', type: 'float' }, { name: 'coneSlope', type: 'float' },
    { name: 'skip', type: 'vec2' }];
  return ranged
    ? Fn(([rayFrom, dir, maxDist, coneSlope, skip, start, end]) =>
        body(rayFrom, dir, maxDist, coneSlope, skip, start, end))
      .setLayout(layout('cardsRanged', 'float', [...inputs, { name: 'start', type: 'int' }, { name: 'end', type: 'int' }]))
    : Fn(([rayFrom, dir, maxDist, coneSlope, skip]) =>
        body(rayFrom, dir, maxDist, coneSlope, skip, int(0), b.count))
      .setLayout(layout('cards', 'float', inputs));
}

// The face normal, exactly on its axis. normalWorld is interpolated and
// renormalised, so it is off-axis by a rounding error that varies by pixel;
// the terrain is boxes, so its dominant axis IS the face.
export const axisNormalTSL = Fn(([v]) => {
  const a = abs(v).toVar();
  const nx = a.x.greaterThanEqual(a.y).and(a.x.greaterThanEqual(a.z));
  const ny = a.y.greaterThan(a.x).and(a.y.greaterThanEqual(a.z));
  return select(nx, vec3(sign(v.x), 0, 0),
                select(ny, vec3(0, sign(v.y), 0), vec3(0, 0, sign(v.z))));
});

// ONE TEXEL, ONE ANSWER - bit for bit. n must be axis-exact (axisNormalTSL):
// across the face each coordinate snaps to its texel centre, and ALONG the
// normal it snaps to the face plane itself, which is a voxel boundary (every
// face of the terrain's boxes is). It used to keep the interpolated value
// there, and that drifts by a rounding error from pixel to pixel - harmless to
// a soft shadow, but a reflection ray started a hair differently marches a
// long way and can land on a different voxel, so one texel reflected bands.
// Every input is now computed from the same integers, so every fragment of a
// texel gets the identical float.
export const texelLockTSL = Fn(([p, n]) => {
  const q = float(VOXEL_METRES);
  const a = abs(n);
  const snapped = p.div(q).floor().add(float(0.5)).mul(q);
  const plane = round(p.div(q)).mul(q);
  return snapped.mul(vec3(1, 1, 1).sub(a)).add(plane.mul(a));
});

// --- The sky, as L2 SH (sky.js) ---
//
// Nine RGB coefficients, projected on the CPU each frame the sky changes, in
// one uniform array. Everything that wants the sky evaluates this: the ambient
// at the bent normal (irradiance), the background and reflection misses at a
// direction (radiance). gain scales irradiance so the noon sky lights an
// up-facing surface at exactly the ambient uniform - sky.js skyAmbientGain.
// The sun's light colour and strength (sky.js sunLight), linear. Module-level,
// so the shading pass and the LPV injection read the one value.
export const sunColourUniform = uniform(new THREE.Color(1, 1, 1));

export function createSkyBindings() {
  return {
    sh: uniformArray(new Array(9).fill(0).map(() => new THREE.Vector4()), 'vec4').setName('bxbSkySH'),
    gain: uniform(float(1))
  };
}
export function writeSkyBindings(b, sh, gain) {
  for (let i = 0; i < 9; i++) b.sh.array[i].set(sh[i][0], sh[i][1], sh[i][2], 0);
  b.gain.value = gain;
  return b;
}
// The mirror of evalSH in sky.js: same basis, same order, same band factors.
export function skyShTSL(b, d, irradiance = false) {
  const x = d.x, y = d.y, z = d.z;
  const Y = [
    float(0.282095),
    y.mul(0.488603), z.mul(0.488603), x.mul(0.488603),
    x.mul(y).mul(1.092548), y.mul(z).mul(1.092548), z.mul(z).mul(3).sub(1).mul(0.315392),
    x.mul(z).mul(1.092548), x.mul(x).sub(y.mul(y)).mul(0.546274)
  ];
  const band = [1, 2 / 3, 2 / 3, 2 / 3, 1 / 4, 1 / 4, 1 / 4, 1 / 4, 1 / 4];
  let out = null;
  for (let i = 0; i < 9; i++) {
    const k = irradiance ? band[i] : 1;
    const term = b.sh.element(int(i)).xyz.mul(Y[i].mul(float(k)));
    out = out ? out.add(term) : term;
  }
  return max(out, vec3(0, 0, 0));
}

// --- AO: cone-traced ambient occlusion (doc §6.2, the short band) ---
//
// Measures what RTAO measures - how much of the hemisphere is blocked within
// AO_DISTANCE - but with six wide cones instead of a few thin random rays. Each
// cone is marched through the same field as the soft sun, and at every step the
// field's distance against the cone's width, d / (R*t), says how much of the
// cone is blocked there - so one cone integrates a whole solid angle rather than
// sampling a line. Blocking weighs less the further out it is (1 - t / reach):
// contact darkens hard, a wall at the edge of reach barely at all.
//
// Deterministic: the same six directions everywhere, no per-texel randomness,
// so there is no noise to hide and the answer is identical frame to frame.
//
// The cones are the usual hemisphere tiling: one along the normal, five tilted
// 60 degrees from it, each 60 degrees wide. Weights are the cosine-weighted
// share of the hemisphere each covers, so open sky sums to exactly 1.
export const AO_CONES = 6;
export const AO_DISTANCE = 1.0;   // metres
const AO_CONE_R = Math.tan(Math.PI / 6);   // 60 degree aperture
const AO_TILT = Math.PI / 3;
const AO_WEIGHT_UP = 0.25, AO_WEIGHT_SIDE = 0.15;

function createAOConeTSL(cascades) {
  return memoCascades(cascades, 'ao', () => Fn(([origin, dir, maxDist, coneR]) => {
    const dim = float(GRID_DIM);
    // A voxel out, as the sun's cone: at t = 0 the cone has no width.
    const t = float(VOXEL_METRES).toVar();
    const occ = float(0).toVar();
    const done = float(0).toVar();
    const hit = float(0).toVar();
    for (const { c } of eachCascade(cascades, float(0))) {
      If(done.equal(float(0)), () => {
        Loop({ start: 0, end: 32, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel);
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => { Break(); });
          const d = decodeFieldTSL(texture3D(c.tex, ringUV(vp, c.ring)).r, c.range);
          const blocked = float(1).sub(clamp(d.mul(c.voxel).div(coneR.mul(t)), float(0), float(1)));
          occ.assign(max(occ, blocked.mul(float(1).sub(t.div(maxDist)))));
          If(d.lessThan(float(HIT_EPS)), () => {
            done.assign(float(1)); hit.assign(float(1)); Break();
          });
          t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }
    // x the occlusion; y how far the axis got, z whether it ended on a surface -
    // which is where the GI gather samples the LPV.
    return vec3(occ, min(t, maxDist), hit);
  }).setLayout(layout('aoCone', 'vec3', [
    { name: 'origin', type: 'vec3' }, { name: 'dir', type: 'vec3' },
    { name: 'maxDist', type: 'float' }, { name: 'coneR', type: 'float' }])));
}

// The GI volumes, sampled trilinearly at a world position. gi.levels holds one
// gi.js binding per volume, finest first: C0's, then C1's.
//
// Each level fades to nothing over its outermost cells, and the finer one is
// laid over the coarser: inside C0 it is C0's answer, crossing its edge it
// blends into C1's, and past C1's edge the bounce is gone. Sampling one volume
// outside itself is what went wrong before - the texture clamps, so everything
// beyond C0 read whatever its boundary cells held.
const LPV_EDGE_CELLS = 2;
export function lpvSampleTSL(gi, p) {
  let out = null;
  for (let i = gi.levels.length - 1; i >= 0; i--) {
    const lv = gi.levels[i];
    const uvw = p.sub(lv.origin).div(float(lv.extent)).toVar();
    // Distance to the nearest face of the volume, in cells.
    const edge = min(min(min(uvw.x, uvw.y), uvw.z),
                     min(min(float(1).sub(uvw.x), float(1).sub(uvw.y)), float(1).sub(uvw.z)))
                   .mul(float(LPV_DIM));
    const w = smoothstep(float(0.5), float(0.5 + LPV_EDGE_CELLS), edge);
    const s = texture3D(lv.tex, uvw).rgb;
    out = out ? mix(out, s, w) : s.mul(w);
  }
  return out;
}

// Returns Fn([p, n, lift, skip]) -> visibility in [0, 1]: n is the hemisphere,
// lift the direction the origin is raised along (the geometry's, as for
// shadows), skip the sprite's own card.
//
// With an lpv it returns vec4(gi.rgb, visibility) instead: the same six cones
// are the GI gather's short band (§6.2), so the bounce costs six texture reads
// on top of the AO, not six more marches. Each cone samples the LPV where it
// ENDED - pulled back half a cell off the surface it hit, or at full reach when
// it escaped - and the samples are weighted as the occlusion is. Sampling at the
// hit rather than at the receiver is what makes the bounce directional: the
// cones that struck a red wall bring back red, the ones that escaped bring back
// whatever fills the open air.
export function createAOTSL({ cascades, distance, biasUniform = null, lpv = null }) {
  const cone = createAOConeTSL(cascades);
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const R = float(AO_CONE_R);
  // A plain function, called inline: it returns three things.
  //   vis   the visibility, as before
  //   gi    the gathered bounce (null without an lpv)
  //   bent  the bent normal - the cone directions weighted by how open each
  //         is. Where the left is blocked it leans right, so the sky ambient
  //         read along it comes from the side that can actually see sky.
  //         Free: the cones were marched anyway.
  //
  // A loop, not six unrolled copies, and its count is a node: `single` (a bool
  // node, or null) drops it to the one cone along the normal, weighted 1, for
  // shading LOD - see createShadowColorNode's lod.
  return (p, n, lift, skip, single = null) => {
    const origin = p.add(lift.mul(bias.mul(float(VOXEL_METRES)))).toVar();
    const helper = select(abs(n.y).lessThan(float(0.999)), vec3(0, 1, 0), vec3(1, 0, 0));
    const T = normalize(cross(helper, n)).toVar();
    const B = cross(n, T).toVar();
    const vis = float(1).toVar();
    const gi = vec3(0, 0, 0).toVar();
    const bent = vec3(0, 0, 0).toVar();
    // single === true compiles the one-cone case in; a node selects at runtime.
    const count = single === true ? int(1)
      : single ? select(single, int(1), int(AO_CONES)) : int(AO_CONES);
    Loop({ start: int(0), end: count, type: 'int', condition: '<', name: 'aoK' }, ({ aoK }) => {
      // Cone 0 straight up the normal; 1-5 tilted AO_TILT, 72 degrees apart.
      const up = aoK.equal(int(0));
      const phi = float(aoK.sub(int(1))).mul(float(2 * Math.PI / 5));
      const st = select(up, float(0), float(Math.sin(AO_TILT)));
      const ct = select(up, float(1), float(Math.cos(AO_TILT)));
      const dir = normalize(T.mul(st.mul(cos(phi))).add(B.mul(st.mul(sin(phi))))
                            .add(n.mul(ct))).toVar();
      const w0 = select(up, float(AO_WEIGHT_UP), float(AO_WEIGHT_SIDE));
      const w = (single === true ? float(1)
        : single ? select(single, float(1), w0) : w0).toVar();
      const r = cone(origin, dir, distance, R).toVar();
      vis.subAssign(r.x.mul(w));
      bent.addAssign(dir.mul(float(1).sub(r.x).mul(w)));
      if (lpv) {
        const back = select(r.z.greaterThan(float(0)),
                            max(r.y.sub(float(LPV_CELL_METRES * 0.5)), float(0)), r.y);
        gi.addAssign(lpvSampleTSL(lpv, origin.add(dir.mul(back))).mul(w));
      }
    });
    const v = clamp(vis, float(0), float(1));
    // Fully enclosed: no open direction, and the ambient is zero anyway.
    const b = select(length(bent).greaterThan(float(1e-4)), normalize(bent), n);
    return { vis: v, gi: lpv ? gi : null, bent: b.toVar() };
  };
}

// The terrain's AO for its PLAIN material - the one it wears with shadows off:
// the static voxel AO of voxelao.js, this is its shader mirror. No field, no
// marches: eight occupancy reads from the block-material volume, at the texel
// centre so it stays one answer per texel. A float for the material's aoNode,
// which in three's lighting darkens the ambient only, not the sun.
// cards: the sun's card bindings, for the sprites' contact discs on the
// ground (createSpriteAOTSL) - analytic, no marches. Null until they load.
export function createVoxelAONode(terrain, cards = null) {
  const vol = terrain.blocks;
  const spriteAO = createSpriteAOTSL(cards);
  return Fn(() => {
    const n = axisNormalTSL(normalWorld).toVar();
    const p = texelLockTSL(positionWorld, n).toVar();
    const block = boxFaceTSL(p, n).block.toVar();
    // The face's two tangent axes, and where the texel sits along each, 0..1.
    const t1 = select(abs(n.x).greaterThan(float(0.5)), vec3(0, 1, 0), vec3(1, 0, 0)).toVar();
    const t2 = select(abs(n.z).greaterThan(float(0.5)), vec3(0, 1, 0), vec3(0, 0, 1)).toVar();
    const l = p.div(float(BLOCK_METRES)).sub(block).toVar();
    const a = l.dot(t1).add(float(0.5)), b = l.dot(t2).add(float(0.5));
    // Occupancy in the layer in front of the face; off the volume is air.
    const front = block.add(n).toVar();
    const occ = (i, j) => {
      const uvw = front.add(t1.mul(float(i))).add(t2.mul(float(j)))
        .sub(vol.origin).add(float(0.5)).div(vol.size);
      const inside = uvw.x.greaterThan(float(0)).and(uvw.y.greaterThan(float(0)))
        .and(uvw.z.greaterThan(float(0))).and(uvw.x.lessThan(float(1)))
        .and(uvw.y.lessThan(float(1))).and(uvw.z.lessThan(float(1)));
      const filled = texture3D(vol.tex, uvw).level(int(0)).r.greaterThan(float(0.5 / 255));
      return select(inside.and(filled), float(1), float(0)).toVar();
    };
    const e = { l: occ(-1, 0), r: occ(1, 0), d: occ(0, -1), u: occ(0, 1) };
    const corner = (s1, s2, c) => select(s1.mul(s2).greaterThan(float(0.5)), float(0),
                                         float(3).sub(s1.add(s2).add(c)).div(float(3)));
    const c00 = corner(e.l, e.d, occ(-1, -1)), c10 = corner(e.r, e.d, occ(1, -1));
    const c01 = corner(e.l, e.u, occ(-1, 1)), c11 = corner(e.r, e.u, occ(1, 1));
    const v = mix(mix(c00, c10, a), mix(c01, c11, a), b);
    const st = boxFaceTSL(p, n).st;
    const texAO = texture(terrain.normal, st).depth(terrainLayerTSL()).b;
    const ao = float(VOXEL_AO_MIN).add(float(1 - VOXEL_AO_MIN).mul(v)).mul(texAO);
    return spriteAO ? ao.mul(spriteAO(p, vec2(1e9, 1e9))) : ao;
  })();
}

// --- Sprite contact AO: a soft disc under each sprite ---
//
// Sprites are flat cards, and cone-tracing AO against them read as noise, so
// their ambient occlusion is the classic analytic blob instead: a disc on the
// ground under the feet, radius half the sprite's width, darkest at the centre
// and fading out both across the disc and with height above the feet. Evaluated
// at the texel-locked position, so it lands in whole texels like everything
// else. Multiplied per sprite, so two standing close darken together.
export const SPRITE_AO_STRENGTH = 0.6;
export const SPRITE_AO_RADIUS = 0.5;   // x the sprite's width

export function createSpriteAOTSL(b) {
  if (!b || !b.atlas) return null;
  return Fn(([p, skip]) => {
    const ao = float(1).toVar();
    Loop({ start: int(0), end: b.count, type: 'int', condition: '<' }, ({ i }) => {
      const A = b.data.element(i.mul(int(4)));
      const B = b.data.element(i.mul(int(4)).add(int(1)));
      const radius = B.z.mul(float(SPRITE_AO_RADIUS * 2));
      const r = length(p.xz.sub(A.xz));
      const above = max(p.y.sub(A.y.sub(B.w)), float(0));   // over the feet
      const occ = float(1).sub(smoothstep(float(0), radius, r))
        .mul(float(1).sub(smoothstep(float(0), radius, above)))
        .mul(float(SPRITE_AO_STRENGTH));
      // Not under its own feet - a sprite does not occlude itself this way.
      const own = length(A.xz.sub(skip)).lessThan(float(0.05));
      ao.mulAssign(select(own, float(1), float(1).sub(occ)));
    });
    return ao;
  });
}

// Terrain has no card of its own to skip.
const NO_SKIP = vec2(1e9, 1e9);

// The sprite equivalent of texelLockTSL: snap to the centre of the SPRITE texel
// the fragment falls in, so a shadow crossing a character lands in whole art
// pixels. The quad is a plane, so p is linear in uv, and the screen derivatives
// of both give dp/du and dp/dv exactly. Computed before any branch, which is
// where derivatives are defined.
export const spriteTexelLockTSL = Fn(([p, st, size]) => {
  const dpx = dFdx(p).toVar(), dpy = dFdy(p).toVar();
  const dux = dFdx(st).toVar(), duy = dFdy(st).toVar();
  const det = dux.x.mul(duy.y).sub(dux.y.mul(duy.x));
  const inv = float(1).div(select(abs(det).greaterThan(float(1e-20)), det, float(1e-20)));
  const dPdu = dpx.mul(duy.y).sub(dpy.mul(dux.y)).mul(inv);
  const dPdv = dpy.mul(dux.x).sub(dpx.mul(duy.x)).mul(inv);
  const d = st.mul(size).floor().add(float(0.5)).div(size).sub(st);
  return p.add(dPdu.mul(d.x)).add(dPdv.mul(d.y));
});

// A sprite's material, lit: its own colour and alpha, times the light at it.
// A sprite's shading normal, in world space. The quad's own axes are the
// tangent frame - x is its width, y its height, z the way it faces - read from
// the model matrix, so it follows the billboard yaw and lean. Two sign flips:
// a mirrored sprite (map.repeat.x = -1) runs u backwards across the quad, and a
// back face faces the other way. No normal map: the quad's own facing.
// Returns { n, ao }: the world normal and the texture's baked AO (LabPBR blue).
// The quad's frame comes from its vertices' own normal and tangent, passed
// flat - the exact attribute values, not interpolated - so a plain quad
// (tangent +x, normal +z) computes precisely what reading the model matrix's
// axes did, and a tree's four planes can share ONE mesh and one draw, each
// plane still facing its own way (sprites.js crossedPlanesGeometry). Every
// sprite geometry must carry a tangent (sprites.js addSpriteTangent).
function spriteShadeNormalTSL(map, normalTex) {
  const tan = varying(attribute('tangent', 'vec4').xyz, 'vSpriteTangent')
    .setInterpolation(THREE.InterpolationSamplingType.FLAT);
  const nrm = varying(attribute('normal', 'vec3'), 'vSpriteNormal')
    .setInterpolation(THREE.InterpolationSamplingType.FLAT);
  const right = normalize(modelWorldMatrix.mul(vec4(tan, 0)).xyz);
  const up = normalize(modelWorldMatrix.mul(vec4(0, 1, 0, 0)).xyz);
  const fwd = normalize(modelWorldMatrix.mul(vec4(nrm, 0)).xyz);
  const face = select(frontFacing, fwd, fwd.negate());
  if (!normalTex) return { n: face, ao: float(1) };
  // Live references to the map's own repeat and offset, so flipping the sprite
  // in updateSpriteFacing flips its normals in the same frame.
  const repeat = uniform(map.repeat), offset = uniform(map.offset);
  const t = labNormalTSL(texture(normalTex, uv().mul(repeat).add(offset))).toVar();
  return { n: normalize(right.mul(t.x.mul(sign(repeat.x)))
                        .add(up.mul(t.y)).add(face.mul(t.z))),
           ao: t.w };
}

export function createSpriteShadowMaterial(base, spriteLight, normalTex = null) {
  const mat = new THREE.MeshBasicNodeMaterial({
    map: base.map, transparent: base.transparent,
    alphaTest: base.alphaTest, side: base.side,
    // Carried because the character clip fix flips these on the live material.
    depthTest: base.depthTest, depthWrite: base.depthWrite
  });
  const base4 = materialColor.toVar();
  const drawn = base4.a.greaterThanEqual(float(base.alphaTest || 0));
  mat.colorNode = base4.mul(vec4(spriteLight(base.map, drawn, normalTex), 1));
  return mat;
}

// A tangent-space normal map applied to a surface with no stored tangents -
// the terrain's boxes have none. The frame comes from screen derivatives of
// position and uv (Schuler's cotangent frame), which for a flat face is exact.
// Derivatives, so call it before any branch.
// A LabPBR normal-map texel to vec4(x, y, z, ao): x right, y UP (the stored
// green is DirectX, down), z rebuilt from the other two, ao from blue.
export const labNormalTSL = Fn(([c]) => {
  const x = c.r.mul(2).sub(1), y = float(1).sub(c.g.mul(2));
  const z = sqrt(max(float(1).sub(x.mul(x)).sub(y.mul(y)), float(0)));
  return vec4(x, y, z, c.b);
});

// t is the tangent-space normal from labNormalTSL, y up.
export const bumpedNormalTSL = Fn(([n, pos, st, t]) => {
  const dp1 = dFdx(pos).toVar(), dp2 = dFdy(pos).toVar();
  const du1 = dFdx(st).toVar(), du2 = dFdy(st).toVar();
  const dp2perp = cross(dp2, n), dp1perp = cross(n, dp1);
  const T = dp2perp.mul(du1.x).add(dp1perp.mul(du2.x)).toVar();
  const B = dp2perp.mul(du1.y).add(dp1perp.mul(du2.y)).toVar();
  const s = inverseSqrt(max(max(T.dot(T), B.dot(B)), float(1e-20)));
  return normalize(T.mul(s).mul(t.x).add(B.mul(s).mul(t.y)).add(n.mul(t.z)));
});

// The same, for the terrain, with the frame read off the face instead of
// reconstructed from derivatives. The terrain is axis-aligned boxes, so every
// face's tangent (the way u runs) and bitangent (the way v runs) are known -
// they are BoxGeometry's own uv layout, the table hitAlbedoTSL uses.
//
// Why not the derivative frame: bumpedNormalTSL scales T and B by the LONGER
// of the two, so on a face seen at an angle - where u and v shrink on screen
// by different amounts - the bent normal leans by a different amount along
// each axis, and that changes with the view. Reflections, which follow the
// normal, squished with the camera; and the derivatives are per 2x2 pixel
// quad, so a texel's pixels did not even agree. This frame is exact and has
// no view in it: one normal per texel, from any angle.
//
//   +x: T -z, B +y    -x: T +z, B +y
//   +y: T +x, B -z    -y: T +x, B +z
//   +z: T +x, B +y    -z: T -x, B +y
export const boxBumpedNormalTSL = Fn(([n, t]) => {
  const ax = abs(n.x).greaterThan(float(0.5));
  const ay = abs(n.y).greaterThan(float(0.5));
  const T = select(ax, vec3(0, 0, n.x.negate()),
                   select(ay, vec3(1, 0, 0), vec3(n.z, 0, 0)));
  const B = select(ay, vec3(0, 0, n.y.negate()), vec3(0, 1, 0));
  return normalize(T.mul(t.x).add(B.mul(t.y)).add(n.mul(t.z)));
});

// Normal maps are data, not colour: no sRGB, nearest so each art texel has one
// normal, repeat because block textures tile.
export function createNormalTexture(data, w, h) {
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------
// 7. Specular (§6.3): LabPBR decode, GGX highlights, one reflection ray
//
// The terrain's specular texel (materials.js has the CPU mirror of the decode)
// becomes three things per fragment:
//
//   highlights  GGX from the sun and every point light, taking the SAME
//               visibility the diffuse term marched - no extra rays.
//   reflection  one ray along reflect(-V, N), marched through the field like
//               any shadow ray. Where it lands, the hit is shaded on the spot:
//               its block's own texture (looked up in the block-material
//               volume, render.js), the sun and the light list with their
//               shadow marches, and the LPV bounce. Past the field, the sky.
//   fresnel     Schlick, roughness-aware, weighting the reflection; the diffuse
//               is scaled by what the reflection took, and metals have none.
//
// There is no surface-radiance grid yet (§7.4), so the hit is lit on the fly
// rather than read from a cache. What that costs: one reflection march plus
// one sun march per reflective fragment, and nothing for any other.
//
// Everything is evaluated at the texel-locked position, the view vector too,
// so a highlight or a reflection lands in whole texels like the shadows do.
// ---------------------------------------------------------------------------
export const REFLECT_MAX_DISTANCE = 24;   // metres
// Rough surfaces fall back from the ray to the ambient estimate over this
// band of alpha (LabPBR roughness, (1 - smoothness)^2). Grass, at alpha 1,
// never marches.
export const REFLECT_ROUGH_START = 0.2;
export const REFLECT_ROUGH_END = 0.45;
// GGX with a near-zero alpha is a delta: a highlight one texel wide and
// thousands bright, flickering in and out as the view moves. The sun is a disc,
// not a point, so its highlight on a mirror is a disc too - this is that floor.
export const SPEC_MIN_ALPHA = 0.03;
// Past this a highlight is clipped white anyway; the cap keeps a torch against
// polished metal from blooming into a flat white rectangle.
export const SPEC_MAX = 4;

// LabPBR specular texel (0-1 per channel) -> { alpha, f0, metal }.
export function decodeSpecularTSL(s, albedo) {
  const g = s.g.mul(float(255)).toVar();
  const metal = select(g.greaterThanEqual(float(229.5)), float(1), float(0)).toVar();
  // 230-237: the standard's predefined metals, F0 from materials.js's n/k table.
  const idx = clamp(round(g.sub(float(230))), float(0), float(7)).toVar();
  let table = vec3(...METAL_F0[7]);
  for (let i = 6; i >= 0; i--) table = select(idx.equal(float(i)), vec3(...METAL_F0[i]), table);
  // 238-255: some other metal - the albedo is its F0.
  const metalF0 = select(g.greaterThanEqual(float(237.5)), albedo, table);
  const f0 = select(metal.greaterThan(float(0.5)), metalF0, vec3(s.g, s.g, s.g)).toVar();
  const rough = float(1).sub(s.r);
  return { alpha: rough.mul(rough).toVar(), f0, metal };
}

// Fresnel for the reflection, Schlick with a roughness term: a rough surface's
// grazing reflection does not climb all the way to white.
function fresnelRoughTSL(f0, nv, alpha) {
  const k = pow(float(1).sub(nv), float(5));
  return f0.add(max(vec3(float(1).sub(alpha)), f0).sub(f0).mul(k));
}

// GGX + height-correlated Smith + Schlick, times N.L and PI - the scale the
// diffuse is in here (albedo x light x N.L: a Lambert BRDF of albedo / PI with
// the PI folded into the light). Returns a function of the light direction,
// for the light loop to call per light.
function specularTSL(n, v, alpha, f0) {
  const a = max(alpha, float(SPEC_MIN_ALPHA));
  const a2 = a.mul(a).toVar();
  const nv = max(n.dot(v), float(1e-4)).toVar();
  return l => {
    const h = normalize(l.add(v));
    const nl = max(n.dot(l), float(0));
    const nh = max(n.dot(h), float(0));
    const vh = max(v.dot(h), float(0));
    const dd = nh.mul(nh).mul(a2.sub(float(1))).add(float(1));
    const D = a2.div(float(Math.PI).mul(dd).mul(dd));
    const vis = float(0.5).div(max(
      nl.mul(sqrt(nv.mul(nv).mul(float(1).sub(a2)).add(a2)))
        .add(nv.mul(sqrt(nl.mul(nl).mul(float(1).sub(a2)).add(a2)))), float(1e-5)));
    const F = f0.add(vec3(1, 1, 1).sub(f0).mul(pow(float(1).sub(vh), float(5))));
    return F.mul(D.mul(vis).mul(nl).mul(float(Math.PI)));
  };
}

// The reflection march: finest level first, carrying t, like the shadow
// traces - but it returns WHERE it stopped. vec4(face normal, t), t < 0 for a
// miss (left the field or ran out of distance). The normal is the field's
// gradient at the hit, snapped to its dominant axis: the terrain is boxes, so
// that is the exact face, and the face is what picks the texture's uv.
export function createReflectTraceTSL(cascades) {
  return memoCascades(cascades, 'reflect', () => Fn(([origin0, dir0, maxDist]) => {
    const origin = origin0.toVar(), dir = dir0.toVar();
    const dim = float(GRID_DIM);
    const t = float(0).toVar();
    const done = float(0).toVar();
    const out = vec4(0, 0, 0, -1).toVar();
    for (const { c } of eachCascade(cascades, float(0))) {
      If(done.equal(float(0)), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel).toVar();
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => { Break(); });
          const at = q => decodeFieldTSL(texture3D(c.tex, ringUV(q, c.ring)).r, c.range);
          const d = at(vp).toVar();
          If(d.lessThan(float(HIT_EPS)), () => {
            const g = vec3(at(vp.add(vec3(1, 0, 0))).sub(at(vp.sub(vec3(1, 0, 0)))),
                           at(vp.add(vec3(0, 1, 0))).sub(at(vp.sub(vec3(0, 1, 0)))),
                           at(vp.add(vec3(0, 0, 1))).sub(at(vp.sub(vec3(0, 0, 1))))).toVar();
            // Inside a solid the gradient can vanish; the ray came from the
            // open side, so the face it met faces back along it.
            g.assign(select(length(g).lessThan(float(1e-4)), dir.negate(), g));
            const ag = abs(g).toVar();
            const nx = ag.x.greaterThanEqual(ag.y).and(ag.x.greaterThanEqual(ag.z));
            const ny = ag.y.greaterThan(ag.x).and(ag.y.greaterThanEqual(ag.z));
            const face = select(nx, vec3(sign(g.x), 0, 0),
                                select(ny, vec3(0, sign(g.y), 0), vec3(0, 0, sign(g.z))));
            out.assign(vec4(face, t));
            done.assign(float(1));
            Break();
          });
          t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }
    return out;
  }).setLayout(layout('reflectTrace', 'vec4', [
    { name: 'origin0', type: 'vec3' }, { name: 'dir0', type: 'vec3' },
    { name: 'maxDist', type: 'float' }])));
}

// Which block a surface point belongs to, and its uv on that face - the uv
// BoxGeometry itself gives that face (the table in boxBumpedNormalTSL), so a
// texture read here matches the mesh's own. p on the face, n axis-exact.
// { block: the block's grid coordinate, st: the face uv }.
function boxFaceTSL(p, n) {
  const block = round(p.sub(n.mul(float(VOXEL_METRES * 0.5))).div(float(BLOCK_METRES))).toVar();
  const l = p.div(float(BLOCK_METRES)).sub(block).toVar();
  const h = float(0.5);
  const uvX = vec2(select(n.x.greaterThan(float(0)), l.z.negate(), l.z).add(h), l.y.add(h));
  const uvY = vec2(l.x.add(h), select(n.y.greaterThan(float(0)), l.z.negate(), l.z).add(h));
  const uvZ = vec2(select(n.z.greaterThan(float(0)), l.x, l.x.negate()).add(h), l.y.add(h));
  const st = select(abs(n.x).greaterThan(float(0.5)), uvX,
                    select(abs(n.y).greaterThan(float(0.5)), uvY, uvZ));
  return { block, st };
}

// A reflection hit's albedo: which block it is (render.js's block-material
// volume), then that block's texture at the face uv - the uv BoxGeometry
// itself gives that face, so a reflected block shows its texture the right
// way round. l is the hit in block units, -0.5..0.5 across the block.
function hitAlbedoTSL(terrain, hp, nh) {
  const f = boxFaceTSL(hp, nh);
  const a = texture(terrain.albedo, f.st).depth(blockLayerTSL(terrain, f.block)).level(int(0)).rgb;
  return mix(a, vec3(1, 1, 1), terrainWhite);
}

// A block's material layer, from render.js's block-material volume. Stored as
// layer + 1; 0 is air, and outside the volume reads as layer 0.
function blockLayerTSL(terrain, block) {
  const vol = terrain.blocks;
  const uvw = block.sub(vol.origin).add(float(0.5)).div(vol.size);
  const code = round(texture3D(vol.tex, uvw).level(int(0)).r.mul(float(255)));
  return int(max(code.sub(float(1)), float(0)));
}

// --- Mirrors: reflected sunlight, gathered (mirrors.js) ---
//
// The light a polished surface throws onto other surfaces. mirrors.js has the
// idea: from a receiving texel, the sun mirrored in a mirror's plane is one
// direction, D = reflect(L, N); if a ray that way lands on the mirror, the
// receiver sees the sun in it. Then it is lit by:
//
//   sun colour x sun scale x max(N_recv . D, 0)       as the direct sun, from D
//   x F                  the mirror TEXEL's Fresnel at the sun's angle, from its
//                        own _s: F0 or metal, so a seam that is rough and
//                        dielectric throws almost nothing and the patch on the
//                        wall carries the plate's pattern
//   x smooth             1 over the smooth band, 0 past REFLECT_ROUGH_END - a
//                        rough surface scatters rather than mirrors
//   x vis (mirror->sun)  the sun's own cone trace and cards at the hit
//   x vis (recv->mirror) a cone trace along D, as wide as the sun's, stopped
//                        short of the mirror so its own face is not a blocker
//
// Two marches, and only for a texel whose ray actually lands on a mirror;
// every other texel pays one plane test per rectangle. The sun only: a point
// light's mirror image depends on which plane, so it needs its own pass.
export function createMirrorBindings(capacity) {
  return {
    capacity,
    data: uniformArray(new Array(capacity * MIRROR_VEC4S).fill(0).map(() => new THREE.Vector4()), 'vec4')
      .setName('bxbMirrors'),
    count: uniform(int(0))
  };
}
export function writeMirrorBindings(b, packed, count) {
  for (let i = 0; i < count * MIRROR_VEC4S; i++) {
    b.data.array[i].set(packed[i * 4], packed[i * 4 + 1], packed[i * 4 + 2], packed[i * 4 + 3]);
  }
  b.count.value = count;
  return b;
}

// --- Sprites in reflections: the card hit test ---
//
// Sprites are not in the field, so a reflection ray cannot land on one by
// marching. It is tested against the VIEW cards instead - one per sprite,
// turned the way the drawn billboard faces (toward the camera's heading) - and
// where it crosses one inside the sprite's opaque pixels, that is a hit. The
// colour comes from the colour atlas (buildCardAtlas), addressed by the same
// rect as the silhouette; the pixel mapping mirrors cardPixelAt in cards.js.
//
// Nearest-neighbour and a hard alpha cut, like the sprite's own draw: a
// reflected sprite is the same pixel art, not a filtered copy of it.
export function createCardColourTexture(atlas) {
  const tex = new THREE.DataTexture(atlas.colour, atlas.width, atlas.height,
                                    THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.generateMipmaps = false;
  tex.needsUpdate = true;
  return tex;
}

// Returns a function, called inline from the shading Fn:
//   (from, dir, maxDist) -> { col, t, skip }
// col the sprite's linear colour, t the distance along the ray (< 0 for no
// hit nearer than maxDist), skip the card's XZ - so the sprite's own lighting
// can leave its own card out of its shadow test.
export function createCardHitTSL(b, colourTex) {
  if (!b || !colourTex) return null;
  return (from, dir, maxDist) => {
    const best = maxDist.toVar();
    const t = float(-1).toVar();
    const col = vec3(0, 0, 0).toVar();
    const skip = vec2(1e9, 1e9).toVar();
    Loop({ start: int(0), end: b.count, type: 'int', condition: '<', name: 'vc' },
         ({ vc }) => {
      const base = vc.mul(int(4));
      const A = b.data.element(base).toVar();
      const B = b.data.element(base.add(int(1))).toVar();
      const C = b.data.element(base.add(int(2))).toVar();
      const D = b.data.element(base.add(int(3))).toVar();
      const ux = B.x, uz = B.y;
      const nx = uz.negate(), nz = ux;
      const denom = dir.x.mul(nx).add(dir.z.mul(nz));
      If(abs(denom).greaterThan(float(1e-6)), () => {
        const th = A.x.sub(from.x).mul(nx).add(A.z.sub(from.z).mul(nz)).div(denom).toVar();
        If(th.greaterThan(float(0)).and(th.lessThan(best)), () => {
          const h = from.add(dir.mul(th)).sub(A.xyz);
          const a = h.x.mul(ux).add(h.z.mul(uz)).mul(A.w);
          const mpp = D.x, cols = D.y, rows = D.z;
          const pc = floor(a.div(mpp).add(cols.mul(float(0.5))));
          const pr = floor(rows.mul(float(0.5)).sub(h.y.div(mpp)));
          If(pc.greaterThanEqual(float(0)).and(pc.lessThan(cols))
               .and(pr.greaterThanEqual(float(0))).and(pr.lessThan(rows)), () => {
            const st = vec2(C.x.add(float(CARD_PAD)).add(pc).add(float(0.5)).div(b.atlasSize.x),
                            C.y.add(float(CARD_PAD)).add(pr).add(float(0.5)).div(b.atlasSize.y));
            const c = texture(colourTex, st).level(int(0)).toVar();
            If(c.a.greaterThanEqual(float(0.5)), () => {
              best.assign(th);
              t.assign(th);
              col.assign(c.rgb);
              skip.assign(A.xz);
            });
          });
        });
      });
    });
    return { col, t, skip };
  };
}

export function createShadowColorNode({ cascades, sunDirection, maxDistance = 12,
                                        ambientUniform = null, shadeMode = 'ground',
                                        rays = 1, shadowOnly = false, lights = null, sun: sunOn = true,
                                        angular = SUN_ANGULAR_SIZE,
                                        bayerTex = null, stepsUniform = null,
                                        biasUniform = null,
                                        cone = false, quantise = true,
                                  fadeStartUniform = null, edgeFadeUniform = null,
                                  cards = null, lightCards = null,
                                  terrain = null, specularOnly = false,
                                  albedo = null, specular: specOn = true,
                                  reflections: reflectOn = true, sky = null,
                                  mirrors = null, mirrorOnly = false,
                                  viewCards = null,
                                  ao: aoOn = true, aoDistanceUniform = null,
                                  aoOnly = false, gi = null, giOnly = false }) {
  // Held as uniforms whose .value is live, so moving the sun is a uniform write
  // rather than a material rebuild.
  const sun = createSunUniforms(sunDirection, angular);
  const maxDist = uniform(float(maxDistance));
  const ambient = ambientUniform || uniform(float(DEFAULT_AMBIENT));
  const sunShadow = createSoftSunTSL({
    cascades, sunDir: sun.dir, sunTangent: sun.tangent,
    sunBitangent: sun.bitangent, maxDist, coneRadiusUniform: sun.coneRadius,
    rayCount: rays, bayerTex, stepsUniform, biasUniform, cone, quantise,
    fadeStartUniform, edgeFadeUniform, cards
  });

  const aoDistance = aoDistanceUniform || uniform(float(AO_DISTANCE));
  // The cones are built for AO, for GI, or both - GI gathers along them - so
  // AO off with GI on still marches them and just ignores the occlusion.
  const cones = aoOn || gi
    ? createAOTSL({ cascades, distance: aoDistance, biasUniform, lpv: gi }) : null;
  const spriteAO = aoOn ? createSpriteAOTSL(cards) : null;
  // { ao, gi }: terrain cones times the sprite discs times the texture's own
  // AO, and the gathered bounce (null without GI). The bounce is scaled by the
  // texture AO too - a crevice in the art receives less of it.
  // bent: the cones' bent normal, or n without cones.
  // single: one AO cone instead of six (see createAOTSL) - what sprites use.
  // Measured (1080p): the six cones on the trees' foliage cost 0.34 ms of
  // 2.35, because an alpha-tested crossed-plane tree shades many layers deep;
  // on the terrain they cost 0.05. A sprite's AO comes mostly from its contact
  // disc and its normal map's own AO, so one cone along its normal is plenty.
  // (A cascade LOD - cheaper shading past C0 - was tried and removed: it was
  // no faster, and it dropped reflections at distance.)
  const occlusion = (p, n, lift, skip, texAO, single = false) => {
    if (!cones) return { ao: texAO, gi: null, bent: n };
    const r = cones(p, n, lift, skip, single || null);
    let vis = aoOn ? r.vis : float(1);
    if (spriteAO) vis = vis.mul(spriteAO(p, skip));
    return {
      ao: vis.mul(texAO).toVar(),
      gi: gi ? r.gi.mul(gi.strength).mul(texAO).toVar() : null,
      bent: aoOn ? r.bent : n
    };
  };

  // --- Ambient: the sky, read along the bent normal ---
  //
  // Without a sky it is the flat ambient uniform, as it always was. With one,
  // the same uniform scales the sky's irradiance (normalised so noon, straight
  // up, is exactly the uniform), so the ambient takes the sky's colour and
  // direction: a wall facing the horizon at dusk goes orange, one facing up
  // goes blue, and an overhang's underside - whose bent normal leans out
  // toward the open side - is lit from that side. One SH evaluation.
  //
  // Deliberately NOT quantised. The ambient is the smoothest light there is;
  // banding it cut steps into a gradient that has no edges, and the dither
  // read as crosshatch. Tried and removed.
  const skyIrradiance = d => (sky
    ? skyShTSL(sky, d, true).mul(ambient).mul(sky.gain)
    : vec3(ambient, ambient, ambient));
  const ambientLight = (bent, ao) => skyIrradiance(bent).mul(ao);
  // The sun's DIRECT term alone - sunShadeTSL with its ambient zeroed, so the
  // ambient can come from the sky instead. Scalar.
  const sunDirect = (visible, n, mode) =>
    sunShadeTSL({ visible, n, sunDir: sun.dir, ambientUniform: ambient, mode, ao: float(0) });

  // Every point light, in one loop over the list. Compiled in whenever there is
  // a list at all: an empty list is a loop of zero iterations, so turning lights
  // on and off is a uniform write rather than a rebuild.
  const pointLight = lights
    ? createPointLightsTSL({ cascades, lights, biasUniform,
                             fadeStartUniform, edgeFadeUniform, cards: lightCards })
    : null;

  // The specular path needs the terrain's arrays (its _s texels) and its albedo,
  // which it now owns: diffuse and specular are no longer one multiply apart.
  const spec = specOn && terrain && albedo;
  const reflectTrace = spec && reflectOn ? createReflectTraceTSL(cascades) : null;
  const reflectMax = float(REFLECT_MAX_DISTANCE);
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  // The sun's light, in the units the diffuse term uses. sunShadeTSL's 'ground'
  // mode divides N.L by a horizontal surface's so the ground reads fully lit
  // at any elevation; the highlight takes the same scale or it would dim as the
  // sun sets while the ground beside it did not.
  const sunLight = () => {
    const e = float(1).sub(ambient);
    return shadeMode === 'ground'
      ? e.div(max(sun.dir.y, float(GROUND_REFERENCE_FLOOR))) : e;
  };

  // Sprites, for the reflection ray: the view cards (see createCardHitTSL).
  const spriteHit = reflectTrace ? createCardHitTSL(viewCards, viewCards && viewCards.colour)
                                 : null;

  // The light at a reflection hit: the sun (with its shadow march - lifted
  // along `lift`, leaving `skip`'s card out), the light list WITH its shadows,
  // and the bounce. `mode` as the surface's own shading.
  //
  // Shadowed on purpose. It was unshadowed to spare a reflective fragment a
  // second loop of marches, and the result was a reflection that disagreed
  // with the thing reflected: a block in a torch's shadow, or round a corner
  // from a lamp, came back lit in the mirror, and no torch or lamp shadow ever
  // appeared in one. The loop is the same one the surface itself runs - same
  // shadow budget, same sprite cards - so the mirror shows what is there.
  const hitLight = (p, n, lift, skip, mode) => {
    const light = skyIrradiance(n).toVar();
    if (sunOn) {
      const vis = float(0).toVar();
      If(n.dot(sun.dir).greaterThan(float(0)), () => {
        vis.assign(sunShadow(p, lift, skip));
      });
      light.addAssign(sunColourUniform.mul(sunDirect(vis, n, mode)));
    }
    // Lifted along `lift` as the sun's ray is; N.L against the hit's normal.
    if (pointLight) light.addAssign(pointLight(p, lift, skip, n));
    if (gi) {
      light.addAssign(lpvSampleTSL(gi, p.add(n.mul(float(LPV_CELL_METRES * 0.5))))
                        .mul(gi.strength));
    }
    return light;
  };

  // Reflected light from the mirrors - see createMirrorBindings. Needs the
  // terrain arrays, for each mirror texel's own reflectivity.
  //
  // The sun AND the light list. A point light's mirror image is exact for a
  // plane, and every rectangle carries its plane: the light at P, reflected in
  // it, sits at P with the plane's axis flipped across the plane. A receiver
  // sees the light in that mirror when the line to the image crosses the plane
  // inside the rectangle; the light then arrives from the image's direction,
  // attenuated over the WHOLE path (receiver -> mirror -> light, which is the
  // distance to the image), and is shadowed on both legs.
  const mirrorTrace = mirrors && terrain && (sunOn || lights)
    ? createConeTraceSunTSL(cascades) : null;
  const mirrorFade = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const mirrorEdge = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));
  // The point lights' sprite cards, for the mirror->light leg. Ranged, as in
  // the light loop: each light walks its own slice.
  const mirrorCards = createCardsTSL(lightCards, { ranged: true });
  // And the sun's, for the sun's receiver->mirror leg. Both card sets are
  // turned to face the REAL light, not its mirror image. For a floor that is
  // exact - the reflection flips height only, so the direction across the
  // ground is unchanged. For an upright mirror the silhouette is a little off
  // its angle; for a sprite-sized shadow that does not show.
  const mirrorSunCards = createCardsTSL(cards);

  // What one mirror texel sends toward the receiver, for light arriving at H
  // from direction Lh: its Fresnel (its own _s, at the bumped normal's angle
  // to the light) times smoothness times the normal-map lobe. back is the way
  // from H to the receiver; minWidth the lobe floor - the source's angular
  // size. vec3; zero for a rough or a tilted texel.
  //
  // The texel's OWN normal matters: the ray was aimed with the flat face
  // normal, and a bevel or a rivet tilted off it sends the light somewhere
  // else. So the lobe weighs how closely where this texel really sends it,
  // reflect(-Lh, Nb), matches `back`. Flat texels pass, tilted ones go dark -
  // the normal map shows in the patch. (Their light lands somewhere this ray
  // is not looking; it is not redistributed.)
  const mirrorTexel = (H, N, Lh, back, minWidth) => {
    const f = boxFaceTSL(H, N);
    const layer = blockLayerTSL(terrain, f.block);
    const s = texture(terrain.specular, f.st).depth(layer).level(int(0));
    const alb = texture(terrain.albedo, f.st).depth(layer).level(int(0)).rgb;
    const d = decodeSpecularTSL(s, alb);
    const smooth = float(1).sub(smoothstep(float(REFLECT_ROUGH_START),
                                           float(REFLECT_ROUGH_END), d.alpha));
    const lab = labNormalTSL(texture(terrain.normal, f.st).depth(layer).level(int(0)));
    const Nb = boxBumpedNormalTSL(N, lab.xyz).toVar();
    const lb = max(Lh.dot(Nb), float(0)).toVar();
    const Rb = Lh.negate().add(Nb.mul(lb.mul(float(2))));
    const width = max(d.alpha, max(minWidth, float(SPEC_MIN_ALPHA)));
    const aligned = exp2(Rb.dot(back).sub(float(1)).div(width.mul(width)).mul(float(1.4427)));
    const F = d.f0.add(vec3(1, 1, 1).sub(d.f0).mul(pow(float(1).sub(lb), float(5))));
    return F.mul(smooth.mul(aligned));
  };

  // skip: the receiving sprite's own card, so it does not block the light
  // reaching itself. Terrain passes NO_SKIP.
  const mirrorLightFn = mirrorTrace ? Fn(([p, nS, lift, skip]) => {
    const sum = vec3(0, 0, 0).toVar();
    const origin = p.add(lift.mul(bias.mul(float(VOXEL_METRES)))).toVar();
    Loop({ start: int(0), end: mirrors.count, type: 'int', condition: '<', name: 'mi' },
         ({ mi }) => {
      const row = mi.mul(int(MIRROR_VEC4S));
      const A = mirrors.data.element(row).toVar();
      const B = mirrors.data.element(row.add(int(1))).toVar();
      // What this mirror can reflect, culled on the CPU (mirrors.js
      // mirrorReach): A.w is a mask - bit i for light i, MIRROR_SUN_BIT for the
      // sun - and C, E the box its reflected sun can reach. Measured: the
      // mirror loop was 0.58 ms of 2.35 at 1080p before this, most of it
      // texels no mirror could ever light.
      const mask = A.w;
      const bit = k => floor(mask.div(exp2(k))).mod(float(2)).greaterThan(float(0.5));
      const C = mirrors.data.element(row.add(int(2)));
      const E = mirrors.data.element(row.add(int(3)));
      const axis = int(A.x);
      const isX = axis.equal(int(0)), isY = axis.equal(int(1));
      const N = select(isX, vec3(A.y, 0, 0), select(isY, vec3(0, A.y, 0), vec3(0, 0, A.y))).toVar();
      const pick = v => select(isX, v.x, select(isY, v.y, v.z));
      const pickU = v => select(isX, v.y, v.x);                   // mirrors.js PLANE_AXES
      const pickV = v => select(isX.or(isY), v.z, v.y);
      // The receiver must be on the mirror's reflecting side.
      const side = pick(origin).sub(A.z).mul(A.y);
      If(side.greaterThan(float(0)), () => {
        // Where a ray from the receiver along dir crosses the plane: the
        // distance, or -1 if it misses the rectangle.
        const crossing = dir => {
          const da = pick(dir);
          const t = A.z.sub(pick(origin)).div(select(abs(da).greaterThan(float(1e-5)), da,
                                                     float(1e-5))).toVar();
          const H = origin.add(dir.mul(t));
          const hu = pickU(H), hv = pickV(H);
          const inside = t.greaterThan(float(VOXEL_METRES))
            .and(hu.greaterThanEqual(B.x)).and(hu.lessThanEqual(B.z))
            .and(hv.greaterThanEqual(B.y)).and(hv.lessThanEqual(B.w));
          return select(inside, t, float(-1));
        };

        // --- the sun, mirrored ---
        if (sunOn) {
          const L = sun.dir;
          const D = L.sub(N.mul(L.dot(N).mul(float(2)))).toVar();     // reflect(L, N)
          const facing = max(nS.dot(D), float(0)).toVar();
          const inBox = origin.x.greaterThanEqual(C.x).and(origin.y.greaterThanEqual(C.y))
            .and(origin.z.greaterThanEqual(C.z)).and(origin.x.lessThanEqual(E.x))
            .and(origin.y.lessThanEqual(E.y)).and(origin.z.lessThanEqual(E.z));
          If(bit(float(Math.log2(MIRROR_SUN_BIT))).and(inBox)
               .and(facing.greaterThan(float(0))).and(L.dot(N).greaterThan(float(0))), () => {
            const t = crossing(D).toVar();
            If(t.greaterThan(float(0)).and(t.lessThan(float(REFLECT_MAX_DISTANCE))), () => {
              // The mirror texel: snapped, like every surface point, so one
              // mirror texel reflects one value.
              const Hl = texelLockTSL(origin.add(D.mul(t)), N).toVar();
              const w = mirrorTexel(Hl, N, L, D.negate(), sun.coneRadius.mul(float(2))).toVar();
              If(w.x.add(w.y).add(w.z).greaterThan(float(1e-3)), () => {
                const toMirror = mirrorTrace(origin, D, t.sub(float(VOXEL_METRES * 2)),
                                             sun.coneRadius, mirrorFade, mirrorEdge).x.toVar();
                // Sprites between the receiver and the mirror - Bob standing on
                // the floor cuts his shadow into its glare on the wall.
                if (mirrorSunCards) {
                  toMirror.mulAssign(mirrorSunCards(origin, D, t, sun.coneRadius, skip));
                }
                const toSun = sunShadow(Hl, N, NO_SKIP);
                sum.addAssign(w.mul(toMirror.mul(toSun).mul(facing))
                               .mul(sunLight()).mul(sunColourUniform));
              });
            });
          });
        }

        // --- every point light, mirrored ---
        if (lights) {
          Loop({ start: int(0), end: lights.count, type: 'int', condition: '<', name: 'ml' },
               ({ ml }) => {
            const base = ml.mul(int(LIGHT_VEC4S));
            const LA = lights.data.element(base).toVar();
            const LB = lights.data.element(base.add(int(1))).toVar();
            const LC = lights.data.element(base.add(int(2))).toVar();
            const P = LA.xyz, level = LA.w;
            // In the mask: on the reflecting side and within reach of it.
            If(bit(float(ml)), () => {
              // Its image: the plane's axis flipped across the plane. N carries
              // the rectangle's sign; times the sign again it is the bare axis.
              const image = P.add(N.mul(A.y).mul(A.z.sub(pick(P)).mul(float(2)))).toVar();
              const toImage = image.sub(origin).toVar();
              const dist = max(length(toImage), float(1e-4)).toVar();
              const D = toImage.div(dist).toVar();
              // The falloff over the whole path, as the light loop's (lights.js
              // lightFalloff): log2(1 + u), reaching zero at the radius.
              const radius = level.mul(float(BLOCK_METRES)).toVar();
              const u = clamp(float(1).sub(dist.div(max(radius, float(1e-4)))), float(0), float(1));
              const falloff = log2(float(1).add(u)).mul(level.div(float(MAX_LIGHT_LEVEL))).toVar();
              const facing = max(nS.dot(D), float(0)).toVar();
              If(falloff.mul(facing).greaterThan(lights.cutoff), () => {
                const t = crossing(D).toVar();
                If(t.greaterThan(float(0)).and(t.lessThan(dist)), () => {
                  const Hl = texelLockTSL(origin.add(D.mul(t)), N).toVar();
                  const toLight = P.sub(Hl).toVar();
                  const legB = max(length(toLight), float(1e-4)).toVar();
                  const Lh = toLight.div(legB).toVar();
                  // The flame's angular size from the mirror is the lobe's floor.
                  const w = mirrorTexel(Hl, N, Lh, D.negate(), LB.w.div(legB)).toVar();
                  If(w.x.add(w.y).add(w.z).greaterThan(float(1e-3)), () => {
                    // Both legs shadowed, as the light loop shadows - the same
                    // budget weight (LC.z) and sprite cards (LC.x, LC.y, LC.w).
                    // The cone opens to the source over the whole path.
                    const slope = LB.w.div(dist);
                    const visible = float(1).toVar();
                    If(LC.z.greaterThan(float(0)), () => {
                      const v = mirrorTrace(origin, D, t.sub(float(VOXEL_METRES * 2)), slope,
                                            mirrorFade, mirrorEdge).x.toVar();
                      const lift2 = Hl.add(N.mul(bias.mul(float(VOXEL_METRES)))).toVar();
                      v.mulAssign(mirrorTrace(lift2, Lh, legB, LB.w.div(legB),
                                              mirrorFade, mirrorEdge).x);
                      if (mirrorCards) {
                        // Both legs: receiver->mirror (leaving the receiver's
                        // own card out) and mirror->light.
                        const start = int(LC.x).toVar();
                        const end = start.add(int(LC.y)).toVar();
                        const cardVis = mirrorCards(origin, D, t, slope, skip, start, end)
                          .mul(mirrorCards(lift2, Lh, legB, LB.w.div(legB), NO_SKIP, start, end));
                        v.mulAssign(mix(float(1), cardVis, LC.w));
                      }
                      visible.assign(mix(float(1), v, LC.z));
                    });
                    sum.addAssign(w.mul(LB.xyz).mul(falloff.mul(facing).mul(visible)));
                  });
                });
              });
            });
          });
        }
      });
    });
    return sum;
  }).setLayout(layout('mirrorLight', 'vec3', [
    { name: 'p', type: 'vec3' }, { name: 'nS', type: 'vec3' }, { name: 'lift', type: 'vec3' },
    { name: 'skip', type: 'vec2' }]))
    : null;
  const mirrorLight = (p, n, nS, lift, skip = NO_SKIP) => mirrorLightFn(p, nS, lift, skip);

  // What a reflection ray sees: the nearer of the terrain it marched to and
  // any sprite card it crosses first. Terrain shows its block's texture; a
  // sprite its own pixels, lit as the sprites are (Lambert, facing back along
  // the ray, since the card was turned to be seen). Past the field, the sky.
  const reflectedTSL = (origin, dir) => {
    const hit = reflectTrace(origin, dir, reflectMax).toVar();
    // A miss sees the sky - the same SH the background draws.
    const col = (sky ? skyShTSL(sky, dir) : vec3(0, 0, 0)).toVar();
    const sp = spriteHit
      ? spriteHit(origin, dir, select(hit.w.greaterThanEqual(float(0)), hit.w, reflectMax))
      : null;
    const terrainWins = sp ? hit.w.greaterThanEqual(float(0)).and(sp.t.lessThan(float(0)))
                           : hit.w.greaterThanEqual(float(0));
    If(terrainWins, () => {
      const nh = hit.xyz.toVar();
      const hp = origin.add(dir.mul(hit.w)).toVar();
      const hpL = texelLockTSL(hp, nh).toVar();
      col.assign(hitAlbedoTSL(terrain, hpL, nh)
                   .mul(hitLight(hpL, nh, nh, NO_SKIP, shadeMode)));
    });
    if (sp) {
      If(sp.t.greaterThanEqual(float(0)), () => {
        const hs = origin.add(dir.mul(sp.t)).toVar();
        const back = vec3(dir.x.negate(), float(0), dir.z.negate());
        const ns = select(length(back).greaterThan(float(1e-4)), normalize(back),
                          vec3(0, 1, 0)).toVar();
        // Lifted straight up, as the sprites' own shading is - see spriteLight.
        col.assign(sp.col.mul(hitLight(hs, ns, vec3(0, 1, 0), sp.skip, 'lambert')));
      });
    }
    return col;
  };

  // Stable per-texel randoms, for the rough-surface jitter below. Hashed from
  // the world texel, so the grain is painted on and does not crawl.
  const texelHash = (wt, k) =>
    fract(sin(wt.dot(vec3(12.9898 + k, 78.233 - k, 37.719 + 2 * k))).mul(float(43758.5453)));

  const node = Fn(() => {
    // normalWorld rather than a derivative reconstruction: the terrain is boxes,
    // so the interpolated normal IS the exact face normal, and the marcher needs
    // it to lift the ray origin off the face along the surface rather than along
    // the ray - see SURFACE_BIAS_VOXELS for why that distinction matters at low
    // sun angles.
    const n = axisNormalTSL(normalWorld).toVar();
    // Locked once, used by every light. Two lights that disagreed about where a
    // surface IS would put their shadows on different texel grids, and the
    // mismatch would read as the torch's shadow crawling against the sun's.
    const p = texelLockTSL(positionWorld, n).toVar();
    // Every texture read at THIS texel's centre, from the locked position -
    // not at the fragment's own uv, which at a texel's edge can read the
    // neighbour's normal or specular while p is still this texel's.
    const st = terrain ? boxFaceTSL(p, n).st.toVar() : null;
    const sample = tex => terrainSampleTSL(tex, st);
    // Shading normal: the face normal bent by the block texture's normal map.
    // The geometric n still lifts rays and decides which faces the sun can
    // reach at all; only N.L sees the bumps.
    // terrain: { normal, specular, albedo, blocks }, from render.js.
    const lab = terrain ? labNormalTSL(sample(terrain.normal)).toVar() : null;
    // The face's exact frame, not a derivative one - see boxBumpedNormalTSL.
    const nS = lab ? boxBumpedNormalTSL(n, lab.xyz).toVar() : n;
    const texAO = lab ? lab.w : float(1);
    // The raw LabPBR specular texel: R smoothness, G F0 / metal, B porosity or
    // SSS - how an authored _s is checked in place.
    if (specularOnly && terrain) return sample(terrain.specular).rgb;
    // The views below show a light term; they keep the albedo multiply they
    // always had.
    // albedo: a function of the face uv (main.js), so it is read here too.
    const base = albedo && st ? albedo(st).toVar() : vec3(1, 1, 1);
    const view = x => base.mul(x);
    // Traced AO times the texture's own. The hemisphere is the geometric face,
    // not the bumped normal - the rays test real geometry.
    const occ = occlusion(p, n, n, NO_SKIP, texAO);
    const ao = occ.ao;
    if (aoOnly) return view(vec3(ao, ao, ao));
    if (giOnly && occ.gi) return view(occ.gi);
    // The bounce adds under everything else - it is light, like the others.
    const plusGI = col => (occ.gi ? col.add(occ.gi) : col);
    // The sky ambient, along the bent normal - bent further by the normal
    // map, so the bumps catch the sky as they catch the sun.
    const amb = ambientLight(normalize(occ.bent.add(nS).sub(n)), ao).toVar();

    // The mirror light alone, raw (not times albedo), so what the mirrors
    // throw can be judged by itself - it is not in the GI bounce view, since
    // it never goes through the LPV.
    if (mirrorOnly) return mirrorTrace ? mirrorLight(p, n, nS, n) : vec3(0, 0, 0);
    // Light thrown here by the mirrors - the iron floor's glare on the wall,
    // from the sun and every lamp. Added under the sun-on and sun-off paths.
    const mirrored = mirrorTrace ? mirrorLight(p, n, nS, n).toVar() : null;
    const plusMirror = col => (mirrored ? col.add(mirrored) : col);

    // The surface's specular response, when it has one. V from the locked
    // position, so the whole texel shares one view direction.
    let surf = null;
    if (spec) {
      const d = decodeSpecularTSL(sample(terrain.specular), base);
      const v = normalize(cameraPosition.sub(p)).toVar();
      surf = { ...d, v, brdf: specularTSL(nS, v, d.alpha, d.f0) };
    }

    // light: the diffuse light (x albedo). direct: the highlights, summed.
    const finish = (light, direct) => {
      if (!surf) return base.mul(light);
      const nv = max(nS.dot(surf.v), float(1e-4));
      const F = fresnelRoughTSL(surf.f0, nv, surf.alpha).toVar();
      // What a rough surface reflects: the ambient light around it, the same
      // estimate the diffuse ambient uses. Smooth surfaces replace it with the
      // ray's answer across the rough band.
      const env = amb.toVar();
      if (occ.gi) env.addAssign(occ.gi);
      if (reflectTrace) {
        const w = float(1).sub(smoothstep(float(REFLECT_ROUGH_START),
                                          float(REFLECT_ROUGH_END), surf.alpha)).toVar();
        If(w.greaterThan(float(0)), () => {
          // The mirror direction, jittered per texel by the roughness - a
          // glossy surface reflects as a stable grain rather than a blur.
          // Half a texel inside, so the along-normal floor is not taken on
          // the face plane itself, where it could round either way.
          const wt = p.sub(n.mul(float(VOXEL_METRES * 0.5))).div(float(VOXEL_METRES))
                      .floor().toVar();
          const jit = vec3(texelHash(wt, 0), texelHash(wt, 1), texelHash(wt, 2))
                        .sub(float(0.5)).mul(surf.alpha.mul(float(2)));
          const r0 = reflect(surf.v.negate(), nS).toVar();
          const rj = normalize(r0.add(jit)).toVar();
          const dir = select(rj.dot(n).greaterThan(float(0.02)), rj, r0).toVar();
          const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();
          env.assign(mix(env, reflectedTSL(origin, dir), w));
        });
      }
      const kd = vec3(1, 1, 1).sub(F).mul(float(1).sub(surf.metal));
      return base.mul(light).mul(kd)
        .add(min(direct, vec3(SPEC_MAX, SPEC_MAX, SPEC_MAX)))
        .add(F.mul(env).mul(texAO));
    };
    const noSpec = vec3(0, 0, 0);
    // The light list: diffuse, and the highlights from the same visibility.
    const points = () => {
      if (!surf) return { diffuse: pointLight(p, n, NO_SKIP, nS), spec: noSpec };
      return pointLight.withSpecular(p, n, NO_SKIP, nS, surf.brdf);
    };

    // Sun off: no march, no shading term, just the ambient floor for anything a
    // dynamic light does not reach.
    if (!sunOn) {
      const dark = plusMirror(amb);
      if (!pointLight) return finish(plusGI(dark), noSpec);
      const pl = points();
      return finish(plusGI(dark.add(pl.diffuse)), pl.spec);
    }
    // Faces turned from the sun get N.L = 0, so the march result is multiplied
    // away - skip it. Not in shadowOnly, which shows the raw term everywhere.
    let visible;
    if (shadowOnly) visible = sunShadow(p, n, NO_SKIP);
    else {
      visible = float(0).toVar();
      If(n.dot(sun.dir).greaterThan(float(0)), () => {
        visible.assign(sunShadow(p, n, NO_SKIP));
      });
    }
    // shadowOnly writes the raw visibility term. Ambient and N.L both compress
    // the shadowed range toward the middle, so acne that is plain here is
    // invisible in the shaded result - which is why it has been hard to name.
    if (shadowOnly) return vec3(visible, visible, visible);
    const sunLit = plusMirror(amb.add(sunColourUniform.mul(sunDirect(visible, nS, shadeMode))));
    const sunSpec = surf
      ? surf.brdf(sun.dir).mul(visible).mul(sunLight()).mul(sunColourUniform) : noSpec;
    if (!pointLight) return finish(plusGI(sunLit), sunSpec);
    const pl = points();
    return finish(plusGI(sunLit.add(pl.diffuse)), sunSpec.add(pl.spec));
  })();

  // Sprites: the same lights, shaded at the sprite's own texel centres and
  // lifted straight up. Up rather than the quad's normal: the quad turns with the
  // camera, and a shadow that moved with the camera would crawl. Built per map,
  // because the texel grid is that texture's.
  //
  // `drawn` gates the marches: a fragment alphaTest is about to discard still
  // ran both of them, and the trees' crossed planes are mostly transparent.
  const spriteLight = (map, drawn, normalTex = null) => Fn(() => {
    const n = vec3(0, 1, 0);
    // Before the branch - the snap takes screen derivatives.
    const p = spriteTexelLockTSL(positionWorld, uv(),
                                 vec2(textureSize(texture(map), int(0)))).toVar();
    const skip = modelWorldMatrix.mul(vec4(0, 0, 0, 1)).xz.toVar();
    const shade = spriteShadeNormalTSL(map, normalTex);
    const nS = shade.n.toVar();
    const texAO = shade.ao.toVar();
    const out = vec3(0, 0, 0).toVar();
    If(drawn, () => {
      // Over the shading normal's hemisphere - a billboard has no geometric
      // face in the field, so the way its texels face is the best there is.
      const occ = occlusion(p, nS, n, skip, texAO, true);
      const ao = occ.ao;
      if (aoOnly) { out.assign(vec3(ao, ao, ao)); return; }
      if (giOnly && occ.gi) { out.assign(occ.gi); return; }
      const plusGI = col => (occ.gi ? col.add(occ.gi) : col);
      if (mirrorOnly) { out.assign(mirrorTrace ? mirrorLight(p, n, nS, n, skip) : vec3(0, 0, 0)); return; }
      // A sprite by a mirror catches its glare too, sun or lamp. Lifted
      // straight up, as the sprite's own rays are.
      const mirrored = mirrorTrace ? mirrorLight(p, n, nS, n, skip) : null;
      const amb0 = ambientLight(occ.bent, ao);
      const amb = mirrored ? amb0.add(mirrored) : amb0;
      if (!sunOn) {
        out.assign(plusGI(pointLight ? amb.add(pointLight(p, n, skip, nS)) : amb));
        return;
      }
      const visible = sunShadow(p, n, skip);
      if (shadowOnly) { out.assign(vec3(visible, visible, visible)); return; }
      // Plain Lambert on the normal-mapped normal: a sprite is lit by which way
      // each of its texels faces, so turning the camera round to the sun's far
      // side shows its dark side.
      const lit = amb.add(sunColourUniform.mul(sunDirect(visible, nS, 'lambert')));
      out.assign(plusGI(pointLight ? lit.add(pointLight(p, n, skip, nS)) : lit));
    });
    return out;
  })();

  return { node, spriteLight, sun, cascades, lights,
           maxDistUniform: maxDist, ambientUniform: ambient };
}

// --- Dynamic point lights, marched through the same field ---
//
// The reason a moving light is affordable at all, and it is worth being explicit
// because it is the opposite of how a shadow map behaves: the distance field is
// GEOMETRY, not light. Moving a torch invalidates nothing. There is no re-bake,
// no re-origin, no second cascade set, no shadow map to re-render from a new
// point of view - the light moves, and the next frame marches the same field it
// already had toward a different point. That is the whole cost.
//
// The march is capped at the distance to the light rather than at a fixed range,
// which is both correct and the cheapest possible cap: anything past the light
// cannot shadow it. Combined with the windowed falloff reaching zero exactly at
// the radius, a level-15 torch is a 22.5 m ray at worst and a level-0 torch is
// no ray at all.

// --- The light list, bound ---
//
// LIGHT_VEC4S vec4 per light, laid out as lights.js packLight writes them. A
// uniform array for the same reason the cards are one: 16 lights is 768 bytes,
// and a uniform read is the cheaper of the two in a per-fragment loop.
export function createLightBindings(capacity = MAX_LIGHTS) {
  return {
    capacity,
    data: uniformArray(new Array(capacity * LIGHT_VEC4S).fill(0)
                         .map(() => new THREE.Vector4()), 'vec4').setName('bxbLights'),
    count: uniform(int(0)),
    // lights.js cutAmount: the sliver of each light's reach not worth a march.
    cutoff: uniform(float(DEFAULT_LIGHT_CUTOFF))
  };
}

export function writeLightBindings(b, packed, count) {
  const v = b.data.array;
  const n = Math.min(count, b.capacity);
  for (let i = 0; i < n * LIGHT_VEC4S; i++) {
    v[i].set(packed[i * 4], packed[i * 4 + 1], packed[i * 4 + 2], packed[i * 4 + 3]);
  }
  b.count.value = n;
  return b;
}

// Returns Fn([p, n, skip, shadeN]) -> the summed colour of every live light.
// The body is the single torch's, unchanged, run once per row of the list.
export function createPointLightsTSL({ cascades, lights, biasUniform, fadeStartUniform,
                                edgeFadeUniform, cards = null }) {
  // Ranged: the point lights share one card buffer and each walks only the
  // slice that was turned to face it.
  const cardsFn = createCardsTSL(cards, { ranged: true });
  // The SUN'S cone marcher, unchanged. Its cone radius is slope * t, and a
  // point light's cone is also linear in t - it just has a different slope, and
  // one that varies per fragment rather than being a uniform. So there is one
  // soft-shadow marcher in the renderer, not two, and a penumbra that disagreed
  // between the sun and a torch would be a bug in one shared function rather
  // than a difference between two plausible ones.
  const trace = createConeTraceSunTSL(cascades);
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const fadeStart = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const edgeFade = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));

  // n lifts the ray off the surface; shadeN is what N.L is taken against - the
  // normal-mapped normal. Separate because the lift has to follow the real
  // geometry, and for a sprite that is not the direction it is shaded as facing.
  //
  // brdf: null, or a function of the light direction returning the specular
  // response per unit of light (see specularTSL) - then the same loop, with the
  // same visibility, also sums each light's highlight.
  const body = (p, n, skip, shadeN, brdf = null) => {
    const sum = vec3(0, 0, 0).toVar();
    const spec = brdf ? vec3(0, 0, 0).toVar() : null;
    // Lifted off the face along the NORMAL, the same as the sun's origin and for
    // the same reason - see SURFACE_BIAS_VOXELS. The same for every light.
    const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();

    // Named: the marcher and the card loop inside both use the default `i`, and
    // an inlined inner loop's `i` would shadow this one's.
    Loop({ start: int(0), end: lights.count, type: 'int', condition: '<', name: 'li' },
         ({ li }) => {
      const base = li.mul(int(LIGHT_VEC4S));
      // Read into variables up front, for the same shadowing reason: an
      // element() left as an expression is re-emitted at each use, inside the
      // inner loops.
      const A = lights.data.element(base).toVar();
      const B = lights.data.element(base.add(int(1))).toVar();
      const C = lights.data.element(base.add(int(2))).toVar();
      const level = A.w;

      const toLight = A.xyz.sub(p).toVar();
      // max() rather than a branch: a fragment exactly at the light would divide
      // by zero, and the answer there is arbitrary anyway.
      const dist = max(length(toLight), float(1e-4)).toVar();
      const dir = toLight.div(dist).toVar();

      // lights.js: the level IS the radius in blocks, and the falloff reaches
      // zero at exactly that radius, so the light ends where the level says and
      // the boundary has no ring.
      const radius = level.mul(float(BLOCK_METRES)).toVar();
      const u = clamp(float(1).sub(dist.div(max(radius, float(1e-4)))),
                      float(0), float(1));
      // log2(1 + u), mirroring lights.js lightFalloff - exact at both ends and
      // far brighter than a square across the middle, which is what a torch
      // looks like.
      const falloff = log2(float(1).add(u)).mul(level.div(float(MAX_LIGHT_LEVEL))).toVar();
      const ndotl = max(shadeN.dot(dir), float(0));

      // lights.js coneSlope(): the cone from this fragment to the emitter opens
      // to sourceRadius at the light, so its slope is sourceRadius / D. Unlike
      // the sun's, this is per fragment - which is the whole difference between
      // a light 150 million km away and one in the player's hand.
      const slope = B.w.div(dist).toVar();
      // Capped at the nearer of the light and its own radius: nothing past
      // either can take away light that is already zero there.
      const reach = min(dist, radius).toVar();

      // Only march where the light can arrive at all. Outside the radius, or on
      // a face turned away, the result is multiplied by zero - and without this
      // gate every such fragment paid a march of up to the full radius, per
      // light. This is also the per-fragment cull §7.1 describes: a light that
      // does not reach here costs its distance test and nothing more.
      //
      // The cutoff remap (lights.js cutAmount) moves the gate inward, so the
      // faint outer band of the light is not marched either, and stays
      // continuous at the new edge rather than leaving a ring.
      const cut = lights.cutoff;
      const amount = max(falloff.mul(ndotl).sub(cut), float(0))
                       .div(float(1).sub(cut)).toVar();
      If(amount.greaterThan(float(0)), () => {
        // C.z is the shadow weight: 0 for a light outside the shadow budget,
        // which lights the surface unshadowed and marches nothing; between 0
        // and 1 while it fades in or out of the budget.
        const weight = C.z;
        const visible = float(1).toVar();
        If(weight.greaterThan(float(0)), () => {
          const v = trace(origin, dir, reach, slope, fadeStart, edgeFade).x.toVar();
          // Sprites are not in the field, so they are tested here - against the
          // cards turned to face THIS light. Same ray, same cap, same cone slope,
          // so a sprite's penumbra matches a wall's.
          // C.w is the card weight: 0 for a light outside the card budget,
          // which tests an empty slice. Faded like the shadow weight.
          if (cardsFn) {
            const start = int(C.x).toVar();
            const cardVis = cardsFn(origin, dir, reach, slope, skip,
                                    start, start.add(int(C.y)));
            v.mulAssign(mix(float(1), cardVis, C.w));
          }
          visible.assign(mix(float(1), v, weight));
        });
        sum.addAssign(B.xyz.mul(amount).mul(visible));
        // The highlight takes the light's falloff, not the cut-remapped
        // diffuse amount - brdf carries its own N.L.
        if (brdf) spec.addAssign(B.xyz.mul(falloff).mul(visible).mul(brdf(dir)));
      });
    });

    return brdf ? { diffuse: sum, spec } : sum;
  };

  // A plain function around the Fn, not the Fn itself: a TSL Fn is a proxy,
  // and properties hung on it read back as node members, not as these.
  const fn = Fn(([p, n, skip, shadeN]) => body(p, n, skip, shadeN))
    .setLayout(layout('pointLights', 'vec3', [
      { name: 'p', type: 'vec3' }, { name: 'n', type: 'vec3' },
      { name: 'skip', type: 'vec2' }, { name: 'shadeN', type: 'vec3' }]));
  const call = (p, n, skip, shadeN) => fn(p, n, skip, shadeN);
  // Called inline from inside another Fn, so their vars land in the caller.
  call.withSpecular = (p, n, skip, shadeN, brdf) => body(p, n, skip, shadeN, brdf);
  return call;
}

// A level's footprint travels with the player, or shading only works where it
// happened to be built. Repopulating in place and rewriting the level's uniforms
// is enough - the texture object and the material are unchanged, so nothing
// rebuilds. The grid remembers its own level, so this works for any cascade.
export function followShadowGrid(grid, tex, bindings, originBlock, renderer = null) {
  createBoxGridAt(originBlock.x, originBlock.z, grid, grid.level);
  return commitShadowGrid(grid, tex, bindings, renderer);
}

// Push a grid's new state to the GPU: bindings, then whatever changed. Used
// both after a main-thread re-origin and after a worker hand-off.
export function commitShadowGrid(grid, tex, bindings, renderer = null) {
  writeCascadeBindings(bindings, grid);
  // A slide changed only the strips that scrolled in, so only those go up.
  // Anything else - a full rebake, or no renderer to copy with - is the old
  // whole-texture upload.
  if (grid.uploadRects && renderer) uploadStrips(renderer, tex, grid, grid.uploadRects);
  else tex.needsUpdate = true;
  grid.uploadRects = [];
  return grid;
}

// Upload a logical rect list straight into the live texture, one
// queue.writeTexture per physically contiguous box - a few hundred KB instead of
// 3 MB, with no staging texture, no copy and no extra submit. The staging-copy
// version of this was cheaper to upload but cost a ~50 ms frame after every step.
let stripBuf = new Uint8Array(0);
function uploadStrips(renderer, tex, grid, rects) {
  renderer.initTexture(tex);
  const gpuTex = renderer.backend.get(tex).texture;
  if (!gpuTex) { tex.needsUpdate = true; return; }
  const queue = renderer.backend.device.queue;
  for (const r of rects) {
    const h = r.y1 - r.y0;
    for (const [xa, xb] of ringSegments(grid, r.x0, r.x1 - 1)) {
      for (const [za, zb] of ringSegmentsZ(grid, r.z0, r.z1 - 1)) {
        const w = xb - xa + 1, d = zb - za + 1;
        if (stripBuf.length < w * h * d) stripBuf = new Uint8Array(w * h * d);
        let o = 0;
        for (let vz = za; vz <= zb; vz++) {
          for (let vy = r.y0; vy < r.y1; vy++, o += w) {
            const base = gridIndex(grid, xa, vy, vz);
            stripBuf.set(grid.data.subarray(base, base + w), o);
          }
        }
        const phys = gridIndex(grid, xa, 0, za);
        queue.writeTexture(
          { texture: gpuTex,
            origin: { x: phys % GRID_DIM, y: r.y0, z: Math.floor(phys / (GRID_DIM * GRID_DIM)) } },
          stripBuf, { offset: 0, bytesPerRow: w, rowsPerImage: h },
          { width: w, height: h, depthOrArrayLayers: d });
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 8. Texel-rate shading: the terrain shaded once per texel, not once per pixel
//
// Everything the terrain shades is texel-locked - one texel, one answer - so
// every pixel inside a texel runs the identical marches and gets the identical
// colour. At play zoom a 12.5 cm texel covers many pixels; profiling put the
// torch, the mirrors and the reflection at ~3 ms of the frame, almost all of
// it repeats.
//
// So the terrain is drawn twice:
//
//   1  a LOW-RES pass (1/scale per axis) with the full shading, writing
//      vec4(colour, texel id) to a float target. Each low-res pixel is one
//      sample of whatever texel lies under it.
//   2  the normal full-res pass, which for each pixel computes only its own
//      texel id (cheap: the lock and a hash), and takes the colour of the
//      first sample in the 3x3 around it with the same id. Only a pixel with
//      no matching sample - a texel smaller than the sample spacing, at a far
//      zoom or a grazing angle - shades itself, in full.
//
// Nothing is approximated: a match means the sample's texel IS this pixel's
// texel, and a texel's colour is the same everywhere in it, bit for bit. No
// atlas, no paging, no invalidation - the cache is rebuilt every frame, so it
// is always the current frame's answer.
//
// The id is the world texel coordinate, each axis mod 128, plus the face (6):
// 7 + 7 + 7 + 3 bits, exact as a float. It only has to tell apart the handful
// of texels in a 3x3 window of samples, which are never 16 m apart.
// Offset by 2 so no id is 0 (cleared) or 1 (the sky's alpha).
export const TEXEL_ID_OFFSET = 2;
export const texelIdTSL = (p, n) => {
  // Half a texel inside the face, so the along-normal floor is not taken on
  // the plane itself.
  const t = p.sub(n.mul(float(VOXEL_METRES * 0.5))).div(float(VOXEL_METRES)).floor();
  const wrap = v => v.sub(floor(v.div(float(128))).mul(float(128)));
  const face = select(abs(n.x).greaterThan(float(0.5)), select(n.x.greaterThan(float(0)), float(0), float(1)),
               select(abs(n.y).greaterThan(float(0.5)), select(n.y.greaterThan(float(0)), float(2), float(3)),
                      select(n.z.greaterThan(float(0)), float(4), float(5))));
  return wrap(t.x).add(wrap(t.y).mul(float(128))).add(wrap(t.z).mul(float(16384)))
    .add(face.mul(float(2097152))).add(float(TEXEL_ID_OFFSET));
};
// This terrain fragment's texel id - the same lock the shading takes.
export const terrainTexelIdTSL = () => {
  const n = axisNormalTSL(normalWorld).toVar();
  const p = texelLockTSL(positionWorld, n);
  return texelIdTSL(p, n);
};

// The low-res pass's output: the full shading and the texel it belongs to.
export function createTexelCacheWriteNode(shade) {
  return vec4(shade, terrainTexelIdTSL());
}

// The full-res lookup. cache: { tex, scale (vec2 low/full), size (vec2 low) }.
//
// A pixel with no matching sample is DISCARDED here, not shaded: the full
// shading lives in a separate miss pass (createTexelMissNode), drawn after,
// where a stencil test keeps it off every pixel this pass filled. It cannot
// live in a branch of this shader - measured, a never-taken branch holding the
// full shading cost 2.5 ms of 3.5, because a GPU sizes a shader's registers for
// its worst path, and every pixel then runs with that fewer in flight.
const TEXEL_CACHE_TAPS = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
                          [1, 1], [-1, 1], [1, -1], [-1, -1]];
export function createTexelCacheLookupNode(cache) {
  return Fn(() => {
    const id = terrainTexelIdTSL().toVar();
    const lc = floor(screenCoordinate.xy.mul(cache.scale)).toVar();
    const hi = cache.size.sub(float(1));
    const out = vec3(0, 0, 0).toVar();
    const found = float(0).toVar();
    // Unrolled on purpose: nine loads, the centre first, each skipped once a
    // match is in hand.
    for (const [dx, dy] of TEXEL_CACHE_TAPS) {
      If(found.equal(float(0)), () => {
        const c = clamp(lc.add(vec2(dx, dy)), vec2(0, 0), hi);
        const s = textureLoad(cache.tex, ivec2(c)).toVar();
        If(s.w.equal(id), () => { out.assign(s.xyz); found.assign(float(1)); });
      });
    }
    If(found.equal(float(0)), () => { Discard(); });
    return out;
  })();
}

// The miss pass's colour: the full shading - or, in the miss view, red, so
// which pixels shade themselves can be seen.
export function createTexelMissNode(cache, shade) {
  return cache.showMisses ? vec3(1, 0, 0) : shade;
}

// Swaps the terrain to a node material whose colour is its own albedo modulated
// by the marched shadow term, keeping the original so it can be restored.
export function applyShadowMaterial(mesh, shadowNode, albedo = null) {
  const mat = createShadowMaterial(mesh, shadowNode, albedo);
  if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
  mesh.material = mat;
  return mat;
}

// The same material, built but not assigned - so it can be compiled off-screen
// while the current one keeps drawing.
// albedo: the surface colour node - for the terrain, its per-block layer with
// whiteworld mixed in (render.js). White when null.
export function createShadowMaterial(mesh, shadowNode, albedo = null) {
  if (!mesh.userData.originalMaterial) mesh.userData.originalMaterial = mesh.material;
  const mat = new THREE.MeshBasicNodeMaterial();
  // .rgb, because the shadow node is a vec3 now that a coloured light adds to it.
  // Per-instance tint (arena greying) lives in instanceColor, so it has to be
  // carried across or battle mode loses its reachable-tile shading.
  mat.colorNode = (albedo || vec3(1, 1, 1)).mul(shadowNode);
  return mat;
}

export function restoreOriginalMaterial(mesh) {
  if (!mesh.userData.originalMaterial) return false;
  if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
  mesh.material = mesh.userData.originalMaterial;
  return true;
}

// ---------------------------------------------------------------------------
// 6. The sun: a marched cone
//
// This replaced a cascaded shadow map, and the replacement is smaller than the
// thing it replaced by about seven hundred lines. See sun.js for the argument;
// the short version is that a depth map needs a bias, a bias needs a
// receiver-plane gradient to be stable under a moving cascade, and a softening
// kernel that widens with distance turns any leftover bias error into a large
// dark region rather than a small speckle. The marched path has none of those
// error sources, because it never compares two depths - it asks the distance
// field whether anything is in the way, which is the same question every other
// light in the renderer asks.
//
// The sun is the ONE light with no termination distance, which is the property
// the CSM existed for and the one thing this does not solve: a ray leaving the
// grid reads as open sky, so a long shadow truncates at the footprint edge. That
// is a RANGE problem, addressed by how far the field reaches, not by how the
// shadow is filtered - so it is deliberately not mixed into this.
// ---------------------------------------------------------------------------

// The dither source. Four by four, nearest, repeating: indexed by integer world
// texel, so the pattern is painted onto surfaces rather than crawling across
// them under camera motion.
export function createBayerTexture() {
  const tex = new THREE.DataTexture(
    new Uint8Array(BAYER4.map(v => v * 16)), 4, 4, THREE.RedFormat);
  tex.minFilter = THREE.NearestFilter;
  tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

// How far off the surface a shadow ray starts, in voxels, along the FACE NORMAL.
//
// Along the normal and not along the ray, which is what the first version did
// and what limited it to steep sun angles. A fragment sits exactly on a voxel
// face, where the field reads zero and floating point can place it either side;
// stepping along the ray escapes that only in proportion to N.L, so a low sun
// grazing a wall barely moves and the surface shadows itself in stripes.
// Stepping along the normal clears the face by the same amount at every sun
// angle. Atlas shading is what makes it available: the job already carries the
// exact face normal, where a screen-space pass reconstructs it from derivatives.
//
// 0.75 of a voxel puts the origin where the field reads about +0.75, comfortably
// clear of HIT_EPS without detaching contact shadows - at 12.5 cm voxels this is
// 9 cm, well under one texel of the art.
export const SURFACE_BIAS_VOXELS = 0.75;

// Where the distance fade begins, as a FRACTION of the march cap. 0.97 spends
// only the last 3% - about 36 cm of a 12 m ray - dissolving the end of a shadow.
//
// That is far narrower than the 0.6 this started at, and the reason is worth
// recording: the discontinuity being hidden is a step of about 0.31 in
// visibility, not a full black-to-white edge, so it only needs a few texels to
// stop reading as a line. A wide fade removes it just as well but costs metres of
// real umbra at the far end of every long shadow, which is a worse trade. Found
// by eye, then confirmed against the neighbouring-texel jump in the tests.
export const SHADOW_FADE_START = 0.97;

// How close to the grid boundary a sample has to be before its occlusion starts
// fading, in voxels. 12 is one block - enough to dissolve an edge, small enough
// that it costs almost no shadow inside the footprint.
export const EDGE_FADE_VOXELS = 12;

// The soft sun. Returns visibility: 1 fully lit, 0 fully occluded, and the values
// between are the fraction of the solar disc this point can see.
//
// Signature is (position, normal) -> float, which is exactly what the atlas shade
// kernel already calls, so the sun technique is interchangeable at that seam -
// the whole point of shading in texel space.
//
// rayCount is a COMPILE-TIME argument, not a uniform: the loop is unrolled on the
// host so the cone offsets are literals and the kernel has no dynamic loop bound.
// Changing it is a kernel rebuild, which is why main.js tracks it alongside the
// other pass-shape flags.
// --- One shading formula, used by both paths ---
//
// This existed twice and the two copies disagreed. The atlas applied Lambert and
// the per-pixel path did not, so at a 20 degree sun flat ground came out at
// 0.35 + 1*sin(20)*0.65 = 0.572 under bxb.atlas() and 1.0 under bxb.shadows() -
// the same geometry, the same sun, 43% darker on one of them, reading as though
// the whole world were in partial shade. Unshaded atlas pages are cleared to 1.0,
// which is why the pages that had not been written yet looked RIGHT and the
// shaded ones looked wrong.
//
// So there is now one function and both call it. Two copies of a formula is a
// disagreement waiting to happen; two copies of a formula where one is a
// simplification of the other is a disagreement that has already happened.
//
// THE N.L PROBLEM, which is an art question rather than a bug. Lambert is
// physically right and it is also why the sun's elevation was controlling the
// whole scene's brightness: at 20 degrees every up-facing surface - which in a
// top-down game is nearly all of them - is multiplied by 0.34. Correct, and too
// dark, with no second light and no tonemapping to recover it.
//
//   'flat'       no N.L at all. What bxb.shadows always did. Bright, but an
//                unshadowed wall facing away from the sun reads exactly as bright
//                as one facing it, which is what made the first atlas look flat.
//   'lambert'    raw N.L. Physically right, dim at a low sun.
//   'ground'     N.L normalised against what a horizontal surface receives, so
//                flat ground under open sky is always fully lit and the sun's
//                ANGLE changes shadow direction and length rather than overall
//                brightness. Faces still differentiate: one turned away still
//                falls to ambient. This is the default.
//
// The reference is floored so that a sun on the horizon does not divide every
// surface up to full brightness and collapse the differentiation entirely.
export const GROUND_REFERENCE_FLOOR = 0.25;
export const DEFAULT_AMBIENT = 0.17;

// ao scales the AMBIENT term only - it is occlusion of light arriving from all
// around, and says nothing about the one direction the sun comes from.
export function sunShadeTSL({ visible, n, sunDir, ambientUniform, mode = 'ground', ao = null }) {
  const amb = ao ? ambientUniform.mul(ao) : ambientUniform;
  if (mode === 'flat') return amb.add(visible.mul(float(1).sub(ambientUniform)));

  const ndotl = max(n.dot(sunDir), float(0)).toVar();
  if (mode === 'ground') {
    // sunDir.y IS the N.L a horizontal up-facing surface would receive.
    const ref = max(sunDir.y, float(GROUND_REFERENCE_FLOOR));
    ndotl.assign(clamp(ndotl.div(ref), float(0), float(1)));
  }
  return amb.add(visible.mul(ndotl).mul(float(1).sub(ambientUniform)));
}

export function createSoftSunTSL({ cascades, sunDir, sunTangent,
                                  sunBitangent, maxDist, coneRadiusUniform,
                                  rayCount = 1, bayerTex = null,
                                  stepsUniform = null, biasUniform = null,
                                  cone = false, quantise = true,
                                  fadeStartUniform = null, edgeFadeUniform = null,
                                  cards = null }) {
  const offsets = coneOffsets(rayCount);
  const cardsFn = createCardsTSL(cards);
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const fadeStart = fadeStartUniform || uniform(float(SHADOW_FADE_START));
  const edgeFade = edgeFadeUniform || uniform(float(EDGE_FADE_VOXELS));
  // Built once per pass, not per invocation: the cascade list is host-side, so
  // the level walk is unrolled into the kernel.
  const coneTrace = createConeTraceSunTSL(cascades);
  const shadowTrace = createShadowTraceTSL(cascades);

  // The cone trace gets the whole penumbra from a single ray, so the sample
  // pattern, its per-texel rotation and the sample count are all irrelevant to
  // it - it is a different way of asking the question, not a cheaper sampling of
  // the same one. Kept as an early return rather than threaded through the loop
  // below so that neither path carries the other's machinery.
  if (cone) {
    return Fn(([p, n, skip]) => {
      // toVar, not a bare expression. TSL hoists a node into a variable at the
      // point it is first REUSED, and here that is inside the C1 loop - which
      // only runs when C0 did not conclude. Everywhere the march ended in C0,
      // the card test below then read an origin that was never assigned, (0,0,0),
      // and every sprite shadow vanished from that region.
      const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();
      const visible = coneTrace(origin, sunDir, maxDist, coneRadiusUniform,
                                fadeStart, edgeFade).x.toVar();
      // Sprites left the distance field so that each light could have them
      // facing IT - see createCardsTSL. Same ray, same cap, same cone radius, so
      // a character's penumbra and a wall's come out of one formula.
      if (cardsFn) {
        visible.mulAssign(cardsFn(origin, sunDir, float(CARD_SUN_REACH),
                                  coneRadiusUniform, skip));
      }
      // The cone trace is ANALYTIC: d/(R*t) is a continuous function of the
      // geometry, so its penumbra is already smooth and there is nothing here a
      // dither is fixing. Quantising it throws that away and replaces it with
      // banding, then breaks the banding up with a crosshatch. Off by default for
      // this path - it is a style choice, not a correctness one.
      if (!quantise || !stepsUniform || !bayerTex) {
        return visible;
      }
      const wt = p.div(float(VOXEL_METRES)).floor();
      const dither = texture(bayerTex, wt.xz.add(vec2(0, wt.y)).mul(float(0.25))).r;
      return clamp(visible.mul(stepsUniform).add(dither).floor().div(stepsUniform),
                   float(0), float(1));
    }).setLayout(layout('sunCone', 'float', [
      { name: 'p', type: 'vec3' }, { name: 'n', type: 'vec3' }, { name: 'skip', type: 'vec2' }]));
  }

  return Fn(([p, n, skip]) => {
    // Lift off the face once, not per ray: the origin depends on the surface,
    // not on which part of the sun is being sampled.
    const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES)))).toVar();

    // Rotate the whole sample pattern by a per-texel angle. Without this the
    // fixed offsets make the penumbra a set of rayCount+1 discrete bands that
    // are identical on every surface, which reads as contouring rather than as
    // softness. Anchored to the world texel for the same reason the quantise
    // dither is - a pattern that moves with the camera crawls.
    const rot = float(0).toVar();
    if (bayerTex) {
      const wt = p.div(float(VOXEL_METRES)).floor();
      rot.assign(texture(bayerTex, wt.xz.add(vec2(0, wt.y)).mul(float(0.25))).r
                   .mul(float(Math.PI * 2)));
    }
    const cs = cos(rot).toVar();
    const sn = sin(rot).toVar();

    const lit = float(0).toVar();
    for (let i = 0; i < offsets.length; i++) {
      const [u, v] = offsets[i];
      // Unrolled, so u and v are literals and the rotation is two multiply-adds.
      const du = cs.mul(float(u)).sub(sn.mul(float(v))).mul(coneRadiusUniform);
      const dv = sn.mul(float(u)).add(cs.mul(float(v))).mul(coneRadiusUniform);
      const dir = normalize(sunDir.add(sunTangent.mul(du)).add(sunBitangent.mul(dv)));
      // Returns weighted occlusion, so visibility is its complement.
      lit.addAssign(float(1).sub(shadowTrace(origin, dir, maxDist, fadeStart, edgeFade).x));
    }
    const visible = lit.div(float(offsets.length)).toVar();
    // The sampled path takes the cards once rather than per disc sample: the
    // card test is already analytic, so sampling it N times would return the
    // same answer N times and pay for it each one.
    if (cardsFn) {
      visible.mulAssign(cardsFn(origin, sunDir, float(CARD_SUN_REACH),
                                coneRadiusUniform, skip));
    }

    // Quantise the soft tail into discrete steps rather than leaving a smooth
    // gradient: under flat-shaded pixel art a gradient reads as a rendering bug,
    // where stepped and dithered reads as intentional (doc 6.1). A single ray is
    // already two values, so there is no tail to quantise and the dither would
    // only punch holes in a binary edge.
    if (!quantise || !stepsUniform || !bayerTex || offsets.length === 1) {
      return visible;
    }

    const wt = p.div(float(VOXEL_METRES)).floor();
    const dither = texture(bayerTex, wt.xz.add(vec2(0, wt.y)).mul(float(0.25))).r;
    return clamp(visible.mul(stepsUniform).add(dither).floor().div(stepsUniform),
                 float(0), float(1));
  }).setLayout(layout('sunSampled', 'float', [
      { name: 'p', type: 'vec3' }, { name: 'n', type: 'vec3' }, { name: 'skip', type: 'vec2' }]));
}

// The uniforms the cone needs, as one object, because the sun direction and its
// two basis vectors must be written TOGETHER or the cone stops being centred on
// the light - a basis left over from the previous sun tilts the penumbra off to
// one side, which looks like a softening bug rather than a stale uniform.
export function createSunUniforms(sunDirection, angular = SUN_ANGULAR_SIZE) {
  const b = sunBasis(sunDirection);
  return {
    dir: uniform(new THREE.Vector3(b.forward.x, b.forward.y, b.forward.z)),
    tangent: uniform(new THREE.Vector3(b.tangent.x, b.tangent.y, b.tangent.z)),
    bitangent: uniform(new THREE.Vector3(b.bitangent.x, b.bitangent.y, b.bitangent.z)),
    coneRadius: uniform(float(coneRadius(angular))),
    angular
  };
}

export function writeSunUniforms(u, sunDirection, angular = u.angular) {
  const b = sunBasis(sunDirection);
  u.dir.value.set(b.forward.x, b.forward.y, b.forward.z);
  u.tangent.value.set(b.tangent.x, b.tangent.y, b.tangent.z);
  u.bitangent.value.set(b.bitangent.x, b.bitangent.y, b.bitangent.z);
  u.coneRadius.value = coneRadius(angular);
  u.angular = angular;
  return u;
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
