
import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, float, vec3, uniform, instanceIndex, storage, texture3D,
  positionWorld, normalWorld, floor, max, texture, uv, abs,
  attribute, varying, vec2, vec4, ivec2, textureStore, min,
  normalize, cross, clamp, cos, sin, smoothstep, length, log2,
  uniformArray, int, dot, select, dFdx, dFdy, textureSize, mix,
  modelWorldMatrix, materialColor, inverseSqrt, frontFacing, sign, sqrt, fract, exp2
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
  const target = storage(buffer, 'float', count);

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
  return Fn(([origin0, dir0, maxDist, coneR, fadeStart, edgeFade]) => {
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
  });
}

// The sampled path's trace. Identical in structure, but returns how much this ray
// should darken - faded at the march's limits - rather than a binary hit.
//
// Separate from traceDistanceTSL because that one is what bxb.parity() diffs
// against the CPU sphere trace, and a hit test that sometimes returns 0.6 is not
// a hit test any more.
export function createShadowTraceTSL(cascades) {
  return Fn(([origin0, dir0, maxDist, fadeStart, edgeFade]) => {
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
  });
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

  const origins = storage(originBuf, 'vec4', n);
  const dirs = storage(dirBuf, 'vec4', n);
  const out = storage(outBuf, 'float', n);
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

export function createCardBindings(capacity = MAX_CARDS) {
  return {
    capacity,
    // Four vec4 per card, flat. A uniform array rather than a storage buffer:
    // 64 cards is 4 KB, far inside the uniform limit, and a uniform read is the
    // cheaper of the two in a per-fragment loop.
    data: uniformArray(new Array(capacity * 4).fill(0).map(() => new THREE.Vector4()),
                       'vec4'),
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

    return vis;
  };
  return ranged
    ? Fn(([from, dir, maxDist, coneSlope, skip, start, end]) =>
        body(from, dir, maxDist, coneSlope, skip, start, end))
    : Fn(([from, dir, maxDist, coneSlope, skip]) =>
        body(from, dir, maxDist, coneSlope, skip, int(0), b.count));
}

export const texelLockTSL = Fn(([p, n]) => {
  const q = float(VOXEL_METRES);
  const a = abs(n);
  const snapped = p.div(q).floor().add(float(0.5)).mul(q);
  return snapped.mul(vec3(1, 1, 1).sub(a)).add(p.mul(a));
});

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
  return Fn(([origin, dir, maxDist, coneR]) => {
    const dim = float(GRID_DIM);
    // A voxel out, as the sun's cone: at t = 0 the cone has no width.
    const t = float(VOXEL_METRES).toVar();
    const occ = float(0).toVar();
    const done = float(0).toVar();
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
          If(d.lessThan(float(HIT_EPS)), () => { done.assign(float(1)); Break(); });
          t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }
    return occ;
  });
}

// Returns Fn([p, n, lift, skip]) -> visibility in [0, 1]: n is the hemisphere,
// lift the direction the origin is raised along (the geometry's, as for
// shadows), skip the sprite's own card.
export function createAOTSL({ cascades, distance, biasUniform = null }) {
  const cone = createAOConeTSL(cascades);
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const R = float(AO_CONE_R);
  return Fn(([p, n, lift, skip]) => {
    const origin = p.add(lift.mul(bias.mul(float(VOXEL_METRES)))).toVar();
    const helper = select(abs(n.y).lessThan(float(0.999)), vec3(0, 1, 0), vec3(1, 0, 0));
    const T = normalize(cross(helper, n)).toVar();
    const B = cross(n, T).toVar();
    const vis = float(1).toVar();
    for (let i = 0; i < AO_CONES; i++) {
      const up = i === 0;
      const phi = (i - 1) * (2 * Math.PI / 5);
      const st = up ? 0 : Math.sin(AO_TILT), ct = up ? 1 : Math.cos(AO_TILT);
      const dir = normalize(T.mul(float(st * Math.cos(phi))).add(B.mul(float(st * Math.sin(phi))))
                            .add(n.mul(float(ct)))).toVar();
      vis.subAssign(cone(origin, dir, distance, R).mul(float(up ? AO_WEIGHT_UP : AO_WEIGHT_SIDE)));
    }
    return clamp(vis, float(0), float(1));
  });
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
function spriteShadeNormalTSL(map, normalTex) {
  const right = normalize(modelWorldMatrix.mul(vec4(1, 0, 0, 0)).xyz);
  const up = normalize(modelWorldMatrix.mul(vec4(0, 1, 0, 0)).xyz);
  const fwd = normalize(modelWorldMatrix.mul(vec4(0, 0, 1, 0)).xyz);
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

// Normal maps are data, not colour: no sRGB, nearest so each art texel has one
// normal, repeat because block textures tile.
export function createNormalTexture(data, w, h) {
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
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
                                  terrainNormal = null,
                                  ao: aoOn = true, aoDistanceUniform = null,
                                  aoOnly = false }) {
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
  const cones = aoOn ? createAOTSL({ cascades, distance: aoDistance, biasUniform }) : null;
  const spriteAO = aoOn ? createSpriteAOTSL(cards) : null;
  // Terrain cones times the sprite discs.
  const rtao = cones ? (p, n, lift, skip) =>
    spriteAO ? cones(p, n, lift, skip).mul(spriteAO(p, skip)) : cones(p, n, lift, skip) : null;

  // Every point light, in one loop over the list. Compiled in whenever there is
  // a list at all: an empty list is a loop of zero iterations, so turning lights
  // on and off is a uniform write rather than a rebuild.
  const pointLight = lights
    ? createPointLightsTSL({ cascades, lights, biasUniform,
                             fadeStartUniform, edgeFadeUniform, cards: lightCards })
    : null;

  const node = Fn(() => {
    // normalWorld rather than a derivative reconstruction: the terrain is boxes,
    // so the interpolated normal IS the exact face normal, and the marcher needs
    // it to lift the ray origin off the face along the surface rather than along
    // the ray - see SURFACE_BIAS_VOXELS for why that distinction matters at low
    // sun angles.
    const n = normalize(normalWorld).toVar();
    // Shading normal: the face normal bent by the block texture's normal map.
    // The geometric n still lifts rays and decides which faces the sun can
    // reach at all; only N.L sees the bumps.
    const lab = terrainNormal ? labNormalTSL(texture(terrainNormal, uv())).toVar() : null;
    const nS = lab ? bumpedNormalTSL(n, positionWorld, uv(), lab.xyz).toVar() : n;
    const texAO = lab ? lab.w : float(1);
    // Locked once, used by every light. Two lights that disagreed about where a
    // surface IS would put their shadows on different texel grids, and the
    // mismatch would read as the torch's shadow crawling against the sun's.
    const p = texelLockTSL(positionWorld, n).toVar();
    // Traced AO times the texture's own. The hemisphere is the geometric face,
    // not the bumped normal - the rays test real geometry.
    const ao = (rtao ? rtao(p, n, n, NO_SKIP).mul(texAO) : texAO).toVar();
    if (aoOnly) return vec3(ao, ao, ao);
    // Sun off: no march, no shading term, just the ambient floor for anything a
    // dynamic light does not reach.
    if (!sunOn) {
      const dark = vec3(ambient, ambient, ambient).mul(ao);
      return pointLight ? dark.add(pointLight(p, n, NO_SKIP, nS)) : dark;
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
    const sunTerm = sunShadeTSL({ visible, n: nS, sunDir: sun.dir,
                                  ambientUniform: ambient, mode: shadeMode, ao });
    if (!pointLight) return vec3(sunTerm, sunTerm, sunTerm);
    return vec3(sunTerm, sunTerm, sunTerm).add(pointLight(p, n, NO_SKIP, nS));
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
      const ao = (rtao ? rtao(p, nS, n, skip).mul(texAO) : texAO).toVar();
      if (aoOnly) { out.assign(vec3(ao, ao, ao)); return; }
      if (!sunOn) {
        const dark = vec3(ambient, ambient, ambient).mul(ao);
        out.assign(pointLight ? dark.add(pointLight(p, n, skip, nS)) : dark);
        return;
      }
      const visible = sunShadow(p, n, skip);
      if (shadowOnly) { out.assign(vec3(visible, visible, visible)); return; }
      // Plain Lambert on the normal-mapped normal: a sprite is lit by which way
      // each of its texels faces, so turning the camera round to the sun's far
      // side shows its dark side.
      const sunTerm = sunShadeTSL({ visible, n: nS, sunDir: sun.dir,
                                    ambientUniform: ambient, mode: 'lambert', ao });
      const lit = vec3(sunTerm, sunTerm, sunTerm);
      out.assign(pointLight ? lit.add(pointLight(p, n, skip, nS)) : lit);
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
                         .map(() => new THREE.Vector4()), 'vec4'),
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
function createPointLightsTSL({ cascades, lights, biasUniform, fadeStartUniform,
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
  return Fn(([p, n, skip, shadeN]) => {
    const sum = vec3(0, 0, 0).toVar();
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
      const falloff = log2(float(1).add(u)).mul(level.div(float(MAX_LIGHT_LEVEL)));
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
          if (cardsFn) {
            const start = int(C.x).toVar();
            v.mulAssign(cardsFn(origin, dir, reach, slope, skip,
                                start, start.add(int(C.y))));
          }
          visible.assign(mix(float(1), v, weight));
        });
        sum.addAssign(B.xyz.mul(amount).mul(visible));
      });
    });

    return sum;
  });
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

// Swaps the terrain to a node material whose colour is its own albedo modulated
// by the marched shadow term, keeping the original so it can be restored.
export function applyShadowMaterial(mesh, shadowNode) {
  const mat = createShadowMaterial(mesh, shadowNode);
  if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
  mesh.material = mat;
  return mat;
}

// The same material, built but not assigned - so it can be compiled off-screen
// while the current one keeps drawing.
// white: a float uniform, 1 for whiteworld - a uniform so toggling it is a
// write, not a shader rebuild.
export function createShadowMaterial(mesh, shadowNode, white = null) {
  if (!mesh.userData.originalMaterial) mesh.userData.originalMaterial = mesh.material;
  const base = mesh.userData.originalMaterial;

  const mat = new THREE.MeshBasicNodeMaterial();
  // .rgb, because the shadow node is a vec3 now that a coloured light adds to it.
  // Whiteworld clears base.map for the unshadowed path, and parks the texture in
  // userData.albedoMap - so the texture is read from there when it exists.
  const map = base.userData.albedoMap !== undefined ? base.userData.albedoMap : base.map;
  let albedo = map ? texture(map, uv()).rgb : vec3(1, 1, 1);
  if (white && map) albedo = mix(albedo, vec3(1, 1, 1), white);
  // Per-instance tint (arena greying) lives in instanceColor, so it has to be
  // carried across or battle mode loses its reachable-tile shading.
  mat.colorNode = albedo.mul(shadowNode);
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
export const DEFAULT_AMBIENT = 0.35;

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
    });
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
  });
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

  const froms = storage(fromBuf, 'vec4', n);
  const dirs = storage(dirBuf, 'vec4', n);
  const out = storage(outBuf, 'float', n);
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
