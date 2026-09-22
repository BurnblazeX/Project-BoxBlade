
import * as THREE from 'three/webgpu';
import {
  Fn, If, Loop, Break, float, vec3, uniform, instanceIndex, storage, texture3D,
  positionWorld, normalWorld, floor, max, texture, uv, abs,
  attribute, varying, vec2, vec4, ivec2, textureStore, min,
  normalize, cross, clamp, cos, sin, smoothstep, length, log2
} from 'three/tsl';
import { GRID_DIM, VOXEL_METRES, createBoxGridAt, DISTANCE_RANGE,
         HIT_EPS, MIN_STEP, MAX_TRACE_STEPS } from './boxgrid.js';
import { BAYER4, SUN_ANGULAR_SIZE, coneOffsets, coneRadius,
         sunBasis } from './sun.js';
import { MAX_LIGHT_LEVEL } from './lights.js';
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
  tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
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
    voxel: uniform(float(grid.voxelSize)),
    range: uniform(float(grid.range))
  };
}

export function writeCascadeBindings(b, grid) {
  b.origin.value.set(grid.origin.x, grid.origin.y, grid.origin.z);
  b.voxel.value = grid.voxelSize;
  b.range.value = grid.range;
  return b;
}

export const traceDistanceTSL = Fn(([sdfTex, gridOrigin, origin, dir, maxDist]) => {
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

    const d = decodeFieldTSL(texture3D(sdfTex, vp.div(dim)).r, float(DISTANCE_RANGE));
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

export function createConeTraceSunTSL(cascades) {
  return Fn(([origin, dir, maxDist, coneR, fadeStart, edgeFade]) => {
    const dim = float(GRID_DIM);
    // Started one voxel out rather than at zero: at t = 0 the cone has no radius,
    // so d/(R*t) divides by zero, and the surface the ray is leaving is the
    // nearest thing to it by construction.
    const t = float(VOXEL_METRES).toVar();
    const res = float(1).toVar();
    const done = float(0).toVar();

    for (const { c, edge } of eachCascade(cascades, edgeFade)) {
      If(done.equal(float(0)), () => {
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

          const d = decodeFieldTSL(texture3D(c.tex, vp.div(dim)).r, c.range);
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
}

// The sampled path's trace. Identical in structure, but returns how much this ray
// should darken - faded at the march's limits - rather than a binary hit.
//
// Separate from traceDistanceTSL because that one is what bxb.parity() diffs
// against the CPU sphere trace, and a hit test that sometimes returns 0.6 is not
// a hit test any more.
export function createShadowTraceTSL(cascades) {
  return Fn(([origin, dir, maxDist, fadeStart, edgeFade]) => {
    const dim = float(GRID_DIM);
    const t = float(0).toVar();
    const occ = float(0).toVar();
    const done = float(0).toVar();

    for (const { c, edge } of eachCascade(cascades, edgeFade)) {
      If(done.equal(float(0)), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel);
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => {
            Break();
          });

          const d = decodeFieldTSL(texture3D(c.tex, vp.div(dim)).r, c.range);
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

  const kernel = Fn(() => {
    const o = origins.element(instanceIndex);
    const d = dirs.element(instanceIndex);
    out.element(instanceIndex).assign(
      traceDistanceTSL(occTex, gridOrigin, o.xyz, d.xyz, d.w)
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
export const texelLockTSL = Fn(([p, n]) => {
  const q = float(VOXEL_METRES);
  const a = abs(n);
  const snapped = p.div(q).floor().add(float(0.5)).mul(q);
  return snapped.mul(vec3(1, 1, 1).sub(a)).add(p.mul(a));
});

export function createShadowColorNode({ cascades, sunDirection, maxDistance = 12,
                                        ambientUniform = null, shadeMode = 'ground',
                                        rays = 1, shadowOnly = false, torch = null, sun: sunOn = true,
                                        angular = SUN_ANGULAR_SIZE,
                                        bayerTex = null, stepsUniform = null,
                                        biasUniform = null,
                                        cone = false, quantise = true,
                                  fadeStartUniform = null, edgeFadeUniform = null }) {
  // Held as uniforms whose .value is live, so moving the sun is a uniform write
  // rather than a material rebuild.
  const sun = createSunUniforms(sunDirection, angular);
  const maxDist = uniform(float(maxDistance));
  const ambient = ambientUniform || uniform(float(DEFAULT_AMBIENT));
  const sunShadow = createSoftSunTSL({
    cascades, sunDir: sun.dir, sunTangent: sun.tangent,
    sunBitangent: sun.bitangent, maxDist, coneRadiusUniform: sun.coneRadius,
    rayCount: rays, bayerTex, stepsUniform, biasUniform, cone, quantise,
    fadeStartUniform, edgeFadeUniform
  });

  const pointLight = torch
    ? createPointLightTSL({ cascades, torch, biasUniform,
                            fadeStartUniform, edgeFadeUniform })
    : null;

  const node = Fn(() => {
    // normalWorld rather than a derivative reconstruction: the terrain is boxes,
    // so the interpolated normal IS the exact face normal, and the marcher needs
    // it to lift the ray origin off the face along the surface rather than along
    // the ray - see SURFACE_BIAS_VOXELS for why that distinction matters at low
    // sun angles.
    const n = normalize(normalWorld);
    // Locked once, used by every light. Two lights that disagreed about where a
    // surface IS would put their shadows on different texel grids, and the
    // mismatch would read as the torch's shadow crawling against the sun's.
    const p = texelLockTSL(positionWorld, n).toVar();
    // Sun off: no march, no shading term, just the ambient floor for anything a
    // dynamic light does not reach.
    if (!sunOn) {
      const dark = vec3(ambient, ambient, ambient);
      return torch ? dark.add(pointLight(p, n)) : dark;
    }
    const visible = sunShadow(p, n);
    // shadowOnly writes the raw visibility term. Ambient and N.L both compress
    // the shadowed range toward the middle, so acne that is plain here is
    // invisible in the shaded result - which is why it has been hard to name.
    if (shadowOnly) return vec3(visible, visible, visible);
    const sunTerm = sunShadeTSL({ visible, n, sunDir: sun.dir,
                                  ambientUniform: ambient, mode: shadeMode });
    if (!torch) return vec3(sunTerm, sunTerm, sunTerm);
    return vec3(sunTerm, sunTerm, sunTerm).add(pointLight(p, n));
  })();

  return { node, sun, cascades, torch,
           maxDistUniform: maxDist, ambientUniform: ambient };
}

// --- A dynamic point light, marched through the same field ---
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
function createPointLightTSL({ cascades, torch, biasUniform, fadeStartUniform,
                               edgeFadeUniform }) {
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

  return Fn(([p, n]) => {
    const toLight = torch.position.sub(p).toVar();
    // max() rather than a branch: a fragment exactly at the light would divide
    // by zero, and the answer there is arbitrary anyway.
    const dist = max(length(toLight), float(1e-4)).toVar();
    const dir = toLight.div(dist).toVar();

    // lights.js: the level IS the radius in blocks, and (1 - d/R)^2 reaches zero
    // at exactly that radius, so the light ends where the level says and the
    // boundary has no ring.
    const radius = torch.level.mul(float(BLOCK_METRES));
    const u = clamp(float(1).sub(dist.div(max(radius, float(1e-4)))),
                    float(0), float(1));
    // log2(1 + u), mirroring lights.js lightFalloff - exact at both ends and far
    // brighter than a square across the middle, which is what a torch looks like.
    const falloff = log2(float(1).add(u)).mul(torch.level.div(float(MAX_LIGHT_LEVEL)));

    const ndotl = max(n.dot(dir), float(0));
    // Lifted off the face along the NORMAL, the same as the sun's origin and for
    // the same reason - see SURFACE_BIAS_VOXELS.
    const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES))));

    // lights.js coneSlope(): the cone from this fragment to the emitter opens to
    // sourceRadius at the light, so its slope is sourceRadius / D. Unlike the
    // sun's, this is per fragment - which is the whole difference between a
    // light that is 150 million km away and one in the player's hand.
    const slope = torch.sourceRadius.div(dist);
    // Capped at the nearer of the light and its own radius: nothing past either
    // can take away light that is already zero there.
    const visible = trace(origin, dir, min(dist, radius), slope,
                          fadeStart, edgeFade).x;

    return torch.colour.mul(falloff).mul(ndotl).mul(visible);
  });
}

// A level's footprint travels with the player, or shading only works where it
// happened to be built. Repopulating in place and rewriting the level's uniforms
// is enough - the texture object and the material are unchanged, so nothing
// rebuilds. The grid remembers its own level, so this works for any cascade.
export function followShadowGrid(grid, tex, bindings, originBlock) {
  createBoxGridAt(originBlock.x, originBlock.z, grid, grid.level);
  writeCascadeBindings(bindings, grid);
  tex.needsUpdate = true;
  return grid;
}

// Swaps the terrain to a node material whose colour is its own albedo modulated
// by the marched shadow term, keeping the original so it can be restored.
export function applyShadowMaterial(mesh, shadowNode) {
  if (!mesh.userData.originalMaterial) mesh.userData.originalMaterial = mesh.material;
  const base = mesh.userData.originalMaterial;

  const mat = new THREE.MeshBasicNodeMaterial();
  // .rgb, because the shadow node is a vec3 now that a coloured light adds to it.
  const albedo = base.map ? texture(base.map, uv()).rgb : vec3(1, 1, 1);
  // Per-instance tint (arena greying) lives in instanceColor, so it has to be
  // carried across or battle mode loses its reachable-tile shading.
  mat.colorNode = albedo.mul(shadowNode);
  mesh.material = mat;
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

export function sunShadeTSL({ visible, n, sunDir, ambientUniform, mode = 'ground' }) {
  if (mode === 'flat') return ambientUniform.add(visible.mul(float(1).sub(ambientUniform)));

  const ndotl = max(n.dot(sunDir), float(0)).toVar();
  if (mode === 'ground') {
    // sunDir.y IS the N.L a horizontal up-facing surface would receive.
    const ref = max(sunDir.y, float(GROUND_REFERENCE_FLOOR));
    ndotl.assign(clamp(ndotl.div(ref), float(0), float(1)));
  }
  return ambientUniform.add(visible.mul(ndotl).mul(float(1).sub(ambientUniform)));
}

export function createSoftSunTSL({ cascades, sunDir, sunTangent,
                                  sunBitangent, maxDist, coneRadiusUniform,
                                  rayCount = 1, bayerTex = null,
                                  stepsUniform = null, biasUniform = null,
                                  cone = false, quantise = true,
                                  fadeStartUniform = null, edgeFadeUniform = null }) {
  const offsets = coneOffsets(rayCount);
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
    return Fn(([p, n]) => {
      const origin = p.add(n.mul(bias.mul(float(VOXEL_METRES))));
      const visible = coneTrace(origin, sunDir, maxDist, coneRadiusUniform,
                                fadeStart, edgeFade).x.toVar();
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

  return Fn(([p, n]) => {
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
