import * as THREE from 'three/webgpu';
import { Fn, If, Loop, Break, float, vec3, uniform, texture3D, max, texture, abs, vec2, min, clamp, smoothstep, int, select } from 'three/tsl';
import { GRID_DIM, VOXEL_METRES, createBoxGridAt, DISTANCE_RANGE, FIELD_CHANNELS, gridIndex, ringSegments, ringSegmentsZ, HIT_EPS, MIN_STEP, MAX_TRACE_STEPS } from '../core/boxgrid.js';

// --- WebGPU compute plumbing and the sphere trace on the GPU ---

// The step ceiling, the hit epsilon and the minimum step all come from
// boxgrid.js rather than being restated here. They are the difference between
// the CPU reference and the shader agreeing and not, and bxb.parity() exists to
// catch exactly that - so there must be one definition, not two.

// ---------------------------------------------------------------------------
// The distance field as a 3D texture
//
// A Data3DTexture rather than a storage buffer, because the passes that consume
// it want a sampler: the field is read at continuous positions along a ray, not
// at integer cells.
// ---------------------------------------------------------------------------
// Two channels, interleaved exactly as grid.data holds them: R the opaque
// distance, G the glass distance (boxgrid.js FIELD_CHANNELS). Every existing read
// samples .r, so it sees the opaque field and nothing else changes.
export function createDistanceTexture(grid) {
  const tex = new THREE.Data3DTexture(grid.data, GRID_DIM, GRID_DIM, GRID_DIM);
  tex.format = THREE.RGFormat;
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
// Sphere tracing in TSL
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
// The cone trace: a soft shadow from ONE ray
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
export function eachCascade(cascades, edgeFade) {
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
export function seamBlend(cascades, origin, march, zero = () => vec2(0, 0)) {
  if (cascades.length < 2) return march(null);
  const w0 = c0WeightTSL(origin, cascades[0]).toVar();
  const both = w0.greaterThan(float(0)).and(w0.lessThan(float(1))).toVar();
  const passes = select(both, int(2), int(1)).toVar();
  const out = zero().toVar();
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
export const levelLive = (i, skipC0, done) =>
  i === 0 && skipC0 ? done.equal(float(0)).and(skipC0.not()) : done.equal(float(0));

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
        const C = FIELD_CHANNELS, row = w * C;
        if (stripBuf.length < row * h * d) stripBuf = new Uint8Array(row * h * d);
        let o = 0;
        for (let vz = za; vz <= zb; vz++) {
          for (let vy = r.y0; vy < r.y1; vy++, o += row) {
            const base = gridIndex(grid, xa, vy, vz) * C;
            stripBuf.set(grid.data.subarray(base, base + row), o);
          }
        }
        const phys = gridIndex(grid, xa, 0, za);
        queue.writeTexture(
          { texture: gpuTex,
            origin: { x: phys % GRID_DIM, y: r.y0, z: Math.floor(phys / (GRID_DIM * GRID_DIM)) } },
          stripBuf, { offset: 0, bytesPerRow: row, rowsPerImage: h },
          { width: w, height: h, depthOrArrayLayers: d });
      }
    }
  }
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
