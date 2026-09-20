import { BLOCK_METRES, Y_MIN, Y_MAX, CHUNK_SIZE } from './world.js';
import { TEXELS_PER_BLOCK, VOXEL_METRES } from './boxgrid.js';

// --- Phase C: the texel atlas ---
//
// The fork the whole rendering architecture turns on. Phase B shaded per SCREEN
// PIXEL: every fragment marched the occupancy grid, so cost scaled with
// resolution and the shadow edge was resolved as finely as the display allowed -
// which is exactly wrong for this art direction, because it means the lighting
// crawls and shimmers under the camera while the albedo stays locked to 12.5 cm
// texels.
//
// Here shading moves into OBJECT SPACE. Every visible block face owns a 12x12
// page in one big atlas - one texel per boxGrid voxel, the same 1:1 lock as
// everywhere else. A compute pass marches one ray per texel and writes the
// result into the page. The terrain fragment shader then does a single nearest
// fetch. Consequences, in order of how much they matter:
//
//   1. A shadow edge can only ever land on a texel boundary, so it is locked to
//      the same grid as the art. It cannot crawl, because it is not a function
//      of where the camera is.
//   2. Cost scales with visible SURFACE AREA, not pixels. Zooming in is free.
//   3. Shading is decoupled from frame rate. A page only needs rewriting when
//      the light or the geometry under it changes, which is what makes radiance,
//      multiple lights and GI affordable later - they are all just more work per
//      texel in a pass that does not run every frame.
//
// This module is the address book and the geometry: which page belongs to which
// face, where each texel sits in the world, and which faces are worth shading.
// No Three.js import, so all of it is testable headlessly - the mapping between
// the writer (compute, texel -> world position) and the reader (fragment,
// uv -> texel) is the one thing that must agree exactly, and a disagreement
// shows up as shadows mirrored or rotated on some faces but not others.

export const PAGE = TEXELS_PER_BLOCK;          // 12x12 texels per block face
export const TEXELS_PER_PAGE = PAGE * PAGE;    // 144
export const FACES_PER_BLOCK = 6;
export const NUDGE = VOXEL_METRES * 0.5;       // push the sample off the face into air

// WebGPU guarantees 8192; staying well under it leaves room on weak adapters.
const MAX_ATLAS_DIM = 4096;

// Face order is BoxGeometry's own: +X, -X, +Y, -Y, +Z, -Z. Anything that reads
// the geometry's faceIndex attribute depends on this staying in step with it.
export const FACE_NORMALS = [
  { x: 1, y: 0, z: 0 }, { x: -1, y: 0, z: 0 },
  { x: 0, y: 1, z: 0 }, { x: 0, y: -1, z: 0 },
  { x: 0, y: 0, z: 1 }, { x: 0, y: 0, z: -1 }
];

// BoxGeometry builds each face with buildPlane(u, v, w, udir, vdir, ...), which
// lays vertices out as
//     local[u] = (uv.x * S - S/2) * udir
//     local[v] = ((1 - uv.y) * S - S/2) * vdir
//     local[w] = +-S/2
// (uv.y is flipped because buildPlane emits 1 - iy/gridY.) These six rows are
// that call table. Getting one wrong transposes or mirrors the lighting on that
// face only, which is why the mapping is data here rather than six branches in
// a shader.
const FACE_PLANES = [
  { u: 'z', v: 'y', w: 'x', udir: -1, vdir: -1, wdir:  1 }, // +X
  { u: 'z', v: 'y', w: 'x', udir:  1, vdir: -1, wdir: -1 }, // -X
  { u: 'x', v: 'z', w: 'y', udir:  1, vdir:  1, wdir:  1 }, // +Y
  { u: 'x', v: 'z', w: 'y', udir:  1, vdir: -1, wdir: -1 }, // -Y
  { u: 'x', v: 'y', w: 'z', udir:  1, vdir: -1, wdir:  1 }, // +Z
  { u: 'x', v: 'y', w: 'z', udir: -1, vdir: -1, wdir: -1 }  // -Z
];

const zero = () => ({ x: 0, y: 0, z: 0 });

// Reference mapping, straight from buildPlane: where on the face does a given
// uv land, in block-local metres? Everything else in this file is derived from
// it, so a test can check the derivation without restating the premise.
export function faceUVToLocal(faceIndex, u, v) {
  const f = FACE_PLANES[faceIndex];
  const S = BLOCK_METRES, h = S / 2;
  const p = zero();
  p[f.u] = (u * S - h) * f.udir;
  p[f.v] = ((1 - v) * S - h) * f.vdir;
  p[f.w] = h * f.wdir;
  return p;
}

// The sampler side, and the only place the fragment shader's arithmetic is
// written down in JS. uv 1.0 would floor to PAGE, one past the end of the page,
// so it is clamped rather than allowed to bleed into the neighbouring page.
export function uvToTexel(u, v) {
  const clamp = t => Math.min(PAGE - 1, Math.max(0, Math.floor(t * PAGE)));
  return { tx: clamp(u), ty: clamp(v) };
}

// The writer side: texel (0,0)'s centre plus the two per-texel steps, in
// block-local metres, already pushed half a voxel clear of the face. Expressing
// it as origin + tx*uStep + ty*vStep means the compute kernel needs no branch
// on face at all - it just reads three vectors out of a buffer.
export function faceBasis(faceIndex) {
  const f = FACE_PLANES[faceIndex];
  const n = FACE_NORMALS[faceIndex];
  const S = BLOCK_METRES, h = S / 2, t = VOXEL_METRES;

  const uStep = zero(), vStep = zero(), origin = zero();
  uStep[f.u] = t * f.udir;
  // Negated because uv.y runs opposite to the vertex layout, and the atlas row
  // index follows uv.y so the sampler and the writer agree.
  vStep[f.v] = -t * f.vdir;
  origin[f.u] = (t * 0.5 - h) * f.udir;
  origin[f.v] = (h - t * 0.5) * f.vdir;
  origin[f.w] = h * f.wdir;

  origin.x += n.x * NUDGE; origin.y += n.y * NUDGE; origin.z += n.z * NUDGE;
  return { normal: n, uStep, vStep, origin };
}

// ---------------------------------------------------------------------------
// Page addressing
//
// Arithmetic, not a free list: page = instanceId * 6 + faceIndex. That reserves
// pages for faces that are never shaded (interiors, undersides), but atlas
// memory is cheap at this world size and a lookup table would cost an extra
// indirection in the hottest shader in the game. What is NOT wasted is GPU
// time: only exposed faces ever appear in a shade job.
// ---------------------------------------------------------------------------
export function atlasLayout(instanceCount) {
  const pageCount = Math.max(1, instanceCount * FACES_PER_BLOCK);
  const maxPagesX = Math.floor(MAX_ATLAS_DIM / PAGE);
  const pagesX = Math.max(1, Math.min(maxPagesX, Math.ceil(Math.sqrt(pageCount))));
  const pagesY = Math.ceil(pageCount / pagesX);
  return {
    pageCount, pagesX, pagesY,
    width: pagesX * PAGE,
    height: pagesY * PAGE,
    fits: pagesY * PAGE <= MAX_ATLAS_DIM,
    bytes: pagesX * PAGE * pagesY * PAGE * 4
  };
}

export function pageIndexFor(instanceId, faceIndex) {
  return instanceId * FACES_PER_BLOCK + faceIndex;
}

export function pageOriginTexels(pageIndex, pagesX) {
  return { x: (pageIndex % pagesX) * PAGE, y: Math.floor(pageIndex / pagesX) * PAGE };
}

// Where does a page texel sit in the world? blockPos is in block coordinates;
// render.js centres a block on blockPos * BLOCK_METRES.
export function texelWorldPosition(blockPos, faceIndex, tx, ty) {
  const b = faceBasis(faceIndex);
  return {
    x: blockPos.x * BLOCK_METRES + b.origin.x + b.uStep.x * tx + b.vStep.x * ty,
    y: blockPos.y * BLOCK_METRES + b.origin.y + b.uStep.y * tx + b.vStep.y * ty,
    z: blockPos.z * BLOCK_METRES + b.origin.z + b.uStep.z * tx + b.vStep.z * ty
  };
}

// ---------------------------------------------------------------------------
// Job list
//
// One job per face that is worth shading, covering the 12x12 block footprint the
// occupancy grid currently sits on - rays can only be marched where there is
// occupancy to march, so shading beyond the grid would write confidently wrong
// pages. Faces buried against a neighbouring block are skipped: they are never
// visible, and at ground level that is five faces out of six.
//
// Undersides ARE included even though the camera lives above the world. They
// cost a sixth of the pass, and leaving them out would mean trusting that no
// view ever catches one - the same assumption the debug overlay makes, but there
// a missing cube is obvious and here it would be a black face nobody can explain.
// ---------------------------------------------------------------------------
export function buildShadeJobs({ originBlockX, originBlockZ, isSolid, instanceIdOf,
                                 pagesX, pageCount,
                                 size = CHUNK_SIZE, yMin = Y_MIN, yMax = Y_MAX }) {
  const origins = [], uSteps = [], vSteps = [];
  let skippedUnmapped = 0;

  for (let bx = originBlockX; bx < originBlockX + size; bx++) {
    for (let bz = originBlockZ; bz < originBlockZ + size; bz++) {
      for (let by = yMin; by <= yMax; by++) {
        if (!isSolid(bx, by, bz)) continue;
        const instanceId = instanceIdOf(bx, by, bz);
        if (instanceId === undefined || instanceId === null) continue;

        for (let f = 0; f < FACES_PER_BLOCK; f++) {
          const n = FACE_NORMALS[f];
          if (isSolid(bx + n.x, by + n.y, bz + n.z)) continue; // buried

          const page = pageIndexFor(instanceId, f);
          if (page >= pageCount) { skippedUnmapped++; continue; }

          const b = faceBasis(f);
          origins.push(
            bx * BLOCK_METRES + b.origin.x,
            by * BLOCK_METRES + b.origin.y,
            bz * BLOCK_METRES + b.origin.z,
            page
          );
          uSteps.push(b.uStep.x, b.uStep.y, b.uStep.z, 0);
          vSteps.push(b.vStep.x, b.vStep.y, b.vStep.z, 0);
        }
      }
    }
  }

  return {
    count: origins.length / 4,
    origins: new Float32Array(origins),
    uSteps: new Float32Array(uSteps),
    vSteps: new Float32Array(vSteps),
    texels: (origins.length / 4) * TEXELS_PER_PAGE,
    skippedUnmapped,
    pagesX
  };
}
