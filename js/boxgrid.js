import { World, getVoxelKey, CHUNK_SIZE, Y_MIN, CHUNK_HEIGHT, BLOCK_METRES } from './world.js';

// --- boxGrid: the fine occupancy / light field ---
//
// The frozen constant: 12 texels per 1.5 m block. Under the 1:1 lock in the
// rendering architecture, boxGrid voxel size IS texel size - one voxel per
// texel, no second resolution to tune. A shadow edge can only be as fine as the
// voxel that occluded it, so this is where the pixelation of the lighting lives.
//
// C0 is sized to exactly one chunk: 12 blocks x 12 texels = 144 on every axis,
// vertically covering the full authorable range (Y_MIN .. Y_MAX, 12 levels).
// A single-chunk encounter therefore has no cascade boundary anywhere in the
// playable area.
//
// Dense Uint8Array for now, per the doc's "start dense single-grid and add
// cascades on hitting the wall". 144^3 at one byte is ~2.99 MB, which is well
// inside budget; sparse bricking lands later, when multi-chunk encounters need it.

export const TEXELS_PER_BLOCK = 12;
export const GRID_DIM = CHUNK_SIZE * TEXELS_PER_BLOCK;      // 144
export const VOXEL_METRES = BLOCK_METRES / TEXELS_PER_BLOCK;  // 0.125 m

// --- Cascades ---
//
// One field was never going to be enough, and the symptom was specific: a shadow
// cut off by a diagonal line once the player walked away from its caster. Shadow
// rays travel toward the sun, so a receiver at the far end of a long shadow has
// to march a long way up-sun to reach what casts it, and C0's 18 m ran out first.
// Leaning the footprint up-sun bought a few metres; this is the structural fix.
//
// Every level is the SAME 144^3 buffer at a different voxel size, so each step
// coarser doubles the reach for the same 2.99 MB:
//
//   C0   12 voxels/block   12.5 cm   12 blocks   18 m
//   C1    6 voxels/block   25 cm     24 blocks   36 m
//   C2    3 voxels/block   50 cm     48 blocks   72 m
//
// The 1:1 lock between voxel and texel holds at C0 only, which is the level the
// atlas is addressed in. C1 exists to answer "is anything in the way" for distant
// casters, where a shadow edge is already smaller than a texel - so half
// resolution there costs nothing visible.
export const CASCADE_COUNT = 3;

// Voxels per block at each level: 12, 6, then 3. Halving rather than any other
// ratio so a C1 voxel is a whole number of C0 voxels, which keeps the two fields
// in phase and stops a cascade boundary landing mid-voxel.
export function cascadeVoxelsPerBlock(level) {
  return TEXELS_PER_BLOCK >> level;
}

export function cascadeVoxelMetres(level) {
  return BLOCK_METRES / cascadeVoxelsPerBlock(level);
}

// Side of the footprint in blocks: 12, then 24.
export function cascadeBlocks(level) {
  return GRID_DIM / cascadeVoxelsPerBlock(level);
}

export function cascadeExtentMetres(level) {
  return cascadeBlocks(level) * BLOCK_METRES;
}

// THE CLAMP IS CONSTANT IN WORLD SPACE, not in voxels - 1 m at every level, so 8
// voxels at C0 and 4 at C1.
//
// Holding it at 8 voxels everywhere instead would have made the coarse levels
// cost MORE to bake than the fine one, which is the opposite of what a cascade is
// for: C1 covers four times the area, so it touches four times the blocks, and
// per-block the band would have been (6+16)^3 against C0's (12+16)^3 - about
// 6.1M writes against 3.2M. Scaling the range with the voxel keeps C1 at (6+8)^3
// over those blocks, which is cheaper than C0 rather than dearer.
//
// It also keeps the cone trace honest. Its penumbra saturates once d/(R*t)
// reaches 1, which depends on the range in METRES - so a constant world-space
// range means a ray crossing from C0 into C1 does not change brightness at the
// seam. A constant voxel range would have doubled the saturation distance at C1
// and put a visible step there.
export const DISTANCE_RANGE_METRES = 1;

export function cascadeRangeVoxels(level) {
  return DISTANCE_RANGE_METRES / cascadeVoxelMetres(level);
}

if (CHUNK_HEIGHT * TEXELS_PER_BLOCK !== GRID_DIM) {
  // Guards the 1:1 lock: the vertical span must produce the same voxel count as
  // the footprint, or C0 stops being a cube and the cascade math in the design
  // doc no longer holds.
  throw new Error(`boxGrid is not cubic: ${GRID_DIM} wide but ${CHUNK_HEIGHT * TEXELS_PER_BLOCK} tall`);
}

// A block at grid coordinate b is rendered centred on b * BLOCK_METRES, so its
// minimum corner sits half a block lower.
function blockMinCorner(b) {
  return (b - 0.5) * BLOCK_METRES;
}

// The grid covers a 12x12 block footprint, but WHERE that footprint sits is the
// caller's choice - see gridOriginFor(). Pass an existing grid as `reuse` to
// repopulate it in place: the occupancy buffer is ~3 MB, so reallocating it
// every time the origin moves is a visible hitch for no reason.
export function createBoxGridAt(originBlockX, originBlockZ, reuse = null, level = 0) {
  const grid = reuse || {
    data: new Uint8Array(GRID_DIM * GRID_DIM * GRID_DIM),
    dim: GRID_DIM,
    level,
    voxelsPerBlock: cascadeVoxelsPerBlock(level),
    voxelSize: cascadeVoxelMetres(level),
    range: cascadeRangeVoxels(level),
    origin: { x: 0, y: blockMinCorner(Y_MIN), z: 0 },
    originBlock: { x: 0, y: Y_MIN, z: 0 }
  };
  // A RE-ORIGIN IS A SCROLL, NOT A REBAKE.
  //
  // Stepping one block moves the footprint by one block and leaves eleven
  // twelfths of the field holding exactly the values it should still hold. The
  // first version re-baked all 144^3 anyway - 18.8 ms at C0, against a 4.2 ms
  // budget at 240 Hz - so every block stepped cost four or five dropped frames
  // while the steady-state framerate sat pinned at the refresh rate. That is the
  // shape of a hitch on movement and nothing about shading load explains it.
  //
  // So: memmove the overlap into place, then bake ONLY the strip that scrolled
  // in. The retained values need no revisiting, and this is worth stating
  // because it is not obvious - a stored distance is the true distance to the
  // nearest geometry, which does not depend on where the footprint happens to
  // sit. A block that has since scrolled out of the footprint still contributed
  // a correct distance, and keeping it is righter than dropping it. What the
  // footprint decides is only which voxels EXIST, not what they mean.
  const prior = grid.filledBlocks ? grid.originBlock : null;
  const dx = prior ? originBlockX - prior.x : 0;
  const dz = prior ? originBlockZ - prior.z : 0;
  const per = grid.voxelsPerBlock || TEXELS_PER_BLOCK;

  grid.origin.x = blockMinCorner(originBlockX);
  grid.origin.z = blockMinCorner(originBlockZ);
  grid.originBlock.x = originBlockX;
  grid.originBlock.z = originBlockZ;

  const sx = dx * per, sz = dz * per;
  const slid = prior && (sx !== 0 || sz !== 0) &&
               Math.abs(sx) < GRID_DIM && Math.abs(sz) < GRID_DIM;
  if (slid) {
    const strips = ringScroll(grid, sx, sz);
    populateDistanceField(grid, strips);
    // What the GPU copy has to be told about - see followShadowGrid.
    grid.uploadRects = strips;
    // Two pieces of bookkeeping the dynamic occluders of occluders.js depend on,
    // both of which are facts about the GRID rather than about the occluders,
    // which is why they are settled here rather than there:
    //
    //   the stored proxy boxes are in grid coordinates, and the scroll has just
    //   moved those coordinates out from under them - so translate, don't
    //   discard. A proxy that did not move in the WORLD then still compares
    //   equal next frame and costs nothing, which is the difference between a
    //   block step being free and a block step re-writing every character in
    //   the scene.
    //
    //   the strip bake has just cleared what slid in to static truth, which
    //   destroys the imprint of any proxy standing there - without that proxy
    //   having moved, so nothing else would notice. Hand the strips over and
    //   applyOccluders re-writes whatever they touched.
    translateOccluderBoxes(grid, sx, sz);
    grid.scrollStrips = strips;
  }
  else if (prior && sx === 0 && sz === 0) {
    populateDistanceField(grid, []);
    grid.uploadRects = [];
  }
  else {
    populateDistanceField(grid);
    grid.uploadRects = null;   // everything changed: a full upload
    // A full bake is not a scroll: nothing was retained, so every imprint is
    // gone and every proxy has to be written again from nothing.
    grid.occluderBoxes = null;
    grid.scrollStrips = null;
  }
  return grid;
}

// Slide the remembered dynamic-occluder boxes along with the field. Kept beside
// scrollField because it is the same move applied to the other thing that stores
// grid coordinates; see the call above for why it is a translate and not a drop.
// --- Handing a scroll between threads ---
//
// The field worker keeps its own mirror of each level and does the scroll and
// strip bake there; only the freshly baked strips come back. packRects and
// unpackRects agree on one order - rect by rect, then z, y, x in LOGICAL
// coordinates - so the two sides can have different rings and still agree.
export function packRects(grid, rects) {
  let n = 0;
  for (const r of rects) n += (r.x1 - r.x0) * (r.y1 - r.y0) * (r.z1 - r.z0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const r of rects) {
    const segs = ringSegments(grid, r.x0, r.x1 - 1);
    for (let vz = r.z0; vz < r.z1; vz++) {
      for (let vy = r.y0; vy < r.y1; vy++) {
        for (const [a, b] of segs) {
          const base = gridIndex(grid, a, vy, vz);
          out.set(grid.data.subarray(base, base + b - a + 1), o);
          o += b - a + 1;
        }
      }
    }
  }
  return out;
}

export function unpackRects(grid, rects, buf) {
  let o = 0;
  for (const r of rects) {
    const segs = ringSegments(grid, r.x0, r.x1 - 1);
    for (let vz = r.z0; vz < r.z1; vz++) {
      for (let vy = r.y0; vy < r.y1; vy++) {
        for (const [a, b] of segs) {
          const len = b - a + 1;
          grid.data.set(buf.subarray(o, o + len), gridIndex(grid, a, vy, vz));
          o += len;
        }
      }
    }
  }
}

// Worker side: re-origin the mirror and describe the result.
export function scrollForHandoff(grid, x, z) {
  createBoxGridAt(x, z, grid, grid.level);
  // null means the mirror did a full rebake - a jump too far to slide - so the
  // whole grid goes back, still in logical order.
  const full = grid.uploadRects === null;
  const rects = full
    ? [{ x0: 0, x1: GRID_DIM, y0: 0, y1: GRID_DIM, z0: 0, z1: GRID_DIM }]
    : grid.uploadRects;
  return { x, z, full, rects, data: packRects(grid, rects),
           occupiedCount: grid.occupiedCount, filledBlocks: grid.filledBlocks };
}

// Main side: the same end state createBoxGridAt would have reached, without
// the bake - origin, ring and the new strips land together, so the shader never
// sees an origin whose strip has not arrived. The ring is this grid's OWN: the
// mirror's can differ (a main-thread full bake keeps the ring, the mirror's
// reset starts at zero), which is why the strips travel in logical order.
export function applyHandoff(grid, m) {
  const per = grid.voxelsPerBlock || TEXELS_PER_BLOCK;
  const sx = (m.x - grid.originBlock.x) * per, sz = (m.z - grid.originBlock.z) * per;
  grid.origin.x = blockMinCorner(m.x);
  grid.origin.z = blockMinCorner(m.z);
  grid.originBlock.x = m.x;
  grid.originBlock.z = m.z;
  grid.occupiedCount = m.occupiedCount;
  grid.filledBlocks = m.filledBlocks;
  if (m.full) {
    unpackRects(grid, m.rects, m.data);
    grid.uploadRects = null;
    grid.occluderBoxes = null;
    grid.scrollStrips = null;
    return grid;
  }
  ringScroll(grid, sx, sz);
  unpackRects(grid, m.rects, m.data);
  translateOccluderBoxes(grid, sx, sz);
  grid.scrollStrips = m.rects;
  grid.uploadRects = m.rects;
  return grid;
}

export function translateOccluderBoxes(grid, sx, sz) {
  const boxes = grid.occluderBoxes;
  if (!boxes) return;
  for (const b of boxes.values()) { b.cx -= sx; b.cz -= sz; }
}

// Slide the field by (sx, sz) voxels and report what scrolled in.
//
// A voxel at new coordinate v held, before the move, the world point now at
// v + s - so new[v] = old[v + s], and the whole thing is one linear offset. It
// is still done row by row rather than as a single copyWithin over the buffer:
// x is the fastest axis, so an x shift makes each row's source spill into its
// neighbour, and a buffer-wide copy would drag that wrapped data into cells
// nothing afterwards clears.
//
// Returns the strips NOT covered by the copy, in new coordinates, as the dirty
// rects the bake then fills. Two of them when the footprint moved diagonally;
// they overlap at the corner, which costs one corner baked twice and is cheaper
// than the arithmetic to avoid it.
export function scrollField(grid, sx, sz) {
  const data = grid.data;
  const x0 = Math.max(0, -sx), x1 = Math.min(GRID_DIM, GRID_DIM - sx);
  const z0 = Math.max(0, -sz), z1 = Math.min(GRID_DIM, GRID_DIM - sz);
  const width = x1 - x0;

  if (width > 0 && z1 > z0) {
    // Ascending when the source lies after the destination, descending when it
    // lies before, so a slab is never read after it has been overwritten. Only
    // z needs the ordering: a nonzero sz puts source and destination in
    // different slabs, and a pure x shift is one self-contained copyWithin per
    // row, which handles its own overlap.
    const ahead = sz > 0 || (sz === 0 && sx > 0);
    for (let i = 0; i < z1 - z0; i++) {
      const vz = ahead ? z0 + i : z1 - 1 - i;
      for (let vy = 0; vy < GRID_DIM; vy++) {
        const dst = voxelIndex(x0, vy, vz);
        const src = voxelIndex(x0 + sx, vy, vz + sz);
        data.copyWithin(dst, src, src + width);
      }
    }
  }

  const rects = [];
  const full = { x0: 0, x1: GRID_DIM, y0: 0, y1: GRID_DIM, z0: 0, z1: GRID_DIM };
  if (sx > 0) rects.push({ ...full, x0: GRID_DIM - sx });
  else if (sx < 0) rects.push({ ...full, x1: -sx });
  if (sz > 0) rects.push({ ...full, z0: GRID_DIM - sz });
  else if (sz < 0) rects.push({ ...full, z1: -sz });
  return rects;
}

// Chunk-aligned convenience wrapper.
export function createBoxGrid(chunkX = 0, chunkZ = 0, reuse = null) {
  return createBoxGridAt(chunkX * CHUNK_SIZE, chunkZ * CHUNK_SIZE, reuse);
}

// Where the grid should sit for a given mode.
//
// Battle is chunk-aligned: the arena IS a chunk, it does not move for the
// duration of the fight, and aligning means the grid and the arena share
// bounds exactly.
//
// Explore follows the player instead, rebuilding once per block stepped rather
// than once per chunk crossed - so lighting coverage travels with them instead of
// falling off wherever they happen to be standing near a chunk edge.
//
// It does NOT centre on them, though; it leans up-sun. See the body for why.

// How far up-sun to push the explore footprint, in blocks.
// 0: every cascade centred on the player. The sun lean (see gridOriginFor)
// dates from before C1 and C2, when a long shadow ran out of C0 with nowhere
// to go; now a sun ray leaving C0 carries on in C1, and the lean only put the
// footprints off-centre. Kept as a setting (bxb.gridbias).
export const SUN_BIAS_BLOCKS = 0;

export function gridOriginFor(mode, gridPos, sunDirection = null,
                              bias = SUN_BIAS_BLOCKS, level = 0) {
  if (mode === 'battle') {
    const cx = Math.floor(gridPos.x / CHUNK_SIZE) * CHUNK_SIZE;
    const cz = Math.floor(gridPos.z / CHUNK_SIZE) * CHUNK_SIZE;
    // C0 IS the arena, exactly. A coarser level is wider than a chunk, so it
    // centres on the same arena rather than aligning to it - padding out equally
    // on all sides keeps the arena in the middle of every level, which is what
    // makes a battle's lighting stable for its whole duration.
    if (level === 0) return { x: cx, z: cz };
    const pad = (cascadeBlocks(level) - CHUNK_SIZE) / 2;
    return { x: cx - pad, z: cz - pad };
  }
  const half = Math.floor(cascadeBlocks(level) / 2);

  // BIASED TOWARD THE SUN, not centred on the player.
  //
  // A shadow ray travels TOWARD the sun, so the occluders that can affect ground
  // the player sees all lie up-sun of it - and the footprint spent down-sun holds
  // nothing that can cast onto anything visible. Centred, the window reaches 9 m
  // each way; at a 20 degree sun a 3 m wall throws an 8.2 m shadow, so a caster
  // that matters sits right on the boundary and its far shadow is cut off. That
  // is the diagonal line across the end of a long shadow once you walk away from
  // its caster.
  //
  // Pushing the window up-sun trades down-sun reach for caster range at no cost -
  // same footprint, same bake. The trade is not free in principle: ground down-sun
  // of the player is still visible and now falls outside the job list, where it
  // keeps its last shaded value rather than being updated. A stale shadow reads
  // far better than a missing one, which is what makes this the right direction to
  // spend the asymmetry.
  //
  // Only the HORIZONTAL component matters - elevation sets shadow length, azimuth
  // sets which way the footprint should lean.
  let x = gridPos.x - half, z = gridPos.z - half;
  if (sunDirection && bias) {
    const len = Math.hypot(sunDirection.x, sunDirection.z);
    if (len > 1e-6) {
      x += Math.round((sunDirection.x / len) * bias);
      z += Math.round((sunDirection.z / len) * bias);
    }
  }

  // COARSE LEVELS SNAP TO A COARSER STRIDE, so they do not rebuild as often as
  // C0 does. Without this every level re-origins on every block stepped and a
  // cascade costs its full bake per step - which is most of the reason to have
  // one gone. C1 moves in 2-block jumps, so it rebuilds half as often, and being
  // off-centre by up to 2 of its 24 blocks is not something a distant shadow can
  // show. Math.floor rather than truncation, so the stride is uniform either side
  // of the origin instead of bunching up around zero.
  //
  // ROUNDED to the nearest stride, not floored: flooring put C1 up to a block
  // and C2 up to three toward -x/-z of the player, every time. Rounding keeps
  // the same stride - so the same rebuild rate - with the player within half
  // a stride of centre.
  const snap = 1 << level;
  return {
    x: Math.round(x / snap) * snap,
    z: Math.round(z / snap) * snap
  };
}

export function voxelIndex(vx, vy, vz) {
  // x fastest, then y, then z - the layout a Data3DTexture expects, so the
  // array can be handed straight to the GPU with no repacking.
  return vx + vy * GRID_DIM + vz * GRID_DIM * GRID_DIM;
}

// --- The ring ---
//
// X and Z are stored as a ring buffer: logical voxel v lives at physical
// (v + ring) mod GRID_DIM. A re-origin then moves the ring instead of the data,
// so a step writes only the strip that scrolled in - on the CPU AND in the GPU
// upload, which used to be all 3 MB of every level on every block stepped. Y
// never scrolls, so it is not ringed. The shader applies the same offset and
// samples with repeat wrapping, so filtering across the physical seam reads the
// logical neighbour.
//
// Everything that indexes grid.data by logical coordinate goes through
// gridIndex; the hot loops walk ringSegments so their idx++ never crosses the
// physical seam.
export function gridIndex(grid, vx, vy, vz) {
  let px = vx + (grid.ringX || 0); if (px >= GRID_DIM) px -= GRID_DIM;
  let pz = vz + (grid.ringZ || 0); if (pz >= GRID_DIM) pz -= GRID_DIM;
  return px + vy * GRID_DIM + pz * GRID_DIM * GRID_DIM;
}

// A logical x range [x0, x1] inclusive, as at most two runs that are each
// contiguous in memory.
export function ringSegments(grid, x0, x1) {
  const cut = GRID_DIM - (grid.ringX || 0);   // first logical x that wraps to 0
  if (x1 < cut || x0 >= cut) return [[x0, x1]];
  return [[x0, cut - 1], [cut, x1]];
}

// The same for z, for the GPU strip copy, which needs physical boxes.
export function ringSegmentsZ(grid, z0, z1) {
  const cut = GRID_DIM - (grid.ringZ || 0);
  if (z1 < cut || z0 >= cut) return [[z0, z1]];
  return [[z0, cut - 1], [cut, z1]];
}

const wrap = v => ((v % GRID_DIM) + GRID_DIM) % GRID_DIM;

// The ring's scroll: new[v] = old[v + s] with no data moved, and the same dirty
// strips scrollField reports.
export function ringScroll(grid, sx, sz) {
  grid.ringX = wrap((grid.ringX || 0) + sx);
  grid.ringZ = wrap((grid.ringZ || 0) + sz);
  const rects = [];
  const full = { x0: 0, x1: GRID_DIM, y0: 0, y1: GRID_DIM, z0: 0, z1: GRID_DIM };
  if (sx > 0) rects.push({ ...full, x0: GRID_DIM - sx });
  else if (sx < 0) rects.push({ ...full, x1: -sx });
  if (sz > 0) rects.push({ ...full, z0: GRID_DIM - sz });
  else if (sz < 0) rects.push({ ...full, z1: -sz });
  return rects;
}

export function inBounds(vx, vy, vz) {
  return vx >= 0 && vy >= 0 && vz >= 0 && vx < GRID_DIM && vy < GRID_DIM && vz < GRID_DIM;
}

// --- The distance field ---
//
// One byte per voxel, same footprint as the binary occupancy this replaced, but
// the byte now means SIGNED DISTANCE TO THE NEAREST SURFACE in voxels, negative
// inside solid, clamped to +-DISTANCE_RANGE voxels. Two things follow, and only
// one of them is about speed:
//
//   1. Slopes and stairs stop needing sub-blocks. A slope is just a distance
//      function whose zero-crossing falls between voxel centres, so occlusion
//      gains sub-voxel precision without any change to the grid. Lighting
//      pixelation is unchanged, because radiance is still one value per voxel -
//      it is only the occlusion TEST that gets finer.
//   2. Rays can sphere-trace, stepping by the stored distance instead of
//      crawling voxel to voxel.
//
// On (2): this used to come with an apology. The range was clamped to one voxel,
// so a step in open air was one voxel and sphere tracing was barely better than
// the DDA it replaced - a win only on diagonals, where DDA pays ~1.7 steps per
// voxel of travel and this pays one. Measured, 92.6% of a populated grid sat
// pinned at the maximum, meaning the field had nothing to say about open space
// anywhere.
//
// The range is now 8 voxels, and that changes three things at once: rays cross
// open air in metre steps rather than 12.5 cm ones, the sun's shadow can be
// SOFTENED from a single ray because the stored distance is a usable estimate of
// how much room there is beside it (see sun.js), and the step cap stops being the
// thing that terminates a long ray.
//
// A NOTE ON WHAT WAS BROKEN HERE. DISTANCE_RANGE was not a working knob: the bake
// inlines the encode into its hot loop, and that inlined copy clamped to +-1
// without dividing by the range. Widening the constant widened the loop bounds,
// cost proportionally more time, and wrote the same bytes - the field stayed
// pinned at 1.6% graded exterior at every range from 1 to 16. Both copies now go
// through the same squared mapping and the test asserts they agree, because a
// constant that silently does nothing is worse than one that is wrong.
export const DISTANCE_RANGE = cascadeRangeVoxels(0);   // 8 voxels at C0
export const FAR_BYTE = 255;                   // the full range: air, nothing near
const SOLID_THRESHOLD = 128;                   // bytes below this decode negative

// --- Why the mapping is SQUARED rather than linear ---
//
// A linear map over +-8 voxels would spend the byte uniformly and lose 8x the
// precision everywhere, including at the zero crossing - which is the one place
// precision is the whole point, because the sub-voxel position of that crossing
// IS the slope mechanism.
//
// So: c = sign(d) * sqrt(|d| / RANGE), stored over the byte, and d = c * |c| *
// RANGE coming back. The derivative vanishes at the surface, so resolution
// concentrates exactly where the geometry is and thins out far away where all a
// ray needs is a rough answer to "how much room is there".
//
// The numbers, at RANGE = 8 and 127.5 steps per unit of c:
//
//   at the surface   0.0005 voxels per step  (16x FINER than the old +-1 linear)
//   at 1 voxel out   0.045                   (coarser, and nothing reads it)
//   at 8 voxels out  0.125                   (a sixteenth of a step; irrelevant)
//
// So this is strictly better than the old encoding at the surface and eight times
// the reach, for one extra multiply in the decode. The shader decode is
// c = sample*2-1 then c*abs(c)*RANGE - see traceDistanceTSL, which must stay the
// exact mirror of decodeDistance() or the CPU reference and the GPU part company.
// The range is a per-level property now, so both of these take it. It defaults to
// C0's so that call sites which only ever meant the fine field keep working.
export function encodeDistance(d, range = DISTANCE_RANGE) {
  const t = Math.min(1, Math.abs(d) / range);
  const c = Math.sign(d) * Math.sqrt(t);
  return Math.round((c + 1) * 127.5);
}

export function decodeDistance(byte, range = DISTANCE_RANGE) {
  const c = byte / 127.5 - 1;
  return c * Math.abs(c) * range;
}

// Distance at a voxel centre, in voxels. Outside the grid reads as open air, so
// a ray that leaves never reports a hit on the way out.
export function distanceAt(grid, vx, vy, vz) {
  const range = grid.range || DISTANCE_RANGE;
  if (!inBounds(vx, vy, vz)) return range;
  return decodeDistance(grid.data[gridIndex(grid, vx, vy, vz)], range);
}

// Kept as the binary view of the field, because a voxel centre is never exactly
// on a surface for axis-aligned geometry - the nearest it gets is half a voxel -
// so the sign is unambiguous here even though the field is continuous.
export function isOccupied(grid, vx, vy, vz) {
  if (!inBounds(vx, vy, vz)) return false;
  return grid.data[gridIndex(grid, vx, vy, vz)] < SOLID_THRESHOLD;
}

// Trilinear sample in world space, the CPU mirror of what texture3D with linear
// filtering does on the GPU. Interpolating a distance field is meaningful in a
// way that interpolating occupancy never was: between a solid voxel centre at
// -0.5 and its air neighbour at +0.5 the zero-crossing lands exactly on the
// shared face, which is where the surface actually is.
export function sampleDistance(grid, wx, wy, wz) {
  const p = worldToVoxelFloat(grid, wx, wy, wz);
  // Texel centres sit at voxel index + 0.5, so shift into centre-relative space.
  const fx = p.x - 0.5, fy = p.y - 0.5, fz = p.z - 0.5;
  const x0 = Math.floor(fx), y0 = Math.floor(fy), z0 = Math.floor(fz);
  const tx = fx - x0, ty = fy - y0, tz = fz - z0;
  const lerp = (a, b, t) => a + (b - a) * t;

  const d = (dx, dy, dz) => distanceAt(grid, x0 + dx, y0 + dy, z0 + dz);
  const y00 = lerp(d(0, 0, 0), d(1, 0, 0), tx), y10 = lerp(d(0, 1, 0), d(1, 1, 0), tx);
  const y01 = lerp(d(0, 0, 1), d(1, 0, 1), tx), y11 = lerp(d(0, 1, 1), d(1, 1, 1), tx);
  return lerp(lerp(y00, y10, ty), lerp(y01, y11, ty), tz);
}

// Exact signed distance from a point to an axis-aligned box, everything in
// voxel units. This is the whole bake for a full or half tile, and the hook for
// every other block type: swap this for the type's own distance function and
// slopes, stairs and quarter tiles all drop out with no change to the grid, the
// traversal, or the atlas. One block, one distance function, one brick.
export function boxDistance(px, py, pz, cx, cy, cz, hx, hy, hz) {
  const qx = Math.abs(px - cx) - hx;
  const qy = Math.abs(py - cy) - hy;
  const qz = Math.abs(pz - cz) - hz;
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
  const inside = Math.min(Math.max(qx, qy, qz), 0);
  return outside + inside;
}

// Continuous voxel-space coordinate of a world-space point (metres).
export function worldToVoxelFloat(grid, wx, wy, wz) {
  return {
    x: (wx - grid.origin.x) / grid.voxelSize,
    y: (wy - grid.origin.y) / grid.voxelSize,
    z: (wz - grid.origin.z) / grid.voxelSize
  };
}

export function worldToVoxel(grid, wx, wy, wz) {
  const f = worldToVoxelFloat(grid, wx, wy, wz);
  return { vx: Math.floor(f.x), vy: Math.floor(f.y), vz: Math.floor(f.z) };
}

export function voxelCentreToWorld(grid, vx, vy, vz) {
  return {
    x: grid.origin.x + (vx + 0.5) * grid.voxelSize,
    y: grid.origin.y + (vy + 0.5) * grid.voxelSize,
    z: grid.origin.z + (vz + 0.5) * grid.voxelSize
  };
}

// Voxelise every solid block whose footprint lands inside this chunk. Each
// block is exactly TEXELS_PER_BLOCK voxels on a side, so this is an AABB fill
// with no rasterisation involved - the payoff of a block-based world.
//
// Note this populates for everything resident in the chunk, NOT only what is
// on screen. Grid population and atlas shading have deliberately different
// visibility criteria: an off-screen torch-lit wall still has to be present
// here, or off-screen reflections silently degrade later.
// Min-in one axis-aligned box, in VOXEL units, over the given rects.
//
// Lifted out of populateDistanceField so that the static bake and the dynamic
// occluders of occluders.js run the identical kernel. Two implementations of
// "write a box into the field" would be two chances to disagree about the
// encoding, the band width or the min - and a proxy whose distances were a
// half-voxel off the terrain's would show as a character's shadow sitting at a
// different depth from the wall's beside it.
//
// Evaluate over the box plus the one-voxel band the clamp allows, and take the
// MINIMUM with whatever is already there. Minimum of distances is union of
// shapes, which is what makes a pile of blocks one continuous surface with
// correct distances along shared faces, edges and corners - and the byte
// encoding is monotonic in distance, so the bytes can be compared directly
// without decoding either side.
//
// Written out rather than calling boxDistance per voxel: this is the hot loop of
// the whole bake (~230k voxels for a ground footprint, every block the player
// steps), and the per-axis terms are constant across the loop that encloses
// them. Math.hypot is also avoided deliberately - its overflow-safe scaling
// costs several times a plain sqrt, and these operands are all within a voxel or
// two of zero.
export function minBoxVoxels(grid, cx, cy, cz, hx, hy, hz, rects,
                             range = grid.range || DISTANCE_RANGE) {
  const data = grid.data;
  const invRange = 1 / range;
  // The band, as a voxel box, before any rect clips it.
  const bLo = (v, h) => Math.ceil(v - h - range);
  const bHi = (v, h) => Math.floor(v + h + range);
  const bx0 = bLo(cx, hx), bx1 = bHi(cx, hx);
  const by0 = bLo(cy, hy), by1 = bHi(cy, hy);
  const bz0 = bLo(cz, hz), bz1 = bHi(cz, hz);

  for (const r of rects) {
    const x0 = Math.max(r.x0, bx0), x1 = Math.min(r.x1 - 1, bx1);
    const y0 = Math.max(r.y0, by0), y1 = Math.min(r.y1 - 1, by1);
    const z0 = Math.max(r.z0, bz0), z1 = Math.min(r.z1 - 1, bz1);
    if (x0 > x1 || y0 > y1 || z0 > z1) continue;
    const segs = ringSegments(grid, x0, x1);

    // The band is a BOX while the clamp is a SPHERE, so its corners are in
    // principle wasted work - a voxel at the corner of an eight-voxel band is
    // 8*sqrt(3) = 13.9 voxels out, past the clamp, so it encodes to FAR_BYTE and
    // the min against an already-cleared grid is a no-op. Clipping each row to the
    // sphere was tried and MEASURED SLOWER: 19.9 ms against 16.9. The rows are 28
    // voxels long, so a sqrt and four clamps per row cost more than the handful of
    // iterations they save - and for a flat ground slab most rows sit inside the
    // box's own y and z extent, where there is nothing to clip at all. Left as a
    // plain box on the strength of the measurement.
    for (let vz = z0; vz <= z1; vz++) {
      const qz = Math.abs(vz + 0.5 - cz) - hz;
      const pz = qz > 0 ? qz * qz : 0;
      for (let vy = y0; vy <= y1; vy++) {
        const qy = Math.abs(vy + 0.5 - cy) - hy;
        const py = qy > 0 ? qy * qy : 0;
        const qyz = qy > qz ? qy : qz;
        const pyz = py + pz;
        // Physically contiguous runs: a row can wrap in the ring.
        for (const [sx0, sx1] of segs) {
          let idx = gridIndex(grid, sx0, vy, vz);
          for (let vx = sx0; vx <= sx1; vx++, idx++) {
            const qx = Math.abs(vx + 0.5 - cx) - hx;
            const outside = qx > 0 ? Math.sqrt(pyz + qx * qx) : Math.sqrt(pyz);
            const mx = qx > qyz ? qx : qyz;
            const d = outside + (mx < 0 ? mx : 0);
            // encodeDistance, inlined. This copy is the one that used to fold the
            // clamp in at a hard-coded +-1 and so pinned the whole field to one
            // voxel whatever DISTANCE_RANGE said; sdf.test.mjs now asserts the two
            // agree byte for byte over the range.
            const t = (d < 0 ? -d : d) * invRange;
            const c = t >= 1 ? (d < 0 ? -1 : 1)
                             : (d < 0 ? -Math.sqrt(t) : Math.sqrt(t));
            const b = ((c + 1) * 127.5 + 0.5) | 0;
            if (b < data[idx]) data[idx] = b;
          }
        }
      }
    }
  }
}

export function populateDistanceField(grid, dirty = null) {
  // `dirty` is the list of voxel rects this call is responsible for, in grid
  // coordinates - what scrollField() just exposed. null means the whole grid,
  // which is a first bake or a jump too far to slide. An EMPTY list is a real
  // answer too: the footprint did not move, so nothing is dirty and there is
  // nothing to do.
  //
  // Everything below is clipped to these rects, not merely culled by them. A
  // block near the strip would otherwise re-write its whole band over retained
  // voxels - correct, but it is the band writes that cost the milliseconds, and
  // a one-block step touches a twelfth of the grid.
  const range = grid.range || DISTANCE_RANGE;
  const per = grid.voxelsPerBlock || TEXELS_PER_BLOCK;
  const band = Math.ceil(range);
  const whole = [{ x0: 0, x1: GRID_DIM, y0: 0, y1: GRID_DIM, z0: 0, z1: GRID_DIM }];
  const rects = dirty === null ? whole : dirty;

  // Clear only what this call will rewrite. A blanket fill of 3 MB is about
  // 0.6 ms, which is most of a frame at 240 Hz and pure waste when a strip is
  // all that changed.
  for (const r of rects) {
    if (r.x0 === 0 && r.x1 === GRID_DIM && r.y0 === 0 && r.y1 === GRID_DIM &&
        r.z0 === 0 && r.z1 === GRID_DIM) { grid.data.fill(FAR_BYTE); continue; }
    const segs = ringSegments(grid, r.x0, r.x1 - 1);
    for (let vz = r.z0; vz < r.z1; vz++) {
      for (let vy = r.y0; vy < r.y1; vy++) {
        for (const [a, b] of segs) {
          const base = gridIndex(grid, a, vy, vz);
          grid.data.fill(FAR_BYTE, base, base + (b - a + 1));
        }
      }
    }
  }

  let filled = 0;
  // The voxel origin of every block that landed in this chunk. Scanning for
  // surface voxels only needs to visit these ranges, not all 144^3 cells.
  const blocks = [];

  for (const key of World.keys()) {
    const [bx, by, bz] = key.split(',').map(Number);

    const vx0 = (bx - grid.originBlock.x) * per;
    const vy0 = (by - grid.originBlock.y) * per;
    const vz0 = (bz - grid.originBlock.z) * per;

    // Skip blocks whose distance BAND misses this chunk - which is not the same
    // as blocks outside it, and the difference was a real artifact.
    //
    // The band is what makes the field honest at its own boundary. Culling on the
    // block instead meant a wall one block past the footprint wrote nothing, so
    // the voxels beside it read FAR - "at least a metre of clear air" - when the
    // wall was 9 cm away. A ray believes that distance and steps a whole metre on
    // it, which at C0's boundary is a metre of path nothing has ever looked at:
    // it lands past the wall, leaves the grid, and hands a cascade that starts
    // BEHIND the occluder to the next level. Whether any given ray cleared the
    // silhouette in that one blind step depends on where its previous steps
    // happened to land, so the shadow edge came out as a staircase with a tread
    // of about a metre - and only when a caster sat within a block of the
    // boundary, which is why it appeared at exactly one distance and not on
    // either side of it.
    //
    // THE BAND AND THE LONGEST STEP ARE THE SAME METRE, and not by coincidence:
    // a saturated sample reads `range` voxels, so the longest step a trace can
    // take is range * voxelSize = DISTANCE_RANGE_METRES, and the band written
    // here reaches exactly that far past the footprint. Baking it means no step
    // can be taken on a distance that does not already account for whatever it
    // is about to jump over. Keep the two equal if either ever moves.
    const reach = band;
    if (vx0 + per + reach <= 0 || vy0 + per + reach <= 0 || vz0 + per + reach <= 0) continue;
    if (vx0 - reach >= GRID_DIM || vy0 - reach >= GRID_DIM || vz0 - reach >= GRID_DIM) continue;

    // Whether the block's own box - not its band - is in the footprint. Only
    // these count toward the occupancy figure, so the number keeps meaning "what
    // is in this chunk" rather than growing by a shell of neighbours.
    const inFootprint = vx0 >= 0 && vy0 >= 0 && vz0 >= 0 &&
                        vx0 < GRID_DIM && vy0 < GRID_DIM && vz0 < GRID_DIM;

    // A block with nothing beneath it - surface ground over empty space, the
    // common case now that there is no sub-surface fill - only needs its top
    // half voxelised. Nothing is ever below to march a ray up from, and the
    // lower 6 voxels can neither be seen nor terminate anything. Halves ground
    // occupancy outright. 12 divides evenly by 2, so the split lands on a
    // voxel boundary with nothing left over. Under the SDF this is not a
    // special case any more, just a box of a different height.
    const supported = World.has(getVoxelKey(bx, by - 1, bz));
    const dyStart = supported ? 0 : per / 2;
    const height = per - dyStart;
    // Floored: C2 has 3 voxels per block, so half a block is 1.5 voxels, and a
    // fractional start here had the overlay indexing the grid at non-integer
    // coordinates - read as empty, which dropped most of C2's surface. The
    // field is unaffected; the box itself is placed continuously below.
    blocks.push(vx0, vy0 + Math.floor(dyStart), vz0);

    // The box in voxel units, as a centre and a half-extent.
    const cx = vx0 + per / 2;
    const cy = vy0 + dyStart + height / 2;
    const cz = vz0 + per / 2;

    minBoxVoxels(grid, cx, cy, cz, per / 2, height / 2, per / 2, rects, range);
    if (inFootprint) filled += per * height * per;
  }

  grid.occupiedCount = filled;
  grid.filledBlocks = blocks;
  return filled;
}

// The old name, kept because "populate the occupancy" is what callers mean even
// though the byte is a distance now.
export const populateOccupancy = populateDistanceField;

// --- Sphere tracing ---
//
// The traversal the design doc specifies, and the reason the field is a distance
// rather than a bit: step by the distance stored at the current point, which can
// never overshoot a surface because that distance is the radius of a guaranteed
// empty sphere. Roughly a third of the DDA's line count and it handles a
// zero-crossing that falls mid-voxel, which is what slopes will be.
//
// MIN_STEP exists because the field is clamped: a ray running parallel to a wall
// sits at a small constant distance and would otherwise inch forward forever.
// Capping the step count and reporting no-hit is the conservative answer there -
// a grazing ray resolves as lit, which is the failure nobody can see.
export const HIT_EPS = 0.05;   // voxels; close enough to a surface to call it a hit
export const MIN_STEP = 0.2;   // voxels; guarantees forward progress
export const MAX_TRACE_STEPS = 2 * GRID_DIM;

export function sphereTrace(grid, origin, dir, maxDistance) {
  const len = Math.hypot(dir.x, dir.y, dir.z);
  if (len === 0) return { hit: false, distance: 0, steps: 0 };
  const d = { x: dir.x / len, y: dir.y / len, z: dir.z / len };

  let t = 0;
  for (let steps = 0; steps < MAX_TRACE_STEPS; steps++) {
    const p = { x: origin.x + d.x * t, y: origin.y + d.y * t, z: origin.z + d.z * t };
    const v = worldToVoxelFloat(grid, p.x, p.y, p.z);
    if (v.x < 0 || v.y < 0 || v.z < 0 ||
        v.x >= GRID_DIM || v.y >= GRID_DIM || v.z >= GRID_DIM) {
      return { hit: false, distance: t, steps, escaped: true };
    }

    const dist = sampleDistance(grid, p.x, p.y, p.z);
    if (dist < HIT_EPS) return { hit: true, distance: t, steps, voxel: worldToVoxel(grid, p.x, p.y, p.z) };

    t += Math.max(dist, MIN_STEP) * grid.voxelSize;
    if (t > maxDistance) return { hit: false, distance: t, steps };
  }
  return { hit: false, distance: t, steps: MAX_TRACE_STEPS, exhausted: true };
}

// --- CPU reference DDA (Amanatides-Woo) ---
//
// This exists to get the traversal provably right before it is ported to a TSL
// compute/fragment pass, where it is far harder to inspect. The GPU version must
// match this behaviour; keep the two in step.
//
// Stepping is done in metres along a normalised direction, so `distance` on the
// result is directly comparable to a light's radius.
export function marchOccupancy(grid, origin, dir, maxDistance, skipOriginVoxel = true) {
  const len = Math.hypot(dir.x, dir.y, dir.z);
  if (len === 0) return { hit: false, distance: 0, steps: 0 };
  const d = { x: dir.x / len, y: dir.y / len, z: dir.z / len };

  let { vx, vy, vz } = worldToVoxel(grid, origin.x, origin.y, origin.z);

  const step = { x: Math.sign(d.x), y: Math.sign(d.y), z: Math.sign(d.z) };
  const tDelta = {
    x: d.x === 0 ? Infinity : Math.abs(grid.voxelSize / d.x),
    y: d.y === 0 ? Infinity : Math.abs(grid.voxelSize / d.y),
    z: d.z === 0 ? Infinity : Math.abs(grid.voxelSize / d.z)
  };

  // Distance along the ray to the first voxel boundary on each axis.
  const firstBoundary = (axis, v, s) => {
    if (s === 0) return Infinity;
    const boundaryVoxel = s > 0 ? v + 1 : v;
    const boundaryWorld = grid.origin[axis] + boundaryVoxel * grid.voxelSize;
    return (boundaryWorld - origin[axis]) / d[axis];
  };
  const tMax = {
    x: firstBoundary('x', vx, step.x),
    y: firstBoundary('y', vy, step.y),
    z: firstBoundary('z', vz, step.z)
  };

  let t = 0;
  let steps = 0;
  let first = true;

  while (t <= maxDistance) {
    // The origin voxel is the shading surface itself; counting it as an
    // occluder would make every lit surface shadow itself.
    if (!(first && skipOriginVoxel) && isOccupied(grid, vx, vy, vz)) {
      return { hit: true, distance: t, steps, voxel: { vx, vy, vz } };
    }
    first = false;

    if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
      vx += step.x; t = tMax.x; tMax.x += tDelta.x;
    } else if (tMax.y <= tMax.z) {
      vy += step.y; t = tMax.y; tMax.y += tDelta.y;
    } else {
      vz += step.z; t = tMax.z; tMax.z += tDelta.z;
    }
    steps++;

    // Leaving the grid means the ray reached open sky for this cascade.
    if (!inBounds(vx, vy, vz)) return { hit: false, distance: t, steps, escaped: true };
  }

  return { hit: false, distance: t, steps };
}

// Convenience: is a world-space point in shadow from a light at lightPos?
export function isInShadow(grid, point, lightPos) {
  const dir = { x: lightPos.x - point.x, y: lightPos.y - point.y, z: lightPos.z - point.z };
  const dist = Math.hypot(dir.x, dir.y, dir.z);
  return marchOccupancy(grid, point, dir, dist).hit;
}
