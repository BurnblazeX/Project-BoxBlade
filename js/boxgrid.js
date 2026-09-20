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
export function createBoxGridAt(originBlockX, originBlockZ, reuse = null) {
  const grid = reuse || {
    data: new Uint8Array(GRID_DIM * GRID_DIM * GRID_DIM),
    dim: GRID_DIM,
    voxelSize: VOXEL_METRES,
    origin: { x: 0, y: blockMinCorner(Y_MIN), z: 0 },
    originBlock: { x: 0, y: Y_MIN, z: 0 }
  };
  grid.origin.x = blockMinCorner(originBlockX);
  grid.origin.z = blockMinCorner(originBlockZ);
  grid.originBlock.x = originBlockX;
  grid.originBlock.z = originBlockZ;
  populateDistanceField(grid);
  return grid;
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
// Explore centres the footprint on the player instead. Lighting coverage then
// stays symmetric around them rather than falling off on whichever side they
// happen to be standing near the chunk edge - which is also why the design doc
// centres its cascades on the camera. The cost is rebuilding once per block
// stepped rather than once per chunk crossed.
export function gridOriginFor(mode, gridPos) {
  if (mode === 'battle') {
    return {
      x: Math.floor(gridPos.x / CHUNK_SIZE) * CHUNK_SIZE,
      z: Math.floor(gridPos.z / CHUNK_SIZE) * CHUNK_SIZE
    };
  }
  const half = Math.floor(CHUNK_SIZE / 2);
  return { x: gridPos.x - half, z: gridPos.z - half };
}

export function voxelIndex(vx, vy, vz) {
  // x fastest, then y, then z - the layout a Data3DTexture expects, so the
  // array can be handed straight to the GPU with no repacking.
  return vx + vy * GRID_DIM + vz * GRID_DIM * GRID_DIM;
}

export function inBounds(vx, vy, vz) {
  return vx >= 0 && vy >= 0 && vz >= 0 && vx < GRID_DIM && vy < GRID_DIM && vz < GRID_DIM;
}

// --- The distance field ---
//
// One byte per voxel, same footprint as the binary occupancy this replaced, but
// the byte now means SIGNED DISTANCE TO THE NEAREST SURFACE in voxels, negative
// inside solid, clamped to +-1 voxel. Two things follow, and only one of them is
// about speed:
//
//   1. Slopes and stairs stop needing sub-blocks. A slope is just a distance
//      function whose zero-crossing falls between voxel centres, so occlusion
//      gains sub-voxel precision without any change to the grid. Lighting
//      pixelation is unchanged, because radiance is still one value per voxel -
//      it is only the occlusion TEST that gets finer.
//   2. Rays can sphere-trace, stepping by the stored distance instead of
//      crawling voxel to voxel.
//
// Honest note on (2): with the range clamped to +-1 voxel, a step in open air is
// one voxel, so this is not the order-of-magnitude win sphere tracing gives over
// a wide-range SDF. It beats DDA mainly on diagonals, where DDA pays ~1.7 steps
// per voxel of travel and this pays one. The clamp is what keeps the byte
// format, and slopes are the reason the field exists.
//
// The encoding maps [-1, +1] voxels onto the full byte range, which makes the
// shader-side decode a single multiply-add: an R8 texture samples normalised to
// [0,1], so distance = sample * 2 - 1. Nothing to divide by 255 anywhere.
export const DISTANCE_RANGE = 1;               // voxels; the clamp, both signs
export const FAR_BYTE = 255;                   // +1 voxel: air, nothing nearby
const SOLID_THRESHOLD = 128;                   // bytes below this decode negative

export function encodeDistance(d) {
  const c = Math.max(-1, Math.min(1, d / DISTANCE_RANGE));
  return Math.round((c + 1) * 127.5);
}

export function decodeDistance(byte) {
  return (byte / 127.5 - 1) * DISTANCE_RANGE;
}

// Distance at a voxel centre, in voxels. Outside the grid reads as open air, so
// a ray that leaves never reports a hit on the way out.
export function distanceAt(grid, vx, vy, vz) {
  if (!inBounds(vx, vy, vz)) return DISTANCE_RANGE;
  return decodeDistance(grid.data[voxelIndex(vx, vy, vz)]);
}

// Kept as the binary view of the field, because a voxel centre is never exactly
// on a surface for axis-aligned geometry - the nearest it gets is half a voxel -
// so the sign is unambiguous here even though the field is continuous.
export function isOccupied(grid, vx, vy, vz) {
  if (!inBounds(vx, vy, vz)) return false;
  return grid.data[voxelIndex(vx, vy, vz)] < SOLID_THRESHOLD;
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
export function populateDistanceField(grid) {
  // Clear only what was written last time, expanded by the field's one-voxel
  // band. A blanket fill memsets ~3 MB on every re-origin, which in explore mode
  // is every block the player steps. All clearing happens before any writing, so
  // over-clearing a neighbour's band cannot erase it.
  const previous = grid.filledBlocks;
  if (previous) {
    for (let k = 0; k < previous.length; k += 3) {
      const x0 = Math.max(0, previous[k] - 1);
      const x1 = Math.min(GRID_DIM - 1, previous[k] + TEXELS_PER_BLOCK);
      const y0 = Math.max(0, previous[k + 1] - 1);
      const y1 = Math.min(GRID_DIM - 1, previous[k + 1] + TEXELS_PER_BLOCK);
      const z0 = Math.max(0, previous[k + 2] - 1);
      const z1 = Math.min(GRID_DIM - 1, previous[k + 2] + TEXELS_PER_BLOCK);
      for (let vz = z0; vz <= z1; vz++) {
        for (let vy = y0; vy <= y1; vy++) {
          const base = voxelIndex(x0, vy, vz);
          grid.data.fill(FAR_BYTE, base, base + (x1 - x0 + 1));
        }
      }
    }
  } else {
    grid.data.fill(FAR_BYTE);
  }

  let filled = 0;
  // The voxel origin of every block that landed in this chunk. Scanning for
  // surface voxels only needs to visit these ranges, not all 144^3 cells.
  const blocks = [];

  for (const key of World.keys()) {
    const [bx, by, bz] = key.split(',').map(Number);

    const vx0 = (bx - grid.originBlock.x) * TEXELS_PER_BLOCK;
    const vy0 = (by - grid.originBlock.y) * TEXELS_PER_BLOCK;
    const vz0 = (bz - grid.originBlock.z) * TEXELS_PER_BLOCK;

    // Skip blocks outside this chunk entirely.
    if (vx0 < 0 || vy0 < 0 || vz0 < 0) continue;
    if (vx0 >= GRID_DIM || vy0 >= GRID_DIM || vz0 >= GRID_DIM) continue;

    // A block with nothing beneath it - surface ground over empty space, the
    // common case now that there is no sub-surface fill - only needs its top
    // half voxelised. Nothing is ever below to march a ray up from, and the
    // lower 6 voxels can neither be seen nor terminate anything. Halves ground
    // occupancy outright. 12 divides evenly by 2, so the split lands on a
    // voxel boundary with nothing left over. Under the SDF this is not a
    // special case any more, just a box of a different height.
    const supported = World.has(getVoxelKey(bx, by - 1, bz));
    const dyStart = supported ? 0 : TEXELS_PER_BLOCK / 2;
    const height = TEXELS_PER_BLOCK - dyStart;
    blocks.push(vx0, vy0 + dyStart, vz0);

    // The box in voxel units, as a centre and a half-extent.
    const cx = vx0 + TEXELS_PER_BLOCK / 2;
    const cy = vy0 + dyStart + height / 2;
    const cz = vz0 + TEXELS_PER_BLOCK / 2;
    const hx = TEXELS_PER_BLOCK / 2, hy = height / 2, hz = TEXELS_PER_BLOCK / 2;

    // Evaluate over the box plus the one-voxel band the clamp allows, and take
    // the MINIMUM with whatever is already there. Minimum of distances is union
    // of shapes, which is what makes a pile of blocks one continuous surface
    // with correct distances along shared faces, edges and corners - and the
    // byte encoding is monotonic in distance, so the bytes can be compared
    // directly without decoding either side.
    // Written out rather than calling boxDistance per voxel: this is the hot
    // loop of the whole bake (~230k voxels for a ground footprint, every block
    // the player steps), and the per-axis terms are constant across the loop
    // that encloses them. Math.hypot is also avoided deliberately - its
    // overflow-safe scaling costs several times a plain sqrt, and these operands
    // are all within a voxel or two of zero.
    const lo = (v, h) => Math.max(0, Math.ceil(v - h - DISTANCE_RANGE));
    const hi = (v, h) => Math.min(GRID_DIM - 1, Math.floor(v + h + DISTANCE_RANGE));
    const x0 = lo(cx, hx), x1 = hi(cx, hx);
    const data = grid.data;

    for (let vz = lo(cz, hz); vz <= hi(cz, hz); vz++) {
      const qz = Math.abs(vz + 0.5 - cz) - hz;
      const pz = qz > 0 ? qz * qz : 0;
      for (let vy = lo(cy, hy); vy <= hi(cy, hy); vy++) {
        const qy = Math.abs(vy + 0.5 - cy) - hy;
        const py = qy > 0 ? qy * qy : 0;
        const qyz = qy > qz ? qy : qz;
        const pyz = py + pz;
        let idx = voxelIndex(x0, vy, vz);
        for (let vx = x0; vx <= x1; vx++, idx++) {
          const qx = Math.abs(vx + 0.5 - cx) - hx;
          const outside = qx > 0 ? Math.sqrt(pyz + qx * qx) : Math.sqrt(pyz);
          const mx = qx > qyz ? qx : qyz;
          const d = outside + (mx < 0 ? mx : 0);
          // encodeDistance, inlined and with the clamp folded in.
          const c = d < -1 ? -1 : (d > 1 ? 1 : d);
          const b = ((c + 1) * 127.5 + 0.5) | 0;
          if (b < data[idx]) data[idx] = b;
        }
      }
    }

    // Counted from the box itself rather than by scanning the written band:
    // bands overlap between neighbouring blocks, boxes never do.
    filled += TEXELS_PER_BLOCK * height * TEXELS_PER_BLOCK;
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
