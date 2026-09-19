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
  populateOccupancy(grid);
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

export function isOccupied(grid, vx, vy, vz) {
  if (!inBounds(vx, vy, vz)) return false;
  return grid.data[voxelIndex(vx, vy, vz)] !== 0;
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
export function populateOccupancy(grid) {
  // Clear only the ranges filled last time. A blanket fill(0) memsets ~3 MB on
  // every re-origin, which in explore mode is every block the player steps.
  const previous = grid.filledBlocks;
  if (previous) {
    for (let k = 0; k < previous.length; k += 3) {
      const vx0 = previous[k], vy0 = previous[k + 1], vz0 = previous[k + 2];
      for (let dz = 0; dz < TEXELS_PER_BLOCK; dz++) {
        for (let dy = 0; vy0 + dy < GRID_DIM && dy < TEXELS_PER_BLOCK; dy++) {
          const base = voxelIndex(vx0, vy0 + dy, vz0 + dz);
          grid.data.fill(0, base, base + TEXELS_PER_BLOCK);
        }
      }
    }
  } else {
    grid.data.fill(0);
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
    // voxel boundary with nothing left over.
    const supported = World.has(getVoxelKey(bx, by - 1, bz));
    const dyStart = supported ? 0 : TEXELS_PER_BLOCK / 2;
    blocks.push(vx0, vy0 + dyStart, vz0);

    for (let dz = 0; dz < TEXELS_PER_BLOCK; dz++) {
      for (let dy = dyStart; dy < TEXELS_PER_BLOCK; dy++) {
        const rowBase = voxelIndex(vx0, vy0 + dy, vz0 + dz);
        grid.data.fill(1, rowBase, rowBase + TEXELS_PER_BLOCK);
        filled += TEXELS_PER_BLOCK;
      }
    }
  }

  grid.occupiedCount = filled;
  grid.filledBlocks = blocks;
  return filled;
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
