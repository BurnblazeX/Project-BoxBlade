// --- Regions: how the world is stored (World plan §0.2a) ---
//
// Minecraft's split: blocks in chunks, chunks in regions. A region is a fixed
// square of REGION_CHUNKS x REGION_CHUNKS chunks (32 x 32, 384 x 384 blocks)
// on the terrain master, whose pixel (x, y) is chunk (x, y), so region (0, 0)
// starts at its top-left. One file per region, a .bxr, will hold a record per
// chunk that differs from the layer below it, behind an offset table of
// REGION_CHUNKS² slots (chunkSlot). A chunk loads from the topmost layer that
// has it:
//   save       the chunks changed in play (part of a save)
//   authored   the chunks built by hand (shipped with the game)
//   generated  the seed and the terrain master (never stored)
// Zones (core/zone.js) are what the player loads, behind a loading screen; they
// own no blocks, and will read the chunks they use from here. Nothing reads or
// writes region files yet: this is the grid they will be on.

export const REGION_CHUNKS = 32;

// The region chunk (cx, cz) is in: [rx, rz]. Negative chunks round down.
export function regionOf(cx, cz) {
  return [Math.floor(cx / REGION_CHUNKS), Math.floor(cz / REGION_CHUNKS)];
}

// The chunk's slot in its region's offset table, 0 to REGION_CHUNKS² - 1:
// x first, then z.
export function chunkSlot(cx, cz) {
  const x = cx - Math.floor(cx / REGION_CHUNKS) * REGION_CHUNKS;
  const z = cz - Math.floor(cz / REGION_CHUNKS) * REGION_CHUNKS;
  return x + REGION_CHUNKS * z;
}

// The chunks region (rx, rz) covers: [cx0, cx1, cz0, cz1], inclusive.
export function regionChunks(rx, rz) {
  const cx0 = rx * REGION_CHUNKS, cz0 = rz * REGION_CHUNKS;
  return [cx0, cx0 + REGION_CHUNKS - 1, cz0, cz0 + REGION_CHUNKS - 1];
}

// The region's file name, Minecraft's pattern: r.<rx>.<rz>.bxr.
export const regionFile = (rx, rz) => `r.${rx}.${rz}.bxr`;
