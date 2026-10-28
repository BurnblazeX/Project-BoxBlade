// --- Windows that follow the player (World plan P2a) ---
//
// No renderer volume covers a whole region: each is a fixed number of blocks
// across, placed around the player, so a region can be any size (the user's
// may be 384 or 512 a side). The field and the LPV have always followed;
// the block volume and the box light use this, and the sun map has its own
// version of the same idea in core/sunmap.js.
//
// One axis at a time, in blocks. A world that fits in the window gets it
// centred on itself and never moves - a small level behaves exactly as it did
// when the volumes were fitted to it. A bigger one keeps the window while the
// focus stays in its middle (margin blocks in from either side) and recentres
// on the focus, snapped to stride, when it leaves - so walking around costs a
// rebuild every so often, never every block.

// The world's blocks' bounds, { min: [x, y, z], max: [x, y, z] } in blocks;
// null for an empty world.
export function worldBlockBounds(world) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const key of world.keys()) {
    const c = key.split(',').map(Number);
    for (let a = 0; a < 3; a++) {
      if (c[a] < min[a]) min[a] = c[a];
      if (c[a] > max[a]) max[a] = c[a];
    }
  }
  return min[0] > max[0] ? null : { min, max };
}

// The window's first block on one axis.
//   prev       its first block now, or null
//   focus      the block the player is on
//   size       blocks across
//   lo, hi     the world's first and last block on this axis (pad included)
export function windowAxis(prev, focus, size, lo, hi, margin = Math.floor(size / 4), stride = 8) {
  if (hi - lo + 1 <= size) return Math.round((lo + hi) / 2 - (size - 1) / 2);
  if (prev !== null && prev !== undefined && focus >= prev + margin && focus < prev + size - margin) return prev;
  return Math.round((focus - size / 2) / stride) * stride;
}

// Both horizontal axes: [x, z] of the window's first block. bounds from
// worldBlockBounds (null: an empty world, a window on the focus); pad blocks
// of air either side of the world count as part of it.
export function followWindow(prev, focus, size, bounds, { pad = 0, margin, stride } = {}) {
  const b = bounds || { min: [focus.x, 0, focus.z], max: [focus.x, 0, focus.z] };
  return [0, 2].map((a, i) => windowAxis(prev ? prev[i] : null, a === 0 ? focus.x : focus.z, size,
                                         b.min[a] - pad, b.max[a] + pad, margin, stride));
}
