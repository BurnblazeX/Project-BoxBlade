// --- The block volume's bytes, laid out as a ring (World plan P2a) ---
//
// render.js keeps a 128 x 48 x 128 window of the world's blocks on the GPU -
// each block's layers and shape, and a model cell's hit-table entry - which
// follows the player over a region wider than it (core/follow.js). Until
// 2026-10-29 every move of the window rebuilt and re-uploaded all of it, 6 MB,
// a hitch each time. Now x and z are a ring: block (x, z) always sits at
// (x mod size, z mod size), wherever the window is, so a move rewrites only
// the strips that slid in and uploads only those (render.js
// followBlockVolume), as the shadow field's slides do. y is not a ring: the
// window spans every level.
//
// The texture is size[2] * 2 deep: a dual block's second half sits size[2]
// further along z (gpu/volume.js blockVolumeTSL, its `second`). The shader
// reads the same ring, and calls anything outside [lo, lo + size) air.
//
// A cell, 4 bytes (gpu/volume.js reads them): R side layer + 1 (0 air), G top
// layer, B bottom layer, A the shape code (+128 for a dual block) - or, in a
// model's cell, G + 256 B its modelHits header and A its GI entry + 1.
// The sky tops (core/skytop.js) are a 2D ring of the same columns, two bytes
// each.

import { blocksIn } from './chunks.js';
import { materialLayer, MATERIAL_LAYERS } from './materials.js';
import { shapeCode, isDual } from './world.js';
import { buildSkyTops } from './skytop.js';

// v's place on a ring of n, for any integer v.
export const ring = (v, n) => ((v % n) + n) % n;

// The byte offset of block (x, y, z) in the volume's data; second: its
// dual half's.
export function volumeOffset(lo, size, x, y, z, second = false) {
  return (((ring(z, size[2]) + (second ? size[2] : 0)) * size[1] + (y - lo[1])) * size[0] + ring(x, size[0])) * 4;
}

// The parts of the window at lo that the window at prev did not cover, as
// boxes of columns [x0, x1, z0, z1] (inclusive, world blocks) - the strips a
// move brings in. null when the two do not overlap at all: everything is new.
export function enteredBoxes(prev, lo, size) {
  const [sx, , sz] = size;
  const dx = lo[0] - prev[0], dz = lo[2] - prev[2];
  if (Math.abs(dx) >= sx || Math.abs(dz) >= sz) return null;
  const boxes = [];
  const z0 = lo[2], z1 = lo[2] + sz - 1;
  if (dx > 0) boxes.push([prev[0] + sx, lo[0] + sx - 1, z0, z1]);
  if (dx < 0) boxes.push([lo[0], prev[0] - 1, z0, z1]);
  // The z strip only over the x the old window had: the x strip has the rest.
  const x0 = Math.max(lo[0], prev[0]), x1 = Math.min(lo[0], prev[0]) + sx - 1;
  if (dz > 0) boxes.push([x0, x1, prev[2] + sz, lo[2] + sz - 1]);
  if (dz < 0) boxes.push([x0, x1, lo[2], prev[2] - 1]);
  return boxes;
}

// A run of world blocks a..b on a ring of n as runs of physical places: one,
// or two where it wraps. [[first, last], ...].
export function ringRuns(a, b, n) {
  const p = ring(a, n), len = b - a + 1;
  return p + len <= n ? [[p, p + len - 1]] : [[p, n - 1], [0, p + len - n - 1]];
}

// Writes the window's cells over box [x0, x1, z0, z1] (inside the window):
// cleared, then the world's blocks there, the model cells' hit entries (hits:
// bbmodel.js buildModelHits, over every model - the same table wherever the
// window is), and the columns' sky tops. vol: { data, skyTop, lo, size } -
// data the volume's bytes, skyTop the sky-top texture's.
// cleared: the caller has zeroed everything already (a full build).
export function writeVolumeBox(vol, world, hits, box, cleared = false) {
  const { data, skyTop, lo, size } = vol;
  const [x0, x1, z0, z1] = box;
  const w = x1 - x0 + 1, d = z1 - z0 + 1;
  if (!cleared) {
    for (let z = z0; z <= z1; z++) {
      for (let x = x0; x <= x1; x++) {
        for (const second of [false, true]) {
          for (let y = lo[1]; y < lo[1] + size[1]; y++) {
            const o = volumeOffset(lo, size, x, y, z, second);
            data[o] = data[o + 1] = data[o + 2] = data[o + 3] = 0;
          }
        }
      }
    }
  }
  for (const [[x, y, z], block] of blocksIn(world, box)) {
    if (y < lo[1] || y >= lo[1] + size[1]) continue;
    const o = volumeOffset(lo, size, x, y, z);
    const layer = materialLayer(block.materialId), f = MATERIAL_LAYERS.faces[layer];
    data[o] = layer + 1;
    data[o + 1] = f ? f.top : layer;
    data[o + 2] = f ? f.bottom : layer;
    data[o + 3] = shapeCode(block);   // its shape and facing (world.js shapeCode)
    if (isDual(block)) {
      data[o + 3] += 128;
      const o2 = volumeOffset(lo, size, x, y, z, true);
      const layer2 = materialLayer(block.second), f2 = MATERIAL_LAYERS.faces[layer2];
      data[o2] = layer2 + 1;
      data[o2 + 1] = f2 ? f2.top : layer2;
      data[o2 + 2] = f2 ? f2.bottom : layer2;
    }
  }
  for (const [key, { header, giIndex }] of hits.index) {
    const [x, y, z] = key.split(',').map(Number);
    if (x < x0 || x > x1 || z < z0 || z > z1 || y < lo[1] || y >= lo[1] + size[1] || header > 65535) continue;
    const o = volumeOffset(lo, size, x, y, z);
    data[o + 1] = header & 255;
    data[o + 2] = header >> 8;
    data[o + 3] = giIndex + 1;
  }
  const tops = buildSkyTops(world, [x0, z0], [w, d]);
  for (let cz = 0; cz < d; cz++) {
    for (let cx = 0; cx < w; cx++) {
      const code = tops[cz * w + cx];
      const i = (ring(z0 + cz, size[2]) * size[0] + ring(x0 + cx, size[0])) * 2;
      skyTop[i] = code & 255;
      skyTop[i + 1] = code >> 8;
    }
  }
}
