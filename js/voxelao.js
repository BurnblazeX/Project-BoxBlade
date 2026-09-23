// --- Static voxel AO: the shadows-off occlusion ---
//
// With shadows on, AO is six cones traced through the distance field
// (gpu.js createAOTSL). With shadows off there is no field to trace - and on
// low-power machines, no budget for one - so the terrain takes the classic
// voxel AO instead: for a texel on a block face, the blocks around that face
// in the layer in front of it decide how enclosed each corner is, and the
// corners blend across the face.
//
// No marches and nothing baked: eight occupancy reads per texel, from the
// block-material volume the renderer already keeps. Static in the sense that
// matters - it changes only when blocks do.
//
// CPU and pure, so it is tested headlessly; gpu.js createVoxelAONode is the
// shader mirror.

// How dark the most enclosed corner gets: 0 would be black. The curve between
// is linear in the corner's open count (0-3 of 3).
export const VOXEL_AO_MIN = 0.45;

// One corner of a face. side1 and side2 are the two edge neighbours that meet
// at it, corner the diagonal one, each 0 or 1 (occupied). Both sides filled is
// fully enclosed whatever the corner holds - the diagonal cannot be seen past
// them. Returns openness, 0..1.
export function cornerAO(side1, side2, corner) {
  if (side1 && side2) return 0;
  return (3 - (side1 + side2 + corner)) / 3;
}

// A face's occlusion at (a, b), each 0..1 across the face along its two
// tangent axes. occ(i, j) is the occupancy of the block at offset (i, j),
// i, j in -1..1, in the layer in front of the face. Bilinear between the
// four corners, then mapped onto VOXEL_AO_MIN..1.
export function faceAO(occ, a, b) {
  const c00 = cornerAO(occ(-1, 0), occ(0, -1), occ(-1, -1));
  const c10 = cornerAO(occ(1, 0), occ(0, -1), occ(1, -1));
  const c01 = cornerAO(occ(-1, 0), occ(0, 1), occ(-1, 1));
  const c11 = cornerAO(occ(1, 0), occ(0, 1), occ(1, 1));
  const v = (c00 * (1 - a) + c10 * a) * (1 - b) + (c01 * (1 - a) + c11 * a) * b;
  return VOXEL_AO_MIN + (1 - VOXEL_AO_MIN) * v;
}
