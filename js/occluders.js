import { GRID_DIM, DISTANCE_RANGE, minBoxVoxels, populateDistanceField,
         voxelIndex, gridIndex } from './boxgrid.js';
import { minSilhouetteVoxels, AXIS_X } from './silhouette.js';

// --- Dynamic occluders: the things that move and still cast ---
//
// The field is baked from the static World map, so until now nothing but terrain
// existed to a shadow ray. A character standing in the sun was lit from every
// direction and cast nothing, which is the most visible hole in the renderer and
// the one thing about it a player would notice first.
//
// The design doc's answer (S9) is a WORLD-LOCKED PROXY: a fixed-orientation
// volume, separate from the camera-facing display quad, and it is the proxy the
// grid ever sees. That separation is not tidiness - a billboard's "thickness" is
// camera-relative, so voxelising the quad itself would make a character's shadow
// change shape as the camera orbits, which reads as a bug rather than as a
// rotation. The proxy does not turn. Starting as an axis-aligned box: it is what
// minBoxVoxels already writes, and at 12.5 cm a 0.6 m character is five voxels
// across, so the silhouette has barely any shape to lose. An alpha-silhouette
// card is the refinement, not the requirement.
//
// --- Why this is affordable, and it is not the obvious reason ---
//
// A full C0 bake is 17 ms, so re-baking the field because someone took a step is
// not on the table. Two things make the incremental version cheap:
//
// 1. populateDistanceField ALREADY takes a rect list and restores exactly those
//    voxels to static truth - clear to FAR, min in every block whose band
//    overlaps. That is precisely "undo the occluder that was here", so no second
//    pristine copy of the field is needed. The same primitive the scrolled
//    re-origin uses, and the same one S4's deformation will use.
//
// 2. THE FIELD IS QUANTISED, SO A PROXY ONLY MOVES WHEN IT CROSSES A VOXEL.
//    Snap the box to the voxel grid and a character walking at 8 units/s changes
//    the field about once every voxel crossed, not once per frame, and a
//    character standing still changes it never. Standing still is the common
//    case in a turn-based tactical game - most of the time most combatants are
//    not the one moving.
//
//    This also falls out looking RIGHT rather than merely cheap. A snapped proxy
//    advances a texel at a time, so a character's shadow steps across the ground
//    on the same grid the terrain's shadow does, instead of sliding smoothly
//    underneath it. Two shadows in one image quantised differently is exactly
//    the mismatch texelLockTSL exists to avoid; this is that rule applied to the
//    caster instead of the receiver.
//
// --- What it actually costs, measured rather than estimated ---
//
// Standing still: nothing. Median frame time is identical with proxies on and
// off (8.3 ms either way over 120 frames), which is the snapping doing its job.
//
// Walking: median unchanged, p95 goes 8.4 -> 11.0 ms. So the frames that do
// rebake cost about 2.6 ms, and THAT IS NOT THE BAKE. A proxy plus its band is
// roughly 21 x 31 x 21 voxels, about 0.5% of the grid, so the restore and
// re-write together are a quarter of a millisecond. The rest is the texture
// upload: needsUpdate on a Data3DTexture re-sends all 3 MB, and there is no
// partial-region path through it.
//
// Two things already keep that in hand, and they are worth knowing before
// anyone optimises the bake by mistake:
//
//   the upload is conditional on applyOccluders reporting a change, per cascade,
//   which is why it returns a boolean at all;
//
//   the coarser cascades change far less often, for free. C1's voxels are 25 cm
//   and C2's are 50 cm, so a snapped box only moves in C2 once the character has
//   walked half a metre. The common changed frame re-uploads C0 alone.
//
// If this ever needs to be cheaper, the target is a partial 3D texture write,
// not the distance bake.

// A humanoid, in metres. Width is the footprint on both horizontal axes: the
// proxy does not turn, so it has no facing to be narrow along.
export const PROXY_WIDTH = 0.6;
export const PROXY_HEIGHT = 1.7;

// Occluders are identified rather than indexed, because the set changes: an
// entity dies, a barrel is destroyed, a summon appears. A Map keyed by id makes
// "what did this one look like last time" a lookup rather than a search, and
// makes a removal detectable as an id present last frame and absent now - which
// has to trigger a restore or the corpse keeps casting.
export function createOccluder(id, { width = PROXY_WIDTH,
                                     height = PROXY_HEIGHT,
                                     silhouettes = null } = {}) {
  // A silhouette is optional, and a proxy without one is still a box. That is
  // not a fallback for its own sake - a barrel or a crate has nothing to gain
  // from a cutout, and the box path is the cheaper of the two per voxel.
  //
  // An ARRAY, one entry per cascade, because a silhouette is expressed in the
  // voxels of the grid it was built for - see createSilhouetteSet. A single
  // shared one is not a smaller version of this, it is the wrong size in every
  // level but the first.
  return { id, x: 0, y: 0, z: 0, width, height, silhouettes, flip: false,
           enabled: true };
}

// The sprite's world position is its FEET - the quad is built standing on the
// ground - so the proxy's centre is half its height above it. Getting this wrong
// buries the box to the waist and the character casts a shadow from the knees
// down.
export function placeOccluder(o, x, footY, z, flip = false) {
  o.x = x;
  o.y = footY + o.height / 2;
  o.z = z;
  // The sprite mirrors when a character turns around, and the cutout has to
  // mirror with it or the shadow faces the wrong way while the art faces the
  // right way - which is more noticeable than either error alone.
  o.flip = flip;
  return o;
}

// The proxy as a box in this grid's voxel coordinates, snapped.
//
// Snapped on the CENTRE, not on the corners, so the box keeps its exact size as
// it moves. Snapping the extents instead would let a 5.0-voxel-wide proxy round
// to 4 or 6 depending on where it stood, and a character's shadow that breathed
// wider and narrower as they walked would be far more obvious than one that
// steps.
function boxInVoxels(grid, o, facing = AXIS_X) {
  const vs = grid.voxelSize;
  // The silhouette built for THIS cascade. Missing one falls back to the box
  // rather than to another level's - a box is the wrong shape but the right
  // size, and a wrong-size cutout is what put a seam across the ground.
  const sil = o.silhouettes ? (o.silhouettes[grid.level || 0] || null) : null;
  let hx, hy, hz;
  if (sil) {
    // The silhouette's own extent, which is already in voxels - it was reduced
    // to the grid's resolution when it was built, so there is nothing to convert
    // and nothing that can disagree about scale. Half-extents are needed here
    // only to size the restore rect; the SHAPE comes from the transform.
    //
    // Taken from the ROTATED card, matching minSilhouetteVoxels' own bounds. A
    // rect sized for an axis-aligned card would be too small for a card turned
    // 45 degrees, and the corners it missed would keep their old values - which
    // is the ghost bandRect exists to prevent, arriving from a new direction.
    const hu = sil.cols / 2, hn = sil.halfThickness;
    hy = sil.rows / 2;
    hx = hu * Math.abs(facing.ux) + hn * Math.abs(facing.uz);
    hz = hu * Math.abs(facing.uz) + hn * Math.abs(facing.ux);
  } else {
    hx = hz = o.width / 2 / vs;
    hy = o.height / 2 / vs;
  }
  const rx = (o.x - grid.origin.x) / vs;
  const ry = (o.y - grid.origin.y) / vs;
  const rz = (o.z - grid.origin.z) / vs;
  if (!sil) {
    return { cx: Math.round(rx), cy: Math.round(ry), cz: Math.round(rz),
             hx, hy, hz, flip: false, sil: null };
  }

  // SNAP TO THE LATTICE THE MASK'S PIXELS ACTUALLY SIT ON, which depends on
  // whether the mask is an odd or even number of voxels across.
  //
  // The bake evaluates at voxel centres, so the offsets it produces are
  // vx + 0.5 - c. The mask's own pixel centres are at (i + 0.5) - cols/2, which
  // are INTEGERS when cols is odd and HALF-INTEGERS when it is even. Rounding
  // the centre to a whole voxel regardless - which is right for a box, whose
  // extents are arbitrary anyway - lands every sample exactly half a voxel off
  // the pixel it was meant to read.
  //
  // For a wide shape that is a soft half-voxel smear. For a ONE-VOXEL-WIDE
  // feature it is the whole feature: a character's arm or a tree's trunk reads
  // from the empty column beside it and vanishes from the shadow, while the gap
  // next to it fills in. The cutout is mostly one-voxel features - that is what
  // distinguishes it from the box - so this is not a rounding nicety.
  const lat = (r, n) => {
    const off = (n & 1) ? 0.5 : 0;
    return Math.round(r - off) + off;
  };
  // The parity snap only means anything while the card IS axis-aligned - it
  // exists to make voxel centres coincide with mask pixel centres, and once the
  // card turns to follow the sun there is no alignment left to preserve. Off
  // axis the centre snaps to whole voxels instead, which still gives the
  // stepping (the shadow advances a texel at a time rather than sliding) without
  // pretending to a correspondence that is no longer there.
  //
  // Vertical is unconditional: v is world up whatever the card's azimuth, so the
  // mask's ROWS stay locked to voxel rows at every orientation.
  const ax = Math.abs(facing.ux) > 1 - 1e-6;   // card lies in XY
  const az = Math.abs(facing.uz) > 1 - 1e-6;   // card lies in ZY
  // The card is one voxel thick, so its normal axis is always the odd case.
  return {
    cx: ax ? lat(rx, sil.cols) : az ? lat(rx, 1) : Math.round(rx),
    cy: lat(ry, sil.rows),
    cz: az ? lat(rz, sil.cols) : ax ? lat(rz, 1) : Math.round(rz),
    hx, hy, hz,
    flip: !!o.flip,
    sil,
    // Carried so sameBox sees it: turning the sun turns every card, which
    // changes the shape in place without moving it. Without this the position
    // comparison reports nothing to do and every silhouette keeps the old sun's
    // orientation until its owner happens to take a step.
    ux: facing.ux, uz: facing.uz
  };
}

function sameBox(a, b) {
  return a.cx === b.cx && a.cy === b.cy && a.cz === b.cz &&
         a.hx === b.hx && a.hy === b.hy && a.hz === b.hz &&
         // Turning around changes the shape without moving it, so the position
         // comparison alone would report nothing to do and leave the old
         // silhouette in the field.
         a.flip === b.flip && a.sil === b.sil &&
         a.ux === b.ux && a.uz === b.uz;
}

// The rect a box's BAND covers, clipped to the grid.
//
// The band, not the box. A stored distance is the distance to the nearest
// surface, so writing a proxy changes voxels up to `range` away from it - and
// restoring it has to clear exactly as far, or a shell of "something solid is
// 30 cm that way" survives around where the character used to be. That ghost
// does not look like a leftover shadow; it looks like the ground has a dent in
// it, because a ray stepping through reads a short distance and creeps.
function bandRect(grid, b, range = grid.range || DISTANCE_RANGE) {
  const pad = Math.ceil(range) + 1;
  return {
    x0: Math.max(0, Math.floor(b.cx - b.hx - pad)),
    x1: Math.min(GRID_DIM, Math.ceil(b.cx + b.hx + pad) + 1),
    y0: Math.max(0, Math.floor(b.cy - b.hy - pad)),
    y1: Math.min(GRID_DIM, Math.ceil(b.cy + b.hy + pad) + 1),
    z0: Math.max(0, Math.floor(b.cz - b.hz - pad)),
    z1: Math.min(GRID_DIM, Math.ceil(b.cz + b.hz + pad) + 1)
  };
}

function emptyRect(r) {
  return r.x0 >= r.x1 || r.y0 >= r.y1 || r.z0 >= r.z1;
}

function rectsOverlap(a, b) {
  return a.x0 < b.x1 && b.x0 < a.x1 &&
         a.y0 < b.y1 && b.y0 < a.y1 &&
         a.z0 < b.z1 && b.z0 < a.z1;
}

// --- The per-frame update ---
//
// Returns true when the field changed, which is the caller's signal to re-upload
// the texture. Returning false on a still frame is the whole point of the
// snapping above: a tactical scene where nobody is mid-move costs one Map walk
// and no writes at all.
// `facing` is the card orientation every silhouette uses this frame - the sun's
// azimuth, in practice. It is an argument rather than per-occluder state because
// it is a property of the LIGHT, and one proxy facing a different way from
// another would be a bug with no physical reading.
export function applyOccluders(grid, occluders, facing = AXIS_X) {
  const prev = grid.occluderBoxes || new Map();
  const now = new Map();
  for (const o of occluders) {
    if (o.enabled === false) continue;
    now.set(o.id, boxInVoxels(grid, o, facing));
  }

  // What has to be restored to static truth this frame. Three sources, and
  // missing any one of them leaves a ghost:
  const rects = [];
  const dirty = new Set();

  //   1. moved or appeared - the place it is going, and the place it left.
  for (const [id, b] of now) {
    const p = prev.get(id);
    if (p && sameBox(p, b)) continue;
    dirty.add(id);
    rects.push(bandRect(grid, b));
    if (p) rects.push(bandRect(grid, p));
  }
  //   2. gone - present last frame, absent now. Nothing else would clear it.
  for (const [id, p] of prev) {
    if (!now.has(id)) rects.push(bandRect(grid, p));
  }
  //   3. scrolled over. The re-origin bakes the strip that slid into the
  //      footprint, which clears it to static truth and destroys the imprint of
  //      any proxy standing there - WITHOUT that proxy having moved, so the
  //      comparison above sees nothing wrong and would leave the character
  //      shadowless until its next step. The strips are recorded by the grid for
  //      exactly this.
  const strips = grid.scrollStrips;
  if (strips && strips.length) {
    for (const [id, b] of now) {
      if (dirty.has(id)) continue;
      const r = bandRect(grid, b);
      if (strips.some(s => rectsOverlap(r, s))) { dirty.add(id); rects.push(r); }
    }
    grid.scrollStrips = null;
  }

  const live = rects.filter(r => !emptyRect(r));
  if (live.length === 0) {
    grid.occluderBoxes = now;
    return false;
  }

  // Restore, then re-write. Every proxy is re-min'd over the whole rect set, not
  // just the ones that moved: the restore is indiscriminate, so a stationary
  // character standing next to a moving one has had its imprint wiped by the
  // mover's rect and has to be put back. Clipping does the filtering - a proxy
  // whose band misses every rect writes nothing.
  populateDistanceField(grid, live);
  for (const b of now.values()) {
    if (b.sil) {
      minSilhouetteVoxels(grid, b.sil, b.cx, b.cy, b.cz, live, b.flip,
                          { ux: b.ux, uz: b.uz }, grid.range);
    } else {
      minBoxVoxels(grid, b.cx, b.cy, b.cz, b.hx, b.hy, b.hz, live, grid.range);
    }
  }

  grid.occluderBoxes = now;
  return true;
}

// Read back what the field holds at a world point, for tests and for bxb.
export function occluderVoxelByte(grid, vx, vy, vz) {
  if (vx < 0 || vy < 0 || vz < 0 ||
      vx >= GRID_DIM || vy >= GRID_DIM || vz >= GRID_DIM) return null;
  return grid.data[gridIndex(grid, vx, vy, vz)];
}
