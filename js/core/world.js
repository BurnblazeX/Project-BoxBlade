import { validExit } from './exits.js';
import { isGlassMaterial } from './materials.js';
import { placeModel } from './bbmodel.js';
import { rectCovered } from './rects.js';
import { createChunkWorld } from './chunks.js';
import { LEVEL_FORMAT, LEVEL_VERSION, migrateLevel, blockRow, chunkBounds, unionBounds } from './region.js';

// --- CHUNK / VERTICAL LAYOUT ---
// A chunk is 12x12 blocks in footprint and 48 blocks tall: 8 below ground,
// ground level itself, and 39 above (60 m) - room for a church on its hill,
// the cliff under it and the riverbed (World plan P1). Ground is y = 0, so the
// vertical span is -8 .. 39 inclusive. Until 2026-10-28 it was -3 .. 8, 12
// levels, the height of one field cube; the field now follows the player up
// and down instead (core/field.js gridOriginFor).
// One block is 1.5 m on a side. This is a property of the world, not of the
// renderer, and core/field.js needs it without pulling Three.js in - so it lives
// here and render.js aliases it as VOXEL_SIZE.
export const BLOCK_METRES = 1.5;
export const CHUNK_SIZE = 12;
export const GROUND_Y = 0;
// Y_MIN/Y_MAX are the AUTHORABLE vertical range, not a volume that gets filled.
// This is not a sandbox game: there is no terrain below the surface unless a
// map actually goes down there, in which case the author places floors and
// walls explicitly. Nothing is generated underground.
export const Y_MIN = -8;
export const Y_MAX = 39;
export const CHUNK_HEIGHT = Y_MAX - Y_MIN + 1; // 48

// The world's blocks, by chunk (core/chunks.js, World plan P3). Answers as the
// Map from "x,y,z" it replaced, plus getAt / hasAt / forEachBlockIn. Blocks it
// hands back are shared, frozen block TYPES: change a cell with World.set,
// never by writing to the block.
export const World = createChunkWorld({ size: CHUNK_SIZE, yMin: Y_MIN, height: CHUNK_HEIGHT });

// --- Who stands where ---
//
// A cell's occupant (an entity's id, a prop's 'prop:<model>') used to be a
// field on its block. The block objects are shared types now, so occupancy is
// its own table, by voxel key: a few entries, not one per block. A cell
// emptied (World.delete) or a world cleared takes its occupants with it.
const Occupants = new Map();
export function occupantAt(key) {
  return Occupants.get(key) ?? null;
}
export function setOccupantAt(key, id) {
  if (id === null || id === undefined) Occupants.delete(key);
  else Occupants.set(key, id);
}
// Off a cell, only if it is the occupant there.
export function vacateAt(key, id) {
  if (Occupants.get(key) === id) Occupants.delete(key);
}
{
  const clear = World.clear, del = World.delete;
  World.clear = () => { clear(); Occupants.clear(); };
  World.delete = key => { Occupants.delete(key); return del(key); };
}

export function getVoxelKey(x, y, z) {
  return `${Math.round(x)},${Math.round(y)},${Math.round(z)}`;
}

// --- SOLID / STANDABLE QUERIES ---
// World holds SOLID blocks. An entity's gridPos.y is the y of the block it is
// standing ON, and its sprite sits on that block's top face - the convention
// the renderer and sprite positioning already used implicitly when everything
// lived at y = 0.
//
// "Standable" is therefore a derived property, not a stored one: a block can be
// stood on when it is solid, its own material permits it, and the cell directly
// above is air (headroom). Keeping it derived means digging a block out
// automatically exposes the one beneath, with no flag to keep in sync.

export function getBlock(x, y, z) {
  return World.getAt(Math.round(x), Math.round(y), Math.round(z));
}

export function isSolid(x, y, z) {
  return World.hasAt(Math.round(x), Math.round(y), Math.round(z));
}

// --- Block shapes ---
//
// A block fills its whole cell, or part of it: block.shape, absent for a full
// block. Every shape is a list of axis-aligned boxes on the 12-voxel grid, so
// every face of every shape lies on a voxel boundary and on its axis - the
// texel lock, the face uv, the texel cache's ids and the snapped hit normals
// hold as they are. The renderer draws each box as an instance of the one
// terrain box, the field bakes each, and nothing downstream knows the shape.
//
// Half tiles: the lower or upper half of the cell, 0.75 m (six texels).
// Vertical halves: the east half for facing E, turned like the slopes. Panes
// (middleVertical): four texels thick through the cell's middle, full width
// and height, facing E across x. Both stand at full height, like a full block.
//
// Slopes are SEEN as planes and BAKED as staircases (doc §0). The renderer
// draws the plane - the terrain box with its downhill top edge dropped, shaded
// on the plane's own texel grid (gpu/terrain.js terrainSurfaceTSL). The boxes here are
// what everything else sees: the field, and so shadows, AO, GI and what
// reflections hit. They are a staircase of 12 steps one texel deep, INSCRIBED
// under the plane - each step's top meets it at the step's downhill edge and
// falls below it across the step - so the plane is never inside its own steps
// and never shadows itself. A slope climbs the whole cell, a half slope half
// of it, and a high half slope the upper half above a half block - so a half
// slope and a high one in a row climb a full block at 26.6 degrees.
// block.facing (N, E, S, W) is the uphill direction; the boxes below are for
// E and turned to the others.
//
// Glass takes the one-box shapes (glass: true below - halves, vertical
// halves, panes); its walk (core/field.js walkThroughGlass, and the shader's)
// steps from box face to box face. A glass slope or stairs is whole.

// Steps are one texel deep: 12 to the block. Step k's top is the plane's
// height at its downhill edge; a first step of no height is left out.
const T = 12;
const steps = (rise, base) => Array.from({ length: T }, (_, k) =>
  [k / T, (k + 1) / T, 0, base + rise * k / T, 0, 1]).filter(b => b[3] > 0);
// { boxes: [x0, x1, y0, y1, z0, z1] in cell fractions, facing E;
//   stand: where an entity stands, as a fraction of the cell's height;
//   low, high: a slope's height at its downhill and uphill edges;
//   turns: turned by block.facing (hasFacing) }
export const BLOCK_SHAPES = {
  full:          { boxes: [[0, 1, 0, 1, 0, 1]],   stand: 1, glass: true },
  halfBottom:    { boxes: [[0, 1, 0, 0.5, 0, 1]], stand: 0.5, glass: true },
  halfTop:       { boxes: [[0, 1, 0.5, 1, 0, 1]], stand: 1, glass: true },
  halfVertical:  { boxes: [[0.5, 1, 0, 1, 0, 1]], stand: 1, turns: true, glass: true },
  middleVertical: { boxes: [[4 / 12, 8 / 12, 0, 1, 0, 1]], stand: 1, turns: true, glass: true },
  slope:         { boxes: steps(1, 0),     stand: 0.5,  low: 0,   high: 1,   slope: true, turns: true },
  halfSlope:     { boxes: steps(0.5, 0),   stand: 0.25, low: 0,   high: 0.5, slope: true, turns: true },
  halfSlopeHigh: { boxes: steps(0.5, 0.5), stand: 0.75, low: 0.5, high: 1,   slope: true, turns: true },
  // Stairs, as Minecraft's: a lower half slab and, on it, the uphill half of
  // the upper half. Boxes on the texel grid, drawn and baked as they are - no
  // plane. Two treads: the downhill half at half height, the uphill half at
  // full; a character steps from one to the other (groundHeightAt).
  stairs:        { boxes: [[0, 1, 0, 0.5, 0, 1], [0.5, 1, 0.5, 1, 0, 1]],
                   stand: 0.75, low: 0.5, high: 1, treads: true, turns: true }
};
// --- Buried faces ---
//
// Which of a drawn box's faces can be seen at all: a face is buried when the
// opaque boxes just past it cover all of it - the next cell's from its near
// side for a face on the cell's boundary, this cell's others for one inside
// it (stairs' upper step on its lower, a dual block's two halves). So two
// half slabs side by side, a slab on a slab, a row of panes or vertical
// halves hide the faces between them, and a face only partly covered is
// drawn whole. The terrain draws a buried face as nothing (render.js
// terrainShapeTSL). Raster only - the field, the block volume, mirrors and
// picking all read the world, not these triangles, so shadows, GI and
// reflections are untouched. A slope's top is its plane, which drops away
// from the cell's top, so it is never buried.
// Bits d: +x, -x, +y, -y, +z, -z (BoxGeometry's face order); set = drawn.
export const FACE_STEPS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
export const ALL_FACES = 63;
// The boxes in a block that hide a face pressed against them, cell
// fractions: every opaque part's (glass hides nothing - it is seen through),
// a slope only its floor (its sides are a profile, its top a plane).
const SLOPE_FLOOR = [[0, 1, 0, 0, 0, 1]];
export function opaqueBoxes(block) {
  if (!block) return [];
  if (isSlope(block)) return isGlassMaterial(block.materialId) ? [] : SLOPE_FLOOR;
  const out = [];
  for (const part of blockParts(block)) if (!isGlassMaterial(part.materialId)) out.push(...part.boxes);
  return out;
}
// The two in-face axes of a face along axis a.
const FACE_PLANE = [[1, 2], [0, 2], [0, 1]];
// box: [x0, x1, y0, y1, z0, z1] in cell fractions; boxesAt(x, y, z): the
// boxes in that cell that hide faces (opaqueBoxes).
export function exposedFaceMask(x, y, z, box, slope, boxesAt) {
  let mask = 0;
  FACE_STEPS.forEach((step, d) => {
    if (slope && d === 2) { mask |= 1 << d; return; }
    const a = step.findIndex(v => v !== 0), s = step[a];
    const [u, v] = FACE_PLANE[a];
    const at = box[2 * a + (s > 0 ? 1 : 0)];
    const edge = s > 0 ? at === 1 : at === 0;
    const others = edge ? boxesAt(x + step[0], y + step[1], z + step[2]) : boxesAt(x, y, z);
    // Past the face: a box whose near side lies on its plane.
    const near = edge ? (s > 0 ? 0 : 1) : at;
    const covers = others.filter(o => o[2 * a + (s > 0 ? 0 : 1)] === near)
      .map(o => [o[2 * u], o[2 * u + 1], o[2 * v], o[2 * v + 1]]);
    if (!rectCovered([box[2 * u], box[2 * u + 1], box[2 * v], box[2 * v + 1]], covers)) mask |= 1 << d;
  });
  return mask;
}

// Uphill, as (x, z).
export const FACINGS = { N: [0, -1], E: [1, 0], S: [0, 1], W: [-1, 0] };

export function blockShape(block) {
  const s = block && BLOCK_SHAPES[block.shape] ? block.shape : 'full';
  return block && isGlassMaterial(block.materialId) && !BLOCK_SHAPES[s].glass ? 'full' : s;
}

// A slope: drawn as a plane over its baked staircase (render.js).
export function isSlope(block) {
  return BLOCK_SHAPES[blockShape(block)].slope === true;
}

// A shape that turns with block.facing: the slopes, the stairs, the vertical
// halves and panes.
export function hasFacing(shape) {
  return !!BLOCK_SHAPES[shape] && BLOCK_SHAPES[shape].turns === true;
}

// A shape of more than one box (slopes' staircases, stairs): what the mirror
// rectangles, which take whole faces, leave out.
export function isCompound(block) {
  return blockBoxes(block).length > 1;
}

export function blockFacing(block) {
  return block && FACINGS[block.facing] ? block.facing : 'E';
}

// A shape's boxes turned to a facing: W mirrors x, S swaps x and z, N swaps
// and mirrors. Cached per shape and facing.
const boxCache = new Map();
export function blockBoxes(block) {
  const shape = blockShape(block);
  const facing = hasFacing(shape) ? blockFacing(block) : 'E';
  const k = shape + facing;
  let out = boxCache.get(k);
  if (!out) {
    // x and z snapped to the texel grid, so a turned face lands on exactly
    // the plane an unturned one does (1 - 8/12 is not 4/12 in floats) and
    // faces compare equal (exposedFaceMask, mirrors.js). Not y: a half
    // slope's steps rise in 24ths.
    const snap = b => b.map((c, k) => (k === 2 || k === 3 ? c : Math.round(c * 12) / 12));
    out = BLOCK_SHAPES[shape].boxes.map(([x0, x1, y0, y1, z0, z1]) => {
      if (facing === 'W') return snap([1 - x1, 1 - x0, y0, y1, z0, z1]);
      if (facing === 'S') return [z0, z1, y0, y1, x0, x1];
      if (facing === 'N') return snap([z0, z1, y0, y1, 1 - x1, 1 - x0]);
      return [x0, x1, y0, y1, z0, z1];
    });
    boxCache.set(k, out);
  }
  return out;
}

// --- What the rays see of a cell (render.js block volume, gpu/volume.js) ---
//
// A block's shape and facing as one byte, the block volume's A: 0 full (and
// stairs - the rays take them whole), a slope kind * 4 + facing (4..15, as
// render.js slopeCode), 16 / 17 the low / high half, 20 + facing a vertical
// half, 24 + facing a pane - facing 0 N, 1 E, 2 S, 3 W.
const FACING_CODE = { N: 0, E: 1, S: 2, W: 3 };
const CODE_FACING = ['N', 'E', 'S', 'W'];
const SLOPE_CODE = { slope: 1, halfSlope: 2, halfSlopeHigh: 3 };
export function shapeCode(block) {
  const shape = blockShape(block), f = FACING_CODE[blockFacing(block)];
  if (SLOPE_CODE[shape]) return SLOPE_CODE[shape] * 4 + f;
  if (shape === 'halfBottom') return 16;
  if (shape === 'halfTop') return 17;
  if (shape === 'halfVertical') return 20 + f;
  if (shape === 'middleVertical') return 24 + f;
  return 0;
}
// The box the rays take for a code, in cell fractions: one box, the whole
// cell for anything not listed above (slopes and stairs as solid).
export function codeBox(code) {
  const shape = code === 16 ? 'halfBottom' : code === 17 ? 'halfTop'
              : code >= 20 && code < 24 ? 'halfVertical' : code >= 24 && code < 28 ? 'middleVertical' : 'full';
  return blockBoxes({ shape, facing: CODE_FACING[code & 3] })[0];
}
// A dual block: a half (low, high or vertical) and, in the rest of its cell,
// a second material's half (block.second). It fills its cell, so the game
// takes it as a full block - standing, headroom, what is solid; only what is
// drawn, baked and seen by the rays tells the halves apart (blockParts).
export function isDual(block) {
  return !!block && block.second != null && !!complementShape(blockShape(block), blockFacing(block));
}
// What a block is made of, part by part: [{ materialId, boxes }] - its own
// shape, then a dual block's other half.
export function blockParts(block) {
  const parts = [{ materialId: block.materialId, boxes: blockBoxes(block) }];
  if (isDual(block)) {
    const other = complementShape(blockShape(block), blockFacing(block));
    parts.push({ materialId: block.second, boxes: blockBoxes({ materialId: block.second, ...other }) });
  }
  return parts;
}
// The other half of a half shape - what a dual block's second material fills
// (block.second): low and high, a vertical half and the one facing away.
// null for a shape with no other half.
const OPPOSITE = { N: 'S', S: 'N', E: 'W', W: 'E' };
export function complementShape(shape, facing) {
  if (shape === 'halfBottom') return { shape: 'halfTop', facing: null };
  if (shape === 'halfTop') return { shape: 'halfBottom', facing: null };
  if (shape === 'halfVertical') return { shape, facing: OPPOSITE[facing] || 'W' };
  return null;
}
// What fills the world point (metres) for the rays: { cell, materialId, box }
// with box in cell fractions - the block's own box, or a dual block's other
// half - or null for air. Half-open as isSolidAt; a slope or stairs whole.
export function partAt(wx, wy, wz, world = World) {
  const cell = [wx, wy, wz].map(v => Math.round(v / BLOCK_METRES));
  const block = world.get(cell.join(','));
  if (!block) return null;
  const f = [wx, wy, wz].map((v, a) => v / BLOCK_METRES - cell[a] + 0.5);
  const inside = b => f[0] >= b[0] && f[0] < b[1] && f[1] >= b[2] && f[1] < b[3] && f[2] >= b[4] && f[2] < b[5];
  const box = codeBox(shapeCode(block));
  if (inside(box)) return { cell, materialId: block.materialId, box };
  if (isDual(block)) {
    const ob = blockParts(block)[1].boxes[0];
    if (inside(ob)) return { cell, materialId: block.second, box: ob };
  }
  return null;
}

// The vertical span a block fills, [lo, hi] in blocks from the cell's floor -
// its boxes' bounds, or for a slope its PLANE's: from the floor to the top of
// its head, which its staircase (inscribed under the plane) stops a texel
// short of. What the slope is drawn over, and its headroom.
export function blockSpan(block) {
  if (isDual(block)) return [0, 1];
  const shape = BLOCK_SHAPES[blockShape(block)];
  if (shape.slope) return [0, shape.high];
  let lo = 1, hi = 0;
  for (const b of blockBoxes(block)) { if (b[2] < lo) lo = b[2]; if (b[3] > hi) hi = b[3]; }
  return [lo, hi];
}

// A slope's surface level at a point of its cell (fractions fx, fz), as a
// fraction of the cell's height: the plane that is drawn, not the steps.
export function slopeLevel(block, fx, fz) {
  const shape = BLOCK_SHAPES[blockShape(block)];
  const [dx, dz] = FACINGS[blockFacing(block)];
  const u = Math.min(1, Math.max(0, (fx - 0.5) * dx + (fz - 0.5) * dz + 0.5));
  return shape.low + (shape.high - shape.low) * u;
}

// Whether a world point (metres) lies inside a block's solid part. Blocks
// are centred on b * BLOCK_METRES, so rounding finds the cell. Half-open like
// the rounding: a point on a top face (a sprite's feet) is outside. A slope
// is solid under its plane, less slopeClearance (a fraction of a block) -
// the room a sprite's feet get on it.
export function isSolidAt(wx, wy, wz, slopeClearance = 0) {
  const bx = Math.round(wx / BLOCK_METRES), by = Math.round(wy / BLOCK_METRES);
  const bz = Math.round(wz / BLOCK_METRES);
  const block = getBlock(bx, by, bz);
  if (!block) return false;
  const fx = wx / BLOCK_METRES - bx + 0.5, fy = wy / BLOCK_METRES - by + 0.5;
  const fz = wz / BLOCK_METRES - bz + 0.5;
  if (isSlope(block)) return fy < slopeLevel(block, fx, fz) - slopeClearance;
  if (isDual(block)) return true;
  return blockBoxes(block).some(b => fx >= b[0] && fx < b[1] && fy >= b[2] && fy < b[3] &&
                                     fz >= b[4] && fz < b[5]);
}

export function isStandable(x, y, z) {
  const block = getBlock(x, y, z);
  if (!block || !block.walkable) return false;
  // Headroom: a whole block of air above the surface. The own cell is clear
  // above its span; the cell above must be air, or solid only from the
  // surface's height up (a half-top over a half-bottom leaves 1.5 m). A slope
  // counts its high edge.
  const top = blockSpan(block)[1];
  const above = getBlock(x, y + 1, z);
  return !above || blockSpan(above)[0] >= top;
}

// The standing surface's height, in blocks, measured from the floor of cell
// y = 0 - so levels compare directly across cells. On a slope, its middle.
export function surfaceLevel(x, y, z) {
  const block = getBlock(x, y, z);
  return y + (isDual(block) ? 1 : BLOCK_SHAPES[blockShape(block)].stand);
}

// The surface's height at the edge toward (dx, dz), in the same levels: on a
// slope its uphill or downhill edge, or the middle across it; elsewhere the
// surface. Two cells join where their facing edges meet.
export function edgeLevel(x, y, z, dx, dz) {
  const block = getBlock(x, y, z);
  const shape = BLOCK_SHAPES[blockShape(block)];
  if (isDual(block)) return y + 1;
  if (shape.high === undefined) return y + shape.stand;
  const [fx, fz] = FACINGS[blockFacing(block)];
  const d = dx * fx + dz * fz;
  return y + (d > 0 ? shape.high : d < 0 ? shape.low : shape.stand);
}

// Where an entity standing on it stands, in world metres: the block's top
// face. Blocks are centred on their coordinate, so the cell's floor is half a
// block below it.
export function surfaceHeight(x, y, z) {
  return (surfaceLevel(x, y, z) - 0.5) * BLOCK_METRES;
}

// The ground under a point, for an entity standing at gridPos (metres): the
// surface of the cell the point is in, at the level a step from gridPos
// lands on - on a slope, the plane's height right there, so a character
// climbs it smoothly rather than in two half-block hops.
export function groundHeightAt(gridPos, wx, wz) {
  const bx = Math.round(wx / BLOCK_METRES), bz = Math.round(wz / BLOCK_METRES);
  let y = gridPos.y;
  if (bx !== gridPos.x || bz !== gridPos.z) {
    y = stepLevel(gridPos.x, gridPos.y, gridPos.z, bx, bz) ?? gridPos.y;
  }
  const block = getBlock(bx, y, bz);
  if (!block) return surfaceHeight(gridPos.x, gridPos.y, gridPos.z);
  if (BLOCK_SHAPES[blockShape(block)].treads) {
    // Stairs: the tread under the point - the lower half-way along the climb,
    // then the upper. followGround eases the step between them.
    const u = slopeLevel({ ...block, shape: 'slope' }, wx / BLOCK_METRES - bx + 0.5, wz / BLOCK_METRES - bz + 0.5);
    const shape = BLOCK_SHAPES.stairs;
    return (y + (u < 0.5 ? shape.low : shape.high) - 0.5) * BLOCK_METRES;
  }
  if (!isSlope(block)) return surfaceHeight(bx, y, bz);
  const level = y + slopeLevel(block, wx / BLOCK_METRES - bx + 0.5, wz / BLOCK_METRES - bz + 0.5);
  return (level - 0.5) * BLOCK_METRES;
}

// Keeps a sprite on the ground under it, for an entity at gridPos. The ground
// is continuous on a slope and followed exactly; a jump in it - a half-block
// ledge - is eased out over a few frames rather than snapped. State lives in
// sprite.userData (groundY, groundLift).
export const GROUND_JUMP = 0.4;          // metres: more than a slope moves in a frame
export function followGround(sprite, gridPos, dt) {
  const g = groundHeightAt(gridPos, sprite.position.x, sprite.position.z);
  const ud = sprite.userData;
  if (ud.groundY === undefined) { ud.groundY = sprite.position.y; ud.groundLift = 0; }
  if (Math.abs(g - ud.groundY) > GROUND_JUMP) ud.groundLift += ud.groundY - g;
  ud.groundY = g;
  ud.groundLift *= Math.exp(-16 * dt);
  if (Math.abs(ud.groundLift) < 1e-3) ud.groundLift = 0;
  sprite.position.y = g + ud.groundLift;
}

// The most a step may rise or drop, in blocks: half a block. A full block
// needs Jump, which does not exist yet.
export const STEP_BLOCKS = 0.5;

// The level to stand at in column (nx, nz), stepping from the block at
// (x, y, z): a standable block in y - 1 .. y + 1 whose edge meets this one's
// within a step (edgeLevel - a slope joins at its ends), the nearest winning,
// then the same y. null when the column cannot be stepped into.
export function stepLevel(x, y, z, nx, nz) {
  const dx = Math.sign(nx - x), dz = Math.sign(nz - z);
  const from = edgeLevel(x, y, z, dx, dz);
  let best = null, bestD = Infinity;
  for (const ny of [y, y + 1, y - 1]) {
    if (!isStandable(nx, ny, nz)) continue;
    const d = Math.abs(edgeLevel(nx, ny, nz, -dx, -dz) - from);
    if (d <= STEP_BLOCKS && d < bestD) { best = ny; bestD = d; }
  }
  return best;
}

// Highest standable level in a column, or null if the column has none.
export function getColumnTop(x, z) {
  for (let y = Y_MAX; y >= Y_MIN; y--) {
    if (isStandable(x, y, z)) return y;
  }
  return null;
}

// --- Placed models (bbmodel.js) ---
//
// A model stands on a block's top face - its origin at the centre of that
// face - turned yaw quarter turns (addWorldModel); or hangs on a wall
// (mountWorldModel). Its cubes are world boxes in world texels (metres /
// 0.125): the renderer draws them, the field bakes them. A model with
// bounding boxes (Blockbench bounding_box elements) takes the tile under it,
// like an entity standing there, so nothing walks into it; one without - a
// wall torch - is walked under.
export const WorldModels = [];
const TEXEL_METRES = BLOCK_METRES / 12;
// Every placed cube as [x0, y0, z0, x1, y1, z1] in world texels - what the
// field bakes (core/field.js), and what the field worker is sent.
export const ModelBoxes = [];

export function addWorldModel(model, x, y, z, yaw = 0) {
  const at = [x * 12, surfaceLevel(x, y, z) * 12 - 6, z * 12];
  return placeEntry(model, at, yaw, [x, y, z], [x, y, z]);
}

// A wall model: its NORTH side (-z in Blockbench) is the side against the
// wall, flush with its cell's north edge. It hangs in the air cell (x, y, z),
// origin at the centre of that cell's floor, turned so its north faces the
// wall - which is `wall` (N, E, S, W) of the cell. It belongs to the wall
// block: in battle it shows and hides with it. Returns null, and places
// nothing, when there is no block on that side to hang on.
const WALL_YAW = { N: 0, W: 1, S: 2, E: 3 };     // quarter turns taking north to it
export function mountWorldModel(model, x, y, z, wall) {
  const [dx, dz] = FACINGS[wall] || [0, 0];
  if (WALL_YAW[wall] === undefined || !getBlock(x + dx, y, z + dz)) return null;
  const at = [x * 12, y * 12 - 6, z * 12];
  const entry = placeEntry(model, at, WALL_YAW[wall], [x + dx, y, z + dz], [x, y - 1, z], wall);
  entry.cell = [x, y, z];
  return entry;
}

// Takes a placed model out again: its cubes, its light, its tile.
export function removeWorldModel(entry) {
  const i = WorldModels.indexOf(entry);
  if (i < 0) return false;
  WorldModels.splice(i, 1);
  vacateAt(getVoxelKey(...entry.tile), 'prop:' + entry.name);
  ModelBoxes.length = 0;
  for (const m of WorldModels) for (const c of m.cubes) if (!c.light) ModelBoxes.push([...c.min, ...c.max]);
  return true;
}

// --- Levels: the world as data ---
//
// What the editor saves and the game loads: every block, every placed model
// by name, and the region's settings, spawns and exits. Entities are not in it
// yet. On disk a region is a .bxr file, level format v2 (core/region.js); v1's
// JSON levels, a row per block, still load:
//   { format: 'boxblade-level', version: 1,
//     blocks: [[x, y, z, materialId, shape?, facing?, second?], ...],
//     models: [[name, x, y, z, yaw], ...]            standing on block (x, y, z)
//             [[name, x, y, z, 0, wall], ...] }      hung in cell (x, y, z)
export { LEVEL_FORMAT };

// The Dev Level (levels/dev_level.bxr): the test map every feature through
// A10 was built and measured on - the church, the glass bed, the torches. The
// game loads it and the editor saves to it, until regions (World plan P2)
// bring a level per region.
export const DEV_LEVEL = 'dev_level';

// The loaded region's settings (World plan P2 §4.1): its name, bounds
// [x0, x1, z0, z1] (inclusive, in blocks), sea level, named spawn points
// [x, y, z], and exits. Set by loadLevel; saved back by serializeLevel.
export const Region = { name: null, bounds: null, seaLevel: null, spawns: {}, exits: [] };

// A spawn point of the loaded region, [x, y, z]: the one named, else its
// 'start', else null.
export function regionSpawn(name = 'start') {
  return Region.spawns[name] || Region.spawns.start || null;
}

// second: a dual block's other half's material (isDual) - kept only for a
// shape that has another half.
export function newBlock(materialId, shape = null, facing = null, second = null) {
  const block = { solid: true, walkable: true, materialId, occupant: null, triggerId: null };
  if (shape && shape !== 'full') block.shape = shape;
  if (facing && hasFacing(shape)) block.facing = facing;
  if (second != null && complementShape(block.shape, block.facing)) block.second = second;
  return block;
}

// The world as a v2 level in memory (core/region.js): copies of its chunks,
// indexing a palette of only the block types in use. The bounds grow to take
// in any block the editor put outside them.
export function serializeLevel() {
  const { size, yMin, height } = World.shape;
  const palette = [null], fileType = new Map(), chunks = [];
  World.forEachChunk((cx, cz, cells, types) => {
    const out = new Uint16Array(cells.length);
    for (let i = 0; i < cells.length; i++) {
      const t = cells[i];
      if (!t) continue;
      let f = fileType.get(t);
      if (f === undefined) {
        const b = types[t];
        f = palette.length;
        palette.push(blockRow(b.materialId, b.shape, b.facing, b.second));
        fileType.set(t, f);
      }
      out[i] = f;
    }
    chunks.push({ cx, cz, cells: out });
  });
  const models = WorldModels.map(m => m.wall
    ? [m.name, ...m.cell, 0, m.wall]
    : [m.name, m.x, m.y, m.z, m.yaw]);
  const level = { format: LEVEL_FORMAT, version: LEVEL_VERSION,
    region: { name: Region.name, bounds: Region.bounds, seaLevel: Region.seaLevel },
    spawns: Region.spawns, exits: Region.exits, models,
    chunk: { size, yMin, height }, palette, chunks };
  level.region.bounds = unionBounds(Region.bounds, chunkBounds(chunks, World.shape));
  return level;
}

// Replaces the world with a level: v2 in memory, its readable form, or v1.
// models: name -> parsed model (models.js); name: the region's, if the level
// does not say. Returns the names of models it could not find, or place.
export function loadLevel(level, models, name = null) {
  level = migrateLevel(level, World.shape);
  clearWorldModels();
  World.clear();
  const blocks = level.palette.map(row => row && newBlock(...row));
  for (const { cx, cz, cells } of level.chunks) World.putChunk(cx, cz, cells, blocks);
  Region.name = level.region.name ?? name;
  Region.bounds = level.region.bounds;
  Region.seaLevel = level.region.seaLevel;
  Region.spawns = level.spawns;
  // One that cannot work (no target, a broken box) is dropped, not tripped on.
  Region.exits = (level.exits || []).map(validExit).filter(Boolean);
  const missing = [];
  for (const [name, x, y, z, yaw, wall] of level.models || []) {
    const model = models.get(name);
    const placed = model && (wall ? mountWorldModel(model, x, y, z, wall)
                                  : addWorldModel(model, x, y, z, yaw || 0));
    if (!placed) missing.push(name);
  }
  return missing;
}

// owner: the block the model is filed under (render.js - picking, battle
// hiding, tint); tile: the block whose tile it takes, if it has bounds.
function placeEntry(model, at, yaw, owner, tile, wall = null) {
  const placed = placeModel(model, at, yaw);
  const [x, y, z] = owner;
  const entry = { model, name: model.name, x, y, z, yaw, wall, tile, at, ...placed };
  // A lightSource cube casts no shadow - it is the light - so it is not
  // baked; its centre and size are where the light is and how big.
  entry.lights = placed.cubes.filter(c => c.light).map(c => ({
    position: c.min.map((v, a) => (v + c.max[a]) / 2 * TEXEL_METRES),
    radius: Math.max(...c.max.map((v, a) => v - c.min[a])) / 2 * TEXEL_METRES,
    level: c.lightLevel              // null: the default (main.js MODEL_LIGHT_LEVEL)
  }));
  WorldModels.push(entry);
  for (const c of placed.cubes) if (!c.light) ModelBoxes.push([...c.min, ...c.max]);
  const tileKey = getVoxelKey(...tile);
  if (model.bounds.length && getBlock(...tile) && !occupantAt(tileKey)) setOccupantAt(tileKey, 'prop:' + model.name);
  return entry;
}

export function clearWorldModels() {
  for (const m of WorldModels) vacateAt(getVoxelKey(...m.tile), 'prop:' + m.name);
  WorldModels.length = 0;
  ModelBoxes.length = 0;
}

// The worker's copy (fieldWorker.js).
export function loadModelBoxes(boxes) {
  ModelBoxes.length = 0;
  for (const b of boxes) ModelBoxes.push(b);
}

// --- The field worker's copy of the world ---
//
// The worker bakes the field too (fieldWorker.js), so it needs what the bake
// reads: which blocks exist, and which of them are glass - glass goes in the
// field's glass distance, not its opaque one. Keys alone lost the second part,
// and every strip the worker baked put glass back in as rock: glass that left
// C0 and came back returned solid, casting a shadow.
// Shapes and facings too: a half block or a slope bakes as its own boxes.
export function worldMirrorEntries(world = World) {
  const out = [];
  for (const [key, block] of world) {
    out.push([key, block ? block.materialId ?? null : null, block ? block.shape ?? null : null,
              block ? block.facing ?? null : null]);
  }
  return out;
}

export function loadWorldMirror(entries, world = World) {
  world.clear();
  for (const [key, materialId, shape, facing] of entries) {
    const block = { materialId };
    if (shape) block.shape = shape;
    if (facing) block.facing = facing;
    world.set(key, block);
  }
}

export function createTestArea(width, depth) {
  World.clear();

  // A single ground layer at y = 0. No sub-surface fill: buried blocks would be
  // invisible geometry inflating both the draw and the field's occupancy set
  // for nothing, since a block is only ever seen or marched against when some
  // face of it is exposed to air.
  for (let x = 0; x < width; x++) {
    for (let z = 0; z < depth; z++) {
      World.set(getVoxelKey(x, GROUND_Y, z), {
        solid: true,
        walkable: true,
        materialId: 'grass', // an id from materials.js
        occupant: null,
        triggerId: null
      });
    }
  }

  addTestElevation();
  // Only when the area holds it: the floor patch replaces ground, so it means
  // nothing past the area's edge. The small test areas the tools build stay
  // exactly as they were.
  if (width > 8 && depth > 17) addReflectionTest();
  if (width > 16 && depth > 13) addGlassTest();
  if (width > 31 && depth > 31) addHalfTileTest();
  if (width > 33 && depth > 31) addSlopeTest();
  if (width > 20 && depth > 12) addStairTest();
}

// Half-tile test bed, beside the raised platform (28-31, y 1, 28-31): two
// half-bottom steps on its west edge make it reachable from the ground (1.0
// -> 1.5 -> 2.0 blocks), and an arch of two pillars under a half-top lintel
// leaves 2.25 m beneath - 1.5 m of headroom over the ground - to walk through.
function addHalfTileTest() {
  const put = (x, y, z, shape = null, materialId = 'grass') => World.set(getVoxelKey(x, y, z), {
    solid: true, walkable: true, materialId, ...(shape ? { shape } : {}), occupant: null, triggerId: null
  });
  put(27, 1, 29, 'halfBottom');
  put(27, 1, 30, 'halfBottom');
  for (const z of [24, 26]) for (let y = 1; y <= 2; y++) put(24, y, z);
  put(24, 2, 25, 'halfTop');
}

// Stair test bed: marble stairs up the west side of the grass wall at x = 20
// (two blocks tall, top at level 3), along z = 12 - ground (1) -> stairs at
// (18, 1) (1.5, 2) -> stairs at (19, 2) on a marble block (2.5, 3) -> the
// wall's top (3). Beside the wall torch at (19, 2, 11).
function addStairTest() {
  const put = (x, y, z, shape = null, facing = null) =>
    World.set(getVoxelKey(x, y, z), newBlock('marble', shape, facing));
  put(18, 1, 12, 'stairs', 'E');
  put(19, 1, 12);
  put(19, 2, 12, 'stairs', 'E');
}

// Slope test bed, also onto the raised platform: a full slope on its north
// side at (29, 1, 27), climbing south (+z) from the ground (level 1) to the
// platform (level 2) in one cell; and a gentle ramp on its east side, a half
// slope at (33, 1, 29) and a high half slope at (32, 1, 29), climbing west
// over two cells.
function addSlopeTest() {
  const put = (x, y, z, shape, facing) => World.set(getVoxelKey(x, y, z), {
    solid: true, walkable: true, materialId: 'grass', shape, facing, occupant: null, triggerId: null
  });
  put(29, 1, 27, 'slope', 'S');
  put(33, 1, 29, 'halfSlope', 'W');
  put(32, 1, 29, 'halfSlopeHigh', 'W');
}

// Glass test bed: the thick case - a single 1.5 m cube, about five feet of
// solid glass - on the grass beside the reflection bed, and a window: a wall
// of glass one block thick, 3 wide and 2 tall, in front of the trees at
// z = 8-9, so sprites are seen through it.
//
// Beside them, in a row along z = 13: a 2 x 2 stained-glass window, whose
// panes colour the light that falls through it, and a frosted cube. Kept off
// the tiles the sprite tests use as open ground (e.g. 18, 11).
function addGlassTest() {
  const put = (x, y, z, materialId = 'glass') => World.set(getVoxelKey(x, y, z), {
    solid: true, walkable: true, materialId, occupant: null, triggerId: null
  });
  put(12, 1, 13);
  for (let x = 13; x <= 15; x++) {
    for (let y = 1; y <= 2; y++) put(x, y, 11);
  }
  for (let x = 16; x <= 17; x++) {
    for (let y = 1; y <= 2; y++) put(x, y, 13, 'stainedGlass');
  }
  put(10, 1, 13, 'frostedGlass');
}

// Reflection test bed: a 5x5 patch of polished iron floor with a polished
// marble wall standing along its far edge, so the floor has something to
// mirror and the wall has the floor and the open sky. Both are LabPBR
// materials with high smoothness (materials.js, and their _s textures). The
// floor replaces ground blocks rather than sitting on them, so it stays
// walkable at ground level.
function addReflectionTest() {
  const put = (x, y, z, materialId) => World.set(getVoxelKey(x, y, z), {
    solid: true, walkable: true, materialId, occupant: null, triggerId: null
  });
  for (let x = 3; x <= 7; x++) {
    for (let z = 12; z <= 16; z++) put(x, GROUND_Y, z, 'ironPlate');
  }
  // Wall: 2 blocks tall, 7 long, along x at z = 17 - overhanging the patch
  // by one block each side.
  for (let x = 2; x <= 8; x++) {
    for (let y = 1; y <= 2; y++) put(x, y, 17, 'marble');
  }
  // Standing iron plates either side, running out from the marble's ends at
  // right angles: 2 tall, 3 long, on the grass just outside the patch. Their
  // inner faces look across the floor at each other, so there are mirrors on
  // three axes - the floor (up), the plates (+x and -x) - and the marble
  // between them to catch what they throw.
  for (let z = 14; z <= 16; z++) {
    for (let y = 1; y <= 2; y++) {
      put(1, y, z, 'ironPlate');
      put(9, y, z, 'ironPlate');
    }
  }
}

// Vertical test geometry, to give the field something that can actually
// occlude a light, which a flat single-layer world cannot. A full block's rise
// needs Jump; the raised platform is reached by the half-tile steps of
// addHalfTileTest.
function addTestElevation() {
  const put = (x, y, z) => World.set(getVoxelKey(x, y, z), {
    solid: true, walkable: true, materialId: 'grass', occupant: null, triggerId: null
  });

  // Wall: 2 blocks tall, 7 long, running along z at x = 20.
  for (let z = 8; z <= 14; z++) {
    for (let y = 1; y <= 2; y++) put(20, y, z);
  }

  // Pillar: 3 blocks tall, isolated, for a long thin shadow.
  for (let y = 1; y <= 3; y++) put(26, y, 20);

  // Raised platform: 4x4, one block up. Reached by half-tile steps.
  for (let x = 28; x <= 31; x++) {
    for (let z = 28; z <= 31; z++) put(x, 1, z);
  }
}

// --- ENTITY SCHEMA ---
export function createEntity(config) {
  return {
    id: config.id || "entity_" + Math.random().toString(36).substr(2, 9),
    name: config.name || "Unknown",
    gridPos: { x: config.gridPos.x, y: config.gridPos.y, z: config.gridPos.z },
    stats: config.stats || { STR: 10, DEX: 10, CON: 10, WIS: 10, INT: 10, CHA: 10 },
    speed: config.speed || 6,
    hp: config.hp || { current: 100, max: 100 },
    ac: config.ac || 10,
    initiative: config.initiative || null,
    weaponDie: config.weaponDie || "1d6",
    facing: config.facing || "N",
    mode: config.mode || "explore",
    inventory: config.inventory || []
  };
}

export function addToInventory(entity, itemId, quantity) {
  const existing = entity.inventory.find(i => i.itemId === itemId);
  if (existing) existing.quantity += quantity;
  else entity.inventory.push({ itemId, quantity });
}

// --- DISTANCE & PATHFINDING ---
export function pathDistance(a, b) {
  // Manhattan distance (4-directional). Includes the vertical term: a half
  // step between levels counts as one, like the step across. Slopes and
  // stairs will need a deliberate cost decision.
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
}

// Walkable neighbours of a standing position. A neighbouring column is
// entered at the level stepLevel picks: its surface within half a block of
// this one (a half block steps up onto a full one). A full block's rise is
// impassable until Jump exists.
function isEnterable(x, y, z) {
  if (!isStandable(x, y, z)) return false;
  return !occupantAt(getVoxelKey(x, y, z));
}

// A* over standable tiles. Called on every click, every held-mouse retarget
// and by the AI, so it has to be cheap: a long path or an unreachable target
// (which searches everything reachable) cost 2.5-11 ms per call when the open
// set was an array scanned in full every step and every lookup built an
// "x,y,z" string - a dropped frame per click at 240 fps.
//
// Now: numeric keys, and a binary heap ordered by (f, the order the tile
// entered the open set). That is exactly the old scan's choice - the first
// tile in the open set with the lowest f, the open set kept in insertion
// order - so the paths are identical, ties included (tools/world.test.mjs
// checks it against the old search). A tile whose f improves while open
// keeps its place: a new heap entry with its old order, and the stale entry
// is skipped when it surfaces.
const pathKey = (x, y, z) =>
  ((Math.round(x) + 32768) * 65536 + (Math.round(z) + 32768)) * 1024 + (Math.round(y) + 512);

export function findPath(start, end, allowDiagonals = false) {
  // A target nothing can stand on (a wall, a covered top, an occupied tile)
  // is never reached - without this the search exhausts the map first.
  if (!isEnterable(end.x, end.y, end.z)) return [];
  const cameFrom = new Map();
  const gScore = new Map();
  const fScore = new Map();
  // Tiles in the open set -> the order they entered it; and each tile's node
  // object as first opened (the old open set held that object, and the path
  // is built from these).
  const openOrder = new Map();
  const nodeOf = new Map();
  // Whether a tile can be entered does not change during a search, and each
  // is asked up to nine times (as a neighbour and as diagonal corners) - each
  // time three string-keyed World lookups. Asked once, then remembered.
  const enterable = new Map();
  const canEnter = (x, y, z) => {
    const k = pathKey(x, y, z);
    let v = enterable.get(k);
    if (v === undefined) { v = isEnterable(x, y, z); enterable.set(k, v); }
    return v;
  };
  const canStepInto = (from, nx, nz) => {
    const ny = stepLevel(from.x, from.y, from.z, nx, nz);
    return ny !== null && canEnter(nx, ny, nz);
  };
  const heap = [];
  const before = (a, b) => a.f < b.f || (a.f === b.f && a.order < b.order);
  const push = e => {
    heap.push(e);
    let i = heap.length - 1;
    while (i > 0) {
      const up = (i - 1) >> 1;
      if (!before(heap[i], heap[up])) break;
      [heap[i], heap[up]] = [heap[up], heap[i]];
      i = up;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < heap.length && before(heap[l], heap[m])) m = l;
        if (r < heap.length && before(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]];
        i = m;
      }
    }
    return top;
  };

  // Octile heuristic for diagonals, Manhattan for 4-way
  const getH = (a, b) => {
    const dx = Math.abs(a.x - b.x);
    const dz = Math.abs(a.z - b.z);
    return allowDiagonals ? (Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz)) : (dx + dz);
  };

  let entered = 0;
  const startKey = pathKey(start.x, start.y, start.z);
  gScore.set(startKey, 0);
  fScore.set(startKey, getH(start, end));
  openOrder.set(startKey, entered);
  nodeOf.set(startKey, start);
  push({ f: fScore.get(startKey), order: entered++, key: startKey });

  const dirs = [
    {x: 0, z: -1, cost: 1},
    {x: 0, z: 1, cost: 1},
    {x: -1, z: 0, cost: 1},
    {x: 1, z: 0, cost: 1}
  ];

  if (allowDiagonals) {
    dirs.push(
      {x: -1, z: -1, cost: Math.SQRT2},
      {x: 1, z: -1, cost: Math.SQRT2},
      {x: -1, z: 1, cost: Math.SQRT2},
      {x: 1, z: 1, cost: Math.SQRT2}
    );
  }

  while (openOrder.size > 0) {
    // The live entry: still open under the same entry, at its current f.
    let e = pop();
    while (openOrder.get(e.key) !== e.order || fScore.get(e.key) !== e.f) e = pop();
    const currentKey = e.key;
    const current = nodeOf.get(currentKey);

    if (current.x === end.x && current.y === end.y && current.z === end.z) {
      const path = [];
      let currNode = current;
      let currKey = currentKey;

      while (cameFrom.has(currKey)) {
        path.unshift(currNode);
        currNode = cameFrom.get(currKey);
        currKey = pathKey(currNode.x, currNode.y, currNode.z);
      }
      return path;
    }

    openOrder.delete(currentKey);

    for (const dir of dirs) {
      // The neighbour's level: the same, or half a block up or down.
      const ny = stepLevel(current.x, current.y, current.z, current.x + dir.x, current.z + dir.z);
      if (ny === null) continue;
      const neighbor = { x: current.x + dir.x, y: ny, z: current.z + dir.z };

      if (currentArena && !currentArena.has(getVoxelKey(neighbor.x, neighbor.y, neighbor.z))) continue;
      if (!canEnter(neighbor.x, neighbor.y, neighbor.z)) continue;

      // Prevent clipping through solid OR occupied corners when moving diagonally:
      // both corners must be steppable from here too.
      if (allowDiagonals && dir.cost > 1) {
        if (!canStepInto(current, current.x + dir.x, current.z)) continue;
        if (!canStepInto(current, current.x, current.z + dir.z)) continue;
      }

      const neighborKey = pathKey(neighbor.x, neighbor.y, neighbor.z);
      const tentative_gScore = (gScore.get(currentKey) ?? Infinity) + dir.cost;

      if (tentative_gScore < (gScore.get(neighborKey) ?? Infinity)) {
        cameFrom.set(neighborKey, current);
        gScore.set(neighborKey, tentative_gScore);
        const f = tentative_gScore + getH(neighbor, end);
        fScore.set(neighborKey, f);

        // Already open: it keeps its place in the order and its object (the
        // old array kept its first copy). Otherwise it joins the end.
        let order = openOrder.get(neighborKey);
        if (order === undefined) {
          order = entered++;
          openOrder.set(neighborKey, order);
          nodeOf.set(neighborKey, neighbor);
        }
        push({ f, order, key: neighborKey });
      }
    }
  }
  return [];
}

// --- BATTLE & RANGE LOGIC ---
export let currentArena = null;

export function attackRange(a, b) {
  // Chebyshev distance (Allows diagonal adjacency)
  return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.z - b.z));
}

export function isInInteractRange(a, b) {
  // Chebyshev adjacency, same shape as melee range - deliberately NOT calling
  // combat.js's isInMeleeRange to keep world.js/objects.js free of combat imports.
  return attackRange(a.gridPos, b.gridPos) <= 1;
}

export function enterBattle(playerGridPos) {
  const arena = new Map();

  const cx = Math.floor(playerGridPos.x / CHUNK_SIZE);
  const cz = Math.floor(playerGridPos.z / CHUNK_SIZE);

  const minX = cx * CHUNK_SIZE;
  const maxX = minX + CHUNK_SIZE - 1;
  const minZ = cz * CHUNK_SIZE;
  const maxZ = minZ + CHUNK_SIZE - 1;

  // The arena is the full 12x12x12 chunk, not just its ground layer: everything
  // in the vertical span belongs to it, so elevated geometry stays visible and
  // keeps occluding once the lighting work lands.
  for (const [key, block] of World.entries()) {
    const [gx, gy, gz] = key.split(',').map(Number);
    if (gx >= minX && gx <= maxX && gz >= minZ && gz <= maxZ && gy >= Y_MIN && gy <= Y_MAX) {
      arena.set(key, block);
    }
  }
  currentArena = arena;
  return { arena, bounds: { minX, maxX, minZ, maxZ, minY: Y_MIN, maxY: Y_MAX } };
}

export function exitBattle() {
  currentArena = null;
}

export function getReachableVoxels(start, speed) {
  const reachable = new Map();
  const queue = [{ pos: start, cost: 0 }];
  reachable.set(getVoxelKey(start.x, start.y, start.z), 0);

  const dirs = [ {x: 0, z: -1}, {x: 0, z: 1}, {x: -1, z: 0}, {x: 1, z: 0} ];

  while (queue.length > 0) {
    queue.sort((a, b) => a.cost - b.cost);
    const current = queue.shift();

    if (current.cost >= speed) continue;

    for (const dir of dirs) {
      const p = current.pos;
      const ny = stepLevel(p.x, p.y, p.z, p.x + dir.x, p.z + dir.z);
      if (ny === null) continue;
      const neighbor = { x: p.x + dir.x, y: ny, z: p.z + dir.z };
      const nKey = getVoxelKey(neighbor.x, neighbor.y, neighbor.z);

      if (currentArena && !currentArena.has(nKey)) continue;
      if (!isEnterable(neighbor.x, neighbor.y, neighbor.z)) continue;

      const newCost = current.cost + 1;
      if (!reachable.has(nKey) || newCost < reachable.get(nKey)) {
        reachable.set(nKey, newCost);
        queue.push({ pos: neighbor, cost: newCost });
      }
    }
  }
  return reachable;
}

export function getAbilityModifier(score) {
  return Math.floor((score - 10) / 2);
}
