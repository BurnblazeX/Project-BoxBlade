import { validExit } from './exits.js';
import { isGlassMaterial } from './materials.js';
import { placeModel } from './bbmodel.js';
import { rectCovered } from './rects.js';
import { createChunkWorld } from './chunks.js';
import { LEVEL_FORMAT, LEVEL_VERSION, migrateLevel, partRow, chunkBounds, unionBounds } from './region.js';

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

// --- Block shapes: parts on the Grid ---
//
// A tile - one block's cell, 1.5 m, 12 texels a side - is 3 x 3 x 3 Grid
// cells (the BoxGrid: 0.5 m, 4 texels). Its block is one to three PARTS, each
// a shape in one of its states, turned to a facing and set at a Grid cell of
// the tile: three slabs of three materials stack into a full tile. Only the
// parts are stored, never the Grid cells - the engine works out which cells a
// tile fills from its parts.
//
// A part: { materialId, shape, state, facing, at }. A block holds its first
// part's fields itself (block.materialId is the tile's main material) and the
// others in block.more (tileParts). at, the Grid cell x + 3y + 9z (0..26),
// moves the part's boxes after the turn; a part never leaves its tile
// (partFits).
//
// Every shape is a list of axis-aligned boxes in cell fractions, facing E, on
// the 12-texel grid, so every face lies on a texel boundary and on its axis -
// the texel lock, the face uv, the texel cache's ids and the snapped hit
// normals hold as they are. The renderer draws each box as an instance of the
// one terrain box, the field bakes each, and nothing downstream knows the shape.
//
//   full   the tile.
//   slab   one Grid layer; at sets its height. Three stack to a full tile.
//   pane   one Grid column, the tile's height; at sets side or centre.
//          Turned, it is every vertical slab. It turns by axis only: E and
//          W are the same pane, N and S too.
//   slope  low, medium or high: rising 1, 2 or 3 Grid cells across the tile.
//          On a slab (at one Grid cell up) for a raised one.
//   steps  three: three treads a Grid cell each. two: the lower two, the
//          first third of the tile left open. flipped: three, upside down.
//
// Slopes are SEEN as planes and BAKED as staircases (doc §0). The renderer
// draws the plane - the terrain box over the slope's span with its downhill
// top edge dropped, shaded on the plane's own texel grid (gpu/terrain.js
// terrainSurfaceTSL). The boxes here are what everything else sees: the
// field, and so shadows, AO, GI and what reflections hit. They are a
// staircase of 12 steps one texel deep, INSCRIBED under the plane - each
// step's top meets it at the step's downhill edge and falls below it across
// the step - so the plane is never inside its own steps and never shadows
// itself. facing (N, E, S, W) is the uphill direction - for steps too.
//
// Glass takes the one-box shapes (glass: true below - full, slab, pane); its
// walk (core/field.js walkThroughGlass, and the shader's) steps from box face
// to box face. A glass slope or steps is whole.

export const GRID = 3;
const G = 1 / GRID;
// Steps are one texel deep: 12 to the block. Step k's top is the plane's
// height at its downhill edge; a first step of no height is left out.
const T = 12;
const staircase = rise => Array.from({ length: T }, (_, k) =>
  [k / T, (k + 1) / T, 0, rise * k / T, 0, 1]).filter(b => b[3] > 0);
// shape: { states: { name: { boxes, rise (a slope's, cell fractions),
//   treads (stood on step by step) } } - the first state the default;
//   turns: turned by facing (hasFacing); axis: turned by axis only (a pane);
//   slope: drawn as a plane; glass: may be glass }
export const BLOCK_SHAPES = {
  full:  { states: { full: { boxes: [[0, 1, 0, 1, 0, 1]] } }, glass: true },
  slab:  { states: { slab: { boxes: [[0, 1, 0, G, 0, 1]] } }, glass: true },
  pane:  { states: { pane: { boxes: [[0, G, 0, 1, 0, 1]] } }, turns: true, axis: true, glass: true },
  slope: { states: { low:    { boxes: staircase(G),     rise: G },
                     medium: { boxes: staircase(2 * G), rise: 2 * G },
                     high:   { boxes: staircase(1),     rise: 1 } },
           slope: true, turns: true },
  steps: { states: { three:   { boxes: [[0, 1, 0, G, 0, 1], [G, 1, G, 2 * G, 0, 1], [2 * G, 1, 2 * G, 1, 0, 1]], treads: true },
                     two:     { boxes: [[G, 1, 0, G, 0, 1], [2 * G, 1, G, 2 * G, 0, 1]], treads: true },
                     flipped: { boxes: [[0, 1, 2 * G, 1, 0, 1], [G, 1, G, 2 * G, 0, 1], [2 * G, 1, 0, G, 0, 1]] } },
           turns: true }
};
export const MAX_PARTS = 3;

// A shape's states, the default first; the one after state (F in the editor).
export const shapeStates = shape => Object.keys((BLOCK_SHAPES[shape] || BLOCK_SHAPES.full).states);
export function nextShapeState(shape, state) {
  const s = shapeStates(shape);
  return s[(s.indexOf(state) + 1) % s.length];
}

// A Grid cell of the tile as [x, y, z] (0..2 each), and back.
export const atCell = at => [at % GRID, Math.floor(at / GRID) % GRID, Math.floor(at / (GRID * GRID))];
export const cellAt = (x, y, z) => x + GRID * y + GRID * GRID * z;

// A block's parts: itself (its own fields are its first part's), then block.more.
const NO_MORE = [];
export function tileParts(block) {
  if (!block) return NO_MORE;
  return block.more ? [block, ...block.more] : [block];
}
// A tile of more than one part.
export const isMulti = block => !!block && !!block.more && block.more.length > 0;

// A part's shape, state and facing, made valid: a glass part takes only the
// glass shapes, whole otherwise.
export function partShape(part) {
  const s = part && BLOCK_SHAPES[part.shape] ? part.shape : 'full';
  return part && isGlassMaterial(part.materialId) && !BLOCK_SHAPES[s].glass ? 'full' : s;
}
export function partState(part) {
  const shape = partShape(part), states = BLOCK_SHAPES[shape].states;
  return part && states[part.state] ? part.state : shapeStates(shape)[0];
}
export function partFacing(part) {
  return part && FACINGS[part.facing] ? part.facing : 'E';
}
// The first part's - the block's own fields.
export const blockShape = partShape;
export const blockFacing = partFacing;

// A slope: drawn as a plane over its baked staircase (render.js).
export function isSlope(part) {
  return BLOCK_SHAPES[partShape(part)].slope === true;
}

// A shape that turns with its facing: slopes, steps, panes.
export function hasFacing(shape) {
  return !!BLOCK_SHAPES[shape] && BLOCK_SHAPES[shape].turns === true;
}

// Cell fractions snapped to 36ths - the Grid's thirds and the slopes' steps
// (a low slope's rise in twelve steps) both land on it - so a turned or moved
// face lands on exactly the plane another does (1 - 8/12 is not 4/12 in
// floats) and faces compare equal (exposedFaceMask, mirrors.js).
const snap36 = c => Math.round(c * 36) / 36;

// A part's boxes, turned and moved to its Grid cell: W mirrors x, S swaps x
// and z, N swaps and mirrors - a pane only swaps. Cached per shape, state,
// facing and cell.
const boxCache = new Map();
export function partBoxes(part) {
  const shape = partShape(part), state = partState(part);
  const def = BLOCK_SHAPES[shape];
  let facing = hasFacing(shape) ? partFacing(part) : 'E';
  if (def.axis) facing = facing === 'W' ? 'E' : facing === 'N' ? 'S' : facing;
  const at = (part && part.at) | 0;
  const k = `${shape}|${state}|${facing}|${at}`;
  let out = boxCache.get(k);
  if (!out) {
    const [ax, ay, az] = atCell(at).map(v => v * G);
    out = def.states[state].boxes.map(([x0, x1, y0, y1, z0, z1]) => {
      let b = [x0, x1, y0, y1, z0, z1];
      if (facing === 'W') b = [1 - x1, 1 - x0, y0, y1, z0, z1];
      if (facing === 'S') b = [z0, z1, y0, y1, x0, x1];
      if (facing === 'N') b = [z0, z1, y0, y1, 1 - x1, 1 - x0];
      return [b[0] + ax, b[1] + ax, b[2] + ay, b[3] + ay, b[4] + az, b[5] + az].map(snap36);
    });
    boxCache.set(k, out);
  }
  return out;
}

// Whether a part stays inside its tile at its Grid cell.
export function partFits(part) {
  const at = part.at | 0;
  return at >= 0 && at < GRID ** 3 && partBoxes(part).every(b => b.every(c => c >= 0 && c <= 1));
}

// Every box of a block, part by part (cell fractions).
export function blockBoxes(block) {
  if (!isMulti(block)) return partBoxes(block);
  return tileParts(block).flatMap(partBoxes);
}

// What a block is made of, part by part: [{ materialId, part, boxes }].
export function blockParts(block) {
  return tileParts(block).map(part => ({ materialId: part.materialId, part, boxes: partBoxes(part) }));
}

// A block of more than one box (slopes' staircases, steps, a stack): what the
// mirror rectangles, which take whole faces, leave out.
export function isCompound(block) {
  return blockBoxes(block).length > 1;
}

// A slope part's rise and the Grid height it starts at, in Grid cells.
function slopeRiseAt(part) {
  return [Math.round(BLOCK_SHAPES.slope.states[partState(part)].rise * GRID), atCell(part.at | 0)[1]];
}
// The slopes there are, by rise and starting height in Grid cells: kind k
// is SLOPE_KINDS[k] (render.js slopeCode and gpu/terrain.js slopeFromCodeTSL
// number them from 1).
export const SLOPE_KINDS = [[3, 0], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1]];
export function slopeKind(part) {
  const [rise, y] = slopeRiseAt(part);
  return SLOPE_KINDS.findIndex(([r, a]) => r === rise && a === y);
}

// --- Buried faces ---
//
// Which of a drawn box's faces can be seen at all: a face is buried when the
// opaque boxes just past it cover all of it - the next cell's from its near
// side for a face on the cell's boundary, this cell's others for one inside
// it (steps' upper treads on their lower, a stack's parts on each other). So
// two slabs side by side, a slab on a slab, a row of panes hide the faces
// between them, and a face only partly covered is drawn whole. The terrain
// draws a buried face as nothing (render.js terrainShapeTSL). Raster only -
// the field, the block volume, mirrors and picking all read the world, not
// these triangles, so shadows, GI and reflections are untouched. A slope's
// top is its plane, which drops away from the cell's top, so it is never
// buried.
// Bits d: +x, -x, +y, -y, +z, -z (BoxGeometry's face order); set = drawn.
export const FACE_STEPS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
export const ALL_FACES = 63;
// The boxes in a block that hide a face pressed against them, cell
// fractions: every opaque part's (glass hides nothing - it is seen through),
// a slope only its floor (its sides are a profile, its top a plane).
export function opaqueBoxes(block) {
  if (!block) return [];
  const out = [];
  for (const part of tileParts(block)) {
    if (isGlassMaterial(part.materialId)) continue;
    if (isSlope(part)) {
      const y = snap36(atCell(part.at | 0)[1] * G);
      out.push([0, 1, y, y, 0, 1]);
    } else out.push(...partBoxes(part));
  }
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

// --- What the rays see of a cell (render.js block volume, gpu/volume.js) ---
//
// A part as one byte, the block volume's A in the part's slot: a box on the
// Grid - its span on each axis one of six (lo, hi) pairs of thirds, x + 6y +
// 36z, 0 the whole tile - or a slope, 216 + kind * 4 + facing (SLOPE_KINDS;
// facing 0 N, 1 E, 2 S, 3 W). The rays take every part as its box: steps and
// slopes whole over their span.
const SPANS = [[0, 3], [0, 1], [0, 2], [1, 2], [1, 3], [2, 3]];
export const SLOPE_CODE_BASE = 216;
const FACING_CODE = { N: 0, E: 1, S: 2, W: 3 };
export function shapeCode(part) {
  if (isSlope(part)) return SLOPE_CODE_BASE + slopeKind(part) * 4 + FACING_CODE[partFacing(part)];
  const boxes = partBoxes(part);
  let code = 0;
  for (let a = 0, m = 1; a < 3; a++, m *= 6) {
    let lo = 1, hi = 0;
    for (const b of boxes) { if (b[2 * a] < lo) lo = b[2 * a]; if (b[2 * a + 1] > hi) hi = b[2 * a + 1]; }
    const l = Math.round(lo * GRID), h = Math.round(hi * GRID);
    code += m * SPANS.findIndex(([sl, sh]) => sl === l && sh === h);
  }
  return code;
}
// The box the rays take for a code, in cell fractions.
export function codeBox(code) {
  if (code >= SLOPE_CODE_BASE) {
    const [rise, y] = SLOPE_KINDS[Math.floor((code - SLOPE_CODE_BASE) / 4)];
    return [0, 1, y / GRID, (y + rise) / GRID, 0, 1];
  }
  const s = [code % 6, Math.floor(code / 6) % 6, Math.floor(code / 36)].map(k => SPANS[k]);
  return [s[0][0] / GRID, s[0][1] / GRID, s[1][0] / GRID, s[1][1] / GRID, s[2][0] / GRID, s[2][1] / GRID];
}
// What fills the world point (metres) for the rays: { cell, materialId, box }
// with box in cell fractions - the first of the block's parts whose ray box
// holds it - or null for air. Half-open as isSolidAt.
export function partAt(wx, wy, wz, world = World) {
  const cell = [wx, wy, wz].map(v => Math.round(v / BLOCK_METRES));
  const block = world.get(cell.join(','));
  if (!block) return null;
  const f = [wx, wy, wz].map((v, a) => v / BLOCK_METRES - cell[a] + 0.5);
  for (const part of tileParts(block)) {
    const b = codeBox(shapeCode(part));
    if (f[0] >= b[0] && f[0] < b[1] && f[1] >= b[2] && f[1] < b[3] && f[2] >= b[4] && f[2] < b[5]) {
      return { cell, materialId: part.materialId, box: b };
    }
  }
  return null;
}

// The vertical span a block fills, [lo, hi] in blocks from the cell's floor -
// its boxes' bounds, or for a slope its PLANE's top, which its staircase
// (inscribed under the plane) stops a texel short of. What the slope is
// drawn over, and its headroom.
export function blockSpan(block) {
  let lo = 1, hi = 0;
  for (const part of tileParts(block)) {
    const [l, h] = partSpan(part);
    if (l < lo) lo = l;
    if (h > hi) hi = h;
  }
  return [lo, hi];
}
// One part's span (never the block's others, though the block is its first).
export function partSpan(part) {
  if (isSlope(part)) {
    const [rise, y] = slopeRiseAt(part);
    return [snap36(y * G), snap36((y + rise) * G)];
  }
  let lo = 1, hi = 0;
  for (const b of partBoxes(part)) { if (b[2] < lo) lo = b[2]; if (b[3] > hi) hi = b[3]; }
  return [lo, hi];
}

// A slope part's surface level at a point of its cell (fractions fx, fz), as
// a fraction of the cell's height: the plane that is drawn, not the steps.
export function slopeLevel(part, fx, fz) {
  const [rise, y] = slopeRiseAt(part);
  const [dx, dz] = FACINGS[partFacing(part)];
  const u = Math.min(1, Math.max(0, (fx - 0.5) * dx + (fz - 0.5) * dz + 0.5));
  return (y + rise * u) * G;
}

// The standing surface over a point of the cell (fractions): the top of the
// highest part over it - a slope's plane, a step's tread, a slab's top - or
// null where no part covers it. Closed: on the edge between two treads, the
// higher; on the cell's edge, what is on that edge.
function surfaceAt(block, fx, fz) {
  let top = null;
  for (const part of tileParts(block)) {
    if (isSlope(part)) {
      const l = slopeLevel(part, fx, fz);
      if (top === null || l > top) top = l;
      continue;
    }
    for (const b of partBoxes(part)) {
      if (fx >= b[0] && fx <= b[1] && fz >= b[4] && fz <= b[5] && (top === null || b[3] > top)) top = b[3];
    }
  }
  return top;
}
// Where no part covers the point: the floor beside open steps, else the
// block's top (a pane is stood on at its top, as a wall).
function surfaceOr(block, fx, fz) {
  const s = surfaceAt(block, fx, fz);
  if (s !== null) return s;
  return tileParts(block).some(p => BLOCK_SHAPES[partShape(p)].states[partState(p)].treads) ? 0 : blockSpan(block)[1];
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
  for (const part of tileParts(block)) {
    if (isSlope(part)) {
      if (fy >= atCell(part.at | 0)[1] * G && fy < slopeLevel(part, fx, fz) - slopeClearance) return true;
      continue;
    }
    if (partBoxes(part).some(b => fx >= b[0] && fx < b[1] && fy >= b[2] && fy < b[3] &&
                                  fz >= b[4] && fz < b[5])) return true;
  }
  return false;
}

export function isStandable(x, y, z) {
  const block = getBlock(x, y, z);
  if (!block || !block.walkable) return false;
  // Headroom: a whole block of air above the surface. The own cell is clear
  // above its span; the cell above must be air, or solid only from the
  // surface's height up (a slab at the top over one at the bottom leaves
  // 2 m). A slope counts its high edge.
  const top = blockSpan(block)[1];
  const above = getBlock(x, y + 1, z);
  return !above || blockSpan(above)[0] >= top;
}

// The standing surface's height, in blocks, measured from the floor of cell
// y = 0 - so levels compare directly across cells. Over the cell's middle.
export function surfaceLevel(x, y, z) {
  return y + surfaceOr(getBlock(x, y, z), 0.5, 0.5);
}

// The surface's height at the edge toward (dx, dz), in the same levels: on a
// slope its uphill or downhill edge, or the middle across it; on steps the
// tread at that edge. Two cells join where their facing edges meet.
export function edgeLevel(x, y, z, dx, dz) {
  return y + surfaceOr(getBlock(x, y, z), 0.5 + dx * 0.5, 0.5 + dz * 0.5);
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
// climbs it smoothly; on steps, the tread under it (followGround eases the
// step between treads).
export function groundHeightAt(gridPos, wx, wz) {
  const bx = Math.round(wx / BLOCK_METRES), bz = Math.round(wz / BLOCK_METRES);
  let y = gridPos.y;
  if (bx !== gridPos.x || bz !== gridPos.z) {
    y = stepLevel(gridPos.x, gridPos.y, gridPos.z, bx, bz) ?? gridPos.y;
  }
  const block = getBlock(bx, y, bz);
  if (!block) return surfaceHeight(gridPos.x, gridPos.y, gridPos.z);
  const level = y + surfaceOr(block, wx / BLOCK_METRES - bx + 0.5, wz / BLOCK_METRES - bz + 0.5);
  return (level - 0.5) * BLOCK_METRES;
}

// Keeps a sprite on the ground under it, for an entity at gridPos. The ground
// is continuous on a slope and followed exactly; a jump in it - a tread, a
// slab's edge - is eased out over a few frames rather than snapped. State
// lives in sprite.userData (groundY, groundLift).
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

// The most a step may rise or drop, in blocks: a Grid cell, a third of a
// tile - a slab, a tread. More needs steps, a slope or Jump (not built yet).
export const STEP_BLOCKS = G + 1e-6;

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
// yet. On disk a region is a .bxr file, level format v3 (core/region.js); v1's
// JSON levels, a row per block in halves, still load:
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

// A block: its first part - materialId, shape, facing, state, at - and the
// tile's other parts (more: [{ materialId, shape, facing, state, at }], up to
// MAX_PARTS in all). Defaults are left off, so equal blocks intern as one
// type; a pane's facing is its axis (E or S).
export function newBlock(materialId, shape = null, facing = null, state = null, at = 0, more = null) {
  const block = { solid: true, walkable: true, occupant: null, triggerId: null,
                  ...cleanPart({ materialId, shape, facing, state, at }) };
  if (more && more.length) block.more = more.slice(0, MAX_PARTS - 1).map(cleanPart);
  return block;
}
function cleanPart({ materialId, shape, facing, state, at }) {
  const part = { materialId };
  if (shape && shape !== 'full' && BLOCK_SHAPES[shape]) part.shape = shape;
  const s = part.shape || 'full';
  if (facing && hasFacing(s)) part.facing = BLOCK_SHAPES[s].axis ? (facing === 'W' ? 'E' : facing === 'N' ? 'S' : facing) : facing;
  if (state && state !== shapeStates(s)[0] && BLOCK_SHAPES[s].states[state]) part.state = state;
  if (at) part.at = at;
  return part;
}
// A palette row's block (core/region.js): its parts' rows.
export function blockFromRows(rows) {
  const [first, ...rest] = rows.map(r => ({ materialId: r[0], shape: r[1], facing: r[2], state: r[3], at: r[4] }));
  return newBlock(first.materialId, first.shape, first.facing, first.state, first.at, rest);
}
// A block as a palette row: its parts' rows.
export const blockRows = block => tileParts(block).map(partRow);

// --- Building on the Grid (tools/editor.js) ---
//
// The Grid cells a part takes, as bits x + 3y + 9z: those its ray box covers
// (codeBox) - a slope or steps the whole of its span.
export function partCells(part) {
  const b = codeBox(shapeCode(part)).map(v => Math.round(v * GRID));
  let mask = 0;
  for (let z = b[4]; z < b[5]; z++) for (let y = b[2]; y < b[3]; y++) for (let x = b[0]; x < b[1]; x++) mask |= 1 << cellAt(x, y, z);
  return mask;
}
// A part's Grid cell for a click in Grid cell g ([x, y, z] of the tile): as
// near g as the part goes and still stays in its tile - a slab at g's
// height, a pane at its column, a slope or steps as high as there is room.
export function gridAt(part, g) {
  const b = codeBox(shapeCode({ ...part, at: 0 })).map(v => Math.round(v * GRID));
  return cellAt(...[0, 1, 2].map(a => Math.max(0, Math.min(g[a], GRID - (b[2 * a + 1] - b[2 * a])))));
}
// A block with a part added - null when it would not fit: a Grid cell
// already taken, or MAX_PARTS there already.
export function withPart(block, part) {
  if (!block) return newBlock(part.materialId, part.shape, part.facing, part.state, part.at);
  const parts = tileParts(block);
  if (parts.length >= MAX_PARTS || !partFits(part)) return null;
  const taken = parts.reduce((m, p) => m | partCells(p), 0);
  if (taken & partCells(part)) return null;
  const [first, ...rest] = [...parts, part];
  return { ...newBlock(first.materialId, first.shape, first.facing, first.state, first.at, rest),
           walkable: block.walkable, triggerId: block.triggerId };
}
// A block with its part i taken out - null when that was its last.
export function withoutPart(block, i) {
  const parts = tileParts(block).filter((p, k) => k !== i);
  if (!parts.length) return null;
  const [first, ...rest] = parts;
  return { ...newBlock(first.materialId, first.shape, first.facing, first.state, first.at, rest),
           walkable: block.walkable, triggerId: block.triggerId };
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
        palette.push(blockRows(b));
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
  const blocks = level.palette.map(row => row && blockFromRows(row));
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
// Every part too, as palette rows (blockRows): a slab or a slope bakes as its
// own boxes, a stack's parts each in their material's channel.
export function worldMirrorEntries(world = World) {
  const out = [];
  for (const [key, block] of world) out.push([key, blockRows(block)]);
  return out;
}

export function loadWorldMirror(entries, world = World) {
  world.clear();
  for (const [key, rows] of entries) world.set(key, blockFromRows(rows));
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

// Slab test bed, beside the raised platform (28-31, y 1, 28-31): a slab and
// then two stacked on its west edge make it reachable from the ground (1 ->
// 1 1/3 -> 1 2/3 -> 2 blocks), and an arch of two pillars under a slab
// lintel at the top of its tile leaves 1 2/3 blocks (2.5 m) over the ground
// to walk through.
function addHalfTileTest() {
  const slab = (at = 0, more = null) => newBlock('grass', 'slab', null, null, at, more);
  for (const z of [29, 30]) {
    World.set(getVoxelKey(26, 1, z), slab());
    World.set(getVoxelKey(27, 1, z), slab(0, [{ materialId: 'grass', shape: 'slab', at: 3 }]));
  }
  for (const z of [24, 26]) for (let y = 1; y <= 2; y++) World.set(getVoxelKey(24, y, z), newBlock('grass'));
  World.set(getVoxelKey(24, 2, 25), slab(6));
}

// Steps test bed: marble steps up the west side of the grass wall at x = 20
// (two blocks tall, top at level 3), along z = 12 - ground (1) -> steps at
// (18, 1) (1 1/3 to 2) -> steps at (19, 2) on a marble block (2 1/3 to 3) ->
// the wall's top (3). Beside the wall torch at (19, 2, 11).
function addStairTest() {
  const put = (x, y, z, shape = null, facing = null) =>
    World.set(getVoxelKey(x, y, z), newBlock('marble', shape, facing));
  put(18, 1, 12, 'steps', 'E');
  put(19, 1, 12);
  put(19, 2, 12, 'steps', 'E');
}

// Slope test bed, also onto the raised platform: a high slope on its north
// side at (29, 1, 27), climbing south (+z) from the ground (level 1) to the
// platform (level 2) in one cell; and a gentle ramp on its east side, a
// medium slope at (33, 1, 29) and a low one on two slabs at (32, 1, 29),
// climbing west over two cells.
function addSlopeTest() {
  const slab = at => ({ materialId: 'grass', shape: 'slab', at });
  World.set(getVoxelKey(29, 1, 27), newBlock('grass', 'slope', 'S', 'high'));
  World.set(getVoxelKey(33, 1, 29), newBlock('grass', 'slope', 'W', 'medium'));
  World.set(getVoxelKey(32, 1, 29), newBlock('grass', 'slope', 'W', 'low', 6, [slab(0), slab(3)]));
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
