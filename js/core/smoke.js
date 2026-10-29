// --- Cloud spells: a spell's volume of smoke (Fog Cloud and the rest) ---
//
// A cloud is cubes on the BoxGrid (core/boxgrid.js) - three a block, half a
// metre each - around the emitter: the middle of the top face of the block it
// was cast at.
//
// Its shape is a sphere of cubes. Which cubes make a round sphere (Red Blob
// Games, "Circle fill on a grid"; the same in 3D): a cube is in when its
// CENTRE is within the radius, and the sphere looks round when its edge falls
// on a boundary between cells - a half-integer radius about a cell centre, an
// integer one about a cell corner (an integer radius about a centre leaves a
// single cube at each pole, the "nub"). The emitter is a cube centre across
// (the block's middle cube) but a cube corner up and down (the top face), so
// the measure is an ellipsoid: across the spell's radius rh (n + 0.5), up and
// down rv = rh + 0.5. A cube is in the sphere at measure <= 1.
//
// Its VOLUME is the sphere's cubes less the part meant to be underground: a
// cube below the emitter's plane that is in solid is simply not there - it
// never spills up and makes the cloud bigger than its radius. Every other
// cube of the sphere is smoke that must go somewhere.
//
// It spreads as CS2's smoke does (its recreations: a flood fill out from the
// grenade, each step through air, bounded by the smoke's reach): by how far
// the smoke must TRAVEL through air to get somewhere, not how far away it is
// in a straight line. The cloud is taken cube by cube, least travel first,
// until it holds its volume - so it bends round corners, pours through doors,
// runs down a corridor past its radius when that is where its volume has to
// go, and never passes through solid. Travel is an any-angle path (Theta*: a
// cube seen in a straight line through air from its parent's anchor is
// reached straight from that anchor), measured in the sphere's own ellipsoid.
// So wherever the emitter is in plain view the travel IS the straight-line
// measure, and in the open the cloud is exactly the sphere above; behind a
// corner it is the way round. The cloud rolls out by the same travel, so the
// smoke visibly flows round corners.
//
// Sizes follow Baldur's Gate 3 (bg3.wiki): its radii in metres, snapped down
// to the nearest half-integer number of cubes (a round sphere). Only Fog Cloud
// grows when upcast, by BG3's measured radii (4.5, 6, 8, 10, 10.5, 12 m for
// levels 1-6), then half a metre a level to 9th - BG3 stops at 6th. The rest
// keep one size, as their rules do: they upcast damage, not area.
//
// Smoke takes up space, and so does everything else (2026-10-30):
//   - Models block it as blocks do: a cell more than a quarter filled by a
//     placed model's cubes is solid to the smoke (a carpet, a quarter of
//     its cell, is not).
//   - Clouds are exclusive. A cloud is cast round every other cloud's cells
//     as round solid, so two clouds side by side grow round each other.
//   - Sprites displace it. Characters and props stand in the cloud as bodies
//     (boxes; render/features/smoke.js gathers them). The cells they cover
//     hold no smoke, and the smoke they push out goes to the cloud's rim, so
//     it keeps its volume. The cast's fill runs on past the volume into a
//     reserve of spare cells, in the same travel order; the cloud is always
//     the first `volume` of them that nothing occupies (reshapeSmokeCloud).
//     A body walking in pushes smoke to the rim; walking out, the hole
//     fills and the rim draws back. The travel is the cast's: smoke is not
//     re-routed round a body, only moved.
//
// Headless: gameplay reads the blocks (line of sight), the renderer draws the
// cubes (render/features/smoke.js).
import { BLOCK_METRES, isSolidAt, partAt, ModelBoxes } from './world.js';
import { isGlassMaterial } from './materials.js';
import { BOX_CELLS, BOX_METRES } from './boxgrid.js';

// A cube is a box of the BoxGrid.
export const SMOKE_CUBES = BOX_CELLS;

export const SMOKE_CUBE = BOX_METRES;

// Fog Cloud's radius at 1st level, in cubes: 4.25 m (BG3: 4.5 m), 17 across.
export const SMOKE_RADIUS = 8.5;

export const MAX_CAST_LEVEL = 9;

// The cloud spells: name, level (its own, the lowest it can be cast at),
// radius in cubes (n + 0.5) - one, or radii from its own level up to 9th -
// colour (linear, the smoke's albedo) and density (extinction per metre
// through a full cube: how dark its shadow is and how much it shades itself).
// One renderer draws them all.
export const CLOUD_SPELLS = {
  fogCloud: { name: 'Fog Cloud', level: 1, colour: [0.82, 0.82, 0.84], density: 1.6,
              radii: [SMOKE_RADIUS, 11.5, 15.5, 19.5, 20.5, 23.5, 24.5, 25.5, 26.5] },
  darkness: { name: 'Darkness', level: 2, radius: 6.5, colour: [0.04, 0.04, 0.05], density: 6 },
  stinkingCloud: { name: 'Stinking Cloud', level: 3, radius: SMOKE_RADIUS, colour: [0.58, 0.66, 0.26], density: 1.4 },
  cloudkill: { name: 'Cloudkill', level: 5, radius: SMOKE_RADIUS, colour: [0.62, 0.72, 0.42], density: 1.8 },
  incendiaryCloud: { name: 'Incendiary Cloud', level: 8, radius: SMOKE_RADIUS, colour: [0.32, 0.28, 0.26], density: 2.4 }
};

// A spell's radius in cubes cast at `level` (its own if lower, 9th at most).
export function smokeRadius(spell, level = spell.level) {
  if (!spell.radii) return spell.radius;
  const at = Math.max(spell.level, Math.min(level, MAX_CAST_LEVEL)) - spell.level;
  return spell.radii[Math.min(at, spell.radii.length - 1)];
}

// The largest radius any spell reaches.
export const SMOKE_MAX_RADIUS = Math.max(...Object.values(CLOUD_SPELLS).map(s => smokeRadius(s, MAX_CAST_LEVEL)));

// A cloud's box on the BoxGrid, in boxes a side, and its min corner from the
// box over the emitter (the first above the top face, under its middle): the
// most a cloud can fill, and the renderer's grid for shadows through it.
export const SMOKE_BOX = 64;
export const SMOKE_BOX_FROM = -32;

// Seconds for Fog Cloud at its own level to roll out to its edge (a bigger
// cloud takes longer by its radius), and for one cube to grow in.
export const SMOKE_GROW_SECONDS = 1.5;
export const SMOKE_GROW_IN = 0.3;

// Half-metre box equivalents along a ray that fully block sunlight.
export const SMOKE_SUN_BOXES = 8;

// A cube's birth for one the cast made (its growth is the roll-out's), and a
// death for one still there - seconds on the smoke's clock, far enough out
// that a float32 still tells them from a moment later.
export const SMOKE_CAST_BORN = -1e4;
export const SMOKE_NEVER_DIES = 1e6;

// Spare cells past a cloud's volume, for the smoke bodies push out: a share
// of the volume, at least SPARE_MIN, at most SPARE_MAX.
export const SMOKE_SPARE_SHARE = 0.25;
const SPARE_MIN = 256, SPARE_MAX = 8192;

// A BoxGrid cell's key, for the sets of occupied cells: one number, x and z
// within +-16384 cells (8 km), y within +-512 (256 m).
const KO = 16384, KS = 32768, YO = 512, YS = 1024;
export const cellKey = (bx, by, bz) => ((bx + KO) * KS + (bz + KO)) * YS + (by + YO);
export function cellOf(key) {
  const by = key % YS, rest = (key - by) / YS, bz = rest % KS, bx = (rest - bz) / KS;
  return [bx - KO, by - YO, bz - KO];
}

// The cells whose centres lie inside a box [x0, y0, z0, x1, y1, z1] in
// metres - what a body occupies. Cell (bx, by, bz) is centred on
// (bx, by, bz) x SMOKE_CUBE (core/boxgrid.js).
export function bodyCells(box, out = new Set()) {
  const lo = d => Math.floor(box[d] / SMOKE_CUBE) + 1, hi = d => Math.ceil(box[d + 3] / SMOKE_CUBE) - 1;
  for (let bz = lo(2); bz <= hi(2); bz++) for (let by = lo(1); by <= hi(1); by++) for (let bx = lo(0); bx <= hi(0); bx++) {
    out.add(cellKey(bx, by, bz));
  }
  return out;
}

// A sprite as a body: its picture's opaque pixels, row by row - how far
// from the upright axis through its middle each row reaches - turned about
// that axis. A billboard turns to face the camera, so what it holds in the
// world is that solid of revolution, not its box: a thin neck, the gap
// under an arm, a trunk under a crown are all smoke. rgba: its pixels (w x
// h, top row first); planeW x planeH: its quad in metres. Returns { rowH,
// radii } - metres a row, and each row's reach in metres, bottom row first
// (0: nothing opaque).
export function silhouetteRadii(rgba, w, h, planeW, planeH, alphaMin = 128) {
  const radii = new Float32Array(h), pxW = planeW / w;
  for (let row = 0; row < h; row++) {
    let reach = 0;
    for (let col = 0; col < w; col++) {
      if (rgba[(row * w + col) * 4 + 3] < alphaMin) continue;
      reach = Math.max(reach, Math.abs(col - w / 2), Math.abs(col + 1 - w / 2));
    }
    radii[h - 1 - row] = reach * pxW;
  }
  return { rowH: planeH / h, radii };
}

// The cells a turned silhouette holds, standing at (x, y, z) - its feet:
// in each layer of cells, those a disc of that layer's widest reach
// touches. { x, y, z, rowH, radii } as silhouetteRadii gives them.
export function profileCells({ x, y, z, rowH, radii }, out = new Set()) {
  const top = y + rowH * radii.length, H = SMOKE_CUBE / 2;
  for (let by = Math.ceil(y / SMOKE_CUBE - 0.5); by * SMOKE_CUBE - H < top; by++) {
    const lo = Math.max(by * SMOKE_CUBE - H, y), hi = Math.min(by * SMOKE_CUBE + H, top);
    if (hi <= lo) continue;
    let r = 0;
    for (let row = Math.floor((lo - y) / rowH); row < Math.min(radii.length, Math.ceil((hi - y) / rowH)); row++) {
      r = Math.max(r, radii[row]);
    }
    if (r <= 0) continue;
    const span = (c, at) => Math.max(Math.abs(at - c * SMOKE_CUBE) - H, 0);
    for (let bz = Math.ceil((z - r - H) / SMOKE_CUBE); bz <= Math.floor((z + r + H) / SMOKE_CUBE); bz++) {
      for (let bx = Math.ceil((x - r - H) / SMOKE_CUBE); bx <= Math.floor((x + r + H) / SMOKE_CUBE); bx++) {
        const dx = span(bx, x), dz = span(bz, z);
        if (dx * dx + dz * dz < r * r) out.add(cellKey(bx, by, bz));
      }
    }
  }
  return out;
}

// What a body occupies, either kind: a box [x0, y0, z0, x1, y1, z1], or a
// turned silhouette { x, y, z, rowH, radii }.
export const occupiedCells = (body, out = new Set()) =>
  (Array.isArray(body) ? bodyCells(body, out) : profileCells(body, out));

// A cell more than this share filled by model cubes is solid to the smoke.
export const SMOKE_MODEL_FILL = 0.25;
const TEXELS = 12 / BOX_CELLS;       // world texels along a cell: 4

// The cells of BoxGrid box [lo, lo + size) (indices) that placed models fill
// past SMOKE_MODEL_FILL - one pass over the model cubes, at a cast. boxes:
// world.js ModelBoxes ([x0, y0, z0, x1, y1, z1] in world texels).
export function modelCells(lo, size, boxes = ModelBoxes) {
  const fill = new Map();
  // Cell b spans texels [TEXELS b - TEXELS / 2, TEXELS b + TEXELS / 2).
  const t0 = lo.map(v => TEXELS * v - TEXELS / 2), t1 = lo.map(v => TEXELS * (v + size) - TEXELS / 2);
  for (const b of boxes) {
    if (b[3] <= t0[0] || b[0] >= t1[0] || b[4] <= t0[1] || b[1] >= t1[1] || b[5] <= t0[2] || b[2] >= t1[2]) continue;
    const c0 = [0, 1, 2].map(d => Math.floor((Math.max(b[d], t0[d]) + TEXELS / 2) / TEXELS));
    const c1 = [0, 1, 2].map(d => Math.ceil((Math.min(b[d + 3], t1[d]) + TEXELS / 2) / TEXELS) - 1);
    for (let z = c0[2]; z <= c1[2]; z++) for (let y = c0[1]; y <= c1[1]; y++) for (let x = c0[0]; x <= c1[0]; x++) {
      const span = (d, v) => Math.max(0, Math.min(b[d + 3], TEXELS * v + TEXELS / 2) - Math.max(b[d], TEXELS * v - TEXELS / 2));
      const v = span(0, x) * span(1, y) * span(2, z);
      if (v > 0) { const k = cellKey(x, y, z); fill.set(k, (fill.get(k) || 0) + v); }
    }
  }
  const out = new Set();
  for (const [k, v] of fill) if (v > SMOKE_MODEL_FILL * TEXELS ** 3) out.add(k);
  return out;
}

// The sphere's measure at cube offset (i, j, k) - centre at emitter + (i, j +
// 0.5, k) cubes; in at <= 1.
export const sphereMeasure = (i, j, k, rh, rv = rh + 0.5) => (i * i + k * k) / (rh * rh) + ((j + 0.5) ** 2) / (rv * rv);

// The sphere's cubes as offsets from the emitter, [i, j, k].
export function sphereCubes(radius = SMOKE_RADIUS) {
  const rh = radius, rv = radius + 0.5;
  const out = [];
  const n = Math.ceil(rv);
  for (let j = -n; j < n; j++) {
    for (let k = -Math.floor(rh); k <= Math.floor(rh); k++) {
      for (let i = -Math.floor(rh); i <= Math.floor(rh); i++) {
        if (sphereMeasure(i, j, k, rh, rv) <= 1) out.push([i, j, k]);
      }
    }
  }
  return out;
}

export const SMOKE_SPHERE = sphereCubes();

// A binary heap of cells by [measure, index] - least first, ties by index, so
// a cloud is the same every time it is cast.
function heap() {
  const m = [], id = [];
  const less = (a, b) => m[a] < m[b] || (m[a] === m[b] && id[a] < id[b]);
  const swap = (a, b) => { [m[a], m[b]] = [m[b], m[a]]; [id[a], id[b]] = [id[b], id[a]]; };
  return {
    get size() { return m.length; },
    push(measure, index) {
      m.push(measure); id.push(index);
      for (let i = m.length - 1; i > 0;) {
        const p = (i - 1) >> 1;
        if (!less(i, p)) break;
        swap(i, p); i = p;
      }
    },
    pop() {
      const top = id[0], last = m.length - 1;
      swap(0, last); m.pop(); id.pop();
      for (let i = 0; ;) {
        const l = 2 * i + 1, r = l + 1;
        let s = i;
        if (l < m.length && less(l, s)) s = l;
        if (r < m.length && less(r, s)) s = r;
        if (s === i) break;
        swap(i, s); i = s;
      }
      return top;
    }
  };
}

const FACES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

// Whether the straight line between points p and q (emitter-relative cube
// coordinates: cube (i, j, k) is [i - 0.5, i + 0.5] x [j, j + 1] x [k - 0.5,
// k + 0.5]) passes only through air. Where the line runs exactly along an
// edge or through a corner, every cube touching it must be air - smoke does
// not squeeze diagonally between two solids.
function lineOfSight(p, q, solid) {
  // Scalars, not arrays: it runs for every step of the spread (thousands a
  // cast), and the arrays it allocated were most of a cast's cost.
  const ox = p[0] + 0.5, oy = p[1], oz = p[2] + 0.5;
  const dx = q[0] + 0.5 - ox, dy = q[1] - oy, dz = q[2] + 0.5 - oz;
  if (Math.hypot(dx, dy, dz) < 1e-9) return true;
  const nudge = v => (v > 0 ? 1e-9 : v < 0 ? -1e-9 : 0);
  const sgn = v => (v > 0 ? 1 : v < 0 ? -1 : 0);
  // Start in the cube the line leaves p into; end in the one it reaches q in.
  let cx = Math.floor(ox + nudge(dx)), cy = Math.floor(oy + nudge(dy)), cz = Math.floor(oz + nudge(dz));
  const lx = Math.floor(q[0] + 0.5 - nudge(dx)), ly = Math.floor(q[1] - nudge(dy)), lz = Math.floor(q[2] + 0.5 - nudge(dz));
  const sx = sgn(dx), sy = sgn(dy), sz = sgn(dz);
  let tx = sx ? ((sx > 0 ? cx + 1 : cx) - ox) / dx : Infinity;
  let ty = sy ? ((sy > 0 ? cy + 1 : cy) - oy) / dy : Infinity;
  let tz = sz ? ((sz > 0 ? cz + 1 : cz) - oz) / dz : Infinity;
  const ux = dx ? Math.abs(1 / dx) : Infinity, uy = dy ? Math.abs(1 / dy) : Infinity, uz = dz ? Math.abs(1 / dz) : Infinity;
  for (let n = 0; n < 1024; n++) {
    if (solid(cx, cy, cz)) return false;
    if (cx === lx && cy === ly && cz === lz) return true;
    const t = Math.min(tx, ty, tz);
    if (t > 1 + 1e-9) return true;
    const ax = tx <= t + 1e-9, ay = ty <= t + 1e-9, az = tz <= t + 1e-9;
    if (ax + ay + az > 1) {
      // Through an edge or corner: each cube beside it, stepped on one axis.
      if (ax && solid(cx + sx, cy, cz)) return false;
      if (ay && solid(cx, cy + sy, cz)) return false;
      if (az && solid(cx, cy, cz + sz)) return false;
    }
    if (ax) { cx += sx; tx += ux; }
    if (ay) { cy += sy; ty += uy; }
    if (az) { cz += sz; tz += uz; }
  }
  return true;
}

// tile: { gx, gy, gz }, the block cast at. solidAt(wx, wy, wz): whether a
// world point is in solid (world.js isSolidAt; a test passes its own).
// glassAt(wx, wy, wz): whether that solid is glass - seen through, so a face
// pressed on it is open as one on air is (else the cloud shows hollow there).
// spell: a CLOUD_SPELLS entry; level: the level it is cast at.
// taken: the cells other clouds hold (a Set of cellKey, or one a cloud) - solid to this one,
// which grows round them. blocked: cells bodies occupy now (a Set) - the
// cloud is cast round them as it is reshaped round them (reshapeSmokeCloud).
// models: the model cubes (world.js ModelBoxes).
// Returns { tile, spell, level, radius (cubes), centre (the emitter, world
// metres), box (the BoxGrid index of its box's min corner), volume (cubes it
// holds), cubes: [[wx, wy, wz, travel, bx, by, bz, air, against, enclosed]], blocks: Map
// 'x,y,z' -> cubes of it, gMax (the farthest travel taken), growSeconds } -
// cubes in the order taken; travel in the ellipsoid's measure (1 at the rim
// in the open); air 1 for a cube with a face open to air that is not smoke
// or glass (always seen); against, for one without, the faces (bit f of FACES) it
// presses on solid with - seen only while that solid is lifted away (the
// interiors take roofs and near walls off). A cube with neither is inside.
// enclosed: the travel at which the last of its smoke neighbours rolls in
// (Infinity for one open to air) - while the cloud rolls out, every cube is
// seen from when it arrives until then, so the front grows layer by layer.
// Glass in the World: the part of its block at the point is a glass material.
function glassAtWorld(wx, wy, wz) {
  const part = partAt(wx, wy, wz);
  return !!part && isGlassMaterial(part.materialId);
}

export function createSmokeCloud(tile, solidAt = isSolidAt, spell = CLOUD_SPELLS.fogCloud, level = spell.level,
                                 glassAt = glassAtWorld, { taken: others = null, blocked = null, models = ModelBoxes } = {}) {
  const { gx, gy, gz } = tile;
  const rh = smokeRadius(spell, level), rv = rh + 0.5;
  const centre = [gx * BLOCK_METRES, (gy + 0.5) * BLOCK_METRES, gz * BLOCK_METRES];
  // The box over the emitter: under the block's middle, just above its top.
  const over = [SMOKE_CUBES * gx, SMOKE_CUBES * gy + 2, SMOKE_CUBES * gz];
  const box = over.map(v => v + SMOKE_BOX_FROM);
  const B = SMOKE_BOX, F = -SMOKE_BOX_FROM;
  const at = (i, j, k) => (i + F) + B * ((j + F) + B * (k + F));
  const inBox = (i, j, k) => i >= -F && j >= -F && k >= -F && i < B - F && j < B - F && k < B - F;
  const pos = (i, j, k) => [centre[0] + i * SMOKE_CUBE, centre[1] + (j + 0.5) * SMOKE_CUBE, centre[2] + k * SMOKE_CUBE];
  // 1 a block's solid, 3 a block's glass, 2 a model's or another cloud's: all
  // walls to the smoke's spread, but only a block's face is hidden with it (a
  // face on glass, a model or other smoke is seen).
  const mCells = models && models.length ? modelCells(box, B, models) : null;
  const otherSets = others instanceof Set ? [others] : others || [];
  const solidCache = new Int8Array(B * B * B).fill(-1);
  const wall = (i, j, k) => {
    if (!inBox(i, j, k)) return 1;
    const n = at(i, j, k);
    if (solidCache[n] < 0) {
      const p = pos(i, j, k), key = cellKey(over[0] + i, over[1] + j, over[2] + k);
      solidCache[n] = solidAt(p[0], p[1], p[2]) ? (glassAt(p[0], p[1], p[2]) ? 3 : 1)
        : (mCells && mCells.has(key)) || otherSets.some(o => o.has(key)) ? 2 : 0;
    }
    return solidCache[n];
  };
  const solid = (i, j, k) => wall(i, j, k) !== 0;
  // The volume: the sphere less what is meant to be underground - below the
  // emitter's plane, a cube counts only where the column down to it from the
  // plane is open air (off a ledge, a pillar's top): air under the ground
  // (a level is often one layer of blocks over nothing) is underground too.
  const open = (i, j, k) => { for (let y = -1; y >= j; y--) if (solid(i, y, k)) return false; return true; };
  let volume = 0;
  for (const [i, j, k] of sphereCubes(rh)) if (j >= 0 || open(i, j, k)) volume++;

  // Travel in the ellipsoid's measure between emitter-relative points.
  const dist = (p, q) => Math.sqrt(((q[0] - p[0]) ** 2 + (q[2] - p[2]) ** 2) / (rh * rh) + ((q[1] - p[1]) ** 2) / (rv * rv));
  const centreOf = (i, j, k) => [i, j + 0.5, k];
  const N = B * B * B;
  const spare = Math.min(SPARE_MAX, Math.max(SPARE_MIN, Math.ceil(volume * SMOKE_SPARE_SHARE)));
  const g = new Float32Array(N).fill(Infinity);
  const anchor = new Array(N);                 // the point each cube is reached straight from
  const taken = new Uint8Array(N);
  const order = [];
  const emitter = [0, 0, 0];
  if (!solid(0, 0, 0)) {
    const frontier = heap();
    const start = at(0, 0, 0);
    g[start] = dist(emitter, centreOf(0, 0, 0));
    anchor[start] = emitter;
    frontier.push(g[start], start);
    while (frontier.size && order.length < volume + spare) {
      const n = frontier.pop();
      if (taken[n]) continue;
      const i = (n % B) - F, j = (Math.floor(n / B) % B) - F, k = Math.floor(n / (B * B)) - F;
      taken[n] = 1;
      order.push([i, j, k, g[n]]);
      const here = centreOf(i, j, k);
      for (const [di, dj, dk] of FACES) {
        const a = i + di, b = j + dj, c = k + dk;
        if (solid(a, b, c)) continue;
        const m = at(a, b, c);
        if (taken[m]) continue;
        const there = centreOf(a, b, c);
        // Straight from this cube's anchor if it can see the next cube; else
        // via this cube, which becomes the next one's anchor.
        let via = anchor[n], cost = g[n] - dist(anchor[n], here) + dist(anchor[n], there);
        if (!lineOfSight(anchor[n], there, solid)) { via = here; cost = g[n] + dist(here, there); }
        if (cost < g[m] - 1e-9) { g[m] = cost; anchor[m] = via; frontier.push(cost, m); }
      }
    }
  }
  // The cast's shape: its first `volume` cells with nothing in them. gMax
  // is the farthest of those - the roll-out's end, fixed from here on.
  let gMax = 0, n = 0;
  for (const [i, j, k, travel] of order) {
    if (n >= volume) break;
    if (blocked && blocked.has(cellKey(over[0] + i, over[1] + j, over[2] + k))) continue;
    if (travel > gMax) gMax = travel;
    n++;
  }
  const cloud = { tile: { gx, gy, gz }, spell, level, radius: rh, centre, box, volume, cubes: [], blocks: new Map(),
                  // Cubes that left, shrinking out: { cube, died }.
                  fading: [],
                  gMax, growSeconds: SMOKE_GROW_SECONDS * rh / SMOKE_RADIUS,
                  // Every cell it may hold: its volume and the spare past it.
                  candidates: order.length };
  // What reshaping needs - the cells in travel order and the walls round
  // them, and what it keeps between reshapes. Not enumerable: not the
  // cloud's data.
  const idx = Int32Array.from(order, ([i, j, k]) => at(i, j, k));   // order entry -> grid index
  const ord = new Int32Array(N).fill(-1);                            // grid index -> order entry
  idx.forEach((m, n) => { ord[m] = n; });
  // Its block's key per cell (for smokeAt), made once.
  const blockKey = order.map(([i, j, k]) => pos(i, j, k).map(v => Math.round(v / BLOCK_METRES)).join());
  Object.defineProperty(cloud, 'shape', { value: {
    order, idx, ord, blockKey, over, wall, pos, at, inBox, g,
    mark: new Uint8Array(N),            // occupied by a body or another cloud (zero between reshapes)
    next: new Uint8Array(N),            // the cells being taken (zero between reshapes)
    cells: new Uint8Array(N),           // the cells it holds
    cubeAt: new Int32Array(N).fill(-1), // grid index -> its cube in cloud.cubes
    list: [],                           // the grid indices it holds
    castEnd: Math.min(volume, order.length),   // its shape with nothing in it: the first this many
    covered: new Set(),                 // grid indices of cubes kept drawn while covered (c[12])
    set: new Set()                      // the same, as cellKeys
  } });
  reshapeSmokeCloud(cloud, blocked);
  return cloud;
}

// A cloud's cells again, round the bodies in the cells `blocked` (a Set of
// cellKey, or null) and other clouds' (`taken`: Sets of cellKey, or one):
// the first `volume` of its cells in travel order that none holds. Returns
// whether any cell changed. Cells past the cast's farthest travel are listed
// at it: they appear as the roll-out ends (or at once, after it).
//
// Incremental: only the cells that came or went change in cubes, blocks and
// the cell set, and only they and their neighbours have their faces looked
// at again - a body walking through a big cloud moves a few dozen cells, not
// the whole of it. cubes is changed in place (a cube that goes takes the
// last one's place); a cube's c[10] is its grid index, c[11] when it was born.
//
// now: the smoke's clock (seconds), for the cubes that come and go - one
// that comes is born now and grows in; one that goes joins cloud.fading,
// died now, and shrinks out (pruneFading drops it after). null at the cast:
// its cubes grow as the cloud rolls out.
//
// settle: false while bodies are moving - the cloud only loses the cells
// they take and refills inside its cast shape, leaving the rim (and the
// volume) be; true once they have stood still (render/features/smoke.js),
// and at the cast - the settled shape above.
export function reshapeSmokeCloud(cloud, blocked = null, taken = null, now = null, settle = true) {
  const S = cloud.shape;
  const { order, idx, ord, blockKey, over, wall, pos, at, inBox, g, mark, next, cells, cubeAt } = S;
  const B = SMOKE_BOX, F = -SMOKE_BOX_FROM;
  // The occupied cells inside its box, marked (and unmarked after).
  const marked = [];
  for (const set of [blocked, ...(taken instanceof Set ? [taken] : taken || [])]) {
    if (!set) continue;
    for (const key of set) {
      const [bx, by, bz] = cellOf(key), i = bx - over[0], j = by - over[1], k = bz - over[2];
      if (!inBox(i, j, k)) continue;
      const m = at(i, j, k);
      if (!mark[m]) { mark[m] = 1; marked.push(m); }
    }
  }
  if (settle) {
    // Settled: the first `volume` free cells - the rim grows or draws back
    // to keep the volume.
    let count = 0;
    for (let n = 0; n < idx.length && count < cloud.volume; n++) {
      const m = idx[n];
      if (!mark[m]) { next[m] = 1; count++; }
    }
  } else {
    // Moving: every cube keeps its cell unless something now stands in it,
    // and every free cell of the cast's shape (its first cells in travel
    // order, as cast) is smoke. The rim is left alone: the cells a body
    // covers come and go as it walks (a column or two as it straddles a cell
    // edge), and a rim following that grew and shrank in place with every
    // step. The volume is short by what the body holds - or over, by a rim
    // grown before - until things settle.
    for (const m of S.list) if (!mark[m]) next[m] = 1;
    for (let n = 0; n < S.castEnd; n++) {
      const m = idx[n];
      if (!mark[m]) next[m] = 1;
    }
  }
  const take = [];
  for (let n = 0; n < idx.length; n++) if (next[idx[n]]) take.push(idx[n]);
  for (const m of marked) mark[m] = 0;
  const added = take.filter(m => !cells[m]), removed = S.list.filter(m => !next[m]);
  for (const m of take) next[m] = 0;
  if (!added.length && !removed.length) return false;

  const cubes = cloud.cubes, blocks = cloud.blocks;
  const keyAt = m => { const [i, j, k] = order[ord[m]]; return cellKey(over[0] + i, over[1] + j, over[2] + k); };
  for (const m of removed) {
    cells[m] = 0;
    if (now !== null) cloud.fading.push({ cube: cubes[cubeAt[m]], died: now });
    const at_ = cubeAt[m], last = cubes.pop();
    if (at_ < cubes.length) { cubes[at_] = last; cubeAt[last[10]] = at_; }
    cubeAt[m] = -1;
    const bk = blockKey[ord[m]], left = blocks.get(bk) - 1;
    if (left > 0) blocks.set(bk, left); else blocks.delete(bk);
    S.set.delete(keyAt(m));
  }
  for (const m of added) {
    cells[m] = 1;
    cubeAt[m] = cubes.length;
    cubes.push(null);
    const bk = blockKey[ord[m]];
    blocks.set(bk, (blocks.get(bk) || 0) + 1);
    S.set.add(keyAt(m));
  }
  S.list = take;

  // Faces again: of every cube that came, and of the cubes beside any change.
  const gMax = cloud.gMax;
  const cubeFor = (m, born) => {
    const [i, j, k, travel0] = order[ord[m]];
    const travel = Math.min(travel0, gMax);
    const p = pos(i, j, k);
    // Its faces: open to air that is not smoke, to a model, another cloud, a
    // body or glass (always seen), or against a block's solid (seen only once
    // that solid is lifted away by the interiors). And when it is enclosed:
    // the travel at which its last smoke neighbour rolls in - while the cloud
    // grows, a cube is seen until then (the front rolls out layer by layer);
    // one open to air never is.
    let air = 0, against = 0, enclosed = travel;
    for (let f = 0; f < 6; f++) {
      const [di, dj, dk] = FACES[f], a = i + di, b = j + dj, c = k + dk;
      if (inBox(a, b, c) && cells[at(a, b, c)]) {
        enclosed = Math.max(enclosed, Math.min(g[at(a, b, c)], gMax));
        continue;
      }
      if (wall(a, b, c) === 1) against |= 1 << f; else air = 1;
    }
    return [p[0], p[1], p[2], travel, over[0] + i, over[1] + j, over[2] + k, air, air ? 0 : against,
            air ? Infinity : enclosed, m, born];
  };
  const touch = new Set(added);
  for (const m of removed.concat(added)) {
    const i = (m % B) - F, j = (Math.floor(m / B) % B) - F, k = Math.floor(m / (B * B)) - F;
    for (const [di, dj, dk] of FACES) {
      if (!inBox(i + di, j + dj, k + dk)) continue;
      const q = at(i + di, j + dj, k + dk);
      if (cells[q]) touch.add(q);
    }
  }
  // A cube looked at again keeps its birth; one that came is born now. One
  // that was seen and is now covered (a newcomer beside it) stays drawn until
  // the newcomer has grown over it - c[12], on the smoke's clock - or the
  // cloud would be seen through while it grows, its inside never drawn.
  for (const m of touch) {
    const prev = cubes[cubeAt[m]];
    const c = cubeFor(m, prev ? prev[11] : now === null ? SMOKE_CAST_BORN : now);
    c[12] = prev ? prev[12] : 0;
    if (now !== null && prev && prev[7] && !c[7]) c[12] = Math.max(c[12], now + SMOKE_GROW_IN + SMOKE_COVER_MARGIN);
    if (c[12]) S.covered.add(m);
    cubes[cubeAt[m]] = c;
  }
  return true;
}

// Seconds past a newcomer's growing in that a cube it covers stays drawn.
const SMOKE_COVER_MARGIN = 0.05;

// Covered cubes whose time to be seen has passed lose it (c[12] = 0), so the
// list can drop them. Returns whether any did.
export function expireCovered(cloud, now) {
  const S = cloud.shape;
  let any = false;
  for (const m of S.covered) {
    const at_ = S.cubeAt[m], c = at_ >= 0 ? cloud.cubes[at_] : null;
    if (c && c[12] > now) continue;
    if (c) c[12] = 0;
    S.covered.delete(m);
    any = true;
  }
  return any;
}

// Drops the cubes that have finished shrinking out. Returns whether any went.
export function pruneFading(cloud, now) {
  const n = cloud.fading.length;
  cloud.fading = cloud.fading.filter(f => now - f.died < SMOKE_GROW_IN);
  return cloud.fading.length !== n;
}

// The cells a cloud holds now (a Set of cellKey, kept by the cloud - not to
// be changed) - for the other clouds to keep out of.
export const smokeCells = cloud => cloud.shape.set;

// The second a cube that travelled `travel` appears, after the cast.
export const smokeAppear = (cloud, travel) => (travel / Math.max(cloud.gMax, 1e-6)) * cloud.growSeconds;

// How much of block (x, y, z) is smoke (0..1, its share of 27 cubes) - for
// gameplay: what a line of sight crosses.
export function smokeAt(cloud, x, y, z) {
  return (cloud.blocks.get(`${x},${y},${z}`) || 0) / SMOKE_CUBES ** 3;
}
