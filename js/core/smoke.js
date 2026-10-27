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
// Headless: gameplay reads the blocks (line of sight), the renderer draws the
// cubes (render/features/smoke.js).
import { BLOCK_METRES, isSolidAt, partAt } from './world.js';
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
                                 glassAt = glassAtWorld) {
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
  const solidCache = new Int8Array(B * B * B).fill(-1);
  const solid = (i, j, k) => {
    if (!inBox(i, j, k)) return true;
    const n = at(i, j, k);
    if (solidCache[n] < 0) { const p = pos(i, j, k); solidCache[n] = solidAt(p[0], p[1], p[2]) ? 1 : 0; }
    return solidCache[n] === 1;
  };
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
    while (frontier.size && order.length < volume) {
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
  const cubes = [], blocks = new Map();
  let gMax = 0;
  for (const [i, j, k, travel] of order) {
    const p = pos(i, j, k);
    if (travel > gMax) gMax = travel;
    // Its faces: open to air that is not smoke, or to glass (always seen), or against
    // solid (seen only once that solid is lifted away by the interiors). And
    // when it is enclosed: the travel at which its last smoke neighbour rolls
    // in - while the cloud grows, a cube is seen until then (the front rolls
    // out layer by layer); one open to air never is.
    let air = 0, against = 0, enclosed = travel;
    FACES.forEach(([di, dj, dk], f) => {
      const a = i + di, b = j + dj, c = k + dk;
      if (inBox(a, b, c) && taken[at(a, b, c)]) { enclosed = Math.max(enclosed, g[at(a, b, c)]); return; }
      if (!solid(a, b, c)) { air = 1; return; }
      const q = pos(a, b, c);
      if (glassAt(q[0], q[1], q[2])) air = 1; else against |= 1 << f;
    });
    cubes.push([p[0], p[1], p[2], travel, over[0] + i, over[1] + j, over[2] + k, air, air ? 0 : against,
                air ? Infinity : enclosed]);
    const key = p.map(v => Math.round(v / BLOCK_METRES)).join();
    blocks.set(key, (blocks.get(key) || 0) + 1);
  }
  return { tile: { gx, gy, gz }, spell, level, radius: rh, centre, box, volume, cubes, blocks, gMax,
           growSeconds: SMOKE_GROW_SECONDS * rh / SMOKE_RADIUS };
}

// The second a cube that travelled `travel` appears, after the cast.
export const smokeAppear = (cloud, travel) => (travel / Math.max(cloud.gMax, 1e-6)) * cloud.growSeconds;

// How much of block (x, y, z) is smoke (0..1, its share of 27 cubes) - for
// gameplay: what a line of sight crosses.
export function smokeAt(cloud, x, y, z) {
  return (cloud.blocks.get(`${x},${y},${z}`) || 0) / SMOKE_CUBES ** 3;
}
