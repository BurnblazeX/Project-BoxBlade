// --- Mirrors: the reflective surfaces, as a few flat rectangles ---
//
// Reflected light (the sun off a polished floor, landing on the wall beside
// it) is GATHERED, not traced forward. A flat mirror with normal N sends the
// sun one way only, so a receiving texel looks along the sun direction
// mirrored in that plane, D = reflect(L, N), and asks whether that ray lands on
// the mirror. That needs the mirrors as analytic planes: this builds them.
//
// Every exposed face of a reflective block, grouped by plane and merged
// greedily into rectangles - the test floor's 5 x 5 top is ONE rectangle, not
// 25 faces. The shader then does one plane test per rectangle per texel, and
// marches only where a ray actually hits one.
//
// CPU and pure, so it is tested headlessly. gpu.js createMirrorLightTSL reads
// the packed form.
import { BLOCK_METRES, GROUND_Y } from './world.js';

export const MAX_MIRRORS = 32;

// A material reflects the sun sharply enough to count when any of its texels
// is smoother than this alpha - the same band edge the view reflection ray
// uses (gpu.js REFLECT_ROUGH_END). Grass, at alpha 1, never does.
export const MIRROR_ALPHA = 0.45;
export function isReflective(specRgba) {
  if (!specRgba) return false;
  for (let i = 0; i < specRgba.length; i += 4) {
    const rough = 1 - specRgba[i] / 255;
    if (rough * rough < MIRROR_ALPHA) return true;
  }
  return false;
}

// The two in-plane axes for a face along `axis` (0 x, 1 y, 2 z), in order.
const PLANE_AXES = [[1, 2], [0, 2], [0, 1]];

// world: a Map of 'x,y,z' -> block (world.js World). reflective: a Set of
// materialIds. Returns rectangles, in metres:
//   { axis, sign, plane, min: [u, v], max: [u, v] }
// where plane is the face's coordinate along axis and u, v run along
// PLANE_AXES[axis].
export function buildMirrors(world, reflective) {
  const groups = new Map();
  for (const [key, block] of world) {
    if (!reflective.has(block.materialId)) continue;
    const c = key.split(',').map(Number);
    for (let axis = 0; axis < 3; axis++) {
      for (const sign of [1, -1]) {
        const nb = c.slice(); nb[axis] += sign;
        if (world.has(nb.join(','))) continue;          // not exposed
        // The underside of the ground layer faces the void under the world -
        // nothing is ever there to receive from it.
        if (axis === 1 && sign === -1 && c[1] <= GROUND_Y) continue;
        const [ua, va] = PLANE_AXES[axis];
        const g = `${axis},${sign},${c[axis]}`;
        if (!groups.has(g)) groups.set(g, { axis, sign, level: c[axis], cells: new Set() });
        groups.get(g).cells.add(`${c[ua]},${c[va]}`);
      }
    }
  }

  const rects = [];
  for (const g of groups.values()) {
    // Greedy: take the lowest remaining cell, grow along u as far as it
    // goes, then along v while the whole row is present.
    const left = new Set(g.cells);
    const sorted = [...left].map(s => s.split(',').map(Number))
      .sort((a, b) => a[1] - b[1] || a[0] - b[0]);
    for (const [u0, v0] of sorted) {
      if (!left.has(`${u0},${v0}`)) continue;
      let u1 = u0;
      while (left.has(`${u1 + 1},${v0}`)) u1++;
      let v1 = v0;
      const rowFull = v => { for (let u = u0; u <= u1; u++) if (!left.has(`${u},${v}`)) return false; return true; };
      while (rowFull(v1 + 1)) v1++;
      for (let v = v0; v <= v1; v++) for (let u = u0; u <= u1; u++) left.delete(`${u},${v}`);
      // Blocks are centred on their grid coordinate, so a block spans
      // (i - 0.5 .. i + 0.5) * BLOCK_METRES.
      rects.push({
        axis: g.axis, sign: g.sign,
        plane: (g.level + 0.5 * g.sign) * BLOCK_METRES,
        min: [(u0 - 0.5) * BLOCK_METRES, (v0 - 0.5) * BLOCK_METRES],
        max: [(u1 + 0.5) * BLOCK_METRES, (v1 + 0.5) * BLOCK_METRES]
      });
    }
  }
  return rects;
}

// How far reflected light is gathered, metres - the receiver-to-mirror leg's
// cap. gpu.js uses it for the same test.
export const MIRROR_REACH = 24;
// The light list's own rows carry a light's level; this many metres per level
// is its radius (lights.js: the level IS the radius in blocks).
const SUN_BIT = 1 << 16;

// What can a mirror actually reflect this frame? Returned per rectangle:
//   mask  bit i: light i reaches it from its reflecting side; SUN_BIT: it
//         faces the sun. 0 means it reflects nothing and is not packed.
//   box   [min, max] around every receiver its reflected SUN can reach: the
//         rectangle swept along the reflected sun for MIRROR_REACH. A texel
//         outside it skips the sun leg for this mirror before any maths.
// sun: { dir: [x, y, z], on }. lights: [{ x, y, z, radius }], list order.
export function mirrorReach(r, sun, lights) {
  const [ua, va] = PLANE_AXES[r.axis];
  const corner = (u, v) => { const p = [0, 0, 0]; p[r.axis] = r.plane; p[ua] = u; p[va] = v; return p; };
  const corners = [corner(r.min[0], r.min[1]), corner(r.max[0], r.min[1]),
                   corner(r.min[0], r.max[1]), corner(r.max[0], r.max[1])];
  let mask = 0;
  for (let i = 0; i < lights.length && i < 16; i++) {
    const L = lights[i], P = [L.x, L.y, L.z];
    if ((P[r.axis] - r.plane) * r.sign <= 0) continue;          // behind it
    // Nearest point of the rectangle to the light.
    const q = [0, 0, 0];
    q[r.axis] = r.plane;
    q[ua] = Math.min(Math.max(P[ua], r.min[0]), r.max[0]);
    q[va] = Math.min(Math.max(P[va], r.min[1]), r.max[1]);
    if (Math.hypot(P[0] - q[0], P[1] - q[1], P[2] - q[2]) < L.radius) mask |= 1 << i;
  }
  let box = null;
  const L = sun.dir;
  if (sun.on && L[r.axis] * r.sign > 0) {
    mask |= SUN_BIT;
    // The reflected sun travels reflect(-L, N): -L with the axis flipped.
    const R = [-L[0], -L[1], -L[2]];
    R[r.axis] = -R[r.axis];
    const pts = corners.concat(corners.map(c => c.map((v, a) => v + R[a] * MIRROR_REACH)));
    const pad = 0.25;
    box = [[0, 1, 2].map(a => Math.min(...pts.map(p => p[a])) - pad),
           [0, 1, 2].map(a => Math.max(...pts.map(p => p[a])) + pad)];
  }
  return { mask, box };
}

// Nearest `capacity` to a point that reflect anything, packed four vec4 each:
//   0  axis, sign, plane, mask          1  umin, vmin, umax, vmax
//   2  sun box min xyz, 0               3  sun box max xyz, 0
// A rectangle's footprint on the ground, { minX, maxX, minZ, maxZ }, metres.
export function mirrorFootprint(r) {
  const [ua, va] = PLANE_AXES[r.axis];
  const lo = [0, 0, 0], hi = [0, 0, 0];
  lo[r.axis] = hi[r.axis] = r.plane;
  lo[ua] = r.min[0]; hi[ua] = r.max[0];
  lo[va] = r.min[1]; hi[va] = r.max[1];
  return { minX: lo[0], maxX: hi[0], minZ: lo[2], maxZ: hi[2] };
}

export const MIRROR_VEC4S = 4;
// area: optional { minX, maxX, minZ, maxZ } - only rectangles overlapping it
// are packed. main.js passes C1's footprint: past C1 the field is too coarse
// for a mirror's two marches to mean much, and a mirror the player is nowhere
// near should cost the frame nothing.
export function packMirrors(rects, near, out, capacity = MAX_MIRRORS,
                            sun = { dir: [0, 1, 0], on: false }, lights = [], area = null) {
  const centre = r => {
    const [ua, va] = PLANE_AXES[r.axis];
    const p = [0, 0, 0];
    p[r.axis] = r.plane;
    p[ua] = (r.min[0] + r.max[0]) / 2; p[va] = (r.min[1] + r.max[1]) / 2;
    return p;
  };
  const d2 = r => { const c = centre(r); return (c[0] - near.x) ** 2 + (c[1] - near.y) ** 2 + (c[2] - near.z) ** 2; };
  const inArea = r => {
    if (!area) return true;
    const f = mirrorFootprint(r);
    return f.maxX >= area.minX && f.minX <= area.maxX && f.maxZ >= area.minZ && f.minZ <= area.maxZ;
  };
  const live = rects.filter(inArea).map(r => ({ r, reach: mirrorReach(r, sun, lights) }))
    .filter(m => m.reach.mask !== 0)
    .sort((a, b) => d2(a.r) - d2(b.r)).slice(0, capacity);
  live.forEach(({ r, reach }, i) => {
    const bmin = reach.box ? reach.box[0] : [0, 0, 0], bmax = reach.box ? reach.box[1] : [0, 0, 0];
    out.set([r.axis, r.sign, r.plane, reach.mask, r.min[0], r.min[1], r.max[0], r.max[1],
             bmin[0], bmin[1], bmin[2], 0, bmax[0], bmax[1], bmax[2], 0], i * 16);
  });
  return live.length;
}
export const MIRROR_SUN_BIT = SUN_BIT;
