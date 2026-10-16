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
// CPU and pure, so it is tested headlessly. gpu/mirrorlight.js createMirrorLightTSL reads
// the packed form.
import { BLOCK_METRES, GROUND_Y, blockBoxes, blockParts, isCompound, isDual } from './world.js';
import { subtractRect } from './rects.js';
import { BLOCK_TEXELS } from './materials.js';
import { lightContribution, shadowScore } from './lights.js';

export const MAX_MIRRORS = 32;

// A material reflects the sun sharply enough to count when any of its texels
// is smoother than this alpha - the same band edge the view reflection ray
// uses (gpu/specular.js REFLECT_ROUGH_END). Grass, at alpha 1, never does.
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
// materialIds. isGlass: materialId -> bool, for glass's back-face glare.
// Returns rectangles, in metres:
//   { axis, sign, plane, min: [u, v], max: [u, v], thickness }
// where plane is the face's coordinate along axis and u, v run along
// PLANE_AXES[axis]. thickness: for a glass face, the metres of glass behind it
// to a back face with air beyond - the slab the second (back-face) reflection
// crosses twice; 0 for anything else, or for glass whose back rests on
// something (a back face against the ground is no mirror). Faces are grouped
// by plane AND thickness, so glass never merges with a coplanar marble face,
// nor two slabs of different depth.
//
// backFaces (default true): glass faces are grouped by the slab behind them,
// which only the back-face glare (bxb.glassglare(2)) reads. Without it the
// thickness means nothing, so coplanar faces merge whatever they are made of -
// the same light, in fewer rectangles: a ray crosses a plane at one point,
// which lies in one rectangle of a group or another, never two.
export function buildMirrors(world, reflective, isGlass = () => false, { backFaces = true } = {}) {
  const groups = new Map();
  for (const [key, whole] of world) {
    const dual = isDual(whole);
    for (const part of blockParts(whole)) {
      // Each half of a dual block reflects as its own material, and the face
      // between the halves is covered.
      const block = { ...whole, materialId: part.materialId };
      if (!reflective.has(block.materialId)) continue;
      // A slope's staircase would be dozens of one-texel rectangles, and stairs
      // are faces at two depths: they reflect through the view ray, but throw
      // no mirror light.
      if (isCompound(block)) continue;
      const c = key.split(',').map(Number);
      const glass = isGlass(block.materialId);
      // The block's one box (cell fractions): a half's, a vertical half's or a
      // pane's faces can sit mid-cell on any axis.
      const box = part.boxes[0];
      for (let axis = 0; axis < 3; axis++) {
        const [ua, va] = PLANE_AXES[axis];
        for (const sign of [1, -1]) {
          const at = box[2 * axis + (sign > 0 ? 1 : 0)];
          const onEdge = sign > 0 ? at === 1 : at === 0;
          // The face, as [u0, u1, v0, v1] of the cell, less what the neighbour
          // across it covers - only a face on the cell's boundary can be
          // covered. Beside a slope or stairs only a top face under it is (each
          // fills its cell's floor); anything else is left whole - conservative,
          // and the legs are traced through the field anyway.
          const face = [box[2 * ua], box[2 * ua + 1], box[2 * va], box[2 * va + 1]];
          let parts = [face];
          const nb = c.slice(); nb[axis] += sign;
          const nbBlock = onEdge ? world.get(nb.join(',')) : null;
          if (!onEdge && dual) parts = [];
          else if (nbBlock && isDual(nbBlock)) parts = [];
          else if (nbBlock && isCompound(nbBlock)) {
            if (axis === 1 && sign > 0) parts = [];
          } else if (nbBlock) {
            const nbox = blockBoxes(nbBlock)[0];
            if (sign > 0 ? nbox[2 * axis] === 0 : nbox[2 * axis + 1] === 1) {
              parts = subtractRect(face, [nbox[2 * ua], nbox[2 * ua + 1], nbox[2 * va], nbox[2 * va + 1]]);
            }
          }
          if (!parts.length) continue;                    // not exposed
          // The underside of the ground layer faces the void under the world -
          // nothing is ever there to receive from it.
          if (axis === 1 && sign === -1 && c[1] <= GROUND_Y && at === 0) continue;
          let thickness = 0;
          if (glass && backFaces && (box[2 * axis] > 0 || box[2 * axis + 1] < 1)) {
            // A glass half or pane: its own depth, with air (or the cell's
            // other half) behind - conservative, as a slab is.
            thickness = (box[2 * axis + 1] - box[2 * axis]) * BLOCK_METRES;
          } else if (glass && backFaces) {
            // Walk back through the glass behind this face.
            const q = c.slice();
            let k = 0;
            for (; k < 16; k++) {
              const b = world.get(q.join(','));
              if (!b || !isGlass(b.materialId)) break;
              q[axis] -= sign;
            }
            thickness = world.has(q.join(',')) ? 0 : k * BLOCK_METRES;
          }
          // The face's plane, in blocks. A part short of the cell along u (v)
          // is grouped by its column (row) and its extent, so it merges only
          // along the other axis; short along both, it stands alone.
          const level = c[axis] - 0.5 + at;
          for (const piece of parts) {
            const fullU = piece[0] === 0 && piece[1] === 1, fullV = piece[2] === 0 && piece[3] === 1;
            const g = `${axis},${sign},${level}` +
                      (fullU && fullV ? '' : fullU ? `,v${c[va]},${piece}` : fullV ? `,u${c[ua]},${piece}` : `,c${c[ua]},${c[va]},${piece}`) +
                      (backFaces ? `,${glass ? 'g' + thickness : 'o'}` : '');
            if (!groups.has(g)) groups.set(g, { axis, sign, level, thickness, part: fullU && fullV ? null : piece, cells: new Set() });
            groups.get(g).cells.add(`${c[ua]},${c[va]}`);
          }
        }
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
      // (i - 0.5 .. i + 0.5) * BLOCK_METRES; a part of a side face spans only
      // its share of the row's height.
      const min = [(u0 - 0.5) * BLOCK_METRES, (v0 - 0.5) * BLOCK_METRES];
      const max = [(u1 + 0.5) * BLOCK_METRES, (v1 + 0.5) * BLOCK_METRES];
      // A part's group holds one column (row) along its short axis.
      if (g.part && !(g.part[0] === 0 && g.part[1] === 1)) {
        min[0] = (u0 - 0.5 + g.part[0]) * BLOCK_METRES;
        max[0] = (u0 - 0.5 + g.part[1]) * BLOCK_METRES;
      }
      if (g.part && !(g.part[2] === 0 && g.part[3] === 1)) {
        min[1] = (v0 - 0.5 + g.part[2]) * BLOCK_METRES;
        max[1] = (v0 - 0.5 + g.part[3]) * BLOCK_METRES;
      }
      rects.push({
        axis: g.axis, sign: g.sign,
        plane: g.level * BLOCK_METRES,
        min, max,
        thickness: g.thickness
      });
    }
  }
  return rects;
}

// How far reflected light is gathered, metres - the receiver-to-mirror leg's
// cap. gpu/mirrorlight.js uses it for the same test.
export const MIRROR_REACH = 24;
// The light list's own rows carry a light's level; this many metres per level
// is its radius (lights.js: the level IS the radius in blocks).
const SUN_BIT = 1 << 16;

const nearestOnRect = (r, P) => {
  const [ua, va] = PLANE_AXES[r.axis];
  const du = P[ua] - Math.min(Math.max(P[ua], r.min[0]), r.max[0]);
  const dv = P[va] - Math.min(Math.max(P[va], r.min[1]), r.max[1]);
  return Math.hypot(P[r.axis] - r.plane, du, dv);
};
export function pickMirrored(lights, rects, focus, budget) {
  const n = Math.max(0, Math.floor(budget));
  if (n >= lights.length) return new Set(lights);
  if (n === 0) return new Set();
  const F = [focus.x, focus.y, focus.z];
  const near = rects.map(r => Math.max(0, 1 - nearestOnRect(r, F) / MIRROR_REACH));
  const score = l => {
    const P = [l.position.x, l.position.y, l.position.z];
    let best = 0;
    rects.forEach((r, i) => {
      if (near[i] <= 0 || (P[r.axis] - r.plane) * r.sign <= 0) return;   // too far, or behind it
      best = Math.max(best, lightContribution(nearestOnRect(r, P), l.level) * near[i]);
    });
    return best + 1e-6 * shadowScore(l, focus);
  };
  const ranked = lights.map(l => ({ l, s: score(l) })).sort((a, b) => b.s - a.s);
  return new Set(ranked.slice(0, n).map(r => r.l));
}

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
//   2  sun box min xyz, glass thickness 3  sun box max xyz, cache base
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

// --- The mirror texel cache (gpu/mirrorlight.js createMirrorLightTSL) ---
//
// What a mirror texel sends does not depend on who receives it - its own sun
// shadow, its view of each light, its F0, smoothness and bumped normal - yet
// every receiving texel recomputed them, two marches and three texture reads a
// hit. A compute pass works them out once per mirror texel into a buffer, and
// the receivers read their mirror texel's slot. Same function, same snapped
// point: the same values.
//
// Each packed rectangle's texels are numbered row by row from its cache base
// (its first texel's index, row 3's w); a texel's slot is index * stride:
// two vec4 for the sun, three per light in the list.
export const MIRROR_TEXEL = BLOCK_METRES / BLOCK_TEXELS;
export const MIRROR_CACHE_VEC4S = 1 << 20;
export const mirrorCacheStride = lightCount => 2 + 3 * lightCount;
export function mirrorTexelDims(r) {
  return [Math.round((r.max[0] - r.min[0]) / MIRROR_TEXEL),
          Math.round((r.max[1] - r.min[1]) / MIRROR_TEXEL)];
}
// area: optional { minX, maxX, minZ, maxZ } - only rectangles overlapping it
// are packed. main.js passes C1's footprint: past C1 the field is too coarse
// for a mirror's two marches to mean much, and a mirror the player is nowhere
// near should cost the frame nothing.
// cache: optional { stride, capacity } - each rectangle gets its texel cache
// base, and packing stops before the cache would overflow (nearest first, as
// the rectangle cap). cache.texels is set to the texels packed: the compute
// pass's count.
export function packMirrors(rects, near, out, capacity = MAX_MIRRORS,
                            sun = { dir: [0, 1, 0], on: false }, lights = [], area = null,
                            cache = null) {
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
  let texels = 0;
  let n = 0;
  for (const { r, reach } of live) {
    const [nu, nv] = mirrorTexelDims(r);
    if (cache && (texels + nu * nv) * cache.stride > cache.capacity) break;
    const bmin = reach.box ? reach.box[0] : [0, 0, 0], bmax = reach.box ? reach.box[1] : [0, 0, 0];
    out.set([r.axis, r.sign, r.plane, reach.mask, r.min[0], r.min[1], r.max[0], r.max[1],
             bmin[0], bmin[1], bmin[2], r.thickness || 0, bmax[0], bmax[1], bmax[2], texels], n * 16);
    texels += nu * nv;
    n++;
  }
  if (cache) cache.texels = texels;
  return n;
}
export const MIRROR_SUN_BIT = SUN_BIT;
