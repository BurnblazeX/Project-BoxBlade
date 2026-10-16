import { voxelIndex, gridIndex, ringSegments, GRID_DIM, DISTANCE_RANGE, FIELD_CHANNELS,
         CASCADE_COUNT, cascadeVoxelMetres } from './boxgrid.js';

// --- Tracing against the cutout, not against a box ---
//
// A box proxy gives a character a rectangular shadow. The silhouette gives it a
// shadow with legs, and the difference between those two is most of what makes a
// shadow read as a character rather than as a blob under one.
//
// THE REASON THIS IS CHEAP IS THE FROZEN CONSTANT. Twelve texels per 1.5 m block
// means 12.5 cm per texel, the boxGrid is locked 1:1 to that, and the character
// sprites are authored at 12 x 24 for a 1.5 x 3.0 m quad. So one sprite pixel is
// one voxel is one texel, exactly, with nothing to resample and no projection
// error to carry. The alpha mask IS a voxel mask already - it just has to be
// turned into distances.
//
// Every sprite in the set holds to it: characters 12 x 24 over 1.5 x 3.0 m,
// the chest 12 x 12 over 1.5 x 1.5, the tree 24 x 36 over 3.0 x 4.5. All 8 px/m.
// downsampleMask below is kept for assets that drift off that anyway - the tree
// was 48 x 72 until it was brought back to spec, so the drift is not theoretical
// and S8's warning about it is worth honouring rather than assuming.
//
// --- The shape, as a distance function ---
//
// Everything the bake needs is "how far is this point from the surface", and for
// a flat cutout that separates cleanly into two independent parts:
//
//   in the plane of the card   a 2D distance transform of the alpha mask
//   across it                  a slab half a voxel thick either side
//
// and those combine with the standard extrusion, which is the same form
// boxDistance already uses one dimension down. Exact, not an approximation.
//
// The 2D transform is computed ONCE per sprite at load and cached. A 12 x 24
// bitmap is nothing, and it never changes unless the art does - so the per-voxel
// cost at bake time is one array read and about six arithmetic ops, against the
// box's own handful. The band is identical, so the measured ~0.25 ms of the bake
// does not move; see occluders.js for where the real per-frame cost actually is.
//
// --- Facing: the card turns to meet the sun ---
//
// A card has an orientation, and a FIXED one degenerates - when the light runs
// parallel to the plane the shadow thins to a line and then vanishes. So the
// card is turned to face the sun's azimuth instead: its normal lies along the
// sun's horizontal direction, so the full silhouette always faces the light and
// the degenerate case cannot arise.
//
// THIS STILL KEEPS S9'S GUARANTEE, which is the part worth being precise about.
// S9 rules out a CAMERA-facing proxy, and for a specific reason: the thickness
// being voxelised would be camera-relative, so a character's shadow would change
// shape as the player orbits - which reads as a bug rather than as a rotation.
// A sun-facing card has no such problem. It is world-locked with respect to the
// viewer; it re-orients only when the LIGHT moves, which is both rare and
// exactly what a real object's silhouette-to-the-sun does. Moving the sun
// already invalidates every shadow in the scene, so it costs nothing extra.
//
// What it gives up: the card is no longer axis-aligned to the voxel grid, so the
// 1:1 pixel-to-voxel correspondence only holds when the sun happens to lie along
// an axis. That is why the transform is sampled BILINEARLY below - the same
// reasoning that makes the boxGrid's own 3D texture LinearFilter rather than
// nearest. Interpolating a distance field is exact in a way that interpolating
// occupancy never was, so a rotated card still resolves its zero-crossing
// properly instead of stair-stepping against its own pixel grid.

// A facing is the card's WIDTH direction in the XZ plane, as a unit vector; its
// normal is that turned 90 degrees. These two are the axis-aligned cases, kept
// named because they are what a test pins against and what the console falls
// back to when the sun is directly overhead and has no azimuth to speak of.
export const AXIS_X = { ux: 1, uz: 0 };   // card lies in XY, normal along Z
export const AXIS_Z = { ux: 0, uz: 1 };   // card lies in ZY, normal along X

// The card turned to face a light. Given the light's horizontal direction the
// card's NORMAL should lie along it, so the width runs at right angles.
//
// THE SIGN IS THE WHOLE FUNCTION, and getting it wrong does not look like a
// wrong angle - it looks like the character's shadow is right-side-left.
//
// Two ways to see which sign is right, and they have to agree:
//
//   the normal. minSilhouetteVoxels builds it as (-uz, ux), so u = (dz, -dx)
//   gives a normal of (dx, dz) - the light's own direction. The card faces the
//   light. The other sign gives (-dx, -dz), a card facing directly away from it,
//   which presents its BACK to the sun and so casts a mirrored silhouette.
//
//   the handedness. A viewer standing at the light looks along -d, so their
//   right is cross(-d, up) = (dz, -dx). The silhouette's +u is increasing column
//   - its right as drawn - and that has to land on the viewer's right or the
//   image is flipped.
//
// Both give u = (dz, -dx). Pinned by 'the cutout is not mirrored' in the tests,
// with an asymmetric mask, because a left-right symmetric one passes either way.
//
// Degenerate directly overhead, where there is no azimuth - the fallback is
// arbitrary because at that angle the shadow is directly underneath and its
// orientation cannot be seen anyway.
export function facingToward(dx, dz) {
  const len = Math.hypot(dx, dz);
  if (len < 1e-6) return AXIS_X;
  return { ux: dz / len, uz: -dx / len };
}

// Alpha above this counts as solid. Matched to the alphaTest the sprite
// materials already use, so the shadow's outline is the same outline the
// renderer draws - a proxy that disagreed would cast from pixels that are not
// visible, or fail to cast from ones that are.
export const ALPHA_THRESHOLD = 0.5;

// RGBA bytes from a canvas to a 0/1 mask. Split out from anything DOM-shaped so
// the transform below can be tested headless against a synthetic mask.
export function maskFromRGBA(rgba, w, h, threshold = ALPHA_THRESHOLD) {
  const cut = threshold * 255;
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) mask[i] = rgba[i * 4 + 3] > cut ? 1 : 0;
  return mask;
}

// Reduce a mask that was authored denser than the voxel grid. ANY set pixel in
// the group wins: a silhouette that shrank under downsampling would drop the
// thin parts - a tree's trunk, a character's arm - which are exactly the parts
// worth having a silhouette for.
export function downsampleMask(mask, w, h, factor) {
  if (factor <= 1) return { mask, w, h };
  const dw = Math.ceil(w / factor), dh = Math.ceil(h / factor);
  const out = new Uint8Array(dw * dh);
  for (let y = 0; y < h; y++) {
    const dy = (y / factor) | 0;
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x]) out[dy * dw + ((x / factor) | 0)] = 1;
    }
  }
  return { mask: out, w: dw, h: dh };
}

// --- The exact Euclidean distance transform (Felzenszwalb & Huttenlocher) ---
//
// Squared distances, one dimension at a time: transform every row, then every
// column of the result, and the two compose into the exact 2D answer. Linear in
// the number of pixels, and exact rather than the chamfer approximation - which
// matters because an approximate distance is an approximate step length, and a
// sphere trace that steps too far goes through the surface it was looking for.
//
// The lower envelope of parabolas: each pixel q contributes a parabola rooted at
// its own value, the transform is the envelope of all of them, and v/z track the
// envelope's vertices and the intervals they win over as the scan advances.
function edt1d(f, n, d, v, z) {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    let s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    while (s <= z[k]) {
      k--;
      s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while (z[k + 1] < q) k++;
    const dq = q - v[k];
    d[q] = dq * dq + f[v[k]];
  }
}

const INF = 1e20;

// Squared distance from every pixel to the nearest SET pixel of `seed`.
function edt2d(seed, w, h) {
  const f = new Float64Array(Math.max(w, h));
  const d = new Float64Array(Math.max(w, h));
  const v = new Int32Array(Math.max(w, h));
  const z = new Float64Array(Math.max(w, h) + 1);
  const out = new Float64Array(w * h);

  for (let i = 0; i < w * h; i++) out[i] = seed[i] ? 0 : INF;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) f[x] = out[y * w + x];
    edt1d(f, w, d, v, z);
    for (let x = 0; x < w; x++) out[y * w + x] = d[x];
  }
  for (let x = 0; x < w; x++) {
    for (let y = 0; y < h; y++) f[y] = out[y * w + x];
    edt1d(f, h, d, v, z);
    for (let y = 0; y < h; y++) out[y * w + x] = d[y];
  }
  return out;
}

// Signed distance per pixel, in pixel units, negative inside the silhouette.
//
// THE HALF-PIXEL IS NOT A FUDGE. Both transforms measure centre to centre, so a
// pixel just outside the shape reads 1 and one just inside reads -1, which puts
// the surface a whole pixel from each of them when it is really on the face
// between the two - half a pixel from each. Left uncorrected every distance near
// the edge is double what it should be, and a sphere trace steps on those
// numbers: the first step off a surface would clear it by twice the available
// room and land inside the geometry. Overestimating distance is the one error
// this field must never make.
export function signedDistanceTransform(mask, w, h) {
  const inv = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) inv[i] = mask[i] ? 0 : 1;

  const toSet = edt2d(mask, w, h);   // 0 on set pixels, >0 elsewhere
  const toClear = edt2d(inv, w, h);  // 0 on clear pixels, >0 inside

  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    out[i] = mask[i] ? -(Math.sqrt(toClear[i]) - 0.5)
                     : Math.sqrt(toSet[i]) - 0.5;
  }
  return out;
}

// --- The silhouette, ready to bake ---
//
// PADDED BY THE BAND WIDTH, deliberately. The bake visits every voxel within
// `range` of the card, which reaches past the sprite's own rectangle on all
// sides. Without the pad those samples fall off the edge of the transform and
// need a clamp plus a correction term - and every cheap version of that
// correction either overestimates the distance diagonally, which is the one
// error a sphere trace cannot take, or underestimates it enough to fatten the
// silhouette. Padding the bitmap first makes the whole band exact for the cost
// of a 28 x 40 array instead of a 12 x 24 one.
export function createSilhouette({ mask, w, h, widthMetres, heightMetres,
                                   voxelMetres, axis = 'x',
                                   pad = Math.ceil(DISTANCE_RANGE) + 1 }) {
  // Reduce to voxel resolution first, so the transform is computed in the units
  // the bake samples it in and nearest sampling is exact - consistent with the
  // S6.4a rule that everything on a surface is sampled nearest.
  const pxPerVoxel = Math.max(1, Math.round((w / widthMetres) * voxelMetres));
  const red = downsampleMask(mask, w, h, pxPerVoxel);

  const pw = red.w + pad * 2, ph = red.h + pad * 2;
  const padded = new Uint8Array(pw * ph);
  for (let y = 0; y < red.h; y++) {
    for (let x = 0; x < red.w; x++) {
      if (red.mask[y * red.w + x]) padded[(y + pad) * pw + (x + pad)] = 1;
    }
  }

  return {
    dt: signedDistanceTransform(padded, pw, ph),
    w: pw, h: ph, pad,
    cols: red.w, rows: red.h,
    axis,
    // Half a voxel either side. A card with no thickness at all is a valid
    // surface mathematically but a ray running along its plane never registers a
    // hit, so a light grazing the ground would shine straight through a
    // character standing in it.
    halfThickness: 0.5
  };
}

// One silhouette per cascade, each reduced to ITS OWN voxel size.
//
// NOT AN OPTIMISATION - a correctness requirement, and the failure it prevents
// is the kind that looks like a rendering artifact rather than a bug. The bake
// reads sil.cols as a count of GRID voxels, so a silhouette built at C0's 12.5 cm
// and handed to C1 describes a card 12 voxels wide at 25 cm each: three metres
// instead of one and a half. Measured before the fix, a 1.12 m character came out
// 2.24 m across in C1 and 4.49 m in C2.
//
// A shadow ray leaves C0 after 18 m and continues in C1, so what that produced
// was a sprite's shadow DOUBLING in width at the cascade seam - a hard, obvious
// line across the ground that reads as a broken cascade blend. There is nothing
// to blend; the two levels simply disagreed about how big the character was.
//
// Reducing per level also gives the right thing for free: C1 carries a 6 x 12
// silhouette and C2 a 3 x 6, so distant casters cost less and quantise coarser,
// which is the lockstep between cascade and detail level that S7.3 asks for.
export function createSilhouetteSet(opts, levels = CASCADE_COUNT) {
  const set = [];
  for (let l = 0; l < levels; l++) {
    set.push(createSilhouette({ ...opts, voxelMetres: cascadeVoxelMetres(l) }));
  }
  return set;
}

// Signed distance in VOXEL units, given a point in card space.
//
// u runs along the card's width and v up its height, both measured from the
// card's centre; n is across it. The mask's rows are top-down as an image is,
// and v is up as the world is, so v inverts - a silhouette baked upside down
// casts a shadow that looks almost right, which is the worst kind of wrong.
//
// BILINEAR, for the reason given in the header: once the card turns to face the
// sun it is no longer aligned to the voxel grid, and nearest sampling would let
// the silhouette stair-step against its own pixels. At a pixel centre the
// interpolation weight is zero, so the axis-aligned case still reads exactly the
// pixel it used to - this is strictly a superset of the nearest behaviour, not a
// softening of it.
export function silhouetteDistance(sil, u, v, n, flip = false) {
  const cu = (flip ? -u : u) + sil.cols / 2 + sil.pad;
  const cv = sil.rows / 2 - v + sil.pad;

  // Sample centres sit at index + 0.5, so shift into centre-relative space -
  // the same convention sampleDistance() uses against the 3D field.
  const fu = cu - 0.5, fv = cv - 0.5;
  const x0 = Math.floor(fu), y0 = Math.floor(fv);
  const tx = fu - x0, ty = fv - y0;
  const cl = (i, n2) => (i < 0 ? 0 : i >= n2 ? n2 - 1 : i);
  const xa = cl(x0, sil.w), xb = cl(x0 + 1, sil.w);
  const ya = cl(y0, sil.h), yb = cl(y0 + 1, sil.h);
  const ra = ya * sil.w, rb = yb * sil.w;
  const top = sil.dt[ra + xa] + (sil.dt[ra + xb] - sil.dt[ra + xa]) * tx;
  const bot = sil.dt[rb + xa] + (sil.dt[rb + xb] - sil.dt[rb + xa]) * tx;
  const d2 = top + (bot - top) * ty;
  const an = (n < 0 ? -n : n) - sil.halfThickness;

  // Extrude a 2D distance to a slab. Outside on both axes, the exact distance is
  // the diagonal; inside on both, it is the larger (least negative) of the two.
  const ou = d2 > 0 ? d2 : 0;
  const on = an > 0 ? an : 0;
  const outside = Math.sqrt(ou * ou + on * on);
  const inside = Math.min(Math.max(d2, an), 0);
  return outside + inside;
}

// --- The bake ---
//
// The silhouette counterpart of minBoxVoxels, and deliberately the same shape:
// same rect clipping, same min-is-union, same inlined encode. A silhouette that
// wrote its bytes even slightly differently from the terrain's would put a
// character's shadow at a different depth from the wall beside it.
export function minSilhouetteVoxels(grid, sil, cx, cy, cz, rects, flip = false,
                                    facing = AXIS_X,
                                    range = grid.range || DISTANCE_RANGE) {
  const data = grid.data;
  const invRange = 1 / range;
  const hu = sil.cols / 2, hv = sil.rows / 2, hn = sil.halfThickness;

  // The card's width direction in the XZ plane, and its normal at right angles
  // to it. One rotation rather than two hard-coded axis cases: an arbitrary
  // facing is what following the sun needs, and the axis-aligned orientations
  // fall out of it as the special cases where one component is zero.
  const ux = facing.ux, uz = facing.uz;
  const nx = -uz, nz = ux;

  // The rotated card's axis-aligned bounds. A point on it is a*(ux,uz) +
  // b*(nx,nz) with |a| <= hu and |b| <= hn, so each world extent is the sum of
  // the two contributions taken at their largest.
  const ex = hu * Math.abs(ux) + hn * Math.abs(nx);
  const ez = hu * Math.abs(uz) + hn * Math.abs(nz);

  const lo = (c, e) => Math.ceil(c - e - range);
  const hi = (c, e) => Math.floor(c + e + range);
  const bx0 = lo(cx, ex), bx1 = hi(cx, ex);
  const by0 = lo(cy, hv), by1 = hi(cy, hv);
  const bz0 = lo(cz, ez), bz1 = hi(cz, ez);

  for (const r of rects) {
    const x0 = Math.max(r.x0, bx0), x1 = Math.min(r.x1 - 1, bx1);
    const y0 = Math.max(r.y0, by0), y1 = Math.min(r.y1 - 1, by1);
    const z0 = Math.max(r.z0, bz0), z1 = Math.min(r.z1 - 1, bz1);
    if (x0 > x1 || y0 > y1 || z0 > z1) continue;
    const segs = ringSegments(grid, x0, x1);

    for (let vz = z0; vz <= z1; vz++) {
      const dz = vz + 0.5 - cz;
      for (let vy = y0; vy <= y1; vy++) {
        const v = vy + 0.5 - cy;
        // Physically contiguous runs: a row can wrap in the ring.
        for (const [sx0, sx1] of segs) {
          // The opaque channel (R): a proxy is an opaque caster.
          let idx = gridIndex(grid, sx0, vy, vz) * FIELD_CHANNELS;
          for (let vx = sx0; vx <= sx1; vx++, idx += FIELD_CHANNELS) {
            const dx = vx + 0.5 - cx;
            const d = silhouetteDistance(sil, dx * ux + dz * uz, v,
                                         dx * nx + dz * nz, flip);
            if (d >= range) continue;   // encodes to FAR_BYTE, min is a no-op
            const t = (d < 0 ? -d : d) * invRange;
            const c = t >= 1 ? (d < 0 ? -1 : 1)
                             : (d < 0 ? -Math.sqrt(t) : Math.sqrt(t));
            const b = ((c + 1) * 127.5 + 0.5) | 0;
            if (b < data[idx]) data[idx] = b;
          }
        }
      }
    }
  }
}
