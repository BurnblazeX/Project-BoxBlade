import { file, section, ok, near, truthy, inRange, note } from './lib/harness.mjs';
import {
  PAGE, TEXELS_PER_PAGE, FACES_PER_BLOCK, NUDGE, FACE_NORMALS,
  faceUVToLocal, uvToTexel, faceBasis, atlasLayout,
  pageIndexFor, pageOriginTexels, texelWorldPosition, buildShadeJobs
} from '../js/atlas.js';
import { BLOCK_METRES } from '../js/world.js';
import { VOXEL_METRES } from '../js/boxgrid.js';

file('atlas.test.mjs - texel atlas addressing and face geometry');

const S = BLOCK_METRES;
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const len = a => Math.hypot(a.x, a.y, a.z);

section('page addressing');
ok('page is one boxGrid voxel per texel', PAGE, 12);
ok('and 144 texels per face', TEXELS_PER_PAGE, 144);
ok('pages are arithmetic: instance * 6 + face', pageIndexFor(7, 3), 45);
truthy('so no two faces of any block collide',
  new Set([0, 1, 2, 3, 4, 5].map(f => pageIndexFor(7, f))).size === 6);
truthy('and no two blocks collide either',
  pageIndexFor(7, 5) < pageIndexFor(8, 0));

const L = atlasLayout(1630);
truthy('the test world fits one atlas', L.fits);
truthy('with room for every page', L.pagesX * L.pagesY >= L.pageCount);
ok('atlas is a whole number of pages wide', L.width % PAGE, 0);
ok('and tall', L.height % PAGE, 0);
note(`1630 blocks -> ${L.width}x${L.height} texels, ${(L.bytes / 1048576).toFixed(1)} MB RGBA8`);

const o = pageOriginTexels(L.pagesX + 2, L.pagesX);
ok('page origin wraps onto the next row (x)', o.x, 2 * PAGE);
ok('page origin wraps onto the next row (y)', o.y, PAGE);

section('face basis is a proper orthonormal frame per face');
for (let f = 0; f < FACES_PER_BLOCK; f++) {
  const b = faceBasis(f);
  near(`face ${f}: u step is one texel`, len(b.uStep), VOXEL_METRES, 1e-9);
  near(`face ${f}: v step is one texel`, len(b.vStep), VOXEL_METRES, 1e-9);
  near(`face ${f}: u and v are perpendicular`, dot(b.uStep, b.vStep), 0, 1e-9);
  near(`face ${f}: u lies in the face plane`, dot(b.uStep, b.normal), 0, 1e-9);
  near(`face ${f}: v lies in the face plane`, dot(b.vStep, b.normal), 0, 1e-9);

  // The page's 144 texel centres must tile the face exactly, so the middle of
  // the page is the middle of the face. 11/2 because origin is texel 0's centre.
  const mid = {
    x: b.origin.x + (b.uStep.x + b.vStep.x) * 5.5,
    y: b.origin.y + (b.uStep.y + b.vStep.y) * 5.5,
    z: b.origin.z + (b.uStep.z + b.vStep.z) * 5.5
  };
  const n = FACE_NORMALS[f];
  const faceCentre = { x: n.x * (S / 2 + NUDGE), y: n.y * (S / 2 + NUDGE), z: n.z * (S / 2 + NUDGE) };
  near(`face ${f}: the page is centred on the face (x)`, mid.x, faceCentre.x, 1e-9);
  near(`face ${f}: the page is centred on the face (y)`, mid.y, faceCentre.y, 1e-9);
  near(`face ${f}: the page is centred on the face (z)`, mid.z, faceCentre.z, 1e-9);

  near(`face ${f}: samples sit half a voxel clear of the surface`,
       dot(b.origin, n), S / 2 + NUDGE, 1e-9);
}

section('the writer and the reader agree');
// This is the assertion the whole phase rests on. The compute pass converts a
// texel index into a world position (faceBasis); the fragment shader converts a
// uv into that same texel index (uvToTexel). If they disagree, the lighting is
// mirrored or transposed on some faces and not others - the single most likely
// way to get this wrong, and invisible on the flat ground where most of the
// world is. faceUVToLocal is the independent reference: BoxGeometry's own
// vertex layout, which is what actually produces the uv being sampled.
let worst = 0;
for (let f = 0; f < FACES_PER_BLOCK; f++) {
  const b = faceBasis(f);
  const n = FACE_NORMALS[f];
  for (let iu = 0; iu < PAGE; iu++) {
    for (let iv = 0; iv < PAGE; iv++) {
      // A uv somewhere inside texel (iu, iv), deliberately off-centre.
      const u = (iu + 0.3) / PAGE, v = (iv + 0.7) / PAGE;
      const t = uvToTexel(u, v);
      if (t.tx !== iu || t.ty !== iv) { worst = Infinity; break; }

      const surface = faceUVToLocal(f, u, v);
      const sample = texelWorldPosition({ x: 0, y: 0, z: 0 }, f, t.tx, t.ty);
      // Along the normal the sample is pushed out by NUDGE; in the plane it must
      // land inside the same texel it was sampled from, so within half a texel.
      const d = { x: sample.x - surface.x - n.x * NUDGE,
                  y: sample.y - surface.y - n.y * NUDGE,
                  z: sample.z - surface.z - n.z * NUDGE };
      near(`face ${f} texel ${iu},${iv}: no drift along the normal`, dot(d, n), 0, 1e-9);
      worst = Math.max(worst, len(d));
    }
  }
}
inRange('every texel resolves inside its own cell, on all six faces',
        worst, 0, VOXEL_METRES * 0.5 + 1e-9);
note(`worst in-plane disagreement ${(worst * 100).toFixed(2)} cm of a ${VOXEL_METRES * 100} cm texel`);

section('uv clamping');
const edge = uvToTexel(1, 1);
ok('uv 1.0 clamps to the last texel, not past the page (x)', edge.tx, PAGE - 1);
ok('uv 1.0 clamps to the last texel, not past the page (y)', edge.ty, PAGE - 1);
ok('uv 0 is the first texel', uvToTexel(0, 0).tx, 0);

section('shade jobs skip what cannot be seen');
// A 3x1x3 slab of ground with one block stacked on its middle column.
const solid = new Set(['0,0,0', '1,0,0', '2,0,0', '0,0,1', '1,0,1', '2,0,1',
                       '0,0,2', '1,0,2', '2,0,2', '1,1,1']);
const isSolid = (x, y, z) => solid.has(`${x},${y},${z}`);
const ids = new Map([...solid].map((k, i) => [k, i]));
const layout = atlasLayout(ids.size);
const jobs = buildShadeJobs({
  originBlockX: 0, originBlockZ: 0, size: 3, yMin: 0, yMax: 1,
  isSolid, instanceIdOf: (x, y, z) => ids.get(`${x},${y},${z}`),
  pagesX: layout.pagesX, pageCount: layout.pageCount
});

// 10 blocks x 6 faces = 60 if nothing were culled. The centre ground block has
// four solid side neighbours and the stacked block above it, leaving only its
// underside; every other block keeps its outward faces.
let expected = 0;
for (const k of solid) {
  const [x, y, z] = k.split(',').map(Number);
  for (const n of FACE_NORMALS) if (!isSolid(x + n.x, y + n.y, z + n.z)) expected++;
}
ok('one job per exposed face', jobs.count, expected);
truthy('which is fewer than six per block', jobs.count < solid.size * 6);
ok('each job carries an origin, a u step and a v step', jobs.origins.length, jobs.count * 4);
ok('and the texel total follows from the job count', jobs.texels, jobs.count * TEXELS_PER_PAGE);
ok('nothing was dropped for want of a page', jobs.skippedUnmapped, 0);
note(`${solid.size} blocks: ${jobs.count} faces shaded, ${solid.size * 6 - jobs.count} buried faces skipped`);

// Every page written must be distinct, or two faces would overwrite each other.
const pages = new Set();
for (let i = 0; i < jobs.count; i++) pages.add(jobs.origins[i * 4 + 3]);
ok('every job writes its own page', pages.size, jobs.count);

section('jobs are positioned in world space');
// The top face of block (1,1,1) is the highest surface in the slab, so its page
// must sit above the block centre by half a block plus the nudge.
let topY = -Infinity;
for (let i = 0; i < jobs.count; i++) topY = Math.max(topY, jobs.origins[i * 4 + 1]);
near('the highest job sits on the top face of the stacked block',
     topY, 1 * S + S / 2 + NUDGE, 1e-9);

section('an empty footprint is not an error');
const none = buildShadeJobs({
  originBlockX: 90, originBlockZ: 90, size: 3, yMin: 0, yMax: 1,
  isSolid, instanceIdOf: () => undefined,
  pagesX: layout.pagesX, pageCount: layout.pageCount
});
ok('no geometry means no jobs', none.count, 0);
ok('and no rays', none.texels, 0);
