import { file, section, ok, near, truthy, falsy, note } from './lib/harness.mjs';
import { World, createTestArea } from '../js/world.js';
import { createBoxGridAt, isOccupied, worldToVoxel, sphereTrace,
         DISTANCE_RANGE, VOXEL_METRES, CASCADE_COUNT,
         cascadeVoxelMetres } from '../js/boxgrid.js';
import {
  maskFromRGBA, downsampleMask, signedDistanceTransform,
  createSilhouette, createSilhouetteSet, silhouetteDistance, ALPHA_THRESHOLD,
  facingToward, AXIS_X, AXIS_Z
} from '../js/silhouette.js';
import { createOccluder, placeOccluder, applyOccluders } from '../js/occluders.js';

file('silhouette.test.mjs - tracing against the sprite cutout');

// Synthetic masks throughout. Reading a real PNG needs a decoder or a canvas,
// and neither belongs in a headless suite - so the DOM-shaped half
// (maskFromRGBA) is fed bytes directly and everything downstream of it is
// exercised against shapes whose right answer can be worked out by hand.

section('alpha becomes a mask');

// Four pixels: opaque, transparent, exactly at the threshold, and above it.
const rgba = new Uint8Array([
  0, 0, 0, 255,
  0, 0, 0, 0,
  0, 0, 0, 127,
  0, 0, 0, 128
]);
const m = maskFromRGBA(rgba, 4, 1);
ok('opaque is solid', m[0], 1);
ok('transparent is not', m[1], 0);
ok('just under the threshold is not', m[2], 0);
ok('just over the threshold is', m[3], 1);
note(`threshold is ${ALPHA_THRESHOLD}, matching the sprite materials' alphaTest`);

section('downsampling keeps thin features');

// A single-pixel vertical line - a trunk, or an arm. Averaging would lose it;
// "any set pixel wins" must not.
const line = new Uint8Array(4 * 4);
for (let y = 0; y < 4; y++) line[y * 4 + 1] = 1;
const half = downsampleMask(line, 4, 4, 2);
ok('halved width', half.w, 2);
ok('halved height', half.h, 2);
ok('the line survives', half.mask[0] + half.mask[2], 2);

section('the distance transform is exact and signed');

// A 2x2 solid block in the middle of a 6x6 field. Every distance below is a
// hand-computed Euclidean one, which is the point of using an exact transform
// rather than a chamfer approximation - an approximate distance is an
// approximate STEP LENGTH, and a sphere trace that steps too far tunnels.
const w = 6, h = 6;
const block = new Uint8Array(w * h);
for (const [x, y] of [[2, 2], [3, 2], [2, 3], [3, 3]]) block[y * w + x] = 1;
const dt = signedDistanceTransform(block, w, h);
const at = (x, y) => dt[y * w + x];

// Inside: nearest clear pixel is one step away, so -(1 - 0.5).
near('inside reads half a pixel to the face', at(2, 2), -0.5, 1e-6);
// Orthogonally adjacent outside: one step out, so +0.5.
near('adjacent outside reads half a pixel', at(1, 2), 0.5, 1e-6);
near('adjacent outside, other side', at(4, 3), 0.5, 1e-6);
// Two out: 2 - 0.5.
near('two pixels out', at(0, 2), 1.5, 1e-6);
// Diagonal from the corner: sqrt(2) - 0.5, and this is the value a chamfer
// transform would get wrong.
near('diagonal is Euclidean, not chamfer', at(1, 1), Math.SQRT2 - 0.5, 1e-6);

// The half-pixel convention is what keeps the surface BETWEEN pixel centres.
// Without it every near-edge distance is doubled and the first step off a
// surface clears it by twice the room actually available.
truthy('the zero crossing sits between the centres',
       at(2, 2) < 0 && at(1, 2) > 0 &&
       Math.abs(at(2, 2)) === Math.abs(at(1, 2)));

section('the extrusion is a slab, not a plane');

// An 8x8 solid square as a card, so in-plane distance is easy to reason about
// and everything interesting is happening across the card.
const solid = new Uint8Array(8 * 8).fill(1);
const sil = createSilhouette({
  mask: solid, w: 8, h: 8,
  widthMetres: 8 * VOXEL_METRES, heightMetres: 8 * VOXEL_METRES,
  voxelMetres: VOXEL_METRES
});
ok('no downsampling when already at voxel density', sil.cols, 8);
ok('padded by the band width on each side', sil.w, 8 + sil.pad * 2);

// Dead centre of the card: inside on both axes, so the distance is to the
// NEAREST face, which is the slab's own surface half a voxel away.
near('the centre of the card is half a voxel inside',
     silhouetteDistance(sil, 0, 0, 0), -0.5, 1e-6);
// Straight out along the normal.
near('one voxel out along the normal',
     silhouetteDistance(sil, 0, 0, 1.5), 1.0, 1e-6);
// A zero-thickness card would report 0 here and a ray running along the plane
// would never register a hit - a light grazing the ground would shine straight
// through a character standing in it.
truthy('the card has real thickness', sil.halfThickness > 0);

section('a cutout casts its own shape, a box does not');

// An H: two legs with a gap between them. The gap is the whole point - it is
// what a box proxy cannot produce and what makes a shadow read as a character.
const W = 7, H = 9;
const hMask = new Uint8Array(W * H);
for (let y = 0; y < H; y++) {
  hMask[y * W + 1] = 1;               // left leg
  hMask[y * W + 5] = 1;               // right leg
}
for (let x = 1; x <= 5; x++) hMask[4 * W + x] = 1;   // crossbar

const hSil = createSilhouette({
  mask: hMask, w: W, h: H,
  widthMetres: W * VOXEL_METRES, heightMetres: H * VOXEL_METRES,
  voxelMetres: VOXEL_METRES
});

// Card space: u from the centre, v up. Column 1 is 2.5 left of centre (centre
// is 3.5), column 3 is 0.5 left. Row 0 is the TOP of the image, so it is +v.
const legU = 1 - W / 2 + 0.5;        // -2.0
const gapU = 3 - W / 2 + 0.5;        //  0.0
const highV = H / 2 - 0.5 - 1;       // row 1, above the crossbar

truthy('a leg is solid', silhouetteDistance(hSil, legU, highV, 0) < 0);
truthy('the gap between the legs is NOT solid',
       silhouetteDistance(hSil, gapU, highV, 0) > 0);
// And the crossbar closes the gap lower down, which proves v is not inverted -
// a silhouette baked upside down looks almost right, which is the worst kind of
// wrong.
const barV = H / 2 - 0.5 - 4;        // row 4
truthy('the crossbar closes the gap', silhouetteDistance(hSil, gapU, barV, 0) < 0);

section('flipping mirrors the cutout');

// An L, so left and right are distinguishable.
const lMask = new Uint8Array(4 * 4);
for (let y = 0; y < 4; y++) lMask[y * 4] = 1;    // a bar down the left edge
const lSil = createSilhouette({
  mask: lMask, w: 4, h: 4,
  widthMetres: 4 * VOXEL_METRES, heightMetres: 4 * VOXEL_METRES,
  voxelMetres: VOXEL_METRES
});
const leftU = -1.5, rightU = 1.5;
truthy('unflipped, the bar is on the left',
       silhouetteDistance(lSil, leftU, 0, 0) < 0 &&
       silhouetteDistance(lSil, rightU, 0, 0) > 0);
truthy('flipped, the bar is on the right',
       silhouetteDistance(lSil, rightU, 0, 0, true) < 0 &&
       silhouetteDistance(lSil, leftU, 0, 0, true) > 0);

section('it bakes into the field, and the gap stays open');

World.clear();
createTestArea(12, 12);
const grid = createBoxGridAt(0, 0, null, 0);
const clean = Uint8Array.from(grid.data);

// The H, stood up on the ground at a known spot, 7 x 9 voxels.
const STAND = { x: 9, z: 9 };
const proxy = createOccluder('h', { silhouettes: [hSil] });
// Card axis defaults to x, so the cutout lies in the XY plane and its normal
// runs along z. Feet on the slab's top face.
placeOccluder(proxy, STAND.x, 0.75, STAND.z);
truthy('applying reports a change', applyOccluders(grid, [proxy]));

// Sample across the shape at a height above the crossbar. The proxy's centre
// sits height/2 above the feet; work in world metres from there.
const centreY = 0.75 + (H * VOXEL_METRES) / 2;
const sampleY = centreY + highV * VOXEL_METRES;
const legWorld = STAND.x + legU * VOXEL_METRES;
const gapWorld = STAND.x + gapU * VOXEL_METRES;

const legV = worldToVoxel(grid, legWorld, sampleY, STAND.z);
const gapV = worldToVoxel(grid, gapWorld, sampleY, STAND.z);
truthy('the leg is solid in the field', isOccupied(grid, legV.vx, legV.vy, legV.vz));
falsy('the gap is open in the field', isOccupied(grid, gapV.vx, gapV.vy, gapV.vz));

// The thing a box proxy could never do: a ray through the gap gets through.
const through = sphereTrace(grid, { x: gapWorld, y: sampleY, z: STAND.z - 2 },
                            { x: 0, y: 0, z: 1 }, 4);
const blocked = sphereTrace(grid, { x: legWorld, y: sampleY, z: STAND.z - 2 },
                            { x: 0, y: 0, z: 1 }, 4);
truthy('a ray at the leg is blocked', blocked.hit);
falsy('a ray through the gap is not', through.hit);
note('a box proxy blocks both - this is the whole reason for the cutout');

section('the restore path still holds with a silhouette');

// The band is irregular now rather than a box, so the rect that clears it has
// to be sized from the silhouette's extent. If it is not, a shell of short
// distances survives around where the character stood.
truthy('removal reports a change', applyOccluders(grid, []));
let diff = 0;
for (let i = 0; i < clean.length; i++) if (clean[i] !== grid.data[i]) diff++;
note(`${diff} bytes differ from a field that never had a silhouette in it`);
ok('field is byte-for-byte what it was before', diff, 0);

section('turning around rewrites the shape without moving it');

applyOccluders(grid, [placeOccluder(proxy, STAND.x, 0.75, STAND.z, false)]);
truthy('flipping is detected as a change',
       applyOccluders(grid, [placeOccluder(proxy, STAND.x, 0.75, STAND.z, true)]));
falsy('and flipping back to the same state is not',
      applyOccluders(grid, [placeOccluder(proxy, STAND.x, 0.75, STAND.z, true)]));
applyOccluders(grid, []);

section('a denser-than-spec sprite still lands on the voxel grid');

// Every shipped sprite is at the frozen 8 px/m today, so this exercises the
// path rather than a current asset - deliberately, because the drift is not
// theoretical: decor_tree.png was 48 x 72 over a 3.0 x 4.5 m quad, twice the
// density, until it was brought back to spec. S8 calls that constant a hard
// requirement precisely because assets wander off it, so the reduction has to
// keep working whether or not anything needs it this week.
const dense = new Uint8Array(48 * 72).fill(1);
const treeSil = createSilhouette({
  mask: dense, w: 48, h: 72,
  widthMetres: 3.0, heightMetres: 4.5,
  voxelMetres: VOXEL_METRES
});
ok('48 px over 3.0 m reduces to 24 voxels', treeSil.cols, 24);
ok('72 px over 4.5 m reduces to 36 voxels', treeSil.rows, 36);
note('2 px per voxel - the asset is at twice the frozen texel density');

section('the card turns to face the light');

// facingToward returns the card's WIDTH direction, at right angles to the light
// it is handed - so that the card's NORMAL points along the light and the full
// silhouette faces it.
const east = facingToward(1, 0);
near('a light along +x puts the card width along z', east.ux, 0, 1e-9);
near('...with unit length', Math.hypot(east.ux, east.uz), 1, 1e-9);
const diag = facingToward(1, 1);
near('a diagonal light gives a diagonal card', Math.abs(diag.ux), Math.SQRT1_2, 1e-9);
near('still unit length', Math.hypot(diag.ux, diag.uz), 1, 1e-9);
// Directly overhead there is no azimuth to face; the fallback must be a usable
// vector rather than a zero one, or the card collapses and the bake divides by
// nothing.
const overhead = facingToward(0, 0);
near('an overhead sun falls back to a unit facing',
     Math.hypot(overhead.ux, overhead.uz), 1, 1e-9);

section('a turned card still casts its cutout, gap and all');

// The whole point of following the sun: at NO azimuth should the silhouette
// degenerate. Sweep the facing right round and check the H's gap survives every
// orientation - the fixed-axis version thins to a line twice per revolution.
World.clear();
createTestArea(12, 12);
const swept = createBoxGridAt(0, 0, null, 0);
const sweepProxy = createOccluder('sweep', { silhouettes: [hSil] });
placeOccluder(sweepProxy, STAND.x, 0.75, STAND.z);

let worstLeg = Infinity, everLostGap = false, orientations = 0;
for (let deg = 0; deg < 360; deg += 15) {
  const r = deg * Math.PI / 180;
  const facing = facingToward(Math.cos(r), Math.sin(r));
  applyOccluders(swept, [sweepProxy], facing);

  // Probe along the card's own width direction, which is where the legs and the
  // gap lie whatever the azimuth. Card space u -> world offset u*(ux,uz).
  const probe = (u) => {
    const wx = STAND.x + u * VOXEL_METRES * facing.ux;
    const wz = STAND.z + u * VOXEL_METRES * facing.uz;
    const v = worldToVoxel(swept, wx, sampleY, wz);
    return isOccupied(swept, v.vx, v.vy, v.vz);
  };
  const legSolid = probe(legU) || probe(legU + 0.5) || probe(legU - 0.5);
  const gapOpen = !probe(gapU);
  if (!legSolid) worstLeg = Math.min(worstLeg, deg);
  if (!gapOpen) everLostGap = true;
  orientations++;
}
applyOccluders(swept, []);

note(`swept ${orientations} orientations at 15 degree steps`);
ok('a leg is solid at every azimuth', worstLeg, Infinity);
falsy('the gap never closes up', everLostGap);

section('turning the sun rewrites the cutouts in place');

const turned = createBoxGridAt(0, 0, null, 0);
const p = placeOccluder(createOccluder('t', { silhouettes: [hSil] }),
                        STAND.x, 0.75, STAND.z);
truthy('first apply writes it', applyOccluders(turned, [p], AXIS_X));
falsy('the same facing again is free', applyOccluders(turned, [p], AXIS_X));
// The proxy has not moved at all - only the light has. Without the facing being
// part of the remembered box this reports nothing to do and every cutout keeps
// the previous sun's orientation until its owner happens to take a step.
truthy('a new facing is detected even though nothing moved',
       applyOccluders(turned, [p], facingToward(1, 1)));
applyOccluders(turned, []);

section('a rotated card still restores exactly');

// The band of a turned card is a rotated slab, so the rect that clears it is
// sized from the rotated bounds. Sized from the axis-aligned ones instead, the
// corners it missed would keep their old values.
const restore = createBoxGridAt(0, 0, null, 0);
const cleanRot = Uint8Array.from(restore.data);
const rp = placeOccluder(createOccluder('r', { silhouettes: [hSil] }),
                         STAND.x, 0.75, STAND.z);
applyOccluders(restore, [rp], facingToward(1, 1));   // worst case, 45 degrees
applyOccluders(restore, []);
let rotDiff = 0;
for (let i = 0; i < cleanRot.length; i++) {
  if (cleanRot[i] !== restore.data[i]) rotDiff++;
}
note(`${rotDiff} bytes differ after a 45-degree card came and went`);
ok('a diagonal card leaves nothing behind', rotDiff, 0);

section('bilinear sampling is exact at pixel centres');

// The rotated case needs interpolation, but the axis-aligned case must not pay
// for it - at a pixel centre the weight is zero, so the old nearest answers have
// to come back bit for bit or every existing expectation above is now measuring
// something subtly different.
for (const [u, v] of [[legU, highV], [gapU, highV], [gapU, barV]]) {
  const d = silhouetteDistance(hSil, u, v, 0);
  const cu = u + W / 2 + hSil.pad, cv = H / 2 - v + hSil.pad;
  const raw = hSil.dt[Math.floor(cv) * hSil.w + Math.floor(cu)];
  const slab = Math.min(Math.max(raw, -hSil.halfThickness), 0);
  const expect = (raw > 0 ? raw : 0) + slab;
  near(`exact at pixel centre (${u}, ${v})`, d, expect, 1e-6);
}

section('the cutout is not mirrored');

// THE TEST THE SYMMETRIC SHAPES ABOVE CANNOT DO. An H is the same shape as its
// own mirror image, so every assertion so far passes whichever way round the
// card faces - which is how a back-to-front card shipped looking like a working
// shadow that happened to be right-side-left.
//
// So: a mask with a bar down its LEFT edge and nothing on the right, and an
// assertion about which side of the world that bar lands on.
//
// Worked through for a sun toward +x, so the expected answer comes from the
// geometry rather than from running it and writing down what happened:
//
//   facing u = (dz, -dx) = (0, -1), so the card's width runs along -z.
//   Column 0's centre is at card-space u = 0.5 - 5/2 = -2.0.
//   Its world offset is u * (ux, uz) * voxel = -2.0 * (0, -1) * 0.125 = +0.25 z.
//
// and independently, as the cross-check: a viewer at the sun looks along -x, so
// their right is (dz, -dx) = -z and their left is +z. The bar is drawn on the
// image's left, so it must appear on that viewer's left, which is +z. The two
// agree, and a flipped sign would put the bar at -z on both counts.
const BW = 5, BH = 8;
const barMask = new Uint8Array(BW * BH);
for (let y = 0; y < BH; y++) barMask[y * BW] = 1;      // left edge only
const barSil = createSilhouette({
  mask: barMask, w: BW, h: BH,
  widthMetres: BW * VOXEL_METRES, heightMetres: BH * VOXEL_METRES,
  voxelMetres: VOXEL_METRES
});

// The normal is what the bake actually builds from the facing, so assert on it
// directly too - it is the more fundamental of the two statements.
const sunEast = facingToward(1, 0);
near('a card facing a +x sun has its normal along +x', -sunEast.uz, 1, 1e-9);
near('...and no z component to that normal', sunEast.ux, 0, 1e-9);

World.clear();
createTestArea(12, 12);
const mir = createBoxGridAt(0, 0, null, 0);
const barProxy = placeOccluder(createOccluder('bar', { silhouettes: [barSil] }),
                               STAND.x, 0.75, STAND.z);

const barY = 0.75 + (BH * VOXEL_METRES) / 2;   // mid-height of the card
const sideIsSolid = (facing, dz) => {
  applyOccluders(mir, [barProxy], facing);
  const v = worldToVoxel(mir, STAND.x, barY, STAND.z + dz);
  const hit = isOccupied(mir, v.vx, v.vy, v.vz);
  applyOccluders(mir, []);
  return hit;
};

truthy('with the sun at +x, the bar lands on +z', sideIsSolid(sunEast, 0.25));
falsy('and not on -z', sideIsSolid(sunEast, -0.25));

// Turn the sun right round: the card turns with it, so the bar has to swap
// sides. A card that did NOT follow would keep the bar where it was, and one
// with the sign wrong would have had it on the wrong side both times.
const sunWest = facingToward(-1, 0);
truthy('with the sun at -x, the bar lands on -z', sideIsSolid(sunWest, -0.25));
falsy('and no longer on +z', sideIsSolid(sunWest, 0.25));

// The same statement once more along the other axis, so the check is not
// accidentally passing on one special case.
const sunNorth = facingToward(0, 1);
near('a card facing a +z sun has its normal along +z', sunNorth.ux, 1, 1e-9);
truthy('with the sun at +z, the bar lands on -x',
       (() => {
         applyOccluders(mir, [barProxy], sunNorth);
         const v = worldToVoxel(mir, STAND.x - 0.25, barY, STAND.z);
         const hit = isOccupied(mir, v.vx, v.vy, v.vz);
         applyOccluders(mir, []);
         return hit;
       })());

section('the held torch escapes its own holder');

// The flame sits inside the character unless it is pushed clear, and once the
// character is a silhouette in the field that means every ray toward the light
// hits the proxy on its first step - the torch shadows itself from every
// direction at once and reads as a dead light.
//
// The trap this pins down is that the two directions involved are INDEPENDENT.
// The card follows the sun's azimuth; "forward" follows the camera. So a single
// sweep over either one proves nothing - it has to be swept over both, and the
// bad case is the pairing where the camera looks along the card's plane.
//
// Modelled on character_Bob measured off the asset: 12 x 24 with the figure
// reaching 5 px from the mask's centre at torch height, which is 0.625 m - too
// wide to escape by moving forward at any distance a held torch should be at.
const BOBW = 12, BOBH = 24;
const bobMask = new Uint8Array(BOBW * BOBH);
for (let y = 8; y < BOBH; y++) {
  for (let x = 1; x <= 9; x++) bobMask[y * BOBW + x] = 1;
}
const bobSil = createSilhouette({
  mask: bobMask, w: BOBW, h: BOBH,
  widthMetres: 1.5, heightMetres: 3.0, voxelMetres: VOXEL_METRES
});

World.clear();
createTestArea(12, 12);
const lit = createBoxGridAt(0, 0, null, 0);
const holder = placeOccluder(createOccluder('holder', { silhouettes: [bobSil] }),
                             STAND.x, 0.75, STAND.z);

const TORCH_H = 0.9, FORWARD = 0.4, CLEAR = 0.25, SIDE = 0.3;

// The position updateTorchPosition computes, reproduced from the same rule so
// the geometry is what is under test rather than the wiring.
//
// `flip` is Bob having turned around: the sprite mirrors, so the hand holding
// the torch swaps sides with him.
function flamePos(facing, camX, camZ, flip = false) {
  const nx = -facing.uz, nz = facing.ux;
  // The sprite's own right, for a flattened forward: (fz, -fx).
  const rx = camZ, rz = -camX;
  const side = SIDE * (flip ? -1 : 1);
  const ox = camX * FORWARD + rx * side;
  const oz = camZ * FORWARD + rz * side;
  // Signed off the FULL offset, not off forward - see updateTorchPosition. With
  // a sideways term in play a clearance signed off forward alone can be pushing
  // one way while the side offset pulls the other.
  const sgn = (ox * nx + oz * nz) >= 0 ? 1 : -1;
  return {
    x: STAND.x + ox + nx * CLEAR * sgn,
    y: 0.75 + TORCH_H,
    z: STAND.z + oz + nz * CLEAR * sgn
  };
}

let insideCount = 0, worstPair = null, pairs = 0;
let forwardOnlyInside = 0;
for (let sun = 0; sun < 360; sun += 30) {
  const sr = sun * Math.PI / 180;
  const facing = facingToward(Math.cos(sr), Math.sin(sr));
  applyOccluders(lit, [holder], facing);

  for (let cam = 0; cam < 360; cam += 30) {
    const cr = cam * Math.PI / 180;
    const cx = Math.cos(cr), cz = Math.sin(cr);
    pairs++;

    // Both ways Bob can be facing. The side offset changes the answer, so a
    // sweep over one of them would only be testing half the positions the
    // flame actually takes.
    for (const flip of [false, true]) {
      const p = flamePos(facing, cx, cz, flip);
      const v = worldToVoxel(lit, p.x, p.y, p.z);
      if (isOccupied(lit, v.vx, v.vy, v.vz)) {
        insideCount++;
        if (!worstPair) {
          worstPair = `sun ${sun} deg, camera ${cam} deg, ` +
                      `${flip ? 'turned' : 'forward'}`;
        }
      }
    }

    // The same check WITHOUT the normal clearance, which is the version that
    // looks reasonable and silently fails - recorded as a number so the fix is
    // shown to be doing something rather than asserted to be.
    const q = { x: STAND.x + cx * FORWARD + cz * SIDE,
                y: 0.75 + TORCH_H,
                z: STAND.z + cz * FORWARD - cx * SIDE };
    const qv = worldToVoxel(lit, q.x, q.y, q.z);
    if (isOccupied(lit, qv.vx, qv.vy, qv.vz)) forwardOnlyInside++;
  }
}
applyOccluders(lit, []);

note(`${pairs} sun/camera pairs at 30 degree steps, each facing both ways`);
note(`forward offset alone would leave the flame inside for ${forwardOnlyInside} of them`);
ok('the flame is never inside the holder', insideCount, 0);
truthy('and the clearance is what achieves that - forward alone is not enough',
       forwardOnlyInside > 0);
if (worstPair) note(`first failure: ${worstPair}`);

section('the holder still shadows its own torch from behind');

// The flame must escape the silhouette, NOT stop interacting with it. A torch
// held in front of a character should still be blocked by that character for
// anything standing behind them - losing that would trade one wrong look for
// another, and it is the easy over-correction here.
const facingE = facingToward(1, 0);
applyOccluders(lit, [holder], facingE);
const flame = flamePos(facingE, 0, 1);   // camera along +z

// A receiver on the far side of the CARD from the flame, which is not the same
// as "behind the character" in a loose sense. The flame is pushed out along the
// card's normal, so a receiver sitting in the card's own plane has a clear line
// to it whatever the z distance - the ray runs down the lit side and never
// crosses the sheet. To be shadowed it has to be on the opposite side of the
// normal: the card's normal here is +x, the flame is at +0.25 x, so the receiver
// goes to -x.
const behind = { x: STAND.x - 1.5, y: 0.75 + TORCH_H, z: STAND.z };
const toLight = {
  x: flame.x - behind.x, y: flame.y - behind.y, z: flame.z - behind.z
};
const len = Math.hypot(toLight.x, toLight.y, toLight.z);
const shadowed = sphereTrace(lit, behind,
                             { x: toLight.x / len, y: toLight.y / len,
                               z: toLight.z / len }, len);
truthy('a point behind the character is shadowed from the torch', shadowed.hit);

// ...and one out to the side, with nothing between it and the flame, is not.
const beside = { x: STAND.x + 2.0, y: 0.75 + TORCH_H, z: STAND.z + 1.0 };
const toL2 = { x: flame.x - beside.x, y: flame.y - beside.y, z: flame.z - beside.z };
const len2 = Math.hypot(toL2.x, toL2.y, toL2.z);
const clearLine = sphereTrace(lit, beside,
                              { x: toL2.x / len2, y: toL2.y / len2,
                                z: toL2.z / len2 }, len2);
falsy('a point beside it, with a clear line, is lit', clearLine.hit);
applyOccluders(lit, []);

section('the torch swaps sides when Bob turns');

// The flame is held in a hand, so it sits off to one side - and when the sprite
// mirrors, the hand mirrors with it. A flame that stayed put while the art
// flipped would read as the torch passing through him.
const camN = { x: 0, z: 1 };                 // camera looking along +z
const faceE = facingToward(1, 0);
const heldRight = flamePos(faceE, camN.x, camN.z, false);
const heldLeft = flamePos(faceE, camN.x, camN.z, true);

// The sprite's right for a forward of (0,1) is (fz, -fx) = (1, 0), so the side
// offset runs along x and the swap shows up there.
truthy('unturned, the flame is to one side', heldRight.x > STAND.x);
truthy('turned, it is to the other', heldLeft.x < STAND.x);
near('and the swap is symmetric about him',
     (heldRight.x - STAND.x) + (heldLeft.x - STAND.x), 0, 1e-9);
near('while the forward reach is unchanged',
     heldRight.z - STAND.z, heldLeft.z - STAND.z, 1e-9);

section('the side offset cannot cancel the clearance');

// The case that made the ORDER matter. With the camera looking along the card's
// width, the side offset is entirely NORMAL to the card - so it pulls on the
// very axis the clearance pushes on, and at 0.3 m against 0.25 m the side wins.
//
// Asserted on the normal component rather than on occupancy, deliberately. The
// quantity the fix actually guarantees is geometric: |n . offset| >= CLEAR, for
// every facing and both ways round. Occupancy cannot express that cleanly here -
// the naive ordering lands 0.05 m inside a 0.0625 m half-thickness, which is a
// tenth of a voxel, so isOccupied rounds it away and a test built on it would
// report the bug as absent rather than as small.
const HALF_THICK = VOXEL_METRES / 2;

function normalComponent(facing, camX, camZ, flip, signOffForwardOnly) {
  const nx = -facing.uz, nz = facing.ux;
  const rx = camZ, rz = -camX;
  const sd = SIDE * (flip ? -1 : 1);
  const ox = camX * FORWARD + rx * sd;
  const oz = camZ * FORWARD + rz * sd;
  const sgn = signOffForwardOnly
    ? ((camX * nx + camZ * nz) >= 0 ? 1 : -1)
    : ((ox * nx + oz * nz) >= 0 ? 1 : -1);
  return (ox + nx * CLEAR * sgn) * nx + (oz + nz * CLEAR * sgn) * nz;
}

let worstCorrect = Infinity, worstNaive = Infinity, cases = 0;
for (let sun = 0; sun < 360; sun += 15) {
  const sr = sun * Math.PI / 180;
  const facing = facingToward(Math.cos(sr), Math.sin(sr));
  for (let cam = 0; cam < 360; cam += 15) {
    const cr = cam * Math.PI / 180;
    const cx2 = Math.cos(cr), cz2 = Math.sin(cr);
    for (const flip of [false, true]) {
      cases++;
      worstCorrect = Math.min(worstCorrect,
        Math.abs(normalComponent(facing, cx2, cz2, flip, false)));
      worstNaive = Math.min(worstNaive,
        Math.abs(normalComponent(facing, cx2, cz2, flip, true)));
    }
  }
}

note(`${cases} facing/camera/turn combinations`);
note(`closest approach to the card: ${worstCorrect.toFixed(4)} m signing off the ` +
     `full offset, ${worstNaive.toFixed(4)} m signing off forward alone ` +
     `(the card's half-thickness is ${HALF_THICK.toFixed(4)} m)`);

truthy('signing off the full offset never gets closer than the clearance',
       worstCorrect >= CLEAR - 1e-9);
truthy('signing off forward alone puts the flame INSIDE the card',
       worstNaive < HALF_THICK);

// And the outer guarantee still holds in the field itself, both ways round.
applyOccluders(lit, [holder], facingToward(0, 1));
let cancelInside = 0;
for (const flip of [false, true]) {
  for (const [cx2, cz2] of [[1, 0], [-1, 0], [0.7071, 0.7071], [-0.7071, 0.7071]]) {
    const p = flamePos(facingToward(0, 1), cx2, cz2, flip);
    const v = worldToVoxel(lit, p.x, p.y, p.z);
    if (isOccupied(lit, v.vx, v.vy, v.vz)) cancelInside++;
  }
}
applyOccluders(lit, []);
ok('and the flame is outside the field in every one of them', cancelInside, 0);

section('every cascade agrees how big the character is');

// The bug this pins: a silhouette is expressed in the voxels of the grid it was
// built for, and minSilhouetteVoxels reads sil.cols as a count of GRID voxels.
// Share one across cascades and C1 - whose voxels are twice the size - describes
// a card twice as wide. Measured before the fix: 1.12 m at C0, 2.24 m at C1,
// 4.49 m at C2.
//
// A shadow ray leaves C0 after 18 m and continues in C1, so what that looked
// like was a sprite's shadow DOUBLING at the cascade seam - a hard line across
// the ground that reads as a broken blend between levels. Nothing needed
// blending; the levels disagreed about the size of the caster.
World.clear();
createTestArea(12, 12);

const setSil = createSilhouetteSet({
  mask: bobMask, w: BOBW, h: BOBH, widthMetres: 1.5, heightMetres: 3.0
});
ok('one silhouette per cascade', setSil.length, CASCADE_COUNT);
note(setSil.map((v, l) => `C${l} ${v.cols}x${v.rows} @ ` +
     `${cascadeVoxelMetres(l).toFixed(3)}m`).join(',  '));

// Each level must describe the SAME world-space object - coarser, not bigger.
for (let l = 0; l < CASCADE_COUNT; l++) {
  near(`C${l} spans 1.5 m of world across its columns`,
       setSil[l].cols * cascadeVoxelMetres(l), 1.5, 1e-9);
  near(`C${l} spans 3.0 m of world down its rows`,
       setSil[l].rows * cascadeVoxelMetres(l), 3.0, 1e-9);
}

// ...and the same again measured out of the baked field, which is what the ray
// actually reads. Scanning for the solid extent rather than trusting the header.
const widths = [];
for (let l = 0; l < CASCADE_COUNT; l++) {
  const g = createBoxGridAt(0, 0, null, l);
  const pr = placeOccluder(createOccluder('p', { silhouettes: setSil }),
                           STAND.x, 0.75, STAND.z);
  applyOccluders(g, [pr], AXIS_X);
  let lo = null, hi = null;
  for (let x = STAND.x - 4; x <= STAND.x + 4; x += 0.01) {
    const v = worldToVoxel(g, x, 0.75 + 0.9, STAND.z);
    if (isOccupied(g, v.vx, v.vy, v.vz)) { if (lo === null) lo = x; hi = x; }
  }
  widths.push(lo === null ? 0 : hi - lo);
  applyOccluders(g, []);
}
note(`baked width per cascade: ${widths.map(w => w.toFixed(2) + ' m').join(', ')}`);

// Not equality - a coarser grid quantises the edges differently, so a voxel of
// slack either way is expected and correct. What must not happen is doubling.
for (let l = 1; l < CASCADE_COUNT; l++) {
  const slack = cascadeVoxelMetres(l) * 2;
  truthy(`C${l} is the same size as C0, within its own quantisation`,
         Math.abs(widths[l] - widths[0]) <= slack);
}
