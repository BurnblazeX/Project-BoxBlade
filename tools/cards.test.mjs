import { file, section, ok, near, truthy, falsy, note } from './lib/harness.mjs';
import { createCard, cardDistance, cardFacingFor, cardVisibility,
         cardsVisibility, buildCardAtlas, packCardInstances,
         cullCardsForLight, CARD_PAD, CARD_RANGE,
         CARD_LOD_C0_METRES, CARD_LOD_LEVELS, SUN_CARD_LOD_SHIFT,
         SUN_CARD_LOD_MAX } from '../js/cards.js';
import { VOXEL_METRES, CASCADE_COUNT } from '../js/boxgrid.js';
import { decodeDistance } from '../js/boxgrid.js';
import { facingToward } from '../js/silhouette.js';

file('cards.test.mjs - analytic sprite cards, one facing per light');

// An H again: two legs and a gap. The gap is what a box cannot do and what every
// assertion about shape here is really checking.
const W = 7, H = 9;
const hMask = new Uint8Array(W * H);
for (let y = 0; y < H; y++) { hMask[y * W + 1] = 1; hMask[y * W + 5] = 1; }
for (let x = 1; x <= 5; x++) hMask[4 * W + x] = 1;

const MPP = 0.125;
const card = createCard({ mask: hMask, w: W, h: H,
                          widthMetres: W * MPP, heightMetres: H * MPP });

section('the card keeps native resolution');

ok('columns are the sprite\'s own', card.cols, W);
ok('no reduction to any voxel size', card.metresPerPixel, MPP);
note('analytic, so there is no grid to match and no cascade to disagree with');

section('distance is in metres, signed, and finds the gap');

// Column 1 centre is 2 pixels left of the 3.5 centre line; column 3 is the gap.
const legA = (1 + 0.5 - W / 2) * MPP;
const gapA = (3 + 0.5 - W / 2) * MPP;
const highB = (H / 2 - 0.5 - 1) * MPP;      // row 1, above the crossbar
const barB = (H / 2 - 0.5 - 4) * MPP;       // row 4, the crossbar

truthy('a leg is inside', cardDistance(card, legA, highB) < 0);
truthy('the gap is outside', cardDistance(card, gapA, highB) > 0);
truthy('the crossbar closes the gap lower down',
       cardDistance(card, gapA, barB) < 0);
// Half a pixel in metres, from the same convention the 2D transform uses.
near('a pixel just outside reads half a pixel in metres',
     cardDistance(card, (0 + 0.5 - W / 2) * MPP, highB), 0.5 * MPP, 1e-6);

section('the facing is per light, per card');

// The whole reason for this path. Two lights in different directions give one
// sprite two different card orientations in the same frame - which is precisely
// what a single baked field cannot represent.
const centre = { x: 5, y: 1, z: 5 };
const toEast = cardFacingFor(centre, 15, 5);
const toNorth = cardFacingFor(centre, 5, 15);
near('a light due east faces the card east', -toEast.uz, 1, 1e-9);
near('a light due north faces it north', toNorth.ux, 1, 1e-9);
truthy('the two orientations differ', Math.abs(toEast.ux - toNorth.ux) > 0.5);
// And it is measured from the SPRITE, not from a global light vector - the same
// light is in a different direction from a sprite standing somewhere else.
const elsewhere = cardFacingFor({ x: 15, y: 1, z: 15 }, 15, 5);
truthy('the same light gives a different facing to a sprite elsewhere',
       Math.abs(elsewhere.ux - toEast.ux) > 0.5);

section('a ray through the gap is not occluded');

const lightE = { x: 15, y: 1, z: 5 };
const facing = cardFacingFor(centre, lightE.x, lightE.z);
// Receiver on the far side of the card from the light, level with it. The card
// faces east, so its width runs along z and the gap is an offset in z.
// ABOVE the crossbar. At the card's vertical centre the H's bar spans the gap,
// so a ray there is blocked by the shape and not by any fault in the test - it
// has to be aimed at the part of the silhouette that actually has a hole in it.
const rayY = centre.y + highB;
function visAt(dz, coneSlope = 0) {
  const from = { x: 1, y: rayY, z: centre.z + dz };
  const to = { x: lightE.x, y: rayY, z: lightE.z + dz };
  const dx = to.x - from.x, dy = to.y - from.y, dzz = to.z - from.z;
  const len = Math.hypot(dx, dy, dzz);
  return cardVisibility({
    card, centre, facing, from,
    dir: { x: dx / len, y: dy / len, z: dzz / len },
    maxDist: len, coneSlope
  });
}

// facing.u for a light due east is (0,-1), so card-space +a is world -z.
ok('a ray at a leg is blocked', visAt(2 * MPP), 0);
ok('a ray through the gap is clear', visAt(0), 1);
ok('a ray at the other leg is blocked', visAt(-2 * MPP), 0);
note('a box proxy blocks all three');

section('only occluders between the receiver and the light count');

const from = { x: 1, y: rayY, z: centre.z + 2 * MPP };
const dir = { x: 1, y: 0, z: 0 };
ok('the card blocks when it is in the way',
   cardVisibility({ card, centre, facing, from, dir, maxDist: 20 }), 0);
// Cap the ray short of the card: nothing between, so nothing blocked. This is
// what stops a sprite standing BEYOND a torch from shadowing it.
ok('a ray that stops short of it does not',
   cardVisibility({ card, centre, facing, from, dir, maxDist: 2 }), 1);
// And one pointing away.
ok('a ray pointing away from it does not',
   cardVisibility({ card, centre, facing, from,
                    dir: { x: -1, y: 0, z: 0 }, maxDist: 20 }), 1);

section('the penumbra widens with distance, like the cone trace');

// Same geometry, same formula: visibility is d / (slope * t), so the soft band
// either side of an edge grows with how far the ray has travelled. Sampled just
// outside a leg, where a hard test reads fully lit.
const edge = 2.6 * MPP;   // a shade outside the leg's edge
const nearSlope = 0.02, farSlope = 0.20;
const vNear = visAt(edge, nearSlope);
const vFar = visAt(edge, farSlope);
note(`just outside the leg: ${vNear.toFixed(3)} at slope ${nearSlope}, ` +
     `${vFar.toFixed(3)} at ${farSlope}`);
ok('a hard test reads this point as lit', visAt(edge, 0), 1);
truthy('a wider cone darkens it', vFar < vNear);
truthy('and it stays a partial value, not a hard edge', vFar > 0 && vFar < 1);

section('several cards each take their own bite');

const twin = [
  { card, centre, facing, flip: false },
  { card, centre: { x: 3, y: centre.y, z: centre.z }, facing, flip: false }
];
const start = { x: 1, y: rayY, z: centre.z + 2 * MPP };
const d2 = { x: 1, y: 0, z: 0 };
ok('two cards in a row still block completely',
   cardsVisibility(twin, start, d2, 20), 0);
// Through both gaps.
const startGap = { x: 1, y: rayY, z: centre.z };
ok('and let a ray through when both have a gap there',
   cardsVisibility(twin, startGap, d2, 20), 1);

section('flip mirrors the card');

const lMask = new Uint8Array(4 * 4);
for (let y = 0; y < 4; y++) lMask[y * 4] = 1;
const lCard = createCard({ mask: lMask, w: 4, h: 4,
                           widthMetres: 4 * MPP, heightMetres: 4 * MPP });
const leftA = (0 + 0.5 - 2) * MPP, rightA = (3 + 0.5 - 2) * MPP;
truthy('unflipped, the bar is at -a',
       cardDistance(lCard, leftA, 0) < 0 && cardDistance(lCard, rightA, 0) > 0);
truthy('flipped, it is at +a',
       cardDistance(lCard, rightA, 0, true) < 0 &&
       cardDistance(lCard, leftA, 0, true) > 0);

section('the atlas carries every card in one texture');

// A shader cannot index a texture by a loop variable, and the loop over cards
// is the whole point - so "which texture" has to become "which rectangle".
const lCardForAtlas = lCard;
const atlas = buildCardAtlas([card, lCardForAtlas]);
ok('as wide as both cards together', atlas.width, card.w + lCardForAtlas.w);
ok('as tall as the taller one', atlas.height, Math.max(card.h, lCardForAtlas.h));
ok('one rect per card', atlas.rects.length, 2);
ok('the second starts where the first ends', atlas.rects[1].x, card.w);

// The bytes must decode back to the distances, through the SAME encoding the
// boxGrid uses - one distance format in the renderer, not two that can drift.
const probeX = atlas.rects[0].x + CARD_PAD + 1;      // inside the H's left leg
const probeY = CARD_PAD + 1;
near('a stored byte decodes to the distance it was built from',
     decodeDistance(atlas.data[probeY * atlas.width + probeX], CARD_RANGE),
     card.dt[probeY * card.w + (CARD_PAD + 1)], 0.05);

section('instances pack to four vec4 each');

const buf = new Float32Array(4 * 16);
const inst = [
  { card, typeIndex: 0, centre: { x: 1, y: 2, z: 3 }, facing: { ux: 0, uz: -1 }, flip: false },
  { card: lCardForAtlas, typeIndex: 1, centre: { x: 4, y: 5, z: 6 }, facing: { ux: 1, uz: 0 }, flip: true }
];
ok('both written', packCardInstances(inst, atlas, buf, 4), 2);
ok('centre x', buf[0], 1);
ok('flip is +1 when not flipped', buf[3], 1);
ok('flip is -1 when flipped', buf[16 + 3], -1);
ok('half width in metres', buf[6], card.widthMetres / 2);
ok('atlas rect of the second card', buf[16 + 8], card.w);
ok('metres per pixel', buf[12], card.metresPerPixel);

// The cap bounds cards TESTED, not sprites in the world.
const small = new Float32Array(1 * 16);
ok('a tight cap drops the overflow rather than overrunning',
   packCardInstances(inst, atlas, small, 1), 1);

section('culling is per light, by that light\'s own reach');

const many = [];
for (let i = 0; i < 40; i++) {
  many.push({ centre: { x: i, y: 0, z: 0 }, enabled: true, id: i });
}
const near5 = cullCardsForLight(many, { x: 0, y: 0, z: 0 }, 5.5, 100);
ok('only what the light can reach', near5.length, 6);
ok('nearest first', near5[0].id, 0);
const capped = cullCardsForLight(many, { x: 0, y: 0, z: 0 }, 100, 8);
ok('the cap keeps the nearest', capped.length, 8);
ok('and drops the far ones', capped[7].id, 7);
note('a scene can hold any number of sprites - what is bounded is how many ' +
     'compete to shadow ONE light in ONE frame');

section('cascade LOD - a card follows the level it stands in');
{
  ok('the LOD copy of C0 voxel size matches boxgrid', CARD_LOD_C0_METRES, VOXEL_METRES);
  ok('the LOD copy of the level count matches boxgrid', CARD_LOD_LEVELS, CASCADE_COUNT);
  // A solid card straight ahead; rays aimed at a sweep of heights across its
  // top edge. Coarser levels move where the edge lands to a coarser grid.
  const solid = createCard({ mask: new Uint8Array(16 * 16).fill(1), w: 16, h: 16,
                             widthMetres: 1, heightMetres: 1 });
  const at = (lod, y, lodShift = 0, lodMax = 2) => cardVisibility({
    card: solid, centre: { x: 0, y: 0, z: 2 }, facing: { ux: 1, uz: 0 },
    from: { x: 0, y, z: 0 }, dir: { x: 0, y: 0, z: 1 }, maxDist: 10, lod, lodShift, lodMax });
  truthy('C0 sees the edge at full resolution', at(0, 0.45) === 0 && at(0, 0.55) === 1);
  // C2 snaps to 0.5 m: 0.45 lands in the cell centred at 0.25, inside.
  // Point lights run the full ladder: C2 snaps to 0.5 m, so 0.45 lands in the
  // cell centred at 0.25, inside.
  ok('point light: C2 range is C2 quality', at(2, 0.45), 0);
  // The sun never goes past C1: 0.25 m cells, so 0.55 is in the one at 0.625.
  ok('sun: C2 range is C1 quality, never 0.5 m',
     at(2, 0.55, SUN_CARD_LOD_SHIFT, SUN_CARD_LOD_MAX), 1);
  // A card 1.1 m tall: its top edge (0.55) falls mid-way through a 25 cm cell,
  // so full resolution and C1 disagree at 0.5 - inside at full resolution, but
  // snapped to that cell's centre at 0.575, outside.
  const tall = createCard({ mask: new Uint8Array(16 * 16).fill(1), w: 16, h: 16,
                            widthMetres: 1.1, heightMetres: 1.1 });
  const tallAt = (lod, lodShift) => cardVisibility({
    card: tall, centre: { x: 0, y: 0, z: 2 }, facing: { ux: 1, uz: 0 },
    from: { x: 0, y: 0.5, z: 0 }, dir: { x: 0, y: 0, z: 1 }, maxDist: 10, lod, lodShift });
  ok('sun: C1 range is full resolution', tallAt(1, SUN_CARD_LOD_SHIFT), 0);
  ok('point light: C1 range is C1 quality', tallAt(1, 0), 1);
  ok('past the outermost band the card has faded out entirely', at(2.999999, 0.0) > 0.99, true);
}
