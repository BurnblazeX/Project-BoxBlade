// --- Occlusion cut-out: the terrain between the camera and the player ---
//
// As voxel games cut away (whole blocks, never below the player's level) with
// Baldur's Gate 3's gate and fade (only while a cast says the player is hidden,
// eased in and out; a circle around them; floors never cut):
//
//   the cut set   which blocks MAY be cut (cellCut): any part of it could be
//                 nearer the camera than the player, above their feet, near
//                 their sight line - generous, so a block joins or leaves it
//                 only where it is not cut at all. Gathered on the CPU along
//                 the sight line (cutTargets), each block eased in and out on
//                 its own (core/interior.js easeCells).
//   the reveal    a buried face shows when the block across it is in the set
//                 (gpu/terrain.js terrainShapeTSL): the cut opens onto the real
//                 floor and the real block behind, never into an empty inside.
//   the fade      per PIXEL, so nothing changes a whole block at once: how
//                 cut is the product of smooth ramps (pixelCut) - its depth
//                 nearer than the player's, its height above their feet, its
//                 place toward the camera from them along the ground - and
//                 a radial gradient on screen around them (cellRadial), and it
//                 is drawn translucent at 1 - that. As the player moves, the
//                 ramps slide across the blocks; nothing pops.
//
// Only while the player is WHOLLY hidden (playerHidden): rays from points all
// over them to the camera, every one blocked by an opaque block. If any part
// is in view - in the open, beside a wall, behind glass - nothing is cut.
// Never under a roof (core/interior.js): the roof's own fade is the effect there.
// Glass is never cut: it hides nothing.
//
// This module is the arithmetic, headless; gpu/occlusion.js is the same in TSL.
import { BLOCK_METRES, isSolidAt, getBlock } from './world.js';
import { VOXEL_METRES } from './boxgrid.js';
import { isGlassMaterial } from './materials.js';
import { maskIndex, opaqueAt } from './interior.js';

export const CENTRE_LIFT = 1;       // the player's centre above the feet, blocks
export const DEPTH_MARGIN = 0.5;    // a pixel starts to cut this much nearer than the centre (blocks)
export const DEPTH_RAMP = 0.5;      // ... and is clear this much nearer again (blocks)
export const FEET_CLEARANCE = 0.1;  // a pixel starts to cut this far above the feet (blocks): floors stay
export const HEIGHT_RAMP = 0.4;     // ... and is clear this much higher again (blocks)
export const AHEAD_RAMP = 0.3;      // toward the camera from the player along the ground, cut by this (blocks):
                                    // never a wall behind them, however high
const HALF_DIAGONAL = 0.87;         // a block's farthest point from its centre (blocks)
export const NEAR_SLACK = 1;        // in the set this far past the circle (blocks), for its extent
export const CUT_DELAY = 0.25;      // hidden this long (s) before it starts to
export const DEFAULT_OCCLUSION = { radius: 2, fade: 0.6 };   // metres, at the player

const clamp01 = x => Math.min(1, Math.max(0, x));
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;

// The frame's inputs: the camera (eye, forward), the player's feet. Plain
// {x, y, z} objects in and out.
export function occlusionFrame(eye, forward, feet) {
  const centre = { x: feet.x, y: feet.y + CENTRE_LIFT * BLOCK_METRES, z: feet.z };
  const a = { x: centre.x - eye.x, y: centre.y - eye.y, z: centre.z - eye.z };
  const dist = Math.hypot(a.x, a.y, a.z) || 1;
  return { eye, forward, centre, feetY: feet.y, dist, axis: { x: a.x / dist, y: a.y / dist, z: a.z / dist },
           playerDepth: dot(a, forward) };
}

// How cut a pixel is before the circle, at its point p (metres): nearer the
// camera than the player, above their feet, toward the camera along the
// ground - each a ramp.
export function pixelCut(p, f) {
  const B = BLOCK_METRES;
  const d = { x: p.x - f.eye.x, y: p.y - f.eye.y, z: p.z - f.eye.z };
  const front = clamp01((f.playerDepth - DEPTH_MARGIN * B - dot(d, f.forward)) / (DEPTH_RAMP * B));
  const above = clamp01((p.y - f.feetY - FEET_CLEARANCE * B) / (HEIGHT_RAMP * B));
  const ahead = clamp01(-groundAhead(p, f) / (AHEAD_RAMP * B));
  return front * above * ahead;
}

// How far beyond the player p is along the ground, away from the camera
// (metres; < 0 toward it).
export function groundAhead(p, f) {
  const hx = f.forward.x, hz = f.forward.z, l = Math.hypot(hx, hz) || 1;
  return ((p.x - f.centre.x) * hx + (p.z - f.centre.z) * hz) / l;
}

// Whether the block whose cell's centre is c (metres) may be cut: 1 when any
// part of it could be (pixelCut above 0, near the circle), else 0.
export function cellCut(c, f, { radius }) {
  const B = BLOCK_METRES, h = HALF_DIAGONAL * B;
  const d = { x: c.x - f.eye.x, y: c.y - f.eye.y, z: c.z - f.eye.z };
  const k = f.dist / Math.max(dot(d, f.axis), 1e-3);
  const r = Math.hypot(f.eye.x + d.x * k - f.centre.x, f.eye.y + d.y * k - f.centre.y,
                       f.eye.z + d.z * k - f.centre.z);
  const nearer = dot(d, f.forward) - h < f.playerDepth - DEPTH_MARGIN * B;
  const above = c.y + 0.5 * B > f.feetY + FEET_CLEARANCE * B;
  const toward = groundAhead(c, f) - h < 0;
  return nearer && above && toward && r < radius + NEAR_SLACK * B ? 1 : 0;
}

// The cell across a box face from one of its corners (gpu/occlusion.js
// occlusionRevealTSL): v the placed corner, raw its sign in the box, n the
// face's normal. Every corner of a block's face gives the same cell.
export function cellAcross(v, raw, n) {
  const q = VOXEL_METRES / 2, B = BLOCK_METRES;
  const at = k => Math.round((v[k] - Math.sign(raw[k]) * (1 - Math.abs(n[k])) * q + n[k] * q) / B) * B;
  return { x: at('x'), y: at('y'), z: at('z') };
}

// The circle's gradient at a pixel distPx from its centre: 1 inside, 0 at
// the edge, over fadePx.
export function cellRadial(distPx, radiusPx, fadePx) {
  return clamp01((radiusPx - distPx) / Math.max(fadePx, 1e-3));
}

// What hides the player: an opaque block's solid part. Glass does not.
export function hidesAt(x, y, z) {
  if (!isSolidAt(x, y, z)) return false;
  const B = BLOCK_METRES;
  const b = getBlock(Math.round(x / B), Math.round(y / B), Math.round(z / B));
  return !(b && isGlassMaterial(b.materialId));
}

// Points over the player's quad (blocks: across the view, up from the feet),
// and the march from each toward the camera: from just past the sprite's
// plane, in steps finer than a half block.
const SAMPLES = [];
for (const a of [-0.35, 0, 0.35]) for (const h of [0.2, 0.75, 1.3, 1.85]) SAMPLES.push([a, h]);
const START = 0.4, STEP = 0.25, REACH = 40;   // metres

// Whether opaque blocks hide the whole player from the camera (eye) - every
// sample's sight line blocked. feet: the player's; right: the camera's right,
// a unit {x, z}. All in metres.
export function playerHidden(eye, feet, right, hides = hidesAt) {
  const B = BLOCK_METRES;
  return SAMPLES.every(([a, h]) => {
    const sx = feet.x + right.x * a * B, sy = feet.y + h * B, sz = feet.z + right.z * a * B;
    const dx = eye.x - sx, dy = eye.y - sy, dz = eye.z - sz;
    const len = Math.hypot(dx, dy, dz);
    const end = Math.min(len, REACH);
    for (let t = START; t < end; t += STEP) {
      const k = t / len;
      if (hides(sx + dx * k, sy + dy * k, sz + dz * k)) return true;
    }
    return false;
  });
}

// The cut set's targets this frame: every opaque block near the player's
// sight line, walked from the player toward the camera a block at a time, in
// a cube that covers the circle (with its slack) at that depth. A Map of mask
// index (origin: the mask's) -> cellCut.
const CUT_REACH = 30;   // blocks along the line
export function cutTargets(f, params, origin, opaque = opaqueAt) {
  const B = BLOCK_METRES, out = new Map(), seen = new Set();
  const end = Math.min(f.dist - B, CUT_REACH * B);
  for (let t = 0; t <= end; t += B) {
    const px = f.centre.x - f.axis.x * t, py = f.centre.y - f.axis.y * t, pz = f.centre.z - f.axis.z * t;
    const n = Math.ceil((params.radius + NEAR_SLACK * B) * (f.dist - t) / f.dist / B);
    const cx = Math.round(px / B), cy = Math.round(py / B), cz = Math.round(pz / B);
    for (let x = cx - n; x <= cx + n; x++) for (let y = cy - n; y <= cy + n; y++) for (let z = cz - n; z <= cz + n; z++) {
      const i = maskIndex(origin, x, y, z);
      if (i < 0 || seen.has(i)) continue;
      seen.add(i);
      if (!opaque(x, y, z)) continue;
      const v = cellCut({ x: x * B, y: y * B, z: z * B }, f, params);
      if (v > 0) out.set(i, v);
    }
  }
  return out;
}
