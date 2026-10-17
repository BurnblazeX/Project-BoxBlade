// --- Occlusion cut-out and interiors, in TSL (core/occlusion.js, core/interior.js) ---
//
// Whole blocks decide, pixels fade. Every decision is made on the CPU and
// eased per block (render/features/occlusion.js), into a mask texture of
// cells around the player - two bytes a cell:
//
//   r  roof       the roof (core/interior.js): whether it fades - by a
//                 fall-off with distance on screen from the player, 100% near
//                 to occlusion.min far, so the roof over them is gone
//   g  cut        the cut-out (core/occlusion.js cellCut): whether it MAY be
//                 cut - how much is per pixel (pixelTSL), by smooth ramps of
//                 its depth and height and the circle's gradient
//
// A block's own value only ever says whether; how much is per pixel, so as
// the player moves the change slides smoothly across the blocks.
//
//   the reveal    the vertex shape (gpu/terrain.js terrainShapeTSL) shows a
//                 buried face when the block across it is in either set: the
//                 opening shows the real floor and the real block behind,
//                 never a solid's empty inside.
//   the cut       per pixel: a block's cut times a smooth radial gradient on
//                 screen around the player - fully cut inside, nothing past
//                 the circle's edge.
//   translucency  a pixel with any fade or cut leaves the opaque passes, and
//                 the twin of the terrain draws it again blended
//                 (interiorFadeTSL) at (1 - fade)(1 - cut) - with the set's
//                 own full shading (render/features/texelcache.js fadeMat), so
//                 a block looks the same solid or fading and never pops.
//
// The mask is every opaque terrain material's maskNode - each of the texel
// cache's passes, the depth prepass, split shading's G-buffer and read, the
// plain material - and the reveal is in all their vertex code, so every pass
// agrees on the holes. Glass is never cut: it hides nothing.
import * as THREE from 'three/webgpu';
import { uniform, float, int, vec3, abs, clamp, max, length, round, bool, attribute, mix, dot,
         smoothstep, texture3D, positionWorld, screenCoordinate, screenSize } from 'three/tsl';
import { BLOCK_METRES } from '../core/world.js';
import { VOXEL_METRES } from '../core/boxgrid.js';
import { INTERIOR_DIMS, DEFAULT_INTERIOR } from '../core/interior.js';
import { DEPTH_MARGIN, DEPTH_RAMP, FEET_CLEARANCE, HEIGHT_RAMP, AHEAD_RAMP } from '../core/occlusion.js';

// The cell mask: two bytes a cell, nearest, read at cell centres. Its
// outermost layer is never written, so a read outside clamps to nothing.
const [IW, IH, ID] = INTERIOR_DIMS;
const maskTex = new THREE.Data3DTexture(new Uint8Array(IW * IH * ID * 2), IW, IH, ID);
maskTex.format = THREE.RGFormat;
maskTex.type = THREE.UnsignedByteType;
maskTex.minFilter = maskTex.magFilter = THREE.NearestFilter;
maskTex.generateMipmaps = false;
maskTex.unpackAlignment = 1;
maskTex.needsUpdate = true;
export const cellMask = {
  tex: maskTex,
  data: maskTex.image.data,                        // r, g per cell, x fastest then y then z
  origin: uniform(new THREE.Vector3())             // the mask's cell 0, in cells
};

// The frame: the camera, the player, and both gradients on screen.
export const occlusion = {
  eye: uniform(new THREE.Vector3()),
  forward: uniform(new THREE.Vector3(0, 0, -1)),   // the camera's view direction
  playerDepth: uniform(1),                         // the player's centre along it (m)
  feetY: uniform(0),
  centreXZ: uniform(new THREE.Vector2()),          // the player's centre on the ground (m)
  aheadXZ: uniform(new THREE.Vector2(0, -1)),      // the camera's view along the ground, unit
  centrePx: uniform(new THREE.Vector2()),          // the player's centre, full-res pixels from the top left
  radiusPx: uniform(100),                          // the cut's circle ...
  fadePx: uniform(30),                             // ... and its gradient's width, inside the edge
  roofNearPx: uniform(100),                        // the roof clear within ...
  roofFarPx: uniform(400),                         // ... down to min from
  roofMin: uniform(DEFAULT_INTERIOR.min),
  fullSize: uniform(new THREE.Vector2(1, 1))       // the full-res drawing buffer
};

const B = BLOCK_METRES;

// A cell's centre from a point inside it, and the fragment's own block: the
// cell just inside its face.
const cellOf = p => round(p.div(float(B))).mul(float(B));
const fragmentCell = () => cellOf(positionWorld.sub(round(attribute('normal', 'vec3')).mul(float(VOXEL_METRES * 0.5))));

// A cell's two mask values (centre c, metres): roof, cut.
function maskAtTSL(c) {
  const u = cellMask;
  const uvw = round(c.div(float(B))).sub(u.origin).add(float(0.5)).div(vec3(IW, IH, ID));
  const m = texture3D(u.tex, uvw).level(int(0));
  return { interior: m.r, cut: m.g };
}

// The vertex half: whether the block across this vertex's face is in either
// set. v: the placed vertex; raw, n: the box geometry's own position and
// normal. A point just across the face, pulled in from the corner: every
// vertex of a block's face lands in the same cell.
export function occlusionRevealTSL(v, raw, n) {
  const q = float(VOXEL_METRES * 0.5);
  const m = maskAtTSL(cellOf(v.sub(raw.sign().mul(abs(n).oneMinus()).mul(q)).add(n.mul(q))));
  return m.interior.greaterThan(float(0)).or(m.cut.greaterThan(float(0)));
}

// This pixel of the fragment's block (at the full-res pixel a low-res pass's
// lies on): its roof fade by its distance on screen from the player, and its
// cut - core/occlusion.js pixelCut's three ramps times the circle's gradient.
function pixelTSL() {
  const u = occlusion, Bm = float(B);
  const m = maskAtTSL(fragmentCell());
  const px = screenCoordinate.xy.mul(u.fullSize.div(screenSize));
  const dist = length(px.sub(u.centrePx));
  const radial = clamp(u.radiusPx.sub(dist).div(max(u.fadePx, float(1e-3))), 0, 1);
  const front = clamp(u.playerDepth.sub(float(DEPTH_MARGIN).mul(Bm))
    .sub(dot(positionWorld.sub(u.eye), u.forward)).div(float(DEPTH_RAMP).mul(Bm)), 0, 1);
  const above = clamp(positionWorld.y.sub(u.feetY).sub(float(FEET_CLEARANCE).mul(Bm))
    .div(float(HEIGHT_RAMP).mul(Bm)), 0, 1);
  const ahead = clamp(dot(positionWorld.xz.sub(u.centreXZ), u.aheadXZ).negate().div(float(AHEAD_RAMP).mul(Bm)), 0, 1);
  const roof = mix(float(1), u.roofMin, smoothstep(u.roofNearPx, max(u.roofFarPx, u.roofNearPx.add(float(1))), dist));
  return { interior: m.interior.mul(roof), cut: m.cut.mul(radial).mul(front).mul(above).mul(ahead) };
}

// The opaque passes' mask: true keeps the fragment - neither faded nor cut.
// skipGlass: the mesh draws glass too (the depth prepass), and its glass
// instances are always kept.
export function occlusionKeepTSL(skipGlass = false) {
  const p = pixelTSL();
  const keep = p.cut.lessThanEqual(float(0)).and(p.interior.lessThanEqual(float(0)));
  return skipGlass ? keep.or(attribute('blockGlass', 'float').mod(float(2)).greaterThan(0.5)) : keep;
}

// Shadow maps (the basic mode's sun) draw the same materials: the cut is the
// view's, so a cut wall still casts its shadow. three reads maskShadowNode
// there before maskNode.
export function applyOcclusionMask(material, skipGlass = false) {
  material.maskNode = occlusionKeepTSL(skipGlass);
  material.maskShadowNode = bool(true);
  return material;
}

// The translucent twin: what the opaque passes left out, drawn again blended
// at (1 - fade)(1 - cut). mask: those pixels with anything to draw.
export function interiorFadeTSL() {
  const p = pixelTSL();
  const opacity = float(1).sub(p.interior).mul(float(1).sub(p.cut));
  const left = p.interior.greaterThan(float(0)).or(p.cut.greaterThan(float(0)));
  return { mask: left.and(opacity.greaterThan(float(1 / 255))), opacity };
}
