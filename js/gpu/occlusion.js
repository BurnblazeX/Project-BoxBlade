// --- Interiors, in TSL (core/interior.js) ---
//
// Whole blocks decide. Every decision is made on the CPU and eased per block
// (render/features/occlusion.js), into a mask texture of cells around the
// player - one byte a cell:
//
//   r  lifted     the roof and the walls lifted with it (core/interior.js):
//                 how far it has gone, 0..1. Its blocks shrink away into their
//                 cells' centres meanwhile (occlusionShrinkTSL) and are then
//                 not drawn at all (occlusionHiddenTSL) - solid throughout,
//                 never translucent
//
//   the shrink    a lifted block's geometry shrinks toward its cell's centre
//                 by its r (gpu/terrain.js terrainClipTSL, every terrain
//                 material's vertexNode): only the clip position moves -
//                 positionWorld, and so all its shading and its texel, stay on
//                 the block's own surface, lit as it was, never from inside a
//                 solid. Solid, so the texel cache shades it as any block: a
//                 translucent roof was two layers a pixel, one shaded per pixel.
//   the hidden    the vertex shape collapses a block whose own cell is fully
//                 lifted - no fragments in any view pass, so a lifted roof
//                 costs nothing. Shadow maps keep it (castShadowPositionNode;
//                 they never read vertexNode, so never see the shrink either).
//   the outside   under a roof, what is outside the building (core/interior.js
//                 inside, a 2D mask of columns) darkens, by outsideShadeTSL in
//                 the view's colour materials only - never the lighting, so
//                 light from outside still reaches in.
//   the reveal    the vertex shape (gpu/terrain.js terrainShapeTSL) shows a
//                 buried face when the block across it is lifted: the opening
//                 shows the real wall top and the real floor, never a solid's
//                 empty inside - drawn its albedo, unlit (terrainRevealedTSL).
//
// A model's cube in a lifted cell (it can span cells, so cannot shrink into
// one) leaves the opaque passes outright, by their maskNode - each of the
// texel cache's passes, the depth prepass, split shading's G-buffer and read,
// the plain material. The reveal is in all their vertex code, so every pass
// agrees on the holes.
import * as THREE from 'three/webgpu';
import { uniform, float, int, vec2, vec3, abs, round, bool, attribute, dot, select,
         texture, texture3D, positionWorld } from 'three/tsl';
import { BLOCK_METRES } from '../core/world.js';
import { VOXEL_METRES } from '../core/boxgrid.js';
import { INTERIOR_DIMS } from '../core/interior.js';

// The cell mask: one byte a cell, nearest, read at cell centres. Its
// outermost layer is never written, so a read outside clamps to nothing.
const [IW, IH, ID] = INTERIOR_DIMS;
const maskTex = new THREE.Data3DTexture(new Uint8Array(IW * IH * ID), IW, IH, ID);
maskTex.format = THREE.RedFormat;
maskTex.type = THREE.UnsignedByteType;
maskTex.minFilter = maskTex.magFilter = THREE.NearestFilter;
maskTex.generateMipmaps = false;
maskTex.unpackAlignment = 1;
maskTex.needsUpdate = true;
export const cellMask = {
  tex: maskTex,
  data: maskTex.image.data,                        // r per cell, x fastest then y then z
  origin: uniform(new THREE.Vector3())             // the mask's cell 0, in cells
};

// The inside: one byte a column of the mask's x z, 255 inside the building,
// filtered so the dark comes in over a column at the walls.
const insideTex = new THREE.DataTexture(new Uint8Array(IW * ID), IW, ID, THREE.RedFormat, THREE.UnsignedByteType);
insideTex.minFilter = insideTex.magFilter = THREE.LinearFilter;
insideTex.generateMipmaps = false;
insideTex.unpackAlignment = 1;
insideTex.needsUpdate = true;
export const insideMask = {
  tex: insideTex,
  data: insideTex.image.data,                      // x fastest, then z
  origin: uniform(new THREE.Vector2())             // its column 0 (x, z), in cells
};

export const occlusion = {
  outsideDark: uniform(0)                          // how dark outside the building is now, 0..1
};

const B = BLOCK_METRES;

// A cell's centre from a point inside it, and the fragment's own block: the
// cell just inside its face.
const cellOf = p => round(p.div(float(B))).mul(float(B));
const fragmentCell = () => cellOf(positionWorld.sub(round(attribute('normal', 'vec3')).mul(float(VOXEL_METRES * 0.5))));

// A cell's lift, 0..1 (centre c, metres).
function liftAtTSL(c) {
  const u = cellMask;
  const uvw = round(c.div(float(B))).sub(u.origin).add(float(0.5)).div(vec3(IW, IH, ID));
  return texture3D(u.tex, uvw).level(int(0)).r;
}

// Whether this vertex's own block is a whole block (not a model's cube, which
// can span cells) whose cell is fully lifted: then it is not drawn.
// v, raw: as occlusionRevealTSL. The corner pulled in on every axis.
export function occlusionHiddenTSL(v, raw) {
  const q = float(VOXEL_METRES * 0.5);
  return liftAtTSL(cellOf(v.sub(raw.sign().mul(q)))).greaterThan(float(254.5 / 255))
    .and(attribute('modelInfo', 'vec4').w.lessThan(float(0.5)));
}

// The vertex's place as its block shrinks away (see the top): toward its own
// cell's centre by the cell's lift - a whole block only. v, raw: as
// occlusionHiddenTSL; v is the shaped vertex. A collapsed vertex (a buried
// face, at the origin) stays collapsed.
export function occlusionShrinkTSL(v, raw) {
  const q = float(VOXEL_METRES * 0.5);
  const c = cellOf(v.sub(raw.sign().mul(q)));
  const k = select(attribute('modelInfo', 'vec4').w.lessThan(float(0.5)), liftAtTSL(c), float(0));
  const shrunk = c.add(v.sub(c).mul(float(1).sub(k)));
  return select(dot(v, v).greaterThan(float(0)), shrunk, v);
}

// The vertex half: whether the block across this vertex's face is lifted.
// v: the placed vertex; raw, n: the box geometry's own position and normal.
// A point just across the face, pulled in from the corner: every vertex of a
// block's face lands in the same cell.
export function occlusionRevealTSL(v, raw, n) {
  const q = float(VOXEL_METRES * 0.5);
  return liftAtTSL(cellOf(v.sub(raw.sign().mul(abs(n).oneMinus()).mul(q)).add(n.mul(q)))).greaterThan(float(0));
}

// The opaque passes' mask: true keeps the fragment - all but a model's cube
// in a lifted cell. skipGlass: the mesh draws glass too (the depth prepass),
// and its glass instances are always kept.
export function occlusionKeepTSL(skipGlass = false) {
  const model = attribute('modelInfo', 'vec4').w.greaterThan(float(0.5));
  const keep = liftAtTSL(fragmentCell()).lessThanEqual(float(0)).or(model.not());
  return skipGlass ? keep.or(round(attribute('blockGlass', 'float')).mod(float(2)).greaterThan(0.5)) : keep;   // rounded: interpolated
}

// Shadow maps (the basic mode's sun) draw the same materials: the lift is the
// view's, so a lifted roof still casts its shadow. three reads maskShadowNode
// there before maskNode.
export function applyOcclusionMask(material, skipGlass = false) {
  material.maskNode = occlusionKeepTSL(skipGlass);
  material.maskShadowNode = bool(true);
  return material;
}

// The view's darkening outside the building: a factor for a colour material's
// output (gpu/texel.js createShadowMaterial, render.js plainTerrainMaterial,
// gpu/sprite.js) - 1 inside, 1 - outsideDark outside. Never in a pass that
// feeds the lighting (the texel cache's low-res pass, the atlas pass).
export function outsideShadeTSL() {
  return float(1).sub(occlusion.outsideDark.mul(float(1).sub(insideAtTSL(positionWorld.xz))));
}

// How far a ground position (xz, metres) is inside the building, 0..1,
// filtered over a column at the walls. An explicit level: safe in a branch.
export function insideAtTSL(xz) {
  const m = insideMask;
  const uv = xz.div(float(B)).sub(m.origin).add(float(0.5)).div(vec2(IW, ID));
  return texture(m.tex, uv).level(float(0)).r;
}
