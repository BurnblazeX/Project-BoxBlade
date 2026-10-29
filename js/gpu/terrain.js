// --- The terrain's surface, in TSL ---
//
// Every terrain shader's view of its fragment: the block's layer, its slope,
// the texel lock, the face uv, a model cube's tile - the node functions the
// terrain materials (render/render.js) and the shading (the rest of gpu/) share. Moved
// out of render.js (2026-10-17) so the shader side never imports the meshes.
import { attribute, texture, uv, int, vec3, vec4, uniform, select, positionLocal, Fn, abs, sign, round, float, vec2, positionWorld, normalWorld, floor, normalize, clamp, exp2,
         cameraProjectionMatrix, cameraViewMatrix, modelWorldMatrix, dFdx, dFdy, max, length, mix, smoothstep } from 'three/tsl';
import { BLOCK_METRES } from '../core/world.js';
import { BLOCK_TEXELS } from '../core/materials.js';
import { VOXEL_METRES } from '../core/field.js';
import { occlusionRevealTSL, occlusionHiddenTSL, occlusionShrinkTSL } from './occlusion.js';

// 1 draws the terrain plain white (whiteworld) - a uniform, so both the plain
// and the shadowed materials follow it with no rebuild.
export const terrainWhite = uniform(0);

// The block's material layer, in the fragment: a per-instance attribute.
export const terrainLayerTSL = () => int(attribute('blockLayer', 'float').add(0.5));
// Which blocks a terrain material draws: glass or not, by the per-instance
// blockGlass flag. The others collapse to a point - zero-area triangles, no
// fragments - so the terrain shader never carries glass shading (a never-taken
// branch still costs registers: doc §0), and glass never runs the terrain's.
// Applied after instancing (NodeMaterial.setupPosition), so positionLocal is
// already the placed block.
// blockGlass also carries the instance's drawn faces: glass + 2 * mask
// (world.js exposedFaceMask), read by terrainShapeTSL.
export const terrainKeepTSL = (glass, reveal = true, hide = !glass) => {
  const isGlass = attribute('blockGlass', 'float').mod(float(2)).greaterThan(0.5);
  return select(glass ? isGlass : isGlass.not(), terrainShapeTSL(reveal, hide), vec3(0, 0, 0));
};

// Every non-glass terrain material's vertexNode: the clip position of the
// shaped vertex (positionNode's), a lifted roof's block shrunk toward its
// cell (gpu/occlusion.js occlusionShrinkTSL). Only the clip position:
// positionWorld - the shading, the texel - is still positionNode's, on the
// block's own surface. Identical in every pass, so their depths agree.
export const terrainClipTSL = () =>
  cameraProjectionMatrix.mul(cameraViewMatrix.mul(modelWorldMatrix
    .mul(vec4(occlusionShrinkTSL(positionLocal, attribute('position', 'vec3')), 1))));
// Both, on a material drawing the terrain's non-glass blocks.
export function applyTerrainVertex(material, reveal = true) {
  material.positionNode = terrainKeepTSL(false, reveal);
  material.vertexNode = terrainClipTSL();
  return material;
}

// --- Slopes, as drawn ---
//
// The instance's blockSlope code, written by render.js slopeCode (see there
// for the encoding), decoded for the shaders.
// What a terrain shader needs of its instance's slope: on (a slope at all),
// f (uphill, x z), grad (rise per metre across), base (the foot's height, a
// fraction of the block), n (the plane's exact normal - the same arithmetic
// in every fragment, so one answer per texel).
export function slopeTSL() {
  return slopeFromCodeTSL(attribute('blockSlope', 'float'));
}
// The same from a code node (a split-shading kernel reads it from the block
// volume's A channel). kind: 1 slope, 2 half slope, 3 high half slope.
export function slopeFromCodeTSL(code) {
  const kind = floor(code.mul(float(0.25)).add(float(0.01)));
  const facing = code.sub(kind.mul(float(4)));
  const f = select(facing.lessThan(float(0.5)), vec2(0, -1),
            select(facing.lessThan(float(1.5)), vec2(1, 0),
            select(facing.lessThan(float(2.5)), vec2(0, 1), vec2(-1, 0))));
  const grad = select(kind.lessThan(float(1.5)), float(1), float(0.5));
  const base = select(kind.greaterThan(float(2.5)), float(0.5), float(0));
  const n = normalize(vec3(f.x.mul(grad).negate(), float(1), f.y.mul(grad).negate()));
  return { on: code.greaterThan(float(0.5)), f, grad, base, n, kind };
}

// Every terrain material's vertex position: a slope's downhill top edge
// dropped by its rise (grad blocks). The geometry's own position picks the
// vertices - top, and on the downhill side of the box's centre. After
// instancing, like terrainKeepTSL, so the drop is in world metres.
//
// A buried face (world.js exposedFaceMask, bits in blockGlass / 2) collapses
// to a point - zero area, no fragments - in every pass that draws the terrain;
// unless the block across it is being cut away from in front of the player
// (gpu/occlusion.js occlusionRevealTSL), when it is the surface the cut opens on.
// reveal false: the buried faces stay buried (the interiors' translucent fade,
// which draws a faded block's outside only).
// hide: a block whose roof has fully faded collapses too (occlusionHiddenTSL)
// - every view pass; false for the shadow maps (castShadowPositionNode), where
// a lifted roof still casts.
export const terrainShapeTSL = (reveal = true, hide = true) => {
  const s = slopeTSL();
  const raw = attribute('position', 'vec3');
  const low = s.on.and(raw.y.greaterThan(float(0)))
    .and(raw.x.mul(s.f.x).add(raw.z.mul(s.f.y)).lessThan(float(0)));
  const shaped = positionLocal.sub(vec3(0, select(low, s.grad.mul(float(BLOCK_METRES)), float(0)), 0));
  const nr = attribute('normal', 'vec3');
  const exposed = terrainFaceExposedTSL();
  const open = reveal ? exposed.or(occlusionRevealTSL(positionLocal, raw, nr)) : exposed;
  const shown = hide ? open.and(occlusionHiddenTSL(positionLocal, raw).not()) : open;
  return select(shown, shaped, vec3(0, 0, 0));
};

// Whether this box face is one of its block's drawn faces (world.js
// exposedFaceMask: bits in blockGlass / 2), by the geometry's own normal - in
// the vertex, or the fragment, where both attributes arrive as varyings.
// Rounded first: interpolated, the whole number blockGlass is off by a hair
// (126 reads 125.99998), and floor would drop a bit - a face lost at random
// pixels along the triangles.
export const terrainFaceExposedTSL = () => {
  const mask = floor(round(attribute('blockGlass', 'float')).mul(float(0.5)));
  const nr = attribute('normal', 'vec3');
  const d = select(abs(nr.x).greaterThan(float(0.5)), select(nr.x.greaterThan(float(0)), float(0), float(1)),
            select(abs(nr.y).greaterThan(float(0.5)), select(nr.y.greaterThan(float(0)), float(2), float(3)),
                   select(nr.z.greaterThan(float(0)), float(4), float(5))));
  return floor(mask.div(exp2(d))).mod(float(2)).greaterThan(float(0.5));
};
// A drawn face that is not one of its block's: buried, and shown only because
// the block across it lifted (the reveal) - a wall's top under a lifted roof,
// the stub under a lifted wall. Its lighting would be from inside a solid
// the lighting still sees, so it is drawn its albedo, unlit
// (render/features/texelcache.js).
export const terrainRevealedTSL = () => terrainFaceExposedTSL().not();

// A terrain array sampled at this fragment's own layer - at st when given
// (the shaded path passes its texel's centre), else the mesh uv.
export const terrainSampleTSL = (tex, st = null) =>
  texture(tex, st || uv()).depth(terrainLayerTSL());

// The face normal, exactly on its axis. normalWorld is interpolated and
// renormalised, so it is off-axis by a rounding error that varies by pixel;
// the terrain is boxes, so its dominant axis IS the face.
export const axisNormalTSL = Fn(([v]) => {
  const a = abs(v).toVar();
  const nx = a.x.greaterThanEqual(a.y).and(a.x.greaterThanEqual(a.z));
  const ny = a.y.greaterThan(a.x).and(a.y.greaterThanEqual(a.z));
  return select(nx, vec3(sign(v.x), 0, 0),
                select(ny, vec3(0, sign(v.y), 0), vec3(0, 0, sign(v.z))));
});

// ONE TEXEL, ONE ANSWER - bit for bit. n must be axis-exact (axisNormalTSL):
// across the face each coordinate snaps to its texel centre, and ALONG the
// normal it snaps to the face plane itself, which is a voxel boundary (every
// face of the terrain's boxes is). It used to keep the interpolated value
// there, and that drifts by a rounding error from pixel to pixel - harmless to
// a soft shadow, but a reflection ray started a hair differently marches a
// long way and can land on a different voxel, so one texel reflected bands.
// Every input is now computed from the same integers, so every fragment of a
// texel gets the identical float.
export const texelLockTSL = Fn(([p, n]) => {
  const q = float(VOXEL_METRES);
  const a = abs(n);
  const snapped = p.div(q).floor().add(float(0.5)).mul(q);
  const plane = round(p.div(q)).mul(q);
  return snapped.mul(vec3(1, 1, 1).sub(a)).add(plane.mul(a));
});

// A slope plane's texel at a point on it - its locked position, block, uv
// in the plane's texture and frame. posW: the point (the fragment's, or the
// locked centre itself); pA: its lock on the world texel grid (only its
// across-the-climb position is read); side: the material's own layer (int);
// first1: the authored plane's first layer + 1, 0 for the side stretched.
// Shared by the terrain surface and split shading's kernels.
export function slopePlaneTSL(sl, posW, pA, side, first1) {
  const Bm = float(BLOCK_METRES), q = float(VOXEL_METRES);
  const r = vec2(sl.f.y.negate(), sl.f.x);                 // across, left to right looking up
  const bxz = round(pA.xz.div(Bm)).toVar();
  const hc = bxz.mul(Bm).dot(sl.f).toVar();                // the cell's centre, along the climb
  const wc = bxz.mul(Bm).dot(r).toVar();                   // and across
  // The plane at the cell's centre, from this fragment - only rounded.
  const yMid = posW.y.add(sl.grad.mul(hc.sub(posW.xz.dot(sl.f))));
  const by = round(yMid.div(Bm).add(float(0.5)).sub(sl.base).sub(sl.grad.mul(float(0.5)))).toVar();
  // The texel: its row along the climb, its column across.
  const rows = select(sl.grad.lessThan(float(0.75)), float(HALF_SLOPE_ROWS), float(SLOPE_ROWS)).toVar();
  const along = posW.xz.dot(sl.f).sub(hc).div(Bm).add(float(0.5));
  const row = clamp(floor(along.mul(rows)), float(0), rows.sub(float(1))).toVar();
  const col = clamp(floor(pA.xz.dot(r).sub(wc).div(q).add(float(6))), float(0), float(11)).toVar();
  // Its centre, from the indices: along the climb and across, then the plane.
  const a = row.add(float(0.5)).div(rows);
  const hS = hc.add(a.sub(float(0.5)).mul(Bm));
  const wS = wc.add(col.add(float(0.5)).sub(float(6)).mul(q));
  const xz = sl.f.mul(hS).add(r.mul(wS));
  const y = by.sub(float(0.5)).add(sl.base).mul(Bm).add(sl.grad.mul(a).mul(Bm));
  const pS = vec3(xz.x, y, xz.y);
  const blockS = vec3(bxz.x, by, bxz.y);
  // Its texel in the plane's texture: authored (modelInfo.x = its first layer
  // + 1), rows 0-11 in that layer and the rest in the next; else the side's
  // own texture stretched nearest along the climb (row * 12 / rows).
  const authored = first1.greaterThan(float(0.5));
  const upper = row.greaterThanEqual(float(12));
  const slopeLayer = select(authored, int(first1.sub(float(0.5))).add(select(upper, int(1), int(0))),
                            side);
  const rowIn = select(authored, row.sub(select(upper, float(12), float(0))),
                       floor(row.mul(float(12)).div(rows)));
  const stS = vec2(col.add(float(0.5)).div(float(12)), rowIn.add(float(0.5)).div(float(12)));
  // Frame: T across, B up the plane.
  const Ts = vec3(r.x, float(0), r.y);
  const Bs = normalize(vec3(sl.f.x, sl.grad, sl.f.y));
  return { n: sl.n, p: vec3(xz.x, y, xz.y), block: vec3(bxz.x, by, bxz.y), st: stS,
           layer: slopeLayer, slopeTexel: vec3(col, row, float(0)), T: Ts, B: Bs };
}

// THE TERRAIN FRAGMENT'S SURFACE: normal, locked position, block, face uv,
// the tangent frame for its normal map, and on a slope its own texel - every
// input to its shading.
//
// A box face: its axis normal, the texel lock, the face uv, exactly as they
// always were (the selects below pass those values through untouched).
//
// A slope's top face is its PLANE (terrainShapeTSL), with a texture of its
// own (terrainTextures.slope*): 12 texels across and, along the climb, 17 on
// a slope (2.12 m) or 13 on a half slope (1.68 m) - square texels of about
// 12.5 cm on the plane. Row 0 is the foot, u runs left to right looking
// uphill. The lock snaps to the centre of THAT texel - across, on the world's
// texel grid as any face; along, to the plane's own rows - and puts y on the
// plane there, from the block and the texel's indices alone: every fragment
// of a texel gets the identical float, so one answer (doc §0). The block is
// found from the plane's height at the cell's centre, never near a boundary.
// The normal is the plane's own (slopeTSL), not an axis.
//
// slopes: false where the mesh carries no blockSlope attribute.
export const SLOPE_ROWS = 17, HALF_SLOPE_ROWS = 13;
//
// The mesh's own normal ATTRIBUTE, not normalWorld: the plain material sets
// its normalNode from this surface, and normalWorld is derived from that node
// - reading it fed the normal back into itself (streaked, flickering terrain).
// The terrain's instances are only moved and scaled along their axes, never
// turned, so the attribute already points along the face in world space; the
// axis snap makes it identical to what normalWorld gave where that was safe.
//
// An edge fragment's interpolated position can land a hair past its own box,
// and the lock then takes the NEIGHBOUR's texel. A block's neighbour shows
// that texel itself, so nothing shows; a model face's map runs on past its uv
// rectangle and reads a pixel of something else - a hairline seam along every
// cube's edge. So the position is first nudged toward the box's centre, across
// the face: instances are only moved and scaled (BoxGeometry is centred), so
// the raw vertex position's sign points outward. A hundredth of a texel.
const EDGE_NUDGE = VOXEL_METRES / 100;
export function terrainSurfaceTSL(slopes = true) {
  const nA = axisNormalTSL(attribute('normal', 'vec3')).toVar();
  const inward = sign(attribute('position', 'vec3')).mul(vec3(1, 1, 1).sub(abs(nA))).mul(float(EDGE_NUDGE));
  const pA = texelLockTSL(positionWorld.sub(inward), nA).toVar();
  const fA = blockFaceTSL(pA, nA);
  const Ta = select(abs(nA.x).greaterThan(float(0.5)), vec3(0, 0, nA.x.negate()),
                    select(abs(nA.y).greaterThan(float(0.5)), vec3(1, 0, 0), vec3(nA.z, 0, 0)));
  const Ba = select(abs(nA.y).greaterThan(float(0.5)), vec3(0, 0, nA.y.negate()), vec3(0, 1, 0));
  if (!slopes) return { on: null, n: nA, p: pA, block: fA.block, st: fA.st, T: Ta, B: Ba };
  const model = modelTexelTSL(nA, pA);
  // A block's face layer: its top and bottom from the instance (fillInstances:
  // modelInfo y, z - the side's own layer where it has none), else its side.
  const info = model.info;
  // Int-typed selects only (see volume.js faceLayerTSL).
  const faceLayer = select(nA.y.greaterThan(float(0.5)), int(info.y.add(float(0.5))),
                           select(nA.y.lessThan(float(-0.5)), int(info.z.add(float(0.5))), terrainLayerTSL()));
  const sl = slopeTSL();
  const on = sl.on.and(nA.y.greaterThan(float(0.5))).toVar();
  const sp = slopePlaneTSL(sl, positionWorld, pA, terrainLayerTSL(), info.x);
  return {
    on,
    slopeTexel: sp.slopeTexel,
    // The one layer every map is read at (terrainTexelTSL): the plane's, a
    // model face's tile (models.js), or the block face's.
    layer: select(on, sp.layer, select(model.on, model.layer, faceLayer)).toVar(),
    // A model's lightSource cube: unlit, its own texel (terrainEmissiveTSL).
    emissive: model.emissive,
    n: select(on, sl.n, nA).toVar(),
    p: select(on, sp.p, pA).toVar(),
    block: select(on, sp.block, fA.block).toVar(),
    st: select(on, sp.st, select(model.on, model.st, fA.st)).toVar(),
    T: select(on, sp.T, Ta).toVar(),
    B: select(on, sp.B, Ba).toVar()
  };
}

// --- Models, as drawn ---
//
// A placed model's cube (bbmodel.js placeModel) is one more instance of the
// terrain box, over its world box. Its faces lie on texel boundaries, so the
// texel lock holds as for any block face; what differs is which texture pixel
// a texel shows. Per instance:
//   modelInfo  (mask, first layer, tiles across, 1; 2 for a lightSource cube,
//              drawn unlit) - 0 for anything else.
//              mask: per face d (+x, -x, +y, -y, +z, -z), bit 2d set when u
//              runs backwards, 2d + 1 when v does; bits 12 and 13 when the
//              top / bottom face has its axes swapped (a yaw turn).
//   modelUV0..2 each face's pixel origins (oU, oV), two faces a vec4.
// pixel = (oU + dU * w[u], oV + dV * w[v]) for the world texel w - the same
// arithmetic as bbmodel.js facePixel, which the tests hold to the uv.
export const MODEL_ATTRS = ['modelInfo', 'modelUV0', 'modelUV1', 'modelUV2'];

export function modelTexelTSL(nA, pA) {
  // Rounded: every value is an integer, and an attribute reaches the fragment
  // interpolated - equal at the three vertices, but not exactly equal after,
  // so 6 could arrive as 5.9999998 and floor to the wrong texel, tile or bit,
  // differently from pixel to pixel (black, flickering texels on the models).
  const info = round(attribute('modelInfo', 'vec4')).toVar();
  const uv0 = round(attribute('modelUV0', 'vec4')).toVar();
  const uv1 = round(attribute('modelUV1', 'vec4')).toVar();
  const uv2 = round(attribute('modelUV2', 'vec4')).toVar();
  return { ...modelTexelCoreTSL(nA, pA, info, uv0, uv1, uv2), info };
}

// The texel arithmetic itself, from a cube's info and uv vectors - the
// instance's attributes above, or a ray hit's record (volume.js hitSurfaceTSL).
// pA: a point on the face (any point within the texel), nA axis-exact.
export function modelTexelCoreTSL(nA, pA, info, uv0, uv1, uv2) {
  const q = float(VOXEL_METRES);
  const t = floor(pA.sub(nA.mul(q.mul(float(0.5)))).div(q)).toVar();
  const ax = abs(nA.x).greaterThan(float(0.5)), ay = abs(nA.y).greaterThan(float(0.5));
  const az = abs(nA.z).greaterThan(float(0.5));
  const fi = select(ax, select(nA.x.greaterThan(float(0)), float(0), float(1)),
             select(ay, select(nA.y.greaterThan(float(0)), float(2), float(3)),
                        select(nA.z.greaterThan(float(0)), float(4), float(5)))).toVar();
  const o = select(fi.lessThan(float(0.5)), uv0.xy, select(fi.lessThan(float(1.5)), uv0.zw,
            select(fi.lessThan(float(2.5)), uv1.xy, select(fi.lessThan(float(3.5)), uv1.zw,
            select(fi.lessThan(float(4.5)), uv2.xy, uv2.zw)))));
  const bit = k => floor(info.x.div(exp2(k))).mod(float(2));
  const dU = float(1).sub(bit(fi.mul(float(2))).mul(float(2)));
  const dV = float(1).sub(bit(fi.mul(float(2)).add(float(1))).mul(float(2)));
  const swap = ay.and(bit(float(10).add(fi)).greaterThan(float(0.5)));
  // The face's own axes: +-x (z, y), +-y (x, z), +-z (x, y) - swapped when set.
  const ca = select(ax, t.z, t.x), cb = select(ay, t.z, t.y);
  const px = o.x.add(dU.mul(select(swap, cb, ca)));
  const py = o.y.add(dV.mul(select(swap, ca, cb)));
  const tx = floor(px.div(float(12))), ty = floor(py.div(float(12)));
  const layer = int(info.y.add(ty.mul(info.z)).add(tx).add(float(0.5)));
  // Tiles are stored bottom-up (writeLayer): image row iy is row 11 - iy.
  const st = vec2(px.sub(tx.mul(float(12))).add(float(0.5)).div(float(12)),
                  float(11.5).sub(py.sub(ty.mul(float(12)))).div(float(12)));
  return { on: info.w.greaterThan(float(0.5)), emissive: info.w.greaterThan(float(1.5)), layer, st };
}

// The shading normal from a surface's frame and a LabPBR tangent-space
// normal - boxBumpedNormalTSL's arithmetic, with the frame handed in.
export const surfaceBumpTSL = (surf, t) =>
  normalize(surf.T.mul(t.x).add(surf.B.mul(t.y)).add(surf.n.mul(t.z)));

// A terrain texture read at a surface's texel: ONE read, at the layer and uv
// the surface chose - a block face's (side, top or bottom), a slope plane's,
// a model tile's. tex: one of terrainTextures' albedo, normal or specular.
// Without a surface (or one built without slopes), the block's own layer.
export function terrainTexelTSL(tex, st, surf = null) {
  return surf && surf.layer ? texture(tex, st).depth(int(surf.layer)) : terrainSampleTSL(tex, st);
}

// The albedo at a surface, for the fragment shaders (basic mode, the plain
// material): its nearest texel while a texel covers a pixel or more, so the
// pixel art stays crisp up close; past that, where the 12-texel faces
// shimmer, filtered - the same array mipmapped, with 16x anisotropic
// filtering (render.js terrainTextures.albedoFiltered). The gradients come
// from the world position, not the face uv: st jumps from 1 to 0 at every
// block's edge, and its own derivatives there would drop a seam of the
// smallest mip along each one. Not in the compute kernels: no derivatives
// there. filtered null: terrainTexelTSL.
export function terrainAlbedoTSL(tex, filtered, surf) {
  if (!filtered) return terrainTexelTSL(tex, surf.st, surf);
  const q = surf.p.mul(float(BLOCK_TEXELS / BLOCK_METRES));   // texels
  const dx = dFdx(q), dy = dFdy(q);
  const n = abs(surf.n);
  // The two axes across the face, as the face uv runs (blockFaceTSL).
  const across = d => select(n.x.greaterThan(float(0.5)), vec2(d.z, d.y),
                             select(n.y.greaterThan(float(0.5)), vec2(d.x, d.z), vec2(d.x, d.y)));
  const gx = across(dx).toVar(), gy = across(dy).toVar();
  const texels = max(length(gx), length(gy));   // texels a pixel, the longer way
  const layer = int(surf.layer);
  const near = texture(tex, surf.st).depth(layer);
  const far = texture(filtered, surf.st).grad(gx.div(float(BLOCK_TEXELS)), gy.div(float(BLOCK_TEXELS))).depth(layer);
  return mix(near, far, smoothstep(float(0.75), float(1.5), texels));
}

// Advanced mode's anisotropic filtering, after the fact. Its kernels shade
// each texel - albedo included - and every pixel takes its own texel's
// colour, so far off, where a texel is smaller than a pixel, the 12-texel
// pattern is point-sampled and shimmers. This is what to multiply that colour
// by: the filtered albedo over the texel's own (both with whiteworld, as the
// kernels apply it), so the pattern is mipmapped and anisotropic while the
// light stays the texel's. Exactly 1 up close, where terrainAlbedoTSL is the
// texel's own. The +EPS keeps a near-black texel from blowing its light up.
const FAR_ALBEDO_EPS = 0.02;
export function terrainFarAlbedoRatioTSL(tex, filtered, surf) {
  const white = c => mix(c, vec3(1, 1, 1), terrainWhite);
  const own = white(terrainTexelTSL(tex, surf.st, surf).rgb);
  const far = white(terrainAlbedoTSL(tex, filtered, surf).rgb);
  return far.add(float(FAR_ALBEDO_EPS)).div(own.add(float(FAR_ALBEDO_EPS)));
}

// Which block a surface point belongs to, and its uv on that face - the uv
// BoxGeometry itself gives a full block's face (the table in normals.js
// boxBumpedNormalTSL), so a texture read here matches the mesh's own. From the
// world position, not the mesh uv: a half block's side shows the half of the
// texture it covers, where the mesh uv would squash the whole texture onto it.
// p on the face, n axis-exact. { block: the block's grid coordinate, st: the
// face uv }.
export function blockFaceTSL(p, n) {
  const block = round(p.sub(n.mul(float(VOXEL_METRES * 0.5))).div(float(BLOCK_METRES))).toVar();
  const l = p.div(float(BLOCK_METRES)).sub(block).toVar();
  const h = float(0.5);
  const uvX = vec2(select(n.x.greaterThan(float(0)), l.z.negate(), l.z).add(h), l.y.add(h));
  const uvY = vec2(l.x.add(h), select(n.y.greaterThan(float(0)), l.z.negate(), l.z).add(h));
  const uvZ = vec2(select(n.z.greaterThan(float(0)), l.x, l.x.negate()).add(h), l.y.add(h));
  const st = select(abs(n.x).greaterThan(float(0.5)), uvX,
                    select(abs(n.y).greaterThan(float(0.5)), uvY, uvZ));
  return { block, st };
}
