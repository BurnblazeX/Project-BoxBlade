import * as THREE from 'three/webgpu';
import { attribute, texture, uv, int, mix, vec3, vec4, uniform, select, positionLocal,
         Fn, abs, sign, round, float, vec2, positionWorld, normalWorld, floor, normalize,
         cameraViewMatrix, clamp, exp2, storage } from 'three/tsl';
import { World, BLOCK_METRES, blockBoxes, blockSpan, isSlope, blockShape, blockFacing, getVoxelKey,
         WorldModels, Y_MIN, CHUNK_HEIGHT, exposedFaceMask, occludesFaces, ALL_FACES } from './world.js';
import { MODELS, TERRAIN_LAYERS } from './models.js';
import { lightColourOf, modelInstanceData, buildModelHits, MODEL_HIT_VEC4S, MODEL_HIT_RECORD } from './bbmodel.js';
export { MODEL_HIT_RECORD };
import { VOXEL_METRES } from './boxgrid.js';
import { MATERIALS, BLOCK_TEXELS, DEFAULT_SPECULAR, materialLayer, isGlassMaterial,
         MATERIAL_LAYERS, SLOPE_SUFFIXES } from './materials.js';
import { terrainNormals } from './normals.js';

// Every block texture, by URL, so Vite bundles them - albedo, _n and _s alike.
const textureUrls = import.meta.glob('../assets/textures/*.png',
                                     { eager: true, query: '?url', import: 'default' });
export function textureUrl(name) {
  const hit = Object.keys(textureUrls).find(k => k.endsWith('/' + name + '.png'));
  return hit ? textureUrls[hit] : null;
}

export const VOXEL_SIZE = BLOCK_METRES; // single source of truth is world.js

export let worldInstancedMesh = null;
// Glass blocks, drawn by a mesh of their own: it shares the terrain's geometry
// and instance buffers (so picking, arena hiding and tinting need nothing
// new), and each material keeps only its own blocks - see terrainKeepTSL.
// null when the world holds no glass, so a map without any pays nothing.
export let glassInstancedMesh = null;
export const voxelIndexMap = new Map(); // grid key -> its instance IDs (a slope has several)
const instanceKeyByIndex = [];          // instance ID -> grid key (raycast picking)

export function keyForInstance(instanceId) {
  return instanceKeyByIndex[instanceId];
}

// --- Terrain textures, one array layer per material ---
//
// Three DataArrayTextures, stacked in materials.js order: albedo (sRGB), normal
// and specular (both LabPBR, both data). One layer per material, every layer
// BLOCK_TEXELS square, so a block picks its layer by the per-instance
// blockLayer attribute and the uv needs no remapping - its derivatives, which
// the texel lock and the normal-map frame both take, stay those of one face.
//
// They exist from the start, filled with placeholders (mid grey, flat, rough),
// and each layer is overwritten as its images arrive. So a material binds them
// once and never waits on a load, and a slow texture is a grey block for a
// moment rather than a rebuild.
//
// Rows are stored BOTTOM-UP (row 0 is v = 0), the orientation normals.js writes
// and a loaded image ends up in on the GPU.
export const terrainTextures = {
  albedo: null, normal: null, specular: null,
  // Source (sRGB) RGBA per layer, top-down, once loaded - what GI averages.
  rgba: [],
  // The same for each layer's _s, or undefined where it has none.
  specRgba: [],
  // What each BLOCK is, as a 3D texture over a fixed box of blocks: what a
  // ray that hit something reads to find its texture (buildBlockVolume).
  // { tex, origin, size } - origin and size in blocks, as vec3 uniforms.
  blocks: null,
  // Placed models' cubes, by cell, for the rays (buildBlockVolume): a
  // read-only storage buffer of vec4. { attr, node }.
  modelHits: null,
  ready: false
};

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
export const terrainKeepTSL = glass => {
  const isGlass = attribute('blockGlass', 'float').mod(float(2)).greaterThan(0.5);
  return select(glass ? isGlass : isGlass.not(), terrainShapeTSL(), vec3(0, 0, 0));
};

// --- Slopes, as drawn ---
//
// A slope is SEEN as a plane (world.js BLOCK_SHAPES; the field holds its
// steps). It is one instance of the terrain box over the slope's span, with
// the top face's downhill edge dropped by the rise, so the top face IS the
// plane; the downhill face collapses (or, on a high half slope, is left at
// half height) and the sides become its profile. Per instance, blockSlope:
// 0 for any other block, else kind * 4 + facing - kind 1 slope, 2 half
// slope, 3 high half slope; facing 0 N, 1 E, 2 S, 3 W (uphill).
const SLOPE_KIND = { slope: 1, halfSlope: 2, halfSlopeHigh: 3 };
const FACING_INDEX = { N: 0, E: 1, S: 2, W: 3 };
export function slopeCode(block) {
  if (!isSlope(block)) return 0;
  return SLOPE_KIND[blockShape(block)] * 4 + FACING_INDEX[blockFacing(block)];
}

// What a terrain shader needs of its instance's slope: on (a slope at all),
// f (uphill, x z), grad (rise per metre across), base (the foot's height, a
// fraction of the block), n (the plane's exact normal - the same arithmetic
// in every fragment, so one answer per texel).
export function slopeTSL() {
  const code = attribute('blockSlope', 'float');
  const kind = floor(code.mul(float(0.25)).add(float(0.01)));
  const facing = code.sub(kind.mul(float(4)));
  const f = select(facing.lessThan(float(0.5)), vec2(0, -1),
            select(facing.lessThan(float(1.5)), vec2(1, 0),
            select(facing.lessThan(float(2.5)), vec2(0, 1), vec2(-1, 0))));
  const grad = select(kind.lessThan(float(1.5)), float(1), float(0.5));
  const base = select(kind.greaterThan(float(2.5)), float(0.5), float(0));
  const n = normalize(vec3(f.x.mul(grad).negate(), float(1), f.y.mul(grad).negate()));
  return { on: code.greaterThan(float(0.5)), f, grad, base, n };
}

// Every terrain material's vertex position: a slope's downhill top edge
// dropped by its rise (grad blocks). The geometry's own position picks the
// vertices - top, and on the downhill side of the box's centre. After
// instancing, like terrainKeepTSL, so the drop is in world metres.
//
// A buried face (world.js exposedFaceMask, bits in blockGlass / 2) collapses
// to a point - zero area, no fragments - in every pass that draws the terrain.
export const terrainShapeTSL = () => {
  const s = slopeTSL();
  const raw = attribute('position', 'vec3');
  const low = s.on.and(raw.y.greaterThan(float(0)))
    .and(raw.x.mul(s.f.x).add(raw.z.mul(s.f.y)).lessThan(float(0)));
  const shaped = positionLocal.sub(vec3(0, select(low, s.grad.mul(float(BLOCK_METRES)), float(0)), 0));
  const mask = floor(attribute('blockGlass', 'float').mul(float(0.5)));
  const nr = attribute('normal', 'vec3');
  const d = select(abs(nr.x).greaterThan(float(0.5)), select(nr.x.greaterThan(float(0)), float(0), float(1)),
            select(abs(nr.y).greaterThan(float(0.5)), select(nr.y.greaterThan(float(0)), float(2), float(3)),
                   select(nr.z.greaterThan(float(0)), float(4), float(5))));
  const shown = floor(mask.div(exp2(d))).mod(float(2)).greaterThan(float(0.5));
  return select(shown, shaped, vec3(0, 0, 0));
};

// The instances a block draws: its boxes (one, unless it is a slope), or for
// a slope one box over its span, which terrainShapeTSL makes the plane.
export function drawnBoxes(block) {
  if (!isSlope(block)) return blockBoxes(block);
  const [lo, hi] = blockSpan(block);
  return [[0, 1, lo, hi, 0, 1]];
}

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
export function terrainSurfaceTSL(slopes = true) {
  const nA = axisNormalTSL(attribute('normal', 'vec3')).toVar();
  const pA = texelLockTSL(positionWorld, nA).toVar();
  const fA = blockFaceTSL(pA, nA);
  const Ta = select(abs(nA.x).greaterThan(float(0.5)), vec3(0, 0, nA.x.negate()),
                    select(abs(nA.y).greaterThan(float(0.5)), vec3(1, 0, 0), vec3(nA.z, 0, 0)));
  const Ba = select(abs(nA.y).greaterThan(float(0.5)), vec3(0, 0, nA.y.negate()), vec3(0, 1, 0));
  if (!slopes) return { on: null, n: nA, p: pA, block: fA.block, st: fA.st, T: Ta, B: Ba };
  const model = modelTexelTSL(nA, pA);
  // A block's face layer: its top and bottom from the instance (fillInstances:
  // modelInfo y, z - the side's own layer where it has none), else its side.
  const info = model.info;
  // Int-typed selects only (see gpu.js faceLayerTSL).
  const faceLayer = select(nA.y.greaterThan(float(0.5)), int(info.y.add(float(0.5))),
                           select(nA.y.lessThan(float(-0.5)), int(info.z.add(float(0.5))), terrainLayerTSL()));
  const sl = slopeTSL();
  const on = sl.on.and(nA.y.greaterThan(float(0.5))).toVar();
  const Bm = float(BLOCK_METRES), q = float(VOXEL_METRES);
  const r = vec2(sl.f.y.negate(), sl.f.x);                 // across, left to right looking up
  const bxz = round(pA.xz.div(Bm)).toVar();
  const hc = bxz.mul(Bm).dot(sl.f).toVar();                // the cell's centre, along the climb
  const wc = bxz.mul(Bm).dot(r).toVar();                   // and across
  // The plane at the cell's centre, from this fragment - only rounded.
  const yMid = positionWorld.y.add(sl.grad.mul(hc.sub(positionWorld.xz.dot(sl.f))));
  const by = round(yMid.div(Bm).add(float(0.5)).sub(sl.base).sub(sl.grad.mul(float(0.5)))).toVar();
  // The texel: its row along the climb, its column across.
  const rows = select(sl.grad.lessThan(float(0.75)), float(HALF_SLOPE_ROWS), float(SLOPE_ROWS)).toVar();
  const along = positionWorld.xz.dot(sl.f).sub(hc).div(Bm).add(float(0.5));
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
  const authored = info.x.greaterThan(float(0.5));
  const upper = row.greaterThanEqual(float(12));
  const slopeLayer = select(authored, int(info.x.sub(float(0.5))).add(select(upper, int(1), int(0))),
                            terrainLayerTSL());
  const rowIn = select(authored, row.sub(select(upper, float(12), float(0))),
                       floor(row.mul(float(12)).div(rows)));
  const stS = vec2(col.add(float(0.5)).div(float(12)), rowIn.add(float(0.5)).div(float(12)));
  // Frame: T across, B up the plane.
  const Ts = vec3(r.x, float(0), r.y);
  const Bs = normalize(vec3(sl.f.x, sl.grad, sl.f.y));
  return {
    on,
    slopeTexel: vec3(col, row, float(0)),
    // The one layer every map is read at (terrainTexelTSL): the plane's, a
    // model face's tile (models.js), or the block face's.
    layer: select(on, slopeLayer, select(model.on, model.layer, faceLayer)).toVar(),
    // A model's lightSource cube: unlit, its own texel (terrainEmissiveTSL).
    emissive: model.emissive,
    n: select(on, sl.n, nA).toVar(),
    p: select(on, pS, pA).toVar(),
    block: select(on, blockS, fA.block).toVar(),
    st: select(on, stS, select(model.on, model.st, fA.st)).toVar(),
    T: select(on, Ts, Ta).toVar(),
    B: select(on, Bs, Ba).toVar()
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
// instance's attributes above, or a ray hit's record (gpu.js hitSurfaceTSL).
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

// Which block a surface point belongs to, and its uv on that face - the uv
// BoxGeometry itself gives a full block's face (the table in gpu.js
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

// The instance matrix of one box a block draws (drawnBoxes, cell fractions
// [x0, x1, y0, y1, z0, z1]): the unit box scaled and moved onto it. A full
// block is the whole cell, a half block half of it, a slope its span.
// Axis-aligned scales only. Scale 0 hides it (battle).
const _pos = new THREE.Vector3(), _scl = new THREE.Vector3(), _rot = new THREE.Quaternion();
export function boxMatrix(out, key, box, visible = true) {
  const [gx, gy, gz] = key.split(',').map(Number);
  const [x0, x1, y0, y1, z0, z1] = box;
  _pos.set((gx + (x0 + x1) / 2 - 0.5) * VOXEL_SIZE, (gy + (y0 + y1) / 2 - 0.5) * VOXEL_SIZE,
           (gz + (z0 + z1) / 2 - 0.5) * VOXEL_SIZE);
  if (visible) _scl.set(x1 - x0, y1 - y0, z1 - z0); else _scl.set(0, 0, 0);
  return out.compose(_pos, _rot, _scl);
}

function makeArray(fill, colorSpace, h = BLOCK_TEXELS, layers = TERRAIN_LAYERS) {
  const n = BLOCK_TEXELS;
  const data = new Uint8Array(n * h * 4 * layers);
  for (let i = 0; i < n * h * layers; i++) data.set(fill, i * 4);
  const tex = new THREE.DataArrayTexture(data, n, h, layers);
  tex.format = THREE.RGBAFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.generateMipmaps = false;
  tex.colorSpace = colorSpace;
  tex.needsUpdate = true;
  return tex;
}

// An image's pixels, top-down, at BLOCK_TEXELS square. A texture authored at
// another size is resampled nearest and warned about - the §4 lock is one
// texel per voxel, so it is a content error, but not one worth a black block.
function imagePixels(img, name, h = BLOCK_TEXELS) {
  const n = BLOCK_TEXELS;
  if (img.width !== n || img.height !== h) {
    console.warn(`[terrain] ${name} is ${img.width}x${img.height}, not ${n}x${h} - resampled`);
  }
  const canvas = document.createElement('canvas');
  canvas.width = n;
  canvas.height = h;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(img, 0, 0, n, h);
  return ctx.getImageData(0, 0, n, h).data;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

// Top-down pixels into one layer of an array, flipped to bottom-up. rows:
// the image's height; the layer's own is the array's.
function writeLayer(tex, layer, topDown, rows = BLOCK_TEXELS) {
  const row = BLOCK_TEXELS * 4;
  const base = layer * tex.image.height * row;
  for (let y = 0; y < rows; y++) {
    tex.image.data.set(topDown.subarray(y * row, (y + 1) * row), base + (rows - 1 - y) * row);
  }
  tex.needsUpdate = true;
}

// Top-down image pixels (12 wide, `rows` tall) flipped to bottom-up.
function bottomUp(topDown, rows) {
  const row = BLOCK_TEXELS * 4, out = new Uint8ClampedArray(rows * row);
  for (let y = 0; y < rows; y++) out.set(topDown.subarray(y * row, (y + 1) * row), (rows - 1 - y) * row);
  return out;
}

// A block layer (bottom-up, 12 x 12) stretched nearest onto `rows` rows,
// bottom-up - a missing slope map's fallback: the foot at row 0.
function stretched(src, srcLayer, rows) {
  const row = BLOCK_TEXELS * 4, from = srcLayer * BLOCK_TEXELS * row;
  const out = new Uint8ClampedArray(rows * row);
  for (let y = 0; y < rows; y++) {
    const sy = Math.floor(y * BLOCK_TEXELS / rows);
    out.set(src.image.data.subarray(from + sy * row, from + (sy + 1) * row), y * row);
  }
  return out;
}

// Bottom-up rows of a slope texture into its two layers: rows 0-11 at
// `first`, the rest from row 0 of first + 1 (terrainSurfaceTSL reads it so).
function writeSplit(tex, first, data, rows) {
  const row = BLOCK_TEXELS * 4, layerBytes = BLOCK_TEXELS * row;
  for (let y = 0; y < rows; y++) {
    const to = (first + Math.floor(y / BLOCK_TEXELS)) * layerBytes + (y % BLOCK_TEXELS) * row;
    tex.image.data.set(data.subarray(y * row, (y + 1) * row), to);
  }
  tex.needsUpdate = true;
}

// One material's slope planes that are authored (materials.js
// MATERIAL_LAYERS), once its block textures are in: each map authored, or -
// for a normal map, generated from an authored albedo - else the side's own
// stretched. A plane with none of the three is not allocated; the shader
// stretches the side itself.
async function loadSlopeLayers(m, layer) {
  const t = terrainTextures;
  const rowsOf = [SLOPE_ROWS, HALF_SLOPE_ROWS];
  const firsts = MATERIAL_LAYERS.faces[layer].slope;
  for (let kind = 0; kind < 2; kind++) {
    const first = firsts[kind], rows = rowsOf[kind];
    if (first < 0) continue;
    const name = m.texture + SLOPE_SUFFIXES[kind];
    const read = async suffix => {
      const url = textureUrl(name + suffix);
      return url ? bottomUp(imagePixels(await loadImage(url), name + suffix, rows), rows) : null;
    };
    const aUrl = textureUrl(name);
    const albedoTD = aUrl ? imagePixels(await loadImage(aUrl), name, rows) : null;
    writeSplit(t.albedo, first, albedoTD ? bottomUp(albedoTD, rows) : stretched(t.albedo, layer, rows), rows);
    const normal = await read('_n');
    writeSplit(t.normal, first, normal || (albedoTD ? terrainNormals(albedoTD, BLOCK_TEXELS, rows)
                                                    : stretched(t.normal, layer, rows)), rows);
    writeSplit(t.specular, first, (await read('_s')) || stretched(t.specular, layer, rows), rows);
  }
}

// A material's top and bottom faces, when it has them (materials.js
// MATERIAL_LAYERS): albedo, then _n (else generated from it) and _s (else
// the side's), as a block's.
async function loadFaceLayers(m, layer) {
  const t = terrainTextures, n = BLOCK_TEXELS;
  const f = MATERIAL_LAYERS.faces[layer];
  for (const [suffix, dst] of [['_top', f.top], ['_bottom', f.bottom]]) {
    if (dst === layer || (suffix === '_bottom' && dst === f.top)) continue;
    const name = m.texture + suffix;
    const aUrl = textureUrl(name), nUrl = textureUrl(name + '_n'), sUrl = textureUrl(name + '_s');
    const rgba = aUrl ? imagePixels(await loadImage(aUrl), name) : null;
    if (rgba) writeLayer(t.albedo, dst, rgba);
    else writeSplit(t.albedo, dst, stretched(t.albedo, layer, n), n);
    if (nUrl) writeLayer(t.normal, dst, imagePixels(await loadImage(nUrl), name + '_n'));
    else writeSplit(t.normal, dst, rgba ? terrainNormals(rgba, n, n) : stretched(t.normal, layer, n), n);
    if (sUrl) writeLayer(t.specular, dst, imagePixels(await loadImage(sUrl), name + '_s'));
    else writeSplit(t.specular, dst, stretched(t.specular, layer, n), n);
  }
}

// What each block is, for the rays, over a FIXED volume: an edit (the
// editor) rewrites it in place, where a volume sized to the world's bounds
// changed size - and every material built on the old texture had to be
// rebuilt. 256 x 12 x 256 blocks around the origin, RGBA8, 3 MB; a block
// outside it reads as air to the rays that ask (reflection and glass hits,
// the GI). Per block:
//   a block       R layer + 1 (its side), G its top layer, B its bottom, A 0;
//   a model cell  R 0 (air to the walk through glass, which ignores models),
//                 G + 256 B its cell in modelHits, A the model's GI entry + 1
//                 (models.js giIndex);
//   air           all 0.
// modelHits: the model cubes by cell (bbmodel.js buildModelHits) - what
// gpu.js hitSurfaceTSL needs to find a hit's texel as the cube's own draw does.
export const BLOCK_VOLUME_ORIGIN = [-96, Y_MIN, -96];
export const BLOCK_VOLUME_SIZE = [256, CHUNK_HEIGHT, 256];
function buildBlockVolume() {
  const lo = BLOCK_VOLUME_ORIGIN, size = BLOCK_VOLUME_SIZE;
  let v = terrainTextures.blocks;
  if (!v) {
    const tex = new THREE.Data3DTexture(new Uint8Array(size[0] * size[1] * size[2] * 4), size[0], size[1], size[2]);
    tex.format = THREE.RGBAFormat;
    tex.type = THREE.UnsignedByteType;
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.wrapS = tex.wrapT = tex.wrapR = THREE.ClampToEdgeWrapping;
    tex.generateMipmaps = false;
    tex.unpackAlignment = 1;
    v = terrainTextures.blocks = {
      tex,
      origin: uniform(new THREE.Vector3(...lo)),
      size: uniform(new THREE.Vector3(...size))
    };
    const attr = new THREE.StorageBufferAttribute(new Float32Array(MODEL_HIT_VEC4S * 4), 4);
    terrainTextures.modelHits = {
      attr, node: storage(attr, 'vec4', MODEL_HIT_VEC4S).toReadOnly().setName('bxbModelHits')
    };
  }
  const data = v.tex.image.data;
  data.fill(0);
  const at = c => {
    const r = c.map((x, a) => x - lo[a]);
    return r.some((x, a) => x < 0 || x >= size[a]) ? -1 : ((r[2] * size[1] + r[1]) * size[0] + r[0]) * 4;
  };
  for (const [key, block] of World.entries()) {
    const o = at(key.split(',').map(Number));
    if (o < 0) continue;
    const layer = materialLayer(block.materialId), f = MATERIAL_LAYERS.faces[layer];
    data[o] = layer + 1;
    data[o + 1] = f ? f.top : layer;
    data[o + 2] = f ? f.bottom : layer;
  }
  const hits = buildModelHits(WorldModels, key => World.has(key));
  if (hits.overflow) console.warn(`[terrain] model hit table full (${MODEL_HIT_VEC4S} vec4): some model cubes reflect as air`);
  for (const [key, { header, giIndex }] of hits.index) {
    const o = at(key.split(',').map(Number));
    if (o < 0 || header > 65535) continue;
    data[o + 1] = header & 255;
    data[o + 2] = header >> 8;
    data[o + 3] = giIndex + 1;
  }
  const mh = terrainTextures.modelHits;
  mh.attr.array.set(hits.data);
  mh.attr.needsUpdate = true;
  v.tex.needsUpdate = true;
}

function initTerrainTextures() {
  if (terrainTextures.albedo) return;
  terrainTextures.albedo = makeArray([128, 128, 128, 255], THREE.SRGBColorSpace);
  terrainTextures.normal = makeArray([128, 128, 255, 255], THREE.NoColorSpace);
  terrainTextures.specular = makeArray(DEFAULT_SPECULAR, THREE.NoColorSpace);
  const loads = MATERIALS.map(async (m, layer) => {
    const albedoUrl = textureUrl(m.texture);
    if (!albedoUrl) { console.warn(`[terrain] no texture ${m.texture}.png`); return; }
    const rgba = imagePixels(await loadImage(albedoUrl), m.texture);
    terrainTextures.rgba[layer] = rgba;
    writeLayer(terrainTextures.albedo, layer, rgba);
    // Normal: authored _n, else generated from the albedo (already bottom-up).
    const nUrl = textureUrl(m.texture + '_n');
    if (nUrl) writeLayer(terrainTextures.normal, layer,
                         imagePixels(await loadImage(nUrl), m.texture + '_n'));
    else {
      const n = BLOCK_TEXELS;
      terrainTextures.normal.image.data.set(terrainNormals(rgba, n, n), layer * n * n * 4);
      terrainTextures.normal.needsUpdate = true;
    }
    // Specular: authored _s, else the placeholder already is DEFAULT_SPECULAR.
    const sUrl = textureUrl(m.texture + '_s');
    if (sUrl) {
      const spec = imagePixels(await loadImage(sUrl), m.texture + '_s');
      terrainTextures.specRgba[layer] = spec;
      writeLayer(terrainTextures.specular, layer, spec);
    }
    await loadFaceLayers(m, layer);
    await loadSlopeLayers(m, layer);
  });
  for (const model of MODELS.values()) loads.push(loadModelTiles(model));
  Promise.allSettled(loads).then(r => {
    for (const x of r) if (x.status === 'rejected') console.warn('[terrain] texture load failed', x.reason);
    terrainTextures.ready = true;
  });
}

// A model's textures (models.js), cut into 12 x 12 tiles in its layers of the
// terrain arrays - row by row, tilesX across. A map it has no companion for
// keeps the arrays' placeholder: flat for normals, DEFAULT_SPECULAR.
async function loadModelTiles(model) {
  // The emissive texture goes in the ALBEDO array, at its own layers: a
  // lightSource cube reads those in place of its albedo (modelInstanceData).
  const maps = [[model.textures.albedo, terrainTextures.albedo, model.modelLayer, 'albedo'],
                [model.textures.normal, terrainTextures.normal, model.modelLayer, 'normal'],
                [model.textures.specular, terrainTextures.specular, model.modelLayer, 'specular'],
                [model.textures.emissive, terrainTextures.albedo, model.emissiveLayer, 'emissive']];
  const n = BLOCK_TEXELS;
  const pixels = {};
  for (const [url, tex, first, kind] of maps) {
    if (!url) continue;
    const img = await loadImage(url);
    const w = model.tilesX * n, h = model.tilesY * n;
    const canvas = document.createElement('canvas');
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(img, 0, 0);
    const all = ctx.getImageData(0, 0, w, h).data;
    pixels[kind] = { all, w };
    for (let ty = 0; ty < model.tilesY; ty++) for (let tx = 0; tx < model.tilesX; tx++) {
      const tile = new Uint8ClampedArray(n * n * 4);
      for (let y = 0; y < n; y++) {
        const from = ((ty * n + y) * w + tx * n) * 4;
        tile.set(all.subarray(from, from + n * 4), y * n * 4);
      }
      writeLayer(tex, first + ty * model.tilesX + tx, tile);
    }
  }
  // What a probe landing on the model bounces (main.js resolveLayerAlbedos).
  if (pixels.albedo) model.albedoPixels = pixels.albedo.all;
  const src = pixels.emissive || pixels.albedo;
  if (src) model.lightColour = lightColourOf(model, src.all, src.w);
}



// What each instance is drawn over: { box } (a block's box, cell fractions)
// or { texels } (a model cube, world texels) - for rebuilding its matrix.
const instanceSource = [];
const TEXEL_METRES = BLOCK_METRES / BLOCK_TEXELS;
function instanceMatrix(out, idx, key, visible = true) {
  const src = instanceSource[idx];
  if (src.box) return boxMatrix(out, key, src.box, visible);
  const { min, max } = src.texels;
  _pos.set((min[0] + max[0]) / 2 * TEXEL_METRES, (min[1] + max[1]) / 2 * TEXEL_METRES,
           (min[2] + max[2]) / 2 * TEXEL_METRES);
  if (visible) _scl.set((max[0] - min[0]) / BLOCK_TEXELS, (max[1] - min[1]) / BLOCK_TEXELS,
                        (max[2] - min[2]) / BLOCK_TEXELS);
  else _scl.set(0, 0, 0);
  return out.compose(_pos, _rot, _scl);
}

// The terrain mesh is made ONCE, with room to spare, and refilled in place
// when the world changes (the editor): the texel cache's twins, the glass mesh
// and every compiled shader set hold this mesh and its buffers, and a new mesh
// would strand them all. Past its capacity an edit cannot be shown until a
// reload (rebuildWorldInstances says so).
export const INSTANCE_HEADROOM = 16384;
let terrain = null;   // { geometry, instanceData, attrs: {...}, capacity }

function instanceCount() {
  let n = 0;
  for (const block of World.values()) n += drawnBoxes(block).length;
  for (const m of WorldModels) n += m.cubes.length;
  return n;
}

export function initWorldRender(scene) {
  const needed = instanceCount();
  if (needed === 0) return;
  initTerrainTextures();
  buildBlockVolume();
  const capacity = needed + INSTANCE_HEADROOM;

  const geometry = new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
  // Every per-instance value in ONE interleaved buffer: WebGPU allows 8
  // vertex buffers a pipeline, and position, normal, uv, the instance matrix
  // and colour take five. One attribute each was 11 - no pipeline at all.
  // Per instance, 20 floats:
  //   0 blockLayer  which material it draws with - its layer in the arrays
  //   1 blockGlass  1 for glass: terrainKeepTSL sends each to its own mesh
  //   2 blockSlope  a slope's kind and facing (slopeCode), 0 otherwise
  //   4.. modelInfo, modelUV0..2 - a model cube's mapping (modelTexelTSL),
  //   zeros for a block.
  const STRIDE = 20;
  const instanceData = new THREE.InstancedInterleavedBuffer(new Float32Array(capacity * STRIDE), STRIDE);
  const field = (name, size, offset) => {
    const a = new THREE.InterleavedBufferAttribute(instanceData, size, offset);
    geometry.setAttribute(name, a);
    return a;
  };
  terrain = {
    geometry, instanceData, capacity,
    layers: field('blockLayer', 1, 0),
    glassFlags: field('blockGlass', 1, 1),
    slopes: field('blockSlope', 1, 2),
    modelAttrs: MODEL_ATTRS.map((name, k) => field(name, 4, 4 + 4 * k))
  };
  // The unshadowed path. Standard lighting, but the colour is the block's own
  // layer - a plain `map` would put layer 0 on everything.
  const material = new THREE.MeshStandardNodeMaterial({ roughness: 0.9, metalness: 0.0 });
  // At the surface's texel (terrainSurfaceTSL): the face uv from the world
  // position, not the mesh uv, and on a slope its own texture. Each node gets
  // a surface of its OWN, built inside its own Fn: the normal is built in a
  // separate pass (three's NORMAL sub-build), and a surface shared with it
  // left the colour reading variables declared there - zero, so every
  // fragment sampled texel (0, 0).
  material.colorNode = Fn(() => {
    const surf = terrainSurfaceTSL();
    const c = mix(terrainTexelTSL(terrainTextures.albedo, surf.st, surf).rgb, vec3(1, 1, 1), terrainWhite);
    return select(surf.emissive, vec3(0, 0, 0), c);
  })();
  // A lightSource cube is unlit: all of its colour is emission.
  material.emissiveNode = Fn(() => {
    const surf = terrainSurfaceTSL();
    return select(surf.emissive, terrainTexelTSL(terrainTextures.albedo, surf.st, surf).rgb, vec3(0, 0, 0));
  })();
  material.positionNode = terrainKeepTSL(false);
  // Lit by the plane's own normal: the mesh's is still the box top's.
  material.normalNode = Fn(() =>
    normalize(cameraViewMatrix.mul(vec4(terrainSurfaceTSL().n, 0)).xyz))();

  worldInstancedMesh = new THREE.InstancedMesh(geometry, material, capacity);
  worldInstancedMesh.receiveShadow = true;
  worldInstancedMesh.castShadow = true;
  worldInstancedMesh.setColorAt(0, new THREE.Color(0xffffff));   // allocates the colour buffer
  const glassCount = fillInstances();
  scene.add(worldInstancedMesh);

  glassInstancedMesh = null;
  if (glassCount > 0) {
    // The unshadowed path's glass: plainly see-through, blended - there is no
    // field to trace without shadows. With them on, main.js swaps in the
    // traced material (gpu.js glassNode).
    // Rough, as the plain terrain is (0.9): the unshadowed path has no specular
    // anywhere - marble and iron show none there either - so glass does not
    // get three's highlight from the directional light.
    const glassMat = new THREE.MeshStandardNodeMaterial({
      roughness: 0.9, metalness: 0.0, transparent: true, opacity: 0.35, depthWrite: false
    });
    glassMat.colorNode = mix(terrainSampleTSL(terrainTextures.albedo).rgb, vec3(1, 1, 1), terrainWhite);
    glassMat.positionNode = terrainKeepTSL(true);
    const g = new THREE.InstancedMesh(geometry, glassMat, capacity);
    g.instanceMatrix = worldInstancedMesh.instanceMatrix;
    g.instanceColor = worldInstancedMesh.instanceColor;
    g.count = worldInstancedMesh.count;
    g.frustumCulled = false;
    g.raycast = () => {};          // picking stays on the terrain mesh, glass included
    g.name = 'glass';
    scene.add(g);
    glassInstancedMesh = g;
  }
}

// Writes every block's and placed model's instances, from index 0, and sets
// the mesh's count. Returns how many blocks are glass.
function fillInstances() {
  const mesh = worldInstancedMesh, t = terrain;
  voxelIndexMap.clear();
  instanceKeyByIndex.length = 0;
  instanceSource.length = 0;
  const matrix = new THREE.Matrix4();
  const white = new THREE.Color(0xffffff);
  const zero4 = [0, 0, 0, 0];
  const buries = (x, y, z) => occludesFaces(World.get(getVoxelKey(x, y, z)));
  const slopeKind = { slope: 0, halfSlope: 1, halfSlopeHigh: 1 };
  let i = 0, glassCount = 0;
  // Every SOLID block renders, not just standable ones: walls, pillars and the
  // underground layers are all real geometry that the lighting work needs to
  // occlude with. Standability is a derived query in world.js, not a render gate.
  for (const [key, block] of World) {
    // Both directions are needed: key -> indices for tinting and visibility,
    // index -> key so a raycast hit (which reports only an instanceId) can be
    // resolved back to the block that was clicked.
    const ids = [];
    const layer = materialLayer(block.materialId);
    const glass = isGlassMaterial(block.materialId);
    const slope = slopeCode(block);
    const [bx, by, bz] = key.split(',').map(Number);
    for (const box of drawnBoxes(block)) {
      instanceSource[i] = { box, glass, slope: !!slope, cell: [bx, by, bz] };
      mesh.setMatrixAt(i, instanceMatrix(matrix, i, key));
      ids.push(i);
      instanceKeyByIndex[i] = key;
      mesh.setColorAt(i, white);
      t.layers.setX(i, layer);
      t.glassFlags.setX(i, (glass ? 1 : 0) + 2 * exposedFaceMask(bx, by, bz, box, !!slope, buries));
      t.slopes.setX(i, slope);
      // A block's modelInfo (w 0): its slope plane's first layer + 1 (0: the
      // side stretched), its top and bottom layers (terrainSurfaceTSL).
      const f = MATERIAL_LAYERS.faces[layer] || { top: layer, bottom: layer, slope: [-1, -1] };
      const sk = slope ? slopeKind[blockShape(block)] : null;
      t.modelAttrs[0].setXYZW(i, sk == null ? 0 : f.slope[sk] + 1, f.top, f.bottom, 0);
      for (let k = 1; k < 4; k++) t.modelAttrs[k].setXYZW(i, ...zero4);
      i++;
    }
    voxelIndexMap.set(key, ids);
    if (glass) glassCount++;
  }
  // Placed models: each cube an instance, filed under the block the model
  // stands on (or hangs from) - picking, battle hiding and tint treat it as
  // that tile's.
  for (const m of WorldModels) {
    const key = getVoxelKey(m.x, m.y, m.z);
    const ids = voxelIndexMap.get(key) || [];
    for (const cube of m.cubes) {
      instanceSource[i] = { texels: cube, model: m };
      mesh.setMatrixAt(i, instanceMatrix(matrix, i, key));
      ids.push(i);
      instanceKeyByIndex[i] = key;
      mesh.setColorAt(i, white);
      t.layers.setX(i, 0);
      t.glassFlags.setX(i, 2 * ALL_FACES);
      t.slopes.setX(i, 0);
      const d = modelInstanceData(cube, m.model);
      t.modelAttrs[0].setXYZW(i, ...d.info);
      for (let k = 0; k < 3; k++) t.modelAttrs[k + 1].setXYZW(i, ...d.uv.slice(k * 4, k * 4 + 4));
      i++;
    }
    voxelIndexMap.set(key, ids);
  }
  mesh.count = i;
  if (glassInstancedMesh) glassInstancedMesh.count = i;
  t.instanceData.needsUpdate = true;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  // Picking and frustum culling bound the instances; they have moved.
  mesh.boundingSphere = null;
  mesh.boundingBox = null;
  return glassCount;
}

// After the world changes (World, WorldModels): the instances, refilled in
// place, and the block-material volume. Returns { ok, count, glassNeeded }:
// ok false when the world has outgrown the mesh (a reload is needed to show
// it); glassNeeded when glass has appeared in a world that was built without
// any (its mesh and shaders do not exist - also a reload).
export function rebuildWorldInstances() {
  if (!worldInstancedMesh) return { ok: false, count: 0, glassNeeded: false };
  const needed = instanceCount();
  if (needed > terrain.capacity) return { ok: false, count: needed, glassNeeded: false };
  const glassCount = fillInstances();
  buildBlockVolume();
  return { ok: true, count: needed, glassNeeded: glassCount > 0 && !glassInstancedMesh };
}

// What an instance is: its block's key, and the placed model it is a cube of
// (null for a block) - for the editor's picking.
export function instanceInfo(id) {
  const src = instanceSource[id];
  return src ? { key: instanceKeyByIndex[id], model: src.model || null } : null;
}

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _scaleHidden = new THREE.Vector3(0, 0, 0);
const _scaleVisible = new THREE.Vector3(1, 1, 1);
const _rotation = new THREE.Quaternion();

export function updateVoxelVisibility(arenaMap, isBattle) {
  if (!worldInstancedMesh) return;
  
  const _matrix = new THREE.Matrix4();
  for (const [key, ids] of voxelIndexMap.entries()) {
    const visible = !(isBattle && (!arenaMap || !arenaMap.has(key)));
    for (const idx of ids) worldInstancedMesh.setMatrixAt(idx, instanceMatrix(_matrix, idx, key, visible));
  }
  worldInstancedMesh.instanceMatrix.needsUpdate = true;
  // A hidden neighbour buries nothing: the arena's edge shows its sides.
  const shown = key => !(isBattle && (!arenaMap || !arenaMap.has(key)));
  const buries = (x, y, z) => {
    const key = getVoxelKey(x, y, z);
    return shown(key) && occludesFaces(World.get(key));
  };
  instanceSource.forEach((src, i) => {
    if (!src.box) return;
    const [x, y, z] = src.cell;
    terrain.glassFlags.setX(i, (src.glass ? 1 : 0) + 2 * exposedFaceMask(x, y, z, src.box, src.slope, buries));
  });
  terrain.instanceData.needsUpdate = true;
}

export function updateVoxelTints(arenaMap, reachableMap, isBattle) {
  if (!worldInstancedMesh) return;
  const colorWhite = new THREE.Color(0xffffff);
  const colorGrey = new THREE.Color(0x333333); 
  
  for (const [key, ids] of voxelIndexMap.entries()) {
    let colour = null;
    if (!isBattle) colour = colorWhite;
    else if (arenaMap.has(key)) colour = reachableMap && reachableMap.has(key) ? colorWhite : colorGrey;
    if (colour) for (const idx of ids) worldInstancedMesh.setColorAt(idx, colour);
  }
  worldInstancedMesh.instanceColor.needsUpdate = true;
}