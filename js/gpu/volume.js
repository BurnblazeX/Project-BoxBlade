import * as THREE from 'three/webgpu';
import { If, Loop, Break, float, vec3, uniform, texture3D, floor, max, texture, abs, vec4, min, clamp, length, int, select, mix, step, sign, fract, round } from 'three/tsl';
import { GRID_DIM, VOXEL_METRES } from '../core/boxgrid.js';
import { BLOCK_METRES } from '../core/world.js';
import { terrainWhite, blockFaceTSL as boxFaceTSL, texelLockTSL, modelTexelCoreTSL } from './terrain.js';
import { MODEL_HIT_RECORD } from '../core/bbmodel.js';
import { MATERIALS } from '../core/materials.js';
import { boxBumpedNormalTSL, labNormalTSL } from './normals.js';

// --- What a ray hit: the block volume, its shapes and the surface at a hit ---

// --- What a ray may land on in battle ---
//
// Battle hides everything outside the arena, but the field still holds the
// whole world - so a reflection, or the view through glass, showed terrain
// that is not on screen. When on, a ray landing outside this box counts as a
// miss (the sky), and the walk through glass reads blocks outside it as air.
// A bounds test per hit, rather than re-baking the field for the arena, which
// would change every shadow and the GI with it. World metres; setHitBounds.
export const hitBounds = {
  min: uniform(new THREE.Vector3(-1e9, -1e9, -1e9)),
  max: uniform(new THREE.Vector3(1e9, 1e9, 1e9))
};

export function setHitBounds(box = null) {
  if (!box) {
    hitBounds.min.value.set(-1e9, -1e9, -1e9);
    hitBounds.max.value.set(1e9, 1e9, 1e9);
    return;
  }
  hitBounds.min.value.set(box.min.x, box.min.y, box.min.z);
  hitBounds.max.value.set(box.max.x, box.max.y, box.max.z);
}

export const inHitBoundsTSL = p => p.x.greaterThanEqual(hitBounds.min.x).and(p.x.lessThanEqual(hitBounds.max.x))
  .and(p.y.greaterThanEqual(hitBounds.min.y)).and(p.y.lessThanEqual(hitBounds.max.y))
  .and(p.z.greaterThanEqual(hitBounds.min.z)).and(p.z.lessThanEqual(hitBounds.max.z));

export function axisFaceTSL(g) {
  const ag = abs(g).toVar();
  const nx = ag.x.greaterThanEqual(ag.y).and(ag.x.greaterThanEqual(ag.z));
  const ny = ag.y.greaterThan(ag.x).and(ag.y.greaterThanEqual(ag.z));
  return select(nx, vec3(sign(g.x), 0, 0),
                select(ny, vec3(0, sign(g.y), 0), vec3(0, 0, sign(g.z))));
}

// The face a ray crossed at p, exactly, for a world of blocks: the block-grid
// plane p is nearest to - the march stops within millimetres of the surface -
// among the planes the ray is not running parallel to, a tie (a corner) going
// to the plane it crosses most squarely. leaving: the normal points along the
// ray (a face it exits); otherwise against it (a face it hits). A field gradient cannot do this at an edge or a
// corner, where two faces meet and the gradient points between them: snapped
// to the wrong axis, an exit bent against the wrong face and sent the ray to
// the sky - the white rims along every glass edge. Mirrored on the CPU by
// boxgrid.js blockFaceNormal.
// terrain: the block volume, so the planes of the shape in p's cell count too
// (world.js codeBox) - a pane's or a half's faces sit mid-cell. Mirrors
// boxgrid.js blockFaceNormal with a world.
export function blockFaceTSL(p, dir, leaving, terrain = null) {
  const f = fract(p.div(float(BLOCK_METRES)).add(float(0.5))).toVar();
  const ad = abs(dir).toVar();
  const edge = min(f, vec3(1, 1, 1).sub(f)).toVar();
  if (terrain) {
    const box = codeBoxTSL(blockVolumeTSL(terrain, round(p.div(float(BLOCK_METRES)))).w);
    edge.assign(min(edge, min(abs(f.sub(box.lo)), abs(f.sub(box.hi)))));
  }
  // Metres to the nearest boundary on each axis, less a millimetre's worth of
  // how squarely the ray crosses it (the tie-break), and ruled out where the ray
  // runs parallel to the planes.
  const score = edge.mul(float(BLOCK_METRES))
    .sub(ad.mul(float(1e-3)))
    .add(step(ad, vec3(1e-3, 1e-3, 1e-3)).mul(float(1e3))).toVar();   // per axis: not a select (one bool)
  const sx = score.x.lessThanEqual(score.y).and(score.x.lessThanEqual(score.z));
  const sy = score.y.lessThan(score.x).and(score.y.lessThanEqual(score.z));
  const s = leaving ? sign(dir) : sign(dir).negate();
  return select(sx, vec3(s.x, 0, 0), select(sy, vec3(0, s.y, 0), vec3(0, 0, s.z)));
}

// A reflection hit's albedo: which block it is (render.js's block-material
// volume), then that block's texture at the face uv - the uv BoxGeometry
// itself gives that face, so a reflected block shows its texture the right
// way round. l is the hit in block units, -0.5..0.5 across the block.
// Everything a hit's shading needs from its texel, read at the hit's own
// block and face like the albedo: the albedo, the normal-mapped shading normal
// and the texture's baked AO (LabPBR blue). The surface seen in a reflection or
// through glass is lit by its bumps as it is when seen directly - the normal
// maps are how most block textures are told apart, and a hit shaded on the
// flat face lost them.
//
// Only inside `near` (a cascade's footprint - C1's): past it the hit is far
// enough that the bumps are below a pixel, and it takes the flat face and no
// AO, skipping the reads.
export function hitSurfaceTSL(terrain, hp, nh, near = null) {
  const f = boxFaceTSL(hp, nh);
  const code = volumeAtTSL(terrain, f.block, hp.sub(nh.mul(float(VOXEL_METRES * 0.5)))).toVar();
  const layer = faceLayerTSL(code, nh).toVar();
  const st = f.st.toVar();
  const emissive = float(0).toVar();     // a model's lightSource cube
  // A placed model's cell (A > 0): find the cube whose face holds the hit's
  // texel, and take that texel as the cube's own draw does (render.js
  // modelTexelCoreTSL) - so models reflect as themselves, not as layer 0.
  // The trace's normal can be off at a thin cube's edge (a chair leg is one
  // texel), putting the texel just outside every box: then the NEAREST cube,
  // at its nearest texel - never the layer-0 fallback (green).
  if (terrain.modelHits) {
    If(code.w.greaterThan(float(0.5)).and(code.x.lessThan(float(0.5))), () => {
      const buf = terrain.modelHits.node;
      const hdr = buf.element(int(code.y.add(code.z.mul(float(256))))).toVar();
      const q = float(VOXEL_METRES);
      const t = floor(hp.sub(nh.mul(q.mul(float(0.5)))).div(q)).toVar();
      const best = int(-1).toVar();
      const bestD = float(1e9).toVar();
      Loop({ start: int(0), end: int(hdr.y), type: 'int', condition: '<' }, ({ i }) => {
        const r = int(hdr.x).add(i.mul(int(MODEL_HIT_RECORD))).toVar();
        const lo = buf.element(r).xyz, hi = buf.element(r.add(int(1))).xyz.sub(float(1));
        // Texels outside the box, per axis (0 inside).
        const dd = length(max(lo.sub(t), vec3(0, 0, 0)).add(max(t.sub(hi), vec3(0, 0, 0)))).toVar();
        If(dd.lessThan(bestD), () => { bestD.assign(dd); best.assign(r); });
        If(dd.lessThan(float(0.5)), () => { Break(); });
      });
      If(best.greaterThanEqual(int(0)), () => {
        const lo = buf.element(best).xyz, hi = buf.element(best.add(int(1))).xyz.sub(float(1));
        // The nearest texel of that cube, as a point on a face along nh.
        const pc = clamp(t, lo, hi).add(float(0.5)).mul(q).add(nh.mul(q.mul(float(0.5))));
        const m = modelTexelCoreTSL(nh, pc, buf.element(best.add(int(2))), buf.element(best.add(int(3))),
                                    buf.element(best.add(int(4))), buf.element(best.add(int(5))));
        layer.assign(m.layer);
        st.assign(m.st);
        emissive.assign(select(m.emissive, float(1), float(0)));
      });
    });
  }
  const a = texture(terrain.albedo, st).depth(int(layer)).level(int(0)).rgb;
  const n = nh.toVar();
  const ao = float(1).toVar();
  const read = () => {
    const lab = labNormalTSL(texture(terrain.normal, st).depth(int(layer)).level(int(0))).toVar();
    n.assign(boxBumpedNormalTSL(nh, lab.xyz));
    ao.assign(lab.w);
  };
  if (near) {
    const dim = float(GRID_DIM);
    const vp = hp.sub(near.origin).div(near.voxel).toVar();
    If(vp.x.greaterThanEqual(float(0)).and(vp.y.greaterThanEqual(float(0)))
       .and(vp.z.greaterThanEqual(float(0))).and(vp.x.lessThan(dim))
       .and(vp.y.lessThan(dim)).and(vp.z.lessThan(dim)), read);
  } else read();
  // The raw _s texel, for a hit that reflects in turn (nestedReflect).
  const spec = texture(terrain.specular, st).depth(int(layer)).level(int(0));
  // albedo is as drawn - white under whiteworld, which reflections show;
  // raw is the texture's own, which is what a surface bounces (texel GI):
  // whiteworld and the GI-only view whiten the surface lit, not its light.
  return { albedo: mix(a, vec3(1, 1, 1), terrainWhite), raw: a, n, ao, layer, st, emissive, spec };
}

// A reflection hit's face and locked texel, checked against the block volume.
// The face: the trace's (the field gradient snapped to an axis) when the cell
// behind it is solid, else the exact block-grid face (blockFaceTSL) when that
// one's is. The texel: locked as usual, then kept inside that solid cell (or,
// if the march stopped a hair short past an edge, the cell one texel further
// along the ray). Near a corner or an edge the gradient can lean into the
// other face, and the lock can land a texel over the edge - either names the
// air beside the hit, read as layer 0: green speckles in reflections (seen on
// Chrome's Dawn).
export function hitTexelTSL(terrain, p, g, dir) {
  const q = float(VOXEL_METRES), Bm = float(BLOCK_METRES);
  const cellAt = x => round(x.div(Bm));
  const solidAt = k => {
    const c = blockVolumeTSL(terrain, k);
    return c.x.greaterThan(float(0.5)).or(c.w.greaterThan(float(0.5)));
  };
  const behind = n => cellAt(p.sub(n.mul(q.mul(float(0.5)))));
  const b = blockFaceTSL(p, dir, false).toVar();
  const n = select(solidAt(behind(g)).or(solidAt(behind(b)).not()), g, b).toVar();
  const kB = behind(n).toVar();
  const kA = cellAt(p.add(dir.mul(q))).toVar();
  const k = select(solidAt(kB).or(solidAt(kA).not()), kB, kA).toVar();
  // The lock, its point behind the face clamped into k's texel centres.
  const half = Bm.sub(q).mul(float(0.5));
  const lo = k.mul(Bm).sub(half), hi = k.mul(Bm).add(half);
  const inner = clamp(texelLockTSL(p, n).sub(n.mul(q.mul(float(0.5)))), lo, hi);
  return { n, p: inner.add(n.mul(q.mul(float(0.5)))) };
}

// A block's entry in render.js's block volume, as bytes (RGBA x 255):
// R side layer + 1 (0 air), G top layer, B bottom layer - or, in a model's
// cell, R 0, G + 256 B its modelHits header, A its GI entry + 1.
export function blockVolumeTSL(terrain, block) {
  const vol = terrain.blocks;
  const uvw = block.sub(vol.origin).add(float(0.5)).div(vol.dims);
  return round(texture3D(vol.tex, uvw).level(int(0)).mul(float(255)));
}

// The layer a block's face n draws with: its top, bottom or side. Air and
// outside the volume read as layer 0. Every select here is INT-typed: a
// float select reused in branching code hands back its cached f32 variable
// unconverted (three's ConditionalNode), and a texture array's layer must
// be an integer - that broke the shaders once.
function faceLayerTSL(code, n) {
  const side = int(max(code.x.sub(float(1)), float(0)));
  const face = select(n.y.greaterThan(float(0.5)), int(code.y),
                      select(n.y.lessThan(float(-0.5)), int(code.z), side));
  return select(code.x.greaterThan(float(0.5)), face, int(0));
}

// p: a point on the face (n its outward normal) - a dual block's half is the
// one behind it.
export function blockLayerTSL(terrain, block, n, p) {
  return faceLayerTSL(volumeAtTSL(terrain, block, p.sub(n.mul(float(VOXEL_METRES * 0.5)))), n);
}

// Whether cell fractions f lie in a codeBoxTSL box (half-open, as partAt).
const inCodeBoxTSL = (f, box) => f.x.greaterThanEqual(box.lo.x).and(f.x.lessThan(box.hi.x))
  .and(f.y.greaterThanEqual(box.lo.y)).and(f.y.lessThan(box.hi.y))
  .and(f.z.greaterThanEqual(box.lo.z)).and(f.z.lessThan(box.hi.z));

// A cell's entry as blockVolumeTSL reads it, for the part holding world point
// q: a dual block (A >= 128; render.js buildBlockVolume) outside its own
// half's box answers its other half's R, G, B from the volume's second half.
export function volumeAtTSL(terrain, cell, q) {
  const v = blockVolumeTSL(terrain, cell).toVar();
  If(v.w.greaterThan(float(127.5)), () => {
    const f = q.div(float(BLOCK_METRES)).sub(cell).add(float(0.5));
    If(inCodeBoxTSL(f, codeBoxTSL(v.w)).not(), () => {
      const v2 = blockVolumeTSL(terrain, cell.add(vec3(0, 0, terrain.blocks.size.z)));
      v.assign(vec4(v2.xyz, v.w));
    });
  });
  return v;
}

// The box the rays take for a shape code (the volume's A; world.js codeBox),
// in cell fractions: a low or high half, a vertical half or a pane turned to
// its facing (0 N, 1 E, 2 S, 3 W), else the whole cell.
function codeBoxTSL(a) {
  const c = a.mod(float(128)).toVar();
  const f = c.mod(float(4));
  const alongX = f.equal(float(1)).or(f.equal(float(3)));
  const plus = f.equal(float(1)).or(f.equal(float(2)));            // E, S: the + side
  const vert = c.greaterThanEqual(float(19.5)).and(c.lessThan(float(23.5)));
  const pane = c.greaterThanEqual(float(23.5)).and(c.lessThan(float(27.5)));
  // Along the shape's axis: [0.5, 1] or [0, 0.5] for a vertical half, the
  // middle four texels for a pane, else [0, 1].
  const a0 = select(vert, select(plus, float(0.5), float(0)), select(pane, float(4 / 12), float(0)));
  const a1 = select(vert, select(plus, float(1), float(0.5)), select(pane, float(8 / 12), float(1)));
  const y0 = select(c.equal(float(17)), float(0.5), float(0));
  const y1 = select(c.equal(float(16)), float(0.5), float(1));
  const lo = vec3(select(alongX, a0, float(0)), y0, select(alongX, float(0), a0));
  const hi = vec3(select(alongX, a1, float(1)), y1, select(alongX, float(1), a1));
  return { lo, hi };
}

// The other half's code for a dual block's (world.js complementShape): low
// and high halves swap, a vertical half turns to face the other way.
function complementCodeTSL(a) {
  const c = a.mod(float(128));
  const f = c.mod(float(4));
  return select(c.equal(float(16)), float(17), select(c.equal(float(17)), float(16),
                c.sub(f).add(f.add(float(2)).mod(float(4)))));
}

// What fills world point q for the walk through glass (world.js partAt, its
// CPU mirror): code, the material's layer + 1 (0 air - and a model's cell,
// which the walk ignores), and that part's box, lo and hi in metres. Outside
// the volume or the battle arena, air.
export function partAtTSL(terrain, q) {
  const Bm = float(BLOCK_METRES);
  const cell = round(q.div(Bm)).toVar();
  const v = blockVolumeTSL(terrain, cell).toVar();
  const f = q.div(Bm).sub(cell).add(float(0.5)).toVar();
  const own = codeBoxTSL(v.w);
  const inOwn = inCodeBoxTSL(f, own).toVar();
  // A dual block's other half fills the rest of the cell: its code from the
  // volume's second half, its box the complement's.
  const dual = v.w.greaterThan(float(127.5)).and(inOwn.not()).toVar();
  const other = codeBoxTSL(complementCodeTSL(v.w));
  If(dual, () => {
    v.assign(vec4(blockVolumeTSL(terrain, cell.add(vec3(0, 0, terrain.blocks.size.z))).x, v.y, v.z, v.w));
  });
  const box = { lo: select(dual, other.lo, own.lo), hi: select(dual, other.hi, own.hi) };
  const inBox = inOwn.or(dual);
  // The volume clamps at its edge, which would repeat the border: outside
  // it is air, as blockCodeTSL reads it.
  const rel = cell.sub(terrain.blocks.origin);
  const inVol = rel.x.greaterThanEqual(float(0)).and(rel.y.greaterThanEqual(float(0)))
    .and(rel.z.greaterThanEqual(float(0))).and(rel.x.lessThan(terrain.blocks.size.x))
    .and(rel.y.lessThan(terrain.blocks.size.y)).and(rel.z.lessThan(terrain.blocks.size.z));
  const code = select(inBox.and(inVol).and(inHitBoundsTSL(cell.mul(Bm))), v.x, float(0));
  const base = cell.sub(float(0.5));
  return { code, lo: base.add(box.lo).mul(Bm), hi: base.add(box.hi).mul(Bm) };
}

// The codes of the glass materials (materials.js glass: true).
const GLASS_CODES = MATERIALS.map((m, i) => (m.glass ? i + 1 : null)).filter(c => c !== null);

export function isGlassCodeTSL(code) {
  if (!GLASS_CODES.length) return code.lessThan(float(-1));   // never
  return GLASS_CODES.map(c => code.equal(float(c))).reduce((a, b) => a.or(b));
}
