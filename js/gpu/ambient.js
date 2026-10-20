import { Fn, If, Loop, Break, float, vec3, vec4, uniform, texture3D, max, abs, vec2, min, normalize, cross, clamp, cos, sin, smoothstep, length, int, select, mix } from 'three/tsl';
import { GRID_DIM, VOXEL_METRES, HIT_EPS, MIN_STEP } from '../core/boxgrid.js';
import { BLOCK_METRES } from '../core/world.js';
import { LPV_DIM, LPV_CELL_METRES } from '../core/lpv.js';
import { terrainSurfaceTSL, terrainTexelTSL } from './terrain.js';
import { VOXEL_AO_MIN } from '../core/voxelao.js';
import { layout, memoCascades } from './layout.js';
import { insideAtTSL } from './occlusion.js';
import { SURFACE_BIAS_VOXELS, decodeFieldTSL, eachCascade, ringUV } from './field.js';

// --- AO: cone-traced ambient occlusion (doc §6.2, the short band) ---
//
// Measures what RTAO measures - how much of the hemisphere is blocked within
// AO_DISTANCE - but with six wide cones instead of a few thin random rays. Each
// cone is marched through the same field as the soft sun, and at every step the
// field's distance against the cone's width, d / (R*t), says how much of the
// cone is blocked there - so one cone integrates a whole solid angle rather than
// sampling a line. Blocking weighs less the further out it is (1 - t / reach):
// contact darkens hard, a wall at the edge of reach barely at all.
//
// Deterministic: the same six directions everywhere, no per-texel randomness,
// so there is no noise to hide and the answer is identical frame to frame.
//
// The cones are the usual hemisphere tiling: one along the normal, five tilted
// 60 degrees from it, each 60 degrees wide. Weights are the cosine-weighted
// share of the hemisphere each covers, so open sky sums to exactly 1.
export const AO_CONES = 6;

export const AO_DISTANCE = 1.0;   // metres

const AO_CONE_R = Math.tan(Math.PI / 6);   // 60 degree aperture

const AO_TILT = Math.PI / 3;

const AO_WEIGHT_UP = 0.25, AO_WEIGHT_SIDE = 0.15;

function createAOConeTSL(cascades) {
  return memoCascades(cascades, 'ao', () => Fn(([origin, dir, maxDist, coneR]) => {
    const dim = float(GRID_DIM);
    // A voxel out, as the sun's cone: at t = 0 the cone has no width.
    const t = float(VOXEL_METRES).toVar();
    const occ = float(0).toVar();
    const done = float(0).toVar();
    const hit = float(0).toVar();
    for (const { c } of eachCascade(cascades, float(0))) {
      If(done.equal(float(0)), () => {
        Loop({ start: 0, end: 32, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel);
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => { Break(); });
          const d = decodeFieldTSL(texture3D(c.tex, ringUV(vp, c.ring)).r, c.range);
          const blocked = float(1).sub(clamp(d.mul(c.voxel).div(coneR.mul(t)), float(0), float(1)));
          occ.assign(max(occ, blocked.mul(float(1).sub(t.div(maxDist)))));
          If(d.lessThan(float(HIT_EPS)), () => {
            done.assign(float(1)); hit.assign(float(1)); Break();
          });
          t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }
    // x the occlusion; y how far the axis got, z whether it ended on a surface -
    // which is where the GI gather samples the LPV.
    return vec3(occ, min(t, maxDist), hit);
  }).setLayout(layout('aoCone', 'vec3', [
    { name: 'origin', type: 'vec3' }, { name: 'dir', type: 'vec3' },
    { name: 'maxDist', type: 'float' }, { name: 'coneR', type: 'float' }])));
}

// The GI volumes, sampled trilinearly at a world position. gi.levels holds one
// gi.js binding per volume, finest first: C0's, then C1's.
//
// Each level fades to nothing over its outermost cells, and the finer one is
// laid over the coarser: inside C0 it is C0's answer, crossing its edge it
// blends into C1's, and past C1's edge the bounce is gone. Sampling one volume
// outside itself is what went wrong before - the texture clamps, so everything
// beyond C0 read whatever its boundary cells held.
const LPV_EDGE_CELLS = 2;

// Half the finest volume's cell: how far off a surface the GI is read, so the
// read is in the air cell before it rather than straddling the face.
export const lpvHalfCell = gi => (gi.levels[0].cellMetres ?? LPV_CELL_METRES) * 0.5;

// level0: read at an explicit level 0 - in a branch only some fragments take
// (gpu/shade.js revealedLight), where an implicit-derivative read is not allowed.
// withSky: vec4 - the light, and the sky visibility (gpu/gi.js, the texture's
// a), which past the volumes is the whole sky (1), as it was before them.
export function lpvSampleTSL(gi, p, level0 = false, withSky = false) {
  let out = null, vis = null;
  for (let i = gi.levels.length - 1; i >= 0; i--) {
    const lv = gi.levels[i];
    const uvw = p.sub(lv.origin).div(float(lv.extent)).toVar();
    // Distance to the nearest face of the volume, in cells.
    const edge = min(min(min(uvw.x, uvw.y), uvw.z),
                     min(min(float(1).sub(uvw.x), float(1).sub(uvw.y)), float(1).sub(uvw.z)))
                   .mul(float(lv.dim ?? LPV_DIM));
    const w = smoothstep(float(0.5), float(0.5 + LPV_EDGE_CELLS), edge).toVar();
    const t = (level0 ? texture3D(lv.tex, uvw).level(float(0)) : texture3D(lv.tex, uvw)).toVar();
    out = out ? mix(out, t.rgb, w) : t.rgb.mul(w);
    if (withSky) vis = mix(vis || float(1), t.a, w);
  }
  return withSky ? vec4(out, vis) : out;
}

// The LPV at q as the shading takes it: { light, the bounce times the
// strength; keep, what is left of the sky ambient - the sky visibility under
// gi.sky (an int uniform, the A/B), else 1 }.
export function lpvAtTSL(gi, q, level0 = false) {
  const s = lpvSampleTSL(gi, q, level0, true).toVar();
  return { light: s.xyz.mul(gi.strength), keep: mix(float(1), s.w, float(gi.sky)) };
}

// The light a revealed face is drawn with (gpu/terrain.js terrainRevealedTSL:
// a wall's top under a lifted roof). Its own would come from inside the
// solid the lifted roof still is to the lighting - black - so it takes the
// room's: the ambient a face there gets (skyIrradiance, the shading's own)
// and the bounce in the room's air beside it - the LPV (gi; none: the ambient
// alone) one block out on each side, just below the face, the room's side kept
// by the interiors' inside mask (outside the walls it is 0). Times a fixed
// form shading, so a block shrinking away still reads as a cube. No direct
// light: the sun and the lamps would need a march, from inside the same solid.
// Returns (p, n) -> rgb: p, n the texel's locked position and its face
// normal, one answer a texel. Every read at an explicit level - it runs in a
// branch only the revealed pixels take.
export const REVEALED_FORM = { top: 1, side: 0.8, bottom: 0.6 };
// room: split shading's room meter (gpu/roommeter.js), or null - where it
// has a mean and is on, that is the light instead: one value for the room.
export function createRevealedLightTSL(skyIrradiance, gi, room = null) {
  return (p, n) => {
    let light = skyIrradiance(n);
    if (gi) {
      // The room's sky visibility too: the most any inside sample sees -
      // the whole sky where none is inside.
      const below = vec3(0, -lpvHalfCell(gi), 0);
      let bounce = null, seen = null, inside = null;
      for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const q = p.add(vec3(dx * BLOCK_METRES, 0, dz * BLOCK_METRES)).add(below);
        const m = insideAtTSL(q.xz).toVar();
        const s = lpvSampleTSL(gi, q, true, true).toVar();
        const b = s.xyz.mul(m), v = s.w.mul(m);
        bounce = bounce ? max(bounce, b) : b;
        seen = seen ? max(seen, v) : v;
        inside = inside ? max(inside, m) : m;
      }
      const vis = select(inside.greaterThan(float(0)), seen, float(1));
      light = light.mul(mix(float(1), vis, float(gi.sky))).add(bounce.mul(gi.strength));
    }
    if (room) {
      const m = room.meanRead.element(int(0));
      light = select(room.on.equal(int(1)).and(m.w.greaterThan(float(0.5))), m.xyz, light);
    }
    const form = select(n.y.greaterThan(float(0.5)), float(REVEALED_FORM.top),
      select(n.y.lessThan(float(-0.5)), float(REVEALED_FORM.bottom), float(REVEALED_FORM.side)));
    return light.mul(form);
  };
}

// Returns Fn([p, n, lift, skip]) -> visibility in [0, 1]: n is the hemisphere,
// lift the direction the origin is raised along (the geometry's, as for
// shadows), skip the sprite's own card.
//
// With an lpv it returns vec4(gi.rgb, visibility) instead: the same six cones
// are the GI gather's short band (§6.2), so the bounce costs six texture reads
// on top of the AO, not six more marches. Each cone samples the LPV where it
// ENDED - pulled back half a cell off the surface it hit, or at full reach when
// it escaped - and the samples are weighted as the occlusion is. Sampling at the
// hit rather than at the receiver is what makes the bounce directional: the
// cones that struck a red wall bring back red, the ones that escaped bring back
// whatever fills the open air.
export function createAOTSL({ cascades, distance, biasUniform = null, lpv = null }) {
  const cone = createAOConeTSL(cascades);
  const bias = biasUniform || uniform(float(SURFACE_BIAS_VOXELS));
  const R = float(AO_CONE_R);
  // A plain function, called inline: it returns three things.
  //   vis   the visibility, as before
  //   gi    the gathered bounce (null without an lpv)
  //   bent  the bent normal - the cone directions weighted by how open each
  //         is. Where the left is blocked it leans right, so the sky ambient
  //         read along it comes from the side that can actually see sky.
  //         Free: the cones were marched anyway.
  //
  // A loop, not six unrolled copies, and its count is a node: `single` (a bool
  // node, or null) drops it to the one cone along the normal, weighted 1, for
  // shading LOD - see createShadowColorNode's lod.
  return (p, n, lift, skip, single = null) => {
    const origin = p.add(lift.mul(bias.mul(float(VOXEL_METRES)))).toVar();
    const helper = select(abs(n.y).lessThan(float(0.999)), vec3(0, 1, 0), vec3(1, 0, 0));
    const T = normalize(cross(helper, n)).toVar();
    const B = cross(n, T).toVar();
    const vis = float(1).toVar();
    const gi = vec3(0, 0, 0).toVar();
    const sky = lpv ? float(0).toVar() : null;
    const bent = vec3(0, 0, 0).toVar();
    // single === true compiles the one-cone case in; a node selects at runtime.
    const count = single === true ? int(1)
      : single ? select(single, int(1), int(AO_CONES)) : int(AO_CONES);
    Loop({ start: int(0), end: count, type: 'int', condition: '<', name: 'aoK' }, ({ aoK }) => {
      // Cone 0 straight up the normal; 1-5 tilted AO_TILT, 72 degrees apart.
      const up = aoK.equal(int(0));
      const phi = float(aoK.sub(int(1))).mul(float(2 * Math.PI / 5));
      const st = select(up, float(0), float(Math.sin(AO_TILT)));
      const ct = select(up, float(1), float(Math.cos(AO_TILT)));
      const dir = normalize(T.mul(st.mul(cos(phi))).add(B.mul(st.mul(sin(phi))))
                            .add(n.mul(ct))).toVar();
      const w0 = select(up, float(AO_WEIGHT_UP), float(AO_WEIGHT_SIDE));
      const w = (single === true ? float(1)
        : single ? select(single, float(1), w0) : w0).toVar();
      const r = cone(origin, dir, distance, R).toVar();
      vis.subAssign(r.x.mul(w));
      bent.addAssign(dir.mul(float(1).sub(r.x).mul(w)));
      if (lpv) {
        const back = select(r.z.greaterThan(float(0)),
                            max(r.y.sub(float(lpvHalfCell(lpv))), float(0)), r.y);
        const s = lpvSampleTSL(lpv, origin.add(dir.mul(back)), false, true).toVar();
        gi.addAssign(s.xyz.mul(w));
        sky.addAssign(s.w.mul(w));
      }
    });
    const v = clamp(vis, float(0), float(1));
    // Fully enclosed: no open direction, and the ambient is zero anyway.
    const b = select(length(bent).greaterThan(float(1e-4)), normalize(bent), n);
    // sky: the LPV's sky visibility where the cones ended, weighted as the
    // occlusion is - the large-scale share the 1 m cones cannot see.
    return { vis: v, gi: lpv ? gi : null, sky: lpv ? sky : null, bent: b.toVar() };
  };
}

// The terrain's AO for its PLAIN material - the one it wears with shadows off:
// the static voxel AO of voxelao.js, this is its shader mirror. No field, no
// marches: eight occupancy reads from the block-material volume, at the texel
// centre so it stays one answer per texel. A float for the material's aoNode,
// which in three's lighting darkens the ambient only, not the sun.
// cards: the sun's card bindings, for the sprites' contact discs on the
// ground (createSpriteAOTSL) - analytic, no marches. Null until they load.
export function createVoxelAONode(terrain, cards = null) {
  const vol = terrain.blocks;
  const spriteAO = createSpriteAOTSL(cards);
  return Fn(() => {
    // A slope's plane is taken as a top face of its own block (approximate).
    const surf = terrainSurfaceTSL();
    const n = select(surf.on, vec3(0, 1, 0), surf.n).toVar();
    const p = surf.p;
    const block = surf.block;
    // The face's two tangent axes, and where the texel sits along each, 0..1.
    const t1 = select(abs(n.x).greaterThan(float(0.5)), vec3(0, 1, 0), vec3(1, 0, 0)).toVar();
    const t2 = select(abs(n.z).greaterThan(float(0.5)), vec3(0, 1, 0), vec3(0, 0, 1)).toVar();
    const l = p.div(float(BLOCK_METRES)).sub(block).toVar();
    const a = l.dot(t1).add(float(0.5)), b = l.dot(t2).add(float(0.5));
    // Occupancy in the layer in front of the face; off the volume is air.
    const front = block.add(n).toVar();
    const occ = (i, j) => {
      const uvw = front.add(t1.mul(float(i))).add(t2.mul(float(j)))
        .sub(vol.origin).add(float(0.5)).div(vol.size);
      const inside = uvw.x.greaterThan(float(0)).and(uvw.y.greaterThan(float(0)))
        .and(uvw.z.greaterThan(float(0))).and(uvw.x.lessThan(float(1)))
        .and(uvw.y.lessThan(float(1))).and(uvw.z.lessThan(float(1)));
      // uvw is over the volume's blocks; the texture holds a dual's halves too.
      const filled = texture3D(vol.tex, uvw.mul(vol.size).div(vol.dims)).level(int(0)).r.greaterThan(float(0.5 / 255));
      return select(inside.and(filled), float(1), float(0)).toVar();
    };
    const e = { l: occ(-1, 0), r: occ(1, 0), d: occ(0, -1), u: occ(0, 1) };
    const corner = (s1, s2, c) => select(s1.mul(s2).greaterThan(float(0.5)), float(0),
                                         float(3).sub(s1.add(s2).add(c)).div(float(3)));
    const c00 = corner(e.l, e.d, occ(-1, -1)), c10 = corner(e.r, e.d, occ(1, -1));
    const c01 = corner(e.l, e.u, occ(-1, 1)), c11 = corner(e.r, e.u, occ(1, 1));
    const v = mix(mix(c00, c10, a), mix(c01, c11, a), b);
    const texAO = terrainTexelTSL(terrain.normal, surf.st, surf).b;
    const ao = float(VOXEL_AO_MIN).add(float(1 - VOXEL_AO_MIN).mul(v)).mul(texAO);
    return spriteAO ? ao.mul(spriteAO(p, vec2(1e9, 1e9))) : ao;
  })();
}

// --- Sprite contact AO: a soft disc under each sprite ---
//
// Sprites are flat cards, and cone-tracing AO against them read as noise, so
// their ambient occlusion is the classic analytic blob instead: a disc on the
// ground under the feet, radius half the sprite's width, darkest at the centre
// and fading out both across the disc and with height above the feet. Evaluated
// at the texel-locked position, so it lands in whole texels like everything
// else. Multiplied per sprite, so two standing close darken together.
export const SPRITE_AO_STRENGTH = 0.6;

export const SPRITE_AO_RADIUS = 0.5;   // x the sprite's width

export function createSpriteAOTSL(b) {
  if (!b || !b.atlas) return null;
  return Fn(([p, skip]) => {
    const ao = float(1).toVar();
    Loop({ start: int(0), end: b.count, type: 'int', condition: '<' }, ({ i }) => {
      const A = b.data.element(i.mul(int(4)));
      const B = b.data.element(i.mul(int(4)).add(int(1)));
      const radius = B.z.mul(float(SPRITE_AO_RADIUS * 2));
      const r = length(p.xz.sub(A.xz));
      const above = max(p.y.sub(A.y.sub(B.w)), float(0));   // over the feet
      const occ = float(1).sub(smoothstep(float(0), radius, r))
        .mul(float(1).sub(smoothstep(float(0), radius, above)))
        .mul(float(SPRITE_AO_STRENGTH));
      // Not under its own feet - a sprite does not occlude itself this way.
      const own = length(A.xz.sub(skip)).lessThan(float(0.05));
      ao.mulAssign(select(own, float(1), float(1).sub(occ)));
    });
    return ao;
  });
}
