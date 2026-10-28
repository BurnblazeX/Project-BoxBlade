import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, int, vec2, vec4, ivec2, uniform, textureLoad, dot, abs, clamp, min, max, floor,
         fract, mix, ceil, positionWorld } from 'three/tsl';
import { SUN_PENUMBRA_CAP, SUN_MAP_MAX_SIDE, SUN_MAP_TAP_SPACING } from '../core/sunmap.js';
import { terrainKeepTSL } from './terrain.js';
import { layout } from './layout.js';

// --- The sun depth map, on the GPU (core/sunmap.js has the why) ---
//
// One target and its uniforms, shared by every shader that shades in sunlight:
// written by render/features/sunmap.js when it redraws, read here. Module
// state on purpose, like sunFarUniform - every shader set reads the same map,
// and a redraw is a few uniform writes, never a rebuild. The target keeps its
// texture object through a resize, so the shaders' binding stays good.
//
// Full float: a stored s is metres up to a region's diagonal, and a receiver
// outside C2 compares it at SUN_MAP_MIN_REACH (1.5 m) - half float's half a
// metre at a few hundred would be its own shadow.

const target = new THREE.RenderTarget(1, 1, { type: THREE.FloatType, format: THREE.RedFormat,
                                              depthBuffer: true });
target.texture.minFilter = target.texture.magFilter = THREE.NearestFilter;
target.texture.generateMipmaps = false;

export const sunMap = {
  target,
  viewProj: uniform(new THREE.Matrix4()),
  // The sun the map was drawn for: receivers measure s along the same one, so a
  // sun that has moved since the last redraw compares like with like.
  dir: uniform(new THREE.Vector3(0, 1, 0)),
  base: uniform(float(0)),
  size: uniform(float(1)),
  texel: uniform(float(1)),      // metres a texel
  on: uniform(int(0)),
  // The softening grid's largest side (0: one hard tap everywhere); each shade
  // uses as few as its penumbra needs (core/sunmap.js sunMapGrid).
  // render/features/sunmap.js sets it from the preset.
  side: uniform(int(4))
};

// The penumbra's widest (core/sunmap.js) - read by the march's cone too.
export const sunPenumbraCap = uniform(float(SUN_PENUMBRA_CAP));

// How far the soft march looks toward the sun: 0 its 12 m cap, 1 on to where
// the ray leaves C1. Never into C2, whose 0.5 m voxels step a shadow's edge -
// past C1 the map, finer, takes over (gpu/sun.js). A uniform: no rebuild.
export const sunSoftReach = uniform(int(1));

// The top of the tallest thing in the world, metres: a sun ray above it can
// meet nothing more, so the march stops there (gpu/trace.js). Huge until the
// world is measured (render/features/sunmap.js), so nothing stops early.
export const sunWorldTop = uniform(float(1e9));

// The pass's material: the terrain's own vertex shape (slopes as their planes,
// glass dropped - it tints the sun, it does not stop it), and none of the
// view's lifting (a roof lifted for the camera still shadows), writing s.
export function createSunMapMaterial() {
  const m = new THREE.MeshBasicNodeMaterial();
  m.positionNode = terrainKeepTSL(false, false, false);
  m.outputNode = vec4(dot(positionWorld, sunMap.dir).sub(sunMap.base), 0, 0, 1);
  m.side = THREE.DoubleSide;
  m.blending = THREE.NoBlending;
  return m;
}

const SEARCH_SIDE = 4;

// The far half's visibility at p (the march's origin): 1 lit, 0 shadowed.
//   reach   how far toward the sun the march covered - only blockers past it
//           (fading in from fadeStart x reach, as the march fades out) count
//   fadeStart  the march's fade start, a fraction of reach
//   coneR   the sun's cone radius: the penumbra widens by this per metre of
//           blocker distance, as the march's does, up to sunPenumbraCap
//   soft    0..1: how much of the softening applies (1 inside C2, fading to 0
//           at its edge); 0 is one hard tap
//
// Smooth, not sampled: every tap is a bilinear blend of its four texels'
// verdicts, on a regular grid with no per-texel turn, never more than
// SUN_MAP_TAP_SPACING texels apart - so an edge is a ramp, not the sawtooth
// and dither of a few nearest taps, nor the offset copies of a grid spread too
// thin (both seen 2026-10-28). The penumbra lies OUTSIDE the shadow's edge,
// as the cone's does (its d / R t is 0 at the edge): the grid's occluded
// share f becomes visibility 1 - 2f, 0 at the edge, 1 a penumbra out. A
// blocker thinner than the penumbra keeps a lighter shadow rather than none.
export const sunMapFarTSL = Fn(([p, reach, fadeStart, coneR, soft]) => {
  const vis = float(1).toVar();
  If(sunMap.on.equal(int(1)), () => {
    const clip = sunMap.viewProj.mul(vec4(p, 1));
    const ndc = clip.xy.div(clip.w).toVar();
    If(abs(ndc.x).lessThan(float(1)).and(abs(ndc.y).lessThan(float(1))), () => {
      // WebGPU's framebuffer: row 0 at the top, ndc y up.
      const px = vec2(ndc.x.mul(0.5).add(0.5), float(0.5).sub(ndc.y.mul(0.5))).mul(sunMap.size).toVar();
      const sp = dot(p, sunMap.dir).sub(sunMap.base).toVar();
      const from = reach.mul(fadeStart).toVar();
      const span = max(reach.sub(from), float(1e-3)).toVar();
      const hi = sunMap.size.sub(float(1));
      // How far toward the sun the blocker over a texel is (negative: none),
      // and how much it counts.
      const blocker = q => textureLoad(sunMap.target.texture,
        ivec2(clamp(q, vec2(0, 0), vec2(hi, hi)))).x.sub(sp);
      const occlusion = b => clamp(b.sub(from).div(span), float(0), float(1));
      // A tap at q, in texels: its four texels' occlusions, blended bilinearly.
      const smoothTap = q => {
        const g = q.sub(float(0.5)).toVar();
        const c = floor(g).toVar(), f = fract(g).toVar();
        const a = occlusion(blocker(c)), b = occlusion(blocker(c.add(vec2(1, 0))));
        const d = occlusion(blocker(c.add(vec2(0, 1)))), e = occlusion(blocker(c.add(vec2(1, 1))));
        return mix(mix(a, b, f.x), mix(d, e, f.x), f.y);
      };
      If(soft.lessThanEqual(float(0)).or(sunMap.side.lessThanEqual(int(0))), () => {
        vis.assign(float(1).sub(smoothTap(px)));
      }).Else(() => {
        const n = float(sunMap.side);
        // Where the blockers are: their mean distance over the widest penumbra,
        // on a 4 x 4 grid of plain taps at most - an average needs no more.
        const ns = min(n, float(SEARCH_SIDE)).toVar();
        const at = (i, j) => vec2(float(i), float(j)).add(float(0.5)).div(ns).mul(float(2)).sub(float(1));
        const search = sunPenumbraCap.mul(soft).div(sunMap.texel).toVar();
        const sum = float(0).toVar(), count = float(0).toVar();
        Loop({ start: 0, end: SEARCH_SIDE, type: 'int', condition: '<', name: 'si' }, ({ si }) => {
          Loop({ start: 0, end: SEARCH_SIDE, type: 'int', condition: '<', name: 'sj' }, ({ sj }) => {
            If(float(si).lessThan(ns).and(float(sj).lessThan(ns)), () => {
              const b = blocker(px.add(at(si, sj).mul(search))).toVar();
              If(b.greaterThan(from), () => { sum.addAssign(b); count.addAssign(float(1)); });
            });
          });
        });
        // None near: one smooth tap (a lone far edge just outside the search).
        // Otherwise filter over the penumbra their mean distance makes - the
        // march's own R t, capped where it is capped.
        If(count.equal(float(0)), () => { vis.assign(float(1).sub(smoothTap(px))); }).Else(() => {
          // The grid this penumbra needs (core/sunmap.js sunMapGrid): enough
          // taps that none is more than SUN_MAP_TAP_SPACING from the next, up
          // to the preset's side - past that the penumbra narrows instead.
          const want = min(coneR.mul(sum.div(count)), sunPenumbraCap).mul(soft).div(sunMap.texel);
          const g = clamp(ceil(want.mul(float(2 / SUN_MAP_TAP_SPACING))), float(1), n).toVar();
          const radius = min(want, g.mul(float(SUN_MAP_TAP_SPACING / 2))).toVar();
          const cell = (i, j) => vec2(float(i), float(j)).add(float(0.5)).div(g).mul(float(2)).sub(float(1));
          const occ = float(0).toVar();
          Loop({ start: 0, end: SUN_MAP_MAX_SIDE, type: 'int', condition: '<', name: 'fi' }, ({ fi }) => {
            Loop({ start: 0, end: SUN_MAP_MAX_SIDE, type: 'int', condition: '<', name: 'fj' }, ({ fj }) => {
              If(float(fi).lessThan(g).and(float(fj).lessThan(g)), () => {
                occ.addAssign(smoothTap(px.add(cell(fi, fj).mul(radius))));
              });
            });
          });
          vis.assign(clamp(float(1).sub(occ.div(g.mul(g)).mul(float(2))), float(0), float(1)));
        });
      });
    });
  });
  return vis;
}).setLayout(layout('sunMapFar', 'float', [
  { name: 'p', type: 'vec3' }, { name: 'reach', type: 'float' }, { name: 'fadeStart', type: 'float' },
  { name: 'coneR', type: 'float' }, { name: 'soft', type: 'float' }]));
