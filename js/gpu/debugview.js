// --- Render debug views: the shader side ---
//
// One view at a time replaces the shaded colour (render/features/debugview.js
// picks it; gpu/shade.js createShadowColorNode takes it as `view`). A kernel
// option like the rest: a view is its own compiled set, and 'final' - no view -
// compiles exactly what it did before views existed.
//
// The views here need no lighting - they read the surface or the field. The
// lighting terms (shadow, direct, indirect, AO, ...) are picked out inside
// createShadowColorNode, where those terms are built.
import { Fn, If, Loop, Break, float, vec3, vec4, floor, fract, sin, max, mix, select,
         clamp, length, normalize, texture3D, cameraPosition } from 'three/tsl';
import { GRID_DIM, VOXEL_METRES, HIT_EPS, MIN_STEP, MAX_TRACE_STEPS } from '../core/field.js';
import { layout, memoCascades } from './layout.js';
import { decodeFieldTSL, ringUV } from './field.js';
import { skyOpenTSL } from './volume.js';

// The views that show light alone, not times the surface's colour - a sprite
// then draws the term itself instead of tinting its art with it.
export const RAW_VIEWS = new Set(['normal', 'facenormal', 'labpbr', 'texels', 'cascades', 'sdf',
                                  'shadow', 'direct', 'indirect', 'ao', 'gi', 'mirror',
                                  'reflection', 'highlights', 'skyopen']);
// The views split shading can show itself (the base kernel writes them): the
// bounce, which only the split kernels trace with texel GI. Any other view
// shades per fragment, as the old views did.
export const SPLIT_VIEWS = new Set(['gi', 'indirect']);

const luma = c => c.dot(vec3(0.2126, 0.7152, 0.0722));
const hash = (v, k) => fract(sin(v.dot(vec3(12.9898 + k, 78.233 - k, 37.719 + 2 * k))).mul(float(43758.5453)));

// Blue - cyan - green - yellow - red over 0..1.
export const heatTSL = x => {
  const t = clamp(x, float(0), float(1)).mul(float(4));
  return clamp(vec3(t.sub(float(2)), select(t.lessThan(float(2)), t, float(4).sub(t)), float(2).sub(t)),
               vec3(0, 0, 0), vec3(1, 1, 1));
};

// The field marched from the camera to the drawn surface p: vec4(where it hit
// (-1: no hit), steps taken, the cascade it hit in, how far p is). The same
// walk as the reflection trace (gpu/trace.js), counting its steps.
function fieldMarch(cascades) {
  return memoCascades(cascades, 'debugField', () => Fn(([surfaceP]) => {
    const origin = cameraPosition.toVar();
    const far = length(surfaceP.sub(origin)).toVar();
    const dir = surfaceP.sub(origin).div(max(far, float(1e-4))).toVar();
    const dim = float(GRID_DIM);
    const t = float(0).toVar(), done = float(0).toVar(), steps = float(0).toVar();
    const out = vec4(-1, 0, 0, 0).toVar();
    cascades.forEach((c, i) => {
      If(done.equal(float(0)), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel).toVar();
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => { Break(); });
          steps.addAssign(float(1));
          const d = decodeFieldTSL(texture3D(c.tex, ringUV(vp, c.ring)).r, c.range).toVar();
          If(d.lessThan(float(HIT_EPS)), () => {
            out.assign(vec4(t, 0, float(i), 0));
            done.assign(float(1));
            Break();
          });
          t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          // A metre past the surface: whatever the field holds behind it is
          // not what this pixel shows.
          If(t.greaterThan(far.add(float(1))), () => { done.assign(float(1)); Break(); });
        });
      });
    });
    return vec4(out.x, steps, out.z, far);
  }).setLayout(layout('debugField', 'vec4', [{ name: 'surfaceP', type: 'vec3' }])));
}

// Which cascade holds p: green C0, yellow C1, orange C2, red none - checkered
// by that cascade's voxels, over the surface's brightness.
function cascadeView(cascades, p, n, base) {
  const tints = [vec3(0.25, 0.9, 0.3), vec3(0.95, 0.85, 0.2), vec3(1, 0.5, 0.15)];
  const out = vec3(0.9, 0.15, 0.15).toVar();
  const dim = float(GRID_DIM);
  for (let i = cascades.length - 1; i >= 0; i--) {
    const c = cascades[i];
    // Half a voxel inside the face, so the checker is the voxel behind it.
    const vp = p.sub(n.mul(c.voxel.mul(float(0.5)))).sub(c.origin).div(c.voxel).toVar();
    const inside = vp.x.greaterThanEqual(float(0)).and(vp.y.greaterThanEqual(float(0)))
      .and(vp.z.greaterThanEqual(float(0))).and(vp.x.lessThan(dim)).and(vp.y.lessThan(dim)).and(vp.z.lessThan(dim));
    const v = floor(vp);
    const odd = fract(v.x.add(v.y).add(v.z).mul(float(0.5))).mul(float(2));
    If(inside, () => { out.assign(tints[Math.min(i, 2)].mul(float(0.8).add(odd.mul(float(0.2))))); });
  }
  return out.mul(float(0.35).add(luma(base).mul(float(0.65))));
}

// The SDF view: the camera ray through the field to this surface. Hue is the
// march's cost in steps (blue few, red 96+), banded every metre of distance.
// Magenta where the field has no surface here (the ray went past: a sprite,
// or the field missing geometry); white where the field's surface stands well
// in front of the drawn one (the field holds something the frame does not).
function sdfView(cascades, p) {
  const r = fieldMarch(cascades)(p).toVar();
  const voxel = select(r.z.lessThan(float(0.5)), cascades[0].voxel,
                       select(r.z.lessThan(float(1.5)), cascades[Math.min(1, cascades.length - 1)].voxel,
                              cascades[cascades.length - 1].voxel));
  const band = float(0.75).add(float(0.25).mul(select(fract(r.x).lessThan(float(0.06)), float(0), float(1))));
  const col = heatTSL(r.y.div(float(96))).mul(band).toVar();
  If(r.x.lessThan(float(0)).or(r.x.greaterThan(r.w.add(voxel.mul(float(2))))), () => {
    col.assign(mix(col, vec3(1, 0, 1), float(0.75)));
  }).ElseIf(r.x.lessThan(r.w.sub(voxel.mul(float(2)))), () => {
    col.assign(mix(col, vec3(1, 1, 1), float(0.75)));
  });
  return col;
}

// The views that need no lighting, or null for the rest.
// s: { p, n, nS, base, specular (a function: the raw LabPBR texel, or null) }
export function surfaceViewTSL(view, s, cascades) {
  switch (view) {
    case 'albedo': return s.base;
    case 'normal': return normalize(s.nS).mul(float(0.5)).add(float(0.5));
    case 'facenormal': return s.n.mul(float(0.5)).add(float(0.5));
    case 'labpbr': return s.specular ? s.specular() : vec3(0, 0, 0);
    case 'texels': {
      // One colour per shaded texel: the lock everything is shaded at.
      const wt = floor(s.p.sub(s.n.mul(float(VOXEL_METRES * 0.5))).div(float(VOXEL_METRES)));
      return vec3(hash(wt, 0), hash(wt, 1), hash(wt, 2)).mul(float(0.7)).add(float(0.3));
    }
    case 'cascades': return cascadeView(cascades, s.p, s.n, s.base);
    case 'sdf': return sdfView(cascades, s.p);
    case 'skyopen': {
      // The sky-top map (volume.js skyOpenTSL) at the texel, half a texel out
      // from its face: green open to the sky, red under a roof.
      if (!s.terrain || !s.terrain.skyTop) return vec3(0, 0, 0);
      const open = skyOpenTSL(s.terrain, s.p.add(s.n.mul(float(VOXEL_METRES * 0.5))));
      return mix(vec3(0.8, 0.15, 0.1), vec3(0.2, 0.8, 0.3), open).mul(float(0.4).add(luma(s.base).mul(float(0.6))));
    }
    default: return null;
  }
}
