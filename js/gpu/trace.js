import { Fn, If, Loop, Break, float, vec3, texture3D, max, abs, vec2, vec4, min, clamp, length, select, step, sign } from 'three/tsl';
import { GRID_DIM, VOXEL_METRES, HIT_EPS, MIN_STEP, MAX_TRACE_STEPS } from '../core/boxgrid.js';
import { layout, memoCascades } from './layout.js';
import { decodeFieldTSL, eachCascade, fadeWeightTSL, levelLive, ringUV, seamBlend } from './field.js';
import { inHitBoundsTSL } from './volume.js';

// --- The marches: the sun's cone, a plain shadow ray, a reflection ray ---

// glass: the variant for a world that has glass - the same march, which also
// reads the field's glass distance (G) in the same sample, steps no further
// than the nearer of the two surfaces, and adds up the metres it spends inside
// glass. Returns vec4(visibility, concluded, metres in glass, 0) instead of
// vec2. Built only when glass exists, so a world without any compiles exactly
// what it did.
export function createConeTraceSunTSL(cascades, glass = false) {
  return memoCascades(cascades, glass ? 'coneGlass' : 'cone', () => Fn(([origin0, dir0, maxDist, coneR, fadeStart, edgeFade]) => {
    // Pinned: both are reused across passes and levels - see the cone path in
    // createSoftSunTSL for what an unpinned shared node does here.
    const origin = origin0.toVar(), dir = dir0.toVar();
    return seamBlend(cascades, origin, skipC0 => {
    const dim = float(GRID_DIM);
    // Started one voxel out rather than at zero: at t = 0 the cone has no radius,
    // so d/(R*t) divides by zero, and the surface the ray is leaving is the
    // nearest thing to it by construction.
    const t = float(VOXEL_METRES).toVar();
    const res = float(1).toVar();
    const done = float(0).toVar();
    const inGlass = float(0).toVar();         // metres, glass variant only
    const entered = float(-1).toVar();        // t where it first met glass

    for (const [i, { c, edge }] of eachCascade(cascades, edgeFade).entries()) {
      If(levelLive(i, skipC0, done), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel);
          // Leaving this level is not a result: t carries into the next one, and
          // only the outermost level's exit means open sky.
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => {
            Break();
          });

          // How much this sample may darken anything: 1 well inside the march's
          // limits, falling to 0 as it approaches them, so a shadow dissolves
          // rather than stopping on a line. See fadeWeightTSL.
          const w = fadeWeightTSL(vp, dim, t, maxDist, fadeStart, edge);

          const sample = texture3D(c.tex, ringUV(vp, c.ring)).toVar();
          const d = decodeFieldTSL(sample.r, c.range);
          // A real hit occludes fully, but only in proportion to w - a hit found
          // at the very end of the march is one the next texel along will miss.
          // Nothing later can lower the minimum, so the whole trace ends here.
          If(d.lessThan(float(HIT_EPS)), () => {
            res.assign(min(res, float(1).sub(w)));
            done.assign(float(1));
            Break();
          });

          // d is in voxels of THIS level, t and the cone radius in metres. The
          // occlusion implied is 1 - ratio; weighting that rather than the
          // visibility is what makes w = 0 mean "contributes nothing" rather
          // than "is fully lit".
          const ratio = d.mul(c.voxel).div(coneR.mul(t));
          const occ = float(1).sub(clamp(ratio, float(0), float(1))).mul(w);
          res.assign(min(res, float(1).sub(occ)));

          if (glass) {
            // Inside glass the exit is -G away; outside, the glass is G away.
            // Step no further than the nearer surface of either kind, so the
            // ray cannot jump over glass, and count the metres spent inside.
            const g = decodeFieldTSL(sample.g, c.range).toVar();
            const step = max(min(d, abs(g)), float(MIN_STEP)).mul(c.voxel).toVar();
            If(g.lessThan(float(0)), () => {
              If(entered.lessThan(float(0)), () => { entered.assign(t); });
              inGlass.addAssign(step);
            });
            t.addAssign(step);
          } else {
            t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          }
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }

    // x is the visibility; y is whether this march CONCLUDED - it hit something,
    // or it spent its whole distance budget - as opposed to running off the edge
    // of the outermost cascade, where it learned nothing at all. The caller uses
    // that to decide whether the answer is worth storing. See the atlas kernel.
    // The glass variant adds z, the metres the ray spent inside glass, and w,
    // how far along it first met glass (-1 never) - for the entry texel's tint.
    if (glass) return vec4(clamp(res, float(0), float(1)), done, inGlass, entered);
    return vec2(clamp(res, float(0), float(1)), done);
    }, glass ? () => vec4(0, 0, 0, 0) : undefined);
  }).setLayout(layout(glass ? 'coneTraceGlass' : 'coneTrace', glass ? 'vec4' : 'vec2', [
    { name: 'origin0', type: 'vec3' }, { name: 'dir0', type: 'vec3' },
    { name: 'maxDist', type: 'float' }, { name: 'coneR', type: 'float' },
    { name: 'fadeStart', type: 'float' }, { name: 'edgeFade', type: 'float' }])));
}

// The sampled path's trace. Identical in structure, but returns how much this ray
// should darken - faded at the march's limits - rather than a binary hit.
//
// Separate from traceDistanceTSL because that one is what bxb.parity() diffs
// against the CPU sphere trace, and a hit test that sometimes returns 0.6 is not
// a hit test any more.
export function createShadowTraceTSL(cascades) {
  return memoCascades(cascades, 'shadow', () => Fn(([origin0, dir0, maxDist, fadeStart, edgeFade]) => {
    const origin = origin0.toVar(), dir = dir0.toVar();
    return seamBlend(cascades, origin, skipC0 => {
    const dim = float(GRID_DIM);
    const t = float(0).toVar();
    const occ = float(0).toVar();
    const done = float(0).toVar();

    for (const [i, { c, edge }] of eachCascade(cascades, edgeFade).entries()) {
      If(levelLive(i, skipC0, done), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel);
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => {
            Break();
          });

          const d = decodeFieldTSL(texture3D(c.tex, ringUV(vp, c.ring)).r, c.range);
          If(d.lessThan(float(HIT_EPS)), () => {
            occ.assign(fadeWeightTSL(vp, dim, t, maxDist, fadeStart, edge));
            done.assign(float(1));
            Break();
          });

          t.addAssign(max(d, float(MIN_STEP)).mul(c.voxel));
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }

    return vec2(occ, done);
    });
  }).setLayout(layout('shadowTrace', 'vec2', [
    { name: 'origin0', type: 'vec3' }, { name: 'dir0', type: 'vec3' },
    { name: 'maxDist', type: 'float' }, { name: 'fadeStart', type: 'float' },
    { name: 'edgeFade', type: 'float' }])));
}

// The reflection march: finest level first, carrying t, like the shadow
// traces - but it returns WHERE it stopped. vec4(face normal, t), t < 0 for a
// miss (left the field or ran out of distance). The normal is the field's
// gradient at the hit, snapped to its dominant axis: the terrain is boxes, so
// that is the exact face, and the face is what picks the texture's uv.
// stopAtGlass: the variant for a ray leaving glass that may meet more of it
// (glass seen through glass). It reads the glass distance too, steps no
// further than the nearer surface of either kind, and stops at glass as it
// does at rock - reporting it as w = -2 - t, the normal left zero (the caller
// takes the face from the block grid). Otherwise it is the same trace.
// sumGlass: the variant for texel GI's rays in a world with glass. It walks
// through glass as the plain trace does, but steps no further than the
// nearer surface of either kind and adds up the metres spent inside glass,
// for the tint of whatever it brings back (glass.js glassRayTransmitTSL).
// It returns vec4(face code, t, metres in glass, t where it first met glass
// (-1 never)): the face packed as x + 2y + 4z (giTraceHitTSL unpacks it).
export function createReflectTraceTSL(cascades, stopAtGlass = false, sumGlass = false) {
  const readG = stopAtGlass || sumGlass;
  const key = sumGlass ? 'reflectGlassSum' : stopAtGlass ? 'reflectGlass' : 'reflect';
  return memoCascades(cascades, key, () => Fn(([origin0, dir0, maxDist]) => {
    const origin = origin0.toVar(), dir = dir0.toVar();
    const dim = float(GRID_DIM);
    const t = float(0).toVar();
    const done = float(0).toVar();
    const out = vec4(0, 0, 0, -1).toVar();
    const inGlass = sumGlass ? float(0).toVar() : null;
    const entered = sumGlass ? float(-1).toVar() : null;
    for (const { c } of eachCascade(cascades, float(0))) {
      If(done.equal(float(0)), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel).toVar();
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => { Break(); });
          const at = q => decodeFieldTSL(texture3D(c.tex, ringUV(q, c.ring)).r, c.range);
          const sample = readG ? texture3D(c.tex, ringUV(vp, c.ring)).toVar() : null;
          const d = (readG ? decodeFieldTSL(sample.r, c.range) : at(vp)).toVar();
          const dG = readG ? decodeFieldTSL(sample.g, c.range).toVar() : null;
          if (stopAtGlass) {
            // Only glass the view can show: in battle, glass outside the arena
            // is hidden, as the rock there is (hitBounds) - it is stepped
            // through, not stopped at. Tested half a voxel on, inside the
            // glass, so a pane standing on the arena's edge is judged by its
            // own side of it.
            const shown = inHitBoundsTSL(origin.add(dir.mul(t.add(c.voxel.mul(float(0.5))))));
            If(dG.lessThan(float(HIT_EPS)).and(d.greaterThanEqual(float(HIT_EPS))).and(shown), () => {
              out.assign(vec4(0, 0, 0, float(-2).sub(t)));
              done.assign(float(1));
              Break();
            });
          }
          If(d.lessThan(float(HIT_EPS)), () => {
            const g = vec3(at(vp.add(vec3(1, 0, 0))).sub(at(vp.sub(vec3(1, 0, 0)))),
                           at(vp.add(vec3(0, 1, 0))).sub(at(vp.sub(vec3(0, 1, 0)))),
                           at(vp.add(vec3(0, 0, 1))).sub(at(vp.sub(vec3(0, 0, 1))))).toVar();
            // Inside a solid the gradient can vanish; the ray came from the
            // open side, so the face it met faces back along it.
            g.assign(select(length(g).lessThan(float(1e-4)), dir.negate(), g));
            const ag = abs(g).toVar();
            const nx = ag.x.greaterThanEqual(ag.y).and(ag.x.greaterThanEqual(ag.z));
            const ny = ag.y.greaterThan(ag.x).and(ag.y.greaterThanEqual(ag.z));
            const face = select(nx, vec3(sign(g.x), 0, 0),
                                select(ny, vec3(0, sign(g.y), 0), vec3(0, 0, sign(g.z))));
            out.assign(vec4(face, t));
            done.assign(float(1));
            Break();
          });
          // Glass is never a wall to step on: outside it, it is dG away; inside
          // (hidden glass being crossed), its far side is -dG away.
          const reach = readG ? min(d, abs(dG)) : d;
          const stepM = max(reach, float(MIN_STEP)).mul(c.voxel).toVar();
          if (sumGlass) {
            If(dG.lessThan(float(0)), () => {
              If(entered.lessThan(float(0)), () => { entered.assign(t); });
              inGlass.addAssign(stepM);
            });
          }
          t.addAssign(stepM);
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }
    if (sumGlass) return vec4(out.x.add(out.y.mul(float(2))).add(out.z.mul(float(4))), out.w, inGlass, entered);
    return out;
  }).setLayout(layout(sumGlass ? 'giTraceGlass' : stopAtGlass ? 'reflectTraceGlass' : 'reflectTrace', 'vec4', [
    { name: 'origin0', type: 'vec3' }, { name: 'dir0', type: 'vec3' },
    { name: 'maxDist', type: 'float' }])));
}

// A sumGlass trace's answer as the plain trace gives it - { hit: vec4(face
// normal, t), glass: vec4(0, 0, metres in glass, entry t) for
// glassRayTransmitTSL }.
export function giTraceHitTSL(r) {
  const code = r.x.toVar();
  const ax = k => select(abs(code).equal(float(k)), sign(code), float(0));
  return { hit: vec4(ax(1), ax(2), ax(4), r.y), glass: vec4(0, 0, r.z, r.w) };
}
