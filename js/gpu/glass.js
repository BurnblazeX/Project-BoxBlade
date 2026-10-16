import { Fn, If, Loop, Break, float, vec3, uniform, texture3D, max, texture, abs, vec2, vec4, min, clamp, length, int, select, pow } from 'three/tsl';
import { GRID_DIM, HIT_EPS, MIN_STEP, MAX_TRACE_STEPS } from '../core/boxgrid.js';
import { BLOCK_METRES } from '../core/world.js';
import { blockFaceTSL as boxFaceTSL, texelLockTSL } from './terrain.js';
import { BLOCK_TEXELS } from '../core/materials.js';
import { layout, memoCascades } from './layout.js';
import { decodeFieldTSL, eachCascade, ringUV } from './field.js';
import { axisFaceTSL, blockFaceTSL, blockLayerTSL } from './volume.js';
import { labNormalTSL } from './normals.js';

// --- Light through glass ---
//
// Glass does not block a shadow ray; it tints it, by how far the ray travels
// inside it (Beer-Lambert, as the view through glass is tinted): tint^(L / 1.5)
// with the tint being glass's through one block, times what the two surfaces
// reflect away: (1 - F)^2, F the Fresnel at the angle the ray meets the entry
// face (materials.js glassSurfaceTransmit) - the same F the glare off that face
// takes, so what glass reflects and what it lets through add up to what
// arrived. (It was the head-on (1 - 0.04)^2 at every angle, which let a low sun
// through a window at 92% while the window also threw half of it as glare.) The
// faces of a block are parallel, so the exit meets the same angle. The ray does
// not bend: the shadow lands where it would if light went straight, the
// standard trade.
//
// The tint is the colour of the texel the ray ENTERED the glass through - so
// stained glass throws its pattern in coloured light, lead lines and all, and
// clear glass its faint cast. One texture read, only for a ray that met glass.
export const GLASS_SURFACE_LOSS = (1 - 0.04) * (1 - 0.04);   // head-on, for reference

// f0: the entry texel's F0 (a dielectric's grey); cosI: |cos| of the angle
// between the ray and the entry face's normal.
export const glassSurfaceTransmitTSL = (f0, cosI) => {
  const F = f0.add(float(1).sub(f0).mul(pow(float(1).sub(clamp(cosI, float(0), float(1))), float(5))));
  const T = float(1).sub(F);
  return T.mul(T);
};

export const glassTransmitTSL = (len, tint, surface = float(GLASS_SURFACE_LOSS)) =>
  pow(max(tint, vec3(1e-3, 1e-3, 1e-3)), vec3(len, len, len).div(float(BLOCK_METRES)))
    .mul(surface);

// What a shadow ray from origin along dir let through, given the glass march's
// answer (traced: z the metres in glass, w how far along the ray it entered).
// terrain: render.js's terrain textures - the entry texel's albedo is the tint.
//
// --- Caustics ---
//
// Light through glass is bent by its surface, and where the normals converge
// it is focused - the bright web under water or a glass brick. Traced properly
// that is a photon problem; the classic cheap form is the DIVERGENCE of the
// surface normals: where the tangent-space normal field converges (a lens-like
// dip), light gathers and brightens; where it spreads (a bump), it thins. And
// the further the receiver is from the glass, the more the bending has moved
// the light, so the effect scales with that distance too:
//   intensity = 1 + strength * distance * div(n.xy)
// read from the entry texel's normal map and its four neighbours on the same
// face. Flat glass has no divergence, so no caustic - as real flat glass casts
// none; textured glass (frosted, a glass-brick ripple) casts its pattern.
// glassCaustics is its strength; 0 skips the reads.
export const glassCaustics = uniform(float(1));

// Glare setting 2: glass's second reflection, off its back face (1 on). The
// light enters (1 - F), reflects off the back (F), and leaves again (1 - F),
// tinted by the glass over the slab crossed twice, and displaced along the face
// by the refraction - a fainter, coloured copy of the front glare beside it.
// A uniform, so the setting needs no rebuild. Only glass rectangles carry a
// thickness (mirrors.js), so nothing else pays for it.
export const glassBackGlare = uniform(float(0));

export const GLASS_CAUSTIC_RANGE = [0.25, 3];   // clamp on the intensity

export function glassRayTransmitTSL(terrain, traced, origin, dir) {
  const out = vec3(1, 1, 1).toVar();
  If(traced.z.greaterThan(float(1e-4)).and(traced.w.greaterThanEqual(float(0))), () => {
    // The march stops within a step past the surface it entered through; the
    // face is the nearest block plane, and the texel is locked onto it as the
    // glass's own shading locks its texels.
    const entry = origin.add(dir.mul(traced.w)).toVar();
    const nf = blockFaceTSL(entry, dir, false, terrain).toVar();
    const f = boxFaceTSL(texelLockTSL(entry, nf), nf);
    const layer = blockLayerTSL(terrain, f.block, nf, entry).toVar();
    const st = f.st.toVar();
    const tint = texture(terrain.albedo, st).depth(int(layer)).level(int(0)).rgb;
    // The two surfaces' reflection at the angle this ray meets them.
    const f0 = texture(terrain.specular, st).depth(int(layer)).level(int(0)).g;
    const surface = glassSurfaceTransmitTSL(f0, abs(dir.dot(nf)));
    out.assign(glassTransmitTSL(traced.z, tint, surface));
    If(glassCaustics.greaterThan(float(0)), () => {
      const du = float(1 / BLOCK_TEXELS);
      const nAt = o => labNormalTSL(texture(terrain.normal, st.add(o)).depth(int(layer)).level(int(0)));
      const div = nAt(vec2(du, 0)).x.sub(nAt(vec2(du.negate(), 0)).x)
        .add(nAt(vec2(0, du)).y.sub(nAt(vec2(0, du.negate())).y)).mul(float(0.5));
      const k = float(1).add(glassCaustics.mul(traced.w).mul(div));
      out.mulAssign(clamp(k, float(GLASS_CAUSTIC_RANGE[0]), float(GLASS_CAUSTIC_RANGE[1])));
    });
  });
  return out;
}

// --- Through glass: the march inside it ---
//
// From a point inside glass, along dir, until the ray either reaches the glass
// surface (the exit) or meets something opaque first (what is seen through the
// glass - under the cube, most rays end in the grass it stands on). One texture
// read a step gives both: R is the opaque distance, G the glass one (negative
// inside, so -G is how far the exit is at least). The step is the smaller.
//
// Returns vec4(normal, w):
//   w in [0, GLASS_OPAQUE_TAG)   exit at distance w; normal is the glass face's,
//                                pointing OUT (G increases outward)
//   w >= GLASS_OPAQUE_TAG        an opaque surface at w - GLASS_OPAQUE_TAG;
//                                normal is that surface's
//   w < 0                        neither within maxDist, or the field ran out
// Normals are the field gradient snapped to its axis, as the reflect trace's.
export const GLASS_OPAQUE_TAG = 10000;

export const GLASS_MAX_DISTANCE = 12;   // metres inside glass, at most

// The block walk through glass (glassNode): steps in all, and internal
// reflections allowed before a ray counts as trapped.
export const GLASS_WALK_STEPS = 32;

export const GLASS_MAX_BOUNCES = 4;

// Panes a view ray refracts through: 2 sees one more glass through the first.
export const GLASS_LAYERS = 2;

// Dispersion, when on: red and blue refract at ior * (1 -/+ this), green at ior.
// Stylised - real crown glass spreads about half a percent.
export const GLASS_DISPERSION = 0.02;

export function createGlassMarchTSL(cascades) {
  return memoCascades(cascades, 'glassMarch', () => Fn(([origin0, dir0, maxDist]) => {
    const origin = origin0.toVar(), dir = dir0.toVar();
    const dim = float(GRID_DIM);
    const t = float(0).toVar();
    const done = float(0).toVar();
    const out = vec4(0, 0, 0, -1).toVar();
    for (const { c } of eachCascade(cascades, float(0))) {
      If(done.equal(float(0)), () => {
        Loop({ start: 0, end: MAX_TRACE_STEPS, type: 'int', condition: '<' }, () => {
          const vp = origin.add(dir.mul(t)).sub(c.origin).div(c.voxel).toVar();
          If(vp.x.lessThan(float(0)).or(vp.y.lessThan(float(0))).or(vp.z.lessThan(float(0)))
             .or(vp.x.greaterThanEqual(dim)).or(vp.y.greaterThanEqual(dim))
             .or(vp.z.greaterThanEqual(dim)), () => { Break(); });
          const s = texture3D(c.tex, ringUV(vp, c.ring)).toVar();
          const dR = decodeFieldTSL(s.r, c.range).toVar();
          const dG = decodeFieldTSL(s.g, c.range).toVar();
          const grad = ch => {
            const at = q => decodeFieldTSL(ch(texture3D(c.tex, ringUV(q, c.ring))), c.range);
            return vec3(at(vp.add(vec3(1, 0, 0))).sub(at(vp.sub(vec3(1, 0, 0)))),
                        at(vp.add(vec3(0, 1, 0))).sub(at(vp.sub(vec3(0, 1, 0)))),
                        at(vp.add(vec3(0, 0, 1))).sub(at(vp.sub(vec3(0, 0, 1)))));
          };
          If(dR.lessThan(float(HIT_EPS)), () => {
            const g = grad(x => x.r).toVar();
            g.assign(select(length(g).lessThan(float(1e-4)), dir.negate(), g));
            out.assign(vec4(axisFaceTSL(g), t.add(float(GLASS_OPAQUE_TAG))));
            done.assign(float(1));
            Break();
          });
          If(dG.greaterThan(float(-HIT_EPS)), () => {
            const g = grad(x => x.g).toVar();
            g.assign(select(length(g).lessThan(float(1e-4)), dir, g));
            out.assign(vec4(axisFaceTSL(g), t));
            done.assign(float(1));
            Break();
          });
          t.addAssign(max(min(dG.negate(), dR), float(MIN_STEP)).mul(c.voxel));
          If(t.greaterThan(maxDist), () => { done.assign(float(1)); Break(); });
        });
      });
    }
    return out;
  }).setLayout(layout('glassMarch', 'vec4', [
    { name: 'origin0', type: 'vec3' }, { name: 'dir0', type: 'vec3' },
    { name: 'maxDist', type: 'float' }])));
}
