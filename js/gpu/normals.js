import * as THREE from 'three/webgpu';
import { Fn, float, vec3, max, abs, vec4, normalize, cross, select, dFdx, dFdy, inverseSqrt, sqrt } from 'three/tsl';

// --- Normal maps: LabPBR decode and the tangent frames they are applied in ---

// A tangent-space normal map applied to a surface with no stored tangents -
// the terrain's boxes have none. The frame comes from screen derivatives of
// position and uv (Schuler's cotangent frame), which for a flat face is exact.
// Derivatives, so call it before any branch.
// A LabPBR normal-map texel to vec4(x, y, z, ao): x right, y UP (the stored
// green is DirectX, down), z rebuilt from the other two, ao from blue.
export const labNormalTSL = Fn(([c]) => {
  const x = c.r.mul(2).sub(1), y = float(1).sub(c.g.mul(2));
  const z = sqrt(max(float(1).sub(x.mul(x)).sub(y.mul(y)), float(0)));
  return vec4(x, y, z, c.b);
});

// t is the tangent-space normal from labNormalTSL, y up.
export const bumpedNormalTSL = Fn(([n, pos, st, t]) => {
  const dp1 = dFdx(pos).toVar(), dp2 = dFdy(pos).toVar();
  const du1 = dFdx(st).toVar(), du2 = dFdy(st).toVar();
  const dp2perp = cross(dp2, n), dp1perp = cross(n, dp1);
  const T = dp2perp.mul(du1.x).add(dp1perp.mul(du2.x)).toVar();
  const B = dp2perp.mul(du1.y).add(dp1perp.mul(du2.y)).toVar();
  const s = inverseSqrt(max(max(T.dot(T), B.dot(B)), float(1e-20)));
  return normalize(T.mul(s).mul(t.x).add(B.mul(s).mul(t.y)).add(n.mul(t.z)));
});

// The same, for the terrain, with the frame read off the face instead of
// reconstructed from derivatives. The terrain is axis-aligned boxes, so every
// face's tangent (the way u runs) and bitangent (the way v runs) are known -
// they are BoxGeometry's own uv layout, the table hitSurfaceTSL uses.
//
// Why not the derivative frame: bumpedNormalTSL scales T and B by the LONGER
// of the two, so on a face seen at an angle - where u and v shrink on screen
// by different amounts - the bent normal leans by a different amount along
// each axis, and that changes with the view. Reflections, which follow the
// normal, squished with the camera; and the derivatives are per 2x2 pixel
// quad, so a texel's pixels did not even agree. This frame is exact and has
// no view in it: one normal per texel, from any angle.
//
//   +x: T -z, B +y    -x: T +z, B +y
//   +y: T +x, B -z    -y: T +x, B +z
//   +z: T +x, B +y    -z: T -x, B +y
export const boxBumpedNormalTSL = Fn(([n, t]) => {
  const ax = abs(n.x).greaterThan(float(0.5));
  const ay = abs(n.y).greaterThan(float(0.5));
  const T = select(ax, vec3(0, 0, n.x.negate()),
                   select(ay, vec3(1, 0, 0), vec3(n.z, 0, 0)));
  const B = select(ay, vec3(0, 0, n.y.negate()), vec3(0, 1, 0));
  return normalize(T.mul(t.x).add(B.mul(t.y)).add(n.mul(t.z)));
});

// Normal maps are data, not colour: no sRGB, nearest so each art texel has one
// normal, repeat because block textures tile.
export function createNormalTexture(data, w, h) {
  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = tex.magFilter = THREE.NearestFilter;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}
