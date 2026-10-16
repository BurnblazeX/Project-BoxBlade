import * as THREE from 'three/webgpu';
import { Fn, float, uniform, floor, texture, uv, abs, attribute, varying, vec2, vec4, ivec2, normalize, clamp, int, select, dFdx, dFdy, textureSize, modelWorldMatrix, materialColor, frontFacing, sign, textureLoad } from 'three/tsl';
import { labNormalTSL } from './normals.js';

// --- Sprite shading: the texel lock, the normal, the lit materials ---

// The sprite equivalent of texelLockTSL: snap to the centre of the SPRITE texel
// the fragment falls in, so a shadow crossing a character lands in whole art
// pixels. The quad is a plane, so p is linear in uv, and the screen derivatives
// of both give dp/du and dp/dv exactly. Computed before any branch, which is
// where derivatives are defined.
export const spriteTexelLockTSL = Fn(([p, st, size]) => {
  const dpx = dFdx(p).toVar(), dpy = dFdy(p).toVar();
  const dux = dFdx(st).toVar(), duy = dFdy(st).toVar();
  const det = dux.x.mul(duy.y).sub(dux.y.mul(duy.x));
  const inv = float(1).div(select(abs(det).greaterThan(float(1e-20)), det, float(1e-20)));
  const dPdu = dpx.mul(duy.y).sub(dpy.mul(dux.y)).mul(inv);
  const dPdv = dpy.mul(dux.x).sub(dpx.mul(duy.x)).mul(inv);
  const d = st.mul(size).floor().add(float(0.5)).div(size).sub(st);
  return p.add(dPdu.mul(d.x)).add(dPdv.mul(d.y));
});

// A sprite's material, lit: its own colour and alpha, times the light at it.
// A sprite's shading normal, in world space. The quad's own axes are the
// tangent frame - x is its width, y its height, z the way it faces - read from
// the model matrix, so it follows the billboard yaw and lean. Two sign flips:
// a mirrored sprite (map.repeat.x = -1) runs u backwards across the quad, and a
// back face faces the other way. No normal map: the quad's own facing.
// Returns { n, ao }: the world normal and the texture's baked AO (LabPBR blue).
// The quad's frame comes from its vertices' own normal and tangent, passed
// flat - the exact attribute values, not interpolated - so a plain quad
// (tangent +x, normal +z) computes precisely what reading the model matrix's
// axes did, and a tree's four planes can share ONE mesh and one draw, each
// plane still facing its own way (sprites.js crossedPlanesGeometry). Every
// sprite geometry must carry a tangent (sprites.js addSpriteTangent).
export function spriteShadeNormalTSL(map, normalTex, front = frontFacing) {
  const tan = varying(attribute('tangent', 'vec4').xyz, 'vSpriteTangent')
    .setInterpolation(THREE.InterpolationSamplingType.FLAT);
  const nrm = varying(attribute('normal', 'vec3'), 'vSpriteNormal')
    .setInterpolation(THREE.InterpolationSamplingType.FLAT);
  const right = normalize(modelWorldMatrix.mul(vec4(tan, 0)).xyz);
  const up = normalize(modelWorldMatrix.mul(vec4(0, 1, 0, 0)).xyz);
  const fwd = normalize(modelWorldMatrix.mul(vec4(nrm, 0)).xyz);
  const face = select(front, fwd, fwd.negate());
  if (!normalTex) return { n: face, ao: float(1) };
  // Live references to the map's own repeat and offset, so flipping the sprite
  // in updateSpriteFacing flips its normals in the same frame.
  const repeat = uniform(map.repeat), offset = uniform(map.offset);
  const t = labNormalTSL(texture(normalTex, uv().mul(repeat).add(offset))).toVar();
  return { n: normalize(right.mul(t.x.mul(sign(repeat.x)))
                        .add(up.mul(t.y)).add(face.mul(t.z))),
           ao: t.w };
}

// atlas: the sprite texel atlas ({ tex, size }), or null to shade every pixel.
// With it, the light is read back from the texel's atlas slot, filled this
// frame by the atlas pass (createSpriteAtlasMaterial) - the same value, shaded
// once per texel instead of once per pixel.
export function createSpriteShadowMaterial(base, spriteLight, normalTex = null, atlas = null) {
  const mat = new THREE.MeshBasicNodeMaterial({
    map: base.map, transparent: base.transparent,
    alphaTest: base.alphaTest, side: base.side,
    // Carried because the character clip fix flips these on the live material.
    depthTest: base.depthTest, depthWrite: base.depthWrite
  });
  const base4 = materialColor.toVar();
  if (atlas) {
    const size = vec2(textureSize(texture(base.map), int(0)));
    const plane = varying(attribute('spritePlane', 'float'), 'vSpritePlane')
      .setInterpolation(THREE.InterpolationSamplingType.FLAT);
    const t = clamp(floor(uv().mul(size)), vec2(0, 0), size.sub(float(1)));
    const slot = spriteAtlasOriginTSL()
      .add(vec2(plane.mul(size.x), select(frontFacing, float(0), size.y))).add(t);
    mat.colorNode = base4.mul(vec4(textureLoad(atlas.tex, ivec2(slot)).rgb, 1));
    return mat;
  }
  const drawn = base4.a.greaterThanEqual(float(base.alphaTest || 0));
  mat.colorNode = base4.mul(vec4(spriteLight(base.map, drawn, normalTex), 1));
  return mat;
}

// The sprite's atlas region, per object: the mesh and its atlas twin carry
// the same Vector2 (main.js ensureSpriteAtlas). In the object's own uniform
// group, which already carries its matrix, so it adds no buffer write.
const spriteAtlasOriginTSL = () => uniform(new THREE.Vector2())
  .onObjectUpdate(({ object }) => object.userData.spriteAtlasOrigin);

// The atlas pass's material, drawn on each sprite mesh's twin
// (sprites.js spriteAtlasGeometry): every plane and side laid flat on the
// mesh's region, one fragment per texel, each running the sprite shading at
// that texel's centre. The world position and frame still come from the real
// vertices, so what it computes is what the sprite's own pixels did - only the
// clip position is the atlas's.
export function createSpriteAtlasMaterial(base, spriteLight, normalTex, atlas) {
  const mat = new THREE.MeshBasicNodeMaterial({
    map: base.map, side: THREE.DoubleSide, depthTest: false, depthWrite: false
  });
  // The art's size as a uniform rather than read off the texture: every
  // sprite's atlas material then compiles to the same code, one pipeline for
  // all of them (a texture query added a binding of its own for some sprites
  // and not others, and a pipeline per sprite). main.js only builds the atlas
  // once the art has loaded.
  const img = base.map.image;
  const size = uniform(new THREE.Vector2(img.width, img.height));
  const side = attribute('spriteSide', 'float');
  const px = spriteAtlasOriginTSL()
    .add(vec2(attribute('spritePlane', 'float').mul(size.x), side.mul(size.y)))
    .add(uv().mul(size));
  // WebGPU clip space: y up, with row 0 of the target at the top.
  mat.vertexNode = vec4(px.x.div(atlas.size.x).mul(2).sub(1),
                        float(1).sub(px.y.div(atlas.size.y).mul(2)), 0, 1);
  const front = varying(side, 'vSpriteSide')
    .setInterpolation(THREE.InterpolationSamplingType.FLAT).lessThan(float(0.5));
  const drawn = materialColor.a.greaterThanEqual(float(base.alphaTest || 0));
  // outputNode, as the texel cache's: the stored light is raw, not tone mapped.
  mat.outputNode = vec4(spriteLight(base.map, drawn, normalTex, front, size), 1);
  return mat;
}
