// --- Fog: a froxel volume (doc §10) ---
//
// A camera-frustum volume of FROXELS.x x .y x .z cells, the depth slices
// spaced exponentially from FOG_NEAR to FOG_FAR, rebuilt every frame by two
// compute kernels, then laid over the frame by a fullscreen quad (makeQuad):
//
//   scatter    per froxel: the fog's density at its centre - height fog
//              times a soft value noise on the texel grid - and the light it
//              scatters toward the camera. The light is the LPV's (gpu/gi.js):
//              its fog kernel lights each 0.75 m air cell with the sun's
//              transmittance (shadowed, tinted by glass, roof-tested) and the
//              light list (shadowed, from every side); the LPV itself gives
//              the bounce and the sky visibility. Density fine, light coarse.
//   integrate  per froxel column: front to back, the light scattered so far
//              and the transmittance, at every slice (Hillaire's energy-
//              conserving step: each slice's light integrated over its own
//              extinction).
//
// Light through the fog scatters more than once - a lit bank of fog glows
// deep inside, past where single scattering would reach. That is the
// octave approximation (Wrenninge): octave i scatters a^i as much, sees b^i
// of the optical depth toward the sun and a phase c^i as peaked. The depth
// toward the sun is the height fog's, analytic: the column above the point,
// over the sun's elevation. ms 0 is single scattering only.
//
// Units are the shading's: a fog cell of unit density lit by the full sun
// scatters back as much as a white face lit head on.
import * as THREE from 'three/webgpu';
import { Fn, If, Loop, float, vec2, vec3, vec4, uvec3, int, uniform, instanceIndex, textureStore,
         texture3D, exp, exp2, pow, log2, max, min, clamp, mix, smoothstep, floor, fract, sin, select, length,
         normalize, abs, ivec2, textureLoad, screenUV, screenCoordinate, positionGeometry } from 'three/tsl';
import { skyShTSL, sunColourUniform } from './sky.js';

export const FROXELS = { x: 160, y: 90, z: 64 };

export const FOG_NEAR = 1;     // metres of view depth: nothing nearer

export const FOG_FAR = 72;     // C2's reach; the far slice holds everything past it

export const DEFAULT_FOG = {
  density: 0.015,    // extinction per metre at and below the base height
  base: 3,           // metres (world y): full density below
  falloff: 4,        // metres for the density to fall by e above the base
  noise: 0.6,        // 0 smooth, 1 the noise's full swing
  noiseSize: 0.125,  // metres: the noise's finest cells - a texel
  wind: 0.25,        // metres a second the noise drifts
  albedo: 0.9,       // the share of extinction that scatters
  g: 0.5,            // forward scattering of the sun (Henyey-Greenstein)
  ms: 1,             // multiple scattering: 0 single, 1 the full octaves
  lamps: 1,          // the light list's share
  sky: 1             // the sky's and the bounce's share
};

const MS_OCTAVES = 3;
const MS = { a: 0.5, b: 0.5, c: 0.5 };

const hash3 = c => fract(sin(c.dot(vec3(127.1, 311.7, 74.7))).mul(float(43758.5453)));

// Value noise on the unit lattice, smoothstepped between the corners - soft,
// one value a lattice cell. 0..1. (gpu/smoke.js too.)
export function valueNoise(q) {
  const i = floor(q).toVar();
  const f = fract(q).toVar();
  const s = f.mul(f).mul(vec3(3, 3, 3).sub(f.mul(float(2)))).toVar();
  const at = (x, y, z) => hash3(i.add(vec3(x, y, z)));
  const x00 = mix(at(0, 0, 0), at(1, 0, 0), s.x), x10 = mix(at(0, 1, 0), at(1, 1, 0), s.x);
  const x01 = mix(at(0, 0, 1), at(1, 0, 1), s.x), x11 = mix(at(0, 1, 1), at(1, 1, 1), s.x);
  return mix(mix(x00, x10, s.y), mix(x01, x11, s.y), s.z);
}

// Henyey-Greenstein, times 4 pi: isotropic is 1. g the forward share, c the
// cosine between the ray and the light's direction. (gpu/smoke.js too.)
export const hg = (g, c) => {
  const d = float(1).add(g.mul(g)).sub(g.mul(c).mul(float(2)));
  return float(1).sub(g.mul(g)).div(max(pow(d, float(1.5)), float(1e-4)));
};

// gi: the LPV's bindings (render/features/gi.js lpvBinding: levels, each
// with its fogSunTex and fogLampTex, strength). sky: the sky bindings.
// ambient: the shading's ambient uniform.
export function createFroxelFog({ gi, sky, ambient }) {
  const { x: X, y: Y, z: Z } = FROXELS;
  const make = () => {
    const t = new THREE.Storage3DTexture(X, Y, Z);
    t.type = THREE.HalfFloatType;
    t.format = THREE.RGBAFormat;
    t.generateMipmaps = false;
    t.minFilter = t.magFilter = THREE.LinearFilter;
    t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
    return t;
  };
  // (light scattered, extinction) per froxel; (light so far, transmittance).
  const scatterTex = make(), integTex = make();

  const u = {
    projInv: uniform(new THREE.Matrix4()),
    camWorld: uniform(new THREE.Matrix4()),
    camPos: uniform(new THREE.Vector3()),
    sunDir: uniform(new THREE.Vector3(0, 1, 0)),
    sunLight: uniform(float(1)),     // the sun's strength in the diffuse's units (0 when off)
    time: uniform(float(0)),
    density: uniform(float(DEFAULT_FOG.density)),
    base: uniform(float(DEFAULT_FOG.base)),
    falloff: uniform(float(DEFAULT_FOG.falloff)),
    noise: uniform(float(DEFAULT_FOG.noise)),
    noiseSize: uniform(float(DEFAULT_FOG.noiseSize)),
    wind: uniform(float(DEFAULT_FOG.wind)),
    albedo: uniform(float(DEFAULT_FOG.albedo)),
    g: uniform(float(DEFAULT_FOG.g)),
    ms: uniform(float(DEFAULT_FOG.ms)),
    lamps: uniform(float(DEFAULT_FOG.lamps)),
    sky: uniform(float(DEFAULT_FOG.sky))
  };

  const logSpan = Math.log2(FOG_FAR / FOG_NEAR);
  // View depth at slice boundary s (0..Z).
  const depthAt = s => float(FOG_NEAR).mul(exp2(s.div(float(Z)).mul(float(logSpan))));
  // A froxel column's ray: the far-plane point in view space, through which
  // the view ray runs; k, metres of ray a metre of view depth.
  const column = (x, y) => {
    const uv = vec2(float(x), float(y)).add(float(0.5)).div(vec2(X, Y));
    const ndc = vec2(uv.x.mul(float(2)).sub(float(1)), float(1).sub(uv.y.mul(float(2))));
    const f = u.projInv.mul(vec4(ndc, float(1), float(1))).toVar();
    const far = f.xyz.div(f.w).toVar();
    return { far, k: length(far).div(far.z.negate()).toVar() };
  };

  // The LPV volumes at p, finest over coarsest, each faded out over its edge
  // cells (as lpvSampleTSL): the sun's transmittance (1 past them - open),
  // the lamps' light and the bounce (0 past them) and the sky visibility (1).
  const lightAt = p => {
    let sunT = vec3(1, 1, 1), lamp = vec3(0, 0, 0), bounce = vec3(0, 0, 0), vis = float(1);
    for (let i = gi.levels.length - 1; i >= 0; i--) {
      const lv = gi.levels[i];
      const uvw = p.sub(lv.origin).div(float(lv.extent)).toVar();
      const edge = min(min(min(uvw.x, uvw.y), uvw.z),
                       min(min(float(1).sub(uvw.x), float(1).sub(uvw.y)), float(1).sub(uvw.z))).mul(float(lv.dim));
      const w = smoothstep(float(0.5), float(2.5), edge).toVar();
      const b = texture3D(lv.tex, uvw).level(float(0)).toVar();
      sunT = mix(sunT, texture3D(lv.fogSunTex, uvw).level(float(0)).rgb, w);
      lamp = mix(lamp, texture3D(lv.fogLampTex, uvw).level(float(0)).rgb, w);
      bounce = mix(bounce, b.rgb, w);
      vis = mix(vis, b.a, w);
    }
    return { sunT, lamp, bounce: bounce.mul(gi.strength), vis };
  };

  const skyIso = () => skyShTSL(sky, vec3(0, 1, 0), true).add(skyShTSL(sky, vec3(0, -1, 0), true))
    .mul(float(0.5)).mul(ambient).mul(sky.gain);

  const scatter = Fn(() => {
    const i = int(instanceIndex);
    const x = i.mod(int(X)), y = i.div(int(X)).mod(int(Y)), z = i.div(int(X * Y));
    const { far } = column(x, y);
    const d = depthAt(float(z).add(float(0.5)));
    const pW = u.camWorld.mul(vec4(far.mul(d.div(far.z.negate())), float(1))).xyz.toVar();
    const ray = normalize(pW.sub(u.camPos)).toVar();
    // Height fog, full below the base.
    const above = max(pW.y.sub(u.base), float(0)).toVar();
    const height = exp(above.negate().div(max(u.falloff, float(0.01))));
    // Soft noise: a texel lattice and one four times coarser, drifting.
    const q = pW.add(vec3(u.wind.mul(u.time), 0, u.wind.mul(u.time).mul(float(0.6)))).div(u.noiseSize).toVar();
    const n = valueNoise(q).mul(float(0.35)).add(valueNoise(q.div(float(4))).mul(float(0.65)));
    const sigma = u.density.mul(height).mul(max(mix(float(1), n.mul(float(2)), u.noise), float(0))).toVar();
    const S = vec3(0, 0, 0).toVar();
    If(sigma.greaterThan(float(1e-5)), () => {
      const L = lightAt(pW);
      // The height fog's optical depth to the sun: the column above, over
      // the sun's elevation.
      const above0 = select(pW.y.lessThan(u.base), u.base.sub(pW.y).add(u.falloff), u.falloff.mul(height));
      const tau = u.density.mul(above0).div(max(u.sunDir.y, float(0.05))).toVar();
      const cosT = ray.dot(u.sunDir).toVar();
      const sunMS = float(0).toVar();
      for (let o = 0; o < MS_OCTAVES; o++) {
        const w = o === 0 ? float(1) : u.ms.mul(float(MS.a ** o));
        sunMS.addAssign(w.mul(exp(tau.mul(float(-(MS.b ** o))))).mul(hg(u.g.mul(float(MS.c ** o)), cosT)));
      }
      const sunIn = sunColourUniform.mul(u.sunLight).mul(L.sunT).mul(sunMS);
      // The isotropic light gains from the octaves too: what scatters more
      // than once has lost its direction.
      const isoMS = float(1).add(u.ms.mul(float(MS.a + MS.a * MS.a)));
      const iso = L.lamp.mul(u.lamps).add(skyIso().mul(L.vis).add(L.bounce).mul(u.sky)).mul(isoMS);
      S.assign(sunIn.add(iso).mul(sigma).mul(u.albedo));
    });
    textureStore(scatterTex, uvec3(x, y, z), vec4(S, sigma));
  })().compute(X * Y * Z).setName('fogScatter');

  const integrate = Fn(() => {
    const i = int(instanceIndex);
    const x = i.mod(int(X)), y = i.div(int(X));
    const { k } = column(x, y);
    const acc = vec3(0, 0, 0).toVar(), T = float(1).toVar();
    Loop({ start: int(0), end: int(Z), type: 'int', condition: '<', name: 'fz' }, ({ fz }) => {
      const uvw = vec3(float(x), float(y), float(fz)).add(float(0.5)).div(vec3(X, Y, Z));
      const s = texture3D(scatterTex, uvw).level(float(0)).toVar();
      const len = depthAt(float(fz).add(float(1))).sub(depthAt(float(fz))).mul(k);
      const st = max(s.w, float(1e-6)).toVar();
      const tr = exp(st.mul(len).negate()).toVar();
      acc.addAssign(T.mul(s.rgb.sub(s.rgb.mul(tr)).div(st)));
      T.mulAssign(tr);
      textureStore(integTex, uvec3(x, y, fz), vec4(acc, T));
    });
  })().compute(X * Y).setName('fogIntegrate');

  // The fog over the frame: a fullscreen quad drawn last, blended
  // dst x transmittance + light (ONE, SRC_ALPHA) - no material is touched, so
  // turning it on compiles one pipeline. Each pixel's depth is its texel's
  // locked position where the G-buffer has one (one answer a texel), else the
  // depth buffer's (slopes, glowing cubes, the sky). Sprites are not in the
  // G-buffer: they take the fog of what is behind them. Nearer than half the
  // first slice fades to none. gbuffer: { id, depth } (texel cache).
  const camFwd = uniform(new THREE.Vector3(0, 0, -1));
  u.camFwd = camFwd;
  const makeQuad = gbuffer => {
    const frag = Fn(() => {
      const px = ivec2(screenCoordinate.xy).toVar();
      const g = textureLoad(gbuffer.id, px).toVar();
      const z = textureLoad(gbuffer.depth, px).x;
      const ndc = vec2(screenUV.x.mul(float(2)).sub(float(1)), float(1).sub(screenUV.y.mul(float(2))));
      const v = u.projInv.mul(vec4(ndc, z, float(1))).toVar();
      const dDepth = v.z.div(v.w).negate();
      const dLock = g.xyz.sub(u.camPos).dot(camFwd);
      const d = select(abs(g.w).greaterThan(float(1.5)), dLock, dDepth);
      const s = log2(max(d, float(FOG_NEAR)).div(float(FOG_NEAR))).div(float(logSpan)).mul(float(Z)).toVar();
      const w = clamp(s.sub(float(0.5)).div(float(Z)), float(0.5 / Z), float(1 - 0.5 / Z));
      const f = texture3D(integTex, vec3(screenUV.x, screenUV.y, w)).level(float(0)).toVar();
      return mix(vec4(0, 0, 0, 1), f, clamp(s.mul(float(2)), float(0), float(1)));
    })();
    const mat = new THREE.MeshBasicNodeMaterial();
    mat.vertexNode = vec4(positionGeometry.xy, float(0), float(1));
    mat.fragmentNode = frag;
    mat.transparent = true;
    mat.depthTest = mat.depthWrite = false;
    mat.blending = THREE.CustomBlending;
    mat.blendEquation = THREE.AddEquation;
    mat.blendSrc = THREE.OneFactor;
    mat.blendDst = THREE.SrcAlphaFactor;
    mat.blendSrcAlpha = THREE.ZeroFactor;
    mat.blendDstAlpha = THREE.OneFactor;
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat);
    mesh.frustumCulled = false;
    mesh.renderOrder = 1e9;
    mesh.name = 'fogQuad';
    return mesh;
  };

  return {
    uniforms: u, kernels: [scatter, integrate], makeQuad, scatterTex, integTex,
    dispose() { scatterTex.dispose(); integTex.dispose(); }
  };
}
