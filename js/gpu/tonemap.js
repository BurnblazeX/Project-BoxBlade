import { Fn, float, vec3, mat3, max, log2, clamp, pow, uniform } from 'three/tsl';
import { layout } from './layout.js';

// --- AgX, with a look ---
//
// three's AgX (nodes/display/ToneMappingFunctions.js agxToneMapping) is the
// base transform: faithful, and flat - low contrast, low saturation, which on
// this game's art reads washed out. This is the same transform with Blender's
// AgX "look" step added: after the sigmoid, in its [0, 1] encoding, a power
// (contrast) and a saturation about the luma. Base is power 1, saturation 1 -
// three's AgX exactly. Punchy is this game's (Blender's is 1.35 and 1.4, which
// was too strong here).
//
// Registered as THREE.CustomToneMapping on the renderer's node library
// (render/features/tonemap.js); the look is two uniforms, so changing it
// rebuilds nothing.
//
// A layout function through gpu/layout.js layout(), built per shader: a raw
// setLayout's text is built once and reused by later shaders with the first
// one's uniform names - rebuilding the output pass (tone mapping off, then on)
// made a shader that read 'object' without declaring it. The look's uniforms
// are parameters as well, read by the inline wrapper.

export const AGX_LOOKS = {
  base: { power: 1, saturation: 1 },
  punchy: { power: 1.1, saturation: 1.05 }
};

const LINEAR_REC2020_TO_LINEAR_SRGB = mat3(vec3(1.6605, -0.1246, -0.0182), vec3(-0.5876, 1.1329, -0.1006),
                                           vec3(-0.0728, -0.0083, 1.1187));
const LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(vec3(0.6274, 0.0691, 0.0164), vec3(0.3293, 0.9195, 0.0880),
                                           vec3(0.0433, 0.0113, 0.8956));
const AGX_INSET = mat3(vec3(0.856627153315983, 0.137318972929847, 0.11189821299995),
                       vec3(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
                       vec3(0.0482516061458583, 0.101439036467562, 0.811302368396859));
const AGX_OUTSET = mat3(vec3(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
                        vec3(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
                        vec3(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405));
const AGX_MIN_EV = -12.47393, AGX_MAX_EV = 4.026069;
const LUMA = vec3(0.2126, 0.7152, 0.0722);

// AgX's sigmoid, as three's agxDefaultContrastApprox: a 6th-order fit.
const contrastApprox = x => {
  const x2 = x.mul(x).toVar();
  const x4 = x2.mul(x2).toVar();
  return float(15.5).mul(x4.mul(x2)).sub(float(40.14).mul(x4.mul(x))).add(float(31.96).mul(x4))
    .sub(float(6.868).mul(x2.mul(x))).add(float(0.4298).mul(x2)).add(float(0.1191).mul(x)).sub(float(0.00232));
};

const agxLookTSL = Fn(([color, exposure, power, saturation]) => {
  const c = vec3(color).mul(exposure).toVar();
  c.assign(AGX_INSET.mul(LINEAR_SRGB_TO_LINEAR_REC2020.mul(c)));
  c.assign(log2(max(c, vec3(1e-10, 1e-10, 1e-10))));
  c.assign(clamp(c.sub(float(AGX_MIN_EV)).div(float(AGX_MAX_EV - AGX_MIN_EV)), 0.0, 1.0));
  c.assign(contrastApprox(c));
  // The look.
  const luma = c.dot(LUMA).toVar();
  c.assign(pow(max(c, vec3(0, 0, 0)), vec3(power, power, power)));
  c.assign(vec3(luma, luma, luma).add(c.sub(vec3(luma, luma, luma)).mul(saturation)));
  c.assign(AGX_OUTSET.mul(c));
  c.assign(pow(max(c, vec3(0, 0, 0)), vec3(2.2, 2.2, 2.2)));
  return clamp(LINEAR_REC2020_TO_LINEAR_SRGB.mul(c), 0.0, 1.0);
}).setLayout(layout('bxbAgXLook', 'vec3', [{ name: 'color', type: 'vec3' }, { name: 'exposure', type: 'float' },
                                            { name: 'power', type: 'float' }, { name: 'saturation', type: 'float' }]));

// The look's uniforms, and the tone mapping function three calls as
// fn(color, exposure) - inline, so each shader declares the uniforms it reads.
export function createAgXLook() {
  const power = uniform(float(AGX_LOOKS.punchy.power)).setName('bxbAgXPower');
  const saturation = uniform(float(AGX_LOOKS.punchy.saturation)).setName('bxbAgXSaturation');
  const fn = Fn(([color, exposure]) => agxLookTSL(color, exposure, power, saturation));
  return { fn, power, saturation };
}
