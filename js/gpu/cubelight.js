// --- The world's light on the cubes of a medium (spell smoke, leaves) ---
//
// Per cube centre p, before the medium itself: the sun (times the sun on and
// its gain), the lamps (through spell smoke's soft copy) and the sky. Shaded
// path: the sun through the shared cone march (glass-tinted, roof-tested),
// then the sun depth map past it, and the sprites' sun cards; the light list;
// the sky by the roof map. Basic mode (world: the box light's walks,
// gpu/boxlight.js): the sun walked to, every lamp walked to and the sky
// straight up, in the basic material's units (irradiance / pi).
//
// gpu/smoke.js and gpu/leaves.js light their cubes by it, once a cube, in
// their light kernels.
import { If, float, vec3 } from 'three/tsl';
import { createConeTraceSunTSL } from './trace.js';
import { createPointLightsTSL } from './pointlights.js';
import { sunRoofTSL } from './sun.js';
import { sunMapFarTSL } from './sunmap.js';
import { glassRayTransmitTSL } from './glass.js';
import { NO_SKIP, createCardsTSL, CARD_SUN_MAX } from './cards.js';
import { skyShTSL, sunColourUniform } from './sky.js';
import { skyOpenTSL } from './volume.js';

// ctx as createSmokeCubes's (gpu/smoke.js); sun: createSunUniforms's; u: the
// cubes' uniforms - sunLight (the sun on, and its share) and sun (its gain).
// Returns { sun(p), lamps(p), sky(p) }.
export function createCubeWorldLight({ cascades, lights, biasUniform, fadeStartUniform, edgeFadeUniform,
                                       ambientUniform, sky, terrain, glass = false, cards = null, lightCards = null,
                                       shared, world = null, sun, u }) {
  const PI_INV = float(1 / Math.PI);
  if (world) {
    return {
      sun: p => world.binding.sunColour.mul(PI_INV).mul(u.sunLight).mul(world.sun(p)).mul(u.sun),
      lamps: p => world.lamps(p).mul(world.binding.lampGain).mul(PI_INV),
      sky: p => world.binding.ambient.mul(PI_INV).mul(world.sky(p))
    };
  }
  const coneTrace = createConeTraceSunTSL(cascades, !!glass);
  const spriteShadow = createCardsTSL(cards);
  // The lamps through the smoke too - its soft copy, as the sun.
  const pointLights = createPointLightsTSL({ cascades, lights, biasUniform, fadeStartUniform,
                                             edgeFadeUniform, cards: lightCards, glass, smoke: shared, smokeSoft: true });
  const sunReach = float(12);
  const skyIso = () => skyShTSL(sky, vec3(0, 1, 0), true).add(skyShTSL(sky, vec3(0, -1, 0), true))
    .mul(float(0.5)).mul(ambientUniform).mul(sky.gain);
  return {
    sun: p => {
      const lit = vec3(0, 0, 0).toVar();
      If(u.sunLight.greaterThan(float(0)), () => {
        const traced = coneTrace(p, sun.dir, sunReach, sun.coneRadius, fadeStartUniform, edgeFadeUniform).toVar();
        const through = glass ? glassRayTransmitTSL(glass, traced, p, sun.dir) : vec3(1, 1, 1);
        // Past the march's 12 m, the long shadows: the sun depth map, hard
        // (gpu/sunmap.js) - as the GI's probes take it. Without it a cloud in
        // a long shadow was lit as if in the open.
        lit.assign(through.mul(traced.x.mul(sunRoofTSL(terrain, p, sun.dir, sunReach))
          .mul(sunMapFarTSL(p, sunReach, fadeStartUniform, sun.coneRadius, float(0)))));
      });
      if (spriteShadow) lit.mulAssign(spriteShadow(p, sun.dir, float(CARD_SUN_MAX), sun.coneRadius, NO_SKIP));
      return sunColourUniform.mul(u.sunLight).mul(lit).mul(u.sun);
    },
    lamps: p => pointLights.withSpecular(p, vec3(0, 0, 0), NO_SKIP, null),
    sky: p => skyIso().mul(terrain && terrain.skyTop ? skyOpenTSL(terrain, p) : float(1))
  };
}
