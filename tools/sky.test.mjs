import { file, section, ok, near, truthy } from './lib/harness.mjs';
import { skyPalette, skyRadiance, projectSH, evalSH, shBasis, skyAmbientGain, sunLight, DEFAULT_SKY_HOUR,
         sunForHour, SKY_KEYS } from '../js/sky.js';

file('sky.test.mjs - authored gradient, L2 SH');

section('palette on the clock');
{
  const noon = skyPalette(12), key = SKY_KEYS.find(k => k.name === 'noon');
  ok('noon is the noon key', noon.zenith.join(), key.zenith.join());
  ok('24 wraps to 0', skyPalette(24).zenith.join(), skyPalette(0).zenith.join());
  const mid = skyPalette(9.25);   // halfway dawn -> noon
  const dawn = SKY_KEYS.find(k => k.name === 'dawn');
  near('between keys it lerps', mid.zenith[2], (dawn.zenith[2] + key.zenith[2]) / 2, 1e-9);
  truthy('night is darker than noon', skyPalette(2).zenith[2] < noon.zenith[2]);
}

section('projection');
{
  // A uniform sky projects to band 0 alone and evaluates back to itself.
  const sh = projectSH(() => [0.5, 0.25, 1]);
  near('uniform radiance round-trips (quadrature, ~1%)', evalSH(sh, [0, 1, 0])[0], 0.5, 0.01);
  near('  in every direction', evalSH(sh, [0.6, -0.8, 0])[2], 1, 0.02);
  near('uniform irradiance / PI is the radiance', evalSH(sh, [1, 0, 0], true)[1], 0.25, 0.005);
  truthy('higher bands all but vanish', sh.slice(1).every(c => Math.abs(c[0]) < 0.01));

  // The basis is orthonormal over the grid: projecting Y_i gives e_i.
  const i = 6;
  const shY = projectSH(d => { const y = shBasis(d)[i]; return [y, y, y]; });
  near('basis is normalised', shY[i][0], 1, 0.02);
  truthy('and orthogonal', shY.every((c, j) => j === i || Math.abs(c[0]) < 0.02));

  // The noon gradient: an up-facing surface sees more of the zenith than a
  // down-facing one, and the fit stays close to the gradient it came from.
  const pal = skyPalette(12), sun = [0, 1, 0];
  const g = projectSH(d => skyRadiance(pal, d, sun));
  const upE = evalSH(g, [0, 1, 0], true), downE = evalSH(g, [0, -1, 0], true);
  truthy('up and down irradiance differ', Math.abs(upE[2] - downE[2]) > 0.01);
  const zen = evalSH(g, [0, 1, 0]), truth = skyRadiance(pal, [0, 1, 0], sun);
  near('zenith radiance is close to the gradient', zen[2], truth[2], 0.15);
}

section('ambient gain');
near('noon, straight up, is exactly 1x ambient',
     (() => {
       const sh = projectSH(d => skyRadiance(skyPalette(12), d, [0, 1, 0]));
       const e = evalSH(sh, [0, 1, 0], true);
       return (0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2]) * skyAmbientGain();
     })(), 1, 1e-9);

section('sun on the clock');
{
  const dawn = sunForHour(6), noon = sunForHour(12), dusk = sunForHour(18);
  near('rises in the east', dawn[0], 1, 1e-9);
  near('sets in the west', dusk[0], -1, 1e-9);
  truthy('highest at noon', noon[1] > sunForHour(9)[1] && noon[1] > 0.8);
  truthy('below the horizon at night', sunForHour(0)[1] < 0);
}

section('the sun\'s own light');
{
  ok('noon sun is exactly white 1 - noon looks as it always did',
     sunLight(skyPalette(12), [0, 1, 0]).join(), '1,1,1');
  const dusk = sunLight(skyPalette(17.5), sunForHour(17.5));
  truthy('dusk sun is warm', dusk[0] > dusk[2] * 1.5);
  truthy('and dimmer than noon', dusk[1] < 0.8);
  ok('night sun is off', sunLight(skyPalette(2), [0, 1, 0]).join(), '0,0,0');
  ok('a sun below the horizon lights nothing', sunLight(skyPalette(12), [0, -0.5, 0.86]).join(), '0,0,0');
  ok('the day starts at 9', DEFAULT_SKY_HOUR, 9);
}
