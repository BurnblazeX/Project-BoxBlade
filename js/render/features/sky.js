// --- The sky (core/sky.js) ---
//
// An authored gradient on a clock, projected to L2 SH whenever the hour or the
// sun moves, and uploaded as one uniform. The ambient reads it at each texel's
// bent normal, the background and reflection misses at their direction.
// The hour is the world clock's (core/clock.js). bxb.time(h) sets it and moves
// the sun along its arc; a running clock (bxb.daycycle) carries the sun along
// as the hour passes. bxb.light still places the sun freely, and the sky's sun
// glow follows wherever it is.
//
// ctx: scene (its background is the sky), sunLight (the plain path's
// DirectionalLight: its position is the sun's direction, its colour is set
// here), clock (the world clock; without one, a stopped clock of its own at
// the default hour), moveSun(x, y, z) (the sun feature's way of moving it).
import { positionWorldDirection } from 'three/tsl';
import { skyPalette, skyRadiance, projectSH, skyAmbientGain, sunForHour, sunLight,
         DEFAULT_SKY_HOUR } from '../../core/sky.js';
import { createSkyBindings, writeSkyBindings, skyShTSL, sunColourUniform } from '../../gpu/sky.js';
import { createWorldClock } from '../../core/clock.js';

function stoppedClock() {
  const c = createWorldClock(DEFAULT_SKY_HOUR);
  c.running = false;
  return c;
}

export function createSky({ scene, sunLight: light, clock = stoppedClock(), moveSun }) {
  const bindings = createSkyBindings();
  const gain = skyAmbientGain();
  let hour = clock.hour();
  let lastKey = '';

  // Re-projected only when the hour or the sun's direction changed. alpha: the
  // tick clock's, for the hour between ticks.
  function update(alpha = 0) {
    const h = clock.hour(alpha);
    // A running clock carries the sun with the hour, at the same resolution as
    // the re-projection (0.001 h: 3.6 game seconds, invisible). A stopped one
    // leaves it where bxb.time or bxb.light put it.
    if (clock.running && h.toFixed(3) !== hour.toFixed(3)) moveSun(...sunForHour(h));
    hour = h;
    const d = light.position.clone().normalize();
    const key = `${hour.toFixed(3)}|${d.x.toFixed(4)},${d.y.toFixed(4)},${d.z.toFixed(4)}`;
    if (key === lastKey) return;
    lastKey = key;
    const pal = skyPalette(hour);
    const sunDir = [d.x, d.y, d.z];
    writeSkyBindings(bindings, projectSH(dir => skyRadiance(pal, dir, sunDir)), gain);
    // The sun's own light, from the same keys. The plain (unshadowed) path's
    // DirectionalLight follows it too, at its own intensity.
    const [r, g, b] = sunLight(pal, sunDir);
    sunColourUniform.value.setRGB(r, g, b);
    light.color.setRGB(r, g, b);
  }
  update();
  scene.backgroundNode = skyShTSL(bindings, positionWorldDirection);

  const commands = {
    time: {
      help: 'time of day in hours: sets the world clock and the sky palette, and moves the sun ' +
            'along its arc. bxb.time(h, false) leaves the sun',
      usage: 'bxb.time(17.5)  |  bxb.time(17.5, false)',
      run: (h = 12, moveSun_ = true) => {
        clock.setHour(h);
        hour = clock.hour();
        if (moveSun_) {
          const [x, y, z] = sunForHour(hour);
          moveSun(x, y, z);
        }
        update();
        return `time ${hour.toFixed(2)}h${moveSun_ ? ', sun moved' : ''}`;
      }
    },
    daycycle: {
      help: 'game time passes: a 48-minute day, the sun following the hour. Off, the clock ' +
            'stands still, battle rounds included',
      usage: 'bxb.daycycle()  |  bxb.daycycle(true)',
      run: (on = !clock.running) => {
        clock.running = !!on;
        return `day cycle ${clock.running ? 'on' : 'off'}, ${clock.hour().toFixed(2)}h`;
      }
    }
  };

  return {
    name: 'sky',
    bindings,
    // Scales irradiance so the noon sky lights an up-facing surface at the ambient.
    gain,
    get hour() { return hour; },
    options: () => ({ sky: bindings }),
    update,
    commands,
    settings: { title: 'Sky', settings: [
      { key: 'time', label: 'Time of day (h)', type: 'range', min: 0, max: 24, step: 0.25,
        help: 'palette on the clock; moves the sun along its arc',
        get: () => hour, set: v => commands.time.run(v) },
      { key: 'daycycle', label: 'Day cycle', type: 'toggle',
        help: 'game time passes, 48 real minutes a day',
        get: () => clock.running, set: v => commands.daycycle.run(v) }
    ]}
  };
}
