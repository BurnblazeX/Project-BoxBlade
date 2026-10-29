import { file, section, ok } from './lib/harness.mjs';
import { worldBlockBounds, windowAxis, followWindow } from '../js/core/follow.js';
import { boxLightRegion, BOX_LIGHT_MAX_BLOCKS } from '../js/core/boxlight.js';

file('follow.test.mjs - windows that follow the player (World plan P2a)');

section('one axis');
{
  ok('a world that fits: centred on it', windowAxis(null, 500, 128, 0, 44), Math.round(22 - 63.5));
  ok('... wherever the focus is', windowAxis(-42, -900, 128, 0, 44), Math.round(22 - 63.5));
  // A 512-block world.
  const lo = -256, hi = 255;
  const first = windowAxis(null, 10, 128, lo, hi);
  ok('a bigger world: around the focus', first <= 10 && 10 < first + 128, true);
  ok('... on the stride', first % 8, 0);
  ok('walking in the middle keeps it', windowAxis(first, 10 + 20, 128, lo, hi), first);
  const moved = windowAxis(first, first + 128 - 10, 128, lo, hi);
  ok('leaving the middle half moves it', moved !== first, true);
  ok('... back around the focus', moved <= first + 118 && first + 118 < moved + 128, true);
  ok('never past the world at its low edge', windowAxis(null, lo + 3, 128, lo, hi), lo);
  ok('nor its high one', windowAxis(null, hi - 3, 128, lo, hi), hi - 127);
  // test_b with its pad: 142 across. Panning over it end to end and back
  // moves the window twice each way at most, and it always covers the focus.
  let at = null, moves = 0, covered = true;
  const pan = [...Array(140).keys()];
  for (const f of [...pan, ...[...pan].reverse(), ...pan]) {
    const next = windowAxis(at, f, 128, -1, 140);
    if (at !== null && next !== at) moves++;
    at = next;
    covered = covered && at <= f && f < at + 128 && at >= -1 && at + 127 <= 140;
  }
  ok('a world a little wider than the window: two places, one move a crossing', moves, 3);
  ok('... always over the focus, inside the world', covered, true);
}

section('both axes, from a world');
{
  const world = new Map([['-9,0,0', {}], ['35,37,44', {}]]);
  const b = worldBlockBounds(world);
  ok('bounds', `${b.min.join()} / ${b.max.join()}`, '-9,0,0 / 35,37,44');
  ok('an empty world has none', worldBlockBounds(new Map()), null);
  const dev = followWindow(null, { x: 10, z: 10 }, 128, b, { pad: 1 });
  ok('the Dev Level fits a 128 window: it never moves', followWindow(dev, { x: 400, z: -300 }, 128, b, { pad: 1 }).join(),
     dev.join());
}

section('the box light: a window over a world wider than it');
{
  const wide = new Map();
  for (let x = -200; x <= 200; x += 4) wide.set(`${x},0,0`, {});
  const a = boxLightRegion(wide, [], { x: 0, z: 0 });
  ok('clipped to its window', `${a.clipped} ${a.size[0]}`, `true ${BOX_LIGHT_MAX_BLOCKS}`);
  ok('around the focus', a.lo[0] <= 0 && 0 < a.lo[0] + a.size[0], true);
  const b = boxLightRegion(wide, [], { x: 150, z: 0 }, [a.lo[0], a.lo[2]]);
  ok('following the player there', b.lo[0] <= 150 && 150 < b.lo[0] + b.size[0], true);
  ok('the same dims, so the kernels stay', b.dims.join(), a.dims.join());
  const small = boxLightRegion(new Map([['0,0,0', {}], ['10,2,10', {}]]), [], { x: 900, z: 900 });
  ok('a world that fits: fitted as before, wherever the player is', `${small.clipped} ${small.size[0]}`, 'false 13');
}
