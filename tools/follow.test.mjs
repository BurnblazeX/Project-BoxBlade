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
