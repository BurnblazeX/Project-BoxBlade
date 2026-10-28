import { file, section, ok, near, truthy } from './lib/harness.mjs';
import { sunMapBounds, sunMapFrame, sunMapTexelMetres, sunMapStale, farOcclusion,
         SUN_PENUMBRA_CAP, SUN_MAP_REDRAW, SUN_MAP_MIN_REACH, sunMapGrid, SUN_MAP_TAP_SPACING,
         SUN_MAP_MAX_SIDE, sunMapMoved, sunMapTexelFor, sunMapSizeFor, SUN_MAP_WINDOW,
         SUN_MAP_HARD_TEXEL } from '../js/core/sunmap.js';
import { coneRadius, SUN_ANGULAR_SIZE } from '../js/core/sun.js';
import { BLOCK_METRES } from '../js/core/world.js';

file('sunmap.test.mjs - the far half of the sun shadow');

const solid = { solid: true };
const world = new Map([['0,0,0', solid], ['10,30,-4', solid], ['-6,-8,2', solid]]);

section('bounds');
{
  const b = sunMapBounds(world, []);
  ok('blocks span +-0.75 m about their centres', b.min.join(), [-9.75, -12.75, -6.75].join());
  ok('... to the far corner', b.max.join(), [15.75, 45.75, 3.75].join());
  const withModel = sunMapBounds(world, [[0, 0, 0, 12, 400, 12]]);
  ok('a model cube reaching higher raises the top', withModel.max[1], 400 * BLOCK_METRES / 12);
  ok('an empty world has none', sunMapBounds(new Map(), []), null);
}

section('the frame looks down the sun over every corner');
{
  const b = sunMapBounds(world, []);
  for (const [label, sun] of [['low sun', { x: 1, y: 0.35, z: 0.4 }], ['overhead', { x: 0, y: 1, z: 0 }],
                              ['from -z', { x: 0, y: 0.5, z: -1 }]]) {
    const f = sunMapFrame(b, sun);
    const dot = (a, c) => a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
    let inside = true, depthOk = true, aboveBase = true;
    for (let i = 0; i < 8; i++) {
      const c = [0, 1, 2].map(a => ((i >> a) & 1 ? b.max[a] : b.min[a]));
      const d = c.map((v, a) => v - f.centre[a]);
      if (Math.abs(dot(d, f.right)) > f.half || Math.abs(dot(d, f.up)) > f.half) inside = false;
      const fromEye = f.eye - dot(d, f.dir);
      if (fromEye < f.near || fromEye > f.far) depthOk = false;
      if (dot(c, f.dir) - f.base <= 0) aboveBase = false;
    }
    ok(`${label}: every corner is on the map`, inside, true);
    ok(`${label}: and within its depth range`, depthOk, true);
    ok(`${label}: every stored s is above 0, the cleared value`, aboveBase, true);
    near(`${label}: right, up and the sun are a right-handed basis`,
         Math.abs(dot(f.right, f.up)) + Math.abs(dot(f.right, f.dir)) + Math.abs(dot(f.up, f.dir)), 0, 1e-9);
  }
  const f = sunMapFrame(b, { x: 1, y: 0.35, z: 0.4 });
  near('a texel is the square over the size', sunMapTexelMetres(f, 1024), 2 * f.half / 1024, 1e-12);
}

section('redraws');
{
  ok('never drawn: stale', sunMapStale(null, { x: 0, y: 1, z: 0 }), true);
  ok('the same sun: not stale', sunMapStale([0, 1, 0], { x: 0, y: 2, z: 0 }), false);
  const turned = a => ({ x: Math.sin(a), y: Math.cos(a), z: 0 });
  ok('a hair under the threshold: not stale', sunMapStale([0, 1, 0], turned(SUN_MAP_REDRAW * 0.9)), false);
  ok('past it: stale', sunMapStale([0, 1, 0], turned(SUN_MAP_REDRAW * 1.1)), true);
}

section('the march and the map hand over without a gap');
{
  // The march's occlusion fades out over fadeStart..1 of its reach; the map's
  // fades in over the same span, so a blocker anywhere is counted in full once.
  const reach = 12, fadeStart = 0.97;
  ok('nearer than the fade: the march has it, the map ignores it', farOcclusion(5, reach, fadeStart), 0);
  ok('past the reach: the map has it in full', farOcclusion(40, reach, fadeStart), 1);
  near('mid-fade: half each', farOcclusion(reach * (fadeStart + 1) / 2, reach, fadeStart), 0.5, 1e-9);
  const R = coneRadius(SUN_ANGULAR_SIZE);
  truthy('the penumbra cap is past what the 12 m march reaches, so Original is unchanged',
         R * 12 < SUN_PENUMBRA_CAP);
  truthy('... and under the field\'s 1 m range, so open air never reads as shadowed', SUN_PENUMBRA_CAP < 1);
  truthy('the nearest a map blocker counts is past the softening reach, so no surface shadows itself',
         SUN_MAP_MIN_REACH > SUN_PENUMBRA_CAP);
}

section('the softening grid never spreads its taps thin');
{
  for (const maxSide of [4, 6, SUN_MAP_MAX_SIDE]) {
    let ok1 = true, ok2 = true;
    for (let r = 0.1; r < 20; r += 0.37) {
      const g = sunMapGrid(r, maxSide);
      if ((2 * g.radius) / g.side > SUN_MAP_TAP_SPACING + 1e-9) ok1 = false;
      if (g.side > maxSide || g.radius > r + 1e-9) ok2 = false;
    }
    ok(`up to ${maxSide} a side: taps never further apart than ${SUN_MAP_TAP_SPACING} texels`, ok1, true);
    ok(`... within the preset's side, never wider than asked`, ok2, true);
  }
  ok('a penumbra the grid covers keeps its width', sunMapGrid(2, 4).radius, 2);
  ok('one too wide narrows rather than spreading the taps', sunMapGrid(10, 4).radius, 4 * SUN_MAP_TAP_SPACING / 2);
}

section('a big world: the map follows the player (World plan P2a)');
{
  const dot = (a, c) => a[0] * c[0] + a[1] * c[1] + a[2] * c[2];
  // A 512-block region, 768 m across.
  const big = { min: [-384, -12, -384], max: [384, 60, 384] };
  const sun = { x: 1, y: 0.4, z: 0.3 };
  const texel = sunMapTexelFor(4);
  const at = { x: 100, y: 3, z: -50 };
  const f = sunMapFrame(big, sun, at, SUN_MAP_WINDOW, texel);
  ok('it follows', f.follows, true);
  ok('over the window, not the region', f.half, SUN_MAP_WINDOW);
  const p = [at.x, at.y, at.z];
  ok('centred on the player, within half a texel', Math.abs(dot(p, f.right) - f.plane[0]) <= texel / 2 + 1e-9 &&
     Math.abs(dot(p, f.up) - f.plane[1]) <= texel / 2 + 1e-9, true);
  ok('the centre sits on whole texels, so the grid stays put on the world',
     [f.plane[0] / texel, f.plane[1] / texel].every(v => Math.abs(v - Math.round(v)) < 1e-9), true);
  const fitted = sunMapFrame(big, sun);
  ok('its depth range still spans the whole world', `${f.eye},${f.far},${f.base}`, `${fitted.eye},${fitted.far},${fitted.base}`);
  ok('a step is not a redraw', sunMapMoved(f, { x: 102, y: 3, z: -50 }), false);
  ok('leaving the middle of the window is', sunMapMoved(f, { x: 100 + SUN_MAP_WINDOW, y: 3, z: -50 }), true);
  const small = sunMapFrame({ min: [-30, 0, -30], max: [30, 20, 30] }, sun, at, SUN_MAP_WINDOW, texel);
  ok('a world that fits keeps a fitted map', small.follows, false);
  ok('... which never asks to move', sunMapMoved(small, { x: 1e4, y: 0, z: 1e4 }), false);
}

section('the texel follows the softening grid');
{
  for (const side of [4, 6, 8]) {
    near(`${side} a side: the grid reaches the full penumbra`,
         sunMapGrid(SUN_PENUMBRA_CAP / sunMapTexelFor(side), side).radius * sunMapTexelFor(side), SUN_PENUMBRA_CAP, 1e-9);
  }
  ok('hard: the texel lock', sunMapTexelFor(0), SUN_MAP_HARD_TEXEL);
  ok('the size covers the window', sunMapSizeFor({ half: SUN_MAP_WINDOW }, 0.25), Math.ceil(2 * SUN_MAP_WINDOW / 0.25));
}
