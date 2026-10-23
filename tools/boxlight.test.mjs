import { file, section, ok, truthy, falsy, near } from './lib/harness.mjs';
import { World, slopeLevel, BLOCK_METRES } from '../js/core/world.js';
import { BOX_METRES } from '../js/core/boxgrid.js';
import { boxLightRegion, buildBoxMasks, buildBoxCells, boxSolid, boxKind, boxRay, boxSun, boxSky,
         boxLampAmount, boxCellCentre, packCell, unpackCell, boxBit, glassBoxTint, skyDir, SKY_RAYS,
         BOX_BLOCK_U32 } from '../js/core/boxlight.js';

file('boxlight.test.mjs - basic mode light on the BoxGrid');

// A 12 x 12 texture in three columns, red | green | blue (top-down sRGB).
const stripes = new Uint8Array(12 * 12 * 4);
for (let y = 0; y < 12; y++) for (let x = 0; x < 12; x++) {
  const o = (y * 12 + x) * 4, c = Math.floor(x / 4);
  stripes.set([c === 0 ? 255 : 0, c === 1 ? 255 : 0, c === 2 ? 255 : 0, 255], o);
}
const glassPixels = id => (id === 'stainedGlass' ? stripes : null);

// A floor of grass blocks x, z in -4..4 at y 0, then whatever a test adds.
function world(extra = []) {
  World.clear();
  for (let x = -4; x <= 4; x++) for (let z = -4; z <= 4; z++) World.set(`${x},0,${z}`, { materialId: 'grass' });
  for (const [k, b] of extra) World.set(k, b);
  const region = boxLightRegion();
  const masks = buildBoxMasks(region, World, glassPixels);
  const { cells, top } = buildBoxCells(region, masks);
  return { region, masks, cells, top };
}
// The local box just above block (x, 0, z)'s top face, in its middle column.
const above = (r, x, z) => [3 * (x - r.lo[0]) + 1, 3 * (0 - r.lo[1]) + 3, 3 * (z - r.lo[2]) + 1];
const blockAt = (r, x, y, z) => (((z - r.lo[2]) * r.size[1] + (y - r.lo[1])) * r.size[0] + (x - r.lo[0])) * BOX_BLOCK_U32;

section('region and masks');
{
  const { region, masks, cells, top } = world();
  ok('one block of margin round the world', JSON.stringify([region.lo[0], region.size[0]]), JSON.stringify([-5, 11]));
  ok('box0 is the lowest box of block lo', region.box0[0], 3 * -5 - 1);
  ok('a full block fills all 27 boxes', masks.data[blockAt(region, 0, 0, 0)], 2 ** 27 - 1);
  ok('cells pack and unpack', JSON.stringify(unpackCell(packCell(287, 38, 5))), JSON.stringify([287, 38, 5]));
  // 9 x 9 blocks of floor: 27 x 27 boxes over it and 27 x 27 under it, plus
  // the four sides' 27 x 3 each.
  ok('surface cells: over, under and round the floor', cells.length, 27 * 27 * 2 + 4 * 27 * 3);
  ok('top is one over the floor', top, 3 * (0 - region.lo[1]) + 3);
  const c = above(region, 0, 0);
  truthy('the box above the floor is air', !boxSolid(region, masks, ...c));
  truthy('the box below it is solid', boxSolid(region, masks, c[0], c[1] - 1, c[2]));
}

section('half blocks and glass');
{
  const { region, masks, cells } = world([['0,1,0', { materialId: 'grass', shape: 'halfBottom' }],
                                          ['2,1,0', { materialId: 'glass' }]]);
  const c = above(region, 0, 0);
  truthy('a bottom half: its lowest box layer is solid', boxSolid(region, masks, ...c));
  falsy('its middle layer is air (a box solid when its centre is)', boxSolid(region, masks, c[0], c[1] + 1, c[2]));
  const g = above(region, 2, 0);
  ok('glass boxes are glass, not solid', boxKind(region, masks, ...g), 2);
  ok('bit order: x, then y, then z', boxBit(1, 2, 0), 7);
  truthy('the floor under the glass is a surface cell', cells.includes(packCell(...g)));
}

section('slopes and models');
{
  // A slope rising north on block (0, 1, 0): over every point of its plane,
  // the box just above the plane is open, so the plane has a cell to read.
  const { region, masks, cells } = world([['0,1,0', { materialId: 'grass', shape: 'slope', facing: 'N' }]]);
  const block = World.get('0,1,0');
  let open = 0, all = 0;
  for (let fx = 0.05; fx < 1; fx += 0.1) for (let fz = 0.05; fz < 1; fz += 0.1) {
    const y = (1 - 0.5 + slopeLevel(block, fx, fz)) * BLOCK_METRES + 0.05;
    const b = [(fx - 0.5) * BLOCK_METRES, y, (fz - 0.5) * BLOCK_METRES].map((v, a) => Math.round(v / BOX_METRES) - region.box0[a]);
    all++;
    if (boxKind(region, masks, ...b) === 0 && cells.includes(packCell(...b))) open++;
  }
  ok('every texel over a slope plane has an open cell in front', open, all);
}
{
  // A chair seat filling block (0, 1, 0)'s lower third, and a thin leg off a
  // box centre in block (2, 1, 0) (world texels, 12 a block).
  World.clear();
  for (let x = -4; x <= 4; x++) for (let z = -4; z <= 4; z++) World.set(`${x},0,${z}`, { materialId: 'grass' });
  const region = boxLightRegion();
  const masks = buildBoxMasks(region, World, () => null, [[-6, 6, -6, 6, 10, 6], [26, 6, 2, 27, 18, 3]]);
  const { cells } = buildBoxCells(region, masks);
  const local = (x, y, z) => [x - region.box0[0], y - region.box0[1], z - region.box0[2]];
  truthy('a box whose centre is inside a model cube is solid', boxSolid(region, masks, ...local(0, 2, 0)));
  falsy('the box above the seat is open', boxSolid(region, masks, ...local(0, 3, 0)));
  truthy('and a surface cell', cells.includes(packCell(...local(0, 3, 0))));
  const leg = local(7, 2, 1);
  falsy('a thin leg off every box centre casts nothing', boxSolid(region, masks, ...leg));
  truthy('but the box it touches is a cell', cells.includes(packCell(...leg)));
  truthy('and so are its neighbours', cells.includes(packCell(leg[0], leg[1] + 2, leg[2])));
}

section('stained glass tints');
{
  const t = (i, k, alongX) => glassBoxTint(stripes, i, 1, k, alongX).map(v => +v.toFixed(3)).join();
  ok('box column 0 takes the left third', t(0, 1, false), '1,0,0');
  ok('box column 2 the right third', t(2, 1, false), '0,0,1');
  ok('crossed along x, the column is box k', t(0, 2, true), '0,0,1');
  // A wall of stained glass along x at z = 2 (y 1..2), sun from +z low down.
  const wall = [];
  for (let x = -4; x <= 4; x++) for (let y = 1; y <= 2; y++) wall.push([`${x},${y},2`, { materialId: 'stainedGlass' }]);
  const { region, masks, top } = world(wall);
  const sun = [0, Math.sin(0.5), Math.cos(0.5)];
  const lit = c => boxSun(region, masks, top, c, sun).map(v => +v.toFixed(3)).join();
  // Behind the wall (z = 1), the sun comes through the box column it crosses.
  const base = above(region, 0, 1);
  ok('through the left third of a pane: red', lit([base[0] - 1, base[1], base[2]]), '1,0,0');
  ok('through the middle: green', lit(base), '0,1,0');
  ok('in front of the wall: untinted', lit(above(region, 0, 3)), '1,1,1');
  const shade = boxSun(region, masks, top, above(region, 0, 1), [0, Math.sin(0.5), -Math.cos(0.5)]);
  ok('the sun from the other side: no glass in the way', shade.join(), '1,1,1');
}

section('the sun');
{
  // A pillar two blocks high on block (0, 1..2, 0).
  const { region, masks, top } = world([['0,1,0', { materialId: 'grass' }], ['0,2,0', { materialId: 'grass' }]]);
  const sunOf = (c, d) => boxSun(region, masks, top, c, d)[0];
  ok('high sun: the floor beside the pillar is lit', sunOf(above(region, 1, 0), [0, 1, 0]), 1);
  ok('below the horizon: dark', sunOf(above(region, 1, 0), [0.6, -0.1, 0.8]), 0);
  // Low sun from +x: the floor on the -x side is in the pillar's shadow.
  const low = [Math.cos(0.4), Math.sin(0.4), 0];
  ok('low sun: shadow on the far side', sunOf(above(region, -1, 0), low), 0);
  ok('low sun: the near side is lit', sunOf(above(region, 1, 0), low), 1);
}

section('the sky');
{
  ok('the sky rays are unit', [...Array(SKY_RAYS).keys()].every(k => Math.abs(Math.hypot(...skyDir(k)) - 1) < 1e-9), true);
  {
    const flat = world();
    ok('open floor: the whole sky', boxSky(flat.region, flat.masks, flat.top, above(flat.region, 0, 0)), 1);
  }
  // A room: walls round x, z in -2..2 (y 1..2) with a roof at y 3, and a door
  // at (0, 1..2, -2).
  const room = [];
  for (let x = -2; x <= 2; x++) for (let z = -2; z <= 2; z++) {
    room.push([`${x},3,${z}`, { materialId: 'grass' }]);
    if (Math.abs(x) === 2 || Math.abs(z) === 2) for (let y = 1; y <= 2; y++) if (!(x === 0 && z === -2)) room.push([`${x},${y},${z}`, { materialId: 'grass' }]);
  }
  const { region, masks, top } = world(room);
  const sky = c => boxSky(region, masks, top, c);
  const yard = sky(above(region, 4, 4));
  truthy('two blocks off the room: the low rays toward it meet its roof', yard > 0.8 && yard < 1);
  ok('inside the room, away from the door: none', sky(above(region, 1, 1)), 0);
  const nearWall = sky(above(region, 3, 0));
  truthy('beside the room\'s outer wall: some, not all', nearWall > 0.5 && nearWall < 1);
}

section('lamps');
{
  const wall = [];
  for (let z = -4; z <= 4; z++) for (let y = 1; y <= 3; y++) wall.push([`0,${y},${z}`, { materialId: 'grass' }]);
  const { region, masks } = world(wall);
  const at = c => { const p = boxCellCentre(region, c); return { x: p[0], y: p[1], z: p[2] }; };
  const amount = (c, l) => boxLampAmount(region, masks, c, l)[0];
  // A level-10 lamp a box over the floor of block (2, 0, 0).
  const lampCell = above(region, 2, 0);
  const lamp = { level: 10, position: at([lampCell[0], lampCell[1] + 1, lampCell[2]]) };
  near('a cell beside the lamp is lit by nearly its peak',
       amount(lampCell, lamp), (Math.log2(1 + (1 - BOX_METRES / 10)) * 10 / 15 - 0.01) / 0.99, 1e-6);
  truthy('the floor on its own side is lit', amount(above(region, 1, 0), lamp) > 0);
  ok('behind the wall: nothing', amount(above(region, -1, 0), lamp), 0);
  ok('past its reach: nothing', amount(above(region, 4, 4), { level: 2, position: lamp.position }), 0);
  // A lamp inside a solid box still lights round it: its own box is the goal.
  const inWall = { level: 10, position: at([3 * (0 - region.lo[0]) + 2, lampCell[1] + 1, lampCell[2]]) };
  truthy('a lamp set in the wall face lights the floor in front', amount(above(region, 1, 0), inWall) > 0);
  ok('a clear ray', boxRay(region, masks, above(region, 1, 0), [0, 0, 1], 6).join(), '1,1,1');
  ok('a ray into the wall', boxRay(region, masks, above(region, 1, 0), [-1, 0, 0], 6).join(), '0,0,0');
}
World.clear();
