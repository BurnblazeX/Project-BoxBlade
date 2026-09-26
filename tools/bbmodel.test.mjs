import { readFileSync } from 'node:fs';
import { file, section, ok, truthy, falsy } from './lib/harness.mjs';
import { parseBBModel, placeModel, facePixel, faceSwapped, transformCube, quarterTurn,
         FACE_DIRS } from '../js/bbmodel.js';

file('bbmodel.test.mjs - Blockbench import and placement');

// The chair is authored art and changes; everything below is derived from
// the file itself, not from its current shape.
const chairSrc = readFileSync(new URL('../assets/objects/decor_chair.bbmodel', import.meta.url), 'utf8');
const chair = parseBBModel(chairSrc);
const chairJson = JSON.parse(chairSrc);
const cubeEls = chairJson.elements.filter(e => (e.type || 'cube') === 'cube');

section('the chair imports cleanly');
ok('its name', chair.name, 'decor_chair');
ok('no warnings', chair.warnings.join(' | '), '');
ok('every cube', chair.cubes.length, cubeEls.length);
ok('every bounding box', chair.bounds.length,
   chairJson.elements.filter(e => e.type === 'bounding_box').length);
ok('its texture is 24 x 24', `${chair.width}x${chair.height}`, '24x24');
truthy('the albedo is embedded', /^data:image\/png;base64,/.test(chair.textures.albedo || ''));
ok('no normal or specular map', `${chair.textures.normal},${chair.textures.specular}`, 'null,null');
truthy('every cube has all six faces', chair.cubes.every(c => c.faces.every(f => f)));

section('each face shows exactly its uv rectangle, one pixel per texel');
{
  // Walk every texel of every face and collect the pixels it shows: they must
  // be the face's own uv rectangle, each pixel once.
  const json = JSON.parse(chairSrc);
  const names = ['east', 'west', 'up', 'down', 'south', 'north'];
  let bad = 0, faces = 0;
  for (const [i, cube] of chair.cubes.entries()) {
    const el = json.elements.filter(e => (e.type || 'cube') === 'cube')[i];
    FACE_DIRS.forEach((n, d) => {
      const f = cube.faces[d];
      const axisN = n.findIndex(x => x !== 0);
      const plane = n[axisN] > 0 ? cube.max[axisN] - 1 : cube.min[axisN];
      const [a, b] = [0, 1, 2].filter(x => x !== axisN);
      const seen = new Set();
      for (let i = cube.min[a]; i < cube.max[a]; i++) for (let j = cube.min[b]; j < cube.max[b]; j++) {
        const w = [0, 0, 0]; w[axisN] = plane; w[a] = i; w[b] = j;
        seen.add(facePixel(f, w).join(','));
      }
      const [u0, v0, u1, v1] = el.faces[names[d]].uv;
      const want = new Set();
      for (let x = Math.min(u0, u1); x < Math.max(u0, u1); x++)
        for (let y = Math.min(v0, v1); y < Math.max(v0, v1); y++) want.add(`${x},${y}`);
      faces++;
      if (seen.size !== want.size || [...seen].some(p => !want.has(p))) bad++;
    });
  }
  ok(`${faces} faces cover their uv rectangles exactly`, bad, 0);
}
{
  // Orientation, Minecraft's convention: a north face seen from the north has
  // its uv's left edge at the +x end and its top row at the top; a top face
  // has its uv origin at the -x, -z corner.
  const seat = chair.cubes[0], el = cubeEls[0];
  const [nu0, nv0, nu1, nv1] = el.faces.north.uv, [tu0, tv0] = el.faces.up.uv;
  const y = seat.max[1] - 1, z = seat.min[2];
  ok('seat north: +x end shows its first u', facePixel(seat.faces[5], [seat.max[0] - 1, y, z])[0], Math.min(nu0, nu1));
  ok('seat north: -x end shows its last u', facePixel(seat.faces[5], [seat.min[0], y, z])[0], Math.max(nu0, nu1) - 1);
  ok('seat north: the top row shows its first v', facePixel(seat.faces[5], [seat.min[0], y, z])[1], Math.min(nv0, nv1));
  ok('seat top: the -x, -z corner shows its uv origin',
     facePixel(seat.faces[2], [seat.min[0], seat.max[1] - 1, seat.min[2]]).join(','), `${tu0},${tv0}`);
}

section('placed in the world');
{
  const at = [60 * 12, 12 * 1 - 6 + 6, 70 * 12];   // any grid point
  const p0 = placeModel(chair, at, 0);
  const seat = chair.cubes[0], placed = p0.cubes[0];
  ok('unturned, the seat is shifted to it', placed.min.join(','), seat.min.map((v, a) => v + at[a]).join(','));
  // A placed face shows the same pixels as the model's, texel for texel.
  const m = [seat.min[0], seat.min[1], seat.min[2]], w = [m[0] + at[0], m[1] + at[1], m[2] + at[2]];
  ok('the same pixel after the shift', facePixel(placed.faces[5], w).join(','),
     facePixel(seat.faces[5], m).join(','));
  // Turned once (+90 about y: x -> -z): the model's north face (-z) now faces
  // west (-x), and shows the same pixels in the same places.
  const p1 = placeModel(chair, at, 1);
  const west = p1.cubes[0].faces[1];
  truthy('turned, the north face faces west', !!west);
  // A model texel on the north face: turned, x -> -z, z -> x, so model
  // (x, y, z) is world texel (at.x + z, at.y + y, at.z - 1 - x).
  const mt = [seat.max[0] - 1, seat.min[1], seat.min[2]];
  const wt = [at[0] + mt[2], at[1] + mt[1], at[2] - 1 - mt[0]];
  ok('the same pixel on the turned face', facePixel(west, wt).join(','),
     facePixel(seat.faces[5], mt).join(','));
  // Top faces can swap their axes under a turn; side faces never do.
  truthy('a turned top face has its axes swapped', faceSwapped(2, p1.cubes[0].faces[2]));
  falsy('unturned, not', faceSwapped(2, p0.cubes[0].faces[2]));
  truthy('side faces never swap', [0, 1, 4, 5].every(d => [0, 1, 2, 3].every(y =>
    placeModel(chair, at, y).cubes.every(c => !c.faces[d] || !faceSwapped(d, c.faces[d])))));
  // Four turns come back to the start.
  const p4 = placeModel(chair, at, 4);
  ok('four turns are none', JSON.stringify(p4.cubes), JSON.stringify(p0.cubes));
  const b0 = chair.bounds[0];
  ok('bounding boxes turn too', p1.bounds[0].min.join(','),
     `${at[0] + b0.min[2]},${at[1] + b0.min[1]},${at[2] - b0.max[0]}`);
}

section('cube rotations');
{
  const cube = (rot, origin = [0, 0, 0]) => ({
    name: 'c', from: [0, 0, 0], to: [2, 1, 3], rotation: rot, origin,
    faces: Object.fromEntries(['north', 'south', 'east', 'west', 'up', 'down'].map((f, i) =>
      [f, { uv: [0, 0, 3, 3], texture: 0 }]))
  });
  const model = els => parseBBModel({ name: 't', textures: [{ name: 't.png', width: 16, height: 16,
    uv_width: 16, uv_height: 16, source: 'data:' }], elements: els });
  const q = model([cube([0, 90, 0])]);
  ok('a quarter turn about y turns the box', `${q.cubes[0].min},${q.cubes[0].max}`, '0,0,-2,3,1,0');
  truthy('a 22.5 turn is refused for now', model([cube([22.5, 0, 0])]).warnings.some(w => /not drawn yet/.test(w)));
  truthy('a 10 degree turn is refused', model([cube([0, 10, 0])]).warnings.some(w => /multiple of 22.5/.test(w)));
  truthy('two axes are refused', model([cube([90, 90, 0])]).warnings.some(w => /more than one axis/.test(w)));
  const off = parseBBModel({ name: 't', elements: [{ name: 'o', from: [0.5, 0, 0], to: [1, 1, 1], faces: {} }] });
  truthy('an off-grid corner is snapped and reported', off.warnings.some(w => /off the unit grid/.test(w)));
  // Transform then its inverse is the identity.
  const T = quarterTurn(1, 1, [3, 0, 5]), U = quarterTurn(1, 3, [3, 0, 5]);
  const c0 = chair.cubes[1];
  ok('a turn and its inverse', JSON.stringify(transformCube(transformCube(c0, T), U)), JSON.stringify(c0));
}

section('a lightSource cube makes a light');
{
  const { lightColourOf } = await import('../js/bbmodel.js');
  const { addWorldModel, clearWorldModels, ModelBoxes, createTestArea: area } = await import('../js/world.js');
  const face = uv => ({ uv, texture: 0 });
  // Sides uv (a column of pixels); top and bottom one pixel of it.
  const allFaces = uv => Object.fromEntries(['north', 'south', 'east', 'west', 'up', 'down'].map(f =>
    [f, face(f === 'up' || f === 'down' ? [uv[0], uv[1], uv[0] + 1, uv[1] + 1] : uv)]));
  const torch = parseBBModel({ name: 'torch', textures: [{ name: 'torch.png', width: 16, height: 16,
    uv_width: 16, uv_height: 16, source: 'data:' }], elements: [
    { name: 'stick', from: [0, 0, 0], to: [1, 6, 1], faces: allFaces([0, 0, 1, 6]) },
    { name: 'lightSource', from: [0, 6, 0], to: [1, 8, 1], faces: allFaces([4, 0, 5, 2]) }] });
  ok('no warnings', torch.warnings.join(' | '), '');
  ok('the flame is the light', torch.cubes.map(c => c.light).join(','), 'false,true');
  ok('its pixel rectangles', JSON.stringify(torch.cubes[1].rects[0]), '[4,0,5,2]');
  area(8, 8);
  const placed = addWorldModel(torch, 3, 0, 3, 0);
  ok('only the stick is baked', ModelBoxes.length, 1);
  ok('one light', placed.lights.length, 1);
  // Cell (3, 0, 3): floor at y 0.75 m; the flame is x 0..1, y 6..8, z 0..1
  // texels from the origin (4.5, 0.75, 4.5) m.
  ok('at the flame\'s centre', placed.lights[0].position.map(v => v.toFixed(4)).join(','),
     [4.5 + 0.0625, 0.75 + 0.875, 4.5 + 0.0625].map(v => v.toFixed(4)).join(','));
  ok('as big as the flame', placed.lights[0].radius, 0.125);
  clearWorldModels();

  // The colour: the flame's pixels (x 4, y 0..1), averaged, brightest channel full.
  const rgba = new Uint8ClampedArray(16 * 16 * 4);
  const put = (x, y, r, g, b, a = 255) => rgba.set([r, g, b, a], (y * 16 + x) * 4);
  put(4, 0, 250, 100, 0); put(4, 1, 125, 50, 0);
  put(0, 0, 0, 0, 255);                           // the stick's: not counted
  ok('averaged and scaled to full', lightColourOf(torch, rgba, 16).toString(16), 'ff6600');
  put(4, 1, 0, 0, 0, 0);                          // transparent: left out
  ok('transparent pixels are left out', lightColourOf(torch, rgba, 16).toString(16), 'ff6600');
  ok('a model without a light has no colour', lightColourOf(chair, rgba, 16), null);
}

section('a light level in the cube name');
{
  const el = name => ({ name, from: [0, 0, 0], to: [1, 1, 1], faces: {} });
  const m = parseBBModel({ name: 'l', elements: [el('lightSource_12'), el('LIGHTSOURCE'), el('lightSource_40'), el('lightSourcey')] });
  ok('lightSource_12 is level 12', m.cubes[0].lightLevel, 12);
  ok('a bare lightSource takes the default', m.cubes[1].light && m.cubes[1].lightLevel, null);
  ok('past 15 is 15', m.cubes[2].lightLevel, 15);
  truthy('and says so', m.warnings.some(w => /past 15/.test(w)));
  falsy('another name is no light', m.cubes[3].light);
  const { addWorldModel, clearWorldModels, createTestArea: area } = await import('../js/world.js');
  area(4, 4);
  ok('the level reaches the placed light', addWorldModel(m, 1, 0, 1).lights[0].level, 12);
  clearWorldModels();
  const torchSrc = readFileSync(new URL('../assets/objects/decor_wallTorch.bbmodel', import.meta.url), 'utf8');
  const torch = parseBBModel(torchSrc);
  ok('the wall torch: no warnings', torch.warnings.join(' | '), '');
  ok('the wall torch: one light, level 12', torch.cubes.filter(c => c.light).map(c => c.lightLevel).join(','), '12');
}

section('wall models hang with their north against the wall');
{
  const { mountWorldModel, clearWorldModels, createTestArea: area, World: W, getVoxelKey: key } =
    await import('../js/world.js');
  const torch = parseBBModel(readFileSync(new URL('../assets/objects/decor_wallTorch.bbmodel', import.meta.url), 'utf8'));
  area(8, 8);
  // A wall block in each direction around cell (4, 1, 4).
  for (const [x, z] of [[4, 3], [5, 4], [4, 5], [3, 4]]) W.set(key(x, 1, z), { solid: true, walkable: true, materialId: 'grass' });
  const touches = (entry, axis, plane) => entry.cubes.some(c => c.min[axis] === plane || c.max[axis] === plane);
  const beyond = (entry, axis, plane, side) => entry.cubes.some(c => side > 0 ? c.max[axis] > plane : c.min[axis] < plane);
  for (const [wall, axis, plane, side] of [['N', 2, 4 * 12 - 6, -1], ['E', 0, 4 * 12 + 6, 1],
                                           ['S', 2, 4 * 12 + 6, 1], ['W', 0, 4 * 12 - 6, -1]]) {
    const e = mountWorldModel(torch, 4, 1, 4, wall);
    truthy(`wall ${wall}: its mount is flush with the wall's face`, touches(e, axis, plane));
    falsy(`wall ${wall}: nothing reaches into the wall`, beyond(e, axis, plane, side));
    ok(`wall ${wall}: filed under the wall block`, `${e.x},${e.y},${e.z}`,
       `${4 + (wall === 'E') - (wall === 'W')},1,${4 + (wall === 'S') - (wall === 'N')}`);
    clearWorldModels();
  }
  ok('no wall, no torch', mountWorldModel(torch, 6, 1, 6, 'N'), null);
  mountWorldModel(torch, 4, 1, 4, 'N');
  falsy('a torch has no bounds, so the tile under it stays free', !!W.get(key(4, 0, 4)).occupant);
  clearWorldModels();
}
