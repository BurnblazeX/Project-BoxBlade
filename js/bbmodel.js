// --- Blockbench models (.bbmodel): import and placement ---
//
// A .bbmodel is JSON: cubes with integer corners, a uv rectangle per face, and
// the textures embedded as data URLs. This reads it into BoxBlade's own model
// form, and places a model in the world as axis-aligned boxes - the form every
// system here already understands: the renderer draws each cube as one more
// instance of the terrain box (render.js), the field bakes it as a box
// (boxgrid.js), so shadows, AO, GI, reflections and the texel cache take a
// model with no path of their own.
//
// The conventions it holds a model to (doc §0):
//   1 Blockbench unit = 1 texel = 12.5 cm; 12 units are one block.
//   Corners on the unit grid, so every face lies on a texel boundary - the
//   texel lock's one requirement. The model's origin is the centre of the
//   floor of the cell it stands in, itself on the grid, so a model built on
//   the grid lands on the world's.
//   One texture pixel per unit: each face's uv rectangle is its own size in
//   pixels.
// What breaks them is reported (warnings) with the cube's name, and the
// nearest thing that works is done.
//
// Pure: no three, no DOM. Textures stay data URLs for the renderer to decode.

// Face directions, in this order everywhere: +x, -x, +y, -y, +z, -z.
export const FACE_DIRS = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
const dirIndex = n => FACE_DIRS.findIndex(d => d[0] === n[0] && d[1] === n[1] && d[2] === n[2]);

// Blockbench's faces (north is -z, east +x) and how a face's uv runs over it:
// u right and v down, as seen from outside - Minecraft's convention. u, v: the
// axis each runs along; rev: against the coordinate (index = max - 1 - m).
const BB_FACES = {
  east:  { dir: 0, u: 2, uRev: true,  v: 1, vRev: true },
  west:  { dir: 1, u: 2, uRev: false, v: 1, vRev: true },
  up:    { dir: 2, u: 0, uRev: false, v: 2, vRev: false },
  down:  { dir: 3, u: 0, uRev: false, v: 2, vRev: true },
  south: { dir: 4, u: 0, uRev: false, v: 1, vRev: true },
  north: { dir: 5, u: 0, uRev: true,  v: 1, vRev: true }
};

const EPS = 1e-3;
// lightSource, or lightSource_<level> (0..15, lights.js): the light's level -
// its reach in blocks and its brightness. No level: the caller's default.
export const LIGHT_CUBE = /^lightSource(?:_(\d+))?$/i;
const snapTo = (v, what, warn) => {
  const r = Math.round(v);
  if (Math.abs(v - r) > EPS) warn(`${what} is ${v}, off the unit grid - snapped to ${r}`);
  return r;
};

// A face's pixel map along one axis: pixel = o + d * m, m the texel's integer
// coordinate along `axis` (the texel [m, m + 1)). a0..a1: its uv span in
// pixels, running backwards when mirrored.
function axisMap(a0, a1, lo, hi, rev, axis) {
  const start = a1 >= a0 ? Math.round(a0) : Math.round(a0) - 1;
  const dir = a1 >= a0 ? 1 : -1;
  // index along the face: rev ? (hi - 1 - m) : (m - lo); pixel = start + dir * index.
  return rev ? { axis, o: start + dir * (hi - 1), d: -dir } : { axis, o: start - dir * lo, d: dir };
}

// --- Moving a cube: a signed axis permutation and a shift ---
//
// T = { axisOf: [[b, s] per source axis a], p: [shift per target axis] }:
// target coordinate b = p[b] + s * source coordinate a. A quarter turn and a
// placement are both one of these, so they share this - the boxes, which way
// each face now points, and each face's pixel map, all at once.
export function transformCube(cube, T) {
  const min = [0, 0, 0], max = [0, 0, 0];
  for (let a = 0; a < 3; a++) {
    const [b, s] = T.axisOf[a];
    if (s > 0) { min[b] = T.p[b] + cube.min[a]; max[b] = T.p[b] + cube.max[a]; }
    else { min[b] = T.p[b] - cube.max[a]; max[b] = T.p[b] - cube.min[a]; }
  }
  const faces = FACE_DIRS.map(wn => {
    // The source face that now faces wn: source normal a = s * target normal b.
    const sn = [0, 0, 0];
    for (let a = 0; a < 3; a++) sn[a] = T.axisOf[a][1] * wn[T.axisOf[a][0]];
    const f = cube.faces[dirIndex(sn)];
    if (!f) return null;
    // Source texel m from target texel w: s > 0, m = w - p; s < 0, m = p - 1 - w.
    const conv = ({ axis, o, d }) => {
      const [b, s] = T.axisOf[axis], p = T.p[b];
      return s > 0 ? { axis: b, o: o - d * p, d } : { axis: b, o: o + d * (p - 1), d: -d };
    };
    return { u: conv(f.u), v: conv(f.v) };
  });
  return { name: cube.name, light: cube.light, lightLevel: cube.lightLevel, rects: cube.rects, min, max, faces };
}

// Quarter turns about `axis` through `pivot`, right-handed (+90 about y takes
// x to -z), as a transform.
export function quarterTurn(axis, turns, pivot = [0, 0, 0]) {
  const b = (axis + 1) % 3, c = (axis + 2) % 3;
  const axisOf = [[0, 1], [1, 1], [2, 1]];
  for (let t = 0; t < ((turns % 4) + 4) % 4; t++) {
    // One turn: coordinate b goes to c, and c to -b.
    for (const m of axisOf) {
      if (m[0] === b) m[0] = c;
      else if (m[0] === c) { m[0] = b; m[1] = -m[1]; }
    }
  }
  // Through the pivot: target = pivot + R (source - pivot).
  const p = [0, 0, 0];
  for (let a = 0; a < 3; a++) { const [tb, s] = axisOf[a]; p[tb] = pivot[tb] - s * pivot[a]; }
  return { axisOf, p };
}

// Parse a .bbmodel (object or JSON text). Returns
//   { name, textures: { albedo, normal, specular } (data URLs or null),
//     width, height (the texture's pixels),
//     cubes: [{ name, min: [x,y,z], max: [x,y,z], faces: [6] }],
//     bounds: [{ min, max }] (its bounding_box elements),
//     warnings: [string] }
// Faces in FACE_DIRS order, each { u: {axis, o, d}, v: {axis, o, d} } -
// pixel = [u.o + u.d * m[u.axis], v.o + v.d * m[v.axis]] for model texel m -
// or null where the face shows nothing.
export function parseBBModel(src, fallbackName = 'model') {
  const json = typeof src === 'string' ? JSON.parse(src) : src;
  const warnings = [];
  const warn = msg => warnings.push(msg);
  const name = json.name || fallbackName;

  // Textures: the albedo, and _n / _s / _e (emissive) beside it by name.
  const texs = json.textures || [];
  const base = t => (t.name || '').replace(/\.png$/i, '');
  const albedoIndex = texs.findIndex(t => !/_(n|s|e)$/.test(base(t)));
  const albedo = texs[albedoIndex] || null;
  const companion = suffix => {
    if (!albedo) return null;
    const t = texs.find(x => base(x) === base(albedo) + suffix);
    return t ? t.source : null;
  };
  if (!albedo) warn('no albedo texture');
  const width = albedo ? albedo.width || json.resolution?.width : 0;
  const height = albedo ? albedo.height || json.resolution?.height : 0;
  // UV units to pixels: a texture can be bigger than the uv space.
  const su = albedo && albedo.uv_width ? width / albedo.uv_width
    : json.resolution?.width ? width / json.resolution.width : 1;
  const sv = albedo && albedo.uv_height ? height / albedo.uv_height
    : json.resolution?.height ? height / json.resolution.height : 1;
  if (su !== 1 || sv !== 1) warn(`texture is ${su} x ${sv} pixels per uv unit - one is expected`);

  const cubes = [], bounds = [];
  for (const el of json.elements || []) {
    const label = `"${el.name || el.uuid}"`;
    const type = el.type || 'cube';
    if (type === 'bounding_box') {
      bounds.push({ min: el.from.map(v => snapTo(v, `${label} from`, warn)),
                    max: el.to.map(v => snapTo(v, `${label} to`, warn)) });
      continue;
    }
    if (type !== 'cube') { warn(`${label}: ${type} elements are not supported - skipped`); continue; }
    if (el.export === false) continue;
    if (el.inflate) warn(`${label}: inflate ${el.inflate} is off the texel grid - ignored`);
    const min = el.from.map(v => snapTo(v, `${label} corner`, warn));
    const max = el.to.map(v => snapTo(v, `${label} corner`, warn));

    // Rotation: one axis at most, in steps of 22.5. Quarter turns are baked in
    // below; 22.5 and 45 are agreed but not drawn yet.
    const rot = el.rotation || [0, 0, 0];
    const axes = rot.map((r, a) => [r, a]).filter(([r]) => Math.abs(r) > EPS);
    if (axes.length > 1) { warn(`${label}: rotated on more than one axis - skipped`); continue; }
    let turn = null;
    if (axes.length === 1) {
      const [angle, axis] = axes[0];
      const steps = angle / 22.5;
      if (Math.abs(steps - Math.round(steps)) > EPS) {
        warn(`${label}: rotation ${angle} is not a multiple of 22.5 - skipped`); continue;
      }
      if (Math.round(steps) % 4 !== 0) {
        warn(`${label}: rotation ${angle} is allowed but not drawn yet - skipped`); continue;
      }
      const pivot = (el.origin || [0, 0, 0]).map(v => snapTo(v, `${label} pivot`, warn));
      turn = quarterTurn(axis, Math.round(steps) / 4, pivot);
    }

    const faces = new Array(6).fill(null);
    // The pixel rectangles its faces show [x0, y0, x1, y1] - what a light's
    // colour is averaged over.
    const rects = [];
    for (const [bbName, spec] of Object.entries(BB_FACES)) {
      const face = el.faces && el.faces[bbName];
      if (!face || face.texture === null || face.texture === undefined || !face.uv) continue;
      if (face.rotation) warn(`${label} ${bbName}: uv rotation ${face.rotation} is not supported - ignored`);
      if (albedo && texs[face.texture] !== albedo && Number(face.texture) !== albedoIndex) {
        warn(`${label} ${bbName}: uses a second texture - drawn from the first`);
      }
      const [u0, v0, u1, v1] = [face.uv[0] * su, face.uv[1] * sv, face.uv[2] * su, face.uv[3] * sv];
      const du = max[spec.u] - min[spec.u], dv = max[spec.v] - min[spec.v];
      if (Math.abs(Math.abs(u1 - u0) - du) > EPS || Math.abs(Math.abs(v1 - v0) - dv) > EPS) {
        warn(`${label} ${bbName}: uv is ${Math.abs(u1 - u0)} x ${Math.abs(v1 - v0)} pixels ` +
             `for a ${du} x ${dv} face - drawn one pixel per texel from its corner`);
      }
      faces[spec.dir] = { u: axisMap(u0, u1, min[spec.u], max[spec.u], spec.uRev, spec.u),
                          v: axisMap(v0, v1, min[spec.v], max[spec.v], spec.vRev, spec.v) };
      rects.push([Math.min(u0, u1), Math.min(v0, v1), Math.max(u0, u1), Math.max(v0, v1)].map(Math.round));
    }
    // A cube named lightSource makes the model a light: a point light at its
    // centre, coloured by its own pixels (the _e texture's, else the albedo's).
    // It is drawn unlit and casts no shadow.
    const lightName = LIGHT_CUBE.exec(el.name || '');
    const light = !!lightName;
    let lightLevel = null;
    if (lightName && lightName[1] !== undefined) {
      lightLevel = Math.min(15, Number(lightName[1]));
      if (Number(lightName[1]) > 15) warn(`${label}: light level ${lightName[1]} is past 15 - 15 is used`);
    }
    const cube = { name: el.name || '', light, lightLevel, rects, min, max, faces };
    cubes.push(turn ? transformCube(cube, turn) : cube);
  }

  return {
    name, width, height, cubes, bounds, warnings,
    textures: { albedo: albedo ? albedo.source : null,
                normal: companion('_n'), specular: companion('_s'), emissive: companion('_e') }
  };
}

// --- Placing a model in the world ---
//
// In world texels (world metres / 0.125). The model's origin goes to `at`
// (integers: the centre of its cell's floor) and it turns yaw quarter turns
// about +y. Returns { cubes, bounds } in world texels, the cubes' face maps
// over WORLD texel indices.
export function placeModel(model, at, yaw = 0) {
  const T = quarterTurn(1, yaw);
  for (let b = 0; b < 3; b++) T.p[b] += at[b];
  return {
    cubes: model.cubes.map(c => transformCube(c, T)),
    bounds: model.bounds.map(b => {
      const t = transformCube({ min: b.min, max: b.max, faces: [] }, T);
      return { min: t.min, max: t.max };
    })
  };
}

// The pixel a placed face shows for world texel w. The shader does the same
// (render.js terrainSurfaceTSL).
export function facePixel(face, w) {
  return [face.u.o + face.u.d * w[face.u.axis], face.v.o + face.v.d * w[face.v.axis]];
}

// The in-face axes the shader assumes for a face direction: +-x (z, y),
// +-y (x, z), +-z (x, y). A top or bottom face can have them swapped by a
// yaw turn; a side face never.
export const FACE_AXES = [[2, 1], [2, 1], [0, 2], [0, 2], [0, 1], [0, 1]];
export function faceSwapped(dir, face) {
  return face.u.axis !== FACE_AXES[dir][0];
}

// A model light's colour: the average of its lightSource cubes' own pixels
// (every face's uv rectangle; transparent pixels left out), scaled so its
// brightest channel is full - the light's level sets how bright it is, the
// texture only which colour. As a 0xRRGGBB, sRGB, like every light colour.
export function lightColourOf(model, rgba, width) {
  let r = 0, g = 0, b = 0, count = 0;
  for (const cube of model.cubes) {
    if (!cube.light) continue;
    for (const [x0, y0, x1, y1] of cube.rects) {
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
        const i = (y * width + x) * 4;
        if (rgba[i + 3] < 8) continue;
        r += rgba[i]; g += rgba[i + 1]; b += rgba[i + 2]; count++;
      }
    }
  }
  if (!count) return null;
  const peak = Math.max(r, g, b) / count;
  if (peak <= 0) return null;
  const k = 255 / peak;
  const c = v => Math.min(255, Math.round(v / count * k));
  return (c(r) << 16) | (c(g) << 8) | c(b);
}
