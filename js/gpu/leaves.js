// --- Leaves on the GPU: cubes that sway, lit as smoke is, textured by noise ---
//
// Every placed tree's crown shell (core/leaves.js) is one
// instanced box mesh, packed: LEAF_CUBE_CAPACITY at most, nearest the player
// first past that. Drawn as spell smoke's cubes are (gpu/smoke.js), lit by
// the same world light (gpu/cubelight.js) and the same way.
// light   per cube, LEAF_LIGHT_SLICE cubes a frame round the list (the whole
//         list at once when it is written): the world's light at its centre -
//         the sun through the world, with how much of it gets through the
//         leaves and any spell smoke (T); the lamps (through the leaves, more
//         dimmed through more of them) and the way they come from; the sky's
//         share, through the leaves.
// corners then per corner of those cubes' cells, the mean of the light of
//         the drawn cubes round it (up to eight; gpu/leafvolume.js's window,
//         an index a box), into a texture on the corner lattice (the sun's
//         half, then the lamps') - so a
//         face reads its four corners bilinearly, at each texel's centre:
//         light blends texel by texel across the crown instead of stepping
//         from cube to cube. A cube outside the window keeps its own light.
// draw    a cube is its full cell, and only its drawn faces are (core/
//         treegen.js shownLeafCells; another collapses to the cube's centre):
//         no face of one cube ever lies on another's. Both sides of a face:
//         through the cut-outs - texels of the noise's darkest gaps, under
//         `cutout` - the eye sees the faces behind and the branches. Round
//         the line from the camera to the player the cut-outs open further
//         (`hide`, within `hideRadius`), texel by texel, so a crown in front
//         of the player thins instead of hiding them. The
//         crown sways as one body: every corner
//         is moved by a smooth field of its REST position - a lean along the
//         wind (gpu/wind.js), a flutter across it and a little lift - so the
//         corners neighbouring cubes share move together and the crown stays
//         watertight. (Cubes swelling each on its own phase, as smoke's do,
//         crossed their neighbours' faces: z-fighting stripes on every leaf,
//         2026-10-30.) Its faces are cellular (Worley) noise on the texel
//         grid - one value a texel, so pixel-sized - read at the rest
//         position, so the pattern moves with the cube instead of swimming
//         over it, and continues across coplanar faces of neighbouring
//         cubes. Four levels, near a feature point lightest, coloured by the
//         species' ramp (darkest to lightest).
//         Lit as smoke's cubes are: of the sun that got through (T), a share
//         T(1 - T) x ms is scattered - lighting every face alike, so a
//         crown's half-lit middle glows - and the rest is direct, wrapped:
//         a face turned from the sun still takes `sss` of it. The lamps are
//         wrapped the same, as far as they come from one way (from all
//         round, evenly); the sky lights every face evenly. The albedo
//         darkens toward the crown's bottom (gradient: the share it loses
//         there).
// view    leafView (a module singleton, as gpu/smokeview.js smokeView): the
//         corners' light, shared by every build of the cubes and the shading.
//         leafViewHitTSL walks a ray through the window box by box - the leaf
//         volume's light says where the drawn cubes are, and their species
//         and height - and shades the first leaf texel it meets, past
//         the cut-outs, as the cubes are (at rest, the lamps from all
//         round): the leaves in reflections and through glass (gpu/shade.js).
import * as THREE from 'three/webgpu';
import { Fn, If, Loop, Break, float, vec2, vec3, vec4, int, uvec3, uniform, uniformArray, instanceIndex, storage,
         select, max, min, sin, fract, floor, round, abs, sign, step, clamp, length, mix, smoothstep, dot,
         varying, frontFacing, texture3D, textureStore, cameraPosition, positionGeometry, normalGeometry } from 'three/tsl';
import { createSunUniforms } from './sun.js';
import { createCubeWorldLight } from './cubelight.js';
import { smokeShared, smokeSunTransmitTSL } from './smokeshadow.js';
import { wind } from './wind.js';
import { leafVolume, leafLightTSL, LEAF_DIMS, LEAF_ID_BITS } from './leafvolume.js';
import { layout } from './layout.js';
import { smokeViewHitTSL } from './smokeview.js';
import { TREE_SPECIES } from '../core/treegen.js';
import { BOX_METRES } from '../core/boxgrid.js';
import { placedLeafCubes } from '../core/leaves.js';

export const LEAF_CUBE_CAPACITY = 131072;
// Cubes lit a frame: the whole list over a few frames.
export const LEAF_LIGHT_SLICE = 16384;
export const DEFAULT_LEAVES = {
  albedo: 1, sway: 0.06, speed: 1.6, waveLength: 6, clump: 3,
  sun: 1, lamps: 1, sky: 1, gradient: 0.35, sss: 0.45, ms: 0.8, cutout: 0.65,
  hide: 1, hideRadius: 2.5
};

const TEXEL = BOX_METRES / 4;
const [WX, WY, WZ] = LEAF_DIMS;
const CORNER_DIMS = [WX + 1, WY + 1, WZ + 1];
// The window's index (gpu/leafvolume.js): the drawn cube + 1 in the low
// bits, then its species and its height in the crown (0 .. 255).
const ID_BITS = LEAF_ID_BITS, ID_MASK = (1 << ID_BITS) - 1;
// Boxes a reflection's walk takes at most.
const LEAF_VIEW_STEPS = 96;
// A point in the unit square for lattice cell v of face plane s.
const hash2 = (v, s) => fract(sin(vec2(v.dot(vec2(127.1, 311.7)).add(s.mul(74.7)),
                                       v.dot(vec2(269.5, 183.3)).add(s.mul(246.1)))).mul(float(43758.5453)));

// The species' ramps, four colours each in species order (core/leaves.js
// LEAF_SPECIES), linear.
function rampRows() {
  return Object.values(TREE_SPECIES).flatMap(({ ramp }) => [0, 1, 2, 3].map(i => {
    const [r, g, b] = ramp[Math.round(i / 3 * (ramp.length - 1))];
    const c = new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
    return new THREE.Vector4(c.r, c.g, c.b, 1);
  }));
}

// The corners' texture: corner j (0 .. LEAF_DIMS) at the min corner of the
// window's box j, the sun's at z = j.z, the lamps' at z = j.z + LEAF_DIMS.z + 1.
function cornerTexture() {
  const t = new THREE.Storage3DTexture(CORNER_DIMS[0], CORNER_DIMS[1], 2 * CORNER_DIMS[2]);
  t.type = THREE.HalfFloatType;
  t.format = THREE.RGBAFormat;
  t.generateMipmaps = false;
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
  return t;
}

export const leafView = {
  // The corners' light: (the world's sun, T), (the lamps, the sky's share).
  corners: cornerTexture(),
  ramps: uniformArray(rampRows(), 'vec4').setName('bxbLeafRamps'),
  uniforms: Object.fromEntries(Object.entries(DEFAULT_LEAVES).map(([k, v]) => [k, uniform(v)])),
  // The player, metres, where the crown thins round the camera's view of them.
  focus: uniform(new THREE.Vector3(0, -1e4, 0)).setName('bxbLeafFocus')
};

// A cube's w (core/leaves.js): drawn faces x 4 + species, + its height share.
const speciesOf = w => int(floor(w)).mod(int(4));
const openOf = w => int(floor(w)).div(int(4));

// Where the sway moves a corner at rest position r (see draw, above): smooth
// in r, so shared corners move together.
function swayTSL(r, u) {
  const along = r.xz.dot(wind.dir).mul(float(2 * Math.PI)).div(u.waveLength).toVar();
  const across = r.xz.dot(vec2(wind.dir.y.negate(), wind.dir.x)).mul(float(2 * Math.PI)).div(u.waveLength.mul(1.7));
  const t = u.time.mul(u.speed);
  const a = u.sway.mul(wind.strength);
  const push = sin(t.sub(along)).mul(0.5).add(0.5);           // leans with the wind and back
  const flutter = sin(t.mul(1.9).sub(along.mul(1.3)).add(across)).mul(0.35);
  const lift = sin(t.mul(1.3).sub(along).add(r.y.mul(0.9))).mul(0.25);
  return vec3(wind.dir.x.mul(push).sub(wind.dir.y.mul(flutter)), lift,
              wind.dir.y.mul(push).add(wind.dir.x.mul(flutter))).mul(a);
}

// The corner lattice coordinate (0 .. 1) of world p, and whether p is in it.
function cornerUVW(p) {
  const uvw = p.div(float(BOX_METRES)).add(1).sub(leafVolume.u.origin).div(vec3(...CORNER_DIMS)).toVar();
  const inside = uvw.x.greaterThanEqual(0).and(uvw.y.greaterThanEqual(0)).and(uvw.z.greaterThanEqual(0))
    .and(uvw.x.lessThanEqual(1)).and(uvw.y.lessThanEqual(1)).and(uvw.z.lessThanEqual(1))
    .and(leafVolume.u.live.equal(int(1)));
  return { uvw: clamp(uvw, vec3(0), vec3(1)), inside };
}
const cornerLightTSL = uvw => ({ sun: texture3D(leafView.corners, vec3(uvw.xy, uvw.z.mul(0.5))).level(0),
                                 lamp: texture3D(leafView.corners, vec3(uvw.xy, uvw.z.mul(0.5).add(0.5))).level(0) });

// The leaf colour on a face with normal n (its geometry's: the pattern) at
// rest position r, seen from side ns (n, or -n from behind: the light) (see
// draw, above); w: the cube's. lightAt(p): { sun, lamp } at a texel centre p
// on the face - sun (the world's sun, T), lamp (the lamps, the sky's share);
// aim: the lamps' way, its length how much of them comes one way. cutAt(p):
// the cut-out threshold at that texel. Returns { colour, kept } - kept false
// in a cut-out.
function leafColourTSL(r, n, ns, w, lightAt, aim, skyColour, u, sunDir, cutAt = () => u.cutout) {
  // Every input to the hash an exact integer, and small: the interpolated
  // normal is 1 give or take its last bit from pixel to pixel, and sin() at
  // ~1e5 turned that bit into a different point - texels near a level's edge
  // broke into shimmering per-pixel static (2026-10-30).
  const nr = round(n).toVar();
  const q = r.div(float(TEXEL)).toVar();
  const an = abs(nr).toVar();
  const uv = select(an.x.greaterThan(0.5), q.zy, select(an.y.greaterThan(0.5), q.xz, q.xy));
  // The face's plane: its own points, shared with coplanar faces - the
  // plane's place along its axis, and which way the face looks (0 .. 6).
  const plane = floor(q.dot(nr).add(0.5)).mod(256).add(nr.dot(vec3(1, 2, 3)).add(3).mul(256));
  const g = floor(uv).add(0.5).div(u.clump).toVar();
  const cell = floor(g).toVar(), f = fract(g).toVar();
  let d = float(4);
  for (let oy = -1; oy <= 1; oy++) for (let ox = -1; ox <= 1; ox++) {
    const o = vec2(ox, oy);
    d = min(d, length(o.add(hash2(cell.add(o).mod(256), plane)).sub(f)));
  }
  const near = float(1).sub(d).toVar();
  const level = int(clamp(floor(near.mul(4)), float(0), float(3)));
  const albedo = leafView.ramps.element(speciesOf(w).mul(int(4)).add(level)).xyz;
  // Darker toward the crown's bottom: its height share in w's fraction.
  const shade = float(1).sub(u.gradient.mul(float(1).sub(fract(w))));
  // This texel's centre: across the face snapped to it, along the normal the
  // face's own plane (per axis by mix: a select takes one bool).
  const centre = mix(floor(q).add(0.5).mul(float(TEXEL)), r, an).toVar();
  const L = lightAt(centre);
  const T = L.sun.w.toVar();
  const glow = T.mul(float(1).sub(T)).mul(u.ms).toVar();
  const wrap = l => u.sss.add(float(1).sub(u.sss).mul(max(l, float(0))));
  const even = float(0.8).add(ns.y.mul(0.2));
  const one = clamp(length(aim), float(0), float(1)).toVar();
  const lampOn = mix(even, wrap(ns.dot(aim.div(max(one, float(1e-4))))), one);
  const sunOn = T.sub(glow).mul(wrap(ns.dot(sunDir))).add(glow);
  const colour = albedo.mul(L.sun.xyz.mul(sunOn).add(L.lamp.xyz.mul(lampOn)).add(skyColour.mul(L.lamp.w).mul(even)))
    .mul(shade).mul(u.albedo);
  return { colour, kept: near.greaterThanEqual(cutAt(centre)) };
}

// The cut-out threshold at texel centre p: `cutout`, rising to 1 (all gaps)
// within hideRadius of the line from the camera to the player, short of the
// player (see draw, above).
function hideCutTSL(p, u) {
  const eye = cameraPosition;
  const line = leafView.focus.sub(eye).toVar();
  const s = clamp(dot(p.sub(eye), line).div(max(dot(line, line), float(1e-4))), float(0), float(1)).toVar();
  const d = length(p.sub(eye.add(line.mul(s))));
  const near = smoothstep(u.hideRadius, u.hideRadius.mul(0.4), d).mul(step(s, float(0.999)));
  return mix(u.cutout, float(1.01), near.mul(u.hide));
}

// Fn([origin, dir, reach, sky, sunDir]) -> vec4(the colour of the first leaf
// texel along the ray within reach, its distance; -1 for none). sky: the
// sky's colour for the leaves (gpu/cubelight.js skyColour); sunDir: the
// shading's. Through the window's boxes, clipped to the leaves' bounds.
export const leafViewHitTSL = Fn(([origin, dir, reach, sky, sunDir]) => {
  const result = vec4(0, 0, 0, -1).toVar();
  const u = leafView.uniforms;
  If(leafVolume.u.live.equal(int(1)), () => {
    const parallel = step(abs(dir), vec3(1e-7)).toVar();
    const inv = vec3(1).div(mix(dir, vec3(1e-7), parallel)).toVar();
    const ta = leafVolume.u.lo.sub(origin).mul(inv), tb = leafVolume.u.hi.sub(origin).mul(inv);
    const near = min(ta, tb).toVar(), far = max(ta, tb);
    const t = max(max(max(near.x, near.y), near.z), float(0)).toVar();
    const end = min(min(min(far.x, far.y), far.z), reach).toVar();
    If(end.greaterThan(t), () => {
      // Box space: box i spans i - 0.5 .. i + 0.5 boxes.
      const cell = floor(origin.add(dir.mul(t.add(1e-4))).div(float(BOX_METRES)).add(0.5)).toVar();
      const s = sign(dir).toVar();
      const nextT = mix(cell.add(step(vec3(0), dir)).sub(0.5).mul(float(BOX_METRES)).sub(origin).mul(inv),
                        vec3(1e20), parallel).toVar();
      const delta = abs(inv).mul(float(BOX_METRES)).toVar();
      // The face the ray came in by: none for a ray starting in a box.
      const normal = vec3(0, 0, 0).toVar();
      If(t.greaterThan(0), () => {
        normal.assign(select(near.x.greaterThanEqual(near.y).and(near.x.greaterThanEqual(near.z)), vec3(s.x.negate(), 0, 0),
          select(near.y.greaterThanEqual(near.z), vec3(0, s.y.negate(), 0), vec3(0, 0, s.z.negate()))));
      });
      Loop({ start: int(0), end: int(LEAF_VIEW_STEPS), type: 'int', condition: '<', name: 'lvw' }, () => {
        If(t.greaterThan(end), () => { Break(); });
        const c = cell.sub(leafVolume.u.origin).toVar();
        If(c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.z.greaterThanEqual(0))
            .and(c.x.lessThan(float(WX))).and(c.y.lessThan(float(WY))).and(c.z.lessThan(float(WZ)))
            .and(dot(normal, normal).greaterThan(0.5)), () => {
          const box = texture3D(leafVolume.light, c.add(0.5).div(vec3(WX, WY, WZ))).level(0).toVar();
          If(box.z.greaterThan(0.5), () => {
            const code = round(box.w.mul(255));
            const w = floor(code.div(64)).add(code.mod(64).div(64));
            const p = origin.add(dir.mul(t)).toVar();
            const leaf = leafColourTSL(p, normal, normal, w, q => cornerLightTSL(cornerUVW(q).uvw), vec3(0, 0, 0),
                                       sky, u, sunDir);
            If(leaf.kept, () => { result.assign(vec4(leaf.colour, t)); Break(); });
          });
        });
        // On to the next box.
        const a = min(min(nextT.x, nextT.y), nextT.z).toVar();
        const m = select(nextT.x.lessThanEqual(nextT.y).and(nextT.x.lessThanEqual(nextT.z)), vec3(1, 0, 0),
          select(nextT.y.lessThanEqual(nextT.z), vec3(0, 1, 0), vec3(0, 0, 1))).toVar();
        cell.addAssign(m.mul(s));
        normal.assign(m.mul(s).negate());
        nextT.addAssign(m.mul(delta));
        t.assign(a);
      });
    });
  });
  return result;
}).setLayout(layout('leafViewHit', 'vec4', [
  { name: 'origin', type: 'vec3' }, { name: 'dir', type: 'vec3' }, { name: 'reach', type: 'float' },
  { name: 'sky', type: 'vec3' }, { name: 'sunDir', type: 'vec3' }]));

// The leaves (leafHit: leafViewHitTSL given the shading's sky and sun, or
// null) and the spell smoke (smoke: on) a secondary ray meets first within
// reach (a var, cut to the leaves): vec4(colour, distance), -1 for neither.
// gpu/shade.js's reflections and its view through glass.
export function nearestMediaTSL(leafHit, smoke, o, d, reach) {
  const seen = vec4(0, 0, 0, -1).toVar();
  if (leafHit) {
    seen.assign(leafHit(o, d, reach));
    If(seen.w.greaterThanEqual(0), () => { reach.assign(seen.w); });
  }
  if (smoke) {
    const cloud = smokeViewHitTSL(o, d, reach).toVar();
    If(cloud.w.greaterThanEqual(0), () => { seen.assign(cloud); });
  }
  return seen;
}

// ctx as createSmokeCubes's (gpu/smoke.js): cascades, lights, ..., or world
// (basic mode, the box light's walks).
export function createLeafCubes({ cascades, lights, biasUniform, fadeStartUniform, edgeFadeUniform,
                                  ambientUniform, sky, terrain, glass = false, cards = null, lightCards = null,
                                  shared = smokeShared, world = null }) {
  const COUNT = LEAF_CUBE_CAPACITY;
  // Per cube: centre (world metres), species + sky share (core/leaves.js).
  const cubeAttr = new THREE.StorageInstancedBufferAttribute(new Float32Array(COUNT * 4), 4);
  const cubes = storage(cubeAttr, 'vec4', COUNT).setName('bxbLeafCubes');
  // Per cube: (the world's sun, T), (the lamps, the sky's share), (the
  // lamps' way: their directions weighted by luminance, over their luminance).
  const lightAttr = new THREE.StorageBufferAttribute(COUNT * 3 * 4, 4);
  const light = storage(lightAttr, 'vec4', COUNT * 3).setName('bxbLeafLight');
  const lightRead = storage(lightAttr, 'vec4', COUNT * 3).toReadOnly().setName('bxbLeafLightRead');
  const { index } = leafVolume, { corners } = leafView;

  const u = {
    sunLight: uniform(float(1)),
    count: uniform(int(0)),          // cubes in the list
    offset: uniform(int(0)),         // the first cube this frame's light pass takes
    time: uniform(float(0)),
    ...leafView.uniforms
  };
  const sun = createSunUniforms(new THREE.Vector3(0, 1, 0));
  const worldLight = createCubeWorldLight({ cascades, lights, biasUniform, fadeStartUniform, edgeFadeUniform,
                                            ambientUniform, sky, terrain, glass, cards, lightCards, shared, world, sun, u,
                                            leaves: leafVolume });

  const lightKernel = Fn(() => {
    const i = int(instanceIndex).add(u.offset);
    If(i.lessThan(u.count), () => {
      const p = cubes.element(i).xyz.toVar();
      // Through the leaves (gpu/leafvolume.js): the crown shades itself.
      const through = leafLightTSL(p).toVar();
      const T = smokeSunTransmitTSL(shared, p, sun.dir).mul(through.x);
      const skyShare = worldLight.skyShare(p).mul(u.sky).mul(through.y);
      const lamps = worldLight.lampsAimed(p);
      const lampSum = lamps.sum.toVar();
      const aim = lamps.aim.div(max(lampSum.dot(vec3(0.2126, 0.7152, 0.0722)), float(1e-6)));
      const at = i.mul(int(3));
      light.element(at).assign(vec4(worldLight.sun(p), T));
      light.element(at.add(int(1))).assign(vec4(lampSum.mul(u.lamps), skyShare));
      light.element(at.add(int(2))).assign(vec4(aim, 0));
    });
  })().compute(COUNT).setName('leafLight');

  // Eight a cube, as the light pass's slice: the cube's corners, each the
  // mean of the drawn cubes round it.
  const box = i => int(i.x).add(int(i.y).mul(int(WX))).add(int(i.z).mul(int(WX * WY)));
  const inWindow = c => c.x.greaterThanEqual(0).and(c.y.greaterThanEqual(0)).and(c.z.greaterThanEqual(0))
    .and(c.x.lessThan(float(WX))).and(c.y.lessThan(float(WY))).and(c.z.lessThan(float(WZ)));
  const cornerKernel = Fn(() => {
    const t = int(instanceIndex);
    const i = t.div(int(8)).add(u.offset).toVar();
    const k = t.mod(int(8)).toVar();
    If(i.lessThan(u.count).and(leafVolume.u.live.equal(int(1))), () => {
      const b = round(cubes.element(i).xyz.div(float(BOX_METRES))).sub(leafVolume.u.origin).toVar();
      If(inWindow(b), () => {
        const j = b.add(vec3(float(k.bitAnd(int(1))), float(k.shiftRight(int(1)).bitAnd(int(1))),
                             float(k.shiftRight(int(2))))).toVar();
        const sumSun = vec4(0, 0, 0, 0).toVar(), sumLamp = vec4(0, 0, 0, 0).toVar(), n = float(0).toVar();
        for (let m = 0; m < 8; m++) {
          const c = j.sub(vec3(1 - (m & 1), 1 - ((m >> 1) & 1), 1 - (m >> 2))).toVar();
          If(inWindow(c), () => {
            const id = index.element(box(c)).bitAnd(int(ID_MASK)).toVar();
            If(id.greaterThan(int(0)), () => {
              const at = id.sub(int(1)).mul(int(3));
              sumSun.addAssign(lightRead.element(at));
              sumLamp.addAssign(lightRead.element(at.add(int(1))));
              n.addAssign(1);
            });
          });
        }
        textureStore(corners, uvec3(j.x, j.y, j.z), sumSun.div(max(n, float(1))));
        textureStore(corners, uvec3(j.x, j.y, j.z.add(float(WZ + 1))), sumLamp.div(max(n, float(1))));
      });
    });
  })().compute(COUNT * 8).setName('leafCorners');

  // The cubes: their drawn faces, each corner swayed from its rest position.
  const cube = cubes.element(instanceIndex);
  const n = normalGeometry;
  const face = select(n.x.greaterThan(0.5), int(0), select(n.x.lessThan(-0.5), int(1),
    select(n.y.greaterThan(0.5), int(2), select(n.y.lessThan(-0.5), int(3), select(n.z.greaterThan(0.5), int(4), int(5))))));
  const open = openOf(cube.w).shiftRight(face).bitAnd(int(1)).equal(int(1));
  const rest = cube.xyz.add(positionGeometry.mul(float(BOX_METRES))).toVar();
  const mat = new THREE.MeshBasicNodeMaterial();
  mat.positionNode = select(open, rest.add(swayTSL(rest, u)), cube.xyz);
  const at = int(instanceIndex).mul(int(3));
  const vSun = varying(lightRead.element(at), 'vLeafSun');
  const vLamp = varying(lightRead.element(at.add(int(1))), 'vLeafLamp');
  const vAim = varying(lightRead.element(at.add(int(2))).xyz, 'vLeafAim');
  const vRest = varying(rest, 'vLeafRest');
  const vW = varying(cube.w, 'vLeafW');
  // In the window, the corners round texel centre p; outside it, the cube's own.
  const lightAt = p => {
    const c = cornerUVW(p);
    const L = cornerLightTSL(c.uvw);
    return { sun: select(c.inside, L.sun, vSun), lamp: select(c.inside, L.lamp, vLamp) };
  };
  const ns = normalGeometry.mul(select(frontFacing, float(1), float(-1)));
  const leaf = leafColourTSL(vRest, normalGeometry, ns, vW, lightAt, vAim, worldLight.skyColour(), u, sun.dir,
                             p => hideCutTSL(p, u));
  mat.colorNode = leaf.colour;
  // Cut out: both sides drawn, so through a gap the faces behind show.
  mat.opacityNode = select(leaf.kept, float(1), float(0));
  mat.alphaTest = 0.5;
  mat.side = THREE.DoubleSide;
  const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), mat, COUNT);
  mesh.frustumCulled = false;
  mesh.count = 0;
  mesh.name = 'leafCubes';

  return {
    uniforms: u, sun, light: lightKernel, corners: cornerKernel, mesh,
    // The leaves of the placed models (core/world.js WorldModels), nearest
    // `near` (metres) kept past the capacity. Returns { count, total }.
    write(models, near = null) {
      const r = placedLeafCubes(models, cubeAttr.array, near);
      if (r.count) {
        cubeAttr.addUpdateRange(0, 4 * r.count);
        cubeAttr.needsUpdate = true;
      }
      mesh.count = r.count;
      u.count.value = r.count;
      return r;
    },
    // The index of the leaf window whose min box is origin: each drawn cube
    // inside it at its box (see ID_BITS).
    fillIndex(origin) {
      const idx = leafVolume.indexAttr.array, arr = cubeAttr.array;
      idx.fill(0);
      for (let i = 0; i < mesh.count; i++) {
        const x = Math.round(arr[4 * i] / BOX_METRES) - origin[0];
        const y = Math.round(arr[4 * i + 1] / BOX_METRES) - origin[1];
        const z = Math.round(arr[4 * i + 2] / BOX_METRES) - origin[2];
        if (x < 0 || y < 0 || z < 0 || x >= WX || y >= WY || z >= WZ) continue;
        const w = arr[4 * i + 3], whole = Math.floor(w);
        idx[x + WX * (y + WY * z)] = (i + 1) | ((whole % 4) << ID_BITS) | (Math.floor((w - whole) * 256) << (ID_BITS + 2));
      }
      leafVolume.indexAttr.needsUpdate = true;
    },
    dispose() { mesh.geometry.dispose(); mat.dispose(); }
  };
}
