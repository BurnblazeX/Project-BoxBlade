import * as THREE from 'three/webgpu';
import { leafLightTSL } from './leafvolume.js';
import {
  Fn, If, Loop, Break, float, int, uint, vec3, vec4, uvec3, uniform, instanceIndex, storage, min,
  textureStore, texture3D, select, max, abs, length, log2, clamp, round, shiftRight, bitAnd, cos, sin, exp
} from 'three/tsl';
import { BOX_CELLS, BOX_METRES } from '../core/boxgrid.js';
import { MAX_LIGHT_LEVEL, LIGHT_VEC4S } from '../core/lights.js';
import { BOX_BLOCK_U32, SKY_RAYS, SKY_REACH } from '../core/boxlight.js';
import { createCardsTSL, NO_SKIP, CARD_SUN_MAX } from './cards.js';
import { smokeShared, smokeTauTSL, smokeTransmitTSL, smokeSunTransmitTSL } from './smokeshadow.js';

// --- Box light on the GPU (core/boxlight.js has the why and the CPU reference) ---
//
// Four kernels over one region:
//   clear   both textures to 0 - "not a surface cell" - after a rebuild
//   sky     per surface cell, the SKY_RAYS walks: its openness into a buffer.
//           Run when the world changes.
//   sun     per surface cell, the walk to the sun: rgb (glass tints it) into
//           a buffer. Run when the sun moves or the world changes.
//   lamps   per surface cell, the light list's sum, each lamp walked to and
//           tested against its sprite cards, and the sun buffer times the
//           sun's cards; the sun, every lamp and the sky through the cloud
//           spells' smoke (gpu/smokeshadow.js: its hard boxes, the sun's
//           eight-box cutoff), as the shaded path has them. Run when a light,
//           a card, the sun or the smoke changed, and while a cloud rolls out.
// And `world`: the same walks from any point, for what lights itself from
// the box light - the cloud spells' cubes in basic mode (gpu/smoke.js).
// It writes the cell's two texels:
//   A  RGBA16F   rgb the lamps, a 1 - so 0 means "not a surface cell"
//   B  RGBA8     rgb the sun, a the sky
// A face reads them FILTERED (boxLightAtTSL): the hardware blends the boxes
// round its texel, and dividing by A's a - how much of that blend was surface
// cells - drops the solid and unwritten ones. So each texel gets a blend of
// the boxes about it instead of a hard 0.5 m step, for the price of the reads
// it had anyway.

// The fragment side: kept for the material's life. The textures behind it
// are swapped when the region changes (setTextures), the uniforms written.
export function createBoxLightBinding() {
  const nodes = { A: [], B: [] };
  return {
    texA: makeTexture(1, 1, 1, THREE.HalfFloatType),
    texB: makeTexture(1, 1, 1, THREE.UnsignedByteType),
    box0: uniform(new THREE.Vector3()),            // world box index of local box 0
    dims: uniform(new THREE.Vector3(1, 1, 1)),     // the region in boxes
    sunDir: uniform(new THREE.Vector3(0, 1, 0)),   // toward the sun
    sunColour: uniform(new THREE.Vector3(1, 1, 1)), // the plain light's colour x intensity
    ambient: uniform(new THREE.Vector3(0.6, 0.6, 0.6)), // the plain AmbientLight's colour x intensity
    lampGain: uniform(float(1.2)),                 // a lamp's peak, as a sun intensity
    skyFloor: uniform(float(0.1)),                 // the ambient where no sky is seen
    // Texture nodes over the current textures, remembered so a swap reaches them.
    atA(uvw) { const t = texture3D(this.texA, uvw).level(int(0)); nodes.A.push(t); return t; },
    atB(uvw) { const t = texture3D(this.texB, uvw).level(int(0)); nodes.B.push(t); return t; },
    setTextures(a, b) {
      this.texA = a; this.texB = b;
      for (const t of nodes.A) t.value = a;
      for (const t of nodes.B) t.value = b;
    }
  };
}

function makeTexture(x, y, z, type) {
  const t = new THREE.Storage3DTexture(x, y, z);
  t.type = type;
  t.format = THREE.RGBAFormat;
  t.generateMipmaps = false;
  t.minFilter = t.magFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
  return t;
}

// What the face at p (world metres, normal n; inside an Fn) is lit by:
// { sun rgb, lamp rgb, sky 0..1 }. Read in the layer of boxes in front of the
// face - at their centres along n, so only the face's neighbours across it
// blend - and normalised by how much of the blend was surface cells. Where
// none were (a slope's plane over a step), half a metre further out. Off the
// region: the sun and the sky unshadowed, no lamps; in it with no cell near,
// no sun or sky (a face only a lifted roof reveals).
export function boxLightAtTSL(b, p, n) {
  const uvwAt = q => q.div(float(BOX_METRES)).sub(b.box0).add(float(0.5)).div(b.dims);
  const uvw = uvwAt(p.add(n.mul(float(BOX_METRES / 2)))).toVar();
  const A = b.atA(uvw).toVar(), B = b.atB(uvw).toVar();
  If(A.w.lessThan(float(0.01)), () => {
    const far = uvwAt(p.add(n.mul(float(BOX_METRES * 1.5))));
    A.assign(b.atA(far));
    B.assign(b.atB(far));
  });
  const inside = uvw.x.greaterThanEqual(float(0)).and(uvw.y.greaterThanEqual(float(0)))
    .and(uvw.z.greaterThanEqual(float(0))).and(uvw.x.lessThanEqual(float(1)))
    .and(uvw.y.lessThanEqual(float(1))).and(uvw.z.lessThanEqual(float(1)));
  const k = float(1).div(max(A.w, float(1e-3))).mul(min(A.w.mul(float(100)), float(1)));
  // ... and through the trees' leaves (gpu/leafvolume.js): their dappled shade.
  const leaves = leafLightTSL(p.add(n.mul(float(BOX_METRES / 2)))).toVar();
  return {
    lamp: select(inside, A.xyz.mul(k), vec3(0, 0, 0)),
    sun: select(inside, B.xyz.mul(k), vec3(1, 1, 1)).mul(leaves.x),
    sky: select(inside, B.w.mul(k), float(1)).mul(leaves.y)
  };
}

// The kernels for one region. capacity: surface cells the buffers hold;
// tintCapacity: glass tints. lights: the light list's bindings
// (gpu/pointlights.js). sunCards, lightCards: the sprites' card bindings
// (gpu/cards.js), or null for no sprite shadows.
export function createBoxLightKernels({ binding, region, capacity, tintCapacity, lights,
                                        sunCards = null, lightCards = null }) {
  const [dx, dy, dz] = region.dims;
  const blocks = region.size[0] * region.size[1] * region.size[2];
  const texA = makeTexture(dx, dy, dz, THREE.HalfFloatType);
  const texB = makeTexture(dx, dy, dz, THREE.UnsignedByteType);
  const blockAttr = new THREE.StorageBufferAttribute(new Uint32Array(blocks * BOX_BLOCK_U32), 1);
  const tintAttr = new THREE.StorageBufferAttribute(new Uint32Array(tintCapacity), 1);
  const cellsAttr = new THREE.StorageBufferAttribute(new Uint32Array(capacity), 1);
  const sunAttr = new THREE.StorageBufferAttribute(new Float32Array(capacity * 4), 4);
  const skyAttr = new THREE.StorageBufferAttribute(new Float32Array(capacity), 1);
  const blockData = storage(blockAttr, 'uint', blocks * BOX_BLOCK_U32).toReadOnly().setName('bxbBoxBlocks');
  const tints = storage(tintAttr, 'uint', tintCapacity).toReadOnly().setName('bxbBoxTints');
  const cells = storage(cellsAttr, 'uint', capacity).toReadOnly().setName('bxbBoxCells');
  const sunBuf = storage(sunAttr, 'vec4', capacity).setName('bxbBoxSun');
  const skyBuf = storage(skyAttr, 'float', capacity).setName('bxbBoxSky');
  const sunCardsFn = createCardsTSL(sunCards);
  const lightCardsFn = createCardsTSL(lightCards, { ranged: true });

  const u = {
    count: uniform(int(0)),
    top: uniform(int(0)),
    cutoff: lights.cutoff
  };
  const D = [dx, dy, dz].map(v => int(v));
  const NX = int(region.size[0]), NY = int(region.size[1]);
  const STEPS = int(dx + dy + dz + 3);
  const C3 = int(BOX_CELLS);

  // Box (x, y, z), inside the region: its block's first u32 and its bit.
  const boxOf = (x, y, z) => {
    const bx = x.div(C3), by = y.div(C3), bz = z.div(C3);
    const bit = x.sub(bx.mul(C3)).add(y.sub(by.mul(C3)).mul(C3)).add(z.sub(bz.mul(C3)).mul(int(BOX_CELLS * BOX_CELLS)));
    return { at: bx.add(NX.mul(by.add(NY.mul(bz)))).mul(int(BOX_BLOCK_U32)).toVar(), bit: uint(bit).toVar() };
  };
  const bitSet = (word, bit) => bitAnd(shiftRight(word, bit), uint(1)).equal(uint(1));
  const tintOf = (at, bit) => {
    const v = tints.element(int(blockData.element(at.add(int(2)))).add(int(bit))).toVar();
    return vec3(float(bitAnd(v, uint(255))), float(bitAnd(shiftRight(v, uint(8)), uint(255))),
                float(bitAnd(shiftRight(v, uint(16)), uint(255)))).div(float(255));
  };

  // core/boxlight.js boxRay, line for line: the rgb that gets through from
  // box (x, y, z) along dir before maxT, the target box, y = top, or the
  // region's edge - 0 at anything solid, each pane entered tinting it.
  const boxRayTSL = (cx, cy, cz, dir, maxT, top, tx, ty, tz) => {
    const x = cx.toVar(), y = cy.toVar(), z = cz.toVar();
    const stepOf = v => select(v.greaterThan(float(0)), int(1), select(v.lessThan(float(0)), int(-1), int(0)));
    const sx = stepOf(dir.x).toVar(), sy = stepOf(dir.y).toVar(), sz = stepOf(dir.z).toVar();
    // Both sides finite: a constant direction (the sky's straight up) folds
    // at compile time, and WGSL rejects a 1 / 0 it can see.
    const deltaOf = v => select(abs(v).greaterThan(float(1e-6)), float(1).div(max(abs(v), float(1e-6))), float(1e30));
    const tdx = deltaOf(dir.x).toVar(), tdy = deltaOf(dir.y).toVar(), tdz = deltaOf(dir.z).toVar();
    const tmx = tdx.mul(float(0.5)).toVar(), tmy = tdy.mul(float(0.5)).toVar(), tmz = tdz.mul(float(0.5)).toVar();
    const t = float(0).toVar();
    const T = vec3(1, 1, 1).toVar();
    // From off the region (a world point, not a cell): nothing to cross.
    If(x.greaterThanEqual(int(0)).and(y.greaterThanEqual(int(0))).and(z.greaterThanEqual(int(0)))
      .and(x.lessThan(D[0])).and(y.lessThan(D[1])).and(z.lessThan(D[2])), () => {
    // Starting in glass (a cell behind a pane): through it already.
    const start = boxOf(x, y, z);
    const inGlass = bitSet(blockData.element(start.at.add(int(1))), start.bit).toVar();
    If(inGlass, () => { T.assign(tintOf(start.at, start.bit)); });
    Loop({ start: int(0), end: STEPS, type: 'int', condition: '<', name: 'bs' }, () => {
      If(tmx.lessThanEqual(tmy).and(tmx.lessThanEqual(tmz)), () => {
        t.assign(tmx); x.addAssign(sx); tmx.addAssign(tdx);
      }).ElseIf(tmy.lessThanEqual(tmz), () => {
        t.assign(tmy); y.addAssign(sy); tmy.addAssign(tdy);
      }).Else(() => {
        t.assign(tmz); z.addAssign(sz); tmz.addAssign(tdz);
      });
      If(t.greaterThanEqual(maxT), () => { Break(); });
      If(x.equal(tx).and(y.equal(ty)).and(z.equal(tz)), () => { Break(); });
      If(y.greaterThanEqual(top), () => { Break(); });
      If(x.lessThan(int(0)).or(y.lessThan(int(0))).or(z.lessThan(int(0)))
        .or(x.greaterThanEqual(D[0])).or(y.greaterThanEqual(D[1])).or(z.greaterThanEqual(D[2])), () => { Break(); });
      const box = boxOf(x, y, z);
      If(bitSet(blockData.element(box.at), box.bit), () => { T.assign(vec3(0, 0, 0)); Break(); });
      const glass = bitSet(blockData.element(box.at.add(int(1))), box.bit).toVar();
      If(glass.and(inGlass.not()), () => { T.mulAssign(tintOf(box.at, box.bit)); });
      inGlass.assign(glass);
    });
    });
    return T;
  };

  const cellOf = i => {
    const v = cells.element(i).toVar();
    return {
      x: int(bitAnd(v, uint(1023))).toVar(),
      y: int(bitAnd(shiftRight(v, uint(10)), uint(1023))).toVar(),
      z: int(shiftRight(v, uint(20))).toVar()
    };
  };
  // The cell's centre in world metres.
  const centreOf = c => vec3(float(c.x), float(c.y), float(c.z)).add(binding.box0).mul(float(BOX_METRES)).toVar();
  const NONE = int(-1 << 20);
  const NO_TOP = int(1 << 30);

  const sun = Fn(() => {
    const i = int(instanceIndex).toVar();
    If(i.lessThan(u.count), () => {
      const c = cellOf(i);
      const T = vec3(0, 0, 0).toVar();
      If(binding.sunDir.y.greaterThan(float(0)), () => {
        T.assign(boxRayTSL(c.x, c.y, c.z, binding.sunDir, float(1e30), u.top, NONE, NONE, NONE));
      });
      sunBuf.element(i).assign(vec4(T, 0));
    });
  })().compute(capacity).setName('BoxLight.Sun');

  // core/boxlight.js skyDir and boxSky.
  const sky = Fn(() => {
    const i = int(instanceIndex).toVar();
    If(i.lessThan(u.count), () => {
      const c = cellOf(i);
      const sum = float(0).toVar();
      Loop({ start: int(0), end: int(SKY_RAYS), type: 'int', condition: '<', name: 'sk' }, ({ sk }) => {
        const high = sk.lessThanEqual(int(6));
        const e = select(high, float(Math.PI / 3), float(Math.PI / 6));
        const a = select(high, float(sk.sub(int(1))).mul(float(Math.PI / 3)),
                         float(sk.sub(int(7))).mul(float(Math.PI / 3)).add(float(Math.PI / 6)));
        const d = select(sk.equal(int(0)), vec3(0, 1, 0),
                         vec3(cos(e).mul(cos(a)), sin(e), cos(e).mul(sin(a)))).toVar();
        const T = boxRayTSL(c.x, c.y, c.z, d, float(SKY_REACH), u.top, NONE, NONE, NONE);
        sum.addAssign(T.x.add(T.y).add(T.z).div(float(3)));
      });
      skyBuf.element(i).assign(sum.div(float(SKY_RAYS)));
    });
  })().compute(capacity).setName('BoxLight.Sky');

  // Every lamp at p (world metres; box c), rgb: walked to, tested against its
  // sprite cards, and through the smoke - its hard boxes for the terrain's
  // cells, its soft copy for the smoke's own cubes (soft).
  const smokeTau = smokeTauTSL(smokeShared, false);
  const lampsTSL = (p, c, soft) => {
    const sum = vec3(0, 0, 0).toVar();
    Loop({ start: int(0), end: lights.count, type: 'int', condition: '<', name: 'li' }, ({ li }) => {
      const base = li.mul(int(LIGHT_VEC4S));
      const A = lights.data.element(base).toVar();
      const colour = lights.data.element(base.add(int(1))).xyz.toVar();
      const level = A.w;
      const to = A.xyz.sub(p).toVar();
      const dist = max(length(to), float(1e-4)).toVar();
      // lights.js lightFalloff and cutAmount, as the shaded path has them;
      // no N.L - the cell is lit from every side.
      const f = clamp(float(1).sub(dist.div(max(level, float(1e-4)))), float(0), float(1));
      const falloff = log2(float(1).add(f)).mul(level.div(float(MAX_LIGHT_LEVEL)));
      const amount = max(falloff.sub(u.cutoff), float(0)).div(float(1).sub(u.cutoff)).toVar();
      If(amount.greaterThan(float(0)), () => {
        const dir = to.div(dist).toVar();
        const goal = round(A.xyz.div(float(BOX_METRES))).sub(binding.box0).toVar();
        const T = boxRayTSL(c.x, c.y, c.z, dir, dist.div(float(BOX_METRES)), NO_TOP,
                            int(goal.x), int(goal.y), int(goal.z)).toVar();
        if (lightCardsFn) {
          // The cards turned to face this light (its slice: row 2's x, y),
          // hard, weighted in as the shaded path weighs them (row 2's w).
          const C = lights.data.element(base.add(int(2))).toVar();
          const start = int(C.x).toVar();
          const v = lightCardsFn(p, dir, dist, float(0), NO_SKIP, start, start.add(int(C.y)));
          T.mulAssign(float(1).sub(float(1).sub(v).mul(C.w)));
        }
        T.mulAssign(smokeTransmitTSL(smokeShared, p, dir, dist, soft));
        sum.addAssign(colour.mul(amount).mul(T));
      });
    });
    return sum;
  };

  const lamps = Fn(() => {
    const i = int(instanceIndex).toVar();
    If(i.lessThan(u.count), () => {
      const c = cellOf(i);
      const p = centreOf(c);
      const sum = lampsTSL(p, c, false);
      const sunT = sunBuf.element(i).xyz.toVar();
      If(sunT.x.add(sunT.y).add(sunT.z).greaterThan(float(0)), () => {
        if (sunCardsFn) sunT.mulAssign(sunCardsFn(p, binding.sunDir, float(CARD_SUN_MAX), float(0), NO_SKIP));
        sunT.mulAssign(smokeSunTransmitTSL(smokeShared, p, binding.sunDir));
      });
      // The sky through the smoke over the cell, as the smoke's cubes take it.
      const sky = skyBuf.element(i).mul(exp(smokeTau(p, vec3(0, 1, 0), float(8))
        .mul(smokeShared.strength).mul(float(-0.5))));
      const at = uvec3(uint(c.x), uint(c.y), uint(c.z));
      textureStore(texA, at, vec4(sum, 1));
      textureStore(texB, at, vec4(sunT, sky));
    });
  })().compute(capacity).setName('BoxLight.Lamps');

  // The walks from any world point p (inside an Fn), for the smoke's cubes:
  // nothing of the smoke in them (the cubes take their own) but the lamps'
  // (its soft copy). Off the region a walk crosses nothing.
  const localOf = p => {
    const b = round(p.div(float(BOX_METRES))).sub(binding.box0).toVar();
    return { x: int(b.x).toVar(), y: int(b.y).toVar(), z: int(b.z).toVar() };
  };
  const world = {
    binding,
    // The sun at p, rgb: the world (glass tints it) and the sprites' cards.
    sun: p => {
      const c = localOf(p);
      const T = vec3(0, 0, 0).toVar();
      If(binding.sunDir.y.greaterThan(float(0)), () => {
        T.assign(boxRayTSL(c.x, c.y, c.z, binding.sunDir, float(1e30), u.top, NONE, NONE, NONE));
        if (sunCardsFn) T.mulAssign(sunCardsFn(p, binding.sunDir, float(CARD_SUN_MAX), float(0), NO_SKIP));
      });
      return T;
    },
    // Every lamp at p, through the smoke's soft copy.
    lamps: p => lampsTSL(p, localOf(p), true),
    // Whether the sky is open straight up from p (glass at its tint's mean).
    sky: p => {
      const c = localOf(p);
      const T = boxRayTSL(c.x, c.y, c.z, vec3(0, 1, 0), float(SKY_REACH), u.top, NONE, NONE, NONE);
      return T.x.add(T.y).add(T.z).div(float(3));
    }
  };

  const texels = dx * dy * dz;
  const clear = Fn(() => {
    const i = int(instanceIndex).toVar();
    If(i.lessThan(int(texels)), () => {
      const at = uvec3(uint(i.mod(int(dx))), uint(i.div(int(dx)).mod(int(dy))), uint(i.div(int(dx * dy))));
      textureStore(texA, at, vec4(0, 0, 0, 0));
      textureStore(texB, at, vec4(0, 0, 0, 0));
    });
  })().compute(texels).setName('BoxLight.Clear');

  return {
    texA, texB, clear, sky, sun, lamps, world, uniforms: u, capacity, tintCapacity,
    cards: { sun: sunCards, light: lightCards },
    // A rebuilt world: its block data, tints, surface cells and top.
    write(masks, cellData, top) {
      blockAttr.array.set(masks.data);
      blockAttr.needsUpdate = true;
      tintAttr.array.set(masks.tints.subarray(0, tintCapacity));
      tintAttr.needsUpdate = true;
      cellsAttr.array.fill(0);
      cellsAttr.array.set(cellData.subarray(0, capacity));
      cellsAttr.needsUpdate = true;
      u.count.value = Math.min(cellData.length, capacity);
      u.top.value = top;
    },
    dispose(renderer = null) {
      texA.dispose();
      texB.dispose();
      const attrs = renderer && renderer._attributes;
      if (attrs) for (const a of [blockAttr, tintAttr, cellsAttr, sunAttr, skyAttr]) attrs.delete(a);
    }
  };
}
