// --- Interiors (core/interior.js, gpu/occlusion.js) ---
//
// Under a roof (for ROOF_DELAY), the building's roof and storeys above
// (interiorTargets) shrink away, solid, block by block (gpu/occlusion.js
// occlusionShrinkTSL) and are then not drawn at all, and what is outside the
// building darkens (core/interior.js inside) - the room shown on its own, as
// Octopath Traveler shows one. The view only: the lighting still sees the
// roof and the world outside. The room's sides facing the camera go with the
// roof (on by default, bxb.interiors('walls')): one square on, a corner's two
// from the battle camera; asked every frame, so a camera turn grows one wall
// back as the next shrinks away. Each block eases on its own (easeCells), so
// nothing pops.
//
// The blocks live in one mask texture of cells that follows the player
// (gpu/occlusion.js cellMask), uploaded only while a cell is easing. Every
// terrain material carries the mask in its vertex code; this feature writes
// the cells and the uniforms, once a frame before the first terrain pass - so
// switching it rebuilds nothing. Basic mode and the effects alike.
//
// ctx:
//   camera
//   feet()   the player's feet, world metres (the sprite's position)
//   picker   the cursor's (render/picking.js): it passes through a lifted roof
import * as THREE from 'three/webgpu';
import { DEFAULT_INTERIOR, ROOF_DELAY, CELL_EASE, INTERIOR_DIMS, playerCell, maskOrigin, maskIndex,
         buildInterior, interiorTargets, easeCells, rekeyCells } from '../../core/interior.js';
import { World } from '../../core/world.js';
import { occlusion, cellMask, insideMask } from '../../gpu/occlusion.js';
import { keyForInstance } from '../render.js';
import { flip } from '../engine.js';

export function createOcclusion({ camera, feet, picker = null }) {
  let lastT = 0;
  const forward = new THREE.Vector3();
  // On, and the building found for the player's cell.
  let roofsOn = true, frontWalls = true, room = null, underFor = 0, darkness = 0;
  const iparams = { ...DEFAULT_INTERIOR };
  // The mask: where it is (it follows the player's cell), each lifted cell's
  // eased value by mask index, and the indices written at the last upload.
  let cellKey = '', worldSize = -1, origin = null, written = [], recentred = false;
  const cells = new Map();

  // The mask follows the player's cell; the building is found again there,
  // or when the world was edited.
  function follow(f) {
    const cell = playerCell(f);
    const key = `${cell.x},${cell.y},${cell.z}`;
    if (key === cellKey && World.size === worldSize) return;
    cellKey = key;
    worldSize = World.size;
    const o = maskOrigin(cell);
    if (origin && (o.x !== origin.x || o.y !== origin.y || o.z !== origin.z)) {
      const moved = rekeyCells(cells, origin, o);
      cells.clear();
      for (const [i, v] of moved) cells.set(i, v);
      recentred = true;
    }
    origin = o;
    room = buildInterior(cell);
    if (room) writeInside(room);
  }

  // The inside mask, for this room; kept when the player leaves, while the
  // dark eases out.
  function writeInside(r) {
    const [W, , D] = INTERIOR_DIMS, data = insideMask.data;
    data.fill(0);
    for (const { x, z } of r.inside) {
      const i = x - origin.x, k = z - origin.z;
      if (i >= 1 && k >= 1 && i <= W - 2 && k <= D - 2) data[i + W * k] = 255;
    }
    insideMask.origin.value.set(origin.x, origin.z);
    insideMask.tex.needsUpdate = true;
  }

  // The targets, eased per block, and the mask uploaded when any moved.
  function update(now = performance.now()) {
    const dt = lastT ? Math.min(0.1, (now - lastT) / 1000) : 0;
    lastT = now;
    camera.updateMatrixWorld();
    camera.getWorldDirection(forward);
    follow(feet());
    const hl = Math.hypot(forward.x, forward.z) || 1;
    underFor = roofsOn && room ? underFor + dt : 0;
    const under = underFor >= ROOF_DELAY;
    const targets = under ? interiorTargets(room, { x: forward.x / hl, z: forward.z / hl }, frontWalls) : new Map();
    const moved = easeCells(cells, targets, dt);
    const dark = under ? iparams.dark : 0, step = dt / CELL_EASE;
    darkness = dark > darkness ? Math.min(dark, darkness + step) : Math.max(dark, darkness - step);
    occlusion.outsideDark.value = darkness;
    if (!moved && !recentred) return;
    recentred = false;
    const data = cellMask.data;
    for (const i of written) data[i] = 0;
    written = [];
    for (const [i, v] of cells) { data[i] = Math.round(v * 255); written.push(i); }
    cellMask.tex.needsUpdate = true;
    cellMask.origin.value.set(origin.x, origin.y, origin.z);
  }

  const describe = () => `interiors ${roofsOn ? `on: outside ${Math.round(iparams.dark * 100)}% dark, ` +
    `walls ${frontWalls ? 'lifted' : 'kept'}` +
    (room ? `, under a roof (${room.roof.length} roof cells, ${room.walls.length} wall cells)` : ', not under a roof')
    : 'off'}`;
  const commands = {
    interiors: {
      help: "under a roof, lift the roof and the storeys above and darken outside the building; " +
            "'walls' toggles lifting the walls facing the camera",
      usage: "bxb.interiors()  toggles  |  bxb.interiors(dark)  how dark outside, 0..1  |  bxb.interiors('walls')",
      run: (v = null) => {
        if (v === null) roofsOn = !roofsOn;
        else if (v === 'walls') frontWalls = !frontWalls;
        else iparams.dark = Math.min(1, Math.max(0, Number(v) || 0));
        return describe();
      }
    }
  };

  // Whether a block is lifted out of the view (its cell more than half gone)
  // - picking passes through it. key: a grid key.
  function hiddenAt(key) {
    if (!origin || !key) return false;
    const [x, y, z] = key.split(',').map(Number);
    return (cells.get(maskIndex(origin, x, y, z)) ?? 0) >= 0.5;
  }
  if (picker) picker.hidden = id => hiddenAt(keyForInstance(id));

  return {
    name: 'occlusion',
    update,
    // Another region loaded: the building is looked for afresh.
    reset() { cellKey = ''; worldSize = -1; },
    hiddenAt,
    commands,
    settings: { title: 'Interiors', settings: [
      { key: 'interiors', label: 'Lift roofs over the player', type: 'toggle',
        help: 'under a roof: the roof and the storeys above go, and outside the building darkens',
        get: () => roofsOn, set: v => flip(roofsOn, v, () => commands.interiors.run()) },
      { key: 'intwalls', label: 'Interiors: lift the walls facing the camera', type: 'toggle',
        help: 'the side of the room facing the camera goes with the roof - two at a corner, as in battle',
        get: () => frontWalls, set: v => flip(frontWalls, v, () => commands.interiors.run('walls')) },
      { key: 'intdark', label: 'Interiors: outside darkness', type: 'range', min: 0, max: 1, step: 0.05,
        help: 'how dark the world outside the building goes - the view only, not its light',
        get: () => iparams.dark, set: v => commands.interiors.run(v) }
    ]}
  };
}
