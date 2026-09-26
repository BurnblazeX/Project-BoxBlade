// --- The level editor ---
//
// Tab in and out. While it is on, the game waits (main.js gates movement and
// the AI on editor.active), the camera flies free, and the mouse edits:
//
//   left click          place the selection on the face under the cursor
//   right click         remove what is under the cursor (a block, or a model)
//   Shift + left drag   fill the box between two cells with the selection
//   Shift + right drag  clear the box between two cells
//   right drag          look around
//   W A S D, Space      fly (Shift: faster); Left Ctrl or C down
//   Q / E               turn the selection (slopes, models)
//   Ctrl + S            save the level (levels/default.json, dev server only)
//   Tab                 back to the game
//
// A model placed on a top face stands on it; on a side face it hangs on that
// wall, its north side against it (world.js mountWorldModel).
//
// Edits go through world.js (World, addWorldModel, ...) and then the one hook
// main.js hands in, onEdit, which rebuilds what depends on the world: the
// terrain instances, the field, the mirrors, the model lights.
import * as THREE from 'three/webgpu';
import { World, getVoxelKey, BLOCK_METRES, Y_MIN, Y_MAX, FACINGS, BLOCK_SHAPES, blockBoxes,
         blockShape, blockFacing, surfaceLevel,
         WorldModels, addWorldModel, mountWorldModel, removeWorldModel, newBlock,
         serializeLevel, hasFacing } from './world.js';
import { MATERIALS } from './materials.js';
import { MODELS } from './models.js';
import { worldInstancedMesh, instanceInfo } from './render.js';
import { createPalette } from './editorPanel.js';
import { placeModel as placeModelBoxes } from './bbmodel.js';

const FLY_SPEED = 10;          // m/s; Shift x3
const LOOK_SPEED = 0.004;      // radians per pixel
const DRAG_PIXELS = 4;         // a right press that moves less is a click
const MAX_BOX_CELLS = 20000;   // one fill or clear at most
const FACING_ORDER = ['N', 'E', 'S', 'W'];

export function createEditor({ scene, camera, dom, onEdit, onToggle, levelName = 'default' }) {
  const state = {
    active: false,
    // The selection: a block (material, shape) or a model; a facing for both.
    kind: 'block', materialId: MATERIALS[0].id, shape: 'full', model: null, facing: 0,
    yaw: 0, pitch: -0.6, pos: new THREE.Vector3(),
    keys: new Set(),
    drag: null,          // { button, x, y, moved, shift, start (a target) }
    hover: null          // the target under the cursor
  };

  // --- The panel (editorPanel.js): everything in it read from disk ---
  const palette = createPalette({
    state, levelName,
    pick: change => { Object.assign(state, change); palette.render(); }
  });
  const renderPanel = () => palette.render();
  const status = msg => { state.message = msg; renderPanel(); };

  // --- The cursor: the outline of what a click would do ---
  //
  // Placing, the outline is the selection's own shape where it would go - a
  // slope's wedge, the stairs' step, a model's cubes as placed; removing, the
  // shape of what would go; dragging a box, the box. Line segments rebuilt
  // per target (a few hundred floats), drawn over everything.
  const cursor = new THREE.LineSegments(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0x66ff88, depthTest: false, transparent: true }));
  cursor.renderOrder = 10;
  cursor.visible = false;
  cursor.raycast = () => {};
  cursor.frustumCulled = false;
  scene.add(cursor);
  const B = BLOCK_METRES, T = B / 12;
  const GROW = 0.01;   // metres the outline stands off the surface

  // The 12 edges of a box, in metres, into out.
  function boxEdges(lo, hi, out) {
    const [x0, y0, z0] = lo.map(v => v - GROW), [x1, y1, z1] = hi.map(v => v + GROW);
    const c = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
               [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
    for (const [i, j] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]) out.push(...c[i], ...c[j]);
  }
  // A block shape's outline in cell (x, y, z): its boxes, or a slope's wedge -
  // the plane, as it is drawn, not the staircase the field bakes.
  function shapeEdges(shape, facing, cell, out) {
    const base = [(cell[0] - 0.5) * B, (cell[1] - 0.5) * B, (cell[2] - 0.5) * B];
    const s = BLOCK_SHAPES[shape] || BLOCK_SHAPES.full;
    if (!s.slope) {
      for (const [x0, x1, y0, y1, z0, z1] of blockBoxes({ shape, facing })) {
        boxEdges([base[0] + x0 * B, base[1] + y0 * B, base[2] + z0 * B],
                 [base[0] + x1 * B, base[1] + y1 * B, base[2] + z1 * B], out);
      }
      return;
    }
    // The wedge, climbing +x (facing E), then turned as blockBoxes turns boxes.
    const turn = ([x, y, z]) => facing === 'W' ? [1 - x, y, z] : facing === 'S' ? [z, y, x]
                              : facing === 'N' ? [z, y, 1 - x] : [x, y, z];
    const P = (x, y, z) => turn([x, y, z]).map((v, a) => base[a] + v * B);
    const prof = [[0, 0], [1, 0], [1, s.high], [0, s.low]];
    const seg = (p, q) => { if (p.some((v, i) => Math.abs(v - q[i]) > 1e-6)) out.push(...p, ...q); };
    for (let i = 0; i < 4; i++) {
      const [x0, y0] = prof[i], [x1, y1] = prof[(i + 1) % 4];
      seg(P(x0, y0, 0), P(x1, y1, 0));
      seg(P(x0, y0, 1), P(x1, y1, 1));
      seg(P(x0, y0, 0), P(x0, y0, 1));
    }
  }
  // A placed model's outline: each cube (not its light), in world texels.
  function cubeEdges(cubes, out) {
    for (const c of cubes) boxEdges(c.min.map(v => v * T), c.max.map(v => v * T), out);
  }
  function setOutline(points, colour) {
    if (!points.length) { cursor.visible = false; return; }
    cursor.geometry.dispose();
    cursor.geometry = new THREE.BufferGeometry();
    cursor.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3));
    cursor.material.color.setHex(colour);
    cursor.visible = true;
  }

  // --- Targeting ---
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  // What is under the screen point: { place: cell [x,y,z] (the empty cell
  // against the face), key: the block hit, normal, model: the placed model
  // hit, if it was one }.
  function targetAt(clientX, clientY) {
    if (!worldInstancedMesh) return null;
    ndc.set((clientX / innerWidth) * 2 - 1, -(clientY / innerHeight) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObject(worldInstancedMesh, false)[0];
    if (!hit || !hit.face) return null;
    const info = instanceInfo(hit.instanceId);
    if (!info) return null;
    const n = hit.face.normal.clone().round();
    const key = info.key.split(',').map(Number);
    // Beside a block: the next cell along the face. Beside a model: the cell
    // the hit point is in, nudged off the face.
    const place = info.model
      ? [0, 1, 2].map(a => Math.round((hit.point.getComponent(a) + n.getComponent(a) * 0.01) / BLOCK_METRES))
      : [key[0] + n.x, key[1] + n.y, key[2] + n.z];
    return { place, key, normal: [n.x, n.y, n.z], model: info.model };
  }

  // What a left click on this target would place - the one answer both the
  // outline and the click use. null: nothing can go there.
  const WALL_YAW = { N: 0, W: 1, S: 2, E: 3 };   // as world.js mountWorldModel
  function planPlacement(target) {
    if (!target) return null;
    if (state.kind === 'block') {
      const c = target.place;
      if (!inBounds(c[1]) || World.has(getVoxelKey(...c))) return null;
      return { kind: 'block', cell: c, shape: state.shape, facing: FACING_ORDER[state.facing] };
    }
    const model = MODELS.get(state.model);
    if (!model || target.model) return null;
    const [nx, ny, nz] = target.normal;
    if (ny > 0) {
      // On a top face: stands on that block, origin at its top.
      const k = target.key;
      const at = [k[0] * 12, surfaceLevel(...k) * 12 - 6, k[2] * 12];
      return { kind: 'model', model, stand: k, yaw: state.facing,
               cubes: placeModelBoxes(model, at, state.facing).cubes };
    }
    if (ny === 0) {
      // On a side face: hangs in the cell in front of it, north to the wall.
      const wall = nx > 0 ? 'W' : nx < 0 ? 'E' : nz > 0 ? 'N' : 'S';
      const c = target.place;
      const at = [c[0] * 12, c[1] * 12 - 6, c[2] * 12];
      return { kind: 'model', model, cell: c, wall,
               cubes: placeModelBoxes(model, at, WALL_YAW[wall]).cubes };
    }
    return null;
  }

  function showCursor(target) {
    if (!target) { cursor.visible = false; return; }
    const d = state.drag;
    const removing = d ? d.button === 2 : false;
    const pts = [];
    if (d && d.shift && d.start) {
      // A box: its whole extent, in cells.
      const a = removing ? d.start.key : d.start.place;
      const b = removing ? target.key : target.place;
      const lo = a.map((v, i) => (Math.min(v, b[i]) - 0.5) * B), hi = a.map((v, i) => (Math.max(v, b[i]) + 0.5) * B);
      boxEdges(lo, hi, pts);
    } else if (removing) {
      // What would go: the model hit, or the block with its own shape.
      if (target.model) cubeEdges(target.model.cubes, pts);
      else {
        const block = World.get(getVoxelKey(...target.key));
        if (block) shapeEdges(blockShape(block), blockFacing(block), target.key, pts);
      }
    } else {
      const plan = planPlacement(target);
      if (plan && plan.kind === 'block') shapeEdges(plan.shape, plan.facing, plan.cell, pts);
      else if (plan) cubeEdges(plan.cubes, pts);
    }
    setOutline(pts, removing ? 0xff6666 : 0x66ff88);
  }

  // --- Edits ---
  const inBounds = y => y >= Y_MIN && y <= Y_MAX;
  // Removing a block takes what stands on it or hangs from it along with it.
  function removeModelsOn(key) {
    for (const m of [...WorldModels]) if (m.x === key[0] && m.y === key[1] && m.z === key[2]) removeWorldModel(m);
  }
  function placeBlockAt(c) {
    const k = getVoxelKey(...c);
    if (!inBounds(c[1]) || World.has(k)) return false;
    World.set(k, newBlock(state.materialId, state.shape, FACING_ORDER[state.facing]));
    return true;
  }
  function removeBlockAt(c) {
    const k = getVoxelKey(...c);
    if (!World.has(k)) return false;
    removeModelsOn(c);
    World.delete(k);
    return true;
  }
  function placeModel(target) {
    const plan = planPlacement(target);
    if (!plan || plan.kind !== 'model') return false;
    return !!(plan.wall ? mountWorldModel(plan.model, ...plan.cell, plan.wall)
                        : addWorldModel(plan.model, ...plan.stand, plan.yaw));
  }
  function boxCells(a, b) {
    const lo = a.map((v, i) => Math.min(v, b[i])), hi = a.map((v, i) => Math.max(v, b[i]));
    const n = (hi[0] - lo[0] + 1) * (hi[1] - lo[1] + 1) * (hi[2] - lo[2] + 1);
    if (n > MAX_BOX_CELLS) { status(`box of ${n} cells is over the ${MAX_BOX_CELLS} limit`); return []; }
    const out = [];
    for (let x = lo[0]; x <= hi[0]; x++) for (let y = lo[1]; y <= hi[1]; y++)
      for (let z = lo[2]; z <= hi[2]; z++) out.push([x, y, z]);
    return out;
  }
  function edited(what) {
    const r = onEdit();
    if (r && r.message) status(r.message);
    else status(what);
  }

  // --- Input ---
  function onKeyDown(e) {
    if (e.key === 'Tab') {
      e.preventDefault(); e.stopPropagation();
      setActive(!state.active);
      return;
    }
    if (!state.active) return;
    e.stopPropagation();
    // Typing in the palette's filter: its keys are text, not flying or turning.
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) {
      if (e.key === 'Escape') e.target.blur();
      return;
    }
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); save(); return; }
    // Left Ctrl descends, so Ctrl is held while flying: keep its browser
    // shortcuts off the fly keys (Ctrl+D would bookmark). Ctrl+W cannot be
    // held back - the browser closes the tab - so C descends too.
    if (e.ctrlKey && ['KeyA', 'KeyD', 'KeyS', 'Space'].includes(e.code)) e.preventDefault();
    if (k === 'q' || k === 'e') {
      state.facing = (state.facing + (k === 'e' ? 1 : 3)) % 4;
      renderPanel();
      showCursor(state.hover);   // the outline turns with it
      return;
    }
    state.keys.add(e.code);
    if (e.code === 'Space') e.preventDefault();
  }
  function onKeyUp(e) {
    state.keys.delete(e.code);
    if (state.active) e.stopPropagation();
  }
  function onPointerDown(e) {
    if (!state.active || e.target !== dom) return;
    e.stopPropagation();
    if (e.button !== 0 && e.button !== 2) return;
    state.drag = { button: e.button, x: e.clientX, y: e.clientY, moved: false, shift: e.shiftKey,
                   start: e.shiftKey ? targetAt(e.clientX, e.clientY) : null };
    showCursor(targetAt(e.clientX, e.clientY));
  }
  function onPointerMove(e) {
    if (!state.active) return;
    e.stopPropagation();
    const d = state.drag;
    if (d && Math.hypot(e.clientX - d.x, e.clientY - d.y) > DRAG_PIXELS) d.moved = true;
    if (d && d.button === 2 && !d.shift) {
      state.yaw -= e.movementX * LOOK_SPEED;
      state.pitch = Math.max(-1.5, Math.min(1.5, state.pitch - e.movementY * LOOK_SPEED));
    }
    state.hover = targetAt(e.clientX, e.clientY);
    showCursor(state.hover);
  }
  function onPointerUp(e) {
    if (!state.active) return;
    e.stopPropagation();
    const d = state.drag;
    state.drag = null;
    if (!d || d.button !== e.button) return;
    const target = targetAt(e.clientX, e.clientY);
    if (d.shift && d.start && target) {
      let n = 0;
      if (d.button === 0 && state.kind === 'block') {
        for (const c of boxCells(d.start.place, target.place)) n += placeBlockAt(c);
      } else if (d.button === 2) {
        for (const c of boxCells(d.start.key, target.key)) n += removeBlockAt(c);
      }
      if (n) edited(`${d.button === 0 ? 'filled' : 'cleared'} ${n} cells`);
    } else if (!d.moved && target) {
      let ok = false;
      if (d.button === 0) ok = state.kind === 'block' ? placeBlockAt(target.place) : placeModel(target);
      else if (target.model) ok = removeWorldModel(target.model);
      else ok = removeBlockAt(target.key);
      if (ok) edited(d.button === 0 ? 'placed' : 'removed');
    }
    showCursor(targetAt(e.clientX, e.clientY));
  }
  const onContextMenu = e => { if (state.active) e.preventDefault(); };
  // A key let go while the window was not looking would stay held.
  window.addEventListener('blur', () => state.keys.clear());
  // Capture phase on window: while the editor is on, the game's own handlers
  // (walking, clicking to move, battle keys) never see these.
  window.addEventListener('keydown', onKeyDown, true);
  window.addEventListener('keyup', onKeyUp, true);
  window.addEventListener('pointerdown', onPointerDown, true);
  window.addEventListener('pointermove', onPointerMove, true);
  window.addEventListener('pointerup', onPointerUp, true);
  window.addEventListener('contextmenu', onContextMenu, true);

  async function save() {
    try {
      const res = await fetch(`/__levels?name=${encodeURIComponent(levelName)}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(serializeLevel())
      });
      status(res.ok ? `saved levels/${levelName}.json` : `save failed: ${await res.text()}`);
    } catch (err) {
      status(`save failed - saving needs the dev server (npm run dev): ${err.message}`);
    }
  }

  function setActive(on) {
    if (on === state.active) return;
    const ok = onToggle ? onToggle(on) : true;
    if (on && ok === false) return;
    state.active = on;
    palette.show(on);
    state.keys.clear();
    state.drag = null;
    if (on) {
      // Start from where the game camera is, looking the way it looks.
      state.pos.copy(camera.position);
      const dir = new THREE.Vector3();
      camera.getWorldDirection(dir);
      state.yaw = Math.atan2(-dir.x, -dir.z);
      state.pitch = Math.asin(Math.max(-1, Math.min(1, dir.y)));
      renderPanel();
    } else {
      cursor.visible = false;
    }
  }

  // The camera, flown: called every frame after the game's camera code, so
  // it has the last word while the editor is on.
  const _fwd = new THREE.Vector3(), _right = new THREE.Vector3();
  function update(dt) {
    if (!state.active) return;
    const k = state.keys;
    const fast = k.has('ShiftLeft') || k.has('ShiftRight') ? 3 : 1;
    _fwd.set(-Math.sin(state.yaw), 0, -Math.cos(state.yaw));
    _right.set(Math.cos(state.yaw), 0, -Math.sin(state.yaw));
    const v = new THREE.Vector3();
    if (k.has('KeyW')) v.add(_fwd);
    if (k.has('KeyS')) v.sub(_fwd);
    if (k.has('KeyD')) v.add(_right);
    if (k.has('KeyA')) v.sub(_right);
    if (k.has('Space')) v.y += 1;
    if (k.has('KeyC') || k.has('ControlLeft')) v.y -= 1;
    if (v.lengthSq() > 0) state.pos.addScaledVector(v.normalize(), FLY_SPEED * fast * dt);
    camera.position.copy(state.pos);
    camera.rotation.set(state.pitch, state.yaw, 0, 'YXZ');
    camera.updateMatrixWorld();
  }

  // Where the field should be centred while editing: the ground the camera
  // looks at (the level of y = 0 blocks), or below it when looking up.
  function focusGridPos() {
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    const groundY = 0.75;
    let p = state.pos.clone();
    if (dir.y < -0.05) p.addScaledVector(dir, (groundY - p.y) / dir.y);
    return { x: Math.round(p.x / BLOCK_METRES), y: 0, z: Math.round(p.z / BLOCK_METRES) };
  }

  return {
    get active() { return state.active; },
    cursor, update, focusGridPos, setActive,
    // Test hooks.
    _state: state, _targetAt: targetAt
  };
}
