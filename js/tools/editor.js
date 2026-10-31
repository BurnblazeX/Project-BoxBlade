// --- The level editor ---
//
// Tab in and out. While it is on, the game waits (main.js gates movement and
// the AI on editor.active), the camera flies free, and the mouse edits:
//
//   left click          place the selection on the face under the cursor, in
//                       the Grid cell against it: a part stacks into a tile
//                       beside the parts already there (world.js withPart)
//   left drag           place a row of blocks along the axis the drag follows
//   right click         remove what is under the cursor (a model, or one
//                       part of a tile)
//   Shift + left drag   fill the box between two cells with the selection
//   Shift + right drag  clear the box between two cells
//   right drag          look around
//   W A S D, Space      fly (Shift: faster); Left Ctrl or C down
//   Q / E               turn the selection (slopes, steps, panes, models)
//   F                   the selection's next state (shapes: world.js
//                       BLOCK_SHAPES states; models: object states, none yet)
//   Ctrl + S            save the level (the loaded region's, levels/<name>.bxr;
//                       dev server only)
//   Tab                 back to the game
//
// With Exit or Spawn picked in the panel (Markers), clicks place those
// instead, on the ground block's top face - the cell the party stands on:
//   left click          a spawn of the name in the panel (moves it if it is
//                       there already); an exit of one cell to the region and
//                       spawn in the panel
//   Shift + left drag   an exit over the box of cells
//   right click         removes the exit or spawns on that cell
// Exits and spawns are drawn while editing (cyan, yellow) and saved in the
// region's header (core/exits.js).
//
// A model placed on a top face stands on it; on a side face it hangs on that
// wall, its north side against it (world.js mountWorldModel).
//
// Edits go through world.js (World, addWorldModel, ...) and then the one hook
// main.js hands in, onEdit, which rebuilds what depends on the world: the
// terrain instances, the field, the mirrors, the model lights.
import * as THREE from 'three/webgpu';
import { World, getVoxelKey, BLOCK_METRES, Y_MIN, Y_MAX, GRID, partBoxes, partSpan, partFacing, isSlope,
         surfaceLevel, tileParts, withPart, withoutPart, gridAt,
         WorldModels, addWorldModel, mountWorldModel, removeWorldModel,
         serializeLevel, DEV_LEVEL, Region, nextShapeState } from '../core/world.js';
import { exitBox, exitsWithout } from '../core/exits.js';
import { encodeRegion } from '../core/region.js';
import { MATERIALS } from '../core/materials.js';
import { MODELS } from '../core/models.js';
import { worldInstancedMesh, instanceInfo } from '../render/render.js';
import { createPalette } from './editorPanel.js';
import { placeModel as placeModelBoxes } from '../core/bbmodel.js';

const FLY_SPEED = 10;          // m/s; Shift x3
const LOOK_SPEED = 0.004;      // radians per pixel
const DRAG_PIXELS = 4;         // a right press that moves less is a click
const MAX_BOX_CELLS = 20000;   // one fill or clear at most
const MAX_ROW_CELLS = 256;     // one row at most, each way
const FACING_ORDER = ['N', 'E', 'S', 'W'];

// levelName: the level saved to, asked at each save - the loaded region's.
export function createEditor({ scene, camera, dom, onEdit, onToggle, levelName = () => DEV_LEVEL }) {
  const state = {
    active: false,
    // The selection: a block (material, shape and its state) or a model; a
    // facing for both.
    kind: 'block', materialId: MATERIALS[0].id, shape: 'full', shapeState: 'full', model: null, facing: 0,
    // Markers (kind 'exit' or 'spawn'): where an exit leads, a spawn's name.
    exitTarget: '', exitSpawn: 'start', spawnName: 'start',
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
  // slope's wedge, the steps' treads, a model's cubes as placed; removing, the
  // part that would go; dragging a box, the box. Line segments rebuilt
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
  // A part's outline in cell (x, y, z): its boxes, or a slope's wedge - the
  // plane, as it is drawn, not the staircase the field bakes.
  function partEdges(part, cell, out) {
    const base = [(cell[0] - 0.5) * B, (cell[1] - 0.5) * B, (cell[2] - 0.5) * B];
    if (!isSlope(part)) {
      for (const [x0, x1, y0, y1, z0, z1] of partBoxes(part)) {
        boxEdges([base[0] + x0 * B, base[1] + y0 * B, base[2] + z0 * B],
                 [base[0] + x1 * B, base[1] + y1 * B, base[2] + z1 * B], out);
      }
      return;
    }
    // The wedge, climbing +x (facing E), then turned as partBoxes turns boxes.
    const facing = partFacing(part);
    const turn = ([x, y, z]) => facing === 'W' ? [1 - x, y, z] : facing === 'S' ? [z, y, x]
                              : facing === 'N' ? [z, y, 1 - x] : [x, y, z];
    const P = (x, y, z) => turn([x, y, z]).map((v, a) => base[a] + v * B);
    const [lo, hi] = partSpan(part);
    const prof = [[0, lo], [1, lo], [1, hi], [0, lo]];
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

  // --- Exits and spawns, drawn while editing ---
  //
  // An exit: the standing room over its cells, two blocks tall; a spawn: a
  // post on its cell. Rebuilt when they change and when the editor opens.
  const markerLines = new THREE.LineSegments(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, transparent: true }));
  markerLines.renderOrder = 9;
  markerLines.visible = false;
  markerLines.raycast = () => {};
  markerLines.frustumCulled = false;
  scene.add(markerLines);
  const EXIT_COLOUR = new THREE.Color(0x55ccff), SPAWN_COLOUR = new THREE.Color(0xffd060);
  // The room above ground cells [x0, y0, z0, x1, y1, z1], h blocks tall,
  // inset by `inset` of a block on each side.
  function standingBox(c, h, inset, out) {
    boxEdges([(c[0] - 0.5 + inset) * B, (c[1] + 0.5) * B, (c[2] - 0.5 + inset) * B],
             [(c[3] + 0.5 - inset) * B, (c[4] + 0.5 + h) * B, (c[5] + 0.5 - inset) * B], out);
  }
  function drawMarkers() {
    const pts = [], cols = [];
    const add = (from, colour) => { for (let i = from; i < pts.length; i += 3) cols.push(colour.r, colour.g, colour.b); };
    for (const e of Region.exits) { const n = pts.length; standingBox(e.cells, 2, 0, pts); add(n, EXIT_COLOUR); }
    for (const at of Object.values(Region.spawns)) {
      const n = pts.length;
      standingBox([...at, ...at], 1.5, 0.35, pts);
      add(n, SPAWN_COLOUR);
    }
    markerLines.geometry.dispose();
    markerLines.geometry = new THREE.BufferGeometry();
    markerLines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
    markerLines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(cols, 3));
    markerLines.visible = state.active && pts.length > 0;
  }
  const isMarker = () => state.kind === 'exit' || state.kind === 'spawn';
  // A ground cell for a marker: the block whose top face is under the cursor.
  const groundOf = target => (target && !target.model && target.normal[1] > 0 ? target.key : null);

  // --- Targeting ---
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  // What is under the screen point: { place: cell [x,y,z] (the cell across
  // the drawn box's face), key: the block hit, normal, model: the placed model
  // hit, if it was one, part: the tile's part hit, grid: the Grid cell
  // against the face - { cell, at: [x, y, z] of the tile } }.
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
    const q = [0, 1, 2].map(a => hit.point.getComponent(a) / BLOCK_METRES + n.getComponent(a) * 0.01);
    const gcell = q.map(Math.round);
    const at = q.map((v, a) => Math.max(0, Math.min(GRID - 1, Math.floor((v - gcell[a] + 0.5) * GRID))));
    return { place, key, normal: [n.x, n.y, n.z], model: info.model, part: info.part, grid: { cell: gcell, at } };
  }

  // What a left click on this target would place - the one answer both the
  // outline and the click use. null: nothing can go there.
  const WALL_YAW = { N: 0, W: 1, S: 2, E: 3 };   // as world.js mountWorldModel
  // The selection as a part at Grid cell g of a tile (world.js gridAt).
  function selectedPart(g) {
    const part = { materialId: state.materialId, shape: state.shape, state: state.shapeState,
                   facing: FACING_ORDER[state.facing], at: 0 };
    part.at = gridAt(part, g);
    return part;
  }
  // The tile a part goes in, and the block it makes there (world.js
  // withPart): the Grid cell against the face, or - full there, or no room -
  // the tile across the drawn box's face, at its near side.
  function blockPlan(target) {
    const tries = [[target.grid.cell, target.grid.at]];
    const g = target.grid.at.map((v, a) => target.normal[a] > 0 ? 0 : target.normal[a] < 0 ? GRID - 1 : v);
    if (target.place.some((v, a) => v !== target.grid.cell[a])) tries.push([target.place, g]);
    for (const [cell, at] of tries) {
      if (!inBounds(cell[1])) continue;
      const part = selectedPart(at);
      const block = withPart(World.get(getVoxelKey(...cell)), part);
      if (block) return { kind: 'block', cell, part, block };
    }
    return null;
  }
  // The tile a row or box starts from: the one the first click would place in.
  const startCell = t => { const p = state.kind === 'block' && planPlacement(t); return p ? p.cell : t.place; };
  function planPlacement(target) {
    if (!target) return null;
    if (state.kind === 'block') return target.model ? null : blockPlan(target);
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

  // A row: the drag picks the axis whose on-screen direction it follows
  // most closely, and runs from the start cell to where the cursor's ray
  // passes nearest that axis's line through it - so a row can climb into the
  // air, where there is no face to point at.
  function rowEnd(start, clientX, clientY) {
    const W = innerWidth, H = innerHeight;
    const toPx = v => { v.project(camera); return [(v.x + 1) / 2 * W, (1 - v.y) / 2 * H]; };
    const c = new THREE.Vector3(...start).multiplyScalar(B);
    const [sx, sy] = toPx(c.clone());
    const mx = clientX - sx, my = clientY - sy, m = Math.hypot(mx, my);
    if (m < 1) return start;
    let axis = -1, bestCos = -1;
    for (let a = 0; a < 3; a++) {
      const e = c.clone(); e.setComponent(a, e.getComponent(a) + B);
      const [ex, ey] = toPx(e);
      const len = Math.hypot(ex - sx, ey - sy);
      if (len < 2) continue;                     // seen end-on: no direction to follow
      const cos = Math.abs((mx * (ex - sx) + my * (ey - sy)) / (len * m));
      if (cos > bestCos) { bestCos = cos; axis = a; }
    }
    if (axis < 0) return start;
    ndc.set((clientX / W) * 2 - 1, -(clientY / H) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const { origin: o, direction: d } = raycaster.ray;
    // Closest approach of the ray o + t d to the line c + s u (u the axis):
    // s = (u.w - (d.u)(d.w)) / (1 - (d.u)^2), w = o - c.
    const w = o.clone().sub(c);
    const du = d.getComponent(axis), denom = 1 - du * du;
    if (denom < 1e-6) return start;
    const s = (w.getComponent(axis) - du * d.dot(w)) / denom;
    const n = Math.max(-MAX_ROW_CELLS, Math.min(MAX_ROW_CELLS, Math.round(s / B)));
    const end = start.slice(); end[axis] += n;
    return end;
  }

  function showCursor(target) {
    const d = state.drag;
    if (isMarker()) {
      // The cell, or the box of cells a Shift drag makes an exit of.
      const g = groundOf(target), pts = [];
      const from = d && d.shift && d.button === 0 && state.kind === 'exit' ? groundOf(d.start) : null;
      if (g) standingBox(from ? exitBox(from, g) : [...g, ...g], state.kind === 'exit' ? 2 : 1.5, 0, pts);
      setOutline(pts, d && d.button === 2 ? 0xff6666 : state.kind === 'exit' ? 0x55ccff : 0xffd060);
      return;
    }
    if (d && d.row) {
      const a = startCell(d.start), b = d.row, pts = [];
      boxEdges(a.map((v, i) => (Math.min(v, b[i]) - 0.5) * B), a.map((v, i) => (Math.max(v, b[i]) + 0.5) * B), pts);
      setOutline(pts, 0x66ff88);
      return;
    }
    if (!target) { cursor.visible = false; return; }
    const removing = d ? d.button === 2 : false;
    const pts = [];
    if (d && d.shift && d.start) {
      // A box: its whole extent, in cells.
      const a = removing ? d.start.key : startCell(d.start);
      const b = removing ? target.key : target.place;
      const lo = a.map((v, i) => (Math.min(v, b[i]) - 0.5) * B), hi = a.map((v, i) => (Math.max(v, b[i]) + 0.5) * B);
      boxEdges(lo, hi, pts);
    } else if (removing) {
      // What would go: the model hit, or the part of the tile.
      if (target.model) cubeEdges(target.model.cubes, pts);
      else {
        const part = tileParts(World.get(getVoxelKey(...target.key)))[target.part || 0];
        if (part) partEdges(part, target.key, pts);
      }
    } else {
      const plan = planPlacement(target);
      if (plan && plan.kind === 'block') partEdges(plan.part, plan.cell, pts);
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
  // The selection in cell c, at the Grid cell of the drag's first tile
  // (part.at), beside whatever is there; false when it does not fit.
  function placeBlockAt(c, part) {
    const k = getVoxelKey(...c);
    if (!inBounds(c[1])) return false;
    const block = withPart(World.get(k), part);
    if (!block) return false;
    World.set(k, block);
    return true;
  }
  function removeBlockAt(c) {
    const k = getVoxelKey(...c);
    if (!World.has(k)) return false;
    removeModelsOn(c);
    World.delete(k);
    return true;
  }
  // A click's placement: the plan's block in its tile. Blocks are shared
  // types (core/chunks.js): a new block is set, never one changed in place.
  function placeAt(target) {
    const plan = planPlacement(target);
    if (!plan) return false;
    World.set(getVoxelKey(...plan.cell), plan.block);
    return true;
  }
  // Right click on a tile of parts: only the part under the cursor goes.
  // Models on it go when its top drops.
  function removePart(c, i) {
    const key = getVoxelKey(...c);
    const block = withoutPart(World.get(key), i);
    if (!block) return removeBlockAt(c);
    const top = surfaceLevel(...c);
    World.set(key, block);
    if (surfaceLevel(...c) !== top) removeModelsOn(c);
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
  // --- Exits and spawns ---
  function placeExit(cells) {
    const target = state.exitTarget.trim();
    if (!target) { status('name the region this exit leads to (Markers)'); return; }
    const spawn = state.exitSpawn.trim() || 'start';
    // One exit per cell: a new one replaces any it overlaps.
    Region.exits = [...exitsWithout(Region.exits, cells), { cells, target, spawn }];
    drawMarkers();
    status(`exit to ${target} (${spawn}) - Ctrl+S to save`);
  }
  function placeSpawn(cell) {
    const name = state.spawnName.trim() || 'start';
    Region.spawns = { ...Region.spawns, [name]: cell.slice() };
    drawMarkers();
    status(`spawn ${name} at ${cell.join(', ')} - Ctrl+S to save`);
  }
  function removeMarkersAt(cell) {
    const box = [...cell, ...cell];
    const exits = exitsWithout(Region.exits, box);
    const spawns = Object.fromEntries(Object.entries(Region.spawns)
      .filter(([, at]) => at[0] !== cell[0] || at[1] !== cell[1] || at[2] !== cell[2]));
    const n = Region.exits.length - exits.length + Object.keys(Region.spawns).length - Object.keys(spawns).length;
    Region.exits = exits;
    Region.spawns = spawns;
    drawMarkers();
    status(n ? `removed ${n} marker${n > 1 ? 's' : ''} - Ctrl+S to save` : 'no exit or spawn on that cell');
  }
  // A marker click or drag, on release. true when it was one.
  function markerRelease(d, target) {
    if (!isMarker()) return false;
    const g = groundOf(target);
    if (d.moved && !(d.shift && d.button === 0)) return true;   // a look-around drag
    if (!g) { status('exits and spawns go on a top face'); return true; }
    if (d.button === 2) removeMarkersAt(g);
    else if (state.kind === 'spawn') placeSpawn(g);
    else {
      const from = d.shift ? groundOf(d.start) : null;
      placeExit(from ? exitBox(from, g) : [...g, ...g]);
    }
    return true;
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
    if (k === 'f' && !e.altKey && !e.ctrlKey && !e.metaKey) {
      if (state.kind === 'block') {
        const next = nextShapeState(state.shape, state.shapeState);
        if (next !== state.shapeState) { state.shapeState = next; renderPanel(); showCursor(state.hover); }
        else status(`${state.shape} has one state`);
      } else if (state.kind === 'model') status('models have no states yet');
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
                   start: e.shiftKey || e.button === 0 ? targetAt(e.clientX, e.clientY) : null,
                   row: null };
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
    if (d && d.moved && d.button === 0 && !d.shift && d.start && state.kind === 'block') {
      d.row = rowEnd(startCell(d.start), e.clientX, e.clientY);
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
    if (markerRelease(d, target)) {
      showCursor(target);
      return;
    }
    // Rows and boxes take the part the first tile would: the same Grid cell in each.
    const startPlan = d.start && state.kind === 'block' ? planPlacement(d.start) : null;
    if (d.row) {
      let n = 0;
      if (startPlan) for (const c of boxCells(startPlan.cell, d.row)) n += placeBlockAt(c, startPlan.part);
      if (n) edited(`placed a row of ${n}`);
    } else if (d.shift && d.start && target) {
      let n = 0;
      if (d.button === 0 && startPlan) {
        const endPlan = planPlacement(target);
        for (const c of boxCells(startPlan.cell, endPlan ? endPlan.cell : target.place)) n += placeBlockAt(c, startPlan.part);
      } else if (d.button === 2) {
        for (const c of boxCells(d.start.key, target.key)) n += removeBlockAt(c);
      }
      if (n) edited(`${d.button === 0 ? 'filled' : 'cleared'} ${n} cells`);
    } else if (!d.moved && target) {
      let ok = false;
      if (d.button === 0) ok = state.kind === 'block' ? placeAt(target) : placeModel(target);
      else if (target.model) ok = removeWorldModel(target.model);
      else ok = removePart(target.key, target.part || 0);
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
      const name = levelName();
      const res = await fetch(`/__levels?name=${encodeURIComponent(name)}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' },
        body: await encodeRegion(serializeLevel())
      });
      status(res.ok ? `saved levels/${name}.bxr` : `save failed: ${await res.text()}`);
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
    drawMarkers();
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
