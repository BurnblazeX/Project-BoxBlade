// --- Occlusion cut-out and interiors (core/occlusion.js, core/interior.js, gpu/occlusion.js) ---
//
// Two sets of blocks fade, each decided here on the CPU and eased per block -
// so nothing pops as the camera moves, blocks fade in and out:
//
//   the cut-out   while the player is wholly hidden (for CUT_DELAY), the
//                 blocks between them and the camera (cutTargets), shown cut
//                 in a circle on screen around them: translucent, by a smooth
//                 radial gradient - clear inside, solid at the edge
//   interiors     under a roof (for ROOF_DELAY), the building's roof and
//                 storeys above (interiorTargets), translucent: gone near the
//                 player on screen, 50% far. The cut-out's circle is off
//                 there - the roof's fade is the whole effect. The walls
//                 toward the camera can fade with the roof too
//                 (bxb.interiors('walls')); asked every frame, so a camera
//                 turn fades one wall in as the next fades out
//
// Both live in one mask texture of cells that follows the player
// (gpu/occlusion.js cellMask), uploaded only while a cell is easing. Every
// opaque terrain material carries the mask and the face reveal; this feature
// writes the cells and the uniforms, once a frame before the first terrain
// pass - so switching it, or changing the circle, rebuilds nothing. Basic
// mode and the effects alike.
//
// The translucent part is a twin of the terrain mesh - its geometry and
// instance buffers, with the showing shader set's own full shading (fadeMat,
// render/features/texelcache.js), or the plain look in the basic mode, so a
// block looks the same solid or fading - drawing after the opaque
// pass what it left out, blended, in two draws: depth first, then colour
// where the depth matches, so only the nearest faded face shows, evenly,
// rather than every layer of a building stacked.
//
// Not a replacement for the characters' on-top variant (render/sprites.js
// updateCharacterClipping): that is for a quad leaning INTO a block - most
// often the wall behind, which this never cuts - and draws the enemy too.
//
// ctx:
//   renderer, scene, camera
//   fadeMat()  the showing shader set's fade material, or null (the basic mode)
//   feet()   the player's feet, world metres (the sprite's position)
import * as THREE from 'three/webgpu';
import { DEFAULT_OCCLUSION, CUT_DELAY, occlusionFrame, playerHidden, cutTargets } from '../../core/occlusion.js';
import { DEFAULT_INTERIOR, ROOF_DELAY, playerCell, maskOrigin, buildInterior, interiorTargets, easeCells,
         rekeyCells } from '../../core/interior.js';
import { World } from '../../core/world.js';
import { occlusion, cellMask, interiorFadeTSL } from '../../gpu/occlusion.js';
import { terrainKeepTSL } from '../../gpu/terrain.js';
import { worldInstancedMesh, plainTerrainMaterial } from '../render.js';
import { flip } from '../engine.js';

// Drawn after the opaque pass, before the characters' on-top variant (10).
const FADE_ORDER = 4;

export function createOcclusion({ renderer, scene, camera, feet, fadeMat = () => null }) {
  let on = true, hidden = false, hiddenFor = 0, lastT = 0;
  const params = { ...DEFAULT_OCCLUSION };
  const forward = new THREE.Vector3(), ndc = new THREE.Vector3();
  const right = new THREE.Vector3(), size = new THREE.Vector2();
  // Interiors: on, and the building found for the player's cell.
  let roofsOn = true, frontWalls = false, room = null, underFor = 0;
  const iparams = { ...DEFAULT_INTERIOR };
  // The mask: where it is (it follows the player's cell), each set's eased
  // values by mask index, and the indices written at the last upload.
  let cellKey = '', worldSize = -1, origin = null, written = [], recentred = false;
  const roofCells = new Map(), cutCells = new Map();

  // The fade twin: [depth, colour] meshes, made with the terrain's.
  let fade = null, plainColour = null;
  function syncFade(show) {
    const src = worldInstancedMesh;
    if (!src || !scene) return;
    if (!fade) {
      const depth = new THREE.MeshBasicNodeMaterial({ colorWrite: false, transparent: true, depthWrite: true });
      depth.maskNode = interiorFadeTSL().mask;
      plainColour = plainTerrainMaterial({ transparent: true, depthWrite: false });
      const f = interiorFadeTSL();
      plainColour.maskNode = f.mask;
      plainColour.opacityNode = f.opacity;
      plainColour.positionNode = terrainKeepTSL(false, false);
      depth.positionNode = terrainKeepTSL(false, false);
      fade = [depth, plainColour].map((mat, k) => {
        const m = new THREE.InstancedMesh(src.geometry, mat, src.count);
        m.instanceMatrix = src.instanceMatrix;
        m.instanceColor = src.instanceColor;
        m.frustumCulled = false;
        m.receiveShadow = true;
        m.raycast = () => {};          // picking stays on the terrain mesh
        m.renderOrder = FADE_ORDER + k;
        scene.add(m);
        // Compiled now, not on the first fade: shown for the call's
        // synchronous part, which is where compileAsync collects what to build.
        try { renderer.compileAsync?.(m, camera, scene)?.catch(() => {}); }
        finally { m.visible = false; }
        return m;
      });
    }
    for (const m of fade) { m.count = src.count; m.visible = show && src.visible; }
    fade[1].material = fadeMat() || plainColour;
  }

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
      for (const cells of [roofCells, cutCells]) {
        const moved = rekeyCells(cells, origin, o);
        cells.clear();
        for (const [i, v] of moved) cells.set(i, v);
      }
      recentred = true;
    }
    origin = o;
    room = buildInterior(cell);
  }

  // Both sets' targets, eased per block, and the mask uploaded when any moved.
  function updateCells(f, fr, dt) {
    follow(f);
    const hl = Math.hypot(forward.x, forward.z) || 1;
    underFor = roofsOn && room ? underFor + dt : 0;
    const under = underFor >= ROOF_DELAY;
    const roofT = under ? interiorTargets(room, f, { x: forward.x / hl, z: forward.z / hl }, frontWalls) : new Map();
    // Under a roof the roof's fade is the whole effect: no circle on top.
    const cutT = on && hiddenFor >= CUT_DELAY && !under ? cutTargets(fr, params, origin) : new Map();
    const movedR = easeCells(roofCells, roofT, dt);
    const movedC = easeCells(cutCells, cutT, dt);
    syncFade(roofCells.size > 0 || cutCells.size > 0);
    if (!movedR && !movedC && !recentred) return;
    recentred = false;
    const data = cellMask.data;
    for (const i of written) data[i] = 0;
    written = [];
    for (const [cells, ch] of [[roofCells, 0], [cutCells, 1]]) {
      for (const [i, v] of cells) { data[2 * i + ch] = Math.round(v * 255); written.push(2 * i + ch); }
    }
    cellMask.tex.needsUpdate = true;
    cellMask.origin.value.set(origin.x, origin.y, origin.z);
  }

  function update(now = performance.now()) {
    const u = occlusion;
    const dt = lastT ? Math.min(0.1, (now - lastT) / 1000) : 0;
    lastT = now;
    camera.updateMatrixWorld();
    const f = feet();
    camera.getWorldDirection(forward);
    right.setFromMatrixColumn(camera.matrixWorld, 0);
    right.y = 0;
    right.normalize();
    hidden = on && playerHidden(camera.position, f, { x: right.x, z: right.z });
    hiddenFor = hidden ? hiddenFor + dt : 0;
    const fr = occlusionFrame(camera.position, forward, f);
    updateCells(f, fr, dt);
    // The circle on screen.
    const depth = Math.max(0.1, fr.playerDepth);
    renderer.getDrawingBufferSize(size);
    const pxPerMetre = size.y / (2 * depth * Math.tan(camera.fov * Math.PI / 360));
    ndc.set(fr.centre.x, fr.centre.y, fr.centre.z).project(camera);
    u.centrePx.value.set((ndc.x + 1) / 2 * size.x, (1 - ndc.y) / 2 * size.y);
    u.radiusPx.value = params.radius * pxPerMetre;
    u.fadePx.value = Math.min(params.fade, params.radius) * pxPerMetre;
    u.roofNearPx.value = iparams.near * pxPerMetre;
    u.roofFarPx.value = iparams.far * pxPerMetre;
    u.fullSize.value.copy(size);
    u.eye.value.copy(camera.position);
    u.forward.value.copy(forward);
    u.playerDepth.value = fr.playerDepth;
    u.feetY.value = fr.feetY;
    u.centreXZ.value.set(fr.centre.x, fr.centre.z);
    const hl2 = Math.hypot(forward.x, forward.z) || 1;
    u.aheadXZ.value.set(forward.x / hl2, forward.z / hl2);
  }

  const describe = () => (on ? `occlusion cut-out on: radius ${params.radius} m, fade ${params.fade} m` +
    ` (player ${hidden ? 'hidden - cutting' : 'in view - nothing cut'})` : 'occlusion cut-out off') +
    `; interiors ${roofsOn ? `on: ${iparams.near}-${iparams.far} m` +
      (room ? `, under a roof (${room.roof.length} roof cells, ${room.walls.length} wall cells)` : ', not under a roof')
      : 'off'}`;
  const commands = {
    occlusion: {
      help: 'dither away the terrain between the camera and the player, in a circle around them, while it hides them',
      usage: 'bxb.occlusion()  toggles  |  bxb.occlusion(radius, fade)  in metres',
      run: (radius = null, fade = null) => {
        if (radius === null) on = !on;
        else {
          params.radius = Math.max(0.1, Number(radius) || DEFAULT_OCCLUSION.radius);
          if (fade !== null) params.fade = Math.max(0, Number(fade) || 0);
        }
        return describe();
      }
    },
    interiors: {
      help: "under a roof, fade the roof and the storeys above - gone near the player on screen, 50% from far " +
            "(the cut-out's circle is off there); 'walls' also fades the walls toward the camera",
      usage: "bxb.interiors()  toggles  |  bxb.interiors(near, far)  in metres  |  bxb.interiors('walls')",
      run: (near = null, far = null) => {
        if (near === null) roofsOn = !roofsOn;
        else if (near === 'walls') frontWalls = !frontWalls;
        else {
          iparams.near = Math.max(0, Number(near) || 0);
          if (far !== null) iparams.far = Math.max(iparams.near, Number(far) || 0);
        }
        return describe();
      }
    }
  };

  return {
    name: 'occlusion',
    update,
    get on() { return on; },
    commands,
    settings: { title: 'Occlusion cut-out', settings: [
      { key: 'occlusion', label: 'Cut out what hides the player', type: 'toggle',
        help: 'terrain between the camera and the player dithers away around them',
        get: () => on, set: v => flip(on, v, () => commands.occlusion.run()) },
      { key: 'occradius', label: 'Radius (m)', type: 'range', min: 0.5, max: 6, step: 0.1,
        get: () => params.radius, set: v => commands.occlusion.run(v, params.fade) },
      { key: 'occfade', label: 'Fade width (m)', type: 'range', min: 0, max: 3, step: 0.1,
        help: 'from fully cut to solid, at the circle\'s edge',
        get: () => params.fade, set: v => commands.occlusion.run(params.radius, v) },
      { key: 'interiors', label: 'See through roofs and front walls', type: 'toggle',
        help: 'under a roof: the roof, the storeys above and the walls toward the camera fade',
        get: () => roofsOn, set: v => flip(roofsOn, v, () => commands.interiors.run()) },
      { key: 'intwalls', label: 'Interiors: fade the front walls too', type: 'toggle',
        help: 'the walls between the player and the camera fade with the roof',
        get: () => frontWalls, set: v => flip(frontWalls, v, () => commands.interiors.run('walls')) },
      { key: 'intnear', label: 'Interiors: fully clear within (m)', type: 'range', min: 0, max: 10, step: 0.5,
        get: () => iparams.near, set: v => commands.interiors.run(v, Math.max(v, iparams.far)) },
      { key: 'intfar', label: 'Interiors: 50% from (m)', type: 'range', min: 1, max: 30, step: 0.5,
        get: () => iparams.far, set: v => commands.interiors.run(Math.min(iparams.near, v), v) }
    ]}
  };
}
