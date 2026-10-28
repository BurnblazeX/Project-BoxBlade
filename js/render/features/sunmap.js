// --- The sun depth map: drawing it (core/sunmap.js has the why) ---
//
// The terrain - blocks and placed models, glass left out - drawn once from the
// sun into gpu/sunmap.js's target, through a twin of the terrain mesh that
// shares its geometry and instance buffers (as the texel cache's twins do), in
// a scene of its own so nothing else is drawn with it. Only when it is stale:
// the sun turned past SUN_MAP_REDRAW, the world changed, the arena hid or
// showed blocks, the preset changed the softening, or - over a world wider
// than SUN_MAP_WINDOW - the player left the middle of the map's window
// (World plan P2a: the map follows the player, it does not cover the region).
// Its texel follows the softening grid (sunMapTexelFor), its size the window.
//
// ctx:
//   renderer, sunLight
//   shadowsOn()  whether the marched shading is on - the map is read only there
//   focus()      the block the player stands on ({ x, y, z }, grid units)
//
// Settings (the Shadows section; the presets set them):
//   farshadow   'off' | 'hard' | 4 | 6 | 8 - the map's softening inside C2:
//               the largest grid of bilinear taps a side, each shade using as
//               few as its penumbra needs ('hard': one smooth tap everywhere;
//               'off': no map, the march alone at its 12 m, as before)
//   sunreach    '12m' | 'c1' - how far the soft march (the cone's analytic
//               penumbra, the wanted look) runs before the map takes over;
//               never into C2, whose 0.5 m voxels step a shadow's edge
import * as THREE from 'three/webgpu';
import { worldInstancedMesh } from '../render.js';
import { sunMapBounds, sunMapFrame, sunMapStale, sunMapMoved, sunMapTexelFor, sunMapSizeFor,
         SUN_MAP_WINDOW } from '../../core/sunmap.js';
import { BLOCK_METRES } from '../../core/world.js';
import { sunMap, sunSoftReach, sunWorldTop, createSunMapMaterial } from '../../gpu/sunmap.js';

const TAP_OPTIONS = ['off', 'hard', 4, 6, 8];
const REACH_OPTIONS = ['12m', 'c1'];

export function createSunMap({ renderer, sunLight, shadowsOn, focus }) {
  let mode = 4;                 // TAP_OPTIONS
  let drawnDir = null;          // the sun the map holds
  let drawnFrame = null;        // the frame it was drawn in, for sunMapMoved
  let bounds;                   // undefined: not measured since the last change
  let dirty = true;
  let scene = null, twin = null;
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const _clear = new THREE.Color();
  let draws = 0;

  function ensureTwin() {
    const src = worldInstancedMesh;
    if (!src) return null;
    if (twin && twin.geometry === src.geometry) return twin;
    scene = scene || new THREE.Scene();
    if (twin) scene.remove(twin);
    twin = new THREE.InstancedMesh(src.geometry, createSunMapMaterial(), src.count);
    twin.instanceMatrix = src.instanceMatrix;
    twin.frustumCulled = false;
    twin.raycast = () => {};
    scene.add(twin);
    return twin;
  }

  // The world's box, measured once per change; its top also stops the soft
  // march early (gpu/trace.js), so it is kept current even with the map off.
  function measure() {
    if (bounds !== undefined) return;
    bounds = sunMapBounds();
    sunWorldTop.value = bounds ? bounds.max[1] : 1e9;
  }

  function draw() {
    const mesh = ensureTwin();
    measure();
    if (!mesh || !bounds) { sunMap.on.value = 0; return; }
    const texel = sunMapTexelFor(sunMap.side.value);
    const f = sunMapFrame(bounds, sunLight.position, focusMetres(), SUN_MAP_WINDOW, texel);
    const size = sunMapSizeFor(f, texel);
    const [cx, cy, cz] = f.centre;
    camera.left = -f.half; camera.right = f.half; camera.top = f.half; camera.bottom = -f.half;
    camera.near = f.near; camera.far = f.far;
    camera.up.set(...f.up);
    camera.position.set(cx + f.dir[0] * f.eye, cy + f.dir[1] * f.eye, cz + f.dir[2] * f.eye);
    camera.lookAt(cx, cy, cz);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld();
    if (sunMap.target.width !== size) sunMap.target.setSize(size, size);
    mesh.count = worldInstancedMesh.count;
    sunMap.dir.value.set(...f.dir);
    sunMap.base.value = f.base;
    sunMap.size.value = size;
    sunMap.texel.value = (2 * f.half) / size;
    sunMap.viewProj.value.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    // Cleared to 0: "nothing toward the sun" (every stored s is above it).
    renderer.getClearColor(_clear);
    const alpha = renderer.getClearAlpha();
    renderer.setClearColor(0x000000, 0);
    renderer.setRenderTarget(sunMap.target);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    renderer.setClearColor(_clear, alpha);
    drawnDir = f.dir;
    drawnFrame = f;
    dirty = false;
    draws++;
    sunMap.on.value = 1;
  }

  // The player's feet, metres - where a following map centres.
  function focusMetres() {
    const g = focus();
    return { x: g.x * BLOCK_METRES, y: g.y * BLOCK_METRES, z: g.z * BLOCK_METRES };
  }

  function apply() {
    sunMap.side.value = typeof mode === 'number' ? mode : 0;
    if (mode === 'off') sunMap.on.value = 0;
    else dirty = true;
  }

  const commands = {
    farshadow: {
      help: "bxb.farshadow('off' | 'hard' | 4 | 6 | 8) - the sun depth map past the march: off, " +
            'hard, or softened inside C2 by a grid of bilinear taps up to this many a side',
      run(v) {
        const want = TAP_OPTIONS.find(o => String(o) === String(v));
        if (want === undefined) return `one of ${TAP_OPTIONS.join(', ')}`;
        mode = want;
        apply();
        return `far shadows: ${mode}`;
      }
    },
    sunmap: {
      help: 'bxb.sunmap() - the sun depth map: size, texel, redraws',
      run() {
        return `sun map ${mode === 'off' ? 'off' : `${sunMap.size.value}², ${sunMap.texel.value.toFixed(3)} m a texel, ` +
                `${drawnFrame && drawnFrame.follows ? `following the player, ${SUN_MAP_WINDOW} m each way` : 'fitted to the world'}, ` +
                `grid up to ${sunMap.side.value}x${sunMap.side.value}, soft march to ` +
                `${REACH_OPTIONS[sunSoftReach.value]}, drawn ${draws} times`}`;
      }
    }
  };

  return {
    name: 'sunmap',
    // Per frame, before anything shades: redraw when stale.
    update() {
      measure();
      if (mode === 'off' || !shadowsOn()) return;
      if (dirty || sunMapStale(drawnDir, sunLight.position) || sunMapMoved(drawnFrame, focusMetres())) draw();
    },
    // The world changed (an edit, the arena hiding blocks): measure and draw again.
    invalidate() { bounds = undefined; sunWorldTop.value = 1e9; dirty = true; },
    commands,
    shadowSettings: [
      { key: 'farshadow', label: 'Far shadows', type: 'select',
        options: TAP_OPTIONS.map(o => ({ value: o, label: typeof o === 'number' ? `soft, up to ${o}x${o}` : o })),
        help: 'shadows past the soft march, from a sun depth map over the region: softened inside C2, ' +
              'hard past it - off: they stop at the march, as before',
        get: () => mode, set: v => commands.farshadow.run(v) },
      { key: 'sunreach', label: 'Soft sun reach', type: 'select',
        options: [{ value: '12m', label: '12 m' }, { value: 'c1', label: 'to C1 edge' }],
        help: 'how far the soft march (the cone penumbra) looks toward the sun before the far map ' +
              'takes over - never into C2, whose half-metre voxels step the edge; it stops early ' +
              'once the ray clears the tallest block',
        get: () => REACH_OPTIONS[sunSoftReach.value] || '12m',
        set: v => { const i = REACH_OPTIONS.indexOf(String(v)); if (i >= 0) sunSoftReach.value = i; } }
    ]
  };
}
