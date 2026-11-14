import * as THREE from 'three/webgpu';
import { BLOCK_METRES, WorldModels, forEachOccupant, surfaceHeight, gridSolidCells, GRID, GRID_METRES,
         BODY_HALF, BODY_HEIGHT } from '../core/world.js';
import { TEXEL_METRES } from '../core/texelcache.js';

// --- Collision overlay (Alt+H) ---
//
// What blocks movement, drawn over the world. Props collide on the Grid: a
// model's bounding_box elements take every half-metre Grid cell they reach
// into, and a character's body - a box round its point - goes where it meets
// none (world.js bodyFits). Characters among themselves, and battle, are
// still tile by tile (occupantAt). So:
//   red     a tile a character holds, one block tall from its floor
//   orange  the Grid cells props take, outlined as one shape a tile
//   yellow  each model's bounding boxes, as authored
//   green   each character's body, and its point
// Drawn through everything; rebuilt each frame while on, only when it changed.
const COLOURS = { entity: new THREE.Color(0xff4444), prop: new THREE.Color(0xff9933),
                  bounds: new THREE.Color(0xffee44), point: new THREE.Color(0x44ff66) };
const B = BLOCK_METRES;

export function createHitboxView(scene, sprites) {
  const lines = new THREE.LineSegments(new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, transparent: true }));
  lines.renderOrder = 11;
  lines.visible = false;
  lines.raycast = () => {};
  scene.add(lines);
  let last = '';

  const pos = [], col = [];
  const line = (a, b, c) => { pos.push(...a, ...b); col.push(c.r, c.g, c.b, c.r, c.g, c.b); };
  function box(lo, hi, c) {
    const [x0, y0, z0] = lo, [x1, y1, z1] = hi;
    const p = [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
               [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]];
    for (const [i, j] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
                          [0, 4], [1, 5], [2, 6], [3, 7]]) line(p[i], p[j], c);
  }

  // The cells as one shape a tile: only the edges where what they take meets
  // what they don't - a crease or a corner - not each cell's twelve. An edge
  // has four cells round it; it is drawn when one or three hold it, or two
  // across a diagonal (two side by side make a flat face). Cells of another
  // tile count as empty, so tiles keep their own outlines. Cell g spans
  // [g, g + 1) Grid cells from half a block below tile 0's middle.
  function outline(cells, c) {
    const G = GRID_METRES, o = -B / 2, tileOf = g => Math.floor(g / GRID);
    const has = new Set(cells), drawn = new Set();
    for (const k of has) {
      const v = k.split(',').map(Number), tile = v.map(tileOf).join();
      const held = p => has.has(p.join()) && p.map(tileOf).join() === tile ? 1 : 0;
      for (let a = 0; a < 3; a++) {
        const b = (a + 1) % 3, d = (a + 2) % 3;
        for (let eb = 0; eb < 2; eb++) for (let ed = 0; ed < 2; ed++) {
          const p = v.slice(); p[b] += eb; p[d] += ed;
          const key = `${tile}|${a}|${p}`;
          if (drawn.has(key)) continue;
          drawn.add(key);
          const at = (sb, sd) => { const q = p.slice(); q[b] += sb; q[d] += sd; return held(q); };
          const n = [at(-1, -1), at(0, -1), at(0, 0), at(-1, 0)];
          const sum = n[0] + n[1] + n[2] + n[3];
          if (sum === 1 || sum === 3 || (sum === 2 && n[0] === n[2])) {
            const q = p.slice(); q[a] += 1;
            line(p.map(u => o + u * G), q.map(u => o + u * G), c);
          }
        }
      }
    }
  }

  function update() {
    if (!lines.visible) return;
    pos.length = 0; col.length = 0;
    forEachOccupant(key => {
      const [x, y, z] = key.split(',').map(Number);
      const floor = surfaceHeight(x, y, z);
      box([(x - 0.5) * B, floor, (z - 0.5) * B], [(x + 0.5) * B, floor + B, (z + 0.5) * B], COLOURS.entity);
    });
    outline(gridSolidCells(), COLOURS.prop);
    for (const m of WorldModels) for (const b of m.bounds || []) {
      box(b.min.map(v => v * TEXEL_METRES), b.max.map(v => v * TEXEL_METRES), COLOURS.bounds);
    }
    const r = 0.2;
    for (const s of sprites()) {
      if (!s.visible) continue;
      const { x, y, z } = s.position;
      box([x - BODY_HALF, y, z - BODY_HALF], [x + BODY_HALF, y + BODY_HEIGHT, z + BODY_HALF], COLOURS.point);
      line([x - r, y, z], [x + r, y, z], COLOURS.point);
      line([x, y, z - r], [x, y, z + r], COLOURS.point);
      line([x, y, z], [x, y + r, z], COLOURS.point);
    }
    const sig = pos.join();
    if (sig === last) return;
    last = sig;
    lines.geometry.dispose();
    lines.geometry = new THREE.BufferGeometry();
    lines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    lines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  }

  function toggle() {
    lines.visible = !lines.visible;
    last = '';
    update();
    return lines.visible
      ? '[hitboxes] on - red: a character\'s tile, orange: Grid cells props take, yellow: model bounding boxes, green: character bodies'
      : '[hitboxes] off';
  }
  return { update, toggle };
}
