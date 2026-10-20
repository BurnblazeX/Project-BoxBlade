import * as THREE from 'three';
import { file, section, ok } from './lib/harness.mjs';
import { buildPickGrid, pickTop } from '../js/core/pickgrid.js';

// The grid walk (core/pickgrid.js) against what it replaced: three's raycast
// of the terrain InstancedMesh, first top face wins (render/picking.js tileAt).
file('picking.test.mjs - grid-walk pick matches the instanced raycast');

const S = 0.75;                     // cell size; any value, the terrain's is BLOCK_METRES
let seed = 12345;
const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32);

// A terrain as fillInstances lays it out: full blocks on a heightfield with
// walls, half and quarter boxes inside cells, model cubes at any texel offset
// (spanning cells), and some instances hidden at scale 0 (battle).
function makeWorld() {
  const geo = new THREE.BoxGeometry(S, S, S);
  const boxes = [];
  const put = (x0, y0, z0, x1, y1, z1, visible = true) => boxes.push({ x0, y0, z0, x1, y1, z1, visible });
  for (let x = 0; x < 20; x++) for (let z = 0; z < 20; z++) {
    const h = 1 + Math.floor(rand() * 4) + (x === 10 ? 4 : 0);
    for (let y = 0; y < h; y++) {
      const c = [(x - 0.5) * S, (y - 0.5) * S, (z - 0.5) * S];
      const r = rand();
      if (y === h - 1 && r < 0.15) put(c[0], c[1], c[2], c[0] + S, c[1] + S / 2, c[2] + S);          // bottom half
      else if (y === h - 1 && r < 0.25) put(c[0], c[1] + S / 2, c[2], c[0] + S / 2, c[1] + S, c[2] + S); // a quarter
      else put(c[0], c[1], c[2], c[0] + S, c[1] + S, c[2] + S, rand() > 0.05);
    }
    if (rand() < 0.1) {                                     // a model cube on top, off the grid
      const t = S / 16, ox = Math.floor(rand() * 16) * t, oz = Math.floor(rand() * 16) * t;
      const w = (2 + Math.floor(rand() * 12)) * t, base = (h - 0.5) * S;
      put((x - 0.5) * S + ox, base, (z - 0.5) * S + oz, (x - 0.5) * S + ox + w, base + w, (z - 0.5) * S + oz + w);
    }
  }
  const mesh = new THREE.InstancedMesh(geo, new THREE.MeshBasicMaterial(), boxes.length);
  const m = new THREE.Matrix4(), p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
  boxes.forEach((b, i) => {
    p.set((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2, (b.z0 + b.z1) / 2);
    if (b.visible) s.set((b.x1 - b.x0) / S, (b.y1 - b.y0) / S, (b.z1 - b.z0) / S); else s.set(0, 0, 0);
    mesh.setMatrixAt(i, m.compose(p, q, s));
  });
  mesh.updateMatrixWorld();
  return mesh;
}

function raycastTop(raycaster, mesh) {
  for (const hit of raycaster.intersectObject(mesh, false)) {
    if (!hit.face || hit.face.normal.y < 0.5) continue;
    return { id: hit.instanceId, t: hit.distance };
  }
  return { id: -1, t: Infinity };
}

const mesh = makeWorld();
const grid = buildPickGrid(mesh.instanceMatrix.array, mesh.count, S);
const raycaster = new THREE.Raycaster();
const instanceT = (id, o, d) => {     // where the ray meets that instance's top, for ties
  const m = new THREE.Matrix4(); mesh.getMatrixAt(id, m);
  const top = new THREE.Vector3().setFromMatrixPosition(m).y + m.elements[5] * S / 2;
  return (top - o.y) / d.y;
};

section('random rays from a game camera');
let rays = 0, hitsTop = 0;
const mismatches = [];
for (let n = 0; n < 4000; n++) {
  // From above and to the side, down at 15-80 degrees, as the orbit camera looks.
  // Aimed at a point over the world, from 5-25 m back along the ray.
  const yaw = rand() * Math.PI * 2, pitch = (15 + rand() * 65) * Math.PI / 180;
  const d = new THREE.Vector3(Math.cos(pitch) * Math.cos(yaw), -Math.sin(pitch), Math.cos(pitch) * Math.sin(yaw));
  const aimAt = new THREE.Vector3(rand() * 20 * S - S, rand() * 6 * S, rand() * 20 * S - S);
  const o = aimAt.clone().addScaledVector(d, -(5 + rand() * 20));
  raycaster.set(o, d);
  const want = raycastTop(raycaster, mesh);
  const got = pickTop(grid, [o.x, o.y, o.z], [d.x, d.y, d.z]);
  rays++;
  if (want.id >= 0) hitsTop++;
  // Two tops at the same height and point (a seam) are both right.
  if (got !== want.id && !(got >= 0 && want.id >= 0 && Math.abs(instanceT(got, o, d) - want.t) < 1e-4)) {
    mismatches.push(`ray ${n}: raycast ${want.id}, grid ${got}`);
  }
}
ok(`${rays} rays, ${hitsTop} on a top face: same instance`, mismatches.slice(0, 5).join('; '), '');

section('edges');
ok('straight down onto a block', pickTop(grid, [3 * S, 50, 3 * S], [0, -1, 0]) >= 0, true);
ok('a ray that misses the world', pickTop(grid, [-100, 5, -100], [-1, 0, 0]), -1);
ok('pointing up from above', pickTop(grid, [3 * S, 50, 3 * S], [0, 1, 0]), -1);
const hidden = buildPickGrid(new Float32Array(16), 1, S);
ok('a hidden instance (scale 0) is never hit', pickTop(hidden, [0, 5, 0], [0, -1, 0]), -1);
const roofOver = new THREE.Matrix4().makeTranslation(0, 3 * S, 0), floorUnder = new THREE.Matrix4();
const stack = buildPickGrid(new Float32Array([...roofOver.elements, ...floorUnder.elements]), 2, S);
ok('a roof over a floor: the roof', pickTop(stack, [0, 50, 0], [0, -1, 0]), 0);
ok('the roof lifted: through to the floor', pickTop(stack, [0, 50, 0], [0, -1, 0], i => i === 0), 1);

section('cost');
const t0 = performance.now();
for (let n = 0; n < 200; n++) {
  raycaster.set(new THREE.Vector3(7, 12, 7), new THREE.Vector3(0.3, -0.8, 0.2).normalize());
  raycastTop(raycaster, mesh);
}
const t1 = performance.now();
for (let n = 0; n < 200; n++) pickTop(grid, [7, 12, 7], [0.3 / 0.877, -0.8 / 0.877, 0.2 / 0.877]);
const t2 = performance.now();
const ray = (t1 - t0) / 200, walk = (t2 - t1) / 200;
console.log(`      ${mesh.count} instances: raycast ${ray.toFixed(3)} ms, grid walk ${walk.toFixed(4)} ms a pick`);
ok('the grid walk is faster than the raycast', walk < ray, true);
