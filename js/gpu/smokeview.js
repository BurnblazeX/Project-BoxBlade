// Shared rendered cubes: secondary rays use the same animated bounds and
// lighting as the mesh. The grid maps each cloud cell to its packed cube.
import * as THREE from 'three/webgpu';
import { frameValues } from './uploads.js';
import { Fn, If, Loop, Break, float, int, vec3, vec4, uniform, storage, texture3D,
         floor, min, max, abs, sign, select, mix, step } from 'three/tsl';
import { SMOKE_BOX, SMOKE_CUBE } from '../core/smoke.js';
import { MAX_SMOKE_CLOUDS, smokeShared } from './smokeshadow.js';
import { valueNoise } from './fog.js';
import { layout } from './layout.js';

export const SMOKE_CUBE_CAPACITY = 131072;
export const DEFAULT_SMOKE = {
  albedo: 1, wave: 0.22, speed: 2.2, noise: 0.35, drift: 0.12,
  ms: 0.8, sun: 1, lamps: 1, sky: 0.6
};
const size = SMOKE_BOX;
const indices = new Float32Array(size ** 3 * MAX_SMOKE_CLOUDS);
const grid = new THREE.Data3DTexture(indices, size, size, size * MAX_SMOKE_CLOUDS);
grid.format = THREE.RedFormat;
grid.type = THREE.FloatType;
grid.minFilter = grid.magFilter = THREE.NearestFilter;
grid.generateMipmaps = false;
grid.needsUpdate = true;
// Only the z layers the clouds cover go up (gpu/smoke.js write).
grid.layerUpdates = new Set();
grid.clearLayerUpdates = () => grid.layerUpdates.clear();
const attr = new THREE.StorageBufferAttribute(SMOKE_CUBE_CAPACITY * 4, 4);
export const smokeView = {
  // spans[k]: the z layers (slot k's own) the grid last held cubes in - the
  // grid outlives each cube mesh, so the next write knows what to clear.
  grid, indices, attr, spans: new Array(MAX_SMOKE_CLOUDS).fill(null),
  read: storage(attr, 'vec4', SMOKE_CUBE_CAPACITY * 4).toReadOnly().setName('bxbSmokeView'),
  uniforms: { time: frameValues.smokeViewTime, ...Object.fromEntries(Object.entries(DEFAULT_SMOKE).map(([k, v]) => [k, uniform(v)])) },
  sunDir: uniform(new THREE.Vector3(0, 1, 0))
};

export function smokeColourTSL(p, n, direct, scattered, iso, colour, u = smokeView.uniforms, sunDir = smokeView.sunDir) {
  const drift = u.drift.mul(u.time);
  const q = p.add(vec3(drift, drift.mul(0.4), drift.mul(-0.7))).div(0.125).toVar();
  const noise = valueNoise(q).mul(0.4).add(valueNoise(q.div(4)).mul(0.6));
  const thin = u.noise.mul(noise.sub(0.5)).mul(2);
  return direct.mul(float(0.45).add(max(n.dot(sunDir), 0).mul(0.55)))
    .add(scattered.mul(float(1).add(thin)))
    .add(iso.mul(float(0.8).add(n.y.mul(0.2))).mul(float(1).add(thin.mul(0.25))))
    .mul(colour).mul(u.albedo);
}

// DDA through the cell grid, testing neighbouring animated boxes too: their
// breathing can extend beyond their home cell. Stop only when the nearest
// intersection is before the next grid boundary. No surface sampling gaps.
export const smokeViewHitTSL = Fn(([origin, dir, reach]) => {
  const best = reach.toVar(), found = int(-1).toVar(), cloud = int(0).toVar();
  const normal = vec3(0, 1, 0).toVar();
  const parallel = step(abs(dir), vec3(1e-7)).toVar();
  const safe = mix(dir, vec3(1e-7), parallel);
  const inv = vec3(1).div(safe).toVar();
  const span = (lo, hi) => {
    const a = lo.sub(origin).mul(inv), b = hi.sub(origin).mul(inv);
    const near = min(a, b), far = max(a, b);
    return { near, entry: max(max(near.x, near.y), near.z), exit: min(min(far.x, far.y), far.z) };
  };
  If(smokeShared.live.greaterThan(0), () => {
    Loop({ start: int(0), end: int(MAX_SMOKE_CLOUDS), type: 'int', condition: '<', name: 'smvCloud' }, ({ smvCloud: k }) => {
      const row = smokeShared.row(k, 0);
      If(row.w.greaterThan(0.5), () => {
        const bounds = span(smokeShared.row(k, 4).xyz, smokeShared.row(k, 5).xyz);
        const t = max(bounds.entry, 0).toVar();
        const end = min(bounds.exit, best).toVar();
        If(end.greaterThan(t), () => {
          const cell = floor(origin.add(dir.mul(t.add(1e-5))).sub(row.xyz).div(SMOKE_CUBE)).toVar();
          Loop({ start: int(0), end: int(SMOKE_BOX * 3 + 6), type: 'int', condition: '<', name: 'smvStep' }, () => {
            If(t.greaterThan(end).or(t.greaterThan(best)), () => { Break(); });
            // A maximum 1.6-cell box overlaps only its immediate neighbours.
            Loop({ start: int(0), end: int(27), type: 'int', condition: '<', name: 'smvNeighbour' }, ({ smvNeighbour: i }) => {
              const offset = vec3(float(i.mod(3)).sub(1), float(i.div(3).mod(3)).sub(1), float(i.div(9)).sub(1));
              const c = cell.add(offset).toVar();
              If(c.greaterThanEqual(vec3(0)).all().and(c.lessThan(vec3(size)).all()), () => {
                const uvw = vec3(c.xy.add(0.5).div(size), c.z.add(float(k).mul(size)).add(0.5).div(size * MAX_SMOKE_CLOUDS));
                const index = int(texture3D(smokeView.grid, uvw).level(0).r).sub(1).toVar();
                If(index.greaterThanEqual(0), () => {
                  const box = smokeView.read.element(index.mul(4));
                  If(box.w.greaterThan(0), () => {
                    const hit = span(box.xyz.sub(box.w), box.xyz.add(box.w));
                    const distance = max(hit.entry, 0).toVar();
                    If(hit.exit.greaterThanEqual(distance).and(distance.lessThan(best)), () => {
                      best.assign(distance); found.assign(index); cloud.assign(k);
                      normal.assign(select(hit.near.x.greaterThanEqual(hit.near.y).and(hit.near.x.greaterThanEqual(hit.near.z)),
                        vec3(sign(dir.x).negate(), 0, 0), select(hit.near.y.greaterThanEqual(hit.near.z),
                          vec3(0, sign(dir.y).negate(), 0), vec3(0, 0, sign(dir.z).negate()))));
                    });
                  });
                });
              });
            });
            const edge = row.xyz.add(cell.add(step(vec3(0), dir)).mul(SMOKE_CUBE));
            const next = mix(edge.sub(origin).mul(inv), vec3(1e20), parallel).toVar();
            const advance = min(min(next.x, next.y), next.z).toVar();
            cell.addAssign(step(next, vec3(advance.add(1e-6))).mul(sign(dir)));
            t.assign(advance);
          });
        });
      });
    });
  });
  const result = vec4(0, 0, 0, -1).toVar();
  If(found.greaterThanEqual(0), () => {
    const base = found.mul(4);
    const colour = smokeColourTSL(origin.add(dir.mul(best)), normal,
      smokeView.read.element(base.add(1)).xyz, smokeView.read.element(base.add(2)).xyz,
      smokeView.read.element(base.add(3)).xyz, smokeShared.row(cloud, 1).xyz);
    result.assign(vec4(colour, best));
  });
  return result;
}).setLayout(layout('smokeViewHit', 'vec4', [
  { name: 'origin', type: 'vec3' }, { name: 'dir', type: 'vec3' }, { name: 'reach', type: 'float' }
]));
