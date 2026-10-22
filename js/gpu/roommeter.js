import * as THREE from 'three/webgpu';
import { Fn, If, float, vec2, vec3, vec4, int, uint, uniform, storage, min, mix, select, round,
         atomicAdd, atomicLoad, atomicStore } from 'three/tsl';
import { VOXEL_METRES } from '../core/field.js';
import { BLOCK_METRES } from '../core/world.js';
import { insideAtTSL } from './occlusion.js';
import { blockVolumeTSL } from './volume.js';

// --- The room meter: the light the visible inside of a building gets ---
//
// A revealed face - a wall's or a pillar's top under a lifted roof - is buried
// in what the lighting sees, so it has no light of its own to show. It takes
// the room's instead (gpu/ambient.js createRevealedLightTSL): the mean diffuse
// light on the inside's surfaces on screen, one value for every revealed face,
// so a cut top reads at the room's brightness - no stripes from the LPV's
// cells, no flat sky under texel GI.
//
// Split shading's kernels feed it (createSplitShading creates it, gpu/shade.js
// taps it): every ROOM_METER_STRIDE-th sample that lies inside a building (the
// interiors' mask) on a face that is not itself buried adds the diffuse light
// each kernel gives it - the base kernel the ambient, bounce and sun (and the
// count), the lamps and mirrors theirs - into four atomic words in fixed
// point. Then ease (one thread, after the kernels) eases the mean toward this
// frame's and clears the sums. Nothing comes back to the CPU.

// An odd prime: no lock-step with the rows of the low-res grid.
export const ROOM_METER_STRIDE = 61;

const SCALE = 256;      // fixed point a unit of light
const CAP = 64;         // one sample's light, at most

export function createRoomMeter() {
  const accAttr = new THREE.StorageBufferAttribute(new Uint32Array(4), 1);
  const acc = storage(accAttr, 'uint', 4).toAtomic().setName('bxbRoomMeter');
  // The eased mean: rgb, and w 1 once there has been a frame to mean.
  const meanAttr = new THREE.StorageBufferAttribute(new Float32Array(4), 4);
  const mean = storage(meanAttr, 'vec4', 1).setName('bxbRoomLight');
  const meanRead = storage(meanAttr, 'vec4', 1).toReadOnly().setName('bxbRoomLightRead');
  // How far the mean moves toward this frame's: 1 - exp(-dt / response), set
  // from the frame's own time (render/features/texelcache.js).
  const blend = uniform(float(1));
  // A/B: 1 revealed faces take the room's light; 0 their own estimate, as before.
  const on = uniform(int(1));

  const ease = Fn(() => {
    const n = atomicLoad(acc.element(int(3))).toVar();
    If(n.greaterThan(uint(0)), () => {
      const sum = vec3(float(atomicLoad(acc.element(int(0)))), float(atomicLoad(acc.element(int(1)))),
                       float(atomicLoad(acc.element(int(2)))));
      const now = sum.div(float(n).mul(float(SCALE)));
      const was = mean.element(int(0)).toVar();
      const a = select(was.w.lessThan(float(0.5)), float(1), blend);
      mean.element(int(0)).assign(vec4(mix(was.xyz, now, a), 1));
    });
    for (let k = 0; k < 4; k++) atomicStore(acc.element(int(k)), uint(0));
  })().compute(1).setName('roomMeterEase');

  return { accAttr, meanAttr, meanRead, blend, on, ease,
    // A split kernel's tap: (idx, p, n, light, counts) - p on a box face, n
    // its axis normal, light the diffuse light this kernel gives the sample;
    // counts in the one kernel that runs for every sample it takes.
    tap: terrain => (idx, p, n, light, counts = false) => {
      const buried = blockVolumeTSL(terrain, round(p.add(n.mul(float(VOXEL_METRES * 0.5))).div(float(BLOCK_METRES))))
        .x.greaterThan(float(0.5));
      If(idx.mod(int(ROOM_METER_STRIDE)).equal(int(0))
        .and(insideAtTSL(vec2(p.x, p.z)).greaterThan(float(0.5))).and(buried.not()), () => {
        const q = min(light, vec3(CAP, CAP, CAP)).mul(float(SCALE));
        atomicAdd(acc.element(int(0)), uint(q.x));
        atomicAdd(acc.element(int(1)), uint(q.y));
        atomicAdd(acc.element(int(2)), uint(q.z));
        if (counts) atomicAdd(acc.element(int(3)), uint(1));
      });
    } };
}
