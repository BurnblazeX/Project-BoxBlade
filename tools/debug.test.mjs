import { file, section, ok, truthy, falsy, inRange, note } from './lib/harness.mjs';
import { createTestArea, World, getVoxelKey } from '../js/world.js';
import { GRID_DIM, TEXELS_PER_BLOCK } from '../js/boxgrid.js';
import { toggleBoxGridDebug, refreshBoxGridDebug, isBoxGridDebugVisible, getBoxGridDebugOrigin } from '../js/debug.js';

file('debug.test.mjs - Alt+X boxGrid overlay');

// debug.js imports Three.js but never constructs a renderer, so the real build
// path runs headlessly against a stub scene. Anything needing a GPU device
// belongs in the in-browser checklist instead.
const solid = () => ({ solid: true, walkable: true, materialId: null, occupant: null, triggerId: null });
createTestArea(36, 36);
for (let y = 1; y <= 2; y++) World.set(getVoxelKey(6, y, 6), solid());

const added = [];
const removed = [];
const scene = { add: o => added.push(o), remove: o => removed.push(o) };
const terrain = { visible: true };
const playerPos = { x: 8, y: 0, z: 8 };

section('starts hidden');
falsy('not visible before first toggle', isBoxGridDebugVisible());
ok('nothing added to the scene yet', added.length, 0);

section('enabling');
const on = toggleBoxGridDebug(scene, playerPos, terrain);
truthy('toggle reports enabled', on);
truthy('state says visible', isBoxGridDebugVisible());
ok('one mesh added', added.length, 1);
falsy('terrain hidden so it cannot z-fight with coplanar voxel faces', terrain.visible);

const mesh = added[0];
truthy('mesh is instanced', mesh.isInstancedMesh);
truthy('has per-instance colour for the height ramp', !!mesh.instanceColor);
truthy('mesh is preallocated above the live count', mesh.instanceMatrix.count >= mesh.count);
falsy('no wireframe - 45k line-geometry instances tanked the framerate', mesh.material.wireframe);
ok('explore centres the footprint on the player', `${getBoxGridDebugOrigin().x},${getBoxGridDebugOrigin().z}`, '2,2');

section('only surface voxels are drawn');
// Ground slab is 144x144 across and 12 voxels deep; almost all of it is
// interior and must be culled. Exact count depends on edge/corner sharing, so
// this brackets it rather than pinning it.
inRange('downward-only faces culled, so well under the full shell', mesh.count, 15000, 45000);
truthy('far below total occupied voxels', mesh.count < 127872 * 0.5);
truthy('far below total grid voxels', mesh.count < GRID_DIM ** 3 * 0.02);
note(`${mesh.count.toLocaleString()} instances drawn for a ${GRID_DIM}^3 grid`);

section('disabling');
const off = toggleBoxGridDebug(scene, playerPos, terrain);
falsy('toggle reports disabled', off);
ok('mesh removed from the scene', removed.length, 1);
truthy('terrain restored', terrain.visible);
falsy('state says hidden', isBoxGridDebugVisible());

section('explore mode - footprint follows the player');
toggleBoxGridDebug(scene, { x: 8, y: 0, z: 8 }, terrain, 'explore');
const addedBefore = added.length;
const explore0 = added[added.length - 1].count;
ok('origin is the player minus half a footprint', `${getBoxGridDebugOrigin().x},${getBoxGridDebugOrigin().z}`, '2,2');

falsy('standing still does not rebuild', refreshBoxGridDebug(scene, { x: 8, y: 0, z: 8 }, 'explore'));

// A re-origin is split over two frames: occupancy on one, mesh refill on the
// next, so neither frame pays for both.
truthy('stepping one block starts a rebuild',
       refreshBoxGridDebug(scene, { x: 9, y: 0, z: 8 }, 'explore'));
ok('origin is claimed immediately', `${getBoxGridDebugOrigin().x},${getBoxGridDebugOrigin().z}`, '3,2');
truthy('the next frame finishes it', refreshBoxGridDebug(scene, { x: 9, y: 0, z: 8 }, 'explore'));
falsy('and then it settles', refreshBoxGridDebug(scene, { x: 9, y: 0, z: 8 }, 'explore'));

// Moving again mid-rebuild must not start a second one on top of the first.
truthy('moving again starts another', refreshBoxGridDebug(scene, { x: 10, y: 0, z: 8 }, 'explore'));
truthy('a further move finishes the one in flight first',
       refreshBoxGridDebug(scene, { x: 14, y: 0, z: 8 }, 'explore'));
ok('the in-flight origin completed', `${getBoxGridDebugOrigin().x},${getBoxGridDebugOrigin().z}`, '4,2');
truthy('then the newer position is picked up',
       refreshBoxGridDebug(scene, { x: 14, y: 0, z: 8 }, 'explore'));
ok('catching up to it', `${getBoxGridDebugOrigin().x},${getBoxGridDebugOrigin().z}`, '8,2');
refreshBoxGridDebug(scene, { x: 14, y: 0, z: 8 }, 'explore'); // finish
ok('re-origins refill the same mesh, never a new one', added.length, addedBefore);

section('battle mode - chunk aligned and static');
truthy('switching to battle snaps the origin to the chunk',
       refreshBoxGridDebug(scene, { x: 9, y: 0, z: 8 }, 'battle'));
refreshBoxGridDebug(scene, { x: 9, y: 0, z: 8 }, 'battle'); // finish the split
ok('origin is chunk aligned', `${getBoxGridDebugOrigin().x},${getBoxGridDebugOrigin().z}`, '0,0');
falsy('moving within the arena does not move the grid',
      refreshBoxGridDebug(scene, { x: 4, y: 0, z: 11 }, 'battle'));
ok('still chunk aligned', `${getBoxGridDebugOrigin().x},${getBoxGridDebugOrigin().z}`, '0,0');
truthy('entering the next chunk does move it',
       refreshBoxGridDebug(scene, { x: 20, y: 0, z: 10 }, 'battle'));
refreshBoxGridDebug(scene, { x: 20, y: 0, z: 10 }, 'battle'); // finish the split
ok('now the next chunk', `${getBoxGridDebugOrigin().x},${getBoxGridDebugOrigin().z}`, '12,0');
const battleWall = added[added.length - 1].count;
truthy('the chunk holding the wall draws more surface', battleWall > explore0);
note(`explore at (8,8): ${explore0.toLocaleString()} | battle chunk with the wall: ${battleWall.toLocaleString()}`);

toggleBoxGridDebug(scene, { x: 20, y: 0, z: 10 }, terrain, 'battle');
falsy('left disabled', isBoxGridDebugVisible());
falsy('refresh is a no-op while hidden', refreshBoxGridDebug(scene, { x: 8, y: 0, z: 8 }, 'explore'));
