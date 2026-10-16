// --- Moving characters over the world ---
//
// One implementation of what the player's path, the enemy's battle path, the
// wander AI and keyboard walking each used to carry their own copy of: a
// character walks across the ground (followGround sets its height), and the
// block it stands on - its gridPos - holds it as the occupant. A character
// only ever clears an occupant it owns, so two overlapping briefly never
// erase each other.
import * as THREE from 'three/webgpu';
import { World, getVoxelKey, stepLevel, surfaceHeight, BLOCK_METRES } from '../core/world.js';

// An entity's gridPos.y is the block it stands ON, so its sprite sits on that
// block's top face - half a block lower on a half-bottom (world.js).
export const getSpriteWorldPos = (gridPos) => new THREE.Vector3(
  gridPos.x * BLOCK_METRES,
  surfaceHeight(gridPos.x, gridPos.y, gridPos.z),
  gridPos.z * BLOCK_METRES
);

// Mirrors the sprite's art so it faces right (true) or left.
export function faceSprite(sprite, isFacingRight) {
  if (isFacingRight) {
    sprite.material.map.repeat.x = -1;
    sprite.material.map.offset.x = 1;
  } else {
    sprite.material.map.repeat.x = 1;
    sprite.material.map.offset.x = 0;
  }
}

// Off the block it stands on - only if it is the occupant there.
export function vacate(entity) {
  const p = entity.gridPos;
  const voxel = World.get(getVoxelKey(p.x, p.y, p.z));
  if (voxel && voxel.occupant === entity.id) voxel.occupant = null;
}

// Onto the block its gridPos names.
export function occupy(entity) {
  const p = entity.gridPos;
  const voxel = World.get(getVoxelKey(p.x, p.y, p.z));
  if (voxel) voxel.occupant = entity.id;
}

// The column the sprite now stands over, if it changed, taken at the level a
// step from the current one reaches (world.js stepLevel; the same level when
// none does).
function trackColumn(entity, sprite) {
  const x = Math.round(sprite.position.x / BLOCK_METRES);
  const z = Math.round(sprite.position.z / BLOCK_METRES);
  const p = entity.gridPos;
  if (x === p.x && z === p.z) return;
  vacate(entity);
  const y = stepLevel(p.x, p.y, p.z, x, z) ?? p.y;
  p.x = x;
  p.y = y;
  p.z = z;
  occupy(entity);
}

// One frame along a path: speed m/s toward path[0], turning to face the way
// it walks as seen from the camera's heading. Reaching it, the sprite snaps
// onto it, path[0] is shifted off into gridPos, and this returns true - the
// caller's cue for whatever a step costs. continuous: occupancy follows the
// column the sprite is over on the way (explore), not only each arrival.
export function followPath(entity, sprite, path, speed, dt, heading, continuous = false) {
  const node = path[0];
  // Across the ground only: followGround sets the height under the sprite.
  const target = new THREE.Vector3(node.x * BLOCK_METRES, sprite.position.y, node.z * BLOCK_METRES);
  const step = speed * dt;
  if (sprite.position.distanceTo(target) <= step) {
    sprite.position.copy(target);
    vacate(entity);
    entity.gridPos = path.shift();
    occupy(entity);
    return true;
  }
  const dir = target.clone().sub(sprite.position).normalize();
  const dot = dir.x * Math.cos(heading) - dir.z * Math.sin(heading);
  if (Math.abs(dot) > 0.05) faceSprite(sprite, dot > 0);
  sprite.position.add(dir.multiplyScalar(step));
  if (continuous) trackColumn(entity, sprite);
  return false;
}

export const WALK_SPEED = 8;    // m/s: Bob, on foot and along a clicked path
export const ENEMY_SPEED = 5;   // m/s: Evil Bob, wandering and in battle

// One frame of keyboard walking: keys { w, a, s, d } relative to the screen,
// heading the camera's settled heading. A column is entered at the same level
// or a half-block step (world.js stepLevel) - a full block's rise needs
// Jump - and not while another character occupies it. Blocked on both axes,
// it slides along whichever one is free.
export function walk(entity, sprite, keys, heading, dt) {
  let rawDx = 0, rawDz = 0;
  if (keys.w) rawDz -= 1;
  if (keys.s) rawDz += 1;
  if (keys.a) rawDx -= 1;
  if (keys.d) rawDx += 1;
  if (rawDx === 0 && rawDz === 0) return;

  if (rawDx !== 0) faceSprite(sprite, rawDx > 0);
  if (rawDx !== 0 && rawDz !== 0) {
    const invSqrt2 = 1 / Math.sqrt(2);
    rawDx *= invSqrt2;
    rawDz *= invSqrt2;
  }
  const cosH = Math.cos(heading);
  const sinH = Math.sin(heading);
  const dx = rawDx * cosH + rawDz * sinH;
  const dz = -rawDx * sinH + rawDz * cosH;

  const newX = sprite.position.x + dx * WALK_SPEED * dt;
  const newZ = sprite.position.z + dz * WALK_SPEED * dt;
  const nextGX = Math.round(newX / BLOCK_METRES);
  const nextGZ = Math.round(newZ / BLOCK_METRES);

  const p = entity.gridPos;
  const levelAt = (gx, gz) => (gx === p.x && gz === p.z)
    ? p.y
    : stepLevel(p.x, p.y, p.z, gx, gz);
  const canWalk = (gx, gz) => {
    const y = levelAt(gx, gz);
    if (y === null) return false;
    const v = World.get(getVoxelKey(gx, y, gz));
    return !v.occupant || v.occupant === entity.id;
  };

  if (canWalk(nextGX, nextGZ)) {
    sprite.position.x = newX;
    sprite.position.z = newZ;
  } else {
    if (canWalk(nextGX, Math.round(sprite.position.z / BLOCK_METRES))) sprite.position.x = newX;
    if (canWalk(Math.round(sprite.position.x / BLOCK_METRES), nextGZ)) sprite.position.z = newZ;
  }

  trackColumn(entity, sprite);
}
