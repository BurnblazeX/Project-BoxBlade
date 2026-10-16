import { World, getVoxelKey, pathDistance, findPath, isStandable } from '../core/world.js';
import { followPath, ENEMY_SPEED } from './movement.js';

export function createWanderAI(entity, sprite, tetherRadius) {
    const spawnPos = { ...entity.gridPos };
    let state = 'IDLE';
    let timer = 2 + Math.random() * 2; // 2 to 4 seconds
    let path = [];

    function pickTarget() {
    const candidates = [];
    for (let x = spawnPos.x - tetherRadius; x <= spawnPos.x + tetherRadius; x++) {
        for (let z = spawnPos.z - tetherRadius; z <= spawnPos.z + tetherRadius; z++) {
        const node = { x, y: spawnPos.y, z };
        if (pathDistance(spawnPos, node) <= tetherRadius) {
            const key = getVoxelKey(node.x, node.y, node.z);
            const voxel = World.get(key);
            if (voxel && isStandable(node.x, node.y, node.z) && !voxel.occupant) {
            candidates.push(node);
            }
        }
        }
    }
    if (candidates.length > 0) {
      const target = candidates[Math.floor(Math.random() * candidates.length)];
      path = findPath(entity.gridPos, target, true); // Allow diagonals for AI explore
    }
  }

  return {
    isPaused: false,
    update: function(dt, currentHeading) {
      if (this.isPaused) return;

      if (path.length > 0) {
        followPath(entity, sprite, path, ENEMY_SPEED, dt, currentHeading, true);
      } else {
        timer -= dt;
        if (timer <= 0) {
          if (state === 'IDLE') {
            state = 'WANDER';
            pickTarget();
          } else {
            state = 'IDLE';
            path = [];
          }
          timer = 2 + Math.random() * 2;
        }
      }
    }
  };
}