import * as THREE from 'three/webgpu';
import { World, getVoxelKey, pathDistance, findPath, isStandable } from './world.js';
import { VOXEL_SIZE } from './render.js';

function updateSpriteFacing(sprite, isFacingRight) {
    if (isFacingRight) {
    sprite.material.map.repeat.x = -1;  
    sprite.material.map.offset.x = 1; 
    } else {
    sprite.material.map.repeat.x = 1;   
    sprite.material.map.offset.x = 0; 
    }
}

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
        const targetNode = path[0];
        const targetWorldPos = new THREE.Vector3(
          targetNode.x * VOXEL_SIZE,
          VOXEL_SIZE / 2,
          targetNode.z * VOXEL_SIZE
        );
        const step = 5 * dt; // Enemy speed
        
        if (sprite.position.distanceTo(targetWorldPos) <= step) {
          sprite.position.copy(targetWorldPos);
          
          const oldVoxel = World.get(getVoxelKey(entity.gridPos.x, entity.gridPos.y, entity.gridPos.z));
          // FIX: Only clear if the enemy actually owns it!
          if (oldVoxel && oldVoxel.occupant === entity.id) oldVoxel.occupant = null;
          
          entity.gridPos = path.shift();
          
          const newVoxel = World.get(getVoxelKey(entity.gridPos.x, entity.gridPos.y, entity.gridPos.z));
          if (newVoxel) newVoxel.occupant = entity.id;
        } else {
          const dir = targetWorldPos.clone().sub(sprite.position).normalize();
          const dot = dir.x * Math.cos(currentHeading) - dir.z * Math.sin(currentHeading);
          if (Math.abs(dot) > 0.05) updateSpriteFacing(sprite, dot > 0);
          
          sprite.position.add(dir.multiplyScalar(step));
          
          // Continuous grid occupancy alignment
          const newGridX = Math.round(sprite.position.x / VOXEL_SIZE);
          const newGridZ = Math.round(sprite.position.z / VOXEL_SIZE);
          if (newGridX !== entity.gridPos.x || newGridZ !== entity.gridPos.z) {
            const oldVoxel = World.get(getVoxelKey(entity.gridPos.x, entity.gridPos.y, entity.gridPos.z));
            // FIX: Strict check
            if (oldVoxel && oldVoxel.occupant === entity.id) oldVoxel.occupant = null;
            
            const newVoxel = World.get(getVoxelKey(newGridX, entity.gridPos.y, newGridZ));
            if (newVoxel) newVoxel.occupant = entity.id;
            entity.gridPos.x = newGridX;
            entity.gridPos.z = newGridZ;
          }
        }
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