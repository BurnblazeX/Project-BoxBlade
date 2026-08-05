import * as THREE from 'three';
import { World } from './world.js';

// Import the texture so Vite knows to bundle and serve it
import grassTextureUrl from '../assets/textures/terrain_grass.png';

export const VOXEL_SIZE = 1.5; 

let worldInstancedMesh = null;

export function initWorldRender(scene) {
  let renderCount = 0;
  for (const voxel of World.values()) {
    if (voxel.walkable) renderCount++;
  }

  if (renderCount === 0) return;

  // 1. Load the texture
  const textureLoader = new THREE.TextureLoader();
  const grassTexture = textureLoader.load(grassTextureUrl);
  
  // CRITICAL for pixel art: prevents the texture from becoming blurry/smudged
  grassTexture.magFilter = THREE.NearestFilter;
  grassTexture.minFilter = THREE.NearestFilter;
  grassTexture.colorSpace = THREE.SRGBColorSpace;

  // 2. Setup geometry (removed the 0.95 scaling so they connect perfectly)
  const geometry = new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
  
  // 3. Apply the texture to the material
  const material = new THREE.MeshStandardMaterial({ 
    map: grassTexture,
    roughness: 0.9,  // High roughness so it doesn't look like shiny plastic
    metalness: 0.0
  });

  worldInstancedMesh = new THREE.InstancedMesh(geometry, material, renderCount);
  worldInstancedMesh.receiveShadow = true;
  worldInstancedMesh.castShadow = true;

  const matrix = new THREE.Matrix4();
  const position = new THREE.Vector3();
  let instanceIndex = 0;

  for (const [key, voxel] of World.entries()) {
    if (voxel.walkable) {
      const [gx, gy, gz] = key.split(',').map(Number);
      
      position.set(gx * VOXEL_SIZE, gy * VOXEL_SIZE, gz * VOXEL_SIZE);
      matrix.setPosition(position);
      
      worldInstancedMesh.setMatrixAt(instanceIndex, matrix);
      instanceIndex++;
    }
  }

  worldInstancedMesh.instanceMatrix.needsUpdate = true;
  
  scene.add(worldInstancedMesh);
}