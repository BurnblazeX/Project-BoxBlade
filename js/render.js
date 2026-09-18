import * as THREE from 'three/webgpu';
import { World } from './world.js';

// Import the texture so Vite knows to bundle and serve it
import grassTextureUrl from '../assets/textures/terrain_grass.png';

export const VOXEL_SIZE = 1.5; 

export let worldInstancedMesh = null;
export const voxelIndexMap = new Map(); // Tracks which instance ID belongs to which grid key

export function initWorldRender(scene) {
  let renderCount = 0;
  for (const voxel of World.values()) {
    if (voxel.walkable) renderCount++;
  }
  if (renderCount === 0) return;

  const textureLoader = new THREE.TextureLoader();
  const grassTexture = textureLoader.load(grassTextureUrl);
  grassTexture.magFilter = THREE.NearestFilter;
  grassTexture.minFilter = THREE.NearestFilter;
  grassTexture.colorSpace = THREE.SRGBColorSpace;

  const geometry = new THREE.BoxGeometry(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
  const material = new THREE.MeshStandardMaterial({ 
    map: grassTexture, roughness: 0.9, metalness: 0.0 
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
      
      // Record instance ID and set default base color
      voxelIndexMap.set(key, instanceIndex);
      worldInstancedMesh.setColorAt(instanceIndex, new THREE.Color(0xffffff));
      instanceIndex++;
    }
  }
  worldInstancedMesh.instanceMatrix.needsUpdate = true;
  if (worldInstancedMesh.instanceColor) worldInstancedMesh.instanceColor.needsUpdate = true;
  scene.add(worldInstancedMesh);
}

const _matrix = new THREE.Matrix4();
const _position = new THREE.Vector3();
const _scaleHidden = new THREE.Vector3(0, 0, 0);
const _scaleVisible = new THREE.Vector3(1, 1, 1);
const _rotation = new THREE.Quaternion();

export function updateVoxelVisibility(arenaMap, isBattle) {
  if (!worldInstancedMesh) return;
  
  const _matrix = new THREE.Matrix4();
  const _position = new THREE.Vector3();
  const _scale = new THREE.Vector3();
  const _quaternion = new THREE.Quaternion(); // Identity (0 rotation)

  for (const [key, idx] of voxelIndexMap.entries()) {
    // Calculate exact position from scratch based on grid coordinates
    const [gx, gy, gz] = key.split(',').map(Number);
    _position.set(gx * VOXEL_SIZE, gy * VOXEL_SIZE, gz * VOXEL_SIZE);
    
    if (isBattle && (!arenaMap || !arenaMap.has(key))) {
      _scale.set(0, 0, 0); // Hide
    } else {
      _scale.set(1, 1, 1); // Show
    }
    
    _matrix.compose(_position, _quaternion, _scale);
    worldInstancedMesh.setMatrixAt(idx, _matrix);
  }
  worldInstancedMesh.instanceMatrix.needsUpdate = true;
}

export function updateVoxelTints(arenaMap, reachableMap, isBattle) {
  if (!worldInstancedMesh) return;
  const colorWhite = new THREE.Color(0xffffff);
  const colorGrey = new THREE.Color(0x333333); 
  
  for (const [key, idx] of voxelIndexMap.entries()) {
    if (!isBattle) {
      worldInstancedMesh.setColorAt(idx, colorWhite);
    } else if (arenaMap.has(key)) {
      if (reachableMap && reachableMap.has(key)) {
        worldInstancedMesh.setColorAt(idx, colorWhite);
      } else {
        worldInstancedMesh.setColorAt(idx, colorGrey);
      }
    }
  }
  worldInstancedMesh.instanceColor.needsUpdate = true;
}