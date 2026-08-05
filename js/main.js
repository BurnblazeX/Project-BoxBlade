import * as THREE from 'three';
import { World, getVoxelKey, createTestArea, createEntity, pathDistance, findPath, enterBattle, exitBattle, getReachableVoxels } from './world.js';
import { initWorldRender, VOXEL_SIZE, updateVoxelTints, updateVoxelVisibility } from './render.js';
import bobTextureUrl from '../assets/sprites/character_Bob.png'
const inputRules = {
  explore: { click: true, keyboard: true },
  battle:  { click: true, keyboard: false }
};

// --- SCENE SETUP ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x222233);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('app').appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
dirLight.position.set(10, 20, 10);
scene.add(dirLight);

// --- WORLD GENERATION ---
createTestArea(48, 48);
initWorldRender(scene);

// --- PLAYER ENTITY & SPRITE SETUP ---
const player = createEntity({
  id: "player_1",
  name: "Bob",
  gridPos: { x: 8, y: 0, z: 8 }, // Placed cleanly inside Chunk (0,0)
  speed: 6,
  mode: "explore"
});
World.get(getVoxelKey(8, 0, 8)).occupant = player.id;

const texLoader = new THREE.TextureLoader();
const bobTexture = texLoader.load(bobTextureUrl);
bobTexture.magFilter = THREE.NearestFilter; 
bobTexture.minFilter = THREE.NearestFilter;

const bobMaterial = new THREE.SpriteMaterial({ map: bobTexture, transparent: true });
const playerSprite = new THREE.Sprite(bobMaterial);

playerSprite.center.set(0.5, 0); 
playerSprite.scale.set(VOXEL_SIZE, VOXEL_SIZE, 1);

const getSpriteWorldPos = (gridPos) => new THREE.Vector3(
  gridPos.x * VOXEL_SIZE,
  VOXEL_SIZE / 2, 
  gridPos.z * VOXEL_SIZE
);
playerSprite.position.copy(getSpriteWorldPos(player.gridPos));
scene.add(playerSprite);

// --- VISUAL AIDS (HIGHLIGHT & PATH DOTS) ---
// 1. Blue Hover Highlight (Edge Outline Only)
const shapeGeo = new THREE.PlaneGeometry(VOXEL_SIZE * 0.95, VOXEL_SIZE * 0.95);
shapeGeo.rotateX(-Math.PI / 2); // Lay flat on the ground
const highlightGeo = new THREE.EdgesGeometry(shapeGeo);
const highlightMat = new THREE.LineBasicMaterial({ 
  color: 0x0088ff, 
  transparent: true, 
  opacity: 0.8
});
const highlightMesh = new THREE.LineSegments(highlightGeo, highlightMat);
highlightMesh.visible = false;
scene.add(highlightMesh);

// 2. Path Dots
const pathGroup = new THREE.Group();
scene.add(pathGroup);
const dotGeo = new THREE.SphereGeometry(0.12, 8, 8);
const dotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

function updatePathDots(path) {
  pathGroup.clear();
  for (const node of path) {
    const dot = new THREE.Mesh(dotGeo, dotMat);
    // Position slightly above the surface
    dot.position.set(node.x * VOXEL_SIZE, (VOXEL_SIZE / 2) + 0.1, node.z * VOXEL_SIZE);
    pathGroup.add(dot);
  }
}


// --- CAMERA RIG SYSTEM ---
let aspect = window.innerWidth / window.innerHeight;

// We use one PerspectiveCamera to allow buttery smooth lerping (no janky swaps).
// We mimic Orthographic for battle mode by lowering the FOV and pulling back distance.
const camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
const pivot = new THREE.Object3D();
let arenaCenter = new THREE.Vector3(); 
scene.add(pivot);

const cameraConfigs = {
  explore: { 
    fov: 45,        
    distance: 22,   
    pitch: Math.PI / 3, 
    headingOffset: 0
  },
  battle: { 
    fov: 8,         
    distance: 125,  
    pitch: Math.atan(1 / Math.sqrt(2)), 
    headingOffset: Math.PI / 4
  }
};

let currentMode = 'explore';
let rotationStep = 0; 
let currentFov = cameraConfigs.explore.fov;
let currentPitch = cameraConfigs.explore.pitch;
let currentHeading = cameraConfigs.explore.headingOffset;
let currentDistance = cameraConfigs.explore.distance;
const CAMERA_LERP_SPEED = 0.1;
const CAMERA_DISTANCE = 50; 

function updateCameraTargets() {
  const config = cameraConfigs[currentMode];
  return {
    targetViewSize: config.viewSize,
    targetPitch: config.pitch,
    targetHeading: (rotationStep * Math.PI / 2) + config.headingOffset
  };
}

// --- MOVEMENT STATE ---
let currentPath = [];
const clock = new THREE.Clock();
const keyState = { w: false, a: false, s: false, d: false };

// --- INPUT HANDLING ---
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const clickPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -(VOXEL_SIZE / 2));

// Helper: Convert screen coords to grid coords
function getGridIntersection(clientX, clientY) {
  mouse.x = (clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(clientY / window.innerHeight) * 2 + 1;
  
  raycaster.setFromCamera(mouse, camera); // CHANGED: Back to standard camera
  const intersection = new THREE.Vector3();
  
  if (raycaster.ray.intersectPlane(clickPlane, intersection)) {
    return {
      gx: Math.round(intersection.x / VOXEL_SIZE),
      gz: Math.round(intersection.z / VOXEL_SIZE)
    };
  }
  return null;
}

let lastHoveredKey = null;
let clickPulseTime = 0; 
let currentReachable = null; 
let currentArenaMap = null;

function refreshReachableTiles() {
  if (currentMode === 'battle') {
    currentReachable = getReachableVoxels(player.gridPos, player.speed);
    updateVoxelTints(currentArenaMap, currentReachable, true);
  } else {
    updateVoxelTints(null, null, false);
  }
}

window.addEventListener('pointermove', (e) => {
  const isWalkingManual = (keyState.w || keyState.a || keyState.s || keyState.d);
  
  if (!inputRules[currentMode].click || currentPath.length > 0 || isWalkingManual) {
    highlightMesh.visible = false;
    pathGroup.clear();
    lastHoveredKey = null;
    return;
  }
  
  if (clickPulseTime > 0) return;

  const intersect = getGridIntersection(e.clientX, e.clientY);
  if (intersect) {
    const { gx, gz } = intersect;
    const targetKey = getVoxelKey(gx, 0, gz);

    if (targetKey !== lastHoveredKey) {
      lastHoveredKey = targetKey;
      const voxel = World.get(targetKey);

      // Only allow hover if in explore mode OR if tile is in battle reachable range
      if (voxel && voxel.walkable && (currentMode === 'explore' || (currentReachable && currentReachable.has(targetKey)))) {
        highlightMesh.position.set(gx * VOXEL_SIZE, (VOXEL_SIZE / 2) + 0.02, gz * VOXEL_SIZE);
        highlightMesh.visible = true;

        if (currentMode === 'battle') {
          const path = findPath(player.gridPos, { x: gx, y: 0, z: gz });
          updatePathDots(path);
        } else {
          pathGroup.clear();
        }
      } else {
        highlightMesh.visible = false;
        pathGroup.clear();
      }
    }
  } else {
    highlightMesh.visible = false;
    pathGroup.clear();
    lastHoveredKey = null;
  }
});

window.addEventListener('pointerup', (e) => {
  if (!inputRules[currentMode].click) return;
  if (currentPath.length > 0) return; // Prevent clicking while currently walking

  const intersect = getGridIntersection(e.clientX, e.clientY);
  if (!intersect) return; // Abort if clicked into the void

  const { gx, gz } = intersect;
  const targetKey = getVoxelKey(gx, 0, gz);

  // STRICT BATTLE REJECTION:
  if (currentMode === 'battle') {
    // If currentReachable doesn't have the tile, it is instantly rejected.
    if (!currentReachable || !currentReachable.has(targetKey)) {
      console.warn(`Rejected: Tile ${targetKey} is outside speed range or arena bounds.`);
      return; 
    }
  }

  if (World.has(targetKey)) {
    const path = findPath(player.gridPos, { x: gx, y: 0, z: gz });
    
    // Double fail-safe: Check actual calculated path length against speed
    if (currentMode === 'battle' && path.length > player.speed) {
       console.warn(`Rejected: Path length (${path.length}) exceeds speed stat (${player.speed}).`);
       return;
    }

    if (path.length > 0) {
      currentPath = path;
      
      clickPulseTime = 1.0; 
      highlightMesh.position.set(gx * VOXEL_SIZE, (VOXEL_SIZE / 2) + 0.02, gz * VOXEL_SIZE);
      highlightMesh.visible = true;
      highlightMesh.material.color.setHex(0xffff00); 
      
      pathGroup.clear(); 
    }
  }
});

window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  
  if (keyState.hasOwnProperty(key)) {
    keyState[key] = true;
  }
  
  if (key === 'b') {
    currentMode = currentMode === 'explore' ? 'battle' : 'explore';
    document.getElementById('mode-text').innerText = currentMode.charAt(0).toUpperCase() + currentMode.slice(1);
    
    currentPath = []; 
    
    highlightMesh.visible = false;
    pathGroup.clear();
    lastHoveredKey = null;
    clickPulseTime = 0;
    highlightMesh.material.color.setHex(0x0088ff);
    
    if (currentMode === 'battle') {
      player.gridPos.x = Math.round(playerSprite.position.x / VOXEL_SIZE);
      player.gridPos.z = Math.round(playerSprite.position.z / VOXEL_SIZE);
      playerSprite.position.copy(getSpriteWorldPos(player.gridPos));
      
      const battleData = enterBattle(player.gridPos);
      currentArenaMap = battleData.arena;
      
      // Calculate exact center of the 16x16 Chunk for the camera
      const centerX = (battleData.bounds.minX + battleData.bounds.maxX) / 2;
      const centerZ = (battleData.bounds.minZ + battleData.bounds.maxZ) / 2;
      arenaCenter.set(centerX * VOXEL_SIZE, 0, centerZ * VOXEL_SIZE);
      
      updateVoxelVisibility(currentArenaMap, true);
      refreshReachableTiles();
    } else {
      exitBattle();
      currentArenaMap = null;
      updateVoxelVisibility(null, false);
      refreshReachableTiles();
    }
  }
  
  if (key === 'q') rotationStep += 1;
  if (key === 'e') rotationStep -= 1;
});

window.addEventListener('keyup', (e) => {
  const key = e.key.toLowerCase();
  if (keyState.hasOwnProperty(key)) {
    keyState[key] = false;
  }
});

window.addEventListener('resize', () => {
  aspect = window.innerWidth / window.innerHeight;
  camera.aspect = aspect;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- RENDER & GAME LOOP ---
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();

  // Handle Highlight Pulse Animation
  if (clickPulseTime > 0) {
    clickPulseTime -= dt * 4; // Roughly 0.25 sec total duration
    const scale = 1 + Math.sin(clickPulseTime * Math.PI) * 0.15; // Pop scale effect
    highlightMesh.scale.set(scale, scale, 1);
    
    if (clickPulseTime <= 0) {
      // Reset after pulse completes
      highlightMesh.material.color.setHex(0x0088ff);
      highlightMesh.material.opacity = 0.5;
      highlightMesh.scale.set(1, 1, 1);
      lastHoveredKey = null; // Force an update check next frame
    }
  }

  // 1. Process Movement Logic
  if (currentPath.length > 0) {
    const targetNode = currentPath[0];
    const targetWorldPos = getSpriteWorldPos(targetNode);
    const step = 8 * dt;
    
    // Inside animate() -> if (currentPath.length > 0) ...
    if (playerSprite.position.distanceTo(targetWorldPos) <= step) {
      playerSprite.position.copy(targetWorldPos);
      
      const oldVoxel = World.get(getVoxelKey(player.gridPos.x, 0, player.gridPos.z));
      if (oldVoxel) oldVoxel.occupant = null;
      
      player.gridPos = currentPath.shift();
      
      const newVoxel = World.get(getVoxelKey(player.gridPos.x, 0, player.gridPos.z));
      if (newVoxel) newVoxel.occupant = player.id;
      
      // Recalculate reachable tiles after completing movement
      if (currentPath.length === 0 && currentMode === 'battle') {
        refreshReachableTiles();
      }
      
    } else {
      const dir = targetWorldPos.clone().sub(playerSprite.position).normalize();
      playerSprite.position.add(dir.multiplyScalar(step));
    }
  } 
  else if (currentMode === 'explore' && inputRules.explore.keyboard) {
    // 1. Raw Input (Opposing keys now cancel each other out)
    let rawDx = 0, rawDz = 0;
    if (keyState.w) rawDz -= 1; // Up
    if (keyState.s) rawDz += 1; // Down
    if (keyState.a) rawDx -= 1; // Left
    if (keyState.d) rawDx += 1; // Right

    if (rawDx !== 0 || rawDz !== 0) {
      // Normalize raw diagonals so you don't run faster diagonally
      if (rawDx !== 0 && rawDz !== 0) {
        const invSqrt2 = 1 / Math.sqrt(2);
        rawDx *= invSqrt2;
        rawDz *= invSqrt2;
      }

      // 2. Rotate input vector based on camera perspective
      const targetHeading = (rotationStep * Math.PI / 2) + cameraConfigs.explore.headingOffset;
      const cosH = Math.cos(targetHeading);
      const sinH = Math.sin(targetHeading);

      // 2D Rotation Matrix applied to X/Z axes
      const dx = rawDx * cosH + rawDz * sinH;
      const dz = -rawDx * sinH + rawDz * cosH;

      const moveSpeed = 8;
      const stepX = dx * moveSpeed * dt;
      const stepZ = dz * moveSpeed * dt;

      const newX = playerSprite.position.x + stepX;
      const newZ = playerSprite.position.z + stepZ;

      const nextGX = Math.round(newX / VOXEL_SIZE);
      const nextGZ = Math.round(newZ / VOXEL_SIZE);
      const voxel = World.get(getVoxelKey(nextGX, 0, nextGZ));

      if (voxel && voxel.walkable) {
        playerSprite.position.x = newX;
        playerSprite.position.z = newZ;
      } else {
        // Wall sliding
        const voxelX = World.get(getVoxelKey(nextGX, 0, Math.round(playerSprite.position.z / VOXEL_SIZE)));
        if (voxelX && voxelX.walkable) playerSprite.position.x = newX;

        const voxelZ = World.get(getVoxelKey(Math.round(playerSprite.position.x / VOXEL_SIZE), 0, nextGZ));
        if (voxelZ && voxelZ.walkable) playerSprite.position.z = newZ;
      }

      // Update gridPos and manage occupant swapping during free-roam
      const newGridX = Math.round(playerSprite.position.x / VOXEL_SIZE);
      const newGridZ = Math.round(playerSprite.position.z / VOXEL_SIZE);
      
      if (newGridX !== player.gridPos.x || newGridZ !== player.gridPos.z) {
        const oldVoxel = World.get(getVoxelKey(player.gridPos.x, 0, player.gridPos.z));
        if (oldVoxel) oldVoxel.occupant = null;
        
        const newVoxel = World.get(getVoxelKey(newGridX, 0, newGridZ));
        if (newVoxel) newVoxel.occupant = player.id;
        
        player.gridPos.x = newGridX;
        player.gridPos.z = newGridZ;
      }
    }
  }

  // 2. Camera Lerping & Pivot Tracking
  if (currentMode === 'explore') {
    pivot.position.lerp(playerSprite.position, 0.1); 
  } else {
    pivot.position.lerp(arenaCenter, 0.1); 
  }

  const config = cameraConfigs[currentMode];
  const targetHeading = (rotationStep * Math.PI / 2) + config.headingOffset;
  
  // Lerp all camera properties smoothly
  currentFov += (config.fov - currentFov) * CAMERA_LERP_SPEED;
  currentPitch += (config.pitch - currentPitch) * CAMERA_LERP_SPEED;
  currentHeading += (targetHeading - currentHeading) * CAMERA_LERP_SPEED;
  currentDistance += (config.distance - currentDistance) * CAMERA_LERP_SPEED;

  camera.fov = currentFov;
  camera.updateProjectionMatrix();

  const xzLen = currentDistance * Math.cos(currentPitch);
  camera.position.x = pivot.position.x + xzLen * Math.sin(currentHeading);
  camera.position.y = pivot.position.y + currentDistance * Math.sin(currentPitch);
  camera.position.z = pivot.position.z + xzLen * Math.cos(currentHeading);
  
  camera.lookAt(pivot.position);

  renderer.render(scene, camera);
}

animate();