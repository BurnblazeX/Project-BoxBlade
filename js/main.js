import * as THREE from 'three';
import { World, getVoxelKey, createTestArea, createEntity, pathDistance, findPath, enterBattle, exitBattle, getReachableVoxels } from './world.js';
import { initWorldRender, VOXEL_SIZE, updateVoxelTints, updateVoxelVisibility } from './render.js';
import { createWanderAI } from './ai.js';
import { rollInitiativeForParticipants, resetBattleState, turnOrder, currentTurnIndex, getCurrentEntity, nextTurn, addParticipant } from './battle.js';
import * as UI from './ui.js';
import { performAttack, isDefeated, takeEnemyTurn, isInMeleeRange } from './combat.js';

import bobTextureUrl from '../assets/sprites/character_Bob.png'
import evilBobTextureUrl from '../assets/sprites/character_EvilBob.png'

UI.initUI(
  // Attack Button Callback
  () => {
    if (currentMode !== 'battle') return;
    const current = getCurrentEntity(battleParticipants);
    if (current.id !== player.id) return; 

    // 1. Check Range 
    if (!isInMeleeRange(player, enemy)) {
       UI.logDiceRoll("<i>Target is out of range!</i>", "system");
       return; 
    }

    // 2. Roll Attack
    const res = performAttack(player, enemy);
    if (!res.success) {
      UI.logDiceRoll("<i>You have no action left!</i>", "system");
      return;
    }

    // 3. Provoke Mechanic: If enemy wasn't in combat, add them now!
    if (!battleParticipants.some(p => p.id === enemy.id)) {
       battleParticipants.push(enemy);
       addParticipant(enemy);
       UI.updateActionOrder(turnOrder, currentTurnIndex, battleParticipants);
       UI.logDiceRoll(`<i>${enemy.name} was provoked and joined the battle!</i>`, "system");
    }

    // 4. Resolve Hit/Damage
    if (res.hit) {
      UI.logDiceRoll(`<b>${player.name}</b> hits ${enemy.name}! <br>[Roll: ${res.attackTotal} vs AC ${enemy.ac}] <br>Deals <b>${res.damage} damage!</b>`, "player-turn");
      if (isDefeated(enemy)) {
        enemySprite.visible = false;
        const v = World.get(getVoxelKey(enemy.gridPos.x, 0, enemy.gridPos.z));
        if (v && v.occupant === enemy.id) v.occupant = null;
        endBattleSequence("VICTORY! You slew Evil Bob!");
      }
    } else {
      UI.logDiceRoll(`<b>${player.name}</b> misses! <br>[Roll: ${res.attackTotal} vs AC ${enemy.ac}]`, "player-turn");
    }
    
    UI.updateHUD(player); 
    UI.updateActionResources(player.turnResources);
  },
  // End Turn Button Callback
  () => {
    if (currentMode !== 'battle') return;
    if (getCurrentEntity(battleParticipants)?.id !== player.id) return;
    
    UI.logDiceRoll(`<i>${player.name} ends their turn.</i>`, "system");
    nextTurn(battleParticipants, onTurnStart);
  }
);

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
createTestArea(36, 36);
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

// Clone the texture so this specific sprite can flip independently
const playerTex = bobTexture.clone();
const bobMaterial = new THREE.SpriteMaterial({ map: playerTex, transparent: true });
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

UI.updatePartyView([player], player.id);

// --- ENEMY ENTITY & SPRITE SETUP ---
const enemy = createEntity({
  id: "enemy_1",
  name: "Evil Bob",
  gridPos: { x: 10, y: 0, z: 10 }, // Placed slightly away from Bob
  stats: { STR: 12, DEX: 12, CON: 12, INT: 8, WIS: 8, CHA: 8 },
  speed: 5,
  hp: { current: 15, max: 15 },
  ac: 12,
  weaponDie: "1d6",
  mode: "explore"
});
World.get(getVoxelKey(10, 0, 10)).occupant = enemy.id;

const evilBobTexture = texLoader.load(evilBobTextureUrl);
evilBobTexture.magFilter = THREE.NearestFilter; 
evilBobTexture.minFilter = THREE.NearestFilter;
// Clone material so EvilBob can flip independently
const evilBobMaterial = new THREE.SpriteMaterial({ map: evilBobTexture.clone(), transparent: true });
const enemySprite = new THREE.Sprite(evilBobMaterial);

enemySprite.center.set(0.5, 0); 
enemySprite.scale.set(VOXEL_SIZE, VOXEL_SIZE, 1);
enemySprite.position.copy(getSpriteWorldPos(enemy.gridPos));
scene.add(enemySprite);

const enemyAI = createWanderAI(enemy, enemySprite, 4);

// --- BATTLE STATE ---
export let battleParticipants = [];
// Bumped every time battle is entered or exited, so timers scheduled by a
// previous battle can detect they're stale even if currentMode flips back
// to 'battle' before they fire.
let battleSessionId = 0;

// SPRITE FLIP HELPER
function updateSpriteFacing(sprite, isFacingRight) {
  if (isFacingRight) {
    sprite.material.map.repeat.x = -1;  
    sprite.material.map.offset.x = 1; 
  } else {
    sprite.material.map.repeat.x = 1;   
    sprite.material.map.offset.x = 0; 
  }
}

// VISUAL AIDS (HIGHLIGHT & PATH DOTS)
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
let isPointerDown = false;
let currentMoveTargetKey = null;
let enemyPath = []; // Tracks AI movement animation

// Handles resetting resources and triggering AI
function onTurnStart(entity) {
  entity.turnResources.actionAvailable = true;
  entity.turnResources.bonusActionAvailable = true;
  entity.turnResources.spellAvailable = true;
  entity.turnResources.moveRemaining = entity.speed;

  if (entity.id === player.id) {
    refreshReachableTiles();
  } else if (entity.id === enemy.id) {
    updateVoxelTints(currentArenaMap, null, true);

    // GUARD: Only trigger AI if the battle wasn't abruptly ended (or replaced by a new one)
    const scheduledSession = battleSessionId;
    setTimeout(() => {
      if (currentMode === 'battle' && battleSessionId === scheduledSession) processEnemyAI();
    }, 500);
  }
}

function endBattleSequence(message) {
  console.log(`[Combat] ${message}`);
  currentMode = 'explore';
  battleSessionId++;
  document.getElementById('mode-text').innerText = 'Explore';
  exitBattle();
  currentArenaMap = null;
  updateVoxelVisibility(null, false);
  
  if (!isDefeated(enemy)) enemyAI.isPaused = false;
  
  resetBattleState(battleParticipants);
  battleParticipants = [];
  refreshReachableTiles();
}

function processEnemyAI() {
  if (currentMode !== 'battle') return;

  const aiDecision = takeEnemyTurn(enemy, player);

  if (aiDecision.action === 'move') {
    enemyPath = aiDecision.path;
  } else if (aiDecision.action === 'attack') {
    const res = aiDecision.result;
    if (res.hit) {
      console.log(`[Combat] ${enemy.name} hits for ${res.damage}! (HP: ${player.hp.current}/${player.hp.max})`);
      if (isDefeated(player)) {
        endBattleSequence("GAME OVER. You have been defeated!");
        return; 
      }
    } else {
      console.log(`[Combat] ${enemy.name} misses! (Rolled ${res.attackTotal} vs AC ${player.ac})`);
    }

    const scheduledSessionAttack = battleSessionId;
    setTimeout(() => {
      if (currentMode === 'battle' && battleSessionId === scheduledSessionAttack) nextTurn(battleParticipants, onTurnStart);
    }, 1000);

  } else if (aiDecision.action === 'end') {
    const scheduledSessionEnd = battleSessionId;
    setTimeout(() => {
      if (currentMode === 'battle' && battleSessionId === scheduledSessionEnd) nextTurn(battleParticipants, onTurnStart);
    }, 500);
  }
}

// Modify refreshReachableTiles to dynamically use moveRemaining
function refreshReachableTiles() {
  if (currentMode === 'battle') {
    const currentSpeed = player.turnResources ? player.turnResources.moveRemaining : player.speed;
    currentReachable = getReachableVoxels(player.gridPos, currentSpeed);
    updateVoxelTints(currentArenaMap, currentReachable, true);
  } else {
    updateVoxelTints(null, null, false);
  }
}

function processClickToMove(clientX, clientY, isDownEvent = false) {
  if (!inputRules[currentMode].click) return;
  // Strict lock for Battle mode turns: No interruptions allowed
  if (currentMode === 'battle' && currentPath.length > 0) return; 

  const intersect = getGridIntersection(clientX, clientY);
  if (!intersect) return;

  const { gx, gz } = intersect;
  const targetKey = getVoxelKey(gx, 0, gz);

  if (currentMode === 'battle') {
    if (!currentReachable || !currentReachable.has(targetKey)) {
      if (isDownEvent) console.warn(`Rejected: Tile ${targetKey} is outside speed range or arena bounds.`);
      return; 
    }
  }

  // Prevent spamming pathfinder calculations if we are holding the mouse and hovering the same tile
  if (!isDownEvent && targetKey === currentMoveTargetKey && currentPath.length > 0) return;

  if (World.has(targetKey)) {
    const allowDiagonals = currentMode === 'explore'; 
    const path = findPath(player.gridPos, { x: gx, y: 0, z: gz }, allowDiagonals);
    
    if (currentMode === 'battle' && path.length > player.speed) return;

    if (path.length > 0) {
      currentPath = path; // Instant override of the path!
      currentMoveTargetKey = targetKey;
      
      highlightMesh.position.set(gx * VOXEL_SIZE, (VOXEL_SIZE / 2) + 0.02, gz * VOXEL_SIZE);
      highlightMesh.visible = true;
      
      if (isDownEvent) {
         clickPulseTime = 1.0; 
         highlightMesh.material.color.setHex(0xffff00); 
      }
      
      if (currentMode === 'explore') {
          pathGroup.clear(); // No dots in explore mode
      } else {
          updatePathDots(path);
      }
    }
  }
}

window.addEventListener('pointerdown', (e) => {
  if (e.target.tagName !== 'CANVAS') return; 
  if (e.button !== 0) return; 

  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  // Intercept clicks on ANY sprite so you can't walk through them
  const spriteIntersects = raycaster.intersectObjects([playerSprite, enemySprite]);
  if (spriteIntersects.length > 0) {
     const hitSprite = spriteIntersects[0].object;
     
     // Only trigger dialogue if we specifically clicked Evil Bob in Explore mode
     if (currentMode === 'explore' && hitSprite === enemySprite) {
        if (pathDistance(player.gridPos, enemy.gridPos) <= 2) {
            document.getElementById('dialogue-panel').style.display = 'block';
            document.getElementById('dialogue-name').innerText = enemy.name;
            isDialogueOpen = true;
        } else {
            console.log("Enemy is too far away to interact!");
        }
     }
     
     // Consume the click if a sprite was hit
     isPointerDown = false; 
     currentPath = [];      
     return; 
  }

  isPointerDown = true;
  processClickToMove(e.clientX, e.clientY, true);
});

// --- DIALOGUE BUTTONS & STATE ---
let isDialogueOpen = false;

// Fix 1: Stop UI clicks from falling through to the game world
document.getElementById('dialogue-panel').addEventListener('pointerdown', (e) => e.stopPropagation());

document.getElementById('btn-talk').addEventListener('click', () => {
  console.log(`${enemy.name} snarls: "You shouldn't be here..."`);
});

document.getElementById('btn-fight').addEventListener('click', () => {
  document.getElementById('dialogue-panel').style.display = 'none';
  isDialogueOpen = false;
  
  // Snap player to grid
  player.gridPos.x = Math.round(playerSprite.position.x / VOXEL_SIZE);
  player.gridPos.z = Math.round(playerSprite.position.z / VOXEL_SIZE);
  playerSprite.position.copy(getSpriteWorldPos(player.gridPos));
  
  // FIX 6: Force Enemy into the Player's 12x12 Arena Chunk
  const chunkSize = 12;
  const minX = Math.floor(player.gridPos.x / chunkSize) * chunkSize;
  const maxX = minX + chunkSize - 1;
  const minZ = Math.floor(player.gridPos.z / chunkSize) * chunkSize;
  const maxZ = minZ + chunkSize - 1;
  
  let ex = Math.round(enemySprite.position.x / VOXEL_SIZE);
  let ez = Math.round(enemySprite.position.z / VOXEL_SIZE);
  
  if (ex < minX || ex > maxX || ez < minZ || ez > maxZ) {
    const oldV = World.get(getVoxelKey(ex, 0, ez));
    if (oldV && oldV.occupant === enemy.id) oldV.occupant = null; 
    
    // Clamp to arena edges
    ex = Math.max(minX, Math.min(ex, maxX));
    ez = Math.max(minZ, Math.min(ez, maxZ));
    
    // Safe Radial Search: Find nearest vacant tile in the chunk
    let found = false;
    for (let radius = 0; radius < 5; radius++) {
      for (let dx = -radius; dx <= radius; dx++) {
        for (let dz = -radius; dz <= radius; dz++) {
          const cx = ex + dx;
          const cz = ez + dz;
          if (cx >= minX && cx <= maxX && cz >= minZ && cz <= maxZ) {
            const checkV = World.get(getVoxelKey(cx, 0, cz));
            if (checkV && checkV.walkable && !checkV.occupant) {
              ex = cx;
              ez = cz;
              found = true;
              break;
            }
          }
        }
        if (found) break;
      }
      if (found) break;
    }
  }
  
  enemy.gridPos.x = ex;
  enemy.gridPos.z = ez;
  enemySprite.position.copy(getSpriteWorldPos(enemy.gridPos));
  
  const newV = World.get(getVoxelKey(ex, 0, ez));
  if (newV) newV.occupant = enemy.id;

  battleParticipants = [player, enemy];
  enemyAI.isPaused = true;

  currentMode = 'battle';
  battleSessionId++;
  document.getElementById('mode-text').innerText = 'Battle';
  highlightMesh.visible = false;
  pathGroup.clear();
  lastHoveredKey = null;
  clickPulseTime = 0;
  
  // Enter Battle Centered on PLAYER'S chunk (since enemy was pulled into it)
  const battleData = enterBattle(player.gridPos);
  currentArenaMap = battleData.arena;
  
  const centerX = (battleData.bounds.minX + battleData.bounds.maxX) / 2;
  const centerZ = (battleData.bounds.minZ + battleData.bounds.maxZ) / 2;
  arenaCenter.set(centerX * VOXEL_SIZE, 0, centerZ * VOXEL_SIZE);
  
  updateVoxelVisibility(currentArenaMap, true);
  refreshReachableTiles();
  
  // PHASE 7: Roll Initiative & Start Turn Queue
  rollInitiativeForParticipants(battleParticipants);
  UI.toggleBattleUI(true);
  UI.updateHUD(player);
  UI.updatePartyView(battleParticipants, player.id);
  UI.updateActionOrder(turnOrder, currentTurnIndex, battleParticipants);
  
  const activeEntity = getCurrentEntity(battleParticipants);
  onTurnStart(activeEntity); 
});

window.addEventListener('pointerup', (e) => {
  if (e.button === 0) isPointerDown = false;
  if (e.target.tagName !== 'CANVAS') return; 
  if (!inputRules[currentMode].click) return;
  if (currentPath.length > 0) return; 

  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  if (raycaster.intersectObjects([playerSprite, enemySprite]).length > 0) return; 

  const intersect = getGridIntersection(e.clientX, e.clientY);
  if (!intersect) return; 

  const { gx, gz } = intersect;
  const targetKey = getVoxelKey(gx, 0, gz);

  if (currentMode === 'battle') {
    if (!currentReachable || !currentReachable.has(targetKey)) {
      console.warn(`Rejected: Tile ${targetKey} is outside speed range or arena bounds.`);
      return; 
    }
  }

  if (World.has(targetKey)) {
    const allowDiagonals = currentMode === 'explore';
    const path = findPath(player.gridPos, { x: gx, y: 0, z: gz }, allowDiagonals);
    
    if (currentMode === 'battle' && path.length > player.turnResources.moveRemaining) {
       console.warn(`Rejected: Path length exceeds speed stat.`);
       return;
    }

    if (path.length > 0) {
      currentPath = path;
      currentMoveTargetKey = targetKey;
      clickPulseTime = 1.0; 
      highlightMesh.position.set(gx * VOXEL_SIZE, (VOXEL_SIZE / 2) + 0.02, gz * VOXEL_SIZE);
      highlightMesh.visible = true;
      highlightMesh.material.color.setHex(0xffff00); 
      
      if (currentMode === 'explore') {
          pathGroup.clear(); 
      } else {
          updatePathDots(path);
      }
    }
  }
});

window.addEventListener('pointermove', (e) => {
  if (e.target.tagName !== 'CANVAS') {
      document.body.style.cursor = 'default';
      return;
  }

  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

  const isWalkingManual = (keyState.w || keyState.a || keyState.s || keyState.d);
  const isWalkingInBattle = (currentMode === 'battle' && currentPath.length > 0);
  
  enemySprite.material.color.setHex(0xffffff);
  document.body.style.cursor = 'default';

  if (currentMode === 'explore' && !isDialogueOpen) {
     raycaster.setFromCamera(mouse, camera);
     const spriteIntersects = raycaster.intersectObject(enemySprite);
     if (spriteIntersects.length > 0 && pathDistance(player.gridPos, enemy.gridPos) <= 2) {
         enemySprite.material.color.setHex(0xffff00); 
         document.body.style.cursor = 'pointer';
     }
  }

  if (!inputRules[currentMode].click || isWalkingManual || isWalkingInBattle) {
    highlightMesh.visible = false;
    lastHoveredKey = null;
    if (!isWalkingInBattle) pathGroup.clear();
    return;
  }
  
  if (isPointerDown && currentMode === 'explore') {
     processClickToMove(e.clientX, e.clientY, false);
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

      if (voxel && voxel.walkable && (currentMode === 'explore' || (currentReachable && currentReachable.has(targetKey)))) {
        highlightMesh.position.set(gx * VOXEL_SIZE, (VOXEL_SIZE / 2) + 0.02, gz * VOXEL_SIZE);
        highlightMesh.visible = true;

        if (currentMode === 'battle') {
          const path = findPath(player.gridPos, { x: gx, y: 0, z: gz }, false);
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

window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();
  
if (key === 'escape' && isDialogueOpen) {
     document.getElementById('dialogue-panel').style.display = 'none';
     isDialogueOpen = false;
  }

  if (keyState.hasOwnProperty(key)) {
    keyState[key] = true;
  }

  if (key === 'b') {
    currentMode = currentMode === 'explore' ? 'battle' : 'explore';
    battleSessionId++;
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
      
      const centerX = (battleData.bounds.minX + battleData.bounds.maxX) / 2;
      const centerZ = (battleData.bounds.minZ + battleData.bounds.maxZ) / 2;
      arenaCenter.set(centerX * VOXEL_SIZE, 0, centerZ * VOXEL_SIZE);
      
      updateVoxelVisibility(currentArenaMap, true);
      
      // FIX: Only the player starts in combat automatically
      battleParticipants = [player];
      
      const enemyInChunk = currentArenaMap.has(getVoxelKey(enemy.gridPos.x, 0, enemy.gridPos.z));
      
      // Freeze them if they are in the chunk, hide them if they aren't
      if (enemyInChunk) {
         enemyAI.isPaused = true;
         enemySprite.visible = true;
      } else {
         enemySprite.visible = false;
      }
      
      refreshReachableTiles();

      rollInitiativeForParticipants(battleParticipants);
      UI.toggleBattleUI(true);
      UI.updateHUD(player);
      UI.updatePartyView(battleParticipants, player.id);
      UI.updateActionOrder(turnOrder, currentTurnIndex, battleParticipants);
      
      const activeEntity = getCurrentEntity(battleParticipants);
      if (activeEntity) onTurnStart(activeEntity);

    } else {
      // EXITING BATTLE
      exitBattle();
      currentArenaMap = null;
      updateVoxelVisibility(null, false);
      refreshReachableTiles();
      
      if (!isDefeated(enemy)) {
         enemyAI.isPaused = false;
         enemySprite.visible = true; 
      }
      
      resetBattleState(battleParticipants);
      battleParticipants = [];
      
      UI.toggleBattleUI(false);
      UI.clearDiceLog();
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

  if (currentMode === 'explore') {
    enemyAI.update(dt, currentHeading);
  }

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
  const isWalkingManual = (keyState.w || keyState.a || keyState.s || keyState.d);

  if (isWalkingManual && currentMode === 'explore' && currentPath.length > 0) {
    currentPath = [];
    currentMoveTargetKey = null;
  }

  if (currentPath.length > 0) {
    const targetNode = currentPath[0];
    const targetWorldPos = getSpriteWorldPos(targetNode);
    const step = 8 * dt;
    
    if (playerSprite.position.distanceTo(targetWorldPos) <= step) {
      playerSprite.position.copy(targetWorldPos);
      
      const oldVoxel = World.get(getVoxelKey(player.gridPos.x, 0, player.gridPos.z));
      if (oldVoxel && oldVoxel.occupant === player.id) oldVoxel.occupant = null;
      
      player.gridPos = currentPath.shift();
      
      const newVoxel = World.get(getVoxelKey(player.gridPos.x, 0, player.gridPos.z));
      if (newVoxel) newVoxel.occupant = player.id;

      if (currentMode === 'battle') {
        player.turnResources.moveRemaining -= 1;
        updatePathDots(currentPath);
      }
      
      if (currentPath.length === 0 && currentMode === 'battle') {
         refreshReachableTiles();
      }
    } else {
      const dir = targetWorldPos.clone().sub(playerSprite.position).normalize();
      
      const dot = dir.x * Math.cos(currentHeading) - dir.z * Math.sin(currentHeading);
      if (Math.abs(dot) > 0.05) updateSpriteFacing(playerSprite, dot > 0);
      
      playerSprite.position.add(dir.multiplyScalar(step));
      
      if (currentMode === 'explore') {
        const newGridX = Math.round(playerSprite.position.x / VOXEL_SIZE);
        const newGridZ = Math.round(playerSprite.position.z / VOXEL_SIZE);
        
        if (newGridX !== player.gridPos.x || newGridZ !== player.gridPos.z) {
          const oldVoxel = World.get(getVoxelKey(player.gridPos.x, 0, player.gridPos.z));
          // FIX: Only clear if we own it
          if (oldVoxel && oldVoxel.occupant === player.id) oldVoxel.occupant = null;
          
          const newVoxel = World.get(getVoxelKey(newGridX, 0, newGridZ));
          if (newVoxel) newVoxel.occupant = player.id;
          
          player.gridPos.x = newGridX;
          player.gridPos.z = newGridZ;
        }
      }
    }
  } 
  // ENEMY AI MOVEMENT LOGIC
  if (enemyPath.length > 0) {
    const targetNode = enemyPath[0];
    const targetWorldPos = getSpriteWorldPos(targetNode);
    const step = 5 * dt; 
    
    if (enemySprite.position.distanceTo(targetWorldPos) <= step) {
      enemySprite.position.copy(targetWorldPos);
      
      const oldVoxel = World.get(getVoxelKey(enemy.gridPos.x, 0, enemy.gridPos.z));
      if (oldVoxel && oldVoxel.occupant === enemy.id) oldVoxel.occupant = null;
      
      enemy.gridPos = enemyPath.shift();
      
      const newVoxel = World.get(getVoxelKey(enemy.gridPos.x, 0, enemy.gridPos.z));
      if (newVoxel) newVoxel.occupant = enemy.id;
      
      enemy.turnResources.moveRemaining -= 1;
      
      if (enemyPath.length === 0) {
         processEnemyAI(); // Trigger the attack check now that movement is finished
      }
    } else {
      const dir = targetWorldPos.clone().sub(enemySprite.position).normalize();
      const dot = dir.x * Math.cos(currentHeading) - dir.z * Math.sin(currentHeading);
      if (Math.abs(dot) > 0.05) updateSpriteFacing(enemySprite, dot > 0);
      enemySprite.position.add(dir.multiplyScalar(step));
    }
  }
  else if (currentMode === 'explore' && inputRules.explore.keyboard) {
    let rawDx = 0, rawDz = 0;
    if (keyState.w) rawDz -= 1; 
    if (keyState.s) rawDz += 1; 
    if (keyState.a) rawDx -= 1; 
    if (keyState.d) rawDx += 1; 

    if (rawDx !== 0 || rawDz !== 0) {
      if (rawDx !== 0) updateSpriteFacing(playerSprite, rawDx > 0);

      if (rawDx !== 0 && rawDz !== 0) {
        const invSqrt2 = 1 / Math.sqrt(2);
        rawDx *= invSqrt2;
        rawDz *= invSqrt2;
      }

      const targetHeading = (rotationStep * Math.PI / 2) + cameraConfigs.explore.headingOffset;
      const cosH = Math.cos(targetHeading);
      const sinH = Math.sin(targetHeading);

      const dx = rawDx * cosH + rawDz * sinH;
      const dz = -rawDx * sinH + rawDz * cosH;

      const moveSpeed = 8;
      const stepX = dx * moveSpeed * dt;
      const stepZ = dz * moveSpeed * dt;

      const newX = playerSprite.position.x + stepX;
      const newZ = playerSprite.position.z + stepZ;

      const nextGX = Math.round(newX / VOXEL_SIZE);
      const nextGZ = Math.round(newZ / VOXEL_SIZE);
      
      const canWalk = (gx, gz) => {
         const v = World.get(getVoxelKey(gx, 0, gz));
         return v && v.walkable && (!v.occupant || v.occupant === player.id);
      };

      if (canWalk(nextGX, nextGZ)) {
        playerSprite.position.x = newX;
        playerSprite.position.z = newZ;
      } else {
        if (canWalk(nextGX, Math.round(playerSprite.position.z / VOXEL_SIZE))) playerSprite.position.x = newX;
        if (canWalk(Math.round(playerSprite.position.x / VOXEL_SIZE), nextGZ)) playerSprite.position.z = newZ;
      }

      const newGridX = Math.round(playerSprite.position.x / VOXEL_SIZE);
      const newGridZ = Math.round(playerSprite.position.z / VOXEL_SIZE);
      
      if (newGridX !== player.gridPos.x || newGridZ !== player.gridPos.z) {
        const oldVoxel = World.get(getVoxelKey(player.gridPos.x, 0, player.gridPos.z));
        // FIX: Strict ownership check
        if (oldVoxel && oldVoxel.occupant === player.id) oldVoxel.occupant = null;
        
        const newVoxel = World.get(getVoxelKey(newGridX, 0, newGridZ));
        if (newVoxel) newVoxel.occupant = player.id;
        
        player.gridPos.x = newGridX;
        player.gridPos.z = newGridZ;
      }
    }
  }

  // FIX 2: Auto-close dialogue if player walks out of range
  if (isDialogueOpen && pathDistance(player.gridPos, enemy.gridPos) > 2) {
     document.getElementById('dialogue-panel').style.display = 'none';
     isDialogueOpen = false;
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