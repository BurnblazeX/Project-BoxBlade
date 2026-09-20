import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import { World, getVoxelKey, createTestArea, createEntity, pathDistance, findPath, enterBattle, exitBattle, getReachableVoxels, addToInventory, isInInteractRange, isStandable, getColumnTop, CHUNK_SIZE, isSolid } from './world.js';
import { createObject, rollLootTable } from './objects.js';
import { initWorldRender, VOXEL_SIZE, updateVoxelTints, updateVoxelVisibility, worldInstancedMesh, keyForInstance, voxelIndexMap } from './render.js';
import { createWanderAI } from './ai.js';
import { toggleBoxGridDebug, refreshBoxGridDebug } from './debug.js';
import { createBoxGridAt, gridOriginFor, GRID_DIM, marchOccupancy, sphereTrace } from './boxgrid.js';
import { makeClipSamples, updateCharacterClipping } from './sprites.js';
import { createPerfOverlay } from './perf.js';
import { runComputeSmokeTest, createDistanceTexture, updateDistanceTexture,
         runGPUMarch, createShadowColorNode, applyShadowMaterial,
         restoreOriginalMaterial, followShadowGrid,
         createTexelAtlas, clearTexelAtlas, createAtlasShadePass, updateShadeJobs,
         runAtlasShade, createAtlasColorNode,
         createCascadeTargets, createCascadeCameras, createShadowScene, addShadowCaster,
         createLightDepthMaterials, updateCascadeCameras, renderCascades,
         createCascadeUniforms, writeCascadeUniforms, createBayerTexture,
         createCsmShadowTSL, readCascade, hideCascadeView, NORMAL_BIAS_TEXELS,
         probeCascade,
         createCascadeBlit, drawCascadeBlit } from './gpu.js';
import { CASCADE_COUNT, fitCascade, cascadeExtent, cascadeDepth, SUN_ANGULAR_SIZE } from './csm.js';
import { atlasLayout, buildShadeJobs } from './atlas.js';
import { createConsole, installConsole } from './console.js';
import { rollInitiativeForParticipants, resetBattleState, turnOrder, currentTurnIndex, getCurrentEntity, nextTurn, addParticipant } from './battle.js';
import * as UI from './ui.js';
import { performAttack, isDefeated, takeEnemyTurn, isInMeleeRange } from './combat.js';

import bobTextureUrl from '../assets/sprites/character_Bob.png'
import evilBobTextureUrl from '../assets/sprites/character_EvilBob.png'
import treeTextureUrl from '../assets/sprites/decor_tree.png'
import chestTextureUrl from '../assets/sprites/container_chest.png'
import chestOpenTextureUrl from '../assets/sprites/container_chest_open.png'

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
        const v = World.get(getVoxelKey(enemy.gridPos.x, enemy.gridPos.y, enemy.gridPos.z));
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

// WebGPU, not WebGL: the texel-space shading architecture needs compute shaders
// and TSL, neither of which WebGL2 can provide. Everything imports from
// 'three/webgpu' so materials resolve to their NodeMaterial equivalents.
const renderer = new THREE.WebGPURenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('app').appendChild(renderer.domElement);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
// ~23 degrees elevation. The old (10,20,10) was ~55 degrees, which put the sun
// nearly overhead and left every shadow hidden under the block casting it.
// A low sun is also what makes voxel shadow blockiness legible.
dirLight.position.set(10, 6, 10);
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
World.get(getVoxelKey(player.gridPos.x, player.gridPos.y, player.gridPos.z)).occupant = player.id;

const texLoader = new THREE.TextureLoader();
const bobTexture = texLoader.load(bobTextureUrl);
bobTexture.magFilter = THREE.NearestFilter;
bobTexture.minFilter = THREE.NearestFilter;
bobTexture.colorSpace = THREE.SRGBColorSpace;

// Ground-anchored plane, same footprint as the old Sprite (VOXEL_SIZE square).
// Characters always rotate around Y to face the camera's heading (set per-frame
// in animate()), plus a partial lean-back toward the camera's pitch, capped at
// MAX_CHARACTER_TILT - a full billboard tilt (matching the camera's real ~35-60
// degree pitch) let the top of the quad swing into adjacent geometry like the
// tree's fixed planes; staying fully vertical (no lean at all) looked squished
// under this game's steep camera angles. The cap is a middle ground.
const MAX_CHARACTER_TILT = THREE.MathUtils.degToRad(22.5);
const characterGeo = new THREE.PlaneGeometry(VOXEL_SIZE, VOXEL_SIZE * 2); // 2 voxels tall
characterGeo.translate(0, VOXEL_SIZE, 0); 

function createCharacterMesh(texture) {
  // alphaTest discards fully-transparent pixels before the depth test, so this
  // mesh's invisible corners don't still write depth and occlude what's behind it.
  const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, alphaTest: 0.5, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(characterGeo, material);
  mesh.rotation.order = 'YXZ'; // yaw (heading) applied before pitch (lean), so lean tilts around the already-yawed local X axis
  mesh.userData.drawingOnTop = false;
  return mesh;
}

// Sample points matching characterGeo's footprint, used to detect the frames
// where the quad is actually penetrating a block. See js/sprites.js.
const characterClipSamples = makeClipSamples(VOXEL_SIZE, VOXEL_SIZE * 2);

// Clone the texture so this specific character can flip independently
const playerTex = bobTexture.clone();
const playerSprite = createCharacterMesh(playerTex);

// An entity's gridPos.y is the block it stands ON, so its sprite sits on that
// block's top face: centre of block y, plus half a block up.
const getSpriteWorldPos = (gridPos) => new THREE.Vector3(
  gridPos.x * VOXEL_SIZE,
  (gridPos.y * VOXEL_SIZE) + (VOXEL_SIZE / 2),
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
World.get(getVoxelKey(enemy.gridPos.x, enemy.gridPos.y, enemy.gridPos.z)).occupant = enemy.id;

const evilBobTexture = texLoader.load(evilBobTextureUrl);
evilBobTexture.magFilter = THREE.NearestFilter;
evilBobTexture.minFilter = THREE.NearestFilter;
evilBobTexture.colorSpace = THREE.SRGBColorSpace;
// Clone the texture so EvilBob can flip independently
const enemySprite = createCharacterMesh(evilBobTexture.clone());
enemySprite.position.copy(getSpriteWorldPos(enemy.gridPos));
scene.add(enemySprite);

const enemyAI = createWanderAI(enemy, enemySprite, 4);

// --- WORLD OBJECTS (Phase 10: Tree, Chest, Barrel) ---
// Untextured colored placeholder sprites - no dedicated object art yet.
function createObjectSprite(color) {
  const mat = new THREE.SpriteMaterial({ color, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.center.set(0.5, 0);
  sprite.scale.set(VOXEL_SIZE, VOXEL_SIZE, 1);
  return sprite;
}

const treeTexture = texLoader.load(treeTextureUrl);
treeTexture.magFilter = THREE.NearestFilter;
treeTexture.minFilter = THREE.NearestFilter;
treeTexture.colorSpace = THREE.SRGBColorSpace;

// Decor (tree): fixed-orientation ground-planted cutout, NOT a camera-facing
// Sprite - meshes stay put as the camera rotates, like cardboard planted in
// the world rather than a character that always faces you.
// Cruciform: 4 planes at 45-degree increments (0/45/90/135), so at least one is
// always within 22.5 degrees of face-on. Plain 0/90 left a gap in battle view,
// since battle mode's camera heading is offset 45 degrees from explore's.
// PlaneGeometry's default local axes already line up as a vertical standee
// (local Y = world up, normal along world Z), so the first plane needs no rotation.
// 2 voxels wide and 3 voxels tall, vs. the 1-block-tall character sprites.
// (A horizontal top face is planned too, once that texture exists - not yet.)
const treePlaneGeo = new THREE.PlaneGeometry(VOXEL_SIZE * 2, VOXEL_SIZE * 3);
treePlaneGeo.translate(0, VOXEL_SIZE * 1.5, 0); // anchor the bottom edge at local origin, like the sprites' center.set(0.5, 0)
const treeMaterial = new THREE.MeshBasicMaterial({
  map: treeTexture, transparent: true, side: THREE.DoubleSide,
  alphaTest: 0.5 // discard fully-transparent pixels before the depth test, so they don't occlude what's behind
});
const treeMeshes = [];
const worldObjects = [];

function createTreeAt(gridPos, id) {
  const treeObject = createObject({
    id,
    name: "Tree",
    subType: "Decor",
    model: "tree",
    gridPos,
    blocking: true
  });

  const treeMesh = new THREE.Group();
  treeMesh.add(
    new THREE.Mesh(treePlaneGeo, treeMaterial),
    new THREE.Mesh(treePlaneGeo, treeMaterial),
    new THREE.Mesh(treePlaneGeo, treeMaterial),
    new THREE.Mesh(treePlaneGeo, treeMaterial)
  );
  treeMesh.children[1].rotation.y = Math.PI / 2;
  treeMesh.children[2].rotation.y = Math.PI / 4;
  treeMesh.children[3].rotation.y = (3 * Math.PI) / 4;
  treeMesh.position.copy(getSpriteWorldPos(treeObject.gridPos));
  scene.add(treeMesh);

  const voxel = World.get(getVoxelKey(treeObject.gridPos.x, treeObject.gridPos.y, treeObject.gridPos.z));
  if (voxel) voxel.occupant = treeObject.id;

  worldObjects.push({ data: treeObject, mesh: treeMesh });
  treeMeshes.push(treeMesh);
  return treeObject;
}

function createTrees(treePositions) {
  treePositions.forEach((gridPos, index) => {
    createTreeAt(gridPos, `obj_tree_${index + 1}`);
  });
}

createTrees([
  { x: 5, y: 0, z: 15 },
  { x: 8, y: 0, z: 14 },
  { x: 13, y: 0, z: 9 },
  { x: 14, y: 0, z: 9 },
  { x: 15, y: 0, z: 9 },
  { x: 14, y: 0, z: 8 },
  { x: 15, y: 0, z: 8 },
  { x: 17, y: 0, z: 9 }
]);



const chest = createObject({
  id: "obj_chest_1",
  name: "Old Chest",
  subType: "Container",
  model: "chest",
  gridPos: { x: 12, y: 0, z: 5 },
  blocking: true,
  lootTable: [
    { itemId: "gold_coin", weight: 5, minQty: 3, maxQty: 10 } // example loot
  ]
});
const chestTexture = texLoader.load(chestTextureUrl);
chestTexture.magFilter = THREE.NearestFilter;
chestTexture.minFilter = THREE.NearestFilter;
chestTexture.colorSpace = THREE.SRGBColorSpace;

const chestOpenTexture = texLoader.load(chestOpenTextureUrl);
chestOpenTexture.magFilter = THREE.NearestFilter;
chestOpenTexture.minFilter = THREE.NearestFilter;
chestOpenTexture.colorSpace = THREE.SRGBColorSpace;

const chestSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: chestTexture, transparent: true }));
chestSprite.center.set(0.5, 0);
chestSprite.scale.set(VOXEL_SIZE, VOXEL_SIZE, 1);
chestSprite.position.copy(getSpriteWorldPos(chest.gridPos));
scene.add(chestSprite);
World.get(getVoxelKey(chest.gridPos.x, chest.gridPos.y, chest.gridPos.z)).occupant = chest.id;
worldObjects.push({ data: chest, mesh: chestSprite });

const barrel = createObject({
  id: "obj_barrel_1",
  name: "Wooden Barrel",
  subType: "Container",
  model: "barrel",
  gridPos: { x: 15, y: 0, z: 15 },
  blocking: true,
  fixedItem: { itemId: "apple", quantity: 2 }
});
const barrelSprite = createObjectSprite(0x8b5a2b);
barrelSprite.position.copy(getSpriteWorldPos(barrel.gridPos));
scene.add(barrelSprite);
World.get(getVoxelKey(barrel.gridPos.x, barrel.gridPos.y, barrel.gridPos.z)).occupant = barrel.id;
worldObjects.push({ data: barrel, mesh: barrelSprite });

// Every world object's data + render node, so battle-arena visibility can be
// driven generically instead of hand-listing objects at each call site.
// Mirrors how enemySprite/updateVoxelVisibility hide things outside the arena:
// in explore mode everything shows; in battle, only objects inside the current
// arena chunk render at all.
function updateObjectVisibility(arenaMap, isBattle) {
  for (const { data, mesh } of worldObjects) {
    mesh.visible = !isBattle || arenaMap.has(getVoxelKey(data.gridPos.x, data.gridPos.y, data.gridPos.z));
  }
}

// Lookup from a clicked/hovered sprite to its underlying data. The tree has no
// entry on purpose - it's pure scenery with no interaction panel at all.
const spriteToTarget = new Map([
  [enemySprite, enemy],
  [chestSprite, chest],
  [barrelSprite, barrel]
]);
// Base tint per hover-able sprite, so the hover-highlight reset restores each
// sprite's own look instead of stomping untextured object sprites back to white.
const spriteBaseColor = new Map([
  [enemySprite, 0xffffff],
  [chestSprite, 0xffffff],
  [barrelSprite, 0x8b5a2b]
]);
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
    dot.position.set(node.x * VOXEL_SIZE, (node.y * VOXEL_SIZE) + (VOXEL_SIZE / 2) + 0.1, node.z * VOXEL_SIZE);
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
    pitch: Math.PI / 3.43, 
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
// Helper: Convert screen coords to the grid cell whose top face was clicked.
// This raycasts the terrain mesh rather than a single horizontal plane - with
// real elevation there is no one ground height, and only a block's TOP face is
// somewhere an entity can stand, so side-face hits on walls are skipped.
// Instances carry no rotation, so the box geometry's local +Y normal is world up.
// Hits arrive sorted by distance, making the first top face the nearest surface.
function getGridIntersection(clientX, clientY) {
  mouse.x = (clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(clientY / window.innerHeight) * 2 + 1;

  if (!worldInstancedMesh) return null;
  raycaster.setFromCamera(mouse, camera);

  for (const hit of raycaster.intersectObject(worldInstancedMesh, false)) {
    if (!hit.face || hit.face.normal.y < 0.5) continue;
    const key = keyForInstance(hit.instanceId);
    if (!key) continue;
    const [gx, gy, gz] = key.split(',').map(Number);
    return { gx, gy, gz };
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
  updateObjectVisibility(null, false);

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

  const { gx, gy, gz } = intersect;
  const targetKey = getVoxelKey(gx, gy, gz);

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
    const path = findPath(player.gridPos, { x: gx, y: gy, z: gz }, allowDiagonals);
    
    if (currentMode === 'battle' && path.length > player.speed) return;

    if (path.length > 0) {
      currentPath = path; // Instant override of the path!
      currentMoveTargetKey = targetKey;
      
      highlightMesh.position.set(gx * VOXEL_SIZE, (gy * VOXEL_SIZE) + (VOXEL_SIZE / 2) + 0.02, gz * VOXEL_SIZE);
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

  // Intercept clicks on ANY sprite (including the tree) so you can't walk through them
  const spriteIntersects = raycaster.intersectObjects([playerSprite, enemySprite, chestSprite, barrelSprite, ...treeMeshes]);
  if (spriteIntersects.length > 0) {
     const hitSprite = spriteIntersects[0].object;
     let target = spriteToTarget.get(hitSprite); // undefined for the tree - no panel, by design

      // Check if the target is an Object instead of an Entity
     if (target && target.subType) {
        // Ignore Decor and Lights; only allow interaction with Containers
        if (target.subType !== "Container") {
           target = undefined; 
        }
     }

     if (currentMode === 'explore' && target) {
        if (pathDistance(player.gridPos, target.gridPos) <= 2) {
            openInteractionPanel(target);
        } else {
            console.log(`${target.name} is too far away to interact!`);
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

// --- INTERACTION PANEL (generic: enemy dialogue + object "Open") ---
let isDialogueOpen = false;
let interactionTarget = null;

function openInteractionPanel(target) {
  interactionTarget = target;
  isDialogueOpen = true;

  document.getElementById('dialogue-panel').style.display = 'block';
  document.getElementById('dialogue-name').innerText = target.name;

  // World objects come from createObject() and always include subType/model.
  // Entities (player/enemies) do not, so they are the only valid fight/talk targets.
  const isObject = !!target && typeof target.subType === 'string' && typeof target.model === 'string';
  const isContainer = isObject && target.subType === 'Container';
  const isEnemy = !isObject;
  document.getElementById('btn-talk').style.display = isEnemy ? 'inline-block' : 'none';
  document.getElementById('btn-fight').style.display = isEnemy ? 'inline-block' : 'none';
  document.getElementById('btn-open').style.display = isContainer ? 'inline-block' : 'none';

  updateInteractionButtons();
}

function closeInteractionPanel() {
  document.getElementById('dialogue-panel').style.display = 'none';
  isDialogueOpen = false;
  interactionTarget = null;
}

// Live adjacency gate: the panel opens from "look" range, but each action button
// only enables at true adjacency, re-checked every frame while the panel is open.
function updateInteractionButtons() {
  if (!interactionTarget) return;
  const inRange = isInInteractRange(player, interactionTarget);

  for (const id of ['btn-talk', 'btn-fight', 'btn-open']) {
    const btn = document.getElementById(id);
    if (btn.style.display === 'none') continue;
    btn.disabled = !inRange;
    btn.classList.toggle('disabled', !inRange);
  }
}

function formatItemName(itemId) {
  return itemId.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

// Fix 1: Stop UI clicks from falling through to the game world
document.getElementById('dialogue-panel').addEventListener('pointerdown', (e) => e.stopPropagation());

document.getElementById('btn-open').addEventListener('click', () => {
  const obj = interactionTarget;
  if (!obj || obj.subType !== 'Container') return;
  if (!isInInteractRange(player, obj)) return; 

  if (!obj.looted) {
    const result = obj.lootTable ? rollLootTable(obj.lootTable) : obj.fixedItem;
    if (result) {
      addToInventory(player, result.itemId, result.quantity);
      UI.logChatMessage(`Found ${result.quantity}x ${formatItemName(result.itemId)} in the ${obj.name}.`);
    }
    obj.looted = true;
  } else {
    UI.logChatMessage(`The ${obj.name} is already empty.`);
  }

  obj.state = 'open';

  // Find the visual mesh associated with this object data
  const objRender = worldObjects.find(wo => wo.data.id === obj.id);
  
  if (objRender && obj.model === 'chest') {
    const targetSprite = objRender.mesh;
    
    // Swap to pre-loaded Open Texture
    targetSprite.material.map = chestOpenTexture;
    targetSprite.material.needsUpdate = true;

    // 2-Second Rummaging State
    setTimeout(() => {
      obj.state = 'closed';
      targetSprite.material.map = chestTexture; 
      targetSprite.material.needsUpdate = true;
      UI.logChatMessage(`Finished checking the ${obj.name}.`);
    }, 2000);
    
  } else if (objRender) {
    // Fallback for Barrel (no open texture exists yet, just state delay)
    setTimeout(() => {
      obj.state = 'closed';
      UI.logChatMessage(`Finished checking the ${obj.name}.`);
    }, 2000);
  }
});

document.getElementById('btn-fight').addEventListener('click', () => {
  if (!interactionTarget || typeof interactionTarget.subType === 'string') return;
  closeInteractionPanel();

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
    const oldV = World.get(getVoxelKey(ex, enemy.gridPos.y, ez));
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
            const checkV = World.get(getVoxelKey(cx, enemy.gridPos.y, cz));
            if (checkV && isStandable(cx, enemy.gridPos.y, cz) && !checkV.occupant) {
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
  
  const newV = World.get(getVoxelKey(ex, enemy.gridPos.y, ez));
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
  updateObjectVisibility(currentArenaMap, true);
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
  if (raycaster.intersectObjects([playerSprite, enemySprite, chestSprite, barrelSprite, ...treeMeshes]).length > 0) return;

  const intersect = getGridIntersection(e.clientX, e.clientY);
  if (!intersect) return; 

  const { gx, gy, gz } = intersect;
  const targetKey = getVoxelKey(gx, gy, gz);

  if (currentMode === 'battle') {
    if (!currentReachable || !currentReachable.has(targetKey)) {
      console.warn(`Rejected: Tile ${targetKey} is outside speed range or arena bounds.`);
      return; 
    }
  }

  if (World.has(targetKey)) {
    const allowDiagonals = currentMode === 'explore';
    const path = findPath(player.gridPos, { x: gx, y: gy, z: gz }, allowDiagonals);
    
    if (currentMode === 'battle' && path.length > player.turnResources.moveRemaining) {
       console.warn(`Rejected: Path length exceeds speed stat.`);
       return;
    }

    if (path.length > 0) {
      currentPath = path;
      currentMoveTargetKey = targetKey;
      clickPulseTime = 1.0; 
      highlightMesh.position.set(gx * VOXEL_SIZE, (gy * VOXEL_SIZE) + (VOXEL_SIZE / 2) + 0.02, gz * VOXEL_SIZE);
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
  
  // Reset each interactable sprite to its own base tint (not a shared white -
  // chest/barrel are untextured color sprites, not tinted photos like enemySprite).
  for (const [sprite, baseColor] of spriteBaseColor) {
    sprite.material.color.setHex(baseColor);
  }
  document.body.style.cursor = 'default';

  if (currentMode === 'explore' && !isDialogueOpen) {
     raycaster.setFromCamera(mouse, camera);
     const hoverIntersects = raycaster.intersectObjects([enemySprite, chestSprite, barrelSprite]);
     if (hoverIntersects.length > 0) {
        const hovered = hoverIntersects[0].object;
        const target = spriteToTarget.get(hovered);
        if (target && pathDistance(player.gridPos, target.gridPos) <= 2) {
           hovered.material.color.setHex(0xffff00);
           document.body.style.cursor = 'pointer';
        }
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
    const { gx, gy, gz } = intersect;
    const targetKey = getVoxelKey(gx, gy, gz);

    if (targetKey !== lastHoveredKey) {
      lastHoveredKey = targetKey;
      const voxel = World.get(targetKey);

      if (voxel && isStandable(gx, gy, gz) && (currentMode === 'explore' || (currentReachable && currentReachable.has(targetKey)))) {
        highlightMesh.position.set(gx * VOXEL_SIZE, (gy * VOXEL_SIZE) + (VOXEL_SIZE / 2) + 0.02, gz * VOXEL_SIZE);
        highlightMesh.visible = true;

        if (currentMode === 'battle') {
          const path = findPath(player.gridPos, { x: gx, y: gy, z: gz }, false);
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

  // Debug: Alt+X toggles the boxGrid occupancy overlay. Matched on e.code, not
  // e.key, because holding Alt changes the reported character on several
  // keyboard layouts while the physical key code stays put. preventDefault
  // stops Alt from reaching the browser's own menu handling.
  if (e.altKey && e.code === 'KeyX') {
    e.preventDefault();
    toggleBoxGridDebug(scene, player.gridPos, worldInstancedMesh, currentMode);
    return;
  }

if (key === 'escape' && isDialogueOpen) {
     closeInteractionPanel();
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
      updateObjectVisibility(currentArenaMap, true);

      // FIX: Only the player starts in combat automatically
      battleParticipants = [player];
      
      const enemyInChunk = currentArenaMap.has(getVoxelKey(enemy.gridPos.x, enemy.gridPos.y, enemy.gridPos.z));
      
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
      updateObjectVisibility(null, false);
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
      
      const oldVoxel = World.get(getVoxelKey(player.gridPos.x, player.gridPos.y, player.gridPos.z));
      if (oldVoxel && oldVoxel.occupant === player.id) oldVoxel.occupant = null;
      
      player.gridPos = currentPath.shift();
      
      const newVoxel = World.get(getVoxelKey(player.gridPos.x, player.gridPos.y, player.gridPos.z));
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
          const oldVoxel = World.get(getVoxelKey(player.gridPos.x, player.gridPos.y, player.gridPos.z));
          // FIX: Only clear if we own it
          if (oldVoxel && oldVoxel.occupant === player.id) oldVoxel.occupant = null;
          
          const newVoxel = World.get(getVoxelKey(newGridX, player.gridPos.y, newGridZ));
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
      
      const oldVoxel = World.get(getVoxelKey(enemy.gridPos.x, enemy.gridPos.y, enemy.gridPos.z));
      if (oldVoxel && oldVoxel.occupant === enemy.id) oldVoxel.occupant = null;
      
      enemy.gridPos = enemyPath.shift();
      
      const newVoxel = World.get(getVoxelKey(enemy.gridPos.x, enemy.gridPos.y, enemy.gridPos.z));
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
         const y = player.gridPos.y; // free movement is same-level until Jump exists
         if (!isStandable(gx, y, gz)) return false;
         const v = World.get(getVoxelKey(gx, y, gz));
         return !v.occupant || v.occupant === player.id;
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
        const oldVoxel = World.get(getVoxelKey(player.gridPos.x, player.gridPos.y, player.gridPos.z));
        // FIX: Strict ownership check
        if (oldVoxel && oldVoxel.occupant === player.id) oldVoxel.occupant = null;
        
        const newVoxel = World.get(getVoxelKey(newGridX, player.gridPos.y, newGridZ));
        if (newVoxel) newVoxel.occupant = player.id;
        
        player.gridPos.x = newGridX;
        player.gridPos.z = newGridZ;
      }
    }
  }

  // Auto-close the panel if the player walks out of look-range of whichever
  // target is open; otherwise keep its action buttons' enabled state live.
  if (isDialogueOpen && interactionTarget) {
     if (pathDistance(player.gridPos, interactionTarget.gridPos) > 2) {
        closeInteractionPanel();
     } else {
        updateInteractionButtons();
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

  // Clamped billboard: characters turn to face the camera's heading, plus lean
  // back toward its pitch up to MAX_CHARACTER_TILT - enough to avoid looking
  // flat/squished, not enough to swing into adjacent geometry like the tree.
  const characterTilt = Math.min(currentPitch, MAX_CHARACTER_TILT);
  playerSprite.rotation.set(-characterTilt, currentHeading, 0);
  enemySprite.rotation.set(-characterTilt, currentHeading, 0);

  // Checked after the rotation is applied, since whether the quad penetrates
  // geometry depends on this frame's yaw and lean.
  updateCharacterClipping(playerSprite, characterClipSamples);
  updateCharacterClipping(enemySprite, characterClipSamples);

  // Rebuilds the overlay if the player has walked into a different chunk.
  // No-op when the overlay is off or the chunk is unchanged.
  refreshBoxGridDebug(scene, player.gridPos, currentMode);

  // Keep the shadow grid centred on the player. Without this it stays wherever
  // it was first built, so shading only works in that one 12x12 patch - which
  // is exactly why the wall at x=20 cast nothing.
  if (shadowsOn || atlasOn) {
    const want = gridOriginFor(currentMode, player.gridPos);
    if (!shadowOrigin || want.x !== shadowOrigin.x || want.z !== shadowOrigin.z) {
      followShadowGrid(shadowGrid, shadowTex, shadowGridOrigin, want);
      shadowOrigin = want;
      // The atlas covers the same footprint, so the pages that just came into
      // range hold no light yet and the ones that left are stale. Rewriting the
      // job list and re-dispatching is a buffer write plus one compute pass,
      // once per block stepped - not per frame.
      if (csmOn) csmRefresh();
      if (atlasOn) {
        const jobs = atlasJobsAt(want);
        if (updateShadeJobs(atlasPass, jobs)) atlasShade();
        else console.warn(`[atlas] ${jobs.count} jobs exceeds capacity ` +
                          `${atlasPass.capacity}; run bxb.atlas() twice to resize`);
      }
    }
  }
  perf.update(dt);

  renderer.render(scene, camera);
  // The depth map goes on screen AFTER the scene, scissored into the corner, so
  // the map and the geometry it was built from are visible side by side.
  if (csmBlit && csmBlit.visible) csmBlit.setMarker(getSpriteWorldPos(player.gridPos));
  drawCascadeBlit(renderer, csmBlit);
}

// --- DEBUG CONSOLE (bxb) ---
let shadowsOn = false;
let shadowGrid = null;
let shadowTex = null;
let shadowSun = null;       // live sun-direction uniform, so bxb.light can move it
let shadowGridOrigin = null; // live grid-origin uniform, so the grid can follow
let shadowOrigin = null;     // block origin the shadow grid is currently built at

// --- Phase C: texel atlas ---
let atlasOn = false;
let atlasTex = null;
let atlasInfo = null;   // layout: page grid, atlas dimensions
let atlasPass = null;   // compute kernel plus its job buffers and uniforms
let atlasShading = false;

function atlasJobsAt(origin) {
  return buildShadeJobs({
    originBlockX: origin.x, originBlockZ: origin.z,
    isSolid,
    instanceIdOf: (x, y, z) => voxelIndexMap.get(getVoxelKey(x, y, z)),
    pagesX: atlasInfo.pagesX, pageCount: atlasInfo.pageCount
  });
}

// Fire and forget. Compute is async, and the atlas simply shows the previous
// light for however many frames the dispatch takes - which is the property that
// makes this affordable in the first place. The guard stops a fast walk from
// queueing several passes over the same buffers.
function atlasShade() {
  if (!atlasPass || atlasShading) return;
  atlasShading = true;
  runAtlasShade(renderer, atlasPass).finally(() => { atlasShading = false; });
}


// --- The sun: cascaded shadow map ---
//
// Doc 6.1b. The sphere-traced path stays available on bxb.atlas() alone for
// comparison, but a sun ray has no termination distance, so marching it always
// truncated at the edge of C0 - which is the notch that showed up at the screen
// edge. The CSM has no such limit and is the doc's accepted exception.
let csmOn = false;
let csmTargets = null, csmCameras = null, csmDepth = null;
let csmUniforms = null, csmBayer = null, csmAngular = null, csmSteps = null;
let csmCascades = null;
let csmBlit = null;
let csmShadowScene = null;
let csmSoft = false;   // hard shadows first; softening is a separate problem

function csmViewCentre() {
  // The cascades follow the VIEW, not the player. The camera looks from 13 m
  // back and sees ~17 m past the player, so centring on the player alone wastes
  // half the near cascade behind the camera - which is exactly the asymmetry
  // that left the far edge of the screen unshaded.
  const p = getSpriteWorldPos(player.gridPos);
  const ahead = cascadeExtent(0) * 0.22;
  return {
    x: p.x + Math.sin(currentHeading) * -ahead,
    y: p.y,
    z: p.z + Math.cos(currentHeading) * -ahead
  };
}

function csmRefresh() {
  if (!csmOn) return;
  csmCascades = [];
  const centre = csmViewCentre();
  for (let i = 0; i < CASCADE_COUNT; i++) {
    csmCascades.push(fitCascade(i, centre, dirLight.position));
  }
  updateCascadeCameras(csmCascades, csmCameras);
  writeCascadeUniforms(csmUniforms, csmCascades, csmCameras);
  renderCascades(renderer, csmShadowScene, csmCameras, csmTargets, csmDepth);
}


// The kernel is compiled with or without the CSM lookup baked in, so switching
// the sun technique is a rebuild, not a uniform write. Tracked explicitly
// because reusing a pass built the other way is silent: everything runs, the
// shadow maps fill, and nothing on screen changes - which is exactly what
// toggling bxb.atlas() off and on used to do.
let atlasPassUsedCsm = false;
let atlasPassUsedSoft = false;
let atlasPassUsedShadowOnly = false;
let atlasPassUsedLevelDebug = false;
let atlasPassUsedDiff = false;
// Live shadow tuning. These are uniforms rather than constants so a sweep costs
// a re-shade instead of a kernel rebuild, which matters because bias and slope
// allowance have to be found by eye.
let csmBias = null, csmSlope = null, csmShadowOnly = false, csmLevelDebug = false;
let csmDiff = null, csmDiffOn = false;

function buildOrReuseAtlasPass(jobs) {
  const stale = !atlasPass || jobs.count > atlasPass.capacity
                || atlasPassUsedCsm !== csmOn || atlasPassUsedSoft !== csmSoft
                || atlasPassUsedShadowOnly !== csmShadowOnly
                || atlasPassUsedLevelDebug !== csmLevelDebug
                || atlasPassUsedDiff !== csmDiffOn;
  if (!stale) { updateShadeJobs(atlasPass, jobs); return false; }

  atlasPass = createAtlasShadePass({
    atlas: atlasTex, occTex: shadowTex, grid: shadowGrid, jobs,
    layout: atlasInfo, sunDirection: dirLight.position,
    shadowOnly: csmShadowOnly,
    sunShadow: csmOn
      ? createCsmShadowTSL(csmTargets.map(t => t.texture), csmUniforms,
                           csmAngular, csmBayer, csmSteps, csmSoft,
                           csmBias, csmSlope, csmLevelDebug,
                           csmDiffOn ? csmDiff : null)
      : null,
    // Room to walk into denser geometry without rebuilding the kernel.
    capacity: Math.min(atlasInfo.pageCount, jobs.count * 2 + 512)
  });
  atlasPassUsedCsm = csmOn;
  atlasPassUsedSoft = csmSoft;
  atlasPassUsedShadowOnly = csmShadowOnly;
  atlasPassUsedLevelDebug = csmLevelDebug;
  atlasPassUsedDiff = csmDiffOn;
  shadowSun = atlasPass.sunDirUniform;
  shadowGridOrigin = atlasPass.gridOriginUniform;
  return true;
}

function bxbLight(x, y, z) {
  dirLight.position.set(x, y, z);
  if (shadowSun) shadowSun.value.set(x, y, z).normalize();
  // The atlas holds baked light, so moving the sun has no effect at all until
  // the pages are rewritten. This is the cost of object-space shading, and the
  // whole reason it is cheap.
  if (csmOn) csmRefresh();
  if (atlasOn) atlasShade();
  return 'sun toward ' + x.toFixed(2) + ',' + y.toFixed(2) + ',' + z.toFixed(2);
}

const perf = createPerfOverlay();
perf.attachRenderer(renderer);

installConsole(createConsole({
  compute: {
    help: 'run the WebGPU compute smoke test',
    run: () => runComputeSmokeTest(renderer)
  },
  parity: {
    help: 'march the same rays on CPU and GPU and diff them',
    run: async (n = 256) => {
      const origin = gridOriginFor(currentMode, player.gridPos);
      const grid = createBoxGridAt(origin.x, origin.z);
      const tex = createDistanceTexture(grid);

      // Spread over the grid in random directions, so the comparison covers
      // axis-aligned, diagonal and grazing rays rather than one easy case.
      const rays = [];
      for (let i = 0; i < n; i++) {
        rays.push({
          origin: {
            x: grid.origin.x + Math.random() * GRID_DIM * 0.125,
            y: grid.origin.y + Math.random() * GRID_DIM * 0.125,
            z: grid.origin.z + Math.random() * GRID_DIM * 0.125
          },
          dir: { x: Math.random() * 2 - 1, y: Math.random() * 2 - 1, z: Math.random() * 2 - 1 },
          maxDist: 18
        });
      }

      const gpu = await runGPUMarch(renderer, tex, grid, rays);
      let agree = 0;
      const disagreements = [];
      for (let i = 0; i < n; i++) {
        // The GPU runs a sphere trace, so the sphere trace is the reference.
        // The binary DDA is reported alongside as a third opinion: it is a
        // completely different traversal over the same field, so if all three
        // move together the field itself is right, and if only the DDA differs
        // the disagreement is about grazing rays and grid boundaries rather
        // than about the shader.
        const cpuHit = sphereTrace(grid, rays[i].origin, rays[i].dir, rays[i].maxDist).hit;
        const gpuHit = gpu[i] > 0.5;
        if (cpuHit === gpuHit) agree++;
        else if (disagreements.length < 5) {
          disagreements.push({ i, cpu: cpuHit, gpu: gpuHit, ray: rays[i],
            dda: marchOccupancy(grid, rays[i].origin, rays[i].dir, rays[i].maxDist).hit });
        }
      }
      // Hit counts on each side: when one is zero the fault is systematic
      // (a bad threshold, an unbound texture) rather than a subtle stepping bug.
      let cpuHits = 0, gpuHits = 0, ddaHits = 0;
      for (let i = 0; i < n; i++) {
        if (sphereTrace(grid, rays[i].origin, rays[i].dir, rays[i].maxDist).hit) cpuHits++;
        if (marchOccupancy(grid, rays[i].origin, rays[i].dir, rays[i].maxDist).hit) ddaHits++;
        if (gpu[i] > 0.5) gpuHits++;
      }
      const out = { rays: n, agree, disagree: n - agree, pct: +(agree / n * 100).toFixed(2),
                    cpuHits, gpuHits, ddaHits, disagreements };
      console.log(out.disagree === 0
        ? `[gpu] sphere-trace parity PASSED - ${n}/${n} rays agree with the CPU reference`
        : `[gpu] sphere-trace parity FAILED - ${out.disagree}/${n} disagree`, out.disagreements);
      return out;
    }
  },
  shadows: {
    help: 'toggle marched shadows on the terrain (one light, per pixel)',
    run: () => {
      if (shadowsOn) {
        restoreOriginalMaterial(worldInstancedMesh);
        shadowsOn = false;
        shadowOrigin = null;
        return 'shadows off';
      }
      if (atlasOn) return 'turn bxb.atlas() off first - they share the terrain material';
      const origin = gridOriginFor(currentMode, player.gridPos);
      shadowGrid = createBoxGridAt(origin.x, origin.z, shadowGrid);
      shadowTex = shadowTex || createDistanceTexture(shadowGrid);
      updateDistanceTexture(shadowTex);
      const { node, sunDirUniform, gridOriginUniform } = createShadowColorNode({
        occTex: shadowTex,
        grid: shadowGrid,
        sunDirection: dirLight.position
      });
      shadowSun = sunDirUniform;
      shadowGridOrigin = gridOriginUniform;
      shadowOrigin = origin;
      applyShadowMaterial(worldInstancedMesh, node);
      shadowsOn = true;
      return `shadows on - directional sun, 12m march cap, grid covering blocks ` +
             `${origin.x}..${origin.x + CHUNK_SIZE - 1} x ${origin.z}..${origin.z + CHUNK_SIZE - 1}`;
    }
  },
  atlas: {
    help: 'toggle texel-space shading: one ray per texel, baked into an atlas',
    run: async () => {
      if (atlasOn) {
        restoreOriginalMaterial(worldInstancedMesh);
        atlasOn = false;
        shadowOrigin = null;
        return 'atlas off';
      }
      if (shadowsOn) return 'turn bxb.shadows() off first - they share the terrain material';

      const origin = gridOriginFor(currentMode, player.gridPos);
      shadowGrid = createBoxGridAt(origin.x, origin.z, shadowGrid);
      shadowTex = shadowTex || createDistanceTexture(shadowGrid);
      updateDistanceTexture(shadowTex);

      atlasInfo = atlasLayout(worldInstancedMesh.count);
      if (!atlasInfo.fits) {
        return `world too large for one atlas: ${atlasInfo.pageCount} pages ` +
               `would need ${atlasInfo.width}x${atlasInfo.height} texels`;
      }
      atlasTex = atlasTex || createTexelAtlas(atlasInfo);
      await clearTexelAtlas(renderer, atlasTex, atlasInfo);

      const jobs = atlasJobsAt(origin);
      buildOrReuseAtlasPass(jobs);
      shadowSun = atlasPass.sunDirUniform;
      shadowGridOrigin = atlasPass.gridOriginUniform;
      shadowOrigin = origin;

      await runAtlasShade(renderer, atlasPass);
      applyShadowMaterial(worldInstancedMesh,
        createAtlasColorNode({ atlas: atlasTex, layout: atlasInfo }));
      atlasOn = true;

      return `atlas on - ${atlasInfo.width}x${atlasInfo.height} texels ` +
             `(${(atlasInfo.bytes / 1048576).toFixed(1)} MB, ${atlasInfo.pageCount} pages), ` +
             `${jobs.count} faces / ${jobs.texels.toLocaleString()} rays shaded over blocks ` +
             `${origin.x}..${origin.x + CHUNK_SIZE - 1} x ${origin.z}..${origin.z + CHUNK_SIZE - 1}`;
    }
  },
  csm: {
    help: 'toggle the cascaded shadow map for the sun (use with bxb.atlas())',
    run: async () => {
      if (csmOn) {
        csmOn = false;
        if (atlasOn) {
          buildOrReuseAtlasPass(atlasJobsAt(shadowOrigin));
          atlasShade();
          return 'csm off - atlas back on the sphere-traced sun';
        }
        return 'csm off';
      }
      if (!csmTargets) {
        csmTargets = createCascadeTargets();
        csmCameras = createCascadeCameras();
        csmDepth = createLightDepthMaterials();
        csmUniforms = createCascadeUniforms();
        csmBayer = createBayerTexture();
        csmAngular = uniform(SUN_ANGULAR_SIZE);
        csmSteps = uniform(4);
        csmBias = uniform(NORMAL_BIAS_TEXELS);
        csmSlope = uniform(0.75);
        csmDiff = uniform(200);
      }
      // The caster set is explicit: a proxy sharing the terrain geometry and
      // instance buffer, in a scene of its own. Models opt in with another
      // addShadowCaster call; camera-facing billboards stay out.
      csmShadowScene = csmShadowScene || createShadowScene();
      addShadowCaster(csmShadowScene, worldInstancedMesh, csmDepth[0].material);
      csmOn = true;
      csmRefresh();
      // Rebuild and re-shade here rather than asking the user to toggle the
      // atlas: the only visible effect of the CSM is via the atlas, so leaving
      // it unbuilt makes the command look like it did nothing.
      let note = ' Run bxb.atlas() to shade with it.';
      if (atlasOn) {
        buildOrReuseAtlasPass(atlasJobsAt(shadowOrigin));
        atlasShade();
        note = ' Atlas re-shaded through it.';
      }
      const ext = [];
      for (let i = 0; i < CASCADE_COUNT; i++) ext.push(cascadeExtent(i).toFixed(0) + 'm');
      return 'csm on - ' + CASCADE_COUNT + ' cascades at 144 texels covering ' +
             ext.join(' / ') + '.' + note;
    }
  },
  csmview: {
    help: 'show the light depth map on screen (banded), or read it back for numbers',
    usage: 'bxb.csmview(level)  |  bxb.csmview(-1) to hide  |  bxb.csmview(0, true) for numbers',
    run: async (level = 0, numbers = false) => {
      if (level < 0) {
        if (csmBlit) csmBlit.visible = false;
        hideCascadeView();
        return 'cascade view hidden';
      }
      if (!csmTargets) return 'run bxb.csm() first';
      // The blit samples the texture through the same path the shadow lookup
      // uses, so it cannot be wrong about the map while the lookup is right.
      // Shown BANDED: the cascade spans 54 m and the terrain sits in a slice of
      // it, so a plain ramp of a healthy map looks like a blank white square.
      // Cleared texels draw purple, exact zeros red.
      csmBlit = csmBlit || createCascadeBlit(csmTargets, csmUniforms);
      csmBlit.setLevel(level);
      csmBlit.visible = true;
      if (!numbers) {
        return 'cascade ' + level + ' bottom-right. Banded depth; purple = never ' +
               'written, red = exactly 0. RED strip marks the map +u edge, GREEN ' +
               'marks +v - if green is at the BOTTOM of the overlay the display ' +
               'is flipped. The YELLOW crosshair is your own position projected ' +
               'through the cascade matrix: if it sits on your own patch of the ' +
               'map, the world-to-map transform is correct. ' +
               'bxb.csmview(' + level + ', true) for numbers, bxb.csmview(-1) to hide.';
      }
      return readCascade(renderer, csmTargets, level,
                         csmCascades && csmCascades[level].depth);
    }
  },
  shadowonly: {
    help: 'show the raw sun visibility term with no albedo or N.L, to judge acne',
    run: () => {
      if (!csmOn) return 'run bxb.csm() first';
      csmShadowOnly = !csmShadowOnly;
      if (atlasOn) { buildOrReuseAtlasPass(atlasJobsAt(shadowOrigin)); atlasShade(); }
      return csmShadowOnly
        ? 'shadow term only: white is lit, black is occluded. Acne shows as ' +
          'speckle or banding on surfaces that should be uniformly white.'
        : 'back to full shading';
    }
  },
  csmlevels: {
    help: 'tint every surface by which cascade shadows it, to find what a band is',
    run: () => {
      if (!csmOn) return 'run bxb.csm() first';
      csmLevelDebug = !csmLevelDebug;
      if (!csmLevelDebug && csmShadowOnly) csmShadowOnly = true;
      if (atlasOn) { buildOrReuseAtlasPass(atlasJobsAt(shadowOrigin)); atlasShade(); }
      return csmLevelDebug
        ? 'cascade levels: darkest = C0 (12.5cm), mid = C1 (25cm), light = C2 ' +
          '(50cm), full bright = beyond every cascade. Use with bxb.shadowonly(). ' +
          'If a band sits on a cascade boundary it is a selection problem; if it ' +
          'sits inside one flat region it is bias in that cascade.'
        : 'cascade level tint off';
    }
  },
  probe: {
    help: 'print both sides of the depth comparison under your feet, in metres',
    usage: 'bxb.probe(level)',
    run: async (level = 0) => {
      if (!csmOn || !csmCascades) return 'run bxb.csm() first';
      // The ground texel the player stands on: the block top face, nudged out
      // the same way the atlas nudges it, so this is the exact point the shade
      // kernel would hand the lookup.
      const w = getSpriteWorldPos(player.gridPos);
      const p = { x: w.x, y: w.y, z: w.z };
      return probeCascade(renderer, csmTargets, csmCascades[level],
                          p, { x: 0, y: 1, z: 0 }, level);
    }
  },
  csmdiff: {
    help: 'show the raw depth comparison instead of a shadow, to tell bias from a transform bug',
    usage: 'bxb.csmdiff(scale)  |  bxb.csmdiff(0) to turn off',
    run: (scale = 200) => {
      if (!csmOn) return 'run bxb.csm() first';
      csmDiffOn = scale > 0;
      if (csmDiffOn) csmDiff.value = scale;
      if (atlasOn) { buildOrReuseAtlasPass(atlasJobsAt(shadowOrigin)); atlasShade(); }
      if (!csmDiffOn) return 'depth diff off';
      return 'depth diff at x' + scale + ': MID GREY means the map and the ' +
             'receiver agree. Mostly mid grey with speckle = bias tuning. A ' +
             'large flat dark or bright region = the receiver depth and the ' +
             'stored depth disagree systematically, which is a transform bug, ' +
             'not a bias one. Try bxb.csmdiff(20) and bxb.csmdiff(2000) to see ' +
             'the magnitude.';
    }
  },
  bias: {
    help: 'normal-offset bias and slope allowance, both in shadow texels',
    usage: 'bxb.bias(normalTexels, slopeScale)',
    run: (b = NORMAL_BIAS_TEXELS, slope = 0.75) => {
      if (!csmBias) return 'run bxb.csm() first';
      csmBias.value = b;
      csmSlope.value = slope;
      if (atlasOn) atlasShade();
      return 'normal bias ' + b + ' texels, slope allowance x' + slope +
             '. Raise until acne clears, then stop - too much detaches shadows ' +
             'from their casters.';
    }
  },
  csmbands: {
    help: 'contour density of the depth map blit',
    usage: 'bxb.csmbands(40)',
    run: (n = 40) => {
      if (!csmBlit) return 'run bxb.csmview(0) first';
      csmBlit.bands.value = n;
      const range = csmCascades ? csmCascades[0].depth : cascadeDepth(0);
      return n + ' bands across the ' + range.toFixed(0) + ' m depth range (' +
             (range / n).toFixed(2) + ' m per stripe)';
    }
  },
  soften: {
    help: 'distance-based soft shadows: off by default until the hard ones are right',
    usage: 'bxb.soften(on?, angularSize?, steps?)',
    run: (on = true, a = SUN_ANGULAR_SIZE, steps = 4) => {
      if (!csmTargets) return 'run bxb.csm() first';
      csmSoft = !!on;
      csmAngular.value = a;
      csmSteps.value = steps;
      if (atlasOn) {
        buildOrReuseAtlasPass(atlasJobsAt(shadowOrigin));
        atlasShade();
      }
      return csmSoft
        ? 'soft shadows on - sun angular size ' + a + ' rad, ' + steps + ' steps'
        : 'soft shadows off - single-tap hard comparison';
    }
  },
  perf: {
    help: 'toggle the frame time graph',
    run: () => perf.toggle() ? 'perf graph on' : 'perf graph off'
  },
  grid: {
    help: 'toggle the boxGrid occupancy overlay (same as Alt+X)',
    run: () => toggleBoxGridDebug(scene, player.gridPos, worldInstancedMesh, currentMode)
      ? 'boxGrid overlay on' : 'boxGrid overlay off'
  },
  stats: {
    help: 'print world, grid and frame statistics',
    run: () => {
      const origin = gridOriginFor(currentMode, player.gridPos);
      const grid = createBoxGridAt(origin.x, origin.z);
      const s = perf.stats();
      const out = {
        blocks: World.size,
        gridOrigin: origin.x + ',' + origin.z,
        gridOccupied: grid.occupiedCount,
        gridVoxels: GRID_DIM ** 3,
        gridMB: +(grid.data.length / 1048576).toFixed(2),
        mode: currentMode,
        fps: +s.fps.toFixed(1),
        frameMs: +s.avg.toFixed(2)
      };
      console.table(out);
      return out;
    }
  },
  where: {
    help: 'print the player position and its chunk',
    run: () => {
      const p = player.gridPos;
      const out = {
        x: p.x, y: p.y, z: p.z,
        chunk: Math.floor(p.x / CHUNK_SIZE) + ',' + Math.floor(p.z / CHUNK_SIZE),
        mode: currentMode
      };
      console.log(out);
      return out;
    }
  },
  tp: {
    help: 'teleport the player to a block (y defaults to the column top)',
    usage: 'bxb.tp(x, z, y?)',
    run: (x, z, y) => {
      const top = (y === undefined || y === null) ? getColumnTop(x, z) : y;
      if (top === null || top === undefined) return 'no standable surface at ' + x + ',' + z;
      const old = World.get(getVoxelKey(player.gridPos.x, player.gridPos.y, player.gridPos.z));
      if (old && old.occupant === player.id) old.occupant = null;
      player.gridPos.x = x; player.gridPos.y = top; player.gridPos.z = z;
      const next = World.get(getVoxelKey(x, top, z));
      if (next) next.occupant = player.id;
      playerSprite.position.copy(getSpriteWorldPos(player.gridPos));
      currentPath = [];
      return 'teleported to ' + x + ',' + top + ',' + z;
    }
  },
  light: {
    help: 'set the sun direction (a vector pointing toward the sun)',
    usage: 'bxb.light(x, y, z)',
    run: (x, y, z) => {
      dirLight.position.set(x, y, z);
      // The shadow march needs the normalised direction, and it lives in a
      // uniform - writing dirLight alone would not reach the shader.
      if (shadowSun) shadowSun.value.set(x, y, z).normalize();
      return shadowsOn
        ? 'sun toward ' + x + ',' + y + ',' + z + ' (shadows updated)'
        : 'sun toward ' + x + ',' + y + ',' + z + ' (run bxb.shadows() to see it)';
    }
  },
  sun: {
    help: 'set the sun by angle - elevation 15-25 gives long readable shadows',
    usage: 'bxb.sun(azimuthDeg, elevationDeg)',
    run: (az = 45, el = 20) => {
      const a = az * Math.PI / 180, e = el * Math.PI / 180;
      const x = Math.cos(e) * Math.sin(a), y = Math.sin(e), z = Math.cos(e) * Math.cos(a);
      return bxbLight(x, y, z);
    }
  }
}));

// setAnimationLoop instead of a manual requestAnimationFrame chain: WebGPURenderer
// needs an async device/adapter init before the first frame, and setAnimationLoop
// awaits it internally. Calling animate() directly would render before the device
// exists.
renderer.setAnimationLoop(animate);