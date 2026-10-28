import * as THREE from 'three/webgpu';
import { uniform, mix, vec3 } from 'three/tsl';
import { MODELS } from './core/models.js';
import { World, getVoxelKey, createTestArea, createEntity, getColumnTop, CHUNK_SIZE,
         worldMirrorEntries, followGround, ModelBoxes, addWorldModel, WorldModels, mountWorldModel,
         loadLevel, findPath, DEV_LEVEL } from './core/world.js';
import { createEditor } from './tools/editor.js';
import { createObject } from './game/objects.js';
import { initWorldRender, rebuildWorldInstances, VOXEL_SIZE, updateVoxelTints,
         updateVoxelVisibility, worldInstancedMesh, glassInstancedMesh, terrainTextures,
         setShadedAlbedo } from './render/render.js';
import { terrainWhite, terrainTexelTSL } from './gpu/terrain.js';
import { createWanderAI } from './game/ai.js';
import { createGameState } from './game/state.js';
import { createEncounter } from './game/encounter.js';
import { createInteraction } from './game/interaction.js';
import { createInput, INPUT_RULES } from './game/input.js';
import { createMarkers } from './render/markers.js';
import { createPicker } from './render/picking.js';
import { createCameraRig } from './render/camera.js';
import { createEngine, flip } from './render/engine.js';
import { createSky } from './render/features/sky.js';
import { createLights } from './render/features/lights.js';
import { createMirrors } from './render/features/mirrors.js';
import { createGI } from './render/features/gi.js';
import { createFog } from './render/features/fog.js';
import { createSmoke } from './render/features/smoke.js';
import { createBoxLight } from './render/features/boxlight.js';
import { createDebugView } from './render/features/debugview.js';
import { createTexelCache } from './render/features/texelcache.js';
import { createSpriteAtlas } from './render/features/spriteatlas.js';
import { createCards } from './render/features/cards.js';
import { createSun } from './render/features/sun.js';
import { createSunMap } from './render/features/sunmap.js';
import { createShadowSets, nextFrame } from './render/shadowsets.js';
import { createToneMap } from './render/features/tonemap.js';
import { createOcclusion } from './render/features/occlusion.js';
import { getSpriteWorldPos, followPath, walk, WALK_SPEED, ENEMY_SPEED } from './game/movement.js';
import { toggleFieldDebug, refreshFieldDebug, isFieldDebugVisible } from './tools/debug.js';
import { createFieldAt, gridOriginFor, GRID_DIM, marchOccupancy, sphereTrace,
         CASCADE_COUNT, applyHandoff } from './core/field.js';
import { makeClipSamples, updateCharacterClipping, addSpriteTangent, crossedPlanesGeometry } from './render/sprites.js';
import { createPerfOverlay } from './tools/perf.js';
import { createFramePacer, describePacing } from './tools/pacing.js';
import { VERSION, bundleHash, buildLabel } from './core/version.js';
import { createVoxelAONode, AO_DISTANCE } from './gpu/ambient.js';
import { runComputeSmokeTest, runGPUMarch } from './gpu/diagnostics.js';
import { installUploads, installSubmitBatching, toggleUpload, uploadFlags, countGPUCalls } from './gpu/uploads.js';
import { textureLimitsToRequest, noteDeviceLimits, fitFallback, deviceLimits } from './gpu/limits.js';
import { measurePasses } from './tools/passtimes.js';
import { createDistanceTexture, updateDistanceTexture, followShadowGrid, commitShadowGrid,
         cascadeBindings, writeCascadeBindings } from './gpu/field.js';
import { GLASS_MAX_BOUNCES, GLASS_LAYERS, GLASS_DISPERSION, glassCaustics } from './gpu/glass.js';
import { createHitCache } from './gpu/hitcache.js';
import { createNormalTexture } from './gpu/normals.js';
import { roughReach } from './gpu/shade.js';
import { reflectBouncesUniform, REFLECT_MAX_BOUNCES } from './gpu/specular.js';
import { setHitBounds } from './gpu/volume.js';
import { sunForHour, DEFAULT_SKY_HOUR } from './core/sky.js';
import { createTickClock, createWorldClock, createTickSmoothing } from './core/clock.js';
import { lightRadiusMetres } from './core/lights.js';
import { createConsole, installConsole } from './tools/console.js';
import { createSettingsPanel } from './tools/settings.js';
import { PRESETS, PRESET_NAMES, applyPreset, matchPreset } from './tools/presets.js';
import { unstableNames, hashText } from './core/shadowsets.js';
import { classifyTextureAlpha, registerCutout, cutoutsEnabled, setCutoutsEnabled } from './render/cutout.js';
import { averageAlbedo } from './core/lpv.js';
import * as UI from './game/ui.js';

import bobTextureUrl from '../assets/sprites/character_Bob.png'
import evilBobTextureUrl from '../assets/sprites/character_EvilBob.png'
import treeTextureUrl from '../assets/sprites/decor_tree.png'
import chestTextureUrl from '../assets/sprites/container_chest.png'
import chestOpenTextureUrl from '../assets/sprites/container_chest_open.png'

// Mode, walk, battle and arena: what the game modules share (game/state.js).
const game = createGameState();

// The battle buttons (game/encounter.js, made below).
UI.initUI(() => encounter.attack(), () => encounter.endTurn());

// --- SCENE SETUP ---
const scene = new THREE.Scene();
// The background is the sky (sky.js): the same L2 SH the ambient and the
// reflection misses read, at the view direction. Built below, with the sky.
scene.background = new THREE.Color(0x222233);

// WebGPU, not WebGL: the texel-space shading architecture needs compute shaders
// and TSL, neither of which WebGL2 can provide. Everything imports from
// 'three/webgpu' so materials resolve to their NodeMaterial equivalents.
//
// The device is requested here rather than by three, whose own request asks for
// featureLevel "compatibility" - Firefox does not support that yet, hands back a
// core adapter anyway and logs a notice on every load. Same features as three
// would ask for: everything the adapter has. Any failure falls back to letting
// three do it.
// High performance: on a dual-GPU laptop the default is the integrated GPU. A
// friend's RTX 3050 laptop ran the game on its Iris Xe at ~55 fps. Chrome on
// Windows may still pin its GPU process to one adapter, in which case only the
// OS graphics setting for the browser moves it.
async function requestGPUDevice() {
  try {
    const adapter = navigator.gpu &&
      await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (adapter?.info) {
      const { vendor, architecture, description } = adapter.info;
      console.info('[renderer] WebGPU adapter:', vendor, architecture, description || '');
    }
    if (!adapter) return undefined;
    // The adapter's highest texture limits, not WebGPU's guaranteed 16 a stage
    // (gpu/limits.js): what the shading may bind follows what was granted.
    const device = await adapter.requestDevice({ requiredFeatures: [...adapter.features],
                                                 requiredLimits: textureLimitsToRequest(adapter.limits) });
    noteDeviceLimits(device.limits);
    return device;
  } catch (err) {
    console.warn('[renderer] own WebGPU device failed, letting three request one', err);
    return undefined;
  }
}
// trackTimestamp gives the perf graph the GPU's own frame time. It only takes
// effect when the device has timestamp-query; without it the graph says so.
// stencil: the texel cache's miss pass is gated by it - see render/features/texelcache.js.
const renderer = new THREE.WebGPURenderer({ antialias: true, trackTimestamp: true, stencil: true,
                                           device: await requestGPUDevice() });
// Fewer writeBuffer calls for the same bytes, and the frame's command buffers
// in fewer submits (gpu/uploads.js, bxb.uploads).
installUploads(renderer);
installSubmitBatching(renderer.backend.parameters.device);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('app').appendChild(renderer.domElement);
// Which refreshes render: every one, a cap, or uncapped - see pacing.js.
// animate is hoisted; the loop itself starts at the bottom of the file.
const pacer = createFramePacer(renderer, animate);

const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
// ~23 degrees elevation. The old (10,20,10) was ~55 degrees, which put the sun
// nearly overhead and left every shadow hidden under the block casting it.
// A low sun is also what makes voxel shadow blockiness legible.
// Where the clock puts the sun at the starting hour - see bxb.time.
dirLight.position.set(...sunForHour(DEFAULT_SKY_HOUR));
scene.add(dirLight);

// --- WORLD GENERATION ---
// The Dev Level (levels/dev_level.json - the editor writes it) if there is
// one; otherwise the test area built in code, with its test props.
async function fetchLevel(name) {
  try {
    const res = await fetch(`${import.meta.env.BASE_URL}levels/${name}.json`, { cache: 'no-store' });
    return res.ok ? await res.json() : null;
  } catch { return null; }
}
const savedLevel = await fetchLevel(DEV_LEVEL);
if (savedLevel) {
  const missing = loadLevel(savedLevel, MODELS);
  if (missing.length) console.warn(`[level] models not found or not placeable: ${missing.join(', ')}`);
} else {
createTestArea(36, 36);
// Test props: the chair from assets/objects (models.js), standing on the
// ground near the start, in three facings.
const testChair = MODELS.get('decor_chair');
if (testChair) {
  addWorldModel(testChair, 6, 0, 8, 0);
  addWorldModel(testChair, 6, 0, 10, 1);
  addWorldModel(testChair, 4, 0, 9, 2);
}
// The wall torch, hung on walls: on the west face of the grass wall at x = 20
// (the wall east of its cell), and on the marble wall behind the reflection
// bed (south of its cell, z = 17) - both walls are two blocks tall, where the iron floor and marble mirror it.
const testTorch = MODELS.get('decor_wallTorch');
if (testTorch) {
  // One block up (cell y 2): at eye level, not at the foot of the wall.
  mountWorldModel(testTorch, 19, 2, 11, 'E');
  mountWorldModel(testTorch, 5, 2, 16, 'S');
}
// Every other model in assets/objects, in a row along z = 6 from x = 4, so a
// new one shows up by being dropped in the folder (until the editor places them).
{
  let x = 4;
  for (const [name, model] of MODELS) {
    if (name === 'decor_chair' || name === 'decor_wallTorch' || x > 10) continue;
    addWorldModel(model, x, 0, 6, 0);
    x += 2;
  }
}
}
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
// A pixel-art sprite texture of its own. Each character needs one so it can flip
// independently - and a separate LOAD rather than a clone: Texture.clone() flags
// the copy for upload at once, while the shared image is still null until the
// file arrives, and a first frame drawn in that window crashes in three.
// The alpha is classified on load: binary alpha lets a cutout draw opaque
// (js/cutout.js).
function loadSpriteTexture(url) {
  const tex = texLoader.load(url, classifyTextureAlpha);
  tex.magFilter = tex.minFilter = THREE.NearestFilter;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}
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
const characterGeo = addSpriteTangent(new THREE.PlaneGeometry(VOXEL_SIZE, VOXEL_SIZE * 2)); // 2 voxels tall
characterGeo.translate(0, VOXEL_SIZE, 0); 

function createCharacterMesh(texture) {
  // alphaTest discards fully-transparent pixels before the depth test, so this
  // mesh's invisible corners don't still write depth and occlude what's behind it.
  const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, alphaTest: 0.5, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(characterGeo, material);
  mesh.rotation.order = 'YXZ'; // yaw (heading) applied before pitch (lean), so lean tilts around the already-yawed local X axis
  mesh.userData.drawingOnTop = false;
  // Opaque once its alpha is known to be binary - except while clipping, when
  // it has to draw last, in the transparent pass.
  registerCutout(material, () => mesh.userData.drawingOnTop);
  return mesh;
}

// Sample points matching characterGeo's footprint, used to detect the frames
// where the quad is actually penetrating a block. See js/sprites.js.
const characterClipSamples = makeClipSamples(VOXEL_SIZE, VOXEL_SIZE * 2);

const playerTex = loadSpriteTexture(bobTextureUrl);
const playerSprite = createCharacterMesh(playerTex);

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
const enemySprite = createCharacterMesh(loadSpriteTexture(evilBobTextureUrl));
enemySprite.position.copy(getSpriteWorldPos(enemy.gridPos));
scene.add(enemySprite);

const enemyAI = createWanderAI(enemy, enemySprite, 4);

// --- Normal maps ---
//
// Generated for every sprite and block texture (normals.js). A hand-authored
// <texture name>_n.png anywhere under assets/ replaces the generated one - drop
// it in and it is picked up, nothing to register.
const normalOverrides = import.meta.glob('../assets/**/*_n.png',
                                         { eager: true, query: '?url', import: 'default' });
function normalMapFor(name, generate) {
  const hit = Object.keys(normalOverrides).find(k => k.endsWith('/' + name + '_n.png'));
  if (hit) {
    const tex = texLoader.load(normalOverrides[hit]);
    tex.minFilter = tex.magFilter = THREE.NearestFilter;
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    return tex;
  }
  const { data, w, h } = generate();
  return createNormalTexture(data, w, h);
}

// The grass texture's mean colour, linear - what the terrain bounces into the
// LPV until the material system gives each block its own albedo. Layer 0 of
// the terrain arrays (render.js); polled, like the cards, until it has loaded.
let terrainAlbedo = null;
function resolveTerrainAlbedo() {
  if (terrainAlbedo || !terrainTextures.rgba[0]) return;
  terrainAlbedo = averageAlbedo(terrainTextures.rgba[0]);
}

// Sprites are turned around by mirroring their UVs - repeat.x goes to -1, see
// updateSpriteFacing - not by scaling the mesh. So that is where any mirror
// state has to be read from. One definition rather than one per caller: the
// cutout and the torch both need it, and a torch that disagreed with the
// silhouette about which way Bob faces would swap sides a frame early or late.
const mirrored = sprite => sprite.material.map.repeat.x < 0;

// --- WORLD OBJECTS (Phase 10: Tree, Chest, Barrel) ---
// Untextured colored placeholder sprites - no dedicated object art yet.
function createObjectSprite(color) {
  const mat = new THREE.SpriteMaterial({ color, transparent: true });
  const sprite = new THREE.Sprite(mat);
  sprite.center.set(0.5, 0);
  sprite.scale.set(VOXEL_SIZE, VOXEL_SIZE, 1);
  return sprite;
}

const treeTexture = loadSpriteTexture(treeTextureUrl);

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
const TREE_ANGLES = [0, Math.PI / 2, Math.PI / 4, (3 * Math.PI) / 4];
const treePlaneGeo = addSpriteTangent(new THREE.PlaneGeometry(VOXEL_SIZE * 2, VOXEL_SIZE * 3));
treePlaneGeo.translate(0, VOXEL_SIZE * 1.5, 0); // anchor the bottom edge at local origin, like the sprites' center.set(0.5, 0)
// The same four planes as one geometry: one draw a tree instead of four. The
// separate planes are kept for bxb.treemerge(), to A/B the two.
const treeCrossGeo = crossedPlanesGeometry(VOXEL_SIZE * 2, VOXEL_SIZE * 3, TREE_ANGLES);
let treeMerge = true;
const treeMaterial = new THREE.MeshBasicMaterial({
  map: treeTexture, transparent: true, side: THREE.DoubleSide,
  alphaTest: 0.5 // discard fully-transparent pixels before the depth test, so they don't occlude what's behind
});
// Four DoubleSide planes a tree: transparent, each is drawn twice (back faces,
// then front). Opaque once the texture's alpha is known to be binary.
registerCutout(treeMaterial);
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

  // The group is the tree: position, visibility, picking and cards all use it.
  // What draws it is either one merged mesh or the four planes.
  const treeMesh = new THREE.Group();
  const planes = TREE_ANGLES.map(a => {
    const m = new THREE.Mesh(treePlaneGeo, treeMaterial);
    m.rotation.y = a;
    return m;
  });
  treeMesh.userData.drawn = { merged: [new THREE.Mesh(treeCrossGeo, treeMaterial)], planes };
  treeMesh.add(...(treeMerge ? treeMesh.userData.drawn.merged : planes));
  treeMesh.position.copy(getSpriteWorldPos(treeObject.gridPos));
  scene.add(treeMesh);

  const voxel = World.get(getVoxelKey(treeObject.gridPos.x, treeObject.gridPos.y, treeObject.gridPos.z));
  if (voxel) voxel.occupant = treeObject.id;

  worldObjects.push({ data: treeObject, mesh: treeMesh });
  treeMeshes.push(treeMesh);
  return treeObject;
}

// Swaps every tree between the merged mesh and the four planes. All of a
// tree's meshes share one material - plain, or the lit one - so the incoming
// meshes take it over and nothing is rebuilt.
function setTreeMerge(on) {
  treeMerge = !!on;
  for (const tree of treeMeshes) {
    const { merged, planes } = tree.userData.drawn;
    const out = tree.children.slice();
    const into = treeMerge ? merged : planes;
    if (out[0] === into[0]) continue;
    for (const m of into) {
      m.material = out[0].material;
      m.userData.originalMaterial = out[0].userData.originalMaterial;
    }
    tree.remove(...out);
    tree.add(...into);
  }
  // The meshes now drawn need their atlas twins and regions.
  if (sets.active && sets.active.atlasMats) spriteAtlas.ensure(sets.active.atlasMats);
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
// Reflections and the view through glass land only on what battle shows:
// the arena's blocks, as a box in world metres (gpu/volume.js hitBounds). Block b
// spans (b - 0.5) to (b + 0.5) blocks.
function setArenaHitBounds(bounds) {
  if (!bounds) { setHitBounds(null); return; }
  const lo = b => (b - 0.5) * VOXEL_SIZE, hi = b => (b + 0.5) * VOXEL_SIZE;
  setHitBounds({
    min: { x: lo(bounds.minX), y: lo(bounds.minY), z: lo(bounds.minZ) },
    max: { x: hi(bounds.maxX), y: hi(bounds.maxY), z: hi(bounds.maxZ) }
  });
}

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

// The hover border and the path dots (render/markers.js).
const markers = createMarkers(scene);

// --- The camera rig (render/camera.js) ---
const rig = createCameraRig(scene);
const camera = rig.camera;
const pivot = rig.pivot;

const timer = new THREE.Timer();

// --- Game time (core/clock.js) ---
// The game steps in 60 Hz ticks; the frame draws the characters between their
// last two (smoothing). Game time - the hour, a battle's rounds - is the world
// clock. Its day cycle starts stopped (bxb.daycycle): the sky re-projects as the
// hour moves, which the time-of-day work is to make cheap first.
const ticks = createTickClock();
const worldClock = createWorldClock(DEFAULT_SKY_HOUR);
worldClock.running = false;
const smoothing = createTickSmoothing([playerSprite, enemySprite]);

// --- What is under the cursor (render/picking.js) ---
const picker = createPicker(camera);

// --- THE EDITOR (js/editor.js): Tab ---
// An edit changes World / WorldModels; this rebuilds what is made from them.
// The terrain mesh is refilled in place (render.js), the block-material
// volume with it; the mirrors rebuild on their next update; the field is
// re-baked in full, and the worker handed the new world; the model lights are
// re-read. Returns a message for the editor when something needs a reload.
function applyWorldEdit() {
  const r = rebuildWorldInstances();
  if (!r.ok) return { message: `${r.count} pieces is past the terrain mesh's room - save and reload` };
  texel.syncCount();
  mirrors.invalidate();
  farShadows.invalidate();
  lighting.syncModelLights();
  boxLight.rebuild();
  if (shadowsOn) {
    // No prior footprint: a full bake, not a scroll (core/field.js createFieldAt).
    for (const g of shadowGrids) if (g) g.filledBlocks = null;
    ensureShadowGrids();
  }
  if (r.glassNeeded) return { message: 'glass in a world built without it: save and reload to see it traced' };
  return null;
}
const editor = createEditor({
  scene, camera, dom: renderer.domElement,
  onEdit: applyWorldEdit,
  onToggle: on => {
    if (on && game.mode !== 'explore') { console.log('the editor opens in explore mode only'); return false; }
    // Nothing carries over: no walk in progress, no key held down.
    game.path = [];
    game.moveTargetKey = null;
    for (const k of Object.keys(game.keys)) game.keys[k] = false;
    return true;
  }
});

// --- Battle and the interaction panel (game/encounter.js, game/interaction.js) ---
//
// The rules live there; what they look like is this view.
const encounter = createEncounter({
  game, player, enemy, playerSprite, enemySprite, enemyAI, ticks, worldClock,
  view: {
    enterArena(arena, bounds) {
      const centerX = (bounds.minX + bounds.maxX) / 2;
      const centerZ = (bounds.minZ + bounds.maxZ) / 2;
      rig.arenaCenter.set(centerX * VOXEL_SIZE, 0, centerZ * VOXEL_SIZE);
      updateVoxelVisibility(arena, true);
      updateObjectVisibility(arena, true);
      farShadows.invalidate();
      setArenaHitBounds(bounds);
    },
    exitArena() {
      updateVoxelVisibility(null, false);
      updateObjectVisibility(null, false);
      farShadows.invalidate();
      setArenaHitBounds(null);
    },
    clearHover() {
      markers.reset();
      input.forgetHover();
    },
    refreshTints() { updateVoxelTints(null, null, false); }
  }
});
const interaction = createInteraction({
  game, player, encounter,
  view: {
    hasMesh: obj => worldObjects.some(wo => wo.data.id === obj.id),
    // A chest swaps to its open texture; a barrel has no open look yet.
    showOpen(obj, open) {
      const r = worldObjects.find(wo => wo.data.id === obj.id);
      if (!r || obj.model !== 'chest') return;
      r.mesh.material.map = open ? chestOpenTexture : chestTexture;
      r.mesh.material.needsUpdate = true;
    }
  }
});

// --- Pointer and keys (game/input.js) ---
const input = createInput({
  game, player, interaction, encounter, picker, markers,
  sprites: {
    clickable: () => [playerSprite, enemySprite, chestSprite, barrelSprite, ...treeMeshes],
    hoverable: () => [enemySprite, chestSprite, barrelSprite],
    targetOf: sprite => spriteToTarget.get(sprite),
    // Each hoverable back to its own base tint (not a shared white - chest/barrel
    // are untextured color sprites, not tinted photos like enemySprite).
    highlight(lit) {
      for (const [sprite, baseColor] of spriteBaseColor) sprite.material.color.setHex(baseColor);
      if (lit) lit.material.color.setHex(0xffff00);
    }
  }
});

// Whiteworld draws the terrain plain white. One uniform, read by both the plain
// and the shadowed terrain materials (render.js), so toggling it is a write.
let whiteWorld = false;
const whiteUniform = terrainWhite;
function toggleWhiteWorld() {
  whiteWorld = !whiteWorld;
  whiteUniform.value = whiteWorld ? 1 : 0;
  return `whiteworld ${whiteWorld ? 'on' : 'off'}`;
}

// R and the "advanced effects" button. One at a time: a second press while a
// turn-on is compiling is ignored rather than queued.
let effectsBusy = false;
let presetBusy = false;   // a graphics preset is being applied (tools/presets.js)
// By id, not a module const: bxb.shadows() can run before the button's wiring.
function syncEffectsButton() {
  const btn = document.getElementById('btn-effects');
  if (!btn) return;
  btn.textContent = shadowsOn ? 'Disable advanced effects' : 'Enable advanced effects';
  btn.disabled = effectsBusy;
  const sel = document.getElementById('sel-preset');
  if (sel) sel.disabled = effectsBusy || presetBusy;
}
async function toggleAdvancedEffects() {
  if (effectsBusy) return;
  effectsBusy = true;
  syncEffectsButton();
  let notice = null;
  try {
    if (shadowsOn) {
      console.log(await bxbApi.shadows());
      return;
    }
    const out = [];
    // The frame graph is left as it is (it is on from load; hidden, it costs
    // Firefox's WebGPU thread nothing - measure fps that way, bxb.gpu).
    if (!sun.cone) out.push(await bxbApi.soften('cone'));
    if (shadowTurnOnCompiles()) {
      notice = showNotice('Preparing shaders\u2026');
      // Two frames, so the notice is on screen before the compile holds them.
      await nextFrame(); await nextFrame();
    }
    out.push(await bxbApi.shadows());
    await sets.done;
    // The GI's compute kernels compile on their first dispatch, the next frame.
    if (notice) { await nextFrame(); await nextFrame(); }
    console.log(out.join('\n'));
  } finally {
    if (notice) notice.remove();
    effectsBusy = false;
    syncEffectsButton();
  }
}

let settingsPanel = null;   // built after the console, below

// --- Graphics presets (tools/presets.js) ---
//
// A preset is a list of settings-panel values, so it goes through the same
// setters as the panel and the console. The effects switch is the one thing it
// drives directly: off before the values, on after, so a switch compiles once.
const PRESET_KEY = 'bxb.preset';
const settingDefs = new Map();   // key -> setting, filled once the panel is built
async function setEffects(on) {
  if (on === shadowsOn) return;
  await toggleAdvancedEffects();
}
function currentPreset() {
  return effectsBusy || presetBusy ? 'custom' : (matchPreset(settingDefs, shadowsOn) || 'custom');
}
async function setPreset(name) {
  if (presetBusy || effectsBusy) return 'busy - a preset or the effects are still building';
  if (!PRESETS[name]) return `no preset '${name}' - use ${PRESET_NAMES.join(', ')}`;
  presetBusy = true;
  syncEffectsButton();
  try {
    const msg = await applyPreset(name, { defs: settingDefs, effectsOn: () => shadowsOn, setEffects });
    try { localStorage.setItem(PRESET_KEY, name); } catch { /* private window */ }
    return `graphics preset: ${msg}`;
  } finally {
    presetBusy = false;
    syncEffectsButton();
  }
}
const presetSetting = {
  key: 'preset', label: 'Preset', type: 'select',
  options: [...PRESET_NAMES.map(n => ({ value: n, label: `${PRESETS[n].label} (${PRESETS[n].target})` })),
            { value: 'custom', label: 'Custom' }],
  help: PRESET_NAMES.map(n => `${PRESETS[n].label}: ${PRESETS[n].help}`).join(' | '),
  get: () => currentPreset(),
  set: v => setPreset(v)
};

// Where the pointer last was, for keys that act on what is under it (Alt+F).
const lastPointer = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
window.addEventListener('pointermove', (e) => { lastPointer.x = e.clientX; lastPointer.y = e.clientY; });

window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();

  // Alt+V: the debug settings panel - every renderer knob in one place.
  if (e.altKey && e.code === 'KeyV') {
    e.preventDefault();
    if (settingsPanel) settingsPanel.toggle();
    return;
  }
  if (settingsPanel && settingsPanel.isOpen) {
    if (key === 'escape') { settingsPanel.close(); return; }
    // Arrow keys on a focused slider, typing in a select - none of it is a
    // game key.
    if (settingsPanel.el.contains(e.target)) return;
  }

  // Alt+X: whiteworld - terrain drawn plain white so light and shadow can be
  // judged without the grass texture in the way. Sprites keep their textures.
  // Alt keys match e.code, not e.key: holding Alt changes the reported character
  // on several layouts. preventDefault keeps Alt from the browser's own menu.
  if (e.altKey && e.code === 'KeyX') {
    e.preventDefault();
    console.log(toggleWhiteWorld());
    return;
  }
  if (debugView.key(e)) return;   // Alt+B: cycle the render debug views

  // Alt+<key> is reserved for debug modes.
  if (e.altKey && e.code === 'KeyC') {
    e.preventDefault();
    rig.freeCam = !rig.freeCam;
    console.log(rig.freeCam ? 'free cam on - WASD flies the camera, Bob stays put'
                        : 'free cam off - camera returns to Bob');
    return;
  }

  // Alt+U: the whole HUD on and off - title, party, map, chat, battle panels.
  // Starts hidden (the class is on body in index.html) while graphics work is
  // the focus.
  if (e.altKey && e.code === 'KeyU') {
    e.preventDefault();
    const hidden = document.body.classList.toggle('ui-hidden');
    console.log(`UI ${hidden ? 'hidden' : 'shown'}`);
    return;
  }

  // Alt+L: drop a lamp where Bob's torch would be. Alt+Shift+L clears them all.
  if (e.altKey && e.code === 'KeyL') {
    e.preventDefault();
    if (e.shiftKey) {
      lighting.clearLamps();
      console.log('lamps cleared');
    } else {
      console.log(lighting.dropLamp());
    }
    return;
  }

  // Alt+F: a Fog Cloud on the block under the pointer. Alt+Shift+F clears them.
  if (e.altKey && e.code === 'KeyF') {
    e.preventDefault();
    if (e.shiftKey) { console.log(smoke.clear()); return; }
    const tile = picker.tileAt(lastPointer.x, lastPointer.y);
    console.log(tile ? smoke.summonAt(tile) : 'no block under the pointer');
    return;
  }

  // Debug: Alt+G toggles the field occupancy overlay.
  if (e.altKey && e.code === 'KeyG') {
    e.preventDefault();
    const msg = toggleFieldDebug(scene, player.gridPos, worldInstancedMesh,
                                   game.mode, { sunDirection: dirLight.position,
                                                  bias: sun.gridBias });
    console.log(msg || '[field debug] off');
    return;
  }

  // The game's keys: walking, Escape, B (game/input.js).
  input.keyDown(key);

  // R: every advanced effect on or off. On: the frame graph, then the
  // cone-traced sun, then the marched shadows (and with them AO cones, GI,
  // reflections, mirrors), in that order, so the shadows build once with the
  // cone already chosen. Off: shadows off - the plain materials and static
  // voxel AO; the frame graph stays up. The first turn-on compiles, behind a
  // "Preparing shaders" notice; after that the set is kept, so both directions
  // are a swap. The button beside Debug does the same.
  if (key === 'r' && !e.altKey && !e.ctrlKey && !e.metaKey) {
    toggleAdvancedEffects();
    return;
  }

  // T: the torch. A held light is the thing the marched field is for - it moves
  // every frame and costs no re-bake, which a shadow map could not have done.
  if (key === 't') {
    console.log(lighting.toggleTorch());
    return;
  }

  if (key === 'q') rig.rotationStep += 1;
  if (key === 'e') rig.rotationStep -= 1;

});

window.addEventListener('keyup', (e) => input.keyUp(e.key.toLowerCase()));

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- GAME TICK ---
// One 1/60 s step of the game: AI and movement. dt is always TICK_SECONDS. Reads
// the camera's heading as the last frame left it.
function gameTick(dt) {
  if (game.mode === 'explore' && !editor.active) {
    enemyAI.update(dt, rig.heading);
  }

  const isWalkingManual = (game.keys.w || game.keys.a || game.keys.s || game.keys.d);

  if (isWalkingManual && !rig.freeCam && game.mode === 'explore' && game.path.length > 0) {
    game.path = [];
    game.moveTargetKey = null;
  }

  if (game.path.length > 0) {
    // In explore the occupied column follows Bob as he walks; in battle only
    // each arrival moves him, and each costs a step of movement.
    const arrived = followPath(player, playerSprite, game.path, WALK_SPEED, dt, rig.heading,
                               game.mode === 'explore');
    if (arrived && game.mode === 'battle') {
      player.turnResources.moveRemaining -= 1;
      markers.path(game.path);
      if (game.path.length === 0) encounter.refreshReachableTiles();
    }
  }
  // The enemy's battle move.
  if (game.enemyPath.length > 0) {
    if (followPath(enemy, enemySprite, game.enemyPath, ENEMY_SPEED, dt, rig.heading)) {
      enemy.turnResources.moveRemaining -= 1;
      // Trigger the attack check now that movement is finished
      if (game.enemyPath.length === 0) encounter.processEnemyAI();
    }
  } else if (game.mode === 'explore' && INPUT_RULES.explore.keyboard && !rig.freeCam) {
    walk(player, playerSprite, game.keys,
         rig.settledHeading('explore'), dt);
  }
  // Characters move across the ground; their height is the ground under
  // them, followed exactly up a slope and eased over a half-block ledge
  // (world.js followGround).
  followGround(playerSprite, player.gridPos, dt);
  followGround(enemySprite, enemy.gridPos, dt);
}
// In order each tick: the characters' starting points for the smoothing, game
// time, then the game.
ticks.every(1, () => smoothing.capture());
ticks.every(1, () => worldClock.tick());
ticks.every(1, gameTick);

// --- RENDER & GAME LOOP ---
function animate(time) {
  const tStart = performance.now();
  timer.update(time);
  const realDt = timer.getDelta();
  // The per-frame eases (camera, markers) are clamped: after a stall (a tab
  // switch, a compile) one huge step would snap every ease to its target. The
  // ticks cap themselves the same way.
  const dt = Math.min(realDt, 0.1);

  // The newest pointer move since the last frame (game/input.js).
  input.flush();

  // The click pop; once it ends the hover is looked at afresh.
  if (markers.update(dt)) input.forgetHover();

  // 1. The game's ticks for the time that passed; the characters are then
  // drawn between their last two tick positions.
  smoothing.restore();
  ticks.advance(realDt);
  smoothing.show(ticks.alpha);

  // The panel closes once Bob walks out of look-range of its target;
  // otherwise its action buttons' enabled state stays live.
  interaction.update();

  // 2. The camera: eased toward the mode's framing, around Bob or the arena.
  rig.update(dt, game.mode, playerSprite.position, game.keys);
  editor.update(dt);

  // Clamped billboard: characters turn to face the camera's heading, plus lean
  // back toward its pitch up to MAX_CHARACTER_TILT - enough to avoid looking
  // flat/squished, not enough to swing into adjacent geometry like the tree.
  const characterTilt = Math.min(rig.pitch, MAX_CHARACTER_TILT);
  playerSprite.rotation.set(-characterTilt, rig.heading, 0);
  enemySprite.rotation.set(-characterTilt, rig.heading, 0);

  // Checked after the rotation is applied, since whether the quad penetrates
  // geometry depends on this frame's yaw and lean.
  updateCharacterClipping(playerSprite, characterClipSamples);
  updateCharacterClipping(enemySprite, characterClipSamples);

  // Rebuilds the overlay if the player has walked into a different chunk.
  // No-op when the overlay is off or the chunk is unchanged.
  refreshFieldDebug(scene, player.gridPos, game.mode,
                      { sunDirection: dirLight.position, bias: sun.gridBias });

  // The cascades follow the player so shading works wherever they are, not only
  // in the patch the grid happened to be built in. Nothing else has to happen
  // when they move: the terrain material marches the field live, so a re-origin
  // is a texture upload and the next frame is already correct.
  const tGrid = performance.now();
  const moved = shadowsOn && followShadowGrids();
  const tCards = performance.now();
  // The sun's cards. The point lights' are packed in lighting.update below, after
  // the torch has moved, so they face where the flame IS this frame.
  // Always: with shadows off the sprites' contact AO still reads the sun's cards.
  resolveTerrainAlbedo();
  cards.update();
  farShadows.update();
  sky.update(ticks.alpha);
  // AFTER the re-origin, never before. A scroll bakes the strip that slid in,
  // which wipes the imprint of any proxy standing there, and it translates the
  // remembered boxes into the new coordinates - so running this first would
  // write proxies the scroll then destroys, and cost the bake twice.
  if (shadowsOn) cards.updateOccluders();
  lighting.moveTorch();
  if (shadowsOn || boxLight.active()) lighting.update();
  // After the light list: the mirrors are culled against where the lights are now.
  mirrors.update();
  // The mirror cache and the LPV volumes: sent with split shading (collectCompute).
  const earlyKernels = collectCompute(() => { runMirrorCache(); if (shadowsOn) gi.update(); });

  const tRender = performance.now();
  occlusion.update();   // the interiors' lifted blocks, before the first terrain pass
  smoke.update();   // cube bounds and light before reflections / glass read them
  texel.render(earlyKernels);
  spriteAtlas.render();
  tonemap.update();   // meters this frame's terrain, before the main render
  fog.update();   // after the texel cache drew this frame's G-buffer
  boxLight.update();
  renderer.render(scene, camera);
  resolveGPUTime();
  const tEnd = performance.now();
  logSpike(tGrid, tCards, tRender, tEnd, moved);
  perf.update(dt, { logic: tGrid - tStart, grids: tCards - tGrid,
                    cards: tRender - tCards, submit: tEnd - tRender }, pacer.mode === 'uncapped');
}

// One resolve in flight at a time: each reads every query since the last and
// returns the newest frame's GPU time; overlapping them would split frames.
// Resolved every frame while timing is on - unresolved queries fill three's
// query pool.
let gpuTimePending = false;
// GPU timing (timestamp queries) runs only while something reads it: the
// frame graph, or a bxb measurement holding it. Each frame it costs ~2 extra
// submits and 2 mapAsync readbacks (9.0 -> 7.25 submits, 1.9 -> 0.1 readbacks,
// Chrome 2026-10-26) - in Firefox each a trip through its one WebGPU thread.
let gpuTimingHolds = 0;
function syncGPUTiming() {
  const b = renderer.backend;
  const supported = !b.device || b.device.features.has('timestamp-query');
  b.trackTimestamp = supported && (perf.visible || gpuTimingHolds > 0);
}
// Runs fn with GPU timing on; wasOff tells it the window holds nothing yet.
async function withGPUTiming(fn) {
  const wasOff = !renderer.backend.trackTimestamp;
  gpuTimingHolds++;
  syncGPUTiming();
  try { return await fn(wasOff); } finally { gpuTimingHolds--; syncGPUTiming(); }
}
function resolveGPUTime() {
  if (gpuTimePending || !renderer.backend.trackTimestamp) return;
  gpuTimePending = true;
  // Render and compute (the LPV) are separate query pools; both resolved, so
  // compute queries do not pile up either.
  Promise.all([renderer.resolveTimestampsAsync('render'),
               renderer.resolveTimestampsAsync('compute')]).then(([ms, cms]) => {
    gpuTimePending = false;
    if (typeof ms === 'number') perf.gpu(ms);
    if (typeof cms === 'number') perf.gpuCompute(cms);
  }, () => { gpuTimePending = false; });
}

// CPU-side breakdown of any slow frame, so a hitch can be pinned to a stage
// instead of guessed at. render covers submission, which is where texture
// uploads and pipeline compiles land. GPU execution time is not in here.
const SPIKE_MS = 8;
function logSpike(tGrid, tCards, tRender, tEnd, moved) {
  const total = tEnd - tGrid;
  if (!spikeLog || total < SPIKE_MS) return;
  const f = x => x.toFixed(1);
  console.log(`[spike] ${f(total)} ms - grids ${f(tCards - tGrid)}` +
              `${moved ? ' (re-origin)' : ''}, cards+occluders ${f(tRender - tCards)}, ` +
              `render ${f(tEnd - tRender)}`);
}
let spikeLog = false;

// --- DEBUG CONSOLE (bxb) ---
let shadowsOn = false;
// What the field is centred on: the player, or while editing, the ground the
// editor camera looks at - so the shadows are fine where the edits are.
function fieldFocus() {
  return editor && editor.active ? editor.focusGridPos() : player.gridPos;
}
// One entry per cascade. C0 is the fine 18 m field the texel lock is aligned to;
// C1 is half resolution over 36 m and exists so a long shadow's ray still finds
// its caster after C0 has run out. See core/field.js for the level table.
let shadowGrids = [];
let shadowTexes = [];
let shadowCascades = [];   // the GPU bindings for each level
let shadowOrigins = [];     // block origin each level is currently built at

// Builds every cascade at its own footprint, reusing the buffers if they exist.
// Called on turn-on and whenever a level's origin moves.
function ensureShadowGrids() {
  for (let l = 0; l < CASCADE_COUNT; l++) {
    const o = gridOriginFor(game.mode, fieldFocus(), dirLight.position, sun.gridBias, l);
    shadowGrids[l] = createFieldAt(o.x, o.z, shadowGrids[l] || null, l, o.y);
    if (!shadowTexes[l]) shadowTexes[l] = createDistanceTexture(shadowGrids[l]);
    updateDistanceTexture(shadowTexes[l]);
    shadowCascades[l] = shadowCascades[l]
      ? writeCascadeBindings(shadowCascades[l], shadowGrids[l])
      : cascadeBindings(shadowTexes[l], shadowGrids[l]);
    shadowOrigins[l] = o;
  }
  resetFieldWorker();
  return shadowCascades;
}

// --- Re-origin off the main thread ---
//
// The worker mirrors every level and does the scroll-and-strip bake; the strips
// come back and applyHandoff lands origin, ring and data together. A level with
// a bake in flight is not re-asked until it lands, and a full bake here bumps
// the generation so any answer already on its way is dropped - the worker has
// been reset behind it.
let fieldWorker = null;
let fieldGen = 0;
const fieldInFlight = [];
let fieldApplied = false;   // for the spike log: a hand-off landed this frame

function resetFieldWorker() {
  if (!fieldWorker) {
    try {
      fieldWorker = new Worker(new URL('./render/fieldWorker.js', import.meta.url), { type: 'module' });
    } catch (err) {
      console.warn('[shadows] no field worker, re-origins stay on the main thread', err);
      fieldWorker = false;
      return;
    }
    fieldWorker.onmessage = ({ data: m }) => {
      fieldInFlight[m.level] = false;
      if (m.gen !== fieldGen || !shadowsOn) return;
      applyHandoff(shadowGrids[m.level], m);
      commitShadowGrid(shadowGrids[m.level], shadowTexes[m.level],
                       shadowCascades[m.level], renderer);
      shadowOrigins[m.level] = { x: m.x, y: m.y, z: m.z };
      fieldApplied = true;
    };
  }
  if (!fieldWorker) return;
  // Every reset, not once. World is static today, but a mirror baking from
  // stale blocks would hand back strips of terrain that is not there, so any
  // future terrain edit only has to trigger a reset to reach it.
  fieldWorker.postMessage({ type: 'world', blocks: worldMirrorEntries(World), models: ModelBoxes });
  fieldGen++;
  fieldInFlight.length = 0;
  for (let l = 0; l < CASCADE_COUNT; l++) {
    const o = shadowOrigins[l];
    fieldWorker.postMessage({ type: 'reset', level: l, x: o.x, y: o.y, z: o.z });
  }
}

// Re-origins only the levels that actually moved. C0 moves every block stepped;
// C1 snaps to a 2-block stride, so it rebuilds half as often - which is most of
// why a second level is affordable at all.
function followShadowGrids() {
  let moved = fieldApplied;
  fieldApplied = false;
  for (let l = 0; l < CASCADE_COUNT; l++) {
    const want = gridOriginFor(game.mode, fieldFocus(), dirLight.position, sun.gridBias, l);
    const at = shadowOrigins[l];
    if (at && want.x === at.x && want.y === at.y && want.z === at.z) continue;
    if (fieldWorker) {
      if (fieldInFlight[l]) continue;
      fieldInFlight[l] = true;
      fieldWorker.postMessage({ type: 'scroll', level: l, x: want.x, y: want.y, z: want.z, gen: fieldGen });
      continue;
    }
    followShadowGrid(shadowGrids[l], shadowTexes[l], shadowCascades[l], want, renderer);
    shadowOrigins[l] = want;
    moved = true;
    // One level per frame: C1 and C2 step on the same frames C0 does, and
    // stacking all three bakes into one frame is the spike. A coarse level a
    // frame late is still a correct field - its bindings move with it.
    break;
  }
  return moved;
}

// Cone-traced AO. On/off is compiled into the kernel; the distance is a
// uniform.
// Always on, by two methods swapped with shadows: traced cones through the
// field with shadows on, static voxel AO with them off (see enableAlwaysOnAO).
// Not a switch: off only inside bxb.gpuprofile, to price the cones.
let aoOn = true;
const aoDistance = uniform(AO_DISTANCE);
// Specular (gpu/specular.js): GGX highlights from every light, and one
// reflection ray per texel on smooth surfaces. Both compiled in. The sky is
// what a ray that leaves the field sees - the background, as a uniform.
let specularOn = true;
let reflectionsOn = true;
// --- The render features (render/engine.js, render/features/) ---
const engine = createEngine();
const sun = engine.add(createSun({
  sunLight: dirLight, shadowsOn: () => shadowsOn, rebuild: () => buildPerPixelShadows(),
  reorigin: () => { shadowOrigins = []; }
}));
// The sun depth map: the sun's shadow past its march (core/sunmap.js).
const farShadows = engine.add(createSunMap({ renderer, sunLight: dirLight, shadowsOn: () => shadowsOn }));
// Before the GI: it is handed their card types.
const cards = engine.add(createCards({
  renderer, camera, pivot, sunLight: dirLight, sunAngular: () => sun.angular,
  sprites: { player: { mesh: playerSprite, texture: bobTexture },
             enemy: { mesh: enemySprite, texture: evilBobTexture },
             trees: () => treeMeshes, treeTexture, mirrored },
  normalMap: normalMapFor,
  grids: () => shadowGrids, textures: () => shadowTexes, shadowsOn: () => shadowsOn,
  rebuild: () => buildPerPixelShadows(), onReady: () => enableAlwaysOnAO()
}));
const sky = engine.add(createSky({ scene, sunLight: dirLight, clock: worldClock,
                                   moveSun: (x, y, z) => sun.move(x, y, z) }));
const lighting = engine.add(createLights({
  // For its messages: whether a light is seen - the box light shows lamps too
  // (boxLight is made below; this runs only on a key or a command).
  scene, shadowsOn: () => shadowsOn || boxLight.active(), mirrorRects: () => mirrors.rects,
  holder: { sprite: playerSprite, facingRight: () => mirrored(playerSprite) },
  // A light's sprite cards, in its slice of the shared point-light buffer.
  cards: {
    active: cards.lightSlices.active,
    pack: (l, used, room) => cards.lightSlices.pack(l.position, lightRadiusMetres(l.level), used, room),
    commit: cards.lightSlices.commit
  }
}));
const mirrors = engine.add(createMirrors({
  sunLight: dirLight, sunOn: () => sun.on, lighting, focus: playerSprite.position,
  footprint: () => shadowGrids[Math.min(1, shadowGrids.length - 1)],
  rebuild: () => { if (shadowsOn) buildPerPixelShadows(); }
}));
const gi = engine.add(createGI({
  renderer, cascades: () => shadowCascades, lighting, sky, mirrors,
  sun: { uniforms: () => sun.uniforms(), on: () => sun.on, angular: () => sun.angular, light: dirLight },
  sprites: { casters: () => cards.casters(), types: cards.types },
  terrainAlbedo: () => terrainAlbedo,
  shadowsOn: () => shadowsOn,
  rebuild: () => { if (shadowsOn) buildPerPixelShadows(); },
  fogOn: () => fog.on()
}));
const fog = engine.add(createFog({
  renderer, scene, camera, texel: { gbuffer: () => texel.gbuffer() }, gi, sky, shadowsOn: () => shadowsOn,
  sun: { light: dirLight, on: () => sun.on, uniforms: () => sun.uniforms() }
}));
const smoke = engine.add(createSmoke({
  renderer, scene, camera, cascades: () => shadowCascades,
  lighting, sky, cards, shadowsOn: () => shadowsOn, rebuild: () => { if (shadowsOn) buildPerPixelShadows(); },
  boxLight: () => boxLight,
  sun: { light: dirLight, on: () => sun.on, angular: () => sun.angular, uniforms: () => sun.uniforms() }
}));
// Basic mode's light: sun shadows and lamps on the BoxGrid, with the advanced
// effects off.
const boxLight = engine.add(createBoxLight({
  renderer, shadowsOn: () => shadowsOn, lighting, cards, sunLight: dirLight, ambientLight, smoke
}));
const debugView = engine.add(createDebugView({
  shadowsOn: () => shadowsOn, shadows: () => bxbApi.shadows(), rebuild: () => buildPerPixelShadows()
}));
const spriteAtlas = engine.add(createSpriteAtlas({
  renderer, scene, camera, meshes: () => spriteMeshes(), shadowsOn: () => shadowsOn,
  activeSet: () => sets.active, rebuild: () => buildPerPixelShadows()
}));
const texel = engine.add(createTexelCache({
  renderer, scene, camera, rig, shadowsOn: () => shadowsOn, activeSet: () => sets.active, gi,
  rebuild: () => buildPerPixelShadows()
}));
// The shader sets: built from every feature's options, kept compiled.
const tonemap = engine.add(createToneMap({ renderer, effectsOn: () => shadowsOn, meterSource: () => texel.meterSource() }));
const occlusion = engine.add(createOcclusion({ camera, picker, feet: () => playerSprite.position }));
const sets = createShadowSets({
  renderer, scene, camera, options: () => shadowNodeOptions(), texel, spriteAtlas, sun,
  sprites: { meshes: () => spriteMeshes(), clipping: () => [playerSprite, enemySprite],
             normalFor: m => cards.normalFor(m) }
});

// Internal reflections a ray through glass may take before it counts as trapped.
let glassBounces = GLASS_MAX_BOUNCES;
// Panes a view ray refracts through (2: one more glass seen through the first).
let glassLayers = GLASS_LAYERS;
// Dispersion: off (0) by default - three traces instead of one. Debug.
let glassDispersion = 0;
// Shadow rays tint through glass. Off only inside bxb.gpuprofile, to price it.
let glassShadows = true;
// Kernels that several features dispatch back to back, collected rather than
// sent: texel.render sends them at the head of split shading's own list, as
// ONE renderer.compute - one pass and one submit for the mirror cache, the LPV
// volumes and the split kernels, where they were three (a submit costs
// Firefox's WebGPU thread ~65 us whatever it holds - profile 2026-10-26). The
// order holds within the pass, so each dispatch still sees the ones before it.
// Nothing drawn between reads them on the GPU (the low-res and id passes only
// write the G-buffer); without split shading the low-res pass shades and reads
// them, so texel.render sends them first, alone. Only for features that set
// every uniform before dispatching and read nothing back (the tone-map meter
// reads back, so it stays out). bxb.computebatch() off: each sent as it comes.
let computeBatchOn = true;
function collectCompute(fn) {
  if (!computeBatchOn) { fn(); return []; }
  const real = renderer.compute, list = [];
  renderer.compute = k => { for (const x of [k].flat()) list.push(x); };
  try { fn(); } finally { renderer.compute = real; }
  return list;
}
// bxb.gpuprofile: how long each change is left to settle before it is measured.
const GPUPROFILE_SETTLE_MS = 1000;
// The cache pass, after the mirrors and lights are packed and before anything
// that reads it is drawn. Only for a set built to read it.
function runMirrorCache() {
  const k = shadowsOn && sets.active && sets.active.mirrorCache;
  // The hit cache's halves swap, and the one written this frame is cleared -
  // in the same submit as the mirror cache.
  const list = [];
  if (shadowsOn && hitCache && hitCacheOn) {
    hitCache.writeHalf.value = 1 - hitCache.writeHalf.value;
    list.push(hitCache.clear);
  }
  if (k && mirrors.cacheTexels) {
    k.count = mirrors.cacheTexels;
    list.push(k);
  }
  if (list.length) renderer.compute(list.length === 1 ? list[0] : list);
}

// Profiling only (bxb.gpuprofile): the model lookup in reflection hits.
let modelHitsOn = true;
// The hit lighting cache (gpu/hitcache.js createHitCache): reflections reuse what the
// terrain lit on screen last frame. A rebuild to switch (bxb.hitcache()).
// Off by default: on the glass bed it saved ~0 (the low-res pass loses early
// depth tests to the writes) and the one-frame lag shimmered in motion.
let hitCacheOn = false;
let hitCache = null;
function ensureHitCache() {
  if (!hitCache) hitCache = createHitCache();
  return hitCache;
}
// Every sprite mesh drawn now.
function spriteMeshes() {
  return [playerSprite, enemySprite, ...treeMeshes.flatMap(g => g.children)];
}
// The shading build's options: the features' slices, and the settings main.js
// still owns (render/shadowsets.js builds and keeps the sets).
let fallbackNoted = '';
function shadowNodeOptions() {
  const all = {
    cascades: shadowCascades,
    terrain: terrainTextures,
    // At the surface's texel - on a slope's plane, its own texture (render.js).
    albedo: (st, surf = null) => mix(terrainTexelTSL(terrainTextures.albedo, st, surf).rgb,
                                     vec3(1, 1, 1), whiteUniform),
    specular: specularOn, reflections: reflectionsOn,
    ...engine.options(),
    ao: aoOn, aoDistanceUniform: aoDistance,
    glassBounces, glassLayers, glassDispersion, glassShadows,
    modelHits: modelHitsOn,
    hitCache: hitCacheOn ? ensureHitCache() : null,
    // The world has glass: shadow rays tint through it (gpu/trace.js createConeTraceSunTSL).
    glass: !!glassInstancedMesh
  };
  // Split shading off: the one-shader fallback gives up what the device's
  // texture limit cannot hold (gpu/limits.js) - said once per change.
  const { opts, dropped } = fitFallback(all);
  const note = dropped.join(', ');
  if (note && note !== fallbackNoted) {
    console.info(`[limits] split shading off: left out ${note} - the device allows ` +
                 `${deviceLimits.sampledTextures} sampled textures a stage`);
  }
  fallbackNoted = note;
  return opts;
}

// --- "Preparing shaders" ---
//
// Turning the effects on compiles two big pipelines (and the GI's compute
// kernels on their first dispatch), and a compile blocks frames in both
// browsers - nothing makes it background work (doc §0). So when a turn-on has
// to compile, it says so first: the notice goes up, two frames pass so it is
// actually on screen, and it comes down once the set is compiled and showing.
// A turn-on that finds its set kept (shadowsets.js) is a swap, and shows none.
function showNotice(text) {
  const notice = document.createElement('div');
  notice.textContent = text;
  Object.assign(notice.style, {
    position: 'fixed', left: '50%', top: '50%', transform: 'translate(-50%, -50%)',
    zIndex: '10000', padding: '10px 18px', borderRadius: '4px',
    background: 'rgba(0,0,0,0.75)', color: '#e5e7eb',
    font: '14px monospace', pointerEvents: 'none'
  });
  document.body.appendChild(notice);
  return notice;
}

// Whether turning shadows on now would compile: no kept, compiled set for the
// settings as they stand. Before the first turn-on there are no cascades to
// key on, and that first one always compiles.
function shadowTurnOnCompiles() {
  return !shadowGrids[0] || sets.wouldCompile();
}

function buildPerPixelShadows() {
  // Turning shadows on builds the field fresh - it did not follow the player
  // while they were off. A rebuild while they are on does not need it baked
  // again.
  if (!shadowsOn || !shadowGrids[0]) ensureShadowGrids();
  const origin = shadowOrigins[0];
  sun.ensureUniforms();
  sets.build();
  shadowsOn = true;
  setShadedAlbedo(true);   // the flatter _a textures (render.js)
  syncEffectsButton();
  lighting.moveTorch();
  lighting.update();
  return `shadows on - ${sun.cone ? 'one cone trace' : sun.rays + ' sun ray' +
           (sun.rays === 1 ? '' : 's')} per texel, ` +
         `12m march cap, grid covering blocks ` +
         `${origin.x}..${origin.x + CHUNK_SIZE - 1} x ${origin.z}..${origin.z + CHUNK_SIZE - 1}` +
         (sun.rays === 1 ? '. bxb.soften() for a penumbra.' : '');
}

const perf = createPerfOverlay();
perf.attachRenderer(renderer);
// On from the start; bxb.perf() still hides it.
perf.toggle();
const bxbApi = installConsole(createConsole({
  compute: {
    help: 'run the WebGPU compute smoke test',
    run: () => runComputeSmokeTest(renderer)
  },
  parity: {
    help: 'march the same rays on CPU and GPU and diff them',
    run: async (n = 256) => {
      const origin = gridOriginFor(game.mode, player.gridPos, dirLight.position, sun.gridBias);
      const grid = createFieldAt(origin.x, origin.z, null, 0, origin.y);
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
    help: 'toggle marched shadows on the terrain (one light, locked per texel)',
    run: () => {
      if (shadowsOn) {
        sets.remove();   // drops any build still compiling
        shadowsOn = false;
        setShadedAlbedo(false);
        syncEffectsButton();
        // The field stops following; turning shadows back on rebuilds it.
        // The plain material's AO is the static voxel kind, which needs none.
        return 'shadows off (static voxel AO)';
      }
      return buildPerPixelShadows();
    }
  },
  proxies: cards.commands.proxies,
  cardparity: cards.commands.cardparity,
  cards: cards.commands.cards,
  preset: {
    help: 'graphics preset: ' + PRESET_NAMES.join(', ') + ' - what each sets is in tools/presets.js',
    usage: "bxb.preset('low')  |  bxb.preset()  which one is on",
    run: (name = null) => (name === null ? `graphics preset: ${currentPreset()}` : setPreset(name))
  },
  setting: {
    help: 'read or set any Alt+V setting by its key (the keys tools/presets.js uses); no key lists them',
    usage: "bxb.setting()  |  bxb.setting('gires')  |  await bxb.setting('gires', 1.5)",
    run: (key = null, value) => {
      if (key === null) return [...settingDefs.keys()].join(', ');
      const def = settingDefs.get(key);
      if (!def || !def.get) return `no setting '${key}'`;
      if (value === undefined) return def.get();
      return Promise.resolve(def.set(value)).then(() => `${key}: ${def.get()}`);
    }
  },
  spikes: {
    help: 'log a CPU breakdown of every frame over 8 ms to the console',
    run: () => {
      spikeLog = !spikeLog;
      return `spike log ${spikeLog ? 'on' : 'off'}`;
    }
  },
  torchside: lighting.commands.torchside,
  cardface: cards.commands.cardface,
  soften: sun.commands.soften,
  farshadow: farShadows.commands.farshadow,
  sunmap: farShadows.commands.sunmap,
  ao: {
    help: "cone-traced ambient occlusion - always on. Reach in metres (bxb.view('ao') shows it alone)",
    usage: 'bxb.ao(1.5)',
    run: (a = null) => {
      if (a !== null) aoDistance.value = Math.max(0.1, Number(a));
      return `AO always on - 6 cones, ${aoDistance.value.toFixed(2)} m`;
    }
  },
  texelcache: texel.commands.texelcache,
  spriteatlas: spriteAtlas.commands.spriteatlas,
  shadowcache: sets.commands.shadowcache,
  treemerge: {
    help: "each tree's four planes as one mesh (one draw); off draws the four planes",
    run: () => {
      setTreeMerge(!treeMerge);
      return `trees ${treeMerge ? 'merged: one draw each' : 'four planes each'}`;
    }
  },
  cutouts: {
    help: 'binary-alpha sprites (trees, characters) drawn opaque: same pixels, half the draws. Off: all transparent',
    run: () => {
      setCutoutsEnabled(!cutoutsEnabled());
      // The lit variants take their flag at build time.
      if (shadowsOn) buildPerPixelShadows();
      return `cutout sprites ${cutoutsEnabled() ? 'opaque (binary alpha only)' : 'transparent, as before'}`;
    }
  },
  glasslayers: {
    help: 'panes a view ray refracts through: 1 sees straight through the second, 2 (default) bends through it too',
    usage: 'bxb.glasslayers(2)',
    run: (n = GLASS_LAYERS) => {
      glassLayers = Math.max(1, Math.min(4, Math.round(n)));
      if (shadowsOn) buildPerPixelShadows();
      return `glass layers: ${glassLayers}`;
    }
  },
  dispersion: {
    help: 'glass dispersion (debug): red and blue refract at their own index - three traces instead of one',
    usage: 'bxb.dispersion()  toggles  |  bxb.dispersion(0.02)  sets the spread',
    run: (k = null) => {
      glassDispersion = k === null ? (glassDispersion > 0 ? 0 : GLASS_DISPERSION) : Math.max(0, +k || 0);
      if (shadowsOn) buildPerPixelShadows();
      return glassDispersion > 0 ? `dispersion on, spread ${glassDispersion}` : 'dispersion off';
    }
  },
  caustics: {
    help: 'strength of the caustics under glass (from its normal map); 0 is off. No rebuild',
    usage: 'bxb.caustics(1)  |  bxb.caustics(0)',
    run: (k = null) => {
      if (k !== null) glassCaustics.value = Math.max(0, +k || 0);
      return `caustics ${glassCaustics.value > 0 ? 'strength ' + glassCaustics.value : 'off'}`;
    }
  },
  glassglare: mirrors.commands.glassglare,
  view: debugView.commands.view,
  mirrorbounce: gi.commands.mirrorbounce,
  mirrorcache: mirrors.commands.mirrorcache,
  gibatch: gi.commands.gibatch,
  gifar: gi.commands.gifar,
  lpvhalf: gi.commands.lpvhalf,
  mirrors: mirrors.commands.mirrors,
  spec: {
    help: 'specular: GGX highlights and reflections. bxb.spec() toggles both, ' +
          "bxb.spec('reflect') just the reflection ray",
    run: (a = null) => {
      if (a === 'reflect') reflectionsOn = !reflectionsOn;
      else specularOn = a === null ? !specularOn : !!a;
      if (!shadowsOn) return 'run bxb.shadows() first';
      buildPerPixelShadows();
      return `specular ${specularOn ? 'on' : 'off'}, reflections ` +
             `${specularOn && reflectionsOn ? 'on' : 'off'}`;
    }
  },
  ambient: sun.commands.ambient,
  daylight: sun.commands.daylight,
  torch: lighting.commands.torch,
  lamp: lighting.commands.lamp,
  lamps: lighting.commands.lamps,
  gi: gi.commands.gi,
  shadowbudget: lighting.commands.shadowbudget,
  cardbudget: lighting.commands.cardbudget,
  roughreach: {
    help: 'rough reflections: shrink their reach through the blur band (on) or keep full reach (off)',
    usage: 'bxb.roughreach()',
    run: () => {
      roughReach.value = roughReach.value > 0 ? 0 : 0.75;
      return `rough reflections ${roughReach.value > 0 ? 'reach less as they blur' : 'at full reach'}`;
    }
  },
  splithalf: texel.commands.splithalf,
  splitresolve: texel.commands.splitresolve,
  splitlegacy: texel.commands.splitlegacy,
  gistats: gi.commands.gistats,
  splitstats: texel.commands.splitstats,
  splitshading: texel.commands.splitshading,
  hitcache: {
    help: 'reflections reuse the light the terrain worked out on screen last frame (rebuilds)',
    usage: 'bxb.hitcache()',
    run: () => {
      hitCacheOn = !hitCacheOn;
      buildPerPixelShadows();
      return `hit lighting cache ${hitCacheOn ? 'on' : 'off - every reflection hit is lit by its own light batch'}`;
    }
  },
  hitshadowbudget: lighting.commands.hitshadowbudget,
  mirrorlightbudget: lighting.commands.mirrorlightbudget,
  lightcutoff: lighting.commands.lightcutoff,
  gridbias: sun.commands.gridbias,
  shade: sun.commands.shade,
  fade: sun.commands.fade,
  sunbias: sun.commands.sunbias,
  internals: {
    help: 'live references for profiling from the console: card bindings, trees, scene. ' +
          'enemyAI.isPaused = true holds Evil Bob still, for image A/Bs',
    run: () => ({ scene, renderer, ...cards.bindings(), treeMeshes,
                  playerSprite, enemySprite, perf, enemyAI })
  },
  gpucalls: {
    help: 'WebGPU queue calls a frame: writeBuffer calls and kB, submits, and the buffers ' +
          'written most. What Firefox\'s WebGPU thread pays for, per call',
    usage: 'await bxb.gpucalls()  |  await bxb.gpucalls(240)  frames to average',
    run: (frames = 120) => countGPUCalls(renderer, frames)
  },
  uploads: {
    help: "A/B the uniform upload fixes: 'coalesce' (one writeBuffer per changed buffer) and " +
          "'unchanged' (skip a table whose bytes did not change)",
    usage: "bxb.uploads('coalesce')  |  bxb.uploads('unchanged')  |  bxb.uploads('batch')",
    run: name => toggleUpload(name)
  },
  gpupasses: {
    help: 'GPU ms per frame by pass and by kernel, averaged over a few seconds. Compute lists ' +
          'are split a kernel a dispatch while it measures, so read the times, not the fps',
    usage: 'await bxb.gpupasses()  |  await bxb.gpupasses(10)  seconds',
    run: (seconds = 5) => withGPUTiming(() => measurePasses(renderer, seconds))
  },
  computebatch: {
    help: 'the mirror cache and the LPV volumes sent in the split shading compute pass and submit. ' +
          'Off: a submit each, as before (A/B)',
    run: () => { computeBatchOn = !computeBatchOn; return `compute batch ${computeBatchOn ? 'on' : 'off'}`; }
  },
  gpu: {
    help: 'average GPU ms over the last frames (render, compute) and the fps, for profiling. ' +
          'bxb.gpu(true) resets the window first and waits for it to refill. Works with the ' +
          'frame graph hidden (bxb.perf()), which is the fairer fps in Firefox',
    usage: 'bxb.gpu()  |  await bxb.gpu(true, 120)',
    run: (reset = false, frames = 120) => withGPUTiming(async wasOff => {
      if (reset || wasOff) {
        perf.resetGPU();
        await new Promise(r => setTimeout(r, 200));   // let in-flight resolves land
        perf.resetGPU();
        while (perf.gpuStats(frames).render.count < frames) await new Promise(r => setTimeout(r, 50));
      }
      const { render, compute } = perf.gpuStats(frames);
      const f = x => +x.toFixed(3);
      return { render: f(render.avg), renderP95: f(render.p95), compute: f(compute.avg),
               total: f(render.avg + compute.avg), renderMax: f(render.max), computeMax: f(compute.max),
               frames: render.count,
               fps: +perf.stats(frames).fps.toFixed(1), graph: perf.visible };
    })
  },
  gpuprofile: {
    help: 'what each feature costs the GPU in this view: turns each off in turn, measures, ' +
          'turns it back on. Stand still. The rebuilding cases compile first - several ' +
          'seconds each in Firefox. Runs uncapped with Evil Bob held still, and lets each ' +
          'change settle before measuring',
    usage: 'await bxb.gpuprofile()  |  await bxb.gpuprofile(480)  frames per measurement',
    run: async (frames = 240) => {
      if (!shadowsOn) return 'run bxb.shadows() first';
      // Steady conditions, put back afterwards:
      // - uncapped, so the GPU is busy every moment: at the vsync cap it idles
      //   half of each frame and its clock moves, and the same work then times
      //   differently from one measurement to the next;
      // - Evil Bob held, so sprites, card shadows and GI sprites stand still.
      const pacing = pacer.mode, ai = enemyAI.isPaused;
      pacer.setMode('uncapped');
      enemyAI.isPaused = true;
      // After any change, a second before measuring: the LPV refills from
      // empty when GI comes back, the hit and texel caches refill, texel GI
      // and the room meter blend in. Measured at once, a case was timed
      // through that refill, and baselines drifted ~0.65 ms (2026-10-26).
      const settle = () => new Promise(r => setTimeout(r, GPUPROFILE_SETTLE_MS));
      const measure = async rebuilt => {
        if (rebuilt) await sets.done;
        await settle();
        return bxbApi.gpu(true, frames);
      };
      try {
        const glass = glassInstancedMesh;
        const glare = mirrors.glare, caustic = glassCaustics.value, layers = glassLayers;
        const texelGI = engine.options().texelGI;
        const rebuild = () => buildPerPixelShadows();
        const flag = set => [() => { set(false); rebuild(); }, () => { set(true); rebuild(); }];
        let shown = [];
        // [name, applies, off, back on, rebuilds]. A case already off is skipped.
        const cases = [
          ['glass surfaces (holes left)', !!glass,
           () => { glass.visible = false; }, () => { glass.visible = true; }, false],
          ['caustics', !!glass && caustic > 0,
           () => bxbApi.caustics(0), () => bxbApi.caustics(caustic), false],
          ['glare off glass', !!glass && glare > 0 && mirrors.on,
           () => bxbApi.glassglare(0), () => bxbApi.glassglare(glare), false],
          ['torch', lighting.torchOn, () => lighting.toggleTorch(), () => lighting.toggleTorch(), false],
          [`model lights (${lighting.modelLightCount})`, lighting.modelLightCount > 0,
           () => { lighting.modelLightsOn = false; }, () => { lighting.modelLightsOn = true; }, false],
          ['sprites (not drawn)', true,
           () => { shown = spriteMeshes().filter(m => m.visible); shown.forEach(m => { m.visible = false; }); },
           () => shown.forEach(m => { m.visible = true; }), false],
          ['sun shadow march', true, ...flag(v => { sun.march = v; }), true],
          [engine.options().gi ? 'AO (cones kept for the LPV)' : 'AO cones', true, ...flag(v => { aoOn = v; }), true],
          ['shadow rays tinting through glass', !!glass, ...flag(v => { glassShadows = v; }), true],
          ['second glass layer', !!glass && layers > 1,
           () => { glassLayers = 1; rebuild(); }, () => { glassLayers = layers; rebuild(); }, true],
          ['mirror light', mirrors.on, () => bxbApi.mirrors(), () => bxbApi.mirrors(), true],
          // GI, whichever method: all of it - then, for texel GI, only the
          // lamps' shadow marches at its ray hits.
          [texelGI ? 'texel GI' : 'LPV GI', !!(texelGI || engine.options().gi),
           () => bxbApi.gi(false), () => bxbApi.gi(true), true],
          ['lamp shadows at texel GI hits', !!texelGI && texelGI.ab.hitShadows.value === 1,
           () => { texelGI.ab.hitShadows.value = 0; }, () => { texelGI.ab.hitShadows.value = 1; }, false],
          // Off costs MORE: every GI ray hit runs the light loop again.
          ['patch direct-light cache (GI hits)', !!texelGI && texelGI.ab.direct.value === 1,
           () => { texelGI.ab.direct.value = 0; }, () => { texelGI.ab.direct.value = 1; }, false],
          ['split shading', texel.splitOn, ...flag(v => { texel.splitOn = v; }), true],
          ['hit lighting cache', hitCacheOn, ...flag(v => { hitCacheOn = v; }), true],
          ['model lookup in reflection hits', !!terrainTextures.modelHits, ...flag(v => { modelHitsOn = v; }), true],
          ['reflections (and glass shading)', reflectionsOn,
           () => bxbApi.spec('reflect'), () => bxbApi.spec('reflect'), true]
        ];
        const f = x => +x.toFixed(3);
        const rows = [];
        // renderP95: the spread within a measurement - a saving smaller than the
        // baselines' drift (the last row) is not a saving.
        const row = (name, m) => ({ case: name, render: m.render, compute: m.compute, renderP95: m.renderP95 });
        const before = await measure(false);
        rows.push(row('baseline', before));
        for (const [name, applies, off, on, rebuilds] of cases) {
          if (!applies) continue;
          off();
          const m = await measure(rebuilds);
          on();
          if (rebuilds) await sets.done;
          rows.push(row(`${name} off`, m));
          console.log(`[gpuprofile] ${name} off: render ${m.render} + compute ${m.compute} ms`);
        }
        const after = await measure(false);
        rows.push(row('baseline again', after));
        // Savings, render + compute (split shading moves work between them),
        // against the mean of the two baselines; their difference is the drift.
        const base = (before.render + before.compute + after.render + after.compute) / 2;
        for (const r of rows) r.saved = f(base - r.render - r.compute);
        console.table(rows);
        console.log(`[gpuprofile] drift between the baselines: ` +
                    `${f(after.render + after.compute - before.render - before.compute)} ms`);
        return rows;
      } finally {
        pacer.setMode(pacing);
        enemyAI.isPaused = ai;
      }
    }
  },
  wgsl: {
    help: "hash of each material's WGSL - drawn, and the texel cache's low-res and the sprite atlas " +
          "passes - to compare across runs and to spot two materials with the same code (one compile). " +
          "Flags names that change per run. bxb.wgsl('text') also returns the source",
    usage: "await bxb.wgsl()  |  await bxb.wgsl('text')",
    run: async (mode = null) => {
      // [label, object]. The low-res materials are only on their meshes inside
      // the low-res pass, so they are read off a stand-in sharing the mesh's
      // instances; the atlas twins wear theirs, on their own layer.
      const stand = (src, mat) => {
        const m = new THREE.InstancedMesh(src.geometry, mat, src.count);
        m.instanceMatrix = src.instanceMatrix;
        m.instanceColor = src.instanceColor;
        return m;
      };
      const set = sets.active;
      const objects = [['terrain', worldInstancedMesh],
                       ['terrain miss', texel.twins.miss],
                       ['terrain low-res', set && set.lowMat && stand(worldInstancedMesh, set.lowMat)],
                       ['glass', glassInstancedMesh],
                       ['glass miss', texel.twins.glassMiss],
                       ['glass low-res', set && set.glassLow && glassInstancedMesh &&
                                         stand(glassInstancedMesh, set.glassLow)],
                       ...spriteMeshes().map(m => ['sprite', m]),
                       ...spriteMeshes().map(m => ['sprite atlas', m.userData.spriteAtlasTwin])];
      const seen = new Set();
      const rows = [];
      const mask = camera.layers.mask;
      for (const [label, o] of objects) {
        if (!o || seen.has(o.material)) continue;
        seen.add(o.material);
        camera.layers.mask |= o.layers.mask;
        let code;
        try { code = await renderer.debug.getShaderAsync(scene, camera, o); }
        finally { camera.layers.mask = mask; }
        const { vertexShader: vs, fragmentShader: fs } = code;
        const unstable = unstableNames(vs + fs);
        rows.push({ object: label, material: o.material.type,
                    vs: hashText(vs), fs: hashText(fs), fsKB: +(fs.length / 1024).toFixed(1),
                    unstable: unstable.join(' ') || '-',
                    ...(mode === 'text' ? { vsText: vs, fsText: fs } : {}) });
      }
      console.table(rows.map(({ vsText, fsText, ...r }) => r));
      const bad = rows.filter(r => r.unstable !== '-').length;
      console.log(bad ? `[wgsl] ${bad} shader(s) carry per-run names - they will miss the pipeline cache`
                      : '[wgsl] no per-run names; hashes should match across runs');
      return rows;
    }
  },
  perf: {
    help: 'toggle the frame time graph',
    run: () => { const on = perf.toggle(); syncGPUTiming(); return on ? 'perf graph on' : 'perf graph off'; }
  },
  perfhz: {
    help: 'frame graph repaints per second (every frame is still sampled); 0 repaints every frame',
    usage: 'bxb.perfhz(10)  |  bxb.perfhz(0)',
    run: (hz = null) => {
      if (hz !== null) perf.drawHz = hz;
      return `frame graph repaints ${perf.drawHz > 0 ? perf.drawHz + ' times a second' : 'every frame'}`;
    }
  },
  pacing: {
    help: "frame pacing: 'vsync' (every refresh), a cap of 30/40/60, or 'uncapped' (benchmark only)",
    usage: "bxb.pacing(30)  |  bxb.pacing('vsync')  |  bxb.pacing('uncapped', 3)",
    run: (mode = null, depth = null) => {
      if (mode !== null && !pacer.setMode(mode)) {
        return `unknown pacing '${mode}' - use 'vsync', 30, 40, 60 or 'uncapped'`;
      }
      if (depth !== null) pacer.depth = depth;
      return `frame pacing: ${describePacing(pacer.mode)}` +
             (pacer.mode === 'uncapped'
               ? `, ${pacer.depth ? pacer.depth + ' frames in flight' : 'unbounded queue'}` : '');
    }
  },
  grid: {
    help: 'toggle the field overlay - both cascades, C1 with C0 subtracted (Alt+G)',
    run: () => toggleFieldDebug(
      scene, player.gridPos, worldInstancedMesh, game.mode,
      { sunDirection: dirLight.position, bias: sun.gridBias }
    ) || 'field overlay off'
  },
  stats: {
    help: 'print world, grid and frame statistics',
    run: () => {
      const origin = gridOriginFor(game.mode, player.gridPos, dirLight.position, sun.gridBias);
      const grid = createFieldAt(origin.x, origin.z, null, 0, origin.y);
      const s = perf.stats();
      const out = {
        blocks: World.size,
        gridOrigin: origin.x + ',' + origin.z,
        gridOccupied: grid.occupiedCount,
        gridVoxels: GRID_DIM ** 3,
        gridMB: +(grid.data.length / 1048576).toFixed(2),
        mode: game.mode,
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
        mode: game.mode
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
      game.path = [];
      return 'teleported to ' + x + ',' + top + ',' + z;
    }
  },
  walk: {
    help: 'walk the player to a block along the explore path, as a click would (for benchmarks)',
    usage: 'bxb.walk(x, z, y?)',
    run: (x, z, y) => {
      const top = (y === undefined || y === null) ? getColumnTop(x, z) : y;
      if (top === null || top === undefined) return 'no standable surface at ' + x + ',' + z;
      const path = findPath(player.gridPos, { x, y: top, z }, true);
      if (!path.length) return 'no path to ' + x + ',' + top + ',' + z;
      game.path = path;
      return 'walking ' + path.length + ' steps to ' + x + ',' + top + ',' + z;
    }
  },
  light: sun.commands.light,
  ...sky.commands, ...tonemap.commands, ...occlusion.commands, ...fog.commands, ...smoke.commands, ...boxLight.commands,
  sun: sun.commands.sun
}));

// --- The settings panel (Alt+V) ---
//
// Every setting routes through the same bxb command the console uses, so the
// two cannot drift - the panel is a front end, not a second owner of the state.
// Ranges marked live:false rebuild the material, so they apply on release.
settingsPanel = createSettingsPanel({ groups: [
  { title: 'Graphics', settings: [presetSetting] },
  // The render debugger: every display that replaces or overlays the frame.
  { title: 'Debug views', settings: [
    debugView.setting,
    { key: 'white', label: 'Whiteworld (Alt+X)', type: 'toggle',
      get: () => whiteWorld, set: v => flip(whiteWorld, v, toggleWhiteWorld) },
    texel.missesSetting,
    gi.missingSetting,
    { key: 'grid', label: 'field overlay (Alt+G)', type: 'toggle',
      get: () => isFieldDebugVisible(),
      set: v => flip(isFieldDebugVisible(), v, () => bxbApi.grid()) }
  ]},
  { title: 'Perf', settings: [
    ...texel.settings,
    sets.setting,
    spriteAtlas.setting,
    { key: 'treemerge', label: 'Merged tree planes', type: 'toggle',
      help: "each tree's four planes drawn as one mesh",
      get: () => treeMerge, set: v => flip(treeMerge, v, () => bxbApi.treemerge()) },
    { key: 'cutouts', label: 'Opaque cutout sprites', type: 'toggle',
      help: 'binary-alpha sprites draw in the opaque pass: one draw, not two',
      get: () => cutoutsEnabled(), set: v => flip(cutoutsEnabled(), v, () => bxbApi.cutouts()) },
    ...gi.perfSettings,
    ...lighting.perfSettings,
    { key: 'coalesce', label: 'One upload per changed buffer', type: 'toggle',
      help: 'a uniform buffer with several changed values goes up in one writeBuffer, not one each',
      get: () => uploadFlags.coalesce, set: v => { uploadFlags.coalesce = v; } },
    { key: 'unchanged', label: 'Skip unchanged tables', type: 'toggle',
      help: 'a table (lights, cards, mirrors, sky...) is uploaded only when its bytes changed',
      get: () => uploadFlags.unchanged, set: v => { uploadFlags.unchanged = v; } },
    { key: 'batchsubmits', label: 'Batch command buffers', type: 'toggle',
      help: "the frame's passes go to the GPU in a few submits, not one each (same work, same order)",
      get: () => uploadFlags.batch, set: v => { uploadFlags.batch = v; } },
    { key: 'pacing', label: 'Frame pacing', type: 'select',
      options: [{ value: 'vsync', label: 'default (vsync)' }, { value: 30, label: 'cap 30' },
                { value: 40, label: 'cap 40' }, { value: 60, label: 'cap 60' },
                { value: 'uncapped', label: 'uncapped (benchmark)' }],
      help: 'which refreshes render; vsync itself and buffering are the browser\'s. ' +
            'Uncapped renders frames that are never shown - for measuring only',
      get: () => pacer.mode, set: v => bxbApi.pacing(v) },
    { key: 'inflight', label: 'Uncapped: frames in flight', type: 'range', min: 0, max: 8, step: 1,
      format: v => (v ? String(v) : 'unbounded'),
      help: 'frames the uncapped loop may queue on the GPU. Too few starves it (the ' +
            'finished signal arrives late); unbounded lets a backlog build and stall',
      get: () => pacer.depth, set: v => bxbApi.pacing(null, v) },
    { key: 'perf', label: 'Frame graph', type: 'toggle',
      get: () => perf.visible, set: v => flip(perf.visible, v, () => bxbApi.perf()) },
    { key: 'spikes', label: 'Log spikes > 8 ms', type: 'toggle',
      get: () => spikeLog, set: v => flip(spikeLog, v, () => bxbApi.spikes()) }
  ]},
  { title: 'Shadows', settings: [
    { key: 'shadows', label: 'Marched shadows', type: 'toggle',
      get: () => shadowsOn, set: v => flip(shadowsOn, v, () => bxbApi.shadows()) },
    ...sun.shadowSettings,
    ...farShadows.shadowSettings
  ]},
  sun.settings,
  tonemap.settings, occlusion.settings,
  { title: 'Shading', settings: [
    ...sun.shadingSettings,
    { key: 'aodist', label: 'AO reach (m)', type: 'range', min: 0.25, max: 3, step: 0.05,
      get: () => aoDistance.value, set: v => bxbApi.ao(v) },
    { key: 'spec', label: 'Specular highlights', type: 'toggle',
      help: 'GGX from the sun and every light, off the LabPBR _s maps',
      get: () => specularOn, set: v => bxbApi.spec(!!v) },
    { key: 'reflect', label: 'Reflections', type: 'toggle',
      help: 'one ray per texel on smooth surfaces, shading what it hits',
      get: () => reflectionsOn, set: v => flip(reflectionsOn, v, () => bxbApi.spec('reflect')) },
    { key: 'reflectbounces', label: 'Reflection bounces', type: 'range', min: 1, max: REFLECT_MAX_BOUNCES, step: 1,
      help: 'reflections in reflections: how many smooth surfaces a reflection ray follows. 1 is the reflection alone',
      get: () => reflectBouncesUniform.value,
      set: v => { reflectBouncesUniform.value = Math.max(1, Math.min(REFLECT_MAX_BOUNCES, Math.round(v))); } },
    ...mirrors.settings,
  ]},
  { title: 'Glass', settings: [
    { key: 'glasslayers', label: 'Glass layers', type: 'range', min: 1, max: 4, step: 1,
      help: 'panes a view ray refracts through; 2 bends through one more glass behind the first',
      get: () => glassLayers, set: v => bxbApi.glasslayers(v) },
    { key: 'caustics', label: 'Caustics', type: 'range', min: 0, max: 3, step: 0.1,
      help: 'light focused by glass, from its normal map; flat glass casts none. 0 is off',
      get: () => glassCaustics.value, set: v => bxbApi.caustics(v) },
    { key: 'dispersion', label: 'Dispersion', type: 'toggle',
      help: 'red and blue refract at their own index: three traces instead of one',
      get: () => glassDispersion > 0, set: v => flip(glassDispersion > 0, v, () => bxbApi.dispersion()) },
    mirrors.glareSetting,
    { key: 'glassbounces', label: 'Internal reflections', type: 'range', min: 0, max: 8, step: 1,
      help: 'reflections inside glass before a ray gives up (then: ambient). Steep rays in a slab need several',
      get: () => glassBounces,
      set: v => { glassBounces = Math.round(v); if (shadowsOn) buildPerPixelShadows(); } }
  ]},
  sky.settings,
  gi.settings,
  fog.settings,
  smoke.settings,
  cards.settings,
  lighting.settings,
  boxLight.settings,
  { title: 'Debug', settings: [
    { key: 'freecam', label: 'Free cam (Alt+C)', type: 'toggle',
      get: () => rig.freeCam, set: v => { rig.freeCam = !!v; } },
    { key: 'hud', label: 'HUD (Alt+U)', type: 'toggle',
      get: () => !document.body.classList.contains('ui-hidden'),
      set: v => { document.body.classList.toggle('ui-hidden', !v); } },
    { key: 'parity', label: 'GPU/CPU march parity', type: 'action', button: 'Run',
      run: () => bxbApi.parity() },
    { key: 'cardparity', label: 'Card parity', type: 'action', button: 'Run',
      run: () => bxbApi.cardparity() },
    { key: 'compute', label: 'Compute smoke test', type: 'action', button: 'Run',
      run: () => bxbApi.compute() },
    { key: 'stats', label: 'Print stats', type: 'action', button: 'Print',
      run: () => JSON.stringify(bxbApi.stats()) }
  ]}
]});

// Every setting by key, for the graphics presets.
for (const g of settingsPanel.groups) for (const def of g.settings) settingDefs.set(def.key, def);

// The top-bar preset select: same as the panel's Graphics row. The choice is
// remembered and put back on the next load (Potato, the default, is the load
// state already). It shows 'custom' once any knob moves off the preset.
{
  const sel = document.getElementById('sel-preset');
  // Keys too: arrows or a letter typed at the open list are not game keys.
  for (const ev of ['pointerdown', 'pointerup', 'click', 'wheel', 'keydown', 'keyup']) {
    sel.addEventListener(ev, e => e.stopPropagation());
  }
  // Just the name here; the target and what it sets are the option's tooltip.
  for (const o of presetSetting.options) {
    const p = PRESETS[o.value];
    const opt = document.createElement('option');
    opt.value = o.value; opt.textContent = p ? p.label : o.label;
    if (p) opt.title = `${p.target}: ${p.help}`;
    opt.disabled = o.value === 'custom';   // shown, never chosen
    sel.appendChild(opt);
  }
  sel.addEventListener('change', () => { setPreset(sel.value).then(m => console.log(m)); sel.blur(); });
  const sync = () => { if (document.activeElement !== sel && !presetBusy) sel.value = currentPreset(); };
  setInterval(sync, 500);
  sync();
  let saved = null;
  try { saved = localStorage.getItem(PRESET_KEY); } catch { /* private window */ }
  if (saved && saved !== 'potato' && PRESETS[saved]) {
    // After the first frames, so the world is on screen while it compiles.
    nextFrame().then(nextFrame).then(() => setPreset(saved)).then(m => console.log(m));
  }
}

// The top-bar Debug button: same toggle as Alt+V. It sits outside every HUD
// layer, so Alt+U leaves it visible. Pointer events stop here so a click never
// also walks Bob to the terrain under the button.
const debugBtn = document.getElementById('btn-debug');
for (const ev of ['pointerdown', 'pointerup', 'click']) {
  debugBtn.addEventListener(ev, e => e.stopPropagation());
}
debugBtn.addEventListener('click', () => { settingsPanel.toggle(); debugBtn.blur(); });

// Bottom right: version, commit and bundle checksum, so a tab can be matched
// to a build at a glance - e.g. that GitHub Pages is serving the latest push.
// Hover for the build time. The commit and time are stamped by vite.config.js.
{
  const stamp = document.createElement('div');
  stamp.id = 'build-stamp';
  stamp.textContent = buildLabel({
    version: VERSION, commit: __BUILD_COMMIT__, dirty: __BUILD_DIRTY__,
    bundle: bundleHash(import.meta.url)
  });
  stamp.title = `built ${__BUILD_TIME__}`;
  document.body.appendChild(stamp);
}

// Beside it: the advanced effects, exactly as R. Labelled with what a click
// will do, and followed by syncEffectsButton, since R and bxb.shadows() can
// change the state too.
const effectsBtn = document.getElementById('btn-effects');
for (const ev of ['pointerdown', 'pointerup', 'click']) {
  effectsBtn.addEventListener(ev, e => e.stopPropagation());
}
effectsBtn.addEventListener('click', () => { toggleAdvancedEffects(); effectsBtn.blur(); });
syncEffectsButton();

// setAnimationLoop instead of a manual requestAnimationFrame chain: WebGPURenderer
// needs an async device/adapter init before the first frame, and setAnimationLoop
// awaits it internally. Calling animate() directly would render before the device
// exists.
// --- AO, always on ---
//
// Two methods, swapped with shadows. Shadows on: the shaded material traces
// six cones through the distance field. Shadows off: the terrain's PLAIN
// material carries the static voxel AO (voxelao.js) as its aoNode - eight
// block reads a texel, no field, no marches - so the cheap path stays cheap on
// low-power machines, and nothing is baked or followed while shadows are off.
// Times the sprites' contact discs (analytic), once the cards have loaded.
function enableAlwaysOnAO() {
  if (!worldInstancedMesh) return;
  const base = worldInstancedMesh.userData.originalMaterial || worldInstancedMesh.material;
  base.aoNode = createVoxelAONode(terrainTextures, cards.ready ? cards.sunBindings : null);
  base.needsUpdate = true;
  if (!worldInstancedMesh.userData.originalMaterial) worldInstancedMesh.userData.originalMaterial = base;
}
enableAlwaysOnAO();

renderer.setAnimationLoop(pacer.onRefresh);
