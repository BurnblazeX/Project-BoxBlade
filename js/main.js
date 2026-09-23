import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import { World, getVoxelKey, createTestArea, createEntity, pathDistance, findPath, enterBattle, exitBattle, getReachableVoxels, addToInventory, isInInteractRange, isStandable, getColumnTop, CHUNK_SIZE, isSolid } from './world.js';
import { createObject, rollLootTable } from './objects.js';
import { initWorldRender, VOXEL_SIZE, updateVoxelTints, updateVoxelVisibility, worldInstancedMesh, keyForInstance, voxelIndexMap } from './render.js';
import { createWanderAI } from './ai.js';
import { toggleBoxGridDebug, refreshBoxGridDebug } from './debug.js';
import { createBoxGridAt, gridOriginFor, GRID_DIM, marchOccupancy, sphereTrace,
         SUN_BIAS_BLOCKS, CASCADE_COUNT, cascadeExtentMetres, cascadeVoxelMetres,
         VOXEL_METRES, applyHandoff } from './boxgrid.js';
import { makeClipSamples, updateCharacterClipping } from './sprites.js';
import { createPerfOverlay } from './perf.js';
import { createCardBindings, createCardAtlasTexture, writeCardBindings,
         runGPUCards, MAX_CARDS } from './gpu.js';
import { runComputeSmokeTest, createDistanceTexture, updateDistanceTexture,
         runGPUMarch, createShadowColorNode, createShadowMaterial,
         restoreOriginalMaterial, followShadowGrid, commitShadowGrid, createSpriteShadowMaterial,
         createNormalTexture,
         createBayerTexture, writeSunUniforms,
         SURFACE_BIAS_VOXELS, SHADOW_FADE_START, EDGE_FADE_VOXELS,
         DEFAULT_AMBIENT, cascadeBindings, writeCascadeBindings, AO_DISTANCE, SEAM_BAND_VOXELS } from './gpu.js';
import { SUN_ANGULAR_SIZE, penumbraTexels } from './sun.js';
import { TORCH_COLOUR, TORCH_LEVEL, TORCH_HEIGHT, TORCH_FORWARD,
         TORCH_SIDE, MAX_LIGHT_LEVEL,
         LIGHT_SOURCE_RADIUS, clampLevel, lightRadiusMetres, colourToRGB,
         pointPenumbraMetres } from './lights.js';
import { createOccluder, placeOccluder, applyOccluders } from './occluders.js';
import { maskFromRGBA, createSilhouetteSet, facingToward,
         AXIS_X, AXIS_Z } from './silhouette.js';
import { createCard, cardFacingFor, buildCardAtlas, packCardInstances,
         cullCardsForLight, cardsVisibility, CARD_PAD, SUN_CARD_LOD_SHIFT,
         SUN_CARD_LOD_MAX } from './cards.js';
import { spriteNormals, terrainNormals } from './normals.js';
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
//
// The device is requested here rather than by three, whose own request asks for
// featureLevel "compatibility" - Firefox does not support that yet, hands back a
// core adapter anyway and logs a notice on every load. Same features as three
// would ask for: everything the adapter has. Any failure falls back to letting
// three do it.
async function requestGPUDevice() {
  try {
    const adapter = navigator.gpu && await navigator.gpu.requestAdapter();
    return adapter ? await adapter.requestDevice({ requiredFeatures: [...adapter.features] })
                   : undefined;
  } catch (err) {
    console.warn('[renderer] own WebGPU device failed, letting three request one', err);
    return undefined;
  }
}
// trackTimestamp gives the perf graph the GPU's own frame time. It only takes
// effect when the device has timestamp-query; without it the graph says so.
const renderer = new THREE.WebGPURenderer({ antialias: true, trackTimestamp: true,
                                           device: await requestGPUDevice() });
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
// A pixel-art sprite texture of its own. Each character needs one so it can flip
// independently - and a separate LOAD rather than a clone: Texture.clone() flags
// the copy for upload at once, while the shared image is still null until the
// file arrives, and a first frame drawn in that window crashes in three.
function loadSpriteTexture(url) {
  const tex = texLoader.load(url);
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

const playerTex = loadSpriteTexture(bobTextureUrl);
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
const enemySprite = createCharacterMesh(loadSpriteTexture(evilBobTextureUrl));
enemySprite.position.copy(getSpriteWorldPos(enemy.gridPos));
scene.add(enemySprite);

const enemyAI = createWanderAI(enemy, enemySprite, 4);

// --- Shadow-casting proxies ---
//
// A sprite is a camera-facing quad with no thickness, so it is not something the
// distance field can be baked from directly - see occluders.js for why
// voxelising the quad itself would make a character's shadow change shape as the
// camera orbits. Each caster gets a world-locked box instead, and the box is all
// the grid ever sees.
//
// Sized against the sprite quads rather than guessed. The character quad is
// VOXEL_SIZE x VOXEL_SIZE*2, translated so its base sits at the mesh position -
// so mesh.position is the FEET, which is what placeOccluder wants. The figure
// does not fill its quad (pixel art carries transparent padding), so the proxy
// is narrower and shorter than the geometry: roughly the body, not the canvas.
const CHARACTER_PROXY = { width: VOXEL_SIZE * 0.5, height: VOXEL_SIZE * 1.6 };
// A tree's quad is VOXEL_SIZE*2 x VOXEL_SIZE*3, and the same reasoning applies -
// a trunk and canopy inside a mostly-empty rectangle.
const TREE_PROXY = { width: VOXEL_SIZE * 0.9, height: VOXEL_SIZE * 2.4 };

// --- Reading the cutout off the sprite ---
//
// The only DOM-shaped part of the silhouette path, and the reason it lives here
// rather than in silhouette.js: a canvas cannot exist in the headless test
// suite, so the module that does the maths takes a plain mask array and this
// function is the one thing that has to be trusted by inspection.
//
// Drawn at native size with smoothing off. The sprite is 12 x 24 and the voxel
// grid it is about to become is also 12 x 24 - any scaling here would resample
// an exact correspondence into an approximate one.
function imageRGBA(image) {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(image, 0, 0);
  return ctx.getImageData(0, 0, image.width, image.height).data;
}

function maskFromTexture(image) {
  return { mask: maskFromRGBA(imageRGBA(image), image.width, image.height),
           w: image.width, h: image.height };
}

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

// The grass texture's normals, once its image has loaded. Polled like the
// cards, for the same reason (Texture has no load event).
let terrainNormalTex = null;
function resolveTerrainNormal() {
  if (terrainNormalTex || !worldInstancedMesh) return false;
  const base = worldInstancedMesh.userData.originalMaterial || worldInstancedMesh.material;
  const map = base.userData.albedoMap !== undefined ? base.userData.albedoMap : base.map;
  const img = map && map.image;
  if (!img || !img.width) return false;
  terrainNormalTex = normalMapFor('terrain_grass', () => ({
    data: terrainNormals(imageRGBA(img), img.width, img.height),
    w: img.width, h: img.height
  }));
  return true;
}

// Textures load asynchronously, so a silhouette cannot be built at module scope.
// Until one arrives the proxy is a box - which is a working shadow, not a
// placeholder, so there is no first-frame flicker of nothing being cast.
//
// RESOLVED BY POLLING, NOT BY AN EVENT, and that is not laziness. THREE.Texture
// dispatches exactly one event, 'dispose' - there is no 'load', and
// TextureLoader only assigns texture.image inside ImageLoader's own callback. An
// addEventListener('load') on the texture therefore never fires, which is how
// every silhouette here silently stayed a box: the fallback is a working shadow,
// so nothing looked broken, it just looked rectangular.
//
// Checking each frame cannot miss a load the way a missing event can, costs one
// truthiness test per proxy once resolved, and self-heals if a texture is ever
// swapped at runtime.
const pendingSilhouettes = [];

function attachSilhouette(proxy, texture, widthMetres, heightMetres) {
  pendingSilhouettes.push({ proxy, texture, widthMetres, heightMetres });
}

function resolveSilhouettes() {
  for (let i = pendingSilhouettes.length - 1; i >= 0; i--) {
    const p = pendingSilhouettes[i];
    const img = p.texture.image;
    if (!img || !img.width) continue;
    const { mask, w, h } = maskFromTexture(img);
    // One per cascade. A single silhouette is expressed in the voxels of the
    // level it was built for, so sharing it across levels makes the character
    // twice as wide in C1 and four times in C2 - see createSilhouetteSet.
    p.proxy.silhouettes = createSilhouetteSet({
      mask, w, h,
      widthMetres: p.widthMetres, heightMetres: p.heightMetres
    });
    pendingSilhouettes.splice(i, 1);
  }
}

// How the cards are turned. 'sun' points each card's normal along the sun's
// azimuth, so the full silhouette always faces the light and the fixed-facing
// degeneracy - a card edge-on to the sun casting a line - cannot arise.
//
// This is still not a CAMERA-facing proxy, which is what S9 rules out and for a
// specific reason: a camera-facing card's thickness is camera-relative, so the
// shadow changes shape as the player orbits. A sun-facing card is world-locked
// with respect to the viewer and re-orients only when the light moves, which is
// what a real silhouette does anyway.
//
// 'x' and 'z' lock to a world plane instead, kept for comparison.
let cardMode = 'sun';

function cardFacing() {
  if (cardMode === 'x') return AXIS_X;
  if (cardMode === 'z') return AXIS_Z;
  return facingToward(dirLight.position.x, dirLight.position.z);
}

// Sprites are turned around by mirroring their UVs - repeat.x goes to -1, see
// updateSpriteFacing - not by scaling the mesh. So that is where any mirror
// state has to be read from. One definition rather than one per caller: the
// cutout and the torch both need it, and a torch that disagreed with the
// silhouette about which way Bob faces would swap sides a frame early or late.
const mirrored = sprite => sprite.material.map.repeat.x < 0;

const playerProxy = createOccluder('player', CHARACTER_PROXY);
const enemyProxy = createOccluder('enemy', CHARACTER_PROXY);
attachSilhouette(playerProxy, bobTexture, VOXEL_SIZE, VOXEL_SIZE * 2);
attachSilhouette(enemyProxy, evilBobTexture, VOXEL_SIZE, VOXEL_SIZE * 2);
// Trees never move, so after the first frame they cost exactly nothing - the
// snapped box compares equal and applyOccluders does no work. They are in the
// list for the same reason the characters are: they are sprites, so they are
// not in the World map, so without a proxy they cast nothing at all.
const treeProxies = [];
// The baked proxies are OFF by default: analytic cards (above) are the shadow
// path now, and running both would have every sprite occlude twice. Kept
// switchable because the bake is what reflections and GI will need.
let proxiesOn = false;

// --- Analytic cards: the shadow path sprites actually use ---
//
// The distance field is geometry shared by every light, so a sprite baked into
// it faces one direction and every other light gets the wrong silhouette. These
// live OUTSIDE the field: each light intersects each card, turned to face that
// light. See cards.js.
//
// The baked proxies above are kept and still work - bxb.proxies() turns them on
// - but they are off by default now, because the two would double-count. What
// they are for from here is reflections and GI (S9), which need sprites present
// in the grid rather than tested against a ray.
const cardTypes = [];          // { card, typeIndex }, one per sprite texture
let cardAtlas = null;
let cardAtlasTex = null;
const pendingCards = [];

// The texture arrives as a THUNK rather than a value, and that is about module
// order rather than laziness. These registrations sit next to the rest of the
// sprite-shadow code, but the textures they name are created wherever their
// sprite happens to be set up - treeTexture is nearly two hundred lines further
// down. Naming one directly here reads it during module evaluation, before the
// const exists, which is a TDZ error at load and not something a build catches.
//
// A thunk is evaluated in resolveCards, long after every module-level const has
// settled, so where a texture is declared stops mattering at all.
// name is the texture's file name, which is what a normal-map override is
// matched by.
function registerCard(key, name, getTexture, widthMetres, heightMetres) {
  const entry = { key, name, getTexture, widthMetres, heightMetres, card: null,
                  normalTex: null };
  pendingCards.push(entry);
  return entry;
}

const bobCard = registerCard('bob', 'character_Bob', () => bobTexture,
                             VOXEL_SIZE, VOXEL_SIZE * 2);
const evilCard = registerCard('evil', 'character_EvilBob', () => evilBobTexture,
                              VOXEL_SIZE, VOXEL_SIZE * 2);
const treeCard = registerCard('tree', 'decor_tree', () => treeTexture,
                              VOXEL_SIZE * 2, VOXEL_SIZE * 3);

// Same polling as the silhouettes, and for the same reason: THREE.Texture has no
// 'load' event, so there is nothing to listen to.
function resolveCards() {
  let built = false;
  for (let i = pendingCards.length - 1; i >= 0; i--) {
    const p = pendingCards[i];
    const tex = p.getTexture();
    const img = tex && tex.image;
    if (!img || !img.width) continue;
    // Per sprite, so one unreadable texture cannot take the rest down with it -
    // and so the reason is reported rather than surfacing as "still loading"
    // forever, which is what it looked like.
    let mask, w, h;
    try {
      ({ mask, w, h } = maskFromTexture(img));
    } catch (err) {
      p.error = String(err && err.message || err);
      continue;
    }
    // How much of the sprite is opaque. Kept because a mask that comes back
    // fully solid is indistinguishable from a working one until you look at the
    // shadow: every card becomes its own bounding box and casts a slab.
    let solid = 0;
    for (let k = 0; k < mask.length; k++) solid += mask[k];
    p.coverage = solid / mask.length;
    p.card = createCard({ mask, w, h,
                          widthMetres: p.widthMetres, heightMetres: p.heightMetres });
    // From the card's own distance transform: the silhouette the shadow uses
    // is the silhouette the bevel follows.
    const card = p.card;
    p.normalTex = normalMapFor(p.name, () => ({
      data: spriteNormals(card.dt, card.w, card.cols, card.rows, CARD_PAD), w, h
    }));
    p.typeIndex = cardTypes.length;
    cardTypes.push(p);
    pendingCards.splice(i, 1);
    built = true;
  }
  if (!built) return false;
  // Repacked whole rather than appended: the atlas is a handful of small
  // bitmaps, and a rect that moved would silently point every instance of that
  // card at the wrong pixels.
  cardAtlas = buildCardAtlas(cardTypes.map(t => t.card));
  return true;
}

// Every sprite that should cast, with no facing yet - the facing is per light
// and is filled in per light below.
function cardCasters() {
  const out = [];
  const add = (entry, sprite, flip) => {
    if (!entry || !entry.card || !sprite.visible) return;
    out.push({
      card: entry.card, typeIndex: entry.typeIndex,
      // The card's centre, not its feet: the mask is centred on the quad.
      centre: { x: sprite.position.x,
                y: sprite.position.y + entry.heightMetres / 2,
                z: sprite.position.z },
      flip, enabled: true
    });
  };
  add(bobCard, playerSprite, mirrored(playerSprite));
  add(evilCard, enemySprite, mirrored(enemySprite));
  for (const m of treeMeshes) add(treeCard, m, false);
  return out;
}

// Rebuilt each frame rather than kept in sync by hand. The set is small and the
// alternative is a registration path that has to be remembered at every spawn
// and despawn - a corpse that keeps casting is exactly the bug that costs.
function activeProxies() {
  if (!proxiesOn) return [];
  const out = [];
  if (playerSprite.visible) {
    out.push(placeOccluder(playerProxy, playerSprite.position.x,
                           playerSprite.position.y, playerSprite.position.z,
                           mirrored(playerSprite)));
  }
  if (enemySprite.visible) {
    out.push(placeOccluder(enemyProxy, enemySprite.position.x,
                           enemySprite.position.y, enemySprite.position.z,
                           mirrored(enemySprite)));
  }
  for (let i = 0; i < treeMeshes.length; i++) {
    const m = treeMeshes[i];
    if (!m.visible) continue;
    if (!treeProxies[i]) {
      treeProxies[i] = createOccluder('tree' + i, TREE_PROXY);
      // One silhouette per tree rather than one shared: createSilhouette is
      // cheap and called once, and sameBox compares the object by identity, so a
      // shared instance would be fine too - but a per-tree one leaves room for
      // variants (a stump, a dead tree) without restructuring.
      attachSilhouette(treeProxies[i], treeTexture, VOXEL_SIZE * 2, VOXEL_SIZE * 3);
    }
    out.push(placeOccluder(treeProxies[i], m.position.x, m.position.y, m.position.z));
  }
  return out;
}

// Write the proxies into every cascade and report whether anything changed.
//
// Every cascade, not just C0: a shadow ray leaves C0 after 18 m and continues in
// C1, so a caster present only in the fine level would stop casting at exactly
// the distance the ray crossed over - a character's shadow vanishing partway
// across the ground for no visible reason.
//
// The texture upload is conditional on the SAME flag, which is the point of
// applyOccluders returning one: a 3 MB Data3DTexture re-upload per cascade per
// frame would cost more than the bake it is reporting, and in a turn-based scene
// most frames have nobody mid-step.
// One binding per LIGHT, not one shared, because the facing is the whole point:
// a card turned toward the sun is not the card the torch needs, and a single
// buffer can only hold one orientation.
let sunCards = null, torchCards = null;
let cardPack = null;
let cardsReady = false;

function ensureCardBindings() {
  if (sunCards) return;
  sunCards = createCardBindings(MAX_CARDS);
  sunCards.lodShift.value = SUN_CARD_LOD_SHIFT;
  sunCards.lodMax.value = SUN_CARD_LOD_MAX;
  torchCards = createCardBindings(MAX_CARDS);
  cardPack = new Float32Array(MAX_CARDS * 16);
}

// Fill one light's buffer: cull to what that light can reach, turn each card to
// face it, pack, upload.
function writeCardsFor(binding, casters, lightPos, radiusMetres, directional) {
  const near = cullCardsForLight(casters, lightPos, radiusMetres, binding.capacity);
  for (const c of near) {
    // A directional light is the same direction from everywhere, so every card
    // turns the same way. A point light is in a DIFFERENT direction from every
    // sprite, which is the case a baked field cannot express at all.
    c.facing = directional
      ? facingToward(lightPos.x, lightPos.z)
      : cardFacingFor(c.centre, lightPos.x, lightPos.z);
  }
  const n = packCardInstances(near, cardAtlas, cardPack, binding.capacity);
  writeCardBindings(binding, cardPack, n);
  return n;
}

let lastCardCount = 0;

// Which cascade a card stands in, as the lod createCardsTSL reads: the level,
// plus how far into the last SEAM_BAND_VOXELS of it toward the next. null when
// it is outside every level - no terrain shadow reaches there either, so the
// card is dropped before it costs any fragment anything.
function cardLod(centre) {
  for (let l = 0; l < CASCADE_COUNT; l++) {
    const g = shadowGrids[l];
    if (!g) return 0;
    const side = cascadeExtentMetres(l);
    const ex = Math.min(centre.x - g.origin.x, g.origin.x + side - centre.x);
    const ez = Math.min(centre.z - g.origin.z, g.origin.z + side - centre.z);
    if (ex < 0 || ez < 0) continue;
    const edge = Math.min(ex, ez) / cascadeVoxelMetres(l);
    const t = Math.min(1, edge / SEAM_BAND_VOXELS);
    return l + (1 - t * t * (3 - 2 * t));   // 1 - smoothstep, as c0WeightTSL
  }
  return null;
}

function updateCards() {
  if (resolveTerrainNormal() && shadowsOn) buildPerPixelShadows();
  if (pendingCards.length && resolveCards()) {
    // The atlas exists now where it did not before, and the kernel branches on
    // whether there is one - so the material has to be rebuilt once. A one-frame
    // hitch on load, and nothing afterwards.
    cardAtlasTex = createCardAtlasTexture(cardAtlas);
    ensureCardBindings();
    sunCards.atlas = torchCards.atlas = cardAtlasTex;
    sunCards.atlasSize.value.set(cardAtlas.width, cardAtlas.height);
    torchCards.atlasSize.value.set(cardAtlas.width, cardAtlas.height);
    cardsReady = true;
    if (shadowsOn) { buildPerPixelShadows(); }
  }
  if (!cardsReady || !cardsOn) { lastCardCount = 0; return; }

  const casters = cardCasters().filter(c => (c.lod = cardLod(c.centre)) !== null);
  // The sun is directional: its "position" is a direction, and everything is
  // within reach of it.
  lastCardCount = writeCardsFor(sunCards, casters, dirLight.position, Infinity, true);
  if (torchOn && torchUniforms) {
    writeCardsFor(torchCards, casters, torchUniforms.position.value,
                  lightRadiusMetres(torchLevel), false);
  } else {
    torchCards.count.value = 0;
  }
}

let cardsOn = true;

function updateOccluders() {
  // Cheap once everything has loaded, and the only thing that turns a box back
  // into a cutout - see attachSilhouette for why this is not an event handler.
  if (pendingSilhouettes.length) resolveSilhouettes();

  const proxies = activeProxies();
  const facing = cardFacing();
  for (let l = 0; l < shadowGrids.length; l++) {
    if (!shadowGrids[l]) continue;
    if (applyOccluders(shadowGrids[l], proxies, facing)) {
      updateDistanceTexture(shadowTexes[l]);
    }
  }
}

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
const timer = new THREE.Timer();
const keyState = { w: false, a: false, s: false, d: false };

// --- Free cam (Alt+C, debug) ---
//
// Detaches the camera from Bob: WASD flies the pivot instead of walking him,
// Q/E still turn it. Only the CAMERA moves - the cascades, the boxGrid overlay
// and everything else that follows the player keep following player.gridPos,
// so flying away shows the field's footprint staying put around Bob.
let freeCam = false;
const FREE_CAM_SPEED = 12;   // m/s
function moveFreeCam(dt) {
  let rx = 0, rz = 0;
  if (keyState.w) rz -= 1;
  if (keyState.s) rz += 1;
  if (keyState.a) rx -= 1;
  if (keyState.d) rx += 1;
  if (!rx && !rz) return;
  const len = Math.hypot(rx, rz);
  // Screen-relative, the same basis Bob walks in.
  const h = (rotationStep * Math.PI / 2) + cameraConfigs[currentMode].headingOffset;
  const c = Math.cos(h), s = Math.sin(h);
  const step = FREE_CAM_SPEED * dt / len;
  pivot.position.x += (rx * c + rz * s) * step;
  pivot.position.z += (-rx * s + rz * c) * step;
}

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

// Whiteworld strips the terrain's albedo by clearing the map on the ORIGINAL
// material - applyShadowMaterial reads its albedo from there, so the shadowed
// path picks it up on rebuild and the unshadowed path picks it up directly.
let whiteWorld = false;
const whiteUniform = uniform(0);
function toggleWhiteWorld() {
  const mesh = worldInstancedMesh;
  if (!mesh) return 'no terrain yet';
  const base = mesh.userData.originalMaterial || mesh.material;
  if (base.userData.albedoMap === undefined) base.userData.albedoMap = base.map;
  whiteWorld = !whiteWorld;
  base.map = whiteWorld ? null : base.userData.albedoMap;
  base.needsUpdate = true;
  // The shadowed material reads this as a uniform, so no rebuild.
  whiteUniform.value = whiteWorld ? 1 : 0;
  return `whiteworld ${whiteWorld ? 'on' : 'off'}`;
}

window.addEventListener('keydown', (e) => {
  const key = e.key.toLowerCase();

  // Alt+X: whiteworld - terrain drawn plain white so light and shadow can be
  // judged without the grass texture in the way. Sprites keep their textures.
  // Matched on e.code, not e.key, because holding Alt changes the reported
  // character on several keyboard layouts while the physical key code stays put.
  // preventDefault stops Alt from reaching the browser's own menu handling.
  if (e.altKey && e.code === 'KeyX') {
    e.preventDefault();
    console.log(toggleWhiteWorld());
    return;
  }

  // Alt+<key> is reserved for debug modes.
  if (e.altKey && e.code === 'KeyC') {
    e.preventDefault();
    freeCam = !freeCam;
    console.log(freeCam ? 'free cam on - WASD flies the camera, Bob stays put'
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

  // Debug: Alt+G toggles the boxGrid occupancy overlay.
  if (e.altKey && e.code === 'KeyG') {
    e.preventDefault();
    const msg = toggleBoxGridDebug(scene, player.gridPos, worldInstancedMesh,
                                   currentMode, { sunDirection: dirLight.position,
                                                  bias: sunGridBias });
    console.log(msg || '[boxGrid debug] off');
    return;
  }

if (key === 'escape' && isDialogueOpen) {
     closeInteractionPanel();
  }

  if (keyState.hasOwnProperty(key)) {
    keyState[key] = true;
  }

  // T: the torch. A held light is the thing the marched field is for - it moves
  // every frame and costs no re-bake, which a shadow map could not have done.
  if (key === 't') {
    console.log(toggleTorch());
    return;
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
function animate(time) {
  const tStart = performance.now();
  // Clamped: after a stall (a tab switch, a compile) one huge step would carry
  // movement through walls and snap every ease to its target.
  timer.update(time);
  const dt = Math.min(timer.getDelta(), 0.1);

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

  if (isWalkingManual && !freeCam && currentMode === 'explore' && currentPath.length > 0) {
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
  else if (currentMode === 'explore' && inputRules.explore.keyboard && !freeCam) {
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
  // The lerp factors were tuned as "per frame at 60 Hz". Converted to the
  // equivalent fraction for this frame's dt, so a 240 Hz screen eases at the
  // same speed as a 60 Hz one instead of four times faster.
  const ease = 1 - Math.pow(1 - CAMERA_LERP_SPEED, dt * 60);
  if (freeCam) {
    moveFreeCam(dt);
  } else if (currentMode === 'explore') {
    pivot.position.lerp(playerSprite.position, ease);
  } else {
    pivot.position.lerp(arenaCenter, ease);
  }

  const config = cameraConfigs[currentMode];
  const targetHeading = (rotationStep * Math.PI / 2) + config.headingOffset;
  
  // Lerp all camera properties smoothly
  currentFov += (config.fov - currentFov) * ease;
  currentPitch += (config.pitch - currentPitch) * ease;
  currentHeading += (targetHeading - currentHeading) * ease;
  currentDistance += (config.distance - currentDistance) * ease;

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
  refreshBoxGridDebug(scene, player.gridPos, currentMode,
                      { sunDirection: dirLight.position, bias: sunGridBias });

  // The cascades follow the player so shading works wherever they are, not only
  // in the patch the grid happened to be built in. Nothing else has to happen
  // when they move: the terrain material marches the field live, so a re-origin
  // is a texture upload and the next frame is already correct.
  const tGrid = performance.now();
  const moved = shadowsOn && followShadowGrids();
  const tCards = performance.now();
  // Before the torch moves? No - after, so a card is tested against where the
  // flame IS this frame rather than where it was last one. updateTorchPosition
  // runs below, so this reads the previous frame's position by one frame, which
  // at walking speed is under a centimetre and invisible; reordering it would
  // mean recomputing the torch position twice.
  if (shadowsOn) updateCards();
  // AFTER the re-origin, never before. A scroll bakes the strip that slid in,
  // which wipes the imprint of any proxy standing there, and it translates the
  // remembered boxes into the new coordinates - so running this first would
  // write proxies the scroll then destroys, and cost the bake twice.
  if (shadowsOn) updateOccluders();
  updateTorchPosition();

  const tRender = performance.now();
  renderer.render(scene, camera);
  const tEnd = performance.now();
  logSpike(tGrid, tCards, tRender, tEnd, moved);
  perf.update(dt, { logic: tGrid - tStart, grids: tCards - tGrid,
                    cards: tRender - tCards, submit: tEnd - tRender });
  resolveGPUTime();
}

// One resolve in flight at a time: each returns the GPU time of everything
// rendered since the last, so overlapping them would split frames in two.
// Resolved every frame even with the graph hidden - unresolved queries fill
// three's query pool.
let gpuTimePending = false;
function resolveGPUTime() {
  if (gpuTimePending || !renderer.backend.trackTimestamp) return;
  gpuTimePending = true;
  renderer.resolveTimestampsAsync('render').then(ms => {
    gpuTimePending = false;
    if (typeof ms === 'number') perf.gpu(ms);
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
// One entry per cascade. C0 is the fine 18 m field the texel lock is aligned to;
// C1 is half resolution over 36 m and exists so a long shadow's ray still finds
// its caster after C0 has run out. See boxgrid.js for the level table.
let shadowGrids = [];
let shadowTexes = [];
let shadowCascades = [];   // the GPU bindings for each level
let shadowSun = null;       // live sun uniform set, so bxb.light can move it
let shadowOrigins = [];     // block origin each level is currently built at

// Builds every cascade at its own footprint, reusing the buffers if they exist.
// Called on turn-on and whenever a level's origin moves.
function ensureShadowGrids() {
  for (let l = 0; l < CASCADE_COUNT; l++) {
    const o = gridOriginFor(currentMode, player.gridPos, dirLight.position, sunGridBias, l);
    shadowGrids[l] = createBoxGridAt(o.x, o.z, shadowGrids[l] || null, l);
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
      fieldWorker = new Worker(new URL('./fieldWorker.js', import.meta.url), { type: 'module' });
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
      shadowOrigins[m.level] = { x: m.x, z: m.z };
      fieldApplied = true;
    };
  }
  if (!fieldWorker) return;
  // Every reset, not once. World is static today, but a mirror baking from
  // stale blocks would hand back strips of terrain that is not there, so any
  // future terrain edit only has to trigger a reset to reach it.
  fieldWorker.postMessage({ type: 'world', keys: [...World.keys()] });
  fieldGen++;
  fieldInFlight.length = 0;
  for (let l = 0; l < CASCADE_COUNT; l++) {
    const o = shadowOrigins[l];
    fieldWorker.postMessage({ type: 'reset', level: l, x: o.x, z: o.z });
  }
}

// Re-origins only the levels that actually moved. C0 moves every block stepped;
// C1 snaps to a 2-block stride, so it rebuilds half as often - which is most of
// why a second level is affordable at all.
function followShadowGrids() {
  let moved = fieldApplied;
  fieldApplied = false;
  for (let l = 0; l < CASCADE_COUNT; l++) {
    const want = gridOriginFor(currentMode, player.gridPos, dirLight.position, sunGridBias, l);
    const at = shadowOrigins[l];
    if (at && want.x === at.x && want.z === at.z) continue;
    if (fieldWorker) {
      if (fieldInFlight[l]) continue;
      fieldInFlight[l] = true;
      fieldWorker.postMessage({ type: 'scroll', level: l, x: want.x, z: want.z, gen: fieldGen });
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

// --- The sun: a marched cone ---
//
// The CSM is gone; see sun.js for why. The sun is now the same sphere-traced ray
// bxb.shadows always cast, widened into a cone over the solar disc, so softening
// and shadowing are one technique rather than a shadow plus a filter.
//
// sunRays is the sample count and is a COMPILE-TIME property of the kernel: 1 is
// the hard reference, and anything above it is soft. Tracked alongside the other
// pass-shape flags because reusing a pass built at a different count is silent -
// everything runs and nothing on screen changes, which is exactly what toggling
// the old bxb.csm() off and on used to do.
let sunRays = 1;
// Which way the penumbra is obtained. false samples the solar disc with sunRays
// rays; true gets the whole thing from a single cone trace against the widened
// field. Both are the same march and the same field - see gpu.js for the one
// thing the cone gives up, which is occluder shape.
let sunCone = false;
let sunAngular = SUN_ANGULAR_SIZE;
let sunBayer = null;
let sunSteps = null;    // quantisation levels for the soft tail
let sunBias = null;     // ray origin lift off the face, in voxels
// Where a shadow starts dissolving as its ray runs out of reach. Uniforms rather
// than constants because this is the knob that decides whether the march's limits
// are visible, and it has to be found by eye.
let sunFadeStart = null, sunEdgeFade = null;
// Set for exactly one pass to overwrite every cached texel. A stored shadow is
// only good while the sun and the world it was measured against hold, so moving
// either invalidates all of them at once.
// How N.L is applied. 'ground' normalises against what a horizontal surface
// receives, so the sun's ANGLE sets shadow direction and length rather than the
// whole scene's brightness - see sunShadeTSL. A kernel branch, so changing it
// rebuilds; ambient beside it is a plain uniform.
let shadeMode = 'ground';
// How far the explore footprint leans toward the sun, in blocks. A plain number
// rather than a uniform: it changes WHERE the grid is baked, so it takes effect
// on the next re-origin, not on the next shade.
let sunGridBias = SUN_BIAS_BLOCKS;
let sunAmbient = null;
// The sun, off. Not a zero direction or a black colour - a KERNEL flag, so the
// material has no sun march in it at all and a torch-only scene pays for one
// light instead of two. This is what a cave or a night is.
let sunOn = true;
let shadowOnly = false; // raw visibility, no albedo or N.L, for judging artifacts
// Cone-traced AO. On/off is compiled into the kernel; the distance is a
// uniform. aoOnly shows the AO term alone.
let aoOn = true;
let aoOnly = false;
const aoDistance = uniform(AO_DISTANCE);

// Quantisation is a KERNEL flag, not a uniform, so changing it rebuilds. Off by
// default now: it was inherited from the sampled path, where a fixed sample
// pattern genuinely does band, and applied to the cone trace, whose output is
// analytic and already smooth. Dithering a smooth gradient only adds crosshatch.
let sunQuantise = false;
// --- The torch: a dynamic point light on the player ---
//
// A level of 0 means off, and off is a KERNEL flag rather than a uniform set to
// zero: with no torch the material has no second march in it at all. Toggling
// costs a material swap, which is a one-frame hitch on a keypress and nothing
// during play - whereas a march that always runs and multiplies out to zero
// would be paid on every fragment of every frame forever.
let torchOn = false;
let torchLevel = TORCH_LEVEL;
// How big the flame is, in metres. A uniform rather than a kernel constant: it
// is the knob the penumbra is found by eye with, so a sweep has to be a uniform
// write rather than a material rebuild.
let torchSourceRadius = LIGHT_SOURCE_RADIUS;
// How far out in front of the holder, and to their side, the flame is held, in
// metres. Plain numbers rather than uniforms: they are read on the CPU when the
// position is written each frame, so there is nothing on the GPU to rebuild when
// either changes.
let torchOffset = TORCH_FORWARD;
let torchSide = TORCH_SIDE;
let torchUniforms = null;

function ensureTorchUniforms() {
  const c = colourToRGB(TORCH_COLOUR);
  torchUniforms = torchUniforms || {
    position: uniform(new THREE.Vector3()),
    colour: uniform(new THREE.Vector3(c.r, c.g, c.b)),
    level: uniform(TORCH_LEVEL),
    sourceRadius: uniform(LIGHT_SOURCE_RADIUS)
  };
  // Off is level 0, not an absent light: the torch is always compiled in, and
  // at level 0 its falloff is zero everywhere, which skips its march entirely.
  // So toggling is this write, never a shader rebuild.
  torchUniforms.level.value = torchOn ? clampLevel(torchLevel) : 0;
  torchUniforms.sourceRadius.value = torchSourceRadius;
  return torchUniforms;
}

// The flame follows the sprite every frame. Nothing else has to happen: the
// distance field is geometry, so a light that moves invalidates none of it.
//
// Held OUT IN FRONT rather than at the sprite's centre. The character is a
// silhouette in the distance field now, so a flame at its own position is inside
// solid geometry - the marcher hits an occluder on the first step and the torch
// shadows itself. TORCH_FORWARD and TORCH_HEIGHT are what carry it clear.
const torchForward = new THREE.Vector3();

function updateTorchPosition() {
  if (!torchOn || !torchUniforms) return;
  // The sprite's own +Z in world space, which is where it is facing - it is
  // yawed to meet the camera every frame, so this is "toward the viewer" without
  // having to re-derive it from currentHeading. Reading the orientation the
  // sprite actually has, rather than the one it was asked for, keeps the flame
  // from lagging a frame behind on a camera spin.
  playerSprite.getWorldDirection(torchForward);
  // Flattened. getWorldDirection carries the character's pitch lean as well as
  // its yaw, and letting that through would bob the flame up and down as the
  // camera tilts - the offset is meant to be horizontal.
  torchForward.y = 0;
  if (torchForward.lengthSq() < 1e-8) torchForward.set(0, 0, 1);
  else torchForward.normalize();

  // The sprite's own right, in world space: X = up cross Z for a right-handed
  // basis, which for a flattened forward is (fz, -fx). Taken from the forward
  // vector rather than from currentHeading for the same reason forward itself is
  // - one source of truth for where the sprite is actually pointing.
  const rightX = torchForward.z, rightZ = -torchForward.x;
  // Bob mirrors when he turns, so the hand holding the torch swaps with him.
  //
  // mirrored() is true when he faces RIGHT (updateSpriteFacing sets repeat.x to
  // -1 for that), and the torch LEADS - it goes to the side he is facing, so
  // facing right puts it on screen-right. Which side reads as correct is a call
  // about the art rather than about the geometry, so torchSide may be negative
  // and bxb.torchside() flips it without a rebuild.
  const side = torchSide * (mirrored(playerSprite) ? 1 : -1);

  const ox = torchForward.x * torchOffset + rightX * side;
  const oz = torchForward.z * torchOffset + rightZ * side;

  torchUniforms.position.value.set(
    playerSprite.position.x + ox,
    playerSprite.position.y + TORCH_HEIGHT,
    playerSprite.position.z + oz);
}

function ensureSunUniforms() {
  sunBayer = sunBayer || createBayerTexture();
  sunSteps = sunSteps || uniform(4);
  sunBias = sunBias || uniform(SURFACE_BIAS_VOXELS);
  sunFadeStart = sunFadeStart || uniform(SHADOW_FADE_START);
  sunEdgeFade = sunEdgeFade || uniform(EDGE_FADE_VOXELS);
  sunAmbient = sunAmbient || uniform(DEFAULT_AMBIENT);
}

function bxbLight(x, y, z) {
  dirLight.position.set(x, y, z);
  // The cone's basis has to be rewritten WITH the direction. A basis left over
  // from the previous sun tilts the penumbra off to one side, which reads as a
  // softening bug rather than as a stale uniform - which is why this goes
  // through writeSunUniforms instead of setting the direction alone.
  if (shadowSun) writeSunUniforms(shadowSun, dirLight.position, sunAngular);
  // The footprint leans toward the sun, so moving the sun moves WHERE the field
  // should be baked, not just how it is lit. Dropping the remembered origin makes
  // the next frame notice and re-origin; without this the grid would keep the
  // previous sun's lean until the player happened to step a block.
  shadowOrigins = [];
  return 'sun toward ' + x.toFixed(2) + ',' + y.toFixed(2) + ',' + z.toFixed(2);
}

// The per-pixel path, as a function rather than only a console command: the ray
// count is baked into the material, so bxb.soften() has to rebuild it, and
// reaching back through globalThis.bxb to do that would make the console the
// owner of state main.js owns.
// Sprites receive the same lights as the terrain. One lit material per ORIGINAL
// material, so the trees' four planes and eight trees still share one.
let spriteShadowMats = [];
function spriteMeshes() {
  return [playerSprite, enemySprite, ...treeMeshes.flatMap(g => g.children)];
}
function restoreSpriteMaterials() {
  for (const m of spriteMeshes()) {
    if (!m.userData.originalMaterial) continue;
    m.material = m.userData.originalMaterial;
    if (m.userData.depthVariants) {
      m.userData.depthVariants = null;
      // The plain material missed any clip transitions while it was parked.
      const top = !!m.userData.drawingOnTop;
      if (m.material.depthTest === top) {
        m.material.depthTest = m.material.depthWrite = !top;
        m.material.needsUpdate = true;
      }
    }
  }
  for (const mat of spriteShadowMats) mat.dispose();
  spriteShadowMats = [];
}
// The characters also get an always-on-top variant for the clip fix in
// sprites.js - compiled here with the rest, so leaning into a wall never
// compiles anything.
const clippingSprites = () => [playerSprite, enemySprite];
// Null until that sprite's card has resolved; the sprite is shaded flat until
// the rebuild that follows.
const spriteNormalFor = m =>
  (m === playerSprite ? bobCard : m === enemySprite ? evilCard : treeCard).normalTex;
function buildSpriteShadows(spriteLight) {
  const made = new Map();
  const pairs = [];
  const variants = new Map();
  for (const m of spriteMeshes()) {
    const base = m.userData.originalMaterial || m.material;
    m.userData.originalMaterial = base;
    if (!made.has(base)) {
      const mat = createSpriteShadowMaterial(base, spriteLight, spriteNormalFor(m));
      mat.depthTest = mat.depthWrite = true;
      made.set(base, mat);
    }
    pairs.push([m, made.get(base)]);
  }
  const mats = [...made.values()];
  for (const m of clippingSprites()) {
    const top = createSpriteShadowMaterial(m.userData.originalMaterial, spriteLight,
                                           spriteNormalFor(m));
    top.depthTest = top.depthWrite = false;
    variants.set(m, { normal: made.get(m.userData.originalMaterial), onTop: top });
    pairs.push([m, top]);
    mats.push(top);
  }
  return { pairs, mats, variants };
}

// Compile the new materials before anything visible uses them, so the swap
// costs nothing and the previous materials keep drawing meanwhile.
//
// Against the REAL mesh in the REAL scene: compileAsync keys its cache on the
// scene it is handed, so compiling stand-ins in a scratch scene produced
// pipelines the actual render never looked up. The new material is put on the
// mesh only for the synchronous part of the call - which is where the render
// object is captured - and taken off again before any frame can draw it. Culling
// is lifted for the same span, or an off-screen sprite would be skipped and
// compile on first sight instead.
//
// One material per frame: building the node graph into WGSL is synchronous
// JavaScript, and doing all four in one frame is a hitch of its own.
const nextFrame = () => new Promise(r => requestAnimationFrame(() => r()));

// Compile everything in the scene that has not been drawn yet, with culling
// lifted - otherwise a sprite first seen on walking into new ground compiles
// its pipeline right then, which is the hitch. Anything already compiled is a
// cache hit, so this is cheap to repeat.
async function warmScene() {
  const lifted = [];
  scene.traverse(o => { if (o.frustumCulled) { lifted.push(o); o.frustumCulled = false; } });
  let pending;
  try { pending = renderer.compileAsync(scene, camera); }
  finally { for (const o of lifted) o.frustumCulled = true; }
  await pending;
}
async function compileOffscreen(pairs) {
  const seen = new Set();
  for (const [mesh, mat] of pairs) {
    if (seen.has(mat)) continue;   // meshes sharing a material share its pipeline
    seen.add(mat);
    const old = mesh.material, culled = mesh.frustumCulled;
    mesh.material = mat;
    mesh.frustumCulled = false;
    let pending;
    try { pending = renderer.compileAsync(mesh, camera, scene); }
    finally { mesh.material = old; mesh.frustumCulled = culled; }
    await pending;
    await nextFrame();
  }
}
let shadowBuild = 0;

function buildPerPixelShadows() {
  ensureShadowGrids();
  const origin = shadowOrigins[0];
  ensureSunUniforms();
  const { node, spriteLight, sun } = createShadowColorNode({
    cascades: shadowCascades,
    sunDirection: dirLight.position,
    rays: sunRays, angular: sunAngular, cone: sunCone, quantise: sunQuantise,
    bayerTex: sunBayer, stepsUniform: sunSteps, biasUniform: sunBias,
    fadeStartUniform: sunFadeStart, edgeFadeUniform: sunEdgeFade,
    ambientUniform: sunAmbient, shadeMode, shadowOnly, sun: sunOn,
    torch: ensureTorchUniforms(),
    // Sprites are not in the field; they are these. Null until the textures have
    // loaded and the atlas is built, which is what the rebuild in updateCards
    // is for.
    cards: cardsReady && cardsOn ? sunCards : null,
    torchCards: cardsReady && cardsOn ? torchCards : null,
    terrainNormal: terrainNormalTex,
    ao: aoOn, aoDistanceUniform: aoDistance, aoOnly
  });
  shadowSun = sun;
  const terrainMat = createShadowMaterial(worldInstancedMesh, node, whiteUniform);
  const sprites = buildSpriteShadows(spriteLight);
  const pairs = [[worldInstancedMesh, terrainMat], ...sprites.pairs];
  const id = ++shadowBuild;
  const swap = () => {
    // A newer build, or shadows turned off, while this one compiled.
    if (id !== shadowBuild) {
      terrainMat.dispose();
      for (const m of sprites.mats) m.dispose();
      return;
    }
    const mesh = worldInstancedMesh;
    if (mesh.material !== mesh.userData.originalMaterial) mesh.material.dispose();
    mesh.material = terrainMat;
    restoreSpriteMaterials();
    for (const m of spriteMeshes()) {
      const v = sprites.variants.get(m);
      if (v) {
        m.userData.depthVariants = v;
        m.material = m.userData.drawingOnTop ? v.onTop : v.normal;
      } else {
        m.material = pairs.find(([pm]) => pm === m)[1];
      }
    }
    spriteShadowMats = sprites.mats;
    warmScene();
  };
  compileOffscreen(pairs).then(swap, err => {
    console.warn('[shadows] background compile failed, swapping anyway', err);
    swap();
  });
  shadowsOn = true;
  updateTorchPosition();
  return `shadows on - ${sunCone ? 'one cone trace' : sunRays + ' sun ray' +
           (sunRays === 1 ? '' : 's')} per texel, ` +
         `12m march cap, grid covering blocks ` +
         `${origin.x}..${origin.x + CHUNK_SIZE - 1} x ${origin.z}..${origin.z + CHUNK_SIZE - 1}` +
         (sunRays === 1 ? '. bxb.soften() for a penumbra.' : '');
}

// Toggling or re-levelling the torch is a uniform write - see ensureTorchUniforms.
// Off costs a branch per fragment; on costs one more march, capped at the
// light's own radius.
function toggleTorch(level = null, flame = null, forward = null) {
  if (flame !== null) torchSourceRadius = Math.max(0, flame);
  if (forward !== null) torchOffset = Math.max(0, forward);
  if (level !== null) torchLevel = clampLevel(level);
  // A flame-size or reach sweep on a lit torch is a CPU-side write - the flame
  // radius is a uniform, the offset is read when the position is set - so
  // neither must be mistaken for a toggle.
  const sizeOnly = level === null && (flame !== null || forward !== null);
  if (!sizeOnly) torchOn = level === null ? !torchOn : torchLevel > 0;
  ensureTorchUniforms();
  if (!torchOn) return 'torch off';
  if (!shadowsOn) return `torch level ${torchLevel} - run bxb.shadows() to see it`;
  // What to expect, so the softening can be checked rather than admired: a
  // penumbra that does NOT shrink as the torch backs away is the sun's cone
  // left in by mistake, not a flame that is too small.
  const at = (h, d) => pointPenumbraMetres(h, d, torchSourceRadius).toFixed(2);
  // Named explicitly because a flame that vanishes when the offset is wound to
  // zero is not a broken light - it is the light sitting inside the character's
  // own silhouette and being occluded by it.
  const held = `held ${torchOffset.toFixed(2)} m in front and ` +
               `${torchSide.toFixed(2)} m to Bob's ` +
               `${(torchSide < 0) === mirrored(playerSprite) ? 'left' : 'right'}` +
               ` while he faces ${mirrored(playerSprite) ? 'right' : 'left'}` +
               ` (bxb.torchside() flips which), ` +
               `${TORCH_HEIGHT.toFixed(2)} m up`;
  return `torch on - ${held}, level ${torchLevel}/${MAX_LIGHT_LEVEL}, reaching ` +
         `${torchLevel} blocks (${lightRadiusMetres(torchLevel).toFixed(1)} m), ` +
         `#${TORCH_COLOUR.toString(16).toUpperCase()}, ` +
         `${(torchSourceRadius * 2).toFixed(2)} m flame. Soft by CONE TRACE, the ` +
         `same marcher the sun uses - the only difference is that its cone opens ` +
         `toward the light instead of staying a fixed angle, so an occluder ` +
         `softens MORE the nearer it sits to the flame and LESS as the torch ` +
         `backs away. Expect a penumbra of ${at(1, 2)} m from an occluder 1 m ` +
         `out with the torch at 2 m, ${at(1, 6)} m with it at 6 m. Contact stays ` +
         `hard either way. bxb.torch(${torchLevel}, ${(torchSourceRadius * 2).toFixed(2)}) ` +
         `to resize the flame.`;
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
      const origin = gridOriginFor(currentMode, player.gridPos, dirLight.position, sunGridBias);
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
    help: 'toggle marched shadows on the terrain (one light, locked per texel)',
    run: () => {
      if (shadowsOn) {
        shadowBuild++;   // drop any build still compiling
        restoreOriginalMaterial(worldInstancedMesh);
        restoreSpriteMaterials();
        shadowsOn = false;
        shadowOrigins = [];
        return 'shadows off';
      }
      return buildPerPixelShadows();
    }
  },
  proxies: {
    help: 'toggle the shadow-casting proxies for sprites, or resize the character box',
    usage: 'bxb.proxies()  |  bxb.proxies(0.75, 2.4)  - width and height in metres',
    run: (w = null, h = null) => {
      if (w !== null) CHARACTER_PROXY.width = Math.max(0.05, w);
      if (h !== null) CHARACTER_PROXY.height = Math.max(0.05, h);
      const sizeOnly = w !== null || h !== null;
      if (sizeOnly) {
        // A resize is not a toggle. The box is remembered by its snapped extent,
        // so changing the size makes every proxy compare unequal next frame and
        // rewrite itself - no invalidation needed beyond setting the number.
        playerProxy.width = enemyProxy.width = CHARACTER_PROXY.width;
        playerProxy.height = enemyProxy.height = CHARACTER_PROXY.height;
      } else {
        proxiesOn = !proxiesOn;
      }
      if (!proxiesOn) return 'proxies off - sprites cast nothing, terrain still does';
      if (!shadowsOn) return 'proxies on - run bxb.shadows() to see them';
      // Which path each proxy is actually on. A silhouette that failed to build
      // falls back to its box silently and still casts, so without this the
      // difference between "the cutout is working" and "the cutout never loaded"
      // is a judgement call about how blocky a shadow looks.
      const sil = playerProxy.silhouettes && playerProxy.silhouettes[0];
      const shape = sil
        ? `CUTOUT ${playerProxy.silhouettes.map(v => v.cols + 'x' + v.rows).join(' / ')}` +
          ` per cascade, facing ${cardMode}`
        : `BOX ${CHARACTER_PROXY.width.toFixed(2)} x ` +
          `${CHARACTER_PROXY.height.toFixed(2)} m (cutout not loaded yet)`;
      const trees = treeProxies.filter(p => p && p.silhouettes).length;
      return `proxies on - characters cast as ${shape}, ` +
             `${trees}/${treeProxies.length} trees as cutouts. ` +
             `Expect the shadow to STEP a texel at a time as you walk, not slide, ` +
             `and to have a GAP between the legs - that gap is the whole ` +
             `difference from a box. bxb.cardface() to compare sun-facing ` +
             `cards against ones locked to a world plane.`;
    }
  },
  cardparity: {
    help: 'run the same rays through the card shader and the CPU reference and diff them',
    run: async (n = 128, line = false) => {
      if (!cardsReady) return 'no cards yet - bxb.shadows() first, then retry';
      const casters = cardCasters();
      for (const c of casters) {
        c.facing = facingToward(dirLight.position.x, dirLight.position.z);
      }
      const near = cullCardsForLight(casters, dirLight.position, Infinity,
                                     sunCards.capacity);
      const count = packCardInstances(near, cardAtlas, cardPack, sunCards.capacity);
      writeCardBindings(sunCards, cardPack, count);

      // Spread over the ground around the player, along the sun. The rays that
      // matter are the ones that graze a card, so the sample box is sized to the
      // sprites rather than to the map.
      const L = dirLight.position.clone().normalize();
      const slope = sunAngular ? Math.tan(sunAngular / 2) : 0.06;
      const samples = [];
      if (line) {
        // A WALK down-sun from the player instead of a scatter. Random points
        // report whether the two agree on average; a line reports WHERE they
        // stop agreeing, which is what a visible edge needs.
        for (let i = 0; i < n; i++) {
          const d = (i / (n - 1)) * 24;
          samples.push({
            from: { x: playerSprite.position.x - L.x * d, y: 0.8,
                    z: playerSprite.position.z - L.z * d },
            dir: { x: L.x, y: L.y, z: L.z }, maxDist: 12
          });
        }
      } else {
        for (let i = 0; i < n; i++) {
          samples.push({
            from: { x: playerSprite.position.x + (Math.random() - 0.5) * 12,
                    y: 0.75 + Math.random() * 2,
                    z: playerSprite.position.z + (Math.random() - 0.5) * 12 },
            dir: { x: L.x, y: L.y, z: L.z },
            maxDist: 12
          });
        }
      }

      const gpu = await runGPUCards(renderer, sunCards, samples, slope);
      if (line) {
        const rows = samples.map((sm, i) => {
          const cpu = cardsVisibility(near, sm.from, sm.dir, 12, slope);
          return `${((i / (n - 1)) * 24).toFixed(1).padStart(5)}m cpu ` +
                 `${cpu.toFixed(3)} gpu ${gpu[i].toFixed(3)}` +
                 (Math.abs(cpu - gpu[i]) > 0.1 ? '  <-- DIVERGE' : '');
        });
        return rows.join(String.fromCharCode(10));
      }
      let worst = 0, worstAt = -1, sum = 0, shadowed = 0;
      for (let i = 0; i < n; i++) {
        const cpu = cardsVisibility(near, samples[i].from, samples[i].dir, 12, slope);
        if (cpu < 0.99) shadowed++;
        const e = Math.abs(cpu - gpu[i]);
        sum += e;
        if (e > worst) { worst = e; worstAt = i; }
      }
      const s0 = samples[worstAt];
      return `${count} cards, ${n} rays, ${shadowed} shadowed on CPU. ` +
             `mean |diff| ${(sum / n).toFixed(4)}, worst ${worst.toFixed(4)} ` +
             `at (${s0.from.x.toFixed(2)}, ${s0.from.y.toFixed(2)}, ` +
             `${s0.from.z.toFixed(2)}) cpu ${cardsVisibility(near, s0.from, s0.dir, 12, slope).toFixed(3)} ` +
             `gpu ${gpu[worstAt].toFixed(3)}. A worst above ~0.05 means the ` +
             `shader and the reference disagree, not that either is noisy.`;
    }
  },
  cards: {
    help: 'analytic sprite shadows - each light gets the cutout turned to face IT',
    usage: 'bxb.cards()  toggles',
    run: () => {
      cardsOn = !cardsOn;
      // Build on demand rather than waiting for a frame. The resolve normally
      // rides the render loop, which does not run while the window is hidden -
      // so asking about the cards from a paused tab would report them as still
      // loading forever, which is a property of the tab and not of the cards.
      if (cardsOn) updateCards();
      // Whether the pass has a card loop in it at all is baked into the kernel,
      // so this is a material rebuild rather than a uniform write - the same
      // trade the torch makes, and for the same reason: off has to be free.
      if (shadowsOn) { buildPerPixelShadows(); }
      if (!cardsOn) return 'sprite cards off - nothing but terrain casts';
      if (!cardsReady) {
        // Say WHY, not just that it has not happened. "Still loading" covers a
        // texture that genuinely has not arrived and one whose image is never
        // going to be readable, and those need opposite fixes.
        const st = pendingCards.map(p => {
          const t = p.getTexture();
          const im = t && t.image;
          return `${p.key}:${p.error ? 'FAILED ' + p.error :
                  !t ? 'no-texture' :
                  !im ? 'no-image' :
                  !im.width ? 'width-0' : 'ready-but-unbuilt'}`;
        }).join(' ');
        return `cards on - no atlas. pending ${pendingCards.length}, ` +
               `built ${cardTypes.length}. ${st || '(nothing pending - the ' +
               'build ran and then something after it failed)'}`;
      }
      const cov = cardTypes.map(t =>
        `${t.key} ${t.card.cols}x${t.card.rows} ${(t.coverage * 100).toFixed(0)}% opaque`
      ).join(', ');
      return `cards on - [${cov}] in a ` +
             `${cardAtlas.width}x${cardAtlas.height} atlas, ${lastCardCount} ` +
             `casting this frame, cap ${MAX_CARDS} per light. Each light tests ` +
             `the cutout turned to face IT, so a torch and the sun disagree ` +
             `about which way Bob is side-on - which is the point, and what a ` +
             `single baked field cannot do. Analytic, so there is no voxel size ` +
             `and no cascade seam.`;
    }
  },
  spikes: {
    help: 'log a CPU breakdown of every frame over 8 ms to the console',
    run: () => {
      spikeLog = !spikeLog;
      return `spike log ${spikeLog ? 'on' : 'off'}`;
    }
  },
  torchside: {
    help: 'which side of Bob the torch is held, and how far - negative swaps sides',
    usage: 'bxb.torchside()  flips  |  bxb.torchside(0.3)  |  bxb.torchside(-0.3)',
    run: (m = null) => {
      // Sign only, or sign and distance. Flipping is the common case - which
      // side reads right depends on the sprite art, not on the geometry, so it
      // is settled by eye rather than derived.
      torchSide = m === null ? -torchSide : m;
      if (!torchOn) return `torch side ${torchSide.toFixed(2)} m - bxb.torch() to light it`;
      const facing = mirrored(playerSprite) ? 'right' : 'left';
      const at = (torchSide < 0) === mirrored(playerSprite) ? 'left' : 'right';
      return `torch ${Math.abs(torchSide).toFixed(2)} m to screen-${at} ` +
             `while Bob faces ${facing}. Turn him round and it should swap; ` +
             `if it does not move at all the sprite is not mirroring.`;
    }
  },
  cardface: {
    help: 'how the sprite cutouts are turned - toward the sun, or locked to a world plane',
    usage: 'bxb.cardface()  cycles sun -> x -> z  |  bxb.cardface("sun")',
    run: (mode = null) => {
      const order = { sun: 'x', x: 'z', z: 'sun' };
      cardMode = mode && order[mode] !== undefined ? mode : order[cardMode];
      // Nothing to invalidate by hand: the facing is part of the remembered box,
      // so sameBox sees it change and every cutout rewrites itself next frame.
      if (cardMode === 'sun') {
        const f = cardFacing();
        return `cutouts face the SUN - card normal along its azimuth ` +
               `(${(-f.uz).toFixed(2)}, ${f.ux.toFixed(2)}). The full ` +
               `silhouette always faces the light, so it cannot thin to a line. ` +
               `bxb.cardface() to lock it to a world plane and see the ` +
               `difference; move the sun with bxb.light() and the cutouts turn ` +
               `with it.`;
      }
      return `cutouts LOCKED to the ${cardMode === 'x' ? 'XY' : 'ZY'} plane. ` +
             `Swing the sun round with bxb.light() and watch the shadow thin ` +
             `toward a line as it goes edge-on - that degeneracy is exactly what ` +
             `the sun-facing mode exists to remove.`;
    }
  },
  soften: {
    help: 'distance-based soft sun. "cone" is one ray, a number is that many over the disc',
    usage: 'bxb.soften("cone")  |  bxb.soften(16)  |  bxb.soften(1) for the hard reference',
    run: async (mode = 'cone', a = sunAngular, steps = 0) => {
      // Two techniques, one seam. Both march the same field from the same origin
      // and both return visibility, so they can be swapped and diffed - which is
      // the whole reason the sampled one was built first even though the cone is
      // sixteen times cheaper.
      if (mode === 'cone') {
        sunCone = true;
        sunRays = 1;
      } else {
        sunCone = false;
        sunRays = Math.max(1, Math.min(64, Math.round(Number(mode) || 16)));
      }
      sunAngular = a;
      ensureSunUniforms();
      // steps = 0 means do not quantise at all, and it is the default. Passing a
      // count opts back into the stepped-and-dithered look.
      sunQuantise = steps > 0;
      if (sunQuantise) sunSteps.value = steps;
      if (shadowSun) writeSunUniforms(shadowSun, dirLight.position, sunAngular);

      // The technique and the ray count are both baked into the kernel, so this
      // is a material swap rather than a uniform write.
      if (shadowsOn) { buildPerPixelShadows(); }

      if (!sunCone && sunRays === 1) {
        return 'hard sun - one ray down the cone axis, which is exactly what ' +
               'bxb.shadows always cast. This is the reference every soft ' +
               'result is judged against.';
      }
      // What to expect, in texels, so the result can be checked rather than
      // admired: a penumbra that does not grow with separation means the cone
      // basis is wrong, not that the angular size is too small.
      const at = s => penumbraTexels(s, a).toFixed(1);
      const expect = `Expect a penumbra of ${at(1)} texels at 1 m separation, ` +
        `${at(4)} at 4 m, ${at(8)} at 8 m. Contact stays hard. If it does not ` +
        `widen with separation the cone basis is wrong, not the angular size.`;
      if (sunCone) {
        return `soft sun by CONE TRACE - one ray, ${a} rad disc, ` +
               `${sunQuantise ? steps + ' quantised steps' : 'smooth (no dither)'}. ` +
               `Analytic rather than sampled, so it is smooth by construction, ` +
               `and ~16x cheaper than sampling. What it gives up is occluder ` +
               `SHAPE: it sees the nearest surface, not how much of the disc that ` +
               `surface covers, so two thin occluders read as one and a shadow ` +
               `through a narrow gap comes out slightly too dark. ` +
               `bxb.soften(16) to diff against the sampled reference. ` + expect;
      }
      return `soft sun by SAMPLING - ${sunRays} rays over a ${a} rad disc, ` +
             `${sunQuantise ? steps + ' quantised steps' : 'no quantisation'}. ` +
             `This is the reference: it resolves ` +
             `occluder shape, at ${sunRays}x the rays. ` + expect;
    }
  },
  ao: {
    help: 'cone-traced ambient occlusion: on/off, reach in metres, or "only" to view it alone',
    usage: 'bxb.ao()  toggles  |  bxb.ao(true, 1.5)  |  bxb.ao("only")',
    run: (a = null, dist = null) => {
      const was = `${aoOn}${aoOnly}`;
      if (a === 'only') aoOnly = !aoOnly;
      else if (a !== null || dist === null) aoOn = a === null ? !aoOn : !!a;
      if (dist !== null) aoDistance.value = Math.max(0.1, Number(dist));
      // On/off and the view are compiled in; the distance is a uniform.
      if (shadowsOn && was !== `${aoOn}${aoOnly}`) buildPerPixelShadows();
      if (!shadowsOn) return 'run bxb.shadows() first';
      return `AO ${aoOn ? '6 cones, ' + aoDistance.value.toFixed(2) + ' m' : 'off'}` +
             `${aoOnly ? ' - showing the AO term alone (white open, black occluded)' : ''}`;
    }
  },
  shadowonly: {
    help: 'show the raw sun visibility term with no albedo or N.L, to judge artifacts',
    run: async () => {
      shadowOnly = !shadowOnly;
      if (!shadowsOn) return 'run bxb.shadows() first';
      buildPerPixelShadows();
      return shadowOnly
        ? 'shadow term only: white is lit, black is occluded. Self-shadowing ' +
          'shows as speckle or stripes on surfaces that should be uniformly ' +
          'white - raise bxb.sunbias() if so.'
        : 'back to full shading';
    }
  },
  ambient: {
    help: 'toggle the ambient floor - off means unlit surfaces are black',
    usage: 'bxb.ambient()  |  bxb.ambient(false)  |  bxb.ambient(0.2) for a level',
    run: (on = null) => {
      ensureSunUniforms();
      const v = on === null ? (sunAmbient.value > 0 ? 0 : DEFAULT_AMBIENT)
              : (on === true ? DEFAULT_AMBIENT : (on === false ? 0 : Number(on)));
      // A uniform, so this is a write rather than a rebuild - it changes how
      // much light a surface gets, not whether the kernel computes one.
      sunAmbient.value = Math.max(0, Math.min(1, v));
      return sunAmbient.value > 0
        ? `ambient ${sunAmbient.value.toFixed(2)}`
        : 'ambient off - anything no light reaches is black. With bxb.daylight(false) ' +
          'too, the torch is the only thing you can see by.';
    }
  },
  daylight: {
    help: 'turn the sun off, leaving ambient plus whatever dynamic lights are lit',
    usage: 'bxb.daylight()  |  bxb.daylight(false)',
    run: (on = !sunOn) => {
      sunOn = !!on;
      if (shadowsOn) { buildPerPixelShadows(); }
      return sunOn
        ? 'sun on'
        : 'sun off - ambient only, plus any dynamic light. The sun march is ' +
          'gone from the kernel entirely, so a torch-lit scene pays for one ' +
          'light rather than two. bxb.torch() if it is too dark to see.';
    }
  },
  torch: {
    help: 'toggle the torch, set its level (0-15), flame size, or how far out it is held',
    usage: 'bxb.torch()  |  bxb.torch(15)  |  bxb.torch(0) off  |  ' +
           'bxb.torch(12, 0.5) flame diameter in m  |  bxb.torch(12, 0.5, 0.45) reach in m',
    run: (level = null, flameDiameter = null, forward = null) =>
      toggleTorch(level, flameDiameter === null ? null : flameDiameter / 2, forward)
  },
  gridbias: {
    help: 'how far the shading footprint leans toward the sun, in blocks',
    usage: 'bxb.gridbias(blocks)  |  bxb.gridbias(0) to centre it on the player',
    run: (blocks = SUN_BIAS_BLOCKS) => {
      sunGridBias = blocks;
      // Takes effect on the next re-origin, which dropping the remembered origin
      // forces on the next frame.
      shadowOrigins = [];
      const half = Math.floor(CHUNK_SIZE / 2);
      const up = (half + blocks) * 1.5, down = (half - blocks) * 1.5;
      return `footprint leans ${blocks} blocks up-sun: ${up.toFixed(1)} m of ` +
             `caster range toward the sun, ${down.toFixed(1)} m away from it. ` +
             `Shadow rays travel TOWARD the sun, so range up-sun is what stops a ` +
             `long shadow being cut off; range down-sun holds nothing that can ` +
             `cast onto anything visible. The cost is that ground down-sun of you ` +
             `falls out of the job list and keeps its last shaded value instead ` +
             `of updating. bxb.gridbias(0) restores the centred footprint.`;
    }
  },
  shade: {
    help: 'how N.L is applied, and the ambient floor - both paths share one formula',
    usage: "bxb.shade('ground'|'lambert'|'flat', ambient?)",
    run: async (mode = 'ground', ambient = DEFAULT_AMBIENT) => {
      const modes = ['ground', 'lambert', 'flat'];
      if (!modes.includes(mode)) return `mode must be one of ${modes.join(', ')}`;
      shadeMode = mode;
      ensureSunUniforms();
      sunAmbient.value = ambient;
      // Ambient is a uniform, the mode is a kernel branch - so only the latter
      // costs a rebuild.
      if (shadowsOn) { buildPerPixelShadows(); }
      if (!shadowsOn) return 'run bxb.shadows() first';
      const el = Math.asin(Math.max(-1, Math.min(1,
        dirLight.position.clone().normalize().y))) * 180 / Math.PI;
      const explain = {
        ground: 'flat ground under open sky reads FULLY LIT at any sun angle, ' +
                'because N.L is normalised against what a horizontal surface ' +
                'receives. The sun angle then sets shadow direction and length ' +
                'rather than overall brightness. Faces turned away still fall to ' +
                'ambient, so nothing goes flat.',
        lambert: 'raw N.L - physically right and DIM at a low sun, because every ' +
                 `up-facing surface is multiplied by sin(${el.toFixed(0)}) = ` +
                 `${Math.sin(el * Math.PI / 180).toFixed(2)}. This is what made ` +
                 'the shaded world look like it was in permanent partial shade.',
        flat: 'no N.L at all - what bxb.shadows() always did. Bright, but an ' +
              'unshadowed wall facing away from the sun reads exactly as bright ' +
              'as one facing it.'
      }[mode];
      return `shade mode '${mode}', ambient ${ambient}. ${explain}`;
    }
  },
  fade: {
    help: 'how a shadow dissolves where its ray runs out of reach, instead of stopping',
    usage: 'bxb.fade(startFraction, edgeVoxels)  |  bxb.fade(1, 0) to see the hard cut',
    run: async (start = SHADOW_FADE_START, edge = EDGE_FADE_VOXELS) => {
      ensureSunUniforms();
      sunFadeStart.value = start;
      sunEdgeFade.value = edge;
      // Both are uniforms, so a sweep is a re-shade rather than a rebuild.
      if (!shadowsOn) return 'run bxb.shadows() first';
      return `shadows fade over the last ${((1 - start) * 100).toFixed(0)}% of the ` +
             `12 m march, and within ${edge} voxels (${(edge * 0.125).toFixed(2)} m) ` +
             `of the grid boundary. This is what stops a long shadow ending on a ` +
             `straight line: past those limits the ray does not know what is out ` +
             `there, and fading is the honest way to say so. bxb.fade(1, 0) turns ` +
             `it off and puts the hard cut back, to confirm that is what you were ` +
             `seeing. Lower start = longer, softer dissolve.`;
    }
  },
  sunbias: {
    help: 'how far a shadow ray starts off the surface, along the face normal, in voxels',
    usage: 'bxb.sunbias(voxels)',
    run: async (v = SURFACE_BIAS_VOXELS) => {
      ensureSunUniforms();
      sunBias.value = v;
      // A uniform, not a rebuild: this is the one knob that has to be found by
      // eye, so a sweep costs a re-shade.
      if (!shadowsOn) return 'run bxb.shadows() first';
      return `ray origin lifted ${v} voxels (${(v * 0.125 * 100).toFixed(1)} cm) ` +
             `along the face normal. Raise until speckle clears, then stop - too ` +
             `much detaches contact shadows from their casters. Along the NORMAL ` +
             `and not the ray, so this is independent of sun angle.`;
    }
  },
  perf: {
    help: 'toggle the frame time graph',
    run: () => perf.toggle() ? 'perf graph on' : 'perf graph off'
  },
  grid: {
    help: 'toggle the boxGrid overlay - both cascades, C1 with C0 subtracted (Alt+G)',
    run: () => toggleBoxGridDebug(
      scene, player.gridPos, worldInstancedMesh, currentMode,
      { sunDirection: dirLight.position, bias: sunGridBias }
    ) || 'boxGrid overlay off'
  },
  stats: {
    help: 'print world, grid and frame statistics',
    run: () => {
      const origin = gridOriginFor(currentMode, player.gridPos, dirLight.position, sunGridBias);
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
    // Goes through bxbLight so the cone basis is rewritten with the direction -
    // see there for why writing the direction alone is not enough now.
    run: (x, y, z) => bxbLight(x, y, z) +
      (shadowsOn ? '' : ' (run bxb.shadows() to see it)')
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