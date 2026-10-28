import * as THREE from 'three/webgpu';
import { file, section, ok } from './lib/harness.mjs';
import { occupantAt, setOccupantAt } from '../js/core/world.js';

// --- A stand-in DOM: just what ui.js and interaction.js touch ---
// Every test file runs in one process: put back what is replaced here, at the end.
const savedDocument = globalThis.document;
const els = new Map();
const element = () => ({ style: {}, classList: { toggle() {}, add() {}, remove() {} }, innerText: '',
                         innerHTML: '', children: [], appendChild() {}, removeChild() {},
                         addEventListener() {} });
const el = id => els.get(id) || (els.set(id, element()), els.get(id));
globalThis.document = { getElementById: el, createElement: element };

const { createTestArea, createEntity, World, getVoxelKey, GROUND_Y, CHUNK_SIZE } = await import('../js/core/world.js');
const { createGameState } = await import('../js/game/state.js');
const { createEncounter } = await import('../js/game/encounter.js');
const { getSpriteWorldPos, occupy } = await import('../js/game/movement.js');
const { createTickClock, createWorldClock, ROUND_TICKS } = await import('../js/core/clock.js');

file('encounter.test.mjs - into battle and out, by B and by Fight');

function setup(enemyAt) {
  createTestArea(30, 30);
  const game = createGameState();
  const player = createEntity({ id: 'p', name: 'Bob', gridPos: { x: 2, y: GROUND_Y, z: 2 } });
  const enemy = createEntity({ id: 'e', name: 'Evil Bob', gridPos: { x: enemyAt.x, y: GROUND_Y, z: enemyAt.z } });
  occupy(player); occupy(enemy);
  const sprite = e => ({ position: getSpriteWorldPos(e.gridPos), visible: true });
  const playerSprite = sprite(player), enemySprite = sprite(enemy);
  const enemyAI = { isPaused: false };
  const calls = [];
  const view = {
    enterArena: (arena, bounds) => calls.push(['enter', bounds]),
    exitArena: () => calls.push(['exit']),
    clearHover: () => calls.push(['clearHover']),
    refreshTints: () => {}
  };
  const ticks = createTickClock(), worldClock = createWorldClock(9);
  const encounter = createEncounter({ game, player, enemy, playerSprite, enemySprite, enemyAI, view,
                                      ticks, worldClock });
  return { game, player, enemy, enemySprite, enemyAI, calls, encounter, ticks, worldClock };
}
const inChunk = (p, b) => p.x >= b.minX && p.x <= b.maxX && p.z >= b.minZ && p.z <= b.maxZ;

section('B: alone, the enemy outside the arena');
{
  const t = setup({ x: 25, z: 25 });
  t.game.path = [{ x: 3, y: 0, z: 2 }];
  t.encounter.startBattle(false);
  ok('battle', t.game.mode, 'battle');
  ok('a new session', t.game.battleSession, 1);
  ok('the label says so', el('mode-text').innerText, 'Battle');
  ok('the walk is dropped', t.game.path.length, 0);
  ok('the player alone', t.game.participants.map(p => p.id).join(), 'p');
  ok('the enemy, outside, is hidden', t.enemySprite.visible, false);
  ok('and keeps wandering', t.enemyAI.isPaused, false);
  ok('the arena is the chunk', t.game.arena.size > 0 && t.calls[1][0] === 'enter', true);
  ok('the panels are up', el('battle-ui-layer').style.display, 'block');
  ok('reach is worked out', t.game.reachable instanceof Map || t.game.reachable instanceof Set, true);

  t.encounter.endBattle();
  ok('explore again', t.game.mode, 'explore');
  ok('another session', t.game.battleSession, 2);
  ok('no participants', t.game.participants.length, 0);
  ok('no arena', t.game.arena, null);
  ok('the enemy shows again', t.enemySprite.visible, true);
  ok('the panels close', el('battle-ui-layer').style.display, 'none');
  ok('everything shown', t.calls.some(c => c[0] === 'exit'), true);
}

section('B: the enemy inside the arena is frozen, not hidden');
{
  const t = setup({ x: 6, z: 6 });
  t.encounter.startBattle(false);
  ok('still visible', t.enemySprite.visible, true);
  ok('frozen', t.enemyAI.isPaused, true);
  ok('not yet fighting', t.game.participants.length, 1);
  t.encounter.endBattle();
  ok('wanders again after', t.enemyAI.isPaused, false);
}

section('Fight: the enemy pulled into the arena');
{
  const t = setup({ x: 25, z: 25 });
  t.encounter.startBattle(true);
  const bounds = t.calls.find(c => c[0] === 'enter')[1];
  ok('arena is the player\'s chunk', bounds.minX === 0 && bounds.maxX === CHUNK_SIZE - 1, true);
  ok('the enemy is inside it', inChunk(t.enemy.gridPos, bounds), true);
  ok('and holds its tile there', occupantAt(getVoxelKey(t.enemy.gridPos.x, t.enemy.gridPos.y, t.enemy.gridPos.z)), 'e');
  ok('its old tile is free', occupantAt(getVoxelKey(25, GROUND_Y, 25)), null);
  ok('both fight', t.game.participants.map(p => p.id).sort().join(), 'e,p');
  ok('the enemy stops wandering', t.enemyAI.isPaused, true);

  // A battle that ends with a result keeps its panels up with the log.
  t.encounter.endBattle('VICTORY! You slew Evil Bob!');
  ok('explore again', t.game.mode, 'explore');
  ok('the panels stay up', el('battle-ui-layer').style.display, 'block');
}
section('game time in battle: 6 s a round');
{
  const t = setup({ x: 25, z: 25 });
  const start = t.worldClock.ticks;
  t.encounter.startBattle(false);
  ok('the first round begins: 6 s', t.worldClock.ticks - start, ROUND_TICKS);
  ok('ticks no longer move it', (t.worldClock.tick(), t.worldClock.ticks - start), ROUND_TICKS);
  t.encounter.endTurn();
  ok('alone, ending the turn begins round 2', t.worldClock.ticks - start, 2 * ROUND_TICKS);
  t.encounter.endBattle();
  t.worldClock.tick();
  ok('explore: the ticks move it again', t.worldClock.ticks - start, 2 * ROUND_TICKS + 1);
}

World.clear();
globalThis.document = savedDocument;
