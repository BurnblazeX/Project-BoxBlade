// --- A battle: entering, turns, the enemy's turn, leaving ---
//
// The rules side of battle. What it looks like - the arena shown, the hover
// cleared, the terrain tints - is the view's (main.js), reached only through
// the callbacks it hands in:
//   enterArena(arena, bounds)  show only the arena; the camera centres on it
//   exitArena()                show everything again
//   clearHover()               no hover border, no path dots
//   refreshTints()             the terrain's reach tint (none, at present)
// ticks: the tick clock, for the pauses between turns. worldClock: game time,
// which in battle stands still but for the rounds: 6 s each, counted as each
// begins (core/clock.js).
import { World, getVoxelKey, enterBattle, exitBattle, getReachableVoxels, isStandable, BLOCK_METRES, occupantAt, setOccupantAt, vacateAt } from '../core/world.js';
import { rollInitiativeForParticipants, resetBattleState, turnOrder, currentTurnIndex, getCurrentEntity,
         nextTurn, addParticipant, roundNumber } from './battle.js';
import { performAttack, isDefeated, takeEnemyTurn, isInMeleeRange } from './combat.js';
import { getSpriteWorldPos, vacate } from './movement.js';
import * as UI from './ui.js';
import { ROUND_TICKS, TICK_RATE } from '../core/clock.js';

export function createEncounter({ game, player, enemy, playerSprite, enemySprite, enemyAI, view,
                                  ticks, worldClock }) {

  // Battle: the tiles this turn's remaining movement reaches.
  function refreshReachableTiles() {
    if (game.mode === 'battle') {
      const currentSpeed = player.turnResources ? player.turnResources.moveRemaining : player.speed;
      game.reachable = getReachableVoxels(player.gridPos, currentSpeed);
    }
    // Never tinted: reach is shown by the hover border and path dots instead.
    view.refreshTints();
  }

  // Handles resetting resources and triggering AI
  function onTurnStart(entity) {
    entity.turnResources.actionAvailable = true;
    entity.turnResources.bonusActionAvailable = true;
    entity.turnResources.spellAvailable = true;
    entity.turnResources.moveRemaining = entity.speed;

    if (entity.id === player.id) {
      refreshReachableTiles();
    } else if (entity.id === enemy.id) {
      // GUARD: Only trigger AI if the battle wasn't abruptly ended (or replaced by a new one)
      later(500, processEnemyAI);
    }
  }

  // Runs fn after ms, in ticks - unless the battle it was scheduled in has
  // ended (or been replaced by a new one) by then.
  function later(ms, fn) {
    const session = game.battleSession;
    ticks.after(ms / 1000 * TICK_RATE, () => {
      if (game.mode === 'battle' && game.battleSession === session) fn();
    });
  }

  // The next turn; a new round is 6 s more of game time.
  function advanceTurn() {
    const round = roundNumber;
    nextTurn(game.participants, onTurnStart);
    if (game.mode === 'battle' && roundNumber > round) worldClock.advance(ROUND_TICKS);
  }

  function processEnemyAI() {
    if (game.mode !== 'battle') return;

    const aiDecision = takeEnemyTurn(enemy, player);

    if (aiDecision.action === 'move') {
      // Walked by the frame loop (movement.js followPath), which calls this
      // again once the move ends.
      game.enemyPath = aiDecision.path;
    } else if (aiDecision.action === 'attack') {
      const res = aiDecision.result;
      if (res.hit) {
        UI.logDiceRoll(`<b>${enemy.name}</b> hits for <b>${res.damage}</b>! <br>[Roll: ${res.attackTotal} vs AC ${player.ac}] ` +
                       `(HP: ${player.hp.current}/${player.hp.max})`, 'enemy-turn');
        if (isDefeated(player)) {
          endBattle('GAME OVER. You have been defeated!');
          return;
        }
      } else {
        UI.logDiceRoll(`<b>${enemy.name}</b> misses! <br>[Roll: ${res.attackTotal} vs AC ${player.ac}]`, 'enemy-turn');
      }
      later(1000, advanceTurn);
    } else if (aiDecision.action === 'end') {
      later(500, advanceTurn);
    }
  }

  // Fight: the enemy is pulled into the player's 12 x 12 arena chunk, onto
  // the nearest free tile there.
  function pullIntoArena() {
    const chunkSize = 12;
    const minX = Math.floor(player.gridPos.x / chunkSize) * chunkSize;
    const maxX = minX + chunkSize - 1;
    const minZ = Math.floor(player.gridPos.z / chunkSize) * chunkSize;
    const maxZ = minZ + chunkSize - 1;

    let ex = Math.round(enemySprite.position.x / BLOCK_METRES);
    let ez = Math.round(enemySprite.position.z / BLOCK_METRES);

    if (ex < minX || ex > maxX || ez < minZ || ez > maxZ) {
      vacateAt(getVoxelKey(ex, enemy.gridPos.y, ez), enemy.id);

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
              const checkKey = getVoxelKey(cx, enemy.gridPos.y, cz);
              if (World.get(checkKey) && isStandable(cx, enemy.gridPos.y, cz) && !occupantAt(checkKey)) {
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

    const newKey = getVoxelKey(ex, enemy.gridPos.y, ez);
    if (World.get(newKey)) setOccupantAt(newKey, enemy.id);
  }

  // Into battle, around the player's chunk. withEnemy (Fight): the enemy is
  // pulled in and fights from the start. Without (B): the player alone, the
  // enemy frozen if it stands in the arena and hidden if not - it joins when
  // provoked (attack).
  function startBattle(withEnemy) {
    // Snap player to grid
    player.gridPos.x = Math.round(playerSprite.position.x / BLOCK_METRES);
    player.gridPos.z = Math.round(playerSprite.position.z / BLOCK_METRES);
    playerSprite.position.copy(getSpriteWorldPos(player.gridPos));
    if (withEnemy) pullIntoArena();

    game.mode = 'battle';
    game.battleSession++;
    // Game time moves by rounds now, the first beginning.
    worldClock.live = false;
    worldClock.advance(ROUND_TICKS);
    UI.setModeText(game.mode);
    game.path = [];
    view.clearHover();

    const battleData = enterBattle(player.gridPos);
    game.arena = battleData.arena;
    game.arenaBounds = battleData.bounds;
    view.enterArena(game.arena, game.arenaBounds);

    if (withEnemy) {
      game.participants = [player, enemy];
      enemyAI.isPaused = true;
    } else {
      game.participants = [player];
      // Freeze them if they are in the chunk, hide them if they aren't
      if (game.arena.has(getVoxelKey(enemy.gridPos.x, enemy.gridPos.y, enemy.gridPos.z))) {
        enemyAI.isPaused = true;
        enemySprite.visible = true;
      } else {
        enemySprite.visible = false;
      }
    }
    refreshReachableTiles();

    rollInitiativeForParticipants(game.participants);
    UI.toggleBattleUI(true);
    UI.updateHUD(player);
    UI.updatePartyView(game.participants, player.id);
    UI.updateActionOrder(turnOrder, currentTurnIndex, game.participants);

    const activeEntity = getCurrentEntity(game.participants);
    if (activeEntity) onTurnStart(activeEntity);
  }

  // Back to explore. message: how it ended (victory, defeat), logged; the
  // battle panels then stay up with the log. Without one (B), they close.
  function endBattle(message = null) {
    if (message) UI.logDiceRoll(`<b>${message}</b>`, 'system');
    game.mode = 'explore';
    game.battleSession++;
    worldClock.live = true;
    UI.setModeText(game.mode);
    game.path = [];
    view.clearHover();

    exitBattle();
    game.arena = null;
    view.exitArena();
    refreshReachableTiles();

    if (!isDefeated(enemy)) {
      enemyAI.isPaused = false;
      enemySprite.visible = true;
    }
    resetBattleState(game.participants);
    game.participants = [];

    if (!message) {
      UI.toggleBattleUI(false);
      UI.clearDiceLog();
    }
  }

  // The Attack button.
  function attack() {
    if (game.mode !== 'battle') return;
    const current = getCurrentEntity(game.participants);
    if (current.id !== player.id) return;

    // 1. Check Range
    if (!isInMeleeRange(player, enemy)) {
      UI.logDiceRoll('<i>Target is out of range!</i>', 'system');
      return;
    }

    // 2. Roll Attack
    const res = performAttack(player, enemy);
    if (!res.success) {
      UI.logDiceRoll('<i>You have no action left!</i>', 'system');
      return;
    }

    // 3. Provoke Mechanic: If enemy wasn't in combat, add them now!
    if (!game.participants.some(p => p.id === enemy.id)) {
      game.participants.push(enemy);
      addParticipant(enemy);
      UI.updateActionOrder(turnOrder, currentTurnIndex, game.participants);
      UI.logDiceRoll(`<i>${enemy.name} was provoked and joined the battle!</i>`, 'system');
    }

    // 4. Resolve Hit/Damage
    if (res.hit) {
      UI.logDiceRoll(`<b>${player.name}</b> hits ${enemy.name}! <br>[Roll: ${res.attackTotal} vs AC ${enemy.ac}] <br>Deals <b>${res.damage} damage!</b>`, 'player-turn');
      if (isDefeated(enemy)) {
        enemySprite.visible = false;
        vacate(enemy);
        endBattle('VICTORY! You slew Evil Bob!');
      }
    } else {
      UI.logDiceRoll(`<b>${player.name}</b> misses! <br>[Roll: ${res.attackTotal} vs AC ${enemy.ac}]`, 'player-turn');
    }

    UI.updateHUD(player);
    UI.updateActionResources(player.turnResources);
  }

  // The End Turn button.
  function endTurn() {
    if (game.mode !== 'battle') return;
    if (getCurrentEntity(game.participants)?.id !== player.id) return;

    UI.logDiceRoll(`<i>${player.name} ends their turn.</i>`, 'system');
    advanceTurn();
  }

  return { startBattle, endBattle, attack, endTurn, processEnemyAI, refreshReachableTiles };
}
