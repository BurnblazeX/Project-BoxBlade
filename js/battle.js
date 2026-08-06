import { rollD20 } from './dice.js';
import { getAbilityModifier } from './world.js';

export let turnOrder = [];
export let currentTurnIndex = 0;
export let roundNumber = 1;

export function rollInitiativeForParticipants(participants) {
  participants.forEach(entity => {
    const dexMod = getAbilityModifier(entity.stats.DEX);
    entity.initiative = rollD20().kept + dexMod;
    
    // PHASE 8: Initialize Action Economy
    entity.turnResources = {
      actionAvailable: true,
      bonusActionAvailable: true,
      spellAvailable: true,
      moveRemaining: entity.speed
    };
  });

  turnOrder = participants.map(entity => ({
    entityId: entity.id,
    initiative: entity.initiative,
    tieBreaker: 0
  }));

  let hasTies = true;
  while (hasTies) {
    hasTies = false;
    for (let i = 0; i < turnOrder.length; i++) {
      for (let j = i + 1; j < turnOrder.length; j++) {
        if (turnOrder[i].initiative === turnOrder[j].initiative && turnOrder[i].tieBreaker === turnOrder[j].tieBreaker) {
          hasTies = true;
          turnOrder[i].tieBreaker = rollD20().kept;
          turnOrder[j].tieBreaker = rollD20().kept;
        }
      }
    }
  }

  turnOrder.sort((a, b) => {
    if (b.initiative === a.initiative) return b.tieBreaker - a.tieBreaker;
    return b.initiative - a.initiative;
  });

  console.log(`[Battle] Initiative rolled! Round ${roundNumber} begins:`, turnOrder);
}

export function getCurrentEntity(participants) {
  if (turnOrder.length === 0) return null;
  const currentId = turnOrder[currentTurnIndex].entityId;
  return participants.find(p => p.id === currentId);
}

export function nextTurn(participants, onTurnStart) {
  currentTurnIndex++;
  
  if (currentTurnIndex >= turnOrder.length) {
    currentTurnIndex = 0;
    roundNumber++;
    console.log(`[Battle] --- Round ${roundNumber} ---`);
  }
  
  const current = getCurrentEntity(participants);
  console.log(`[Battle] Turn: ${current.name}`);
  
  if (onTurnStart && current) {
    onTurnStart(current);
  }
}

export function resetBattleState(participants) {
  turnOrder = [];
  currentTurnIndex = 0;
  roundNumber = 1;
  participants.forEach(p => p.initiative = null);
}