import { rollD20, rollDice } from './dice.js';
import { getAbilityModifier, attackRange, getVoxelKey, World, currentArena, findPath } from './world.js';

export function isInMeleeRange(attacker, defender) {
  return attackRange(attacker.gridPos, defender.gridPos) <= 1;
}

export function applyDamage(defender, amount) {
  defender.hp.current = Math.max(0, defender.hp.current - amount);
}

export function isDefeated(entity) {
  return entity.hp.current <= 0;
}

export function performAttack(attacker, defender) {
  if (!attacker.turnResources.actionAvailable) {
    return { success: false, message: "No action available." };
  }

  const strMod = getAbilityModifier(attacker.stats.STR);
  const attackRoll = rollD20();
  const attackTotal = attackRoll.kept + strMod;

  attacker.turnResources.actionAvailable = false; // Consume action

  let hit = false;
  let damageTotal = 0;

  if (attackTotal >= defender.ac) {
    hit = true;
    const damageRoll = rollDice(attacker.weaponDie);
    damageTotal = Math.max(1, damageRoll.total + strMod); // Minimum 1 damage
    applyDamage(defender, damageTotal);
  }

  return { success: true, hit, attackTotal, damage: damageTotal, rollDetails: attackRoll };
}

export function takeEnemyTurn(enemy, player) {
  // 1. Attack if already in range
  if (isInMeleeRange(enemy, player)) {
    if (enemy.turnResources.actionAvailable) {
      return { action: 'attack', result: performAttack(enemy, player) };
    }
    return { action: 'end' };
  }

  // 2. Move if not in range
  if (enemy.turnResources.moveRemaining > 0) {
    // Attack Range is Chebyshev (8-way), so enemy can path to any 8 adjacent squares
    const dirs = [ 
      {x:0,z:-1}, {x:0,z:1}, {x:-1,z:0}, {x:1,z:0},
      {x:-1,z:-1}, {x:1,z:-1}, {x:-1,z:1}, {x:1,z:1}
    ];
    let bestPath = [];
    let shortest = Infinity;

    for (const dir of dirs) {
      const target = { x: player.gridPos.x + dir.x, y: player.gridPos.y, z: player.gridPos.z + dir.z };
      const tKey = getVoxelKey(target.x, target.y, target.z);
      const v = World.get(tKey);
      
      // Ensure target is valid and inside the battle arena
      if (v && v.walkable && !v.occupant && (!currentArena || currentArena.has(tKey))) {
        const path = findPath(enemy.gridPos, target, false); // Movement pathing is still 4-way
        if (path.length > 0 && path.length < shortest) {
          shortest = path.length;
          bestPath = path;
        }
      }
    }

    if (bestPath.length > 0) {
      const cappedPath = bestPath.slice(0, enemy.turnResources.moveRemaining);
      return { action: 'move', path: cappedPath };
    }
  }

  return { action: 'end' };
}