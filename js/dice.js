/**
 * Rolls a single die of the specified size.
 * @param {number} sides - Number of sides on the die.
 * @returns {number} The rolled result (1 to sides).
 */
export function rollDie(sides) {
  return Math.floor(Math.random() * sides) + 1;
}

/**
 * Parses dice notation (e.g., "2d6", "1d20") and rolls them.
 * @param {string} notation - The dice string.
 * @returns {Object} { rolls: [number], total: number }
 */
export function rollDice(notation) {
  const match = notation.match(/^(\d+)d(\d+)$/i);
  if (!match) {
    throw new Error(`Invalid dice notation: ${notation}. Expected format like "2d6".`);
  }

  const count = parseInt(match[1], 10);
  const sides = parseInt(match[2], 10);
  
  const rolls = [];
  let total = 0;
  
  for (let i = 0; i < count; i++) {
    const result = rollDie(sides);
    rolls.push(result);
    total += result;
  }
  
  return { rolls, total };
}

/**
 * Rolls a d20 with optional advantage/disadvantage logic.
 * @param {Object} options - { advantage: boolean, disadvantage: boolean }
 * @returns {Object} { rolls: [number], kept: number, advantage: boolean, disadvantage: boolean }
 */
export function rollD20({ advantage = false, disadvantage = false } = {}) {
  // D&D Rule: If you have both advantage and disadvantage, they cancel out.
  if (advantage && disadvantage) {
    advantage = false;
    disadvantage = false;
  }

  const rolls = [rollDie(20)];
  
  if (advantage || disadvantage) {
    rolls.push(rollDie(20));
  }

  let kept = rolls[0];
  if (advantage) {
    kept = Math.max(...rolls);
  } else if (disadvantage) {
    kept = Math.min(...rolls);
  }

  return { rolls, kept, advantage, disadvantage };
}

/**
 * Applies a flat modifier to a roll result object.
 * @param {Object} rollResult - The output from rollDice or rollD20.
 * @param {number} modifier - The flat number to add (or subtract).
 * @returns {Object} A new object with the modifier applied to the final total.
 */
export function applyModifier(rollResult, modifier) {
  // rollD20 uses 'kept', rollDice uses 'total'. 
  // We resolve the base value accordingly.
  const baseValue = rollResult.kept !== undefined ? rollResult.kept : rollResult.total;
  
  return {
    ...rollResult,
    modifier,
    total: baseValue + modifier
  };
}


// ==========================================
// DEV-ONLY SELF TESTS
// ==========================================
if (import.meta.env.DEV) {
  console.log("--- 🎲 DICE ENGINE INITIALIZED 🎲 ---");
  
  console.log("1. rollDie(8):", rollDie(8));
  
  console.log("2. rollDice('3d6'):", rollDice("3d6"));
  
  console.log("3. rollD20 (Normal):", rollD20());
  console.log("4. rollD20 (Advantage):", rollD20({ advantage: true }));
  console.log("5. rollD20 (Disadvantage):", rollD20({ disadvantage: true }));
  console.log("6. rollD20 (Both - Cancels out):", rollD20({ advantage: true, disadvantage: true }));
  
  const rawAttackRoll = rollD20({ advantage: true });
  console.log("7. applyModifier (Attack +4):", applyModifier(rawAttackRoll, 4));
  
  const rawDamageRoll = rollDice("2d6");
  console.log("8. applyModifier (Damage +2):", applyModifier(rawDamageRoll, 2));
  
  console.log("-------------------------------------");
}