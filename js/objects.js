// --- OBJECT SCHEMA ---
// Objects are a distinct type from Voxel and Entity: stationary/interactable
// world dressing (trees, chests, barrels) with no stats, HP, or turn resources.

export function createObject(config) {
  return {
    id: config.id,
    name: config.name,
    subType: config.subType, // "Decor", "Container", or "Light"
    model: config.model,     // e.g., "tree", "chest", "barrel"
    gridPos: config.gridPos,
    blocking: config.blocking !== undefined ? config.blocking : true,
    materialId: null,
    state: "closed",         // "closed" or "open"
    looted: false,
    lootTable: config.lootTable || null,
    fixedItem: config.fixedItem || null
  };
}

// Weighted-random pick across the table's entries, then a flat quantity roll
// between that entry's minQty/maxQty. Not a d20/dice-notation roll - loot
// weighting isn't a D&D mechanic, so it doesn't go through dice.js.
export function rollLootTable(table) {
  const totalWeight = table.reduce((sum, entry) => sum + entry.weight, 0);
  let roll = Math.random() * totalWeight;

  let chosen = table[table.length - 1];
  for (const entry of table) {
    if (roll < entry.weight) {
      chosen = entry;
      break;
    }
    roll -= entry.weight;
  }

  const quantity = Math.floor(Math.random() * (chosen.maxQty - chosen.minQty + 1)) + chosen.minQty;
  return { itemId: chosen.itemId, quantity };
}
