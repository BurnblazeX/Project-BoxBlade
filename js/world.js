export const World = new Map();

export function getVoxelKey(x, y, z) {
  return `${Math.round(x)},${Math.round(y)},${Math.round(z)}`;
}

export function createTestArea(width, depth) {
  World.clear();
  for (let x = 0; x < width; x++) {
    for (let z = 0; z < depth; z++) {
      World.set(getVoxelKey(x, 0, z), {
        walkable: true, elevation: 0, occupant: null, triggerId: null
      });
    }
  }
}

// --- ENTITY SCHEMA ---
export function createEntity(config) {
  return {
    id: config.id || "entity_" + Math.random().toString(36).substr(2, 9),
    name: config.name || "Unknown",
    gridPos: { x: config.gridPos.x, y: config.gridPos.y, z: config.gridPos.z },
    stats: config.stats || { STR: 10, DEX: 10, CON: 10, WIS: 10, INT: 10, CHA: 10 },
    speed: config.speed || 6,
    hp: config.hp || { current: 100, max: 100 },
    ac: config.ac || 10,                  
    initiative: config.initiative || null, 
    weaponDie: config.weaponDie || "1d6",
    facing: config.facing || "N",
    mode: config.mode || "explore",
    inventory: config.inventory || []
  };
}

export function addToInventory(entity, itemId, quantity) {
  const existing = entity.inventory.find(i => i.itemId === itemId);
  if (existing) existing.quantity += quantity;
  else entity.inventory.push({ itemId, quantity });
}

// --- DISTANCE & PATHFINDING ---
export function pathDistance(a, b) {
  // Manhattan distance (4-directional)
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
}

export function findPath(start, end, allowDiagonals = false) {
  const openSet = [start];
  const cameFrom = new Map();
  
  const gScore = new Map();
  gScore.set(getVoxelKey(start.x, start.y, start.z), 0);
  
  // Octile heuristic for diagonals, Manhattan for 4-way
  const getH = (a, b) => {
    const dx = Math.abs(a.x - b.x);
    const dz = Math.abs(a.z - b.z);
    return allowDiagonals ? (Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz)) : (dx + dz);
  };
  
  const fScore = new Map();
  fScore.set(getVoxelKey(start.x, start.y, start.z), getH(start, end));
  
  const dirs = [
    {x: 0, y: 0, z: -1, cost: 1}, 
    {x: 0, y: 0, z: 1, cost: 1},  
    {x: -1, y: 0, z: 0, cost: 1}, 
    {x: 1, y: 0, z: 0, cost: 1}   
  ];

  if (allowDiagonals) {
    dirs.push(
      {x: -1, y: 0, z: -1, cost: Math.SQRT2},
      {x: 1, y: 0, z: -1, cost: Math.SQRT2},
      {x: -1, y: 0, z: 1, cost: Math.SQRT2},
      {x: 1, y: 0, z: 1, cost: Math.SQRT2}
    );
  }

  while (openSet.length > 0) {
    let current = openSet[0];
    let lowestIndex = 0;
    let currentKey = getVoxelKey(current.x, current.y, current.z);
    
    for (let i = 1; i < openSet.length; i++) {
      const nodeKey = getVoxelKey(openSet[i].x, openSet[i].y, openSet[i].z);
      if ((fScore.get(nodeKey) ?? Infinity) < (fScore.get(currentKey) ?? Infinity)) {
        current = openSet[i];
        lowestIndex = i;
        currentKey = nodeKey;
      }
    }

    if (current.x === end.x && current.y === end.y && current.z === end.z) {
      const path = [];
      let currNode = current;
      let currKey = getVoxelKey(currNode.x, currNode.y, currNode.z);
      
      while (cameFrom.has(currKey)) {
        path.unshift(currNode);
        currNode = cameFrom.get(currKey);
        currKey = getVoxelKey(currNode.x, currNode.y, currNode.z);
      }
      return path;
    }

    openSet.splice(lowestIndex, 1);

    for (const dir of dirs) {
      const neighbor = { x: current.x + dir.x, y: current.y + dir.y, z: current.z + dir.z };
      const neighborKey = getVoxelKey(neighbor.x, neighbor.y, neighbor.z);
      
      if (currentArena && !currentArena.has(neighborKey)) continue;

      const voxel = World.get(neighborKey);
      if (!voxel || !voxel.walkable || voxel.occupant) continue; 
      
      // Prevent clipping through solid OR occupied corners when moving diagonally
      if (allowDiagonals && dir.cost > 1) {
        const v1 = World.get(getVoxelKey(current.x + dir.x, 0, current.z));
        const v2 = World.get(getVoxelKey(current.x, 0, current.z + dir.z));
        if ((!v1 || !v1.walkable || v1.occupant) || (!v2 || !v2.walkable || v2.occupant)) continue;
      }

      const tentative_gScore = (gScore.get(currentKey) ?? Infinity) + dir.cost;

      if (tentative_gScore < (gScore.get(neighborKey) ?? Infinity)) {
        cameFrom.set(neighborKey, current);
        gScore.set(neighborKey, tentative_gScore);
        fScore.set(neighborKey, tentative_gScore + getH(neighbor, end));
        
        if (!openSet.find(n => n.x === neighbor.x && n.y === neighbor.y && n.z === neighbor.z)) {
          openSet.push(neighbor);
        }
      }
    }
  }
  return []; 
}

// --- BATTLE & RANGE LOGIC ---
export let currentArena = null;

export function attackRange(a, b) {
  // Chebyshev distance (Allows diagonal adjacency)
  return Math.max(Math.abs(a.x - b.x), Math.abs(a.y - b.y), Math.abs(a.z - b.z));
}

export function isInInteractRange(a, b) {
  // Chebyshev adjacency, same shape as melee range - deliberately NOT calling
  // combat.js's isInMeleeRange to keep world.js/objects.js free of combat imports.
  return attackRange(a.gridPos, b.gridPos) <= 1;
}

export function enterBattle(playerGridPos) {
  const arena = new Map();
  const chunkSize = 12;
  
  const cx = Math.floor(playerGridPos.x / chunkSize);
  const cz = Math.floor(playerGridPos.z / chunkSize);
  
  const minX = cx * chunkSize;
  const maxX = minX + chunkSize - 1;
  const minZ = cz * chunkSize;
  const maxZ = minZ + chunkSize - 1; 
  
  for (const [key, voxel] of World.entries()) {
    const [gx, gy, gz] = key.split(',').map(Number);
    if (gx >= minX && gx <= maxX && gz >= minZ && gz <= maxZ) {
      arena.set(key, voxel);
    }
  }
  currentArena = arena;
  return { arena, bounds: { minX, maxX, minZ, maxZ } };
}

export function exitBattle() {
  currentArena = null;
}

export function getReachableVoxels(start, speed) {
  const reachable = new Map();
  const queue = [{ pos: start, cost: 0 }];
  reachable.set(getVoxelKey(start.x, start.y, start.z), 0);
  
  const dirs = [ {x: 0,y: 0,z: -1}, {x: 0,y: 0,z: 1}, {x: -1,y: 0,z: 0}, {x: 1,y: 0,z: 0} ];

  while (queue.length > 0) {
    queue.sort((a, b) => a.cost - b.cost); 
    const current = queue.shift();
    
    if (current.cost >= speed) continue;

    for (const dir of dirs) {
      const neighbor = { x: current.pos.x + dir.x, y: current.pos.y + dir.y, z: current.pos.z + dir.z };
      const nKey = getVoxelKey(neighbor.x, neighbor.y, neighbor.z);
      
      if (currentArena && !currentArena.has(nKey)) continue;
      
      const voxel = World.get(nKey);
      if (!voxel || !voxel.walkable || voxel.occupant) continue; 

      const newCost = current.cost + 1;
      if (!reachable.has(nKey) || newCost < reachable.get(nKey)) {
        reachable.set(nKey, newCost);
        queue.push({ pos: neighbor, cost: newCost });
      }
    }
  }
  return reachable;
}

export function getAbilityModifier(score) {
  return Math.floor((score - 10) / 2);
}