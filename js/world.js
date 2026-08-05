export const World = new Map();

export function getVoxelKey(x, y, z) {
  return `${x},${y},${z}`;
}

export function createTestArea(width, depth) {
  World.clear();
  const halfWidth = Math.floor(width / 2);
  const halfDepth = Math.floor(depth / 2);

  for (let x = -halfWidth; x <= halfWidth; x++) {
    for (let z = -halfDepth; z <= halfDepth; z++) {
      const y = 0; 
      World.set(getVoxelKey(x, y, z), {
        walkable: true,
        elevation: y,
        occupant: null,
        triggerId: null
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
    facing: config.facing || "N",
    mode: config.mode || "explore"
  };
}

// --- DISTANCE & PATHFINDING ---
export function pathDistance(a, b) {
  // Manhattan distance (4-directional)
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
}

export function findPath(start, end) {
  const openSet = [start];
  const cameFrom = new Map();
  
  const gScore = new Map();
  gScore.set(getVoxelKey(start.x, start.y, start.z), 0);
  
  const fScore = new Map();
  fScore.set(getVoxelKey(start.x, start.y, start.z), pathDistance(start, end));
  
  const dirs = [
    {x: 0, y: 0, z: -1}, 
    {x: 0, y: 0, z: 1},  
    {x: -1, y: 0, z: 0}, 
    {x: 1, y: 0, z: 0}   
  ];

  while (openSet.length > 0) {
    let current = openSet[0];
    let lowestIndex = 0;
    let currentKey = getVoxelKey(current.x, current.y, current.z);
    
    for (let i = 1; i < openSet.length; i++) {
      const nodeKey = getVoxelKey(openSet[i].x, openSet[i].y, openSet[i].z);
      // CRITICAL FIX: Use ?? instead of || so 0 isn't treated as Infinity
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
      const voxel = World.get(neighborKey);
      
      if (!voxel || !voxel.walkable || voxel.occupant) {
        continue; 
      }

      // CRITICAL FIX: Use ?? instead of ||
      const tentative_gScore = (gScore.get(currentKey) ?? Infinity) + 1;

      if (tentative_gScore < (gScore.get(neighborKey) ?? Infinity)) {
        cameFrom.set(neighborKey, current);
        gScore.set(neighborKey, tentative_gScore);
        fScore.set(neighborKey, tentative_gScore + pathDistance(neighbor, end));
        
        if (!openSet.find(n => n.x === neighbor.x && n.y === neighbor.y && n.z === neighbor.z)) {
          openSet.push(neighbor);
        }
      }
    }
  }
  return []; 
}