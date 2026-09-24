export const World = new Map();

// --- CHUNK / VERTICAL LAYOUT ---
// A chunk is 12x12 blocks in footprint and 12 blocks tall: 3 below ground,
// ground level itself, and 8 above. Ground is y = 0, so the vertical span is
// -3 .. 8 inclusive (12 distinct levels).
// One block is 1.5 m on a side. This is a property of the world, not of the
// renderer, and boxgrid.js needs it without pulling Three.js in - so it lives
// here and render.js aliases it as VOXEL_SIZE.
export const BLOCK_METRES = 1.5;
export const CHUNK_SIZE = 12;
export const GROUND_Y = 0;
// Y_MIN/Y_MAX are the AUTHORABLE vertical range, not a volume that gets filled.
// This is not a sandbox game: there is no terrain below the surface unless a
// map actually goes down there, in which case the author places floors and
// walls explicitly. Nothing is generated underground.
export const Y_MIN = -3;
export const Y_MAX = 8;
export const CHUNK_HEIGHT = Y_MAX - Y_MIN + 1; // 12

export function getVoxelKey(x, y, z) {
  return `${Math.round(x)},${Math.round(y)},${Math.round(z)}`;
}

// --- SOLID / STANDABLE QUERIES ---
// World holds SOLID blocks. An entity's gridPos.y is the y of the block it is
// standing ON, and its sprite sits on that block's top face - the convention
// the renderer and sprite positioning already used implicitly when everything
// lived at y = 0.
//
// "Standable" is therefore a derived property, not a stored one: a block can be
// stood on when it is solid, its own material permits it, and the cell directly
// above is air (headroom). Keeping it derived means digging a block out
// automatically exposes the one beneath, with no flag to keep in sync.

export function getBlock(x, y, z) {
  return World.get(getVoxelKey(x, y, z));
}

export function isSolid(x, y, z) {
  return World.has(getVoxelKey(x, y, z));
}

export function isStandable(x, y, z) {
  const block = getBlock(x, y, z);
  if (!block || !block.walkable) return false;
  return !isSolid(x, y + 1, z); // headroom
}

// Highest standable level in a column, or null if the column has none.
export function getColumnTop(x, z) {
  for (let y = Y_MAX; y >= Y_MIN; y--) {
    if (isStandable(x, y, z)) return y;
  }
  return null;
}

// --- The field worker's copy of the world ---
//
// The worker bakes the field too (fieldWorker.js), so it needs what the bake
// reads: which blocks exist, and which of them are glass - glass goes in the
// field's glass distance, not its opaque one. Keys alone lost the second part,
// and every strip the worker baked put glass back in as rock: glass that left
// C0 and came back returned solid, casting a shadow.
export function worldMirrorEntries(world = World) {
  const out = [];
  for (const [key, block] of world) out.push([key, block ? block.materialId ?? null : null]);
  return out;
}

export function loadWorldMirror(entries, world = World) {
  world.clear();
  for (const [key, materialId] of entries) world.set(key, { materialId });
}

export function createTestArea(width, depth) {
  World.clear();

  // A single ground layer at y = 0. No sub-surface fill: buried blocks would be
  // invisible geometry inflating both the draw and the boxGrid's occupancy set
  // for nothing, since a block is only ever seen or marched against when some
  // face of it is exposed to air.
  for (let x = 0; x < width; x++) {
    for (let z = 0; z < depth; z++) {
      World.set(getVoxelKey(x, GROUND_Y, z), {
        solid: true,
        walkable: true,
        materialId: 'grass', // an id from materials.js
        occupant: null,
        triggerId: null
      });
    }
  }

  addTestElevation();
  // Only when the area holds it: the floor patch replaces ground, so it means
  // nothing past the area's edge. The small test areas the tools build stay
  // exactly as they were.
  if (width > 8 && depth > 17) addReflectionTest();
  if (width > 16 && depth > 13) addGlassTest();
}

// Glass test bed: the thick case - a single 1.5 m cube, about five feet of
// solid glass - on the grass beside the reflection bed, and a window: a wall
// of glass one block thick, 3 wide and 2 tall, in front of the trees at
// z = 8-9, so sprites are seen through it.
//
// Beside them, in a row along z = 13: a 2 x 2 stained-glass window, whose
// panes colour the light that falls through it, and a frosted cube. Kept off
// the tiles the sprite tests use as open ground (e.g. 18, 11).
function addGlassTest() {
  const put = (x, y, z, materialId = 'glass') => World.set(getVoxelKey(x, y, z), {
    solid: true, walkable: true, materialId, occupant: null, triggerId: null
  });
  put(12, 1, 13);
  for (let x = 13; x <= 15; x++) {
    for (let y = 1; y <= 2; y++) put(x, y, 11);
  }
  for (let x = 16; x <= 17; x++) {
    for (let y = 1; y <= 2; y++) put(x, y, 13, 'stainedGlass');
  }
  put(10, 1, 13, 'frostedGlass');
}

// Reflection test bed: a 5x5 patch of polished iron floor with a polished
// marble wall standing along its far edge, so the floor has something to
// mirror and the wall has the floor and the open sky. Both are LabPBR
// materials with high smoothness (materials.js, and their _s textures). The
// floor replaces ground blocks rather than sitting on them, so it stays
// walkable at ground level.
function addReflectionTest() {
  const put = (x, y, z, materialId) => World.set(getVoxelKey(x, y, z), {
    solid: true, walkable: true, materialId, occupant: null, triggerId: null
  });
  for (let x = 3; x <= 7; x++) {
    for (let z = 12; z <= 16; z++) put(x, GROUND_Y, z, 'ironPlate');
  }
  // Wall: 2 blocks tall, 7 long, along x at z = 17 - overhanging the patch
  // by one block each side.
  for (let x = 2; x <= 8; x++) {
    for (let y = 1; y <= 2; y++) put(x, y, 17, 'marble');
  }
  // Standing iron plates either side, running out from the marble's ends at
  // right angles: 2 tall, 3 long, on the grass just outside the patch. Their
  // inner faces look across the floor at each other, so there are mirrors on
  // three axes - the floor (up), the plates (+x and -x) - and the marble
  // between them to catch what they throw.
  for (let z = 14; z <= 16; z++) {
    for (let y = 1; y <= 2; y++) {
      put(1, y, z, 'ironPlate');
      put(9, y, z, 'ironPlate');
    }
  }
}

// Vertical test geometry. Phase A has no slopes, stairs or Jump action, so none
// of this is climbable yet - it exists to give the boxGrid something that can
// actually occlude a light, which a flat single-layer world cannot. The raised
// platform is deliberately left unreachable as a standing demonstration of what
// the Jump action is for.
function addTestElevation() {
  const put = (x, y, z) => World.set(getVoxelKey(x, y, z), {
    solid: true, walkable: true, materialId: 'grass', occupant: null, triggerId: null
  });

  // Wall: 2 blocks tall, 7 long, running along z at x = 20.
  for (let z = 8; z <= 14; z++) {
    for (let y = 1; y <= 2; y++) put(20, y, z);
  }

  // Pillar: 3 blocks tall, isolated, for a long thin shadow.
  for (let y = 1; y <= 3; y++) put(26, y, 20);

  // Raised platform: 4x4, one block up. Standable on top, but not reachable
  // from ground level until step/jump traversal exists.
  for (let x = 28; x <= 31; x++) {
    for (let z = 28; z <= 31; z++) put(x, 1, z);
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
  // Manhattan distance (4-directional). Includes the vertical term, which is
  // inert while all traversal is same-level, but will need a deliberate cost
  // decision once slopes and stairs land.
  return Math.abs(a.x - b.x) + Math.abs(a.y - b.y) + Math.abs(a.z - b.z);
}

// Walkable neighbours of a standing position. Phase A is strictly same-level:
// a neighbouring column is only enterable when its standable surface sits at
// the same y. Walls and pillars are therefore simply impassable, which is
// exactly what gives the lighting work something to cast shadows from.
// Step/slope traversal will widen this to a vertical-delta allowance.
function isEnterable(x, y, z) {
  if (!isStandable(x, y, z)) return false;
  const block = getBlock(x, y, z);
  return !block.occupant;
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
    {x: 0, z: -1, cost: 1},
    {x: 0, z: 1, cost: 1},
    {x: -1, z: 0, cost: 1},
    {x: 1, z: 0, cost: 1}
  ];

  if (allowDiagonals) {
    dirs.push(
      {x: -1, z: -1, cost: Math.SQRT2},
      {x: 1, z: -1, cost: Math.SQRT2},
      {x: -1, z: 1, cost: Math.SQRT2},
      {x: 1, z: 1, cost: Math.SQRT2}
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
      const neighbor = { x: current.x + dir.x, y: current.y, z: current.z + dir.z };
      const neighborKey = getVoxelKey(neighbor.x, neighbor.y, neighbor.z);

      if (currentArena && !currentArena.has(neighborKey)) continue;
      if (!isEnterable(neighbor.x, neighbor.y, neighbor.z)) continue;

      // Prevent clipping through solid OR occupied corners when moving diagonally.
      // Checked at the mover's own level, not a hardcoded y = 0.
      if (allowDiagonals && dir.cost > 1) {
        if (!isEnterable(current.x + dir.x, current.y, current.z)) continue;
        if (!isEnterable(current.x, current.y, current.z + dir.z)) continue;
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

  const cx = Math.floor(playerGridPos.x / CHUNK_SIZE);
  const cz = Math.floor(playerGridPos.z / CHUNK_SIZE);

  const minX = cx * CHUNK_SIZE;
  const maxX = minX + CHUNK_SIZE - 1;
  const minZ = cz * CHUNK_SIZE;
  const maxZ = minZ + CHUNK_SIZE - 1;

  // The arena is the full 12x12x12 chunk, not just its ground layer: everything
  // in the vertical span belongs to it, so elevated geometry stays visible and
  // keeps occluding once the lighting work lands.
  for (const [key, block] of World.entries()) {
    const [gx, gy, gz] = key.split(',').map(Number);
    if (gx >= minX && gx <= maxX && gz >= minZ && gz <= maxZ && gy >= Y_MIN && gy <= Y_MAX) {
      arena.set(key, block);
    }
  }
  currentArena = arena;
  return { arena, bounds: { minX, maxX, minZ, maxZ, minY: Y_MIN, maxY: Y_MAX } };
}

export function exitBattle() {
  currentArena = null;
}

export function getReachableVoxels(start, speed) {
  const reachable = new Map();
  const queue = [{ pos: start, cost: 0 }];
  reachable.set(getVoxelKey(start.x, start.y, start.z), 0);

  const dirs = [ {x: 0, z: -1}, {x: 0, z: 1}, {x: -1, z: 0}, {x: 1, z: 0} ];

  while (queue.length > 0) {
    queue.sort((a, b) => a.cost - b.cost);
    const current = queue.shift();

    if (current.cost >= speed) continue;

    for (const dir of dirs) {
      const neighbor = { x: current.pos.x + dir.x, y: current.pos.y, z: current.pos.z + dir.z };
      const nKey = getVoxelKey(neighbor.x, neighbor.y, neighbor.z);

      if (currentArena && !currentArena.has(nKey)) continue;
      if (!isEnterable(neighbor.x, neighbor.y, neighbor.z)) continue;

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
