// --- Exits: where a region leads to another (World plan P2, step 3) ---
//
// An exit is a box of the cells the party stands on - [x0, y0, z0, x1, y1,
// z1], inclusive, the ground blocks' own keys (player.gridPos) - and where it
// goes: { cells, target: region name, spawn: spawn name in it }. They ride in
// the region's header (core/region.js) beside the spawns, { name: [x, y, z] },
// and the editor places both (tools/editor.js).
//
// Standing on an exit offers the way on (main.js: a prompt, F or its button);
// nothing is forced. Arriving ON an exit - a spawn placed in one, or the far
// side's door - does not offer it back at once: the tracker arms only once the
// party has stood off every exit.
//
// Pure: no DOM, no world.

// The box two corner cells span, either order.
export function exitBox(a, b) {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.min(a[2], b[2]),
          Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.max(a[2], b[2])];
}

export function inExit(exit, x, y, z) {
  const c = exit.cells;
  return x >= c[0] && x <= c[3] && y >= c[1] && y <= c[4] && z >= c[2] && z <= c[5];
}

// The exit at cell (x, y, z), or null. The first one, where boxes overlap.
export function exitAt(exits, x, y, z) {
  for (const e of exits || []) if (inExit(e, x, y, z)) return e;
  return null;
}

// What a region's exit list is without those touching the cells of box b -
// how the editor removes an exit (a right click on any of its cells).
export function exitsWithout(exits, b) {
  return (exits || []).filter(e => {
    const c = e.cells;
    return c[3] < b[0] || c[0] > b[3] || c[4] < b[1] || c[1] > b[4] || c[5] < b[2] || c[2] > b[5];
  });
}

// An exit read from a file, checked: null when it cannot work.
export function validExit(e) {
  if (!e || !Array.isArray(e.cells) || e.cells.length !== 6 || !e.cells.every(Number.isInteger)) return null;
  if (typeof e.target !== 'string' || !e.target) return null;
  return { cells: exitBox(e.cells.slice(0, 3), e.cells.slice(3)), target: e.target,
           spawn: typeof e.spawn === 'string' && e.spawn ? e.spawn : 'start' };
}

// Which exit to offer, frame by frame. update(exits, pos) answers the exit
// the party stands on once armed, else null; disarm() on arrival.
export function createExitTracker() {
  let armed = true;
  return {
    update(exits, pos) {
      const e = exitAt(exits, pos.x, pos.y, pos.z);
      if (!e) { armed = true; return null; }
      return armed ? e : null;
    },
    disarm() { armed = false; },
    get armed() { return armed; }
  };
}
