// --- The game's state ---
//
// One object, not a module's worth of loose variables: an imported binding
// cannot be reassigned, so state that several modules change has to live
// where they can all reach it. main.js makes one and hands it to the rest.
export function createGameState() {
  return {
    mode: 'explore',              // 'explore' or 'battle'
    // The player's walk: the nodes still to reach, and the tile it leads to.
    path: [],
    moveTargetKey: null,
    // The enemy's battle move, animating (combat.js takeEnemyTurn).
    enemyPath: [],
    // Screen-relative walking keys held down.
    keys: { w: false, a: false, s: false, d: false },
    // Who is in the battle (battle.js keeps the turn order).
    participants: [],
    // Bumped every time battle is entered or exited, so timers scheduled by a
    // previous battle can detect they're stale even if the mode flips back
    // to 'battle' before they fire.
    battleSession: 0,
    // Battle: the tiles this turn's remaining movement reaches.
    reachable: null,
    // Battle: the arena's tiles (world.js enterBattle) and its bounds in blocks.
    arena: null,
    arenaBounds: null,
    // The last click-to-move search that found nothing, as "start>target":
    // not searched again while the mouse is held or hovers on it.
    failedPath: null,
    // The interaction panel's target (interaction.js), null while it is closed.
    interactionTarget: null
  };
}
