// --- Pointer and keys: walking, hovering, picking targets ---
//
// Turns the mouse and the game keys into game actions. What is under the
// cursor comes from picker (render/picking.js), the border and dots go to
// markers (render/markers.js), and the sprites are described by sprites:
//   clickable()       a click on one of these is never a walk
//   hoverable()       these light up under the cursor when in look-range
//   targetOf(sprite)  its panel target; undefined for scenery (the tree)
//   highlight(s)      s lit, every other hoverable at its own colour (null: none)
import { World, getVoxelKey, findPath, pathDistance, isStandable } from '../core/world.js';

// What each mode lets the player do.
export const INPUT_RULES = {
  explore: { click: true, keyboard: true },
  battle:  { click: true, keyboard: false }
};

export function createInput({ game, player, interaction, encounter, picker, markers, sprites }) {
  let lastHoveredKey = null;   // the tile the hover last looked at
  let pointerDown = false;     // left button held (explore: drag to re-path)
  let warned = null;           // the tile this click already warned about

  const isWalkingManual = () => game.keys.w || game.keys.a || game.keys.s || game.keys.d;

  // A rejected click says why once - not again for the same tile on release.
  function reject(key, message) {
    if (key === warned) return;
    warned = key;
    console.warn(message);
  }

  // Walk to the tile under the cursor. kind: 'press' and 'release' (a click,
  // which pops the border and says why it was refused) or 'hold' (dragging
  // with the button down in explore, which re-paths quietly).
  function clickToMove(clientX, clientY, kind) {
    if (!INPUT_RULES[game.mode].click) return;
    // Strict lock for Battle mode turns: No interruptions allowed
    if (game.mode === 'battle' && game.path.length > 0) return;

    const tile = picker.tileAt(clientX, clientY);
    if (!tile) return;
    const { gx, gy, gz } = tile;
    const targetKey = getVoxelKey(gx, gy, gz);

    if (game.mode === 'battle' && (!game.reachable || !game.reachable.has(targetKey))) {
      if (kind !== 'hold') reject(targetKey, `Rejected: Tile ${targetKey} is outside speed range or arena bounds.`);
      return;
    }

    // Prevent spamming pathfinder calculations if we are holding the mouse and hovering the same tile
    if (kind === 'hold' && targetKey === game.moveTargetKey && game.path.length > 0) return;

    // A target that just failed from this tile is not searched again while
    // the mouse is held or hovers on it (an unreachable one searches all).
    const failKey = `${getVoxelKey(player.gridPos.x, player.gridPos.y, player.gridPos.z)}>${targetKey}`;
    if (failKey === game.failedPath) return;
    if (!World.has(targetKey)) return;

    const allowDiagonals = game.mode === 'explore';
    const path = findPath(player.gridPos, { x: gx, y: gy, z: gz }, allowDiagonals);
    game.failedPath = path.length ? null : failKey;

    // What is left of this turn's movement, not the whole stat.
    if (game.mode === 'battle' && path.length > player.turnResources.moveRemaining) {
      if (kind !== 'hold') reject(targetKey, 'Rejected: Path length exceeds speed stat.');
      return;
    }
    if (path.length === 0) return;

    game.path = path; // Instant override of the path!
    game.moveTargetKey = targetKey;
    markers.target(tile, kind !== 'hold');
    if (game.mode === 'explore') markers.clearPath(); // No dots in explore mode
    else markers.path(path);
  }

  window.addEventListener('pointerdown', (e) => {
    // A move still waiting for the frame came before this: handle it first, in order.
    flush();
    if (e.target.tagName !== 'CANVAS') return;
    if (e.button !== 0) return;
    warned = null;

    // Intercept clicks on ANY sprite (including the tree) so you can't walk through them
    const hitSprite = picker.objectAt(e.clientX, e.clientY, sprites.clickable());
    if (hitSprite) {
      let target = sprites.targetOf(hitSprite); // undefined for the tree - no panel, by design

      // Check if the target is an Object instead of an Entity
      if (target && target.subType) {
        // Ignore Decor and Lights; only allow interaction with Containers
        if (target.subType !== 'Container') {
          target = undefined;
        }
      }

      if (game.mode === 'explore' && target) {
        if (pathDistance(player.gridPos, target.gridPos) <= 2) {
          interaction.open(target);
        } else {
          console.log(`${target.name} is too far away to interact!`);
        }
      }

      // Consume the click if a sprite was hit
      pointerDown = false;
      game.path = [];
      return;
    }

    pointerDown = true;
    clickToMove(e.clientX, e.clientY, 'press');
  });

  // Released without a walk under way (the press was refused, or the cursor
  // moved): the tile it is released on.
  window.addEventListener('pointerup', (e) => {
    flush();
    if (e.button === 0) pointerDown = false;
    if (e.target.tagName !== 'CANVAS') return;
    if (game.path.length > 0) return;
    if (picker.objectAt(e.clientX, e.clientY, sprites.clickable())) return;
    clickToMove(e.clientX, e.clientY, 'release');
  });

  // Pointer moves are handled once a frame, from the frame loop, with the latest
  // event. A mouse can report far faster than the frame rate (150-190 moves a
  // second in a Firefox profile, some faster still), and each move raycast the
  // terrain - and, with the button held, re-pathed to the tile under the cursor,
  // which moves as the camera follows Bob. That was a steady fps drop while
  // click-walking. Only the newest position matters: the path it sets is read
  // once a frame anyway.
  let pendingPointerMove = null;
  window.addEventListener('pointermove', (e) => { pendingPointerMove = e; });
  function flush() {
    const e = pendingPointerMove;
    if (!e) return;
    pendingPointerMove = null;
    hover(e);
  }

  function hover(e) {
    if (e.target.tagName !== 'CANVAS') {
      document.body.style.cursor = 'default';
      return;
    }

    const isWalkingInBattle = (game.mode === 'battle' && game.path.length > 0);

    // A target in look-range lights up, and the cursor says it can be clicked.
    let lit = null;
    document.body.style.cursor = 'default';
    if (game.mode === 'explore' && !game.interactionTarget) {
      const hovered = picker.objectAt(e.clientX, e.clientY, sprites.hoverable());
      const target = hovered && sprites.targetOf(hovered);
      if (target && pathDistance(player.gridPos, target.gridPos) <= 2) {
        lit = hovered;
        document.body.style.cursor = 'pointer';
      }
    }
    sprites.highlight(lit);

    if (!INPUT_RULES[game.mode].click || isWalkingManual() || isWalkingInBattle) {
      markers.hideBorder();
      lastHoveredKey = null;
      if (!isWalkingInBattle) markers.clearPath();
      return;
    }

    if (pointerDown && game.mode === 'explore') {
      clickToMove(e.clientX, e.clientY, 'hold');
      return;
    }

    if (markers.pulsing()) return;

    const tile = picker.tileAt(e.clientX, e.clientY);
    if (!tile) {
      markers.hideBorder();
      markers.clearPath();
      lastHoveredKey = null;
      return;
    }
    const { gx, gy, gz } = tile;
    const targetKey = getVoxelKey(gx, gy, gz);
    if (targetKey === lastHoveredKey) return;
    lastHoveredKey = targetKey;

    const voxel = World.get(targetKey);
    const inArena = game.mode === 'explore' || !game.arena || game.arena.has(targetKey);
    if (!(voxel && isStandable(gx, gy, gz) && inArena)) {
      markers.hideBorder();
      markers.clearPath();
      return;
    }

    if (game.mode === 'battle') {
      // Every standable tile gets a border and a path; reachable ones are
      // white, the rest red, with the dots turning red where movement runs
      // out. No path at all is red with no dots.
      const reachable = !!(game.reachable && game.reachable.has(targetKey));
      const path = findPath(player.gridPos, { x: gx, y: gy, z: gz }, false);
      const reach = player.turnResources ? player.turnResources.moveRemaining : player.speed;
      markers.hover(tile, reachable ? 'reach' : 'out');
      markers.path(path, reachable ? Infinity : reach);
    } else {
      markers.hover(tile, 'explore');
      markers.clearPath();
    }
  }

  // The game's keys, down: walking, Escape closes the panel, B into battle
  // around Bob's chunk alone (the enemy joins when provoked) and out again.
  function keyDown(key) {
    if (key === 'escape' && game.interactionTarget) interaction.close();
    if (game.keys.hasOwnProperty(key)) game.keys[key] = true;
    if (key === 'b') {
      if (game.mode === 'explore') encounter.startBattle(false);
      else encounter.endBattle();
    }
  }
  function keyUp(key) {
    if (game.keys.hasOwnProperty(key)) game.keys[key] = false;
  }

  return {
    flush, keyDown, keyUp,
    // The hover is looked at afresh on the next move (the click pop ended,
    // the view was reset).
    forgetHover() { lastHoveredKey = null; }
  };
}
