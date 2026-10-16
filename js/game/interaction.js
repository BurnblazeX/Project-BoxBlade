// --- The interaction panel: talk to or fight an enemy, open a container ---
//
// Opens from "look" range (two tiles), and each action button enables only at
// true adjacency, re-checked every frame while it is open (update). The open
// target is game.interactionTarget, null while closed. What a container looks
// like open is the view's: view.showOpen(obj, open), and view.hasMesh(obj).
import { addToInventory, isInInteractRange, pathDistance } from '../core/world.js';
import { rollLootTable } from './objects.js';
import * as UI from './ui.js';

const el = id => document.getElementById(id);

function formatItemName(itemId) {
  return itemId.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

export function createInteraction({ game, player, encounter, view }) {
  function open(target) {
    game.interactionTarget = target;

    el('dialogue-panel').style.display = 'block';
    el('dialogue-name').innerText = target.name;

    // World objects come from createObject() and always include subType/model.
    // Entities (player/enemies) do not, so they are the only valid fight/talk targets.
    const isObject = !!target && typeof target.subType === 'string' && typeof target.model === 'string';
    const isContainer = isObject && target.subType === 'Container';
    const isEnemy = !isObject;
    el('btn-talk').style.display = isEnemy ? 'inline-block' : 'none';
    el('btn-fight').style.display = isEnemy ? 'inline-block' : 'none';
    el('btn-open').style.display = isContainer ? 'inline-block' : 'none';

    updateButtons();
  }

  function close() {
    el('dialogue-panel').style.display = 'none';
    game.interactionTarget = null;
  }

  // Live adjacency gate: the panel opens from "look" range, but each action button
  // only enables at true adjacency, re-checked every frame while the panel is open.
  function updateButtons() {
    const target = game.interactionTarget;
    if (!target) return;
    const inRange = isInInteractRange(player, target);

    for (const id of ['btn-talk', 'btn-fight', 'btn-open']) {
      const btn = el(id);
      if (btn.style.display === 'none') continue;
      btn.disabled = !inRange;
      btn.classList.toggle('disabled', !inRange);
    }
  }

  // Each frame: closes once the player walks out of look-range of the
  // target, otherwise keeps the buttons' enabled state live.
  function update() {
    const target = game.interactionTarget;
    if (!target) return;
    if (pathDistance(player.gridPos, target.gridPos) > 2) close();
    else updateButtons();
  }

  // Stop UI clicks from falling through to the game world
  el('dialogue-panel').addEventListener('pointerdown', (e) => e.stopPropagation());

  el('btn-open').addEventListener('click', () => {
    const obj = game.interactionTarget;
    if (!obj || obj.subType !== 'Container') return;
    if (!isInInteractRange(player, obj)) return;

    if (!obj.looted) {
      const result = obj.lootTable ? rollLootTable(obj.lootTable) : obj.fixedItem;
      if (result) {
        addToInventory(player, result.itemId, result.quantity);
        UI.logChatMessage(`Found ${result.quantity}x ${formatItemName(result.itemId)} in the ${obj.name}.`);
      }
      obj.looted = true;
    } else {
      UI.logChatMessage(`The ${obj.name} is already empty.`);
    }

    obj.state = 'open';
    if (!view.hasMesh(obj)) return;
    // Open for a 2-second rummage (a chest shows it; a barrel has no open look).
    view.showOpen(obj, true);
    setTimeout(() => {
      obj.state = 'closed';
      view.showOpen(obj, false);
      UI.logChatMessage(`Finished checking the ${obj.name}.`);
    }, 2000);
  });

  el('btn-fight').addEventListener('click', () => {
    const target = game.interactionTarget;
    if (!target || typeof target.subType === 'string') return;
    close();
    encounter.startBattle(true);
  });

  return { open, close, update };
}
