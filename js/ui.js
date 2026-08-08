export function initUI(onAttack, onEndTurn) {
  document.getElementById('btn-action-attack').addEventListener('click', onAttack);
  document.getElementById('btn-end-turn').addEventListener('click', onEndTurn);
}

export function toggleBattleUI(isVisible) {
  document.getElementById('battle-ui-layer').style.display = isVisible ? 'block' : 'none';
}

export function updateHUD(player) {
  document.getElementById('hud-name').innerText = player.name;
  document.getElementById('hud-ac').innerText = `AC: ${player.ac}`;
  document.getElementById('hud-hp-text').innerText = `${player.hp.current} / ${player.hp.max}`;
  
  const pct = Math.max(0, (player.hp.current / player.hp.max) * 100);
  document.getElementById('hud-hp-bar').style.width = `${pct}%`;
}

export function updatePartyView(participants, playerId) {
  const partyList = document.getElementById('party-list');
  const party = participants.filter(p => p.id === playerId);
  partyList.innerHTML = party.map(p => `<div class="party-member">👤<br>${p.name}</div>`).join('');
}

export function updateActionOrder(turnOrder, currentTurnIndex, participants) {
  const container = document.getElementById('action-order-list');
  container.innerHTML = turnOrder.map((t, idx) => {
    const ent = participants.find(p => p.id === t.entityId);
    const activeClass = idx === currentTurnIndex ? 'active-turn' : '';
    const name = ent ? ent.name.substring(0, 5) : '?';
    return `<div class="turn-box ${activeClass}">${name}</div>`;
  }).join('');
}

export function logDiceRoll(message, type = 'system') {
  const log = document.getElementById('dice-log');
  const entry = document.createElement('div');
  entry.className = `dice-log-entry ${type}`;
  entry.innerHTML = message;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

export function clearDiceLog() {
  document.getElementById('dice-log').innerHTML = '';
}

export function updateActionResources(turnResources) {
  if (!turnResources) return;
  document.getElementById('res-action').className = `res-box ${turnResources.actionAvailable ? 'res-avail' : 'res-spent'}`;
  document.getElementById('res-bonus').className = `res-box ${turnResources.bonusActionAvailable ? 'res-avail' : 'res-spent'}`;
  document.getElementById('res-spell').className = `res-box ${turnResources.spellAvailable ? 'res-avail' : 'res-spent'}`;
  document.getElementById('res-move').innerText = turnResources.moveRemaining;
}