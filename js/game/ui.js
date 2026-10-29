export function initUI(onAttack, onEndTurn) {
  document.getElementById('btn-action-attack').addEventListener('click', onAttack);
  document.getElementById('btn-end-turn').addEventListener('click', onEndTurn);
}

// The mode label: 'explore' -> Explore.
export function setModeText(mode) {
  document.getElementById('mode-text').innerText = mode.charAt(0).toUpperCase() + mode.slice(1);
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

export function logChatMessage(message) {
  const log = document.getElementById('chat-log');
  const entry = document.createElement('div');
  entry.className = 'chat-log-entry';
  entry.innerHTML = message;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

export function updateActionResources(turnResources) {
  if (!turnResources) return;
  document.getElementById('res-action').className = `res-box ${turnResources.actionAvailable ? 'res-avail' : 'res-spent'}`;
  document.getElementById('res-bonus').className = `res-box ${turnResources.bonusActionAvailable ? 'res-avail' : 'res-spent'}`;
  document.getElementById('res-spell').className = `res-box ${turnResources.spellAvailable ? 'res-avail' : 'res-spent'}`;
  document.getElementById('res-move').innerText = turnResources.moveRemaining;
}

// The loading screen, while another region loads (main.js loadRegion): its
// name, then progress through the steps - fraction 0..1 and what is happening.
export function showLoading(name) {
  document.getElementById('loading-name').innerText = name;
  setLoadingProgress(0, '');
  document.getElementById('loading-screen').style.display = 'flex';
}
export function setLoadingProgress(fraction, step) {
  document.getElementById('loading-fill').style.width = `${Math.round(Math.max(0, Math.min(1, fraction)) * 100)}%`;
  document.getElementById('loading-step').innerText = step;
}
export function hideLoading() {
  document.getElementById('loading-screen').style.display = 'none';
}
