// --- Debug settings panel (Alt+V) ---
//
// One place for every renderer knob, instead of forty console commands. The
// panel owns no state: each setting is a get/set pair supplied by main.js, which
// keeps owning the variables, and set() usually just calls the same function the
// bxb command does - so the panel and the console cannot disagree about what a
// setting means, or about what it costs to change.
//
// A setting:
//   { key, label, type: 'toggle' | 'range' | 'select' | 'action',
//     get(), set(value), help?,
//     min, max, step, format?(v)      range
//     options: [value | { value, label }]   select
//     live?: false                    range: apply on release, not while dragging -
//                                     for anything that rebuilds the material
//     run()                           action (a button; no get/set) }
//
// Only coerceSetting is pure; the rest needs a DOM.

export function optionList(def) {
  return (def.options || []).map(o => (typeof o === 'object' ? o : { value: o, label: String(o) }));
}

// What a raw control value becomes before it reaches set(): numbers clamped to
// the range and snapped to the step, toggles to booleans, select values to one
// of the options (matched by string, since a <select> only speaks strings).
export function coerceSetting(def, raw) {
  if (def.type === 'toggle') return !!raw;
  if (def.type === 'range') {
    let v = Number(raw);
    if (!Number.isFinite(v)) v = def.min ?? 0;
    if (def.min !== undefined) v = Math.max(def.min, v);
    if (def.max !== undefined) v = Math.min(def.max, v);
    if (def.step) {
      const base = def.min ?? 0;
      v = base + Math.round((v - base) / def.step) * def.step;
      // Kill the float dust a step like 0.1 leaves behind.
      const places = (String(def.step).split('.')[1] || '').length;
      v = Number(v.toFixed(places));
    }
    return v;
  }
  if (def.type === 'select') {
    const hit = optionList(def).find(o => String(o.value) === String(raw));
    return hit ? hit.value : optionList(def)[0]?.value;
  }
  return raw;
}

function formatValue(def, v) {
  if (def.format) return def.format(v);
  if (typeof v !== 'number') return String(v);
  const places = def.step ? (String(def.step).split('.')[1] || '').length : 2;
  return v.toFixed(places);
}

const STORE_KEY = 'bxb.settings.collapsed';
const POS_KEY = 'bxb.settings.position';
function loadPosition() {
  try { return JSON.parse(localStorage.getItem(POS_KEY) || 'null'); }
  catch { return null; }
}
function savePosition(pos) {
  try { localStorage.setItem(POS_KEY, JSON.stringify(pos)); } catch { /* private window */ }
}
function loadCollapsed() {
  try { return new Set(JSON.parse(localStorage.getItem(STORE_KEY) || '[]')); }
  catch { return new Set(); }
}
function saveCollapsed(set) {
  try { localStorage.setItem(STORE_KEY, JSON.stringify([...set])); } catch { /* private window */ }
}

export function createSettingsPanel({ groups, title = 'Debug settings', parent = document.body }) {
  const el = document.createElement('div');
  el.id = 'settings-panel';
  el.hidden = true;
  // The game listens for pointer events on window; a click on a slider must not
  // also walk Bob to wherever the cursor is over the terrain.
  for (const ev of ['pointerdown', 'pointerup', 'pointermove', 'wheel', 'click']) {
    el.addEventListener(ev, e => e.stopPropagation());
  }

  const head = document.createElement('div');
  head.className = 'sp-head';
  head.innerHTML = `<span>${title}</span><span class="sp-hint">drag · dbl-click resets · Alt+V</span>`;
  el.appendChild(head);

  // --- Dragging, by the header ---
  //
  // Placed by left/top once moved, clamped so at least the header stays on
  // screen, and remembered across reloads.
  function placeAt(x, y) {
    const w = el.offsetWidth || 340, hh = head.offsetHeight || 30;
    x = Math.max(40 - w, Math.min(window.innerWidth - 40, x));
    y = Math.max(0, Math.min(window.innerHeight - hh, y));
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    el.style.right = 'auto';
    return { x, y };
  }
  let placed = loadPosition();
  let drag = null;
  head.addEventListener('pointerdown', e => {
    if (e.button !== 0) return;
    const r = el.getBoundingClientRect();
    drag = { dx: e.clientX - r.left, dy: e.clientY - r.top, id: e.pointerId };
    head.setPointerCapture(e.pointerId);
    head.classList.add('sp-dragging');
    e.preventDefault();   // no text selection while dragging
  });
  head.addEventListener('pointermove', e => {
    if (!drag || e.pointerId !== drag.id) return;
    placed = placeAt(e.clientX - drag.dx, e.clientY - drag.dy);
  });
  const endDrag = e => {
    if (!drag || e.pointerId !== drag.id) return;
    drag = null;
    head.classList.remove('sp-dragging');
    if (placed) savePosition(placed);
  };
  head.addEventListener('pointerup', endDrag);
  head.addEventListener('pointercancel', endDrag);
  // Double-click the header to send it back to the corner.
  head.addEventListener('dblclick', () => {
    placed = null;
    el.style.left = el.style.top = el.style.right = '';
    try { localStorage.removeItem(POS_KEY); } catch { /* private window */ }
  });
  // A window that shrank must not strand the panel off screen.
  window.addEventListener('resize', () => { if (placed) placed = placeAt(placed.x, placed.y); });

  const body = document.createElement('div');
  body.className = 'sp-body';
  el.appendChild(body);

  const status = document.createElement('div');
  status.className = 'sp-status';
  el.appendChild(status);

  const collapsed = loadCollapsed();
  const rows = [];   // { def, sync() }

  function say(msg) {
    if (typeof msg !== 'string' || !msg) return;
    status.textContent = msg.length > 220 ? msg.slice(0, 217) + '...' : msg;
    status.title = msg;
  }

  function apply(def, value) {
    let r;
    try { r = def.type === 'action' ? def.run() : def.set(value); }
    catch (err) { say(`${def.label}: ${err.message}`); console.error(err); return; }
    // Some setters are async (a rebuild waits on compile); refresh once they land
    // too, since a rebuild can change what other settings read back.
    Promise.resolve(r).then(m => { say(m); refresh(); }, err => say(String(err)));
    refresh();
  }

  function buildRow(def) {
    const row = document.createElement('label');
    row.className = `sp-row sp-${def.type}`;
    if (def.help) row.title = def.help;
    const name = document.createElement('span');
    name.className = 'sp-label';
    name.textContent = def.label;
    row.appendChild(name);

    let sync = () => {};
    if (def.type === 'toggle') {
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.addEventListener('change', () => apply(def, coerceSetting(def, box.checked)));
      row.appendChild(box);
      sync = () => { box.checked = !!def.get(); };
    } else if (def.type === 'range') {
      const wrap = document.createElement('span');
      wrap.className = 'sp-range';
      const slider = document.createElement('input');
      slider.type = 'range';
      slider.min = def.min; slider.max = def.max; slider.step = def.step ?? 'any';
      const out = document.createElement('span');
      out.className = 'sp-value';
      const push = () => apply(def, coerceSetting(def, slider.value));
      slider.addEventListener('input', () => {
        out.textContent = formatValue(def, coerceSetting(def, slider.value));
        if (def.live !== false) push();
      });
      if (def.live === false) slider.addEventListener('change', push);
      wrap.append(slider, out);
      row.appendChild(wrap);
      sync = () => {
        // Never yank a slider out from under the hand dragging it.
        if (document.activeElement === slider) return;
        const v = def.get();
        slider.value = v;
        out.textContent = formatValue(def, v);
      };
    } else if (def.type === 'select') {
      const sel = document.createElement('select');
      for (const o of optionList(def)) {
        const opt = document.createElement('option');
        opt.value = String(o.value); opt.textContent = o.label;
        sel.appendChild(opt);
      }
      sel.addEventListener('change', () => apply(def, coerceSetting(def, sel.value)));
      row.appendChild(sel);
      sync = () => { if (document.activeElement !== sel) sel.value = String(def.get()); };
    } else if (def.type === 'action') {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = def.button || 'Run';
      btn.addEventListener('click', () => apply(def));
      row.appendChild(btn);
    }
    rows.push({ def, sync });
    return row;
  }

  for (const g of groups) {
    const sec = document.createElement('details');
    sec.className = 'sp-group';
    sec.open = !collapsed.has(g.title);
    const sum = document.createElement('summary');
    sum.textContent = g.title;
    sec.appendChild(sum);
    sec.addEventListener('toggle', () => {
      if (sec.open) collapsed.delete(g.title); else collapsed.add(g.title);
      saveCollapsed(collapsed);
    });
    for (const def of g.settings) sec.appendChild(buildRow(def));
    body.appendChild(sec);
  }
  parent.appendChild(el);

  function refresh() {
    if (el.hidden) return;
    for (const r of rows) {
      try { r.sync(); } catch { /* a getter reading state that is not built yet */ }
    }
  }

  // Values change from the console and from hotkeys too, so an open panel
  // re-reads them a few times a second rather than trusting its own writes.
  let timer = null;
  function open() {
    el.hidden = false;
    // Applied on open, when the panel has a size to clamp against.
    if (placed) placed = placeAt(placed.x, placed.y);
    refresh();
    timer = timer || setInterval(refresh, 300);
  }
  function close() {
    el.hidden = true;
    clearInterval(timer);
    timer = null;
    // Keyboard focus left on a slider would keep eating the arrow keys.
    if (el.contains(document.activeElement)) document.activeElement.blur();
  }

  return {
    el, open, close, refresh, say,
    get isOpen() { return !el.hidden; },
    toggle() { if (el.hidden) open(); else close(); return !el.hidden; }
  };
}
