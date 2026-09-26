// --- The editor's palette ---
//
// Everything in it comes from what is on disk, so a new material or model shows
// up with no code: materials from materials.js MATERIALS (built-in, plus every
// terrain_<id>.png found), models from models.js MODELS (every .bbmodel in
// assets/objects), shapes from world.js BLOCK_SHAPES. Each gets a picture made
// from its own data - a material its texture, a model a front view built from
// its cubes' own pixels, a shape its profile - built once and kept.
import { MATERIALS } from './materials.js';
import { MODELS } from './models.js';
import { BLOCK_SHAPES, hasFacing } from './world.js';
import { textureUrl } from './render.js';
import { facePixel } from './bbmodel.js';

const FACING_ORDER = ['N', 'E', 'S', 'W'];
const FACING_ARROW = { N: '↑', E: '→', S: '↓', W: '←' };
export const SHAPE_ORDER = ['full', ...Object.keys(BLOCK_SHAPES).filter(s => s !== 'full')];

// 'ironPlate' -> 'Iron Plate'; 'decor_wallTorch' -> { category: 'decor', name: 'Wall Torch' }.
export const titleCase = s => s.replace(/([a-z0-9])([A-Z])/g, '$1 $2').replace(/[_-]+/g, ' ')
  .replace(/\b\w/g, c => c.toUpperCase()).trim();
export function modelLabel(name) {
  const i = name.indexOf('_');
  return i > 0 ? { category: name.slice(0, i), name: titleCase(name.slice(i + 1)) }
               : { category: '', name: titleCase(name) };
}
const SHAPE_LABELS = { full: 'Full', halfBottom: 'Half, low', halfTop: 'Half, high', slope: 'Slope',
                       halfSlope: 'Half slope', halfSlopeHigh: 'Half slope, high', stairs: 'Stairs' };
export const shapeLabel = s => SHAPE_LABELS[s] || titleCase(s);

// --- Pictures ---
const pixelated = (w, h, css) => {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  c.style.cssText = `image-rendering:pixelated;display:block;${css || ''}`;
  return c;
};
const loadImage = src => new Promise((resolve, reject) => {
  const img = new Image();
  img.onload = () => resolve(img);
  img.onerror = reject;
  img.src = src;
});

const pictures = new Map();
function picture(key, make) {
  if (!pictures.has(key)) pictures.set(key, make());
  // One canvas can only be in one place: a copy each time it is shown.
  const src = pictures.get(key);
  const c = pixelated(src.width, src.height, src.style.cssText);
  const draw = () => c.getContext('2d').drawImage(src, 0, 0);
  draw();
  src.addEventListener('drawn', draw, { once: true });
  return c;
}

// A material: its albedo texture, 12 x 12.
function materialPicture(m) {
  return picture('m:' + m.id, () => {
    const c = pixelated(12, 12);
    const url = textureUrl(m.texture);
    if (url) loadImage(url).then(img => {
      c.getContext('2d').drawImage(img, 0, 0, 12, 12);
      c.dispatchEvent(new Event('drawn'));
    }).catch(() => {});
    return c;
  });
}

// A shape: its profile seen from the side, climbing to the right (facing E).
function shapePicture(shape) {
  return picture('s:' + shape, () => {
    const n = 24, c = pixelated(n, n);
    const g = c.getContext('2d');
    g.fillStyle = '#c8ccd4';
    const s = BLOCK_SHAPES[shape];
    if (s.slope) {
      g.beginPath();
      g.moveTo(0, n); g.lineTo(n, n);
      g.lineTo(n, n - s.high * n); g.lineTo(0, n - s.low * n);
      g.closePath(); g.fill();
    } else {
      for (const [x0, x1, y0, y1] of s.boxes) g.fillRect(x0 * n, n - y1 * n, (x1 - x0) * n, (y1 - y0) * n);
    }
    return c;
  });
}

// A model: seen from the south (+z), its cubes' south faces in their own
// pixels, far ones first - a front view in the model's own art.
function modelPicture(name, model) {
  return picture('o:' + name, () => {
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const cube of model.cubes) {
      lo = [Math.min(lo[0], cube.min[0]), Math.min(lo[1], cube.min[1])];
      hi = [Math.max(hi[0], cube.max[0]), Math.max(hi[1], cube.max[1])];
    }
    const w = Math.max(1, hi[0] - lo[0]), h = Math.max(1, hi[1] - lo[1]), size = Math.max(w, h);
    const c = pixelated(size, size);
    if (!model.textures.albedo) return c;
    loadImage(model.textures.albedo).then(img => {
      const t = pixelated(img.width, img.height);
      t.getContext('2d').drawImage(img, 0, 0);
      const px = t.getContext('2d').getImageData(0, 0, img.width, img.height).data;
      const out = c.getContext('2d');
      const ox = Math.floor((size - w) / 2), oy = size - h;
      for (const cube of [...model.cubes].sort((a, b) => a.max[2] - b.max[2])) {
        const face = cube.faces[4];               // +z, south
        if (!face) continue;
        const z = cube.max[2] - 1;
        for (let x = cube.min[0]; x < cube.max[0]; x++) for (let y = cube.min[1]; y < cube.max[1]; y++) {
          const [u, v] = facePixel(face, [x, y, z]);
          if (u < 0 || v < 0 || u >= img.width || v >= img.height) continue;
          const i = (v * img.width + u) * 4;
          if (px[i + 3] < 8) continue;
          out.fillStyle = `rgb(${px[i]},${px[i + 1]},${px[i + 2]})`;
          out.fillRect(ox + x - lo[0], oy + hi[1] - 1 - y, 1, 1);
        }
      }
      c.dispatchEvent(new Event('drawn'));
    }).catch(() => {});
    return c;
  });
}

// --- The panel ---
const CSS = `
#editor-panel { position:fixed; left:12px; top:48px; z-index:50; width:340px; display:none;
  max-height:calc(100vh - 72px); overflow-y:auto; overflow-x:hidden; box-sizing:border-box; background:rgba(18,20,24,0.94); color:#e6e8ec;
  font:12px/1.35 system-ui, sans-serif; border:1px solid #3a3f48; border-radius:8px; padding:10px; }
#editor-panel h3 { margin:12px 0 6px; font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:#8fa3bf; }
#editor-panel .top { display:flex; justify-content:space-between; align-items:baseline; }
#editor-panel .top b { font-size:13px; letter-spacing:.06em; }
#editor-panel .hint { color:#7f8896; font-size:11px; }
#editor-panel .sel { display:flex; gap:10px; align-items:center; margin-top:8px; padding:8px;
  background:#22262e; border:1px solid #3a3f48; border-radius:6px; }
#editor-panel .sel canvas { width:44px; height:44px; background:#0e1014; border-radius:4px; }
#editor-panel .sel .what { font-weight:600; }
#editor-panel .sel .facing { font-size:18px; color:#9cf; margin-left:auto; text-align:center; line-height:1; }
#editor-panel .sel .facing small { display:block; font-size:10px; color:#7f8896; }
#editor-panel input { width:100%; box-sizing:border-box; margin-top:8px; padding:5px 7px; border-radius:5px;
  border:1px solid #3a3f48; background:#0e1014; color:#e6e8ec; font:12px system-ui, sans-serif; }
#editor-panel .grid { display:grid; grid-template-columns:repeat(5, minmax(0, 1fr)); gap:6px; }
#editor-panel .tile { cursor:pointer; text-align:center; padding:4px 2px; border-radius:5px; min-width:0;
  border:1px solid transparent; background:#22262e; }
#editor-panel .tile:hover { border-color:#56606e; }
#editor-panel .tile.on { border-color:#6db3ff; background:#1f3550; }
#editor-panel .tile canvas { width:40px; height:40px; max-width:100%; margin:0 auto 3px; border-radius:3px; }
#editor-panel .tile span { display:block; font-size:10px; color:#c2c8d0; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }
#editor-panel .shapes canvas { background:#0e1014; }
#editor-panel .row { display:flex; gap:8px; align-items:center; cursor:pointer; padding:4px; margin-bottom:4px;
  border-radius:5px; border:1px solid transparent; background:#22262e; }
#editor-panel .row:hover { border-color:#56606e; }
#editor-panel .row.on { border-color:#6db3ff; background:#1f3550; }
#editor-panel .row canvas { width:36px; height:36px; background:#0e1014; border-radius:3px; flex:none; }
#editor-panel .row .cat { font-size:10px; color:#7f8896; text-transform:uppercase; letter-spacing:.06em; }
#editor-panel .msg { margin-top:10px; padding:6px 8px; border-radius:5px; background:#2e2a1c; color:#f1d58a; }
#editor-panel details { margin-top:10px; color:#9aa3ae; }
#editor-panel details div { white-space:pre-wrap; font:11px/1.5 ui-monospace, monospace; margin-top:4px; }
#editor-panel .empty { color:#7f8896; font-size:11px; }
`;

// state: the editor's (kind, materialId, shape, model, facing, message).
// pick(change): apply a selection change.
export function createPalette({ state, pick, levelName }) {
  const style = document.createElement('style');
  style.textContent = CSS;
  document.head.appendChild(style);
  const panel = document.createElement('div');
  panel.id = 'editor-panel';
  // The panel is not the world: its pointer events stop here, so the game
  // under it never takes a click on a tile for a click on the ground.
  for (const t of ['pointerdown', 'pointerup', 'pointermove', 'wheel']) {
    panel.addEventListener(t, e => e.stopPropagation());
  }
  document.body.appendChild(panel);
  let filter = '';
  const match = (...labels) => !filter || labels.some(l => l.toLowerCase().includes(filter));
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  function render() {
    const keepFocus = document.activeElement && document.activeElement.id === 'editor-filter';
    const caret = keepFocus ? document.activeElement.selectionStart : 0;
    panel.textContent = '';
    const top = el('div', 'top');
    top.append(el('b', '', 'EDITOR'), el('span', 'hint', `levels/${levelName}.json · Tab to leave`));
    panel.appendChild(top);

    // What is selected, and which way it faces.
    const sel = el('div', 'sel');
    const turns = state.kind === 'model' || hasFacing(state.shape);
    if (state.kind === 'model') {
      const model = MODELS.get(state.model);
      const lab = modelLabel(state.model || '');
      if (model) sel.appendChild(modelPicture(state.model, model));
      const t = el('div');
      t.append(el('div', 'what', lab.name), el('div', 'hint', 'model' + (lab.category ? ` · ${lab.category}` : '')));
      sel.appendChild(t);
    } else {
      const m = MATERIALS.find(x => x.id === state.materialId);
      if (m) sel.appendChild(materialPicture(m));
      const t = el('div');
      t.append(el('div', 'what', titleCase(state.materialId)), el('div', 'hint', shapeLabel(state.shape)));
      sel.appendChild(t);
    }
    if (turns) {
      const f = FACING_ORDER[state.facing];
      const d = el('div', 'facing', FACING_ARROW[f]);
      d.appendChild(el('small', '', `${f} · Q/E`));
      sel.appendChild(d);
    }
    panel.appendChild(sel);

    const input = el('input');
    input.id = 'editor-filter';
    input.placeholder = 'Filter materials and models…';
    input.value = filter;
    input.oninput = () => { filter = input.value.trim().toLowerCase(); render(); };
    panel.appendChild(input);
    if (keepFocus) { input.focus(); input.setSelectionRange(caret, caret); }

    // Materials.
    panel.appendChild(el('h3', '', `Materials (${MATERIALS.length})`));
    const mats = el('div', 'grid');
    for (const m of MATERIALS) {
      if (!match(m.id, titleCase(m.id))) continue;
      const on = state.kind === 'block' && state.materialId === m.id;
      const tile = el('div', 'tile' + (on ? ' on' : ''));
      tile.title = m.id + (m.glass ? ' (glass)' : '');
      tile.append(materialPicture(m), el('span', '', titleCase(m.id)));
      tile.onclick = () => pick({ kind: 'block', materialId: m.id });
      mats.appendChild(tile);
    }
    panel.appendChild(mats.children.length ? mats : el('div', 'empty', 'no material matches'));

    // Shapes.
    panel.appendChild(el('h3', '', 'Shape'));
    const shapes = el('div', 'grid shapes');
    for (const s of SHAPE_ORDER) {
      const on = state.kind === 'block' && state.shape === s;
      const tile = el('div', 'tile' + (on ? ' on' : ''));
      tile.title = s + (hasFacing(s) ? ' - turns with Q/E' : '');
      tile.append(shapePicture(s), el('span', '', shapeLabel(s)));
      tile.onclick = () => pick({ kind: 'block', shape: s });
      shapes.appendChild(tile);
    }
    panel.appendChild(shapes);

    // Models.
    panel.appendChild(el('h3', '', `Models (${MODELS.size})`));
    let shown = 0;
    for (const [name, model] of MODELS) {
      const lab = modelLabel(name);
      if (!match(name, lab.name, lab.category)) continue;
      const on = state.kind === 'model' && state.model === name;
      const row = el('div', 'row' + (on ? ' on' : ''));
      row.title = name + (model.cubes.some(c => c.light) ? ' - emits light' : '');
      const t = el('div');
      t.append(el('div', '', lab.name + (model.cubes.some(c => c.light) ? ' ☀' : '')),
               el('div', 'cat', lab.category || 'model'));
      row.append(modelPicture(name, model), t);
      row.onclick = () => pick({ kind: 'model', model: name });
      panel.appendChild(row);
      shown++;
    }
    if (!shown) panel.appendChild(el('div', 'empty', MODELS.size ? 'no model matches' : 'no .bbmodel in assets/objects'));

    if (state.message) panel.appendChild(el('div', 'msg', state.message));
    const help = el('details');
    help.appendChild(el('summary', '', 'Controls'));
    help.appendChild(el('div', '',
      'LMB           place on the face\nRMB           remove\nShift+LMB drag  fill a box\n' +
      'Shift+RMB drag  clear a box\nRMB drag      look\nWASD, Space   fly  (Shift: fast)\n' +
      'L-Ctrl or C   down\n' +
      'Q / E         turn\nCtrl+S        save'));
    panel.appendChild(help);
  }

  return {
    el: panel, render,
    show(on) { panel.style.display = on ? 'block' : 'none'; if (on) render(); }
  };
}
