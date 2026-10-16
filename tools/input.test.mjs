import { file, section, ok } from './lib/harness.mjs';

// --- Stand-ins: the window's listeners, the body's cursor ---
// Every test file runs in one process: put back what is replaced here, at the end.
const saved = { window: globalThis.window, document: globalThis.document, warn: console.warn };
const handlers = {};
globalThis.window = { addEventListener: (type, fn) => { handlers[type] = fn; }, innerWidth: 800, innerHeight: 600 };
globalThis.document = { body: { style: {} } };

const { createTestArea, createEntity, getVoxelKey, GROUND_Y } = await import('../js/core/world.js');
const { createGameState } = await import('../js/game/state.js');
const { createInput } = await import('../js/game/input.js');

file('input.test.mjs - one click-to-move for press, hold and release');

let tileUnder = null;               // what the picker reports under the cursor
const warnings = [];
console.warn = msg => warnings.push(msg);

function setup(mode) {
  createTestArea(30, 30);
  const game = createGameState();
  game.mode = mode;
  const player = createEntity({ id: 'p', gridPos: { x: 2, y: GROUND_Y, z: 2 } });
  player.turnResources = { moveRemaining: 6 };
  const marks = [];
  const markers = {
    target: (t, pop) => marks.push(['target', pop]), hover: () => {}, hideBorder: () => {},
    path: () => marks.push(['path']), clearPath: () => {}, pulsing: () => false
  };
  const input = createInput({
    game, player, interaction: { open() {}, close() {} }, encounter: {},
    picker: { tileAt: () => tileUnder, objectAt: () => null },
    markers,
    sprites: { clickable: () => [], hoverable: () => [], targetOf: () => undefined, highlight() {} }
  });
  warnings.length = 0;
  return { game, player, marks, input };
}
const canvas = { tagName: 'CANVAS' };
const press = () => handlers.pointerdown({ target: canvas, button: 0, clientX: 0, clientY: 0 });
const release = () => handlers.pointerup({ target: canvas, button: 0, clientX: 0, clientY: 0 });
const at = (x, z) => ({ gx: x, gy: GROUND_Y, gz: z });

section('battle: a refused click says so once');
{
  const t = setup('battle');
  t.game.reachable = new Set([getVoxelKey(3, GROUND_Y, 2)]);
  tileUnder = at(9, 9);
  press();
  ok('the press warns', warnings.length, 1);
  release();
  ok('the release on the same tile does not again', warnings.length, 1);
  ok('and no walk', t.game.path.length, 0);
  press();
  tileUnder = at(8, 9);
  release();
  ok('released on another refused tile: that one warns', warnings.length, 3);
}

section('battle: the path may use what is left of the turn');
{
  const t = setup('battle');
  const far = getVoxelKey(6, GROUND_Y, 2);
  t.game.reachable = new Set([far]);
  t.player.turnResources.moveRemaining = 2;   // 4 steps away, 2 left
  tileUnder = at(6, 2);
  press();
  ok('refused', t.game.path.length, 0);
  ok('with the reason', warnings[0], 'Rejected: Path length exceeds speed stat.');
  t.player.turnResources.moveRemaining = 4;
  t.game.failedPath = null;
  warnings.length = 0;
  press();
  ok('with enough left, it walks', t.game.path.length, 4);
  ok('the target pops', JSON.stringify(t.marks.find(m => m[0] === 'target')), '["target",true]');
  ok('and the dots show', t.marks.some(m => m[0] === 'path'), true);
}

section('explore: press walks, release only when the press did not');
{
  const t = setup('explore');
  tileUnder = at(5, 5);
  press();
  const path = t.game.path;
  ok('the press sets a walk', path.length > 0, true);
  release();
  ok('the release leaves it be', t.game.path, path);
  t.game.path = [];
  tileUnder = null;
  press();                       // nothing under the cursor: no walk
  tileUnder = at(4, 6);
  release();
  ok('released over a tile, it walks there', t.game.path.length > 0 && t.game.moveTargetKey === getVoxelKey(4, GROUND_Y, 6), true);
  ok('no warnings in explore', warnings.length, 0);
}

globalThis.window = saved.window;
globalThis.document = saved.document;
console.warn = saved.warn;
