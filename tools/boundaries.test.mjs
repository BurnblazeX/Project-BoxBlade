import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, resolve, relative, sep } from 'node:path';
import { file, section, ok, note } from './lib/harness.mjs';

// The engine's layers, one folder each under js/, and which may import which.
// main.js wires them together and may import anything.
//   core    the world and every headless calculation - what these tests import
//   game    rules, AI, game UI; reads the world, never the renderer
//   render  meshes, render features, the frame's passes
//   gpu     TSL: shader nodes and compute kernels
//   tools   editor, console, settings, debug, perf - may reach anything
// Without this check the folders are only a convention, and the code drifts
// back to everything importing everything (2026-10-17 architecture review).
file('boundaries.test.mjs - layers import only what they may');

const ALLOWED = {
  core:   ['core'],
  game:   ['core', 'game'],
  render: ['core', 'gpu', 'render'],
  gpu:    ['core', 'gpu'],
  tools:  ['core', 'game', 'render', 'gpu', 'tools']
};
// Core stays headless: no renderer, no shader nodes, no DOM.
const CORE_BANNED = /^three\/(webgpu|tsl)$/;
const DOM = /\b(document|window)\.\w/;

// Known violations, each removed by a later step. An entry that no longer
// occurs fails too, so the list only shrinks.
const EXCEPTIONS = new Set([]);

// Line budgets: a ratchet. A file or function over its budget fails; one well
// under it fails too, asking for the budget to come down to where it now is.
const SLACK = 50;
const FILE_BUDGETS = {
  'main.js': 4292
};
const FUNCTION_BUDGETS = {
  'gpu/shade.js createShadowColorNode': 1579
};

const here = dirname(fileURLToPath(import.meta.url));
const js = join(here, '..', 'js');
const files = readdirSync(js, { recursive: true })
  .filter(f => f.endsWith('.js'))
  .map(f => f.split(sep).join('/'));
const layerOf = f => f.includes('/') ? f.split('/')[0] : null;

// Static, dynamic and side-effect imports, and worker URLs.
const SPEC = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|new URL\(\s*)(['"])([^'"]+)\1/g;

section('every file is in a layer');
const stray = files.filter(f => !layerOf(f) && f !== 'main.js');
ok('only main.js at the root', stray.join(', '), '');
const unknown = files.filter(f => layerOf(f) && !ALLOWED[layerOf(f)]);
ok('no folder outside the layers', unknown.join(', '), '');

section('imports');
const violations = [], seen = new Set();
let edges = 0;
for (const f of files) {
  const layer = layerOf(f);
  if (!layer || !ALLOWED[layer]) continue;
  const src = readFileSync(join(js, f), 'utf8');
  for (const [, , spec] of src.matchAll(SPEC)) {
    if (!spec.startsWith('.')) {
      if (layer === 'core' && CORE_BANNED.test(spec)) violations.push(`${f} -> ${spec}`);
      continue;
    }
    const target = relative(js, resolve(join(js, dirname(f)), spec)).split(sep).join('/');
    if (target.startsWith('..')) continue;   // assets
    edges++;
    const to = layerOf(target);
    if (!to) { violations.push(`${f} -> ${target} (main.js is imported by nothing)`); continue; }
    if (ALLOWED[layer].includes(to)) continue;
    const edge = `${f} -> ${target}`;
    if (EXCEPTIONS.has(edge)) { seen.add(edge); continue; }
    violations.push(edge);
  }
  if (layer === 'core' && DOM.test(src)) violations.push(`${f} touches the DOM`);
}
note(`${edges} imports checked`);
ok('none crosses a boundary', violations.join('; '), '');
ok('every exception still occurs', [...EXCEPTIONS].filter(e => !seen.has(e)).join('; '), '');

// Headless in fact, not only by its imports: every core module loads in node.
section('core loads headless');
const failed = [];
for (const f of files.filter(f => layerOf(f) === 'core')) {
  try { await import(pathToFileURL(join(js, f)).href); }
  catch (e) { failed.push(`${f}: ${e.message}`); }
}
ok('every core module imports in node', failed.join('; '), '');

section('line budgets');
const budget = (label, lines, max) => {
  ok(`${label}: ${lines} lines, budget ${max}`,
     lines > max ? `over by ${lines - max}` : lines < max - SLACK ? `lower the budget to ${lines}` : '', '');
};
for (const [f, max] of Object.entries(FILE_BUDGETS)) {
  budget(f, readFileSync(join(js, f), 'utf8').split('\n').length - 1, max);
}
for (const [key, max] of Object.entries(FUNCTION_BUDGETS)) {
  const [f, name] = key.split(' ');
  const lines = readFileSync(join(js, f), 'utf8').split('\n');
  const start = lines.findIndex(l => new RegExp(`^export (async )?function ${name}\\b`).test(l));
  const end = lines.findIndex((l, i) => i > start && /^}/.test(l));
  budget(name, start < 0 ? -1 : end - start + 1, max);
}
