import { readFileSync, writeFileSync, mkdirSync, rmSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { createHash } from 'node:crypto';
import { file, section, ok, note } from './lib/harness.mjs';
import { buildAllShaders } from './lib/shaders.mjs';

// Every shader, byte for byte, against the accepted set. A refactor that only
// moves code must leave all of them identical - this is what proves it, where
// the eye can only say nothing looks different.
//
// A change that is MEANT to change shaders is accepted with
//   node tools/run-tests.mjs shaders --update
// which rewrites tools/snapshots/shaders.json. The full WGSL of each run is in
// tools/.shaders/current, and of the last accepted one in tools/.shaders/accepted
// (neither committed) - diff the two to see what moved.
file('shaders.test.mjs - every shader as accepted');

const here = dirname(fileURLToPath(import.meta.url));
const snapFile = join(here, 'snapshots', 'shaders.json');
const out = join(here, '.shaders');
const update = process.argv.includes('--update');

const t0 = performance.now();
const shaders = await buildAllShaders();
note(`${Object.keys(shaders).length} shaders built in ${((performance.now() - t0) / 1000).toFixed(1)} s`);

const hash = s => createHash('sha256').update(s).digest('hex').slice(0, 16);
const dump = (dir, key, stages) => {
  for (const [stage, wgsl] of Object.entries(stages)) {
    const f = join(dir, `${key.replace(/[^\w.-]+/g, '_')}.${stage}.wgsl`);
    writeFileSync(f, wgsl);
  }
};
const current = {};
rmSync(join(out, 'current'), { recursive: true, force: true });
mkdirSync(join(out, 'current'), { recursive: true });
for (const [key, stages] of Object.entries(shaders)) {
  current[key] = Object.fromEntries(Object.entries(stages).map(([s, w]) => [s, hash(w)]));
  dump(join(out, 'current'), key, stages);
}

if (update || !existsSync(snapFile)) {
  mkdirSync(dirname(snapFile), { recursive: true });
  writeFileSync(snapFile, JSON.stringify(current, null, 1) + '\n');
  rmSync(join(out, 'accepted'), { recursive: true, force: true });
  mkdirSync(join(out, 'accepted'), { recursive: true });
  for (const [key, stages] of Object.entries(shaders)) dump(join(out, 'accepted'), key, stages);
  note('snapshot written');
}

section('against the snapshot');
const accepted = JSON.parse(readFileSync(snapFile, 'utf8'));
const changed = [], missing = [], added = [];
for (const [key, stages] of Object.entries(accepted)) {
  if (!current[key]) { missing.push(key); continue; }
  for (const [s, h] of Object.entries(stages)) if (current[key][s] !== h) changed.push(`${key}.${s}`);
}
for (const key of Object.keys(current)) if (!accepted[key]) added.push(key);
ok('none changed', changed.join(', '), '');
ok('none gone', missing.join(', '), '');
ok('none new', added.join(', '), '');
if (changed.length) note('diff tools/.shaders/accepted against tools/.shaders/current');
