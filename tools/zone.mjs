// Zones on disk (js/core/zone.js), from the command line.
//
//   node tools/zone.mjs info <file>            what a level holds, and its sizes
//   node tools/zone.mjs dump <file> [out.json] a level as readable JSON, a row a block
//   node tools/zone.mjs pack <file> <out.bxz>  any level (v1 .json, a dump, a .bxz) to a .bxz
//
// A .bxz is binary; dump is how to read or diff one, and pack takes the dump
// back. Packing a v1 .json is the migration: bounds from its blocks, the name
// from the output file's.

import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';
import { decodeZone, encodeZone, migrateLevel, levelRows, packZone } from '../js/core/zone.js';
import { World } from '../js/core/world.js';

async function read(file) {
  const bytes = readFileSync(file);
  return file.endsWith('.json') ? JSON.parse(bytes.toString('utf8')) : decodeZone(new Uint8Array(bytes));
}

// One block row, spawn, exit or model a line: a diff shows the cells that moved.
function readable(level) {
  const list = rows => rows.length ? `[\n    ${rows.map(r => JSON.stringify(r)).join(',\n    ')}\n  ]` : '[]';
  const { blocks, models, exits, ...head } = level;
  const lines = Object.entries(head).map(([k, v]) => `  ${JSON.stringify(k)}: ${JSON.stringify(v)}`);
  lines.push(`  "exits": ${list(exits)}`, `  "models": ${list(models)}`, `  "blocks": ${list(blocks)}`);
  return `{\n${lines.join(',\n')}\n}\n`;
}

const [command, input, output] = process.argv.slice(2);
if (!command || !input || (command === 'pack' && !output)) {
  console.error('usage: node tools/zone.mjs info|dump|pack <file> [out]');
  process.exit(1);
}
const level = migrateLevel(await read(input), World.shape);
if (command === 'info') {
  const raw = packZone(level).length, gz = (await encodeZone(level)).length;
  const blocks = levelRows(level).blocks.length;
  console.log(`${input}: zone ${JSON.stringify(level.zone)}`);
  console.log(`  ${blocks} blocks in ${level.chunks.length} chunks, ${level.palette.length - 1} kinds of block, ` +
              `${level.models.length} models, ${Object.keys(level.spawns).length} spawns, ${level.exits.length} exits`);
  console.log(`  ${raw} bytes packed, ${gz} as a .bxz`);
} else if (command === 'dump') {
  const text = readable(levelRows(level));
  if (output) writeFileSync(output, text); else process.stdout.write(text);
} else if (command === 'pack') {
  level.zone.name ??= basename(output).replace(/\.[^.]*$/, '');
  writeFileSync(output, await encodeZone(level));
  console.log(`wrote ${output}`);
} else {
  console.error(`unknown command ${command}`);
  process.exit(1);
}
