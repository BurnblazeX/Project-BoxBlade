// Regions on disk (js/core/region.js), from the command line.
//
//   node tools/region.mjs info <file>            what a level holds, and its sizes
//   node tools/region.mjs dump <file> [out.json] a level as readable JSON, a row a block
//   node tools/region.mjs pack <file> <out.bxr>  any level (v1 .json, a dump, a .bxr) to a .bxr
//
// A .bxr is binary; dump is how to read or diff one, and pack takes the dump
// back. Packing a v1 .json is the migration: bounds from its blocks, the name
// from the output file's.

import { readFileSync, writeFileSync } from 'node:fs';
import { basename } from 'node:path';
import { decodeRegion, encodeRegion, migrateLevel, levelRows, packRegion } from '../js/core/region.js';
import { World } from '../js/core/world.js';

async function read(file) {
  const bytes = readFileSync(file);
  return file.endsWith('.json') ? JSON.parse(bytes.toString('utf8')) : decodeRegion(new Uint8Array(bytes));
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
  console.error('usage: node tools/region.mjs info|dump|pack <file> [out]');
  process.exit(1);
}
const level = migrateLevel(await read(input), World.shape);
if (command === 'info') {
  const raw = packRegion(level).length, gz = (await encodeRegion(level)).length;
  const blocks = levelRows(level).blocks.length;
  console.log(`${input}: region ${JSON.stringify(level.region)}`);
  console.log(`  ${blocks} blocks in ${level.chunks.length} chunks, ${level.palette.length - 1} kinds of block, ` +
              `${level.models.length} models, ${Object.keys(level.spawns).length} spawns, ${level.exits.length} exits`);
  console.log(`  ${raw} bytes packed, ${gz} as a .bxr`);
} else if (command === 'dump') {
  const text = readable(levelRows(level));
  if (output) writeFileSync(output, text); else process.stdout.write(text);
} else if (command === 'pack') {
  level.region.name ??= basename(output).replace(/\.[^.]*$/, '');
  writeFileSync(output, await encodeRegion(level));
  console.log(`wrote ${output}`);
} else {
  console.error(`unknown command ${command}`);
  process.exit(1);
}
