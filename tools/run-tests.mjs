// Smoke-test runner. Discovers every *.test.mjs in tools/ and runs it.
//
//   npm test                 - run everything
//   npm test boxgrid         - run only files whose name contains "boxgrid"
//
// Tests must stay headless: no WebGPU device, no DOM. Modules that only import
// Three.js are fine (debug.js does), but anything constructing a renderer is not
// testable here - that is what the in-browser checklist is for.

import { readdir } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';
import { results } from './lib/harness.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const filter = process.argv[2];

const entries = (await readdir(here))
  .filter(f => f.endsWith('.test.mjs'))
  .filter(f => !filter || f.includes(filter))
  .sort();

if (entries.length === 0) {
  console.error(filter ? `No test files match "${filter}"` : 'No test files found');
  process.exit(1);
}

for (const entry of entries) {
  // pathToFileURL, not a string join: Windows paths need proper URL encoding
  // (drive letters, spaces) before dynamic import will accept them.
  await import(pathToFileURL(join(here, entry)).href);
}

const { pass, fail, failures } = results();
const total = pass + fail;

console.log('\n' + '-'.repeat(60));
if (fail === 0) {
  console.log(`\x1b[32m${pass}/${total} passed\x1b[0m across ${entries.length} file(s)`);
} else {
  console.log(`\x1b[31m${fail} failed\x1b[0m, ${pass} passed across ${entries.length} file(s)\n`);
  for (const f of failures) console.log(`  \x1b[31m*\x1b[0m ${f}`);
}
process.exit(fail ? 1 : 0);
