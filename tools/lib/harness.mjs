// Minimal zero-dependency test harness.
//
// Counts are module-level and shared: the runner imports every test file into
// one process, so totals accumulate across them and a single summary is printed
// at the end. Tests are plain top-level code - importing the file runs it.

let pass = 0;
let fail = 0;
const failures = [];
let currentFile = '(unknown)';
let currentSection = '';

export function file(name) {
  currentFile = name;
  console.log(`\n\x1b[1m${name}\x1b[0m`);
}

export function section(name) {
  currentSection = name;
  console.log(`  \x1b[2m${name}\x1b[0m`);
}

function record(passed, label, detail) {
  if (passed) {
    pass++;
    console.log(`    \x1b[32mPASS\x1b[0m  ${label}`);
  } else {
    fail++;
    failures.push(`${currentFile} > ${currentSection} > ${label}: ${detail}`);
    console.log(`    \x1b[31mFAIL\x1b[0m  ${label} \x1b[31m(${detail})\x1b[0m`);
  }
}

export function ok(label, got, want) {
  record(got === want, label, `got ${format(got)}, want ${format(want)}`);
}

export function near(label, got, want, epsilon = 1e-6) {
  record(Math.abs(got - want) < epsilon, label, `got ${got}, want ~${want} (±${epsilon})`);
}

export function truthy(label, got) {
  record(!!got, label, `got ${format(got)}, want truthy`);
}

export function falsy(label, got) {
  record(!got, label, `got ${format(got)}, want falsy`);
}

export function inRange(label, got, min, max) {
  record(got >= min && got <= max, label, `got ${got}, want ${min}..${max}`);
}

export function throws(label, fn) {
  let threw = false;
  try { fn(); } catch { threw = true; }
  record(threw, label, 'did not throw');
}

// Bare informational line - not an assertion, for numbers worth eyeballing.
export function note(text) {
  console.log(`          \x1b[2m${text}\x1b[0m`);
}

function format(v) {
  if (typeof v === 'string') return `"${v}"`;
  return String(v);
}

export function results() {
  return { pass, fail, failures };
}
