import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { file, section, ok, note } from './lib/harness.mjs';

// Every name we hand WGSL verbatim - a layout function's parameters and a
// Loop's counter, both written as `name: '...'` - must not be a WGSL keyword or
// reserved word. One was (`smooth`, a glass function's parameter): the shader
// failed to parse and every glass block drew as a flat white box. Checked here
// because the shader only exists in the browser.
file('wgslnames.test.mjs - names given to WGSL are not reserved');

// WGSL keywords and reserved words (WGSL spec, sections 15.1-15.2).
const RESERVED = new Set(`
alias break case const const_assert continue continuing default diagnostic discard else
enable false fn for if let loop override requires return struct switch true var while
NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await
become binding_array cast catch class co_await co_return co_yield coherent column_major
common compile compile_fragment concept const_cast consteval constexpr constinit crate
debugger decltype delete demote demote_to_helper do dynamic_cast enum explicit export
extends extern external fallthrough filter final finally friend from fxgroup get goto
groupshared highp impl implements import inline instanceof interface layout lowp macro
macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept
noinline nointerpolation noperspective null nullptr of operator package packoffset
partition pass patch pixelfragment precise precision premerge priv protected pub public
readonly ref regardless register reinterpret_cast require resource restrict self set
shared sizeof smooth snorm static static_assert static_cast std subroutine super target
template this thread_local throw trait try type typedef typeid typename typeof union
unless unorm unsafe unsized use using varying virtual volatile wgsl where with writeonly
yield`.split(/\s+/).filter(Boolean));

const here = dirname(fileURLToPath(import.meta.url));
const js = join(here, '..', 'js');
const names = [];
for (const f of readdirSync(js).filter(f => f.endsWith('.js'))) {
  const src = readFileSync(join(js, f), 'utf8');
  for (const m of src.matchAll(/name:\s*'([A-Za-z_][A-Za-z0-9_]*)'/g)) names.push([f, m[1]]);
}

section('layout parameters and loop counters');
note(`${names.length} names checked`);
const bad = names.filter(([, n]) => RESERVED.has(n)).map(([f, n]) => `${f}: ${n}`);
ok('none is reserved', bad.join(', '), '');
ok('the check catches one', RESERVED.has('smooth'), true);
