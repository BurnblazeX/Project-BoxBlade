import { file, section, ok, truthy } from './lib/harness.mjs';
import { VERSION, bundleHash, buildLabel } from '../js/version.js';

file('version.test.mjs - version and build stamp');

section('the version follows A##//#a');
truthy(`${VERSION} is well formed`, /^[ABCR]\d{2}\/\/\d+[a-z]?$/.test(VERSION));

section('bundle checksum');
ok('from a built entry', bundleHash('https://x.github.io/Project-BoxBlade/assets/index-BdlFL22P.js'), 'BdlFL22P');
ok('ignores a query', bundleHash('/assets/index-Ab_3-xYz.js?v=1'), 'Ab_3-xYz');
ok('none under the dev server', bundleHash('http://localhost:3000/js/main.js'), null);

section('label');
ok('built', buildLabel({ version: 'A03//6b', commit: 'c9b52f2', dirty: false, bundle: 'BdlFL22P' }),
   'A03//6b  c9b52f2  #BdlFL22P');
ok('dev, uncommitted changes', buildLabel({ version: 'A03//6b', commit: 'c9b52f2', dirty: true, bundle: null }),
   'A03//6b  c9b52f2+  dev');
