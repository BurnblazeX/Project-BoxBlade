import { file, section, ok, truthy } from './lib/harness.mjs';
import { coerceSetting, optionList } from '../js/settings.js';

file('settings.test.mjs - what a panel control hands to a setter');

section('ranges clamp and snap');
const r = { type: 'range', min: 0, max: 1, step: 0.05 };
ok('in range, on step', coerceSetting(r, '0.35'), 0.35);
ok('snapped to the step', coerceSetting(r, 0.33), 0.35);
ok('clamped low', coerceSetting(r, -3), 0);
ok('clamped high', coerceSetting(r, 7), 1);
ok('no float dust from 0.1-style steps', coerceSetting({ type: 'range', min: 0, max: 1, step: 0.1 }, 0.3), 0.3);
ok('garbage falls to the minimum', coerceSetting(r, 'abc'), 0);
ok('steps count from the minimum', coerceSetting({ type: 'range', min: 1, max: 15, step: 2 }, 4), 5);
ok('negative ranges', coerceSetting({ type: 'range', min: -0.6, max: 0.6, step: 0.05 }, -0.31), -0.3);

section('toggles and selects');
ok('toggle to boolean', coerceSetting({ type: 'toggle' }, 1), true);
const sel = { type: 'select', options: ['cone', 1, 16] };
ok('string select value matches a number option', coerceSetting(sel, '16'), 16);
ok('and keeps a string option a string', coerceSetting(sel, 'cone'), 'cone');
ok('unknown falls back to the first option', coerceSetting(sel, 'nope'), 'cone');
truthy('bare options get labels', optionList(sel)[1].label === '1');
