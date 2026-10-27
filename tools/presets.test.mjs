import { file, section, ok, truthy } from './lib/harness.mjs';
import { PRESETS, PRESET_NAMES, matchPreset, applyPreset } from '../js/tools/presets.js';

file('presets.test.mjs - graphics presets over the settings panel');

// Stand-in settings: a value each, and a log of what was set.
function fakeDefs(keys) {
  const state = {}, sets = [];
  const defs = new Map(keys.map(k => [k, {
    key: k, get: () => state[k], set: v => { state[k] = v; sets.push(k); }
  }]));
  return { defs, state, sets };
}
const allKeys = [...new Set(PRESET_NAMES.flatMap(n => Object.keys(PRESETS[n].values)))];

section('the table');
ok('five tiers, cheapest first', PRESET_NAMES.join(','), 'potato,low,original,high,ultra');
truthy('potato is basic mode, the rest advanced',
  !PRESETS.potato.effects && PRESET_NAMES.slice(1).every(n => PRESETS[n].effects));
const texelOnly = ['gireach', 'gidirectevery'];
const advanced = PRESET_NAMES.filter(n => PRESETS[n].effects);
const keysOf = n => Object.keys(PRESETS[n].values).filter(k => !texelOnly.includes(k)).sort().join(',');
truthy('every advanced preset names the same keys (texel GI\'s own aside)',
  advanced.every(n => keysOf(n) === keysOf('original')));
ok('the GI method is set first, so a match never reads texel GI unasked',
  advanced.map(n => Object.keys(PRESETS[n].values)[0]).join(','), advanced.map(() => 'gimode').join(','));
ok('only ultra uses texel GI', advanced.filter(n => PRESETS[n].values.gimode === 'texel').join(','), 'ultra');

section('apply and match');
{
  const { defs, state, sets } = fakeDefs(allKeys);
  let effects = false;
  const calls = [];
  const setEffects = async on => { calls.push(on); effects = on; };
  await applyPreset('low', { defs, effectsOn: () => effects, setEffects });
  ok('effects turned on after the values', calls.join(','), 'true');
  ok('matched afterwards', matchPreset(defs, effects), 'low');
  ok('not potato while effects are on', matchPreset(defs, true) === 'potato', false);
  await applyPreset('original', { defs, effectsOn: () => effects, setEffects });
  ok('a switch turns effects off first, then on', calls.join(','), 'true,false,true');
  ok('original matched', matchPreset(defs, effects), 'original');
  state.gibounces = 9;
  ok('a moved knob is custom', matchPreset(defs, effects), null);
  const before = sets.length;
  await applyPreset('original', { defs, effectsOn: () => effects, setEffects });
  ok('reapplying sets only what moved', sets.length - before, 1);
}
{
  const { defs } = fakeDefs(['gimode']);
  const msg = await applyPreset('high', { defs, effectsOn: () => true, setEffects: async () => {} });
  truthy('settings this build lacks are skipped, and said', msg.includes('skipped'));
}
