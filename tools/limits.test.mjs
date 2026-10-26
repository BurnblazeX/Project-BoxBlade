import { file, section, ok } from './lib/harness.mjs';
import { textureLimitsToRequest, noteDeviceLimits, deviceLimits, fitFallback, FALLBACK_TEXTURES,
         FALLBACK_OPTIONAL, GUARANTEED_SAMPLED_TEXTURES } from '../js/gpu/limits.js';

file('limits.test.mjs - texture limits, and the fallback fitted to them');

section('the device asks for what the adapter has');
ok('a desktop adapter: its highest', JSON.stringify(textureLimitsToRequest(
  { maxSampledTexturesPerShaderStage: 48, maxSamplersPerShaderStage: 16 })), '{"maxSampledTexturesPerShaderStage":48}');
ok('an adapter at the guarantee: nothing to ask', JSON.stringify(textureLimitsToRequest(
  { maxSampledTexturesPerShaderStage: 16, maxSamplersPerShaderStage: 16 })), '{}');
ok('no limits object: nothing', JSON.stringify(textureLimitsToRequest(undefined)), '{}');
noteDeviceLimits({ maxSampledTexturesPerShaderStage: 48, maxSamplersPerShaderStage: 32 });
ok('the granted limits are what the builds use', deviceLimits.sampledTextures + '/' + deviceLimits.samplers, '48/32');
noteDeviceLimits({});
ok('a device without them: the guarantee', deviceLimits.sampledTextures, GUARANTEED_SAMPLED_TEXTURES);

section('split shading on: nothing given up');
{
  const opts = { split: {}, smokeView: true };
  const r = fitFallback(opts, 16);
  ok('the same options', r.opts === opts && r.dropped.length === 0, true);
}

section('split shading off');
{
  const opts = { split: null, smokeView: true, smoke: {} };
  const tight = fitFallback(opts, 16);
  ok('at 16: smoke in reflections goes', tight.opts.smokeView, false);
  ok('... and is reported', tight.dropped.join(), 'smokeView');
  ok('... the smoke itself stays', !!tight.opts.smoke, true);
  ok('... the caller\'s options are not changed', opts.smokeView, true);
  const roomy = fitFallback(opts, 48);
  ok('with room (48): kept', roomy.opts === opts && roomy.opts.smokeView, true);
  ok('at exactly what it needs: kept', fitFallback(opts, FALLBACK_TEXTURES).opts.smokeView, true);
  const off = fitFallback({ split: null, smokeView: false }, 16);
  ok('already off: nothing reported', off.dropped.length, 0);
  const saved = FALLBACK_OPTIONAL.reduce((s, [, n]) => s + n, 0);
  ok('everything optional given up fits the guarantee', FALLBACK_TEXTURES - saved <= GUARANTEED_SAMPLED_TEXTURES, true);
}
