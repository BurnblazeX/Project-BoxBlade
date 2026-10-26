// --- The device's texture limits, and what the shading gives up to fit them ---
//
// WebGPU guarantees 16 sampled textures (and 16 samplers) per shader stage.
// Desktop adapters allow far more (48 on the dev box's RTX), so the device asks
// for the adapter's highest (main.js requestGPUDevice) and the builds use what
// it was granted. The target (Adreno 610) stays at 16, so every path must
// still fit 16 when it has to.
//
// Split shading lights the texels in compute kernels, each with its own
// budget. Its fallback - one shader per low-res sample, when split shading is
// off (a GPU error in its first 20 s, the texel cache off, or an A/B) - binds
// every texture the shading reads in ONE fragment shader: 17 on the test map
// since smoke showed in reflections (A10//00). There, optional reads are given
// up, cheapest loss first, until the shader fits the device. Measured in the
// game: Firefox and Chrome both reported 17.
//
// Pure - no three, no DOM - so the rule tests headless.

export const GUARANTEED_SAMPLED_TEXTURES = 16;

// What the device was granted (main.js notes it once the device exists). A
// device three requested itself has the defaults.
export const deviceLimits = {
  sampledTextures: GUARANTEED_SAMPLED_TEXTURES,
  samplers: GUARANTEED_SAMPLED_TEXTURES
};

// The adapter's highest texture limits, for requestDevice's requiredLimits.
export function textureLimitsToRequest(adapterLimits) {
  const out = {};
  for (const k of ['maxSampledTexturesPerShaderStage', 'maxSamplersPerShaderStage']) {
    if (adapterLimits && adapterLimits[k] > GUARANTEED_SAMPLED_TEXTURES) out[k] = adapterLimits[k];
  }
  return out;
}

export function noteDeviceLimits(limits) {
  if (!limits) return;
  deviceLimits.sampledTextures = limits.maxSampledTexturesPerShaderStage || GUARANTEED_SAMPLED_TEXTURES;
  deviceLimits.samplers = limits.maxSamplersPerShaderStage || GUARANTEED_SAMPLED_TEXTURES;
}

// The fallback's sampled textures with every optional read on, and the
// optional reads in the order they are given up: [build option, textures it
// binds there]. Smoke in reflections and through glass first - the clouds
// still show directly.
export const FALLBACK_TEXTURES = 17;
export const FALLBACK_OPTIONAL = [['smokeView', 1]];

// The build options with what the fallback cannot afford turned off: the same
// object when nothing has to go (split shading on, or the device has room).
// dropped: the options given up, for the console.
export function fitFallback(opts, limit = deviceLimits.sampledTextures) {
  if (opts.split) return { opts, dropped: [] };
  let need = FALLBACK_TEXTURES;
  const out = { ...opts }, dropped = [];
  for (const [key, textures] of FALLBACK_OPTIONAL) {
    if (need <= limit) break;
    if (!out[key]) continue;
    out[key] = false;
    need -= textures;
    dropped.push(key);
  }
  return { opts: dropped.length ? out : opts, dropped };
}
