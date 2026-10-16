import * as THREE from 'three/webgpu';

// --- Real shader functions, not inlined copies ---
//
// A TSL Fn WITHOUT a layout is pasted inline at every call site. The march
// helpers are called from many places - the sun, every light, reflection
// hits, the mirrors, sprites - so the terrain shader held about ten copies of
// the cascade walk, each three step loops, and compiling it froze the game for
// tens of seconds. With setLayout each is ONE WGSL function, emitted once per
// shader and called. Names must be unique per shader, hence the counter; and
// the per-cascade ones are memoised, so every caller shares one function.
//
// PER SHADER, NOT PER BACKEND. three caches a layout function's generated
// WGSL per backend, keyed by the function node, and hands that same text to
// every shader that calls it. That is right for its own layout functions,
// which are pure maths; ours read uniforms and textures (the cascades, the
// cards, the light list), and the text names them as the FIRST shader to build
// it did - so every other shader got "no definition for nodeUniform2". The
// body is built by whichever builder calls it (flowShaderNode), registering
// its uniforms there correctly, so the fix is only the cache: functions whose
// layout carries perShader are cached per builder instead.
const perBuilderFns = new WeakMap();

// Per builder too: how many functions of each base name it has named. A name is
// base_n, n counting that base's functions in THIS shader in the order they are
// first used - so the same graph gives the same WGSL every run. (A global
// serial was unique too, but it counted every function made before, in any
// build, so the text - and the browser's pipeline cache key - varied between
// runs with what had been built earlier.)
const perBuilderNames = new WeakMap();

const baseBuildFunctionNode = THREE.NodeBuilder.prototype.buildFunctionNode;
THREE.NodeBuilder.prototype.buildFunctionNode = function (shaderNode) {
  if (!(shaderNode.layout && shaderNode.layout.perShader)) {
    return baseBuildFunctionNode.call(this, shaderNode);
  }
  let fns = perBuilderFns.get(this);
  if (!fns) perBuilderFns.set(this, fns = new Map());
  let fn = fns.get(shaderNode);
  if (!fn) {
    fn = new THREE.FunctionNode();
    let counts = perBuilderNames.get(this);
    if (!counts) perBuilderNames.set(this, counts = new Map());
    const layout = shaderNode.layout;
    const n = counts.get(layout.baseName) || 0;
    counts.set(layout.baseName, n + 1);
    // Only read while the code is generated (the call sites take the name from
    // the code), so it is set for that and put back.
    const resting = layout.name;
    layout.name = `${layout.baseName}_${n}`;
    const previous = this.currentFunctionNode;
    this.currentFunctionNode = fn;
    try {
      fn.code = this.buildFunctionCode(shaderNode);
    } finally {
      this.currentFunctionNode = previous;
      layout.name = resting;
    }
    fns.set(shaderNode, fn);
  }
  return fn;
};

let fnSerial = 0;

const fnName = base => `${base}_${fnSerial++}`;

// Every layout here goes through this: a unique name, and per-shader caching.
export const layout = (base, type, inputs) => ({ name: fnName(base), baseName: base, type, inputs,
                                         perShader: true });

const perCascades = new WeakMap();

export function memoCascades(cascades, key, make) {
  if (!perCascades.has(cascades)) perCascades.set(cascades, {});
  const m = perCascades.get(cascades);
  return m[key] || (m[key] = make());
}
