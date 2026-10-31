// --- The sky-top map: how high each column's roof is ---
//
// One code per block column: the top of the highest block in it that is not
// glass - what a GI or reflection ray that missed everything must be above to
// count as having reached the sky (gpu/volume.js skyOpenTSL). A ray that only
// ran out of reach indoors ends under its room's roof, so it brings no sky in;
// one that left through a door or a window ends outside, over open ground.
//
// Glass is not a roof: a skylight lets the sky through (tinted by the ray's
// own path through the glass). Placed models are left out - a model roof does
// not block the sky (first version).
//
// The code is 0 for a column with no roof at all, else 1 + the top in
// SKY_TOP_STEPS ths of a block above the authorable range's floor, rounded
// up: a roof is never placed lower than it is, and at most a step (9 cm)
// higher. 16 bits: 48 levels at 16 steps is 769 codes, past a byte's 255
// (World plan P1). The texture carries it as two bytes, low in R, high in G
// (render.js, gpu/volume.js skyOpenTSL).
import { BLOCK_METRES, Y_MIN, isSlope, partSpan, blockParts } from './world.js';
import { isGlassMaterial } from './materials.js';
import { blocksIn } from './chunks.js';

export const SKY_TOP_STEPS = 16;

// The top of a block's non-glass parts, in blocks above its cell's floor -
// null for a block that is glass all through.
export function roofTop(block) {
  let hi = null;
  for (const { materialId, part, boxes } of blockParts(block)) {
    if (isGlassMaterial(materialId)) continue;
    // A slope's top is its plane's, over the staircase.
    const top = isSlope(part) ? partSpan(part)[1] : Math.max(...boxes.map(b => b[3]));
    if (hi === null || top > hi) hi = top;
  }
  return hi;
}

// The map over a box of columns: origin [x, z] and size [w, d] in blocks,
// row-major by z (data[z * w + x]). world: a Map of 'x,y,z' -> block.
export function buildSkyTops(world, origin, size) {
  const out = new Uint16Array(size[0] * size[1]);
  const box = [origin[0], origin[0] + size[0] - 1, origin[1], origin[1] + size[1] - 1];
  for (const [[x, y, z], block] of blocksIn(world, box)) {
    const cx = x - origin[0], cz = z - origin[1];
    if (cx < 0 || cz < 0 || cx >= size[0] || cz >= size[1]) continue;
    const hi = roofTop(block);
    if (hi === null) continue;
    const code = encodeSkyTop((y - 0.5 + hi) * BLOCK_METRES);
    const i = cz * size[0] + cx;
    if (code > out[i]) out[i] = code;
  }
  return out;
}

// World metres to the code and back (the shader mirrors decodeSkyTop).
export const SKY_TOP_MAX_CODE = 65535;
export function encodeSkyTop(metres) {
  const steps = Math.ceil((metres / BLOCK_METRES - (Y_MIN - 0.5)) * SKY_TOP_STEPS - 1e-6);
  return Math.max(1, Math.min(SKY_TOP_MAX_CODE, 1 + steps));
}

// Codes as the texture's bytes: low byte in R, high in G.
export function packSkyTops(codes, out = new Uint8Array(codes.length * 2)) {
  for (let i = 0; i < codes.length; i++) { out[2 * i] = codes[i] & 255; out[2 * i + 1] = codes[i] >> 8; }
  return out;
}

export function decodeSkyTop(code) {
  return code === 0 ? -Infinity : ((code - 1) / SKY_TOP_STEPS + Y_MIN - 0.5) * BLOCK_METRES;
}
