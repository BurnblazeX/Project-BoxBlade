// --- Normal maps, generated ---
//
// Every sprite and block texture gets a tangent-space normal map. Generated
// here from what the texture already is; a hand-authored <name>_n.png beside
// the texture replaces the generated one (see main.js).
//
// Output is RGBA8 in the LabPBR normal layout, the standard BoxBlade follows:
//   R  normal x, along the texture's u (right); 128 is flat
//   G  normal y, DirectX convention - higher tilts DOWN the texture; 128 flat
//   B  ambient occlusion baked into the texture: 255 none, 0 fully occluded
//   A  height (unused yet); 255 is the surface
// The normal's z is not stored - it is rebuilt from x and y, since the normal
// is unit length and faces out of the surface.
// Rows are written BOTTOM-UP, so row 0 is v = 0 - the same orientation a
// loaded PNG ends up in on the GPU, which is what lets an authored override
// drop in with no special case.

// Sprites: a rounded bevel off the silhouette. How far in from the edge, in
// sprite pixels, the surface takes to curve from edge-on to facing the viewer.
export const SPRITE_BEVEL_PX = 3;

// Terrain: how steep a full black-to-white step in luminance reads, as a
// height in texels. Higher is bumpier.
export const TERRAIN_BUMP = 1.5;

// ao and height in [0, 1], 1 meaning none / at the surface.
export function encodeNormal(out, o, x, y, z, ao = 1, height = 1) {
  const len = Math.hypot(x, y, z) || 1;
  out[o] = Math.round((x / len * 0.5 + 0.5) * 255);
  out[o + 1] = Math.round((-y / len * 0.5 + 0.5) * 255);   // DirectX: green is down
  out[o + 2] = Math.round(ao * 255);
  out[o + 3] = Math.round(height * 255);
}

// [x, y, z, ao] with y up and z rebuilt - the mirror of labNormalTSL in gpu.js.
export function decodeNormal(data, o) {
  const x = data[o] / 127.5 - 1, y = 1 - data[o + 1] / 127.5;
  return [x, y, Math.sqrt(Math.max(0, 1 - x * x - y * y)), data[o + 2] / 255];
}

// Height field to normals by central differences. heightAt(x, row) takes image
// coordinates (row 0 at the top); y in the normal is up, so it is the negative
// of the row direction.
function normalsFromHeight(w, h, heightAt) {
  const out = new Uint8Array(w * h * 4);
  for (let row = 0; row < h; row++) {
    for (let x = 0; x < w; x++) {
      const hx = (heightAt(x + 1, row) - heightAt(x - 1, row)) / 2;
      const hy = (heightAt(x, row - 1) - heightAt(x, row + 1)) / 2;   // up = -row
      encodeNormal(out, ((h - 1 - row) * w + x) * 4, -hx, -hy, 1);
    }
  }
  return out;
}

// From a sprite's signed distance transform (card.dt in cards.js: padded by
// `pad` on every side, negative inside, in pixels). Height is a quarter-circle
// bevel of SPRITE_BEVEL_PX: zero at the edge, flat once that far in. Outside
// the silhouette the pixel is transparent and never shaded, so it is left
// facing the viewer.
export function spriteNormals(dt, paddedW, cols, rows, pad, bevel = SPRITE_BEVEL_PX) {
  const heightAt = (x, row) => {
    const cx = Math.min(Math.max(x, -pad), cols + pad - 1) + pad;
    const cy = Math.min(Math.max(row, -pad), rows + pad - 1) + pad;
    const t = Math.min(Math.max(-dt[cy * paddedW + cx] / bevel, 0), 1);
    return bevel * Math.sqrt(1 - (1 - t) * (1 - t));
  };
  return normalsFromHeight(cols, rows, heightAt);
}

// From a tiling block texture's RGBA: height is luminance, wrapped at the edges
// because the texture repeats across neighbouring blocks.
export function terrainNormals(rgba, w, h, bump = TERRAIN_BUMP) {
  const lum = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    lum[i] = (0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]) / 255;
  }
  const heightAt = (x, row) =>
    bump * lum[((row % h + h) % h) * w + ((x % w + w) % w)];
  return normalsFromHeight(w, h, heightAt);
}
