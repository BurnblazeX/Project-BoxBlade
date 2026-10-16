// --- Cutout sprites drawn opaque ---
//
// The pixel art's alpha is binary: every texel is fully opaque or fully clear.
// Sampled nearest with no mips, and alpha-tested at 0.5, a binary-alpha texture
// blends nothing - each drawn pixel has alpha 1 - so drawing it in the opaque
// pass puts exactly the same pixels on screen. What changes is the cost. three
// draws a transparent DoubleSide material TWICE (back faces, then front), sorts
// it every frame, and writes its uniforms per pass; a tree is four such planes.
// Opaque, the same material is one draw and leaves the sort.
//
// Decided per texture, from its pixels, not per asset by hand: a texture with
// any partial alpha (glass, smoke, a soft edge) keeps its transparent flag and
// is drawn and blended as before. So this never has to be revisited when
// translucent art arrives - it simply does not apply to it.
//
// bxb.cutouts() turns it off (everything back to transparent) for A/B.

let enabled = true;
const materials = new Set();

// Pure, so it can be tested headless: true when every alpha is 0 or 255.
export function alphaIsBinary(rgba) {
  for (let i = 3; i < rgba.length; i += 4) {
    const a = rgba[i];
    if (a !== 0 && a !== 255) return false;
  }
  return true;
}

// The loader's onLoad. Reads the image's alpha once and re-syncs every
// registered material using it. Anything unreadable counts as not binary -
// the safe answer is the old one, transparent.
export function classifyTextureAlpha(tex) {
  let binary = false;
  try {
    const img = tex.image;
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const g = canvas.getContext('2d', { willReadFrequently: true });
    g.drawImage(img, 0, 0);
    binary = alphaIsBinary(g.getImageData(0, 0, canvas.width, canvas.height).data);
  } catch {
    binary = false;
  }
  tex.userData.alphaBinary = binary;
  for (const m of materials) if (m.map === tex) syncCutout(m);
  return tex;
}

// Whether this material may draw opaque. Needs an alpha test too: without one
// the clear texels would draw as solid quads.
export function cutoutOpaque(material) {
  return enabled && material.alphaTest > 0 && material.map?.userData?.alphaBinary === true;
}

// The transparent flag the material should carry. onTop forces transparent: the
// characters' clip fix draws them last, over everything, and "last" is the
// transparent pass.
export function wantsTransparent(material, onTop = false) {
  return onTop || !cutoutOpaque(material);
}

// onTop is asked on every sync, so it follows the character's clip state.
export function registerCutout(material, onTop = () => false) {
  material.userData.cutoutOnTop = onTop;
  materials.add(material);
  syncCutout(material);
  return material;
}

// Flag changes rebuild the pipeline, so only on an actual change.
export function syncCutout(material) {
  const onTop = material.userData.cutoutOnTop?.() ?? false;
  const t = wantsTransparent(material, onTop);
  if (material.transparent !== t) {
    material.transparent = t;
    material.needsUpdate = true;
  }
}

export function cutoutsEnabled() { return enabled; }

export function setCutoutsEnabled(on) {
  enabled = !!on;
  for (const m of materials) syncCutout(m);
}
