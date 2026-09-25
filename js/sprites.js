import * as THREE from 'three/webgpu';
import { isSolid, BLOCK_METRES } from './world.js';
import { wantsTransparent } from './cutout.js';

// --- Character quad vs. world geometry ---
//
// Characters depth-test against the world normally, so one standing behind a
// wall is properly occluded by it. The problem is that the quad is a full block
// wide, so its corners sit exactly BLOCK_METRES/2 from centre - the cell
// half-width. Any yaw puts them on the voxel boundary, and the backward lean
// pushes the top edge past it, so alongside a tall block the quad genuinely
// penetrates geometry and gets sliced open.
//
// Rather than give up depth testing everywhere, detect the penetration and only
// draw on top for the frames where it is actually happening. Occlusion stays
// correct the rest of the time.

export const CHARACTER_RENDER_ORDER = 10;

// --- Sprite geometry ---
//
// Sprite shading takes each quad's frame from its vertices: normal for the way
// it faces, tangent for its art's +x (gpu.js spriteShadeNormalTSL). A plain
// PlaneGeometry has the normal but no tangent, so add one: +x, the art's right.
// Also which plane of the mesh each vertex is on (spritePlane, 0 for a single
// quad) - the sprite texel atlas keeps each plane's texels apart.
export function addSpriteTangent(geo) {
  const n = geo.attributes.position.count;
  const t = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) { t[i * 4] = 1; t[i * 4 + 3] = 1; }
  geo.setAttribute('tangent', new THREE.BufferAttribute(t, 4));
  geo.setAttribute('spritePlane', new THREE.BufferAttribute(new Float32Array(n), 1));
  geo.userData.spritePlanes = 1;
  return geo;
}

// Crossed planes as ONE geometry: a width x height quad, bottom edge at the
// origin, once per angle about Y. Four separate meshes were four draws (eight,
// while transparent) with four uniform writes; merged, a tree is one. Each
// plane keeps its own normal and tangent, which is all its shading reads, and
// the mesh's origin is where each plane's was, so the card skip (the model
// matrix's position) is unchanged.
export function crossedPlanesGeometry(width, height, angles) {
  const parts = angles.map((a, k) => {
    const g = addSpriteTangent(new THREE.PlaneGeometry(width, height));
    g.translate(0, height / 2, 0);
    g.rotateY(a);
    g.attributes.spritePlane.array.fill(k);
    return g;
  });
  const names = ['position', 'normal', 'uv', 'tangent', 'spritePlane'];
  const merged = new THREE.BufferGeometry();
  for (const name of names) {
    const size = parts[0].attributes[name].itemSize;
    const total = parts.reduce((k, g) => k + g.attributes[name].array.length, 0);
    const out = new Float32Array(total);
    let at = 0;
    for (const g of parts) { out.set(g.attributes[name].array, at); at += g.attributes[name].array.length; }
    merged.setAttribute(name, new THREE.BufferAttribute(out, size));
  }
  const index = [];
  let base = 0;
  for (const g of parts) {
    for (const i of g.index.array) index.push(i + base);
    base += g.attributes.position.count;
  }
  merged.setIndex(index);
  merged.computeBoundingBox();
  merged.computeBoundingSphere();
  merged.userData.spritePlanes = angles.length;
  for (const g of parts) g.dispose();
  return merged;
}

// --- Sprite texel atlas ---
//
// A sprite's shading is locked to its art's texels (gpu.js spriteTexelLockTSL),
// and at play zoom a texel covers some 80 pixels - each of which ran the full
// shading, torch and mirror light and all, to the same answer. So each texel
// is shaded ONCE, into a float atlas, and the sprite's own draw reads it back.
//
// Each sprite mesh gets a region: its art's width x height, once per plane
// across and once per side down (front row block, then back - a back face
// faces the other way, so it is lit differently). The atlas pass draws each
// mesh's twin: this geometry, every vertex twice - once per side, spriteSide 0
// then 1 - and the vertex shader lays each plane and side out flat on its
// region, one fragment per texel.
const atlasGeometries = new WeakMap();
export function spriteAtlasGeometry(geo) {
  let out = atlasGeometries.get(geo);
  if (out) return out;
  out = new THREE.BufferGeometry();
  const n = geo.attributes.position.count;
  for (const name of ['position', 'normal', 'uv', 'tangent', 'spritePlane']) {
    const a = geo.attributes[name];
    const arr = new Float32Array(a.array.length * 2);
    arr.set(a.array, 0);
    arr.set(a.array, a.array.length);
    out.setAttribute(name, new THREE.BufferAttribute(arr, a.itemSize));
  }
  const side = new Float32Array(n * 2);
  side.fill(1, n);
  out.setAttribute('spriteSide', new THREE.BufferAttribute(side, 1));
  const idx = geo.index.array;
  const index = new Array(idx.length * 2);
  for (let i = 0; i < idx.length; i++) { index[i] = idx[i]; index[i + idx.length] = idx[i] + n; }
  out.setIndex(index);
  // Culled with the sprite it shades: the same bounds.
  if (!geo.boundingSphere) geo.computeBoundingSphere();
  out.boundingSphere = geo.boundingSphere.clone();
  atlasGeometries.set(geo, out);
  return out;
}

// The region a sprite needs: art width per plane across, art height per side down.
export const spriteAtlasRegion = (texW, texH, planes) => [texW * planes, texH * 2];

// Shelf packing: sizes [[w, h], ...] into rows of the given width, tallest
// first so the shelves stay full. Returns each one's [x, y] (in input order)
// and the height used. Whole pixels, so a region's texels land on pixel centres.
export function packSpriteAtlas(sizes, width) {
  const order = sizes.map((s, i) => i).sort((a, b) => sizes[b][1] - sizes[a][1]);
  const at = new Array(sizes.length);
  let x = 0, y = 0, shelf = 0;
  for (const i of order) {
    const [w, h] = sizes[i];
    if (w > width) throw new Error(`sprite region ${w} wider than the atlas (${width})`);
    if (x + w > width) { y += shelf; x = 0; shelf = 0; }
    at[i] = [x, y];
    x += w;
    shelf = Math.max(shelf, h);
  }
  return { at, height: y + shelf };
}

// Four corners plus the top-edge midpoint. The top edge is what leans, so it is
// where penetration starts; block-scale geometry cannot pass through the quad's
// interior without touching one of these.
export function makeClipSamples(width, height) {
  return [
    new THREE.Vector3(-width / 2, 0, 0),
    new THREE.Vector3( width / 2, 0, 0),
    new THREE.Vector3(-width / 2, height, 0),
    new THREE.Vector3( width / 2, height, 0),
    new THREE.Vector3(0, height, 0)
  ];
}

const _point = new THREE.Vector3();

// Takes a world matrix rather than a mesh so it can be exercised without a
// renderer or a scene graph.
export function isQuadClipping(matrixWorld, samples) {
  for (const local of samples) {
    _point.copy(local).applyMatrix4(matrixWorld);
    // Blocks are centred on b * BLOCK_METRES, so rounding maps a world point to
    // the cell that contains it.
    if (isSolid(
      Math.round(_point.x / BLOCK_METRES),
      Math.round(_point.y / BLOCK_METRES),
      Math.round(_point.z / BLOCK_METRES)
    )) return true;
  }
  return false;
}

export function updateCharacterClipping(mesh, samples) {
  mesh.updateMatrixWorld(true); // rotation was set this frame; the matrix is stale until asked
  const clipping = isQuadClipping(mesh.matrixWorld, samples);
  if (clipping === mesh.userData.drawingOnTop) return clipping; // only touch state on a transition

  mesh.userData.drawingOnTop = clipping;
  mesh.renderOrder = clipping ? CHARACTER_RENDER_ORDER : 0;
  // Lit materials come as a prebuilt pair. Flipping depth state on a lit
  // material means a fresh pipeline - a shader compile of 100 ms or more, the
  // first time a character leans into a wall - where swapping to a variant
  // compiled in advance costs nothing.
  const v = mesh.userData.depthVariants;
  if (v) { mesh.material = clipping ? v.onTop : v.normal; return clipping; }
  mesh.material.depthTest = !clipping;
  mesh.material.depthWrite = !clipping;
  // Drawing on top means drawing last, which is the transparent pass; clear of
  // the wall, a binary-alpha cutout goes back to opaque (js/cutout.js).
  mesh.material.transparent = wantsTransparent(mesh.material, clipping);
  // Depth state is baked into the render pipeline, so the material must be
  // rebuilt. Gated on the transition above, so this costs nothing per frame.
  mesh.material.needsUpdate = true;
  return clipping;
}
