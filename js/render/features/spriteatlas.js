// --- Sprite texel atlas (render/sprites.js) ---
//
// The sprites' light, shaded once per art texel into a float target by one
// pass of twins, then read back by the sprites' own draws - the texel cache's
// idea for sprites, with a fixed slot per texel instead of a screen lookup.
// Each sprite mesh has a twin: its child (so the same world matrix), on
// SPRITE_ATLAS_LAYER only, drawing sprites.js spriteAtlasGeometry. Culled with
// its sprite; a slot off screen goes stale, and nothing reads it.
// bxb.spriteatlas() turns it off, back to shading every pixel, for A/B.
// Paced: the atlas is shaded at most SPRITE_ATLAS_HZ times a second (Alt+V
// Perf, 0 = every frame). Its slots do not move with the camera, only the
// light on them changes, and a few ms of lag on that is not visible - while
// above ~150 fps it skips most frames' pass.
//
// ctx:
//   renderer, scene, camera
//   meshes()     the sprite meshes drawn now
//   shadowsOn()  whether the marched shading is on
//   activeSet()  the shader set showing: its atlas materials, if it reads the atlas
//   rebuild()    the shading rebuilt, when the switch changed
import * as THREE from 'three/webgpu';
import { uniform } from 'three/tsl';
import { spriteAtlasGeometry, spriteAtlasRegion, packSpriteAtlas } from '../sprites.js';

const SPRITE_ATLAS_LAYER = 2;
const SPRITE_ATLAS_WIDTH = 512;
const SPRITE_ATLAS_HZ = 144;

export function createSpriteAtlas({ renderer, scene, camera, meshes, shadowsOn, activeSet, rebuild }) {
  let spriteAtlasOn = true;
  let atlasHz = SPRITE_ATLAS_HZ;
  let lastShaded = -Infinity;
  const spriteAtlas = {
    rt: null,
    tex: null,
    size: uniform(new THREE.Vector2(1, 1))
  };
  // Every sprite's art must be loaded to size its region.
  const spriteArtReady = () => meshes().every(m => {
    const map = (m.userData.originalMaterial || m.material).map;
    return map && map.image && map.image.width > 0;
  });
  // Twins for the meshes drawn now, and their regions. Run on each build and
  // whenever the drawn meshes change (bxb.treemerge). mats: base material ->
  // atlas material, for the twins; null leaves their materials as they are.
  function ensureSpriteAtlas(mats = null) {
    const drawn = meshes();
    const sizes = drawn.map(m => {
      const img = (m.userData.originalMaterial || m.material).map.image;
      return spriteAtlasRegion(img.width, img.height, m.geometry.userData.spritePlanes || 1);
    });
    const { at, height } = packSpriteAtlas(sizes, SPRITE_ATLAS_WIDTH);
    drawn.forEach((m, i) => {
      let twin = m.userData.spriteAtlasTwin;
      if (!twin) {
        twin = new THREE.Mesh(spriteAtlasGeometry(m.geometry), new THREE.MeshBasicNodeMaterial());
        twin.layers.set(SPRITE_ATLAS_LAYER);
        twin.raycast = () => {};
        twin.userData.spriteAtlasOrigin = m.userData.spriteAtlasOrigin = new THREE.Vector2();
        m.userData.spriteAtlasTwin = twin;
        m.add(twin);
      }
      twin.userData.spriteAtlasOrigin.set(at[i][0], at[i][1]);
      const lit = mats && mats.get(m.userData.originalMaterial || m.material);
      if (lit) twin.material = lit;
    });
    lastShaded = -Infinity;   // new twins or regions: shade them on the next frame
    const h = Math.max(1, height);
    if (!spriteAtlas.rt) {
      spriteAtlas.rt = new THREE.RenderTarget(SPRITE_ATLAS_WIDTH, h,
                                              { type: THREE.FloatType, depthBuffer: false });
      spriteAtlas.tex = spriteAtlas.rt.texture;
      spriteAtlas.tex.minFilter = spriteAtlas.tex.magFilter = THREE.NearestFilter;
      spriteAtlas.tex.generateMipmaps = false;
    } else if (spriteAtlas.rt.height !== h) {
      spriteAtlas.rt.setSize(SPRITE_ATLAS_WIDTH, h);
    }
    spriteAtlas.size.value.set(SPRITE_ATLAS_WIDTH, h);
  }
  // The atlas pass, before the main render. Only while the showing set reads it.
  function renderSpriteAtlas() {
    const set = activeSet();
    if (!shadowsOn() || !set || !set.atlasMats) return;
    const now = performance.now();
    // 2 ms of slack, so a 144 Hz display still shades every frame.
    if (atlasHz > 0 && now - lastShaded < 1000 / atlasHz - 2) return;
    lastShaded = now;
    const mask = camera.layers.mask;
    camera.layers.set(SPRITE_ATLAS_LAYER);
    renderer.setRenderTarget(spriteAtlas.rt);
    renderer.render(scene, camera);
    renderer.setRenderTarget(null);
    camera.layers.mask = mask;
  }

  const commands = {
    spriteatlas: {
      help: "sprite texel atlas: shade each sprite art texel once and read it back. Off shades every pixel, as before",
      run: () => {
        spriteAtlasOn = !spriteAtlasOn;
        if (shadowsOn()) rebuild();
        return `sprite atlas ${spriteAtlasOn ? 'on: one shading per art texel' : 'off: every pixel shades'}`;
      }
    }
  };

  return {
    name: 'spriteatlas',
    // The target and its size, as the sprite materials read them.
    bindings: spriteAtlas,
    // Whether a set built now shades through the atlas (and its key part).
    ready: () => spriteAtlasOn && spriteArtReady(),
    ensure: ensureSpriteAtlas,
    render: renderSpriteAtlas,
    // For the Perf section.
    setting: { key: 'atlashz', label: 'Sprite light rate (Hz, 0 every frame)', type: 'range',
      min: 0, max: 480, step: 24,
      help: 'how often the light on the sprites is worked out (the atlas pass); between, they keep the last',
      get: () => atlasHz, set: v => { atlasHz = Math.max(0, Math.round(v)); lastShaded = -Infinity; } },
    commands
  };
}
