// --- Sprite shadows: analytic cards, and the baked proxies ---
//
// Sprites are not in the World map, so they cast through two paths of their
// own: cards each light tests analytically (core/cards.js, the default), and
// proxies voxelised into the distance field (core/occluders.js, off; kept for
// GI and reflections). The cards have other readers too: the sprites' contact
// AO and, turned toward the camera, the reflections.
//
// ctx:
//   renderer, camera
//   pivot        the camera rig's pivot: the billboards face from it to the camera
//   sunLight     the DirectionalLight: its position is the sun's direction
//   sunAngular() the sun's angular size (the card parity's penumbra)
//   sprites      { player, enemy: { mesh, texture }, trees() the tree meshes,
//                  treeTexture, mirrored(sprite) whether it is turned round }
//   normalMap(name, generate)  a sprite's normal map: an _n.png override or generated
//   grids(), textures()  the shadow cascades' grids and distance textures
//   shadowsOn()  whether the marched shading is on
//   rebuild()    the shading rebuilt (a kernel flag changed, or the atlas arrived)
//   onReady()    the cards' atlas has arrived (the plain AO takes their discs)
import { VOXEL_SIZE } from '../render.js';
import { CASCADE_COUNT, cascadeExtentMetres, cascadeVoxelMetres } from '../../core/field.js';
import { createCardBindings, createCardAtlasTexture, createCardColourTexture, writeCardBindings,
         MAX_CARDS } from '../../gpu/cards.js';
import { runGPUCards } from '../../gpu/diagnostics.js';
import { updateDistanceTexture, SEAM_BAND_VOXELS } from '../../gpu/field.js';
import { POINT_CARD_CAPACITY } from '../../core/lights.js';
import { createOccluder, placeOccluder, applyOccluders } from '../../core/occluders.js';
import { maskFromRGBA, createSilhouetteSet, facingToward,
         AXIS_X, AXIS_Z } from '../../core/silhouette.js';
import { createCard, cardFacingFor, buildCardAtlas, packCardInstances,
         cullCardsForLight, cardsVisibility, CARD_PAD, SUN_CARD_LOD_SHIFT,
         SUN_CARD_LOD_MAX } from '../../core/cards.js';
import { spriteNormals } from '../../core/normals.js';
import { flip } from '../engine.js';

export function createCards({ renderer, camera, pivot, sunLight, sunAngular, sprites, normalMap,
                              grids, textures, shadowsOn, rebuild, onReady }) {
  // --- The shadow-casting proxies ---
  //
  // A sprite is a camera-facing quad with no thickness, so it is not something the
  // distance field can be baked from directly - see occluders.js for why
  // voxelising the quad itself would make a character's shadow change shape as the
  // camera orbits. Each caster gets a world-locked box instead, and the box is all
  // the grid ever sees.
  //
  // Sized against the sprite quads rather than guessed. The character quad is
  // VOXEL_SIZE x VOXEL_SIZE*2, translated so its base sits at the mesh position -
  // so mesh.position is the FEET, which is what placeOccluder wants. The figure
  // does not fill its quad (pixel art carries transparent padding), so the proxy
  // is narrower and shorter than the geometry: roughly the body, not the canvas.
  const CHARACTER_PROXY = { width: VOXEL_SIZE * 0.5, height: VOXEL_SIZE * 1.6 };
  // A tree's quad is VOXEL_SIZE*2 x VOXEL_SIZE*3, and the same reasoning applies -
  // a trunk and canopy inside a mostly-empty rectangle.
  const TREE_PROXY = { width: VOXEL_SIZE * 0.9, height: VOXEL_SIZE * 2.4 };

  // --- Reading the cutout off the sprite ---
  //
  // The only DOM-shaped part of the silhouette path, and the reason it lives here
  // rather than in silhouette.js: a canvas cannot exist in the headless test
  // suite, so the module that does the maths takes a plain mask array and this
  // function is the one thing that has to be trusted by inspection.
  //
  // Drawn at native size with smoothing off. The sprite is 12 x 24 and the voxel
  // grid it is about to become is also 12 x 24 - any scaling here would resample
  // an exact correspondence into an approximate one.
  function imageRGBA(image) {
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(image, 0, 0);
    return ctx.getImageData(0, 0, image.width, image.height).data;
  }

  function maskFromTexture(image) {
    return { mask: maskFromRGBA(imageRGBA(image), image.width, image.height),
             w: image.width, h: image.height };
  }

  // Textures load asynchronously, so a silhouette cannot be built at module scope.
  // Until one arrives the proxy is a box - which is a working shadow, not a
  // placeholder, so there is no first-frame flicker of nothing being cast.
  //
  // RESOLVED BY POLLING, NOT BY AN EVENT, and that is not laziness. THREE.Texture
  // dispatches exactly one event, 'dispose' - there is no 'load', and
  // TextureLoader only assigns texture.image inside ImageLoader's own callback. An
  // addEventListener('load') on the texture therefore never fires, which is how
  // every silhouette here silently stayed a box: the fallback is a working shadow,
  // so nothing looked broken, it just looked rectangular.
  //
  // Checking each frame cannot miss a load the way a missing event can, costs one
  // truthiness test per proxy once resolved, and self-heals if a texture is ever
  // swapped at runtime.
  const pendingSilhouettes = [];

  function attachSilhouette(proxy, texture, widthMetres, heightMetres) {
    pendingSilhouettes.push({ proxy, texture, widthMetres, heightMetres });
  }

  function resolveSilhouettes() {
    for (let i = pendingSilhouettes.length - 1; i >= 0; i--) {
      const p = pendingSilhouettes[i];
      const img = p.texture.image;
      if (!img || !img.width) continue;
      const { mask, w, h } = maskFromTexture(img);
      // One per cascade. A single silhouette is expressed in the voxels of the
      // level it was built for, so sharing it across levels makes the character
      // twice as wide in C1 and four times in C2 - see createSilhouetteSet.
      p.proxy.silhouettes = createSilhouetteSet({
        mask, w, h,
        widthMetres: p.widthMetres, heightMetres: p.heightMetres
      });
      pendingSilhouettes.splice(i, 1);
    }
  }

  // How the cards are turned. 'sun' points each card's normal along the sun's
  // azimuth, so the full silhouette always faces the light and the fixed-facing
  // degeneracy - a card edge-on to the sun casting a line - cannot arise.
  //
  // This is still not a CAMERA-facing proxy, which is what S9 rules out and for a
  // specific reason: a camera-facing card's thickness is camera-relative, so the
  // shadow changes shape as the player orbits. A sun-facing card is world-locked
  // with respect to the viewer and re-orients only when the light moves, which is
  // what a real silhouette does anyway.
  //
  // 'x' and 'z' lock to a world plane instead, kept for comparison.
  let cardMode = 'sun';

  function cardFacing() {
    if (cardMode === 'x') return AXIS_X;
    if (cardMode === 'z') return AXIS_Z;
    return facingToward(sunLight.position.x, sunLight.position.z);
  }

  const playerProxy = createOccluder('player', CHARACTER_PROXY);
  const enemyProxy = createOccluder('enemy', CHARACTER_PROXY);
  attachSilhouette(playerProxy, sprites.player.texture, VOXEL_SIZE, VOXEL_SIZE * 2);
  attachSilhouette(enemyProxy, sprites.enemy.texture, VOXEL_SIZE, VOXEL_SIZE * 2);
  // Trees never move, so after the first frame they cost exactly nothing - the
  // snapped box compares equal and applyOccluders does no work. They are in the
  // list for the same reason the characters are: they are sprites, so they are
  // not in the World map, so without a proxy they cast nothing at all.
  const treeProxies = [];
  // The baked proxies are OFF by default: analytic cards (above) are the shadow
  // path now, and running both would have every sprite occlude twice. Kept
  // switchable because the bake is what reflections and GI will need.
  let proxiesOn = false;

  // --- Analytic cards: the shadow path sprites actually use ---
  //
  // The distance field is geometry shared by every light, so a sprite baked into
  // it faces one direction and every other light gets the wrong silhouette. These
  // live OUTSIDE the field: each light intersects each card, turned to face that
  // light. See cards.js.
  //
  // The baked proxies above are kept and still work - bxb.proxies() turns them on
  // - but they are off by default now, because the two would double-count. What
  // they are for from here is reflections and GI (S9), which need sprites present
  // in the grid rather than tested against a ray.
  const cardTypes = [];          // { card, typeIndex }, one per sprite texture
  let cardAtlas = null;
  let cardAtlasTex = null;
  const pendingCards = [];

  // The texture arrives as a THUNK rather than a value: it is read in
  // resolveCards, each frame until its image has loaded, so a texture swapped
  // in later is still the one built.
  // name is the texture's file name, which is what a normal-map override is
  // matched by.
  function registerCard(key, name, getTexture, widthMetres, heightMetres) {
    const entry = { key, name, getTexture, widthMetres, heightMetres, card: null,
                    normalTex: null };
    pendingCards.push(entry);
    return entry;
  }

  const bobCard = registerCard('bob', 'character_Bob', () => sprites.player.texture,
                               VOXEL_SIZE, VOXEL_SIZE * 2);
  const evilCard = registerCard('evil', 'character_EvilBob', () => sprites.enemy.texture,
                                VOXEL_SIZE, VOXEL_SIZE * 2);
  const treeCard = registerCard('tree', 'decor_tree', () => sprites.treeTexture,
                                VOXEL_SIZE * 2, VOXEL_SIZE * 3);

  // Same polling as the silhouettes, and for the same reason: THREE.Texture has no
  // 'load' event, so there is nothing to listen to.
  function resolveCards() {
    let built = false;
    for (let i = pendingCards.length - 1; i >= 0; i--) {
      const p = pendingCards[i];
      const tex = p.getTexture();
      const img = tex && tex.image;
      if (!img || !img.width) continue;
      // Per sprite, so one unreadable texture cannot take the rest down with it -
      // and so the reason is reported rather than surfacing as "still loading"
      // forever, which is what it looked like.
      let mask, w, h, rgba;
      try {
        rgba = imageRGBA(img);
        w = img.width; h = img.height;
        mask = maskFromRGBA(rgba, w, h);
      } catch (err) {
        p.error = String(err && err.message || err);
        continue;
      }
      // How much of the sprite is opaque. Kept because a mask that comes back
      // fully solid is indistinguishable from a working one until you look at the
      // shadow: every card becomes its own bounding box and casts a slab.
      let solid = 0;
      for (let k = 0; k < mask.length; k++) solid += mask[k];
      p.coverage = solid / mask.length;
      p.card = createCard({ mask, w, h, rgba,
                            widthMetres: p.widthMetres, heightMetres: p.heightMetres });
      // From the card's own distance transform: the silhouette the shadow uses
      // is the silhouette the bevel follows.
      const card = p.card;
      p.normalTex = normalMap(p.name, () => ({
        data: spriteNormals(card.dt, card.w, card.cols, card.rows, CARD_PAD), w, h
      }));
      p.typeIndex = cardTypes.length;
      cardTypes.push(p);
      pendingCards.splice(i, 1);
      built = true;
    }
    if (!built) return false;
    // Repacked whole rather than appended: the atlas is a handful of small
    // bitmaps, and a rect that moved would silently point every instance of that
    // card at the wrong pixels.
    cardAtlas = buildCardAtlas(cardTypes.map(t => t.card));
    return true;
  }

  // Every sprite that should cast, with no facing yet - the facing is per light
  // and is filled in per light below.
  function cardCasters() {
    const out = [];
    const add = (entry, sprite, flip) => {
      if (!entry || !entry.card || !sprite.visible) return;
      out.push({
        card: entry.card, typeIndex: entry.typeIndex,
        // The card's centre, not its feet: the mask is centred on the quad.
        centre: { x: sprite.position.x,
                  y: sprite.position.y + entry.heightMetres / 2,
                  z: sprite.position.z },
        flip, enabled: true
      });
    };
    add(bobCard, sprites.player.mesh, sprites.mirrored(sprites.player.mesh));
    add(evilCard, sprites.enemy.mesh, sprites.mirrored(sprites.enemy.mesh));
    for (const m of sprites.trees()) add(treeCard, m, false);
    return out;
  }

  // Rebuilt each frame rather than kept in sync by hand. The set is small and the
  // alternative is a registration path that has to be remembered at every spawn
  // and despawn - a corpse that keeps casting is exactly the bug that costs.
  function activeProxies() {
    if (!proxiesOn) return [];
    const out = [];
    const playerSprite = sprites.player.mesh, enemySprite = sprites.enemy.mesh;
    if (playerSprite.visible) {
      out.push(placeOccluder(playerProxy, playerSprite.position.x,
                             playerSprite.position.y, playerSprite.position.z,
                             sprites.mirrored(playerSprite)));
    }
    if (enemySprite.visible) {
      out.push(placeOccluder(enemyProxy, enemySprite.position.x,
                             enemySprite.position.y, enemySprite.position.z,
                             sprites.mirrored(enemySprite)));
    }
    const treeMeshes = sprites.trees();
    for (let i = 0; i < treeMeshes.length; i++) {
      const m = treeMeshes[i];
      if (!m.visible) continue;
      if (!treeProxies[i]) {
        treeProxies[i] = createOccluder('tree' + i, TREE_PROXY);
        // One silhouette per tree rather than one shared: createSilhouette is
        // cheap and called once, and sameBox compares the object by identity, so a
        // shared instance would be fine too - but a per-tree one leaves room for
        // variants (a stump, a dead tree) without restructuring.
        attachSilhouette(treeProxies[i], sprites.treeTexture, VOXEL_SIZE * 2, VOXEL_SIZE * 3);
      }
      out.push(placeOccluder(treeProxies[i], m.position.x, m.position.y, m.position.z));
    }
    return out;
  }

  // Write the proxies into every cascade and report whether anything changed.
  //
  // Every cascade, not just C0: a shadow ray leaves C0 after 18 m and continues in
  // C1, so a caster present only in the fine level would stop casting at exactly
  // the distance the ray crossed over - a character's shadow vanishing partway
  // across the ground for no visible reason.
  //
  // The texture upload is conditional on the SAME flag, which is the point of
  // applyOccluders returning one: a 3 MB Data3DTexture re-upload per cascade per
  // frame would cost more than the bake it is reporting, and in a turn-based scene
  // most frames have nobody mid-step.
  // One slice per LIGHT, because the facing is the whole point: a card turned
  // toward the sun is not the card the torch needs, and one slice can only hold
  // one orientation. The sun has its own buffer; the point lights share one, each
  // owning the slice its light-list row points at.
  let sunCards = null, lightCards = null;
  // A third set, for reflections: every sprite turned the way its billboard is
  // drawn, toward the camera's heading, with the sprites' colour atlas beside the
  // silhouette one. A reflection ray that crosses one of these sees the sprite.
  let viewCards = null;
  let cardPack = null;
  let cardsReady = false;

  function ensureCardBindings() {
    if (sunCards) return;
    sunCards = createCardBindings(MAX_CARDS, 'bxbSunCards');
    sunCards.lodShift.value = SUN_CARD_LOD_SHIFT;
    sunCards.lodMax.value = SUN_CARD_LOD_MAX;
    lightCards = createCardBindings(POINT_CARD_CAPACITY, 'bxbLightCards');
    viewCards = createCardBindings(MAX_CARDS, 'bxbViewCards');
    cardPack = new Float32Array(POINT_CARD_CAPACITY * 16);
  }

  // Fill one light's buffer: cull to what that light can reach, turn each card to
  // face it, pack, upload.
  function writeCardsFor(binding, casters, lightPos, radiusMetres, directional) {
    const n = packCardsFor(casters, lightPos, radiusMetres, directional,
                           cardPack, binding.capacity);
    writeCardBindings(binding, cardPack, n);
    return n;
  }

  // Cull, face and pack one light's cards into `out`, at most `capacity` of them.
  // Returns how many were written. Split from the upload so the point lights can
  // pack their slices back to back into one buffer.
  function packCardsFor(casters, lightPos, radiusMetres, directional, out, capacity) {
    const near = cullCardsForLight(casters, lightPos, radiusMetres, capacity);
    for (const c of near) {
      // A directional light is the same direction from everywhere, so every card
      // turns the same way. A point light is in a DIFFERENT direction from every
      // sprite, which is the case a baked field cannot express at all.
      c.facing = directional
        ? facingToward(lightPos.x, lightPos.z)
        : cardFacingFor(c.centre, lightPos.x, lightPos.z);
    }
    return packCardInstances(near, cardAtlas, out, capacity);
  }

  let lastCardCount = 0;

  // Which cascade a card stands in, as the lod createCardsTSL reads: the level,
  // plus how far into the last SEAM_BAND_VOXELS of it toward the next. null when
  // it is outside every level - no terrain shadow reaches there either, so the
  // card is dropped before it costs any fragment anything.
  function cardLod(centre) {
    for (let l = 0; l < CASCADE_COUNT; l++) {
      const g = grids()[l];
      if (!g) return 0;
      const side = cascadeExtentMetres(l);
      const ex = Math.min(centre.x - g.origin.x, g.origin.x + side - centre.x);
      const ez = Math.min(centre.z - g.origin.z, g.origin.z + side - centre.z);
      if (ex < 0 || ez < 0) continue;
      const edge = Math.min(ex, ez) / cascadeVoxelMetres(l);
      const t = Math.min(1, edge / SEAM_BAND_VOXELS);
      return l + (1 - t * t * (3 - 2 * t));   // 1 - smoothstep, as c0WeightTSL
    }
    return null;
  }

  function updateCards() {
    if (pendingCards.length && resolveCards()) {
      // The atlas exists now where it did not before, and the kernel branches on
      // whether there is one - so the material has to be rebuilt once. A one-frame
      // hitch on load, and nothing afterwards.
      cardAtlasTex = createCardAtlasTexture(cardAtlas);
      ensureCardBindings();
      sunCards.atlas = lightCards.atlas = viewCards.atlas = cardAtlasTex;
      if (viewCards.colour) viewCards.colour.dispose();
      viewCards.colour = createCardColourTexture(cardAtlas);
      for (const b of [sunCards, lightCards, viewCards]) {
        b.atlasSize.value.set(cardAtlas.width, cardAtlas.height);
      }
      cardsReady = true;
      if (shadowsOn()) { rebuild(); }
      // The plain material's AO takes the sprites' contact discs from here on.
      onReady();
    }
    if (!cardsReady || !cardsOn) {
      lastCardCount = 0; frameCasters = null;
      if (sunCards) sunCards.count.value = 0;   // no discs either
      return;
    }

    // Each card's cascade LOD - with shadows off the cascades are not kept, and
    // the only reader is the contact disc, which has no LOD: every card counts.
    const casters = cardCasters()
      .filter(c => (c.lod = shadowsOn() ? cardLod(c.centre) : 0) !== null);
    // The sun is directional: its "position" is a direction, and everything is
    // within reach of it.
    lastCardCount = writeCardsFor(sunCards, casters, sunLight.position, Infinity, true);
    // The view cards: the direction from the pivot to the camera IS the way the
    // billboards face (rotation.y = heading), so every card takes it. Nearest the
    // camera first, so the cap drops the sprites least likely to be seen.
    const near = cullCardsForLight(casters, camera.position, Infinity, viewCards.capacity);
    const facing = facingToward(camera.position.x - pivot.position.x,
                                camera.position.z - pivot.position.z);
    for (const c of near) c.facing = facing;
    writeCardBindings(viewCards, cardPack,
                      packCardInstances(near, cardAtlas, cardPack, viewCards.capacity));
    // The point lights' cards are packed with the light list, in lighting.update -
    // after the torch has moved this frame rather than before.
    frameCasters = casters;
  }
  let frameCasters = null;

  let cardsOn = true;

  function updateOccluders() {
    // Cheap once everything has loaded, and the only thing that turns a box back
    // into a cutout - see attachSilhouette for why this is not an event handler.
    if (pendingSilhouettes.length) resolveSilhouettes();

    const proxies = activeProxies();
    const facing = cardFacing();
    const shadowGrids = grids();
    for (let l = 0; l < shadowGrids.length; l++) {
      if (!shadowGrids[l]) continue;
      if (applyOccluders(shadowGrids[l], proxies, facing)) {
        updateDistanceTexture(textures()[l]);
      }
    }
  }

  const commands = {
    proxies: {
      help: 'toggle the shadow-casting proxies for sprites, or resize the character box',
      usage: 'bxb.proxies()  |  bxb.proxies(0.75, 2.4)  - width and height in metres',
      run: (w = null, h = null) => {
        if (w !== null) CHARACTER_PROXY.width = Math.max(0.05, w);
        if (h !== null) CHARACTER_PROXY.height = Math.max(0.05, h);
        const sizeOnly = w !== null || h !== null;
        if (sizeOnly) {
          // A resize is not a toggle. The box is remembered by its snapped extent,
          // so changing the size makes every proxy compare unequal next frame and
          // rewrite itself - no invalidation needed beyond setting the number.
          playerProxy.width = enemyProxy.width = CHARACTER_PROXY.width;
          playerProxy.height = enemyProxy.height = CHARACTER_PROXY.height;
        } else {
          proxiesOn = !proxiesOn;
        }
        if (!proxiesOn) return 'proxies off - sprites cast nothing, terrain still does';
        if (!shadowsOn()) return 'proxies on - run bxb.shadows() to see them';
        // Which path each proxy is actually on. A silhouette that failed to build
        // falls back to its box silently and still casts, so without this the
        // difference between "the cutout is working" and "the cutout never loaded"
        // is a judgement call about how blocky a shadow looks.
        const sil = playerProxy.silhouettes && playerProxy.silhouettes[0];
        const shape = sil
          ? `CUTOUT ${playerProxy.silhouettes.map(v => v.cols + 'x' + v.rows).join(' / ')}` +
            ` per cascade, facing ${cardMode}`
          : `BOX ${CHARACTER_PROXY.width.toFixed(2)} x ` +
            `${CHARACTER_PROXY.height.toFixed(2)} m (cutout not loaded yet)`;
        const trees = treeProxies.filter(p => p && p.silhouettes).length;
        return `proxies on - characters cast as ${shape}, ` +
               `${trees}/${treeProxies.length} trees as cutouts. ` +
               `Expect the shadow to STEP a texel at a time as you walk, not slide, ` +
               `and to have a GAP between the legs - that gap is the whole ` +
               `difference from a box. bxb.cardface() to compare sun-facing ` +
               `cards against ones locked to a world plane.`;
      }
    },
    cardparity: {
      help: 'run the same rays through the card shader and the CPU reference and diff them',
      run: async (n = 128, line = false) => {
        if (!cardsReady) return 'no cards yet - bxb.shadows() first, then retry';
        const casters = cardCasters();
        for (const c of casters) {
          c.facing = facingToward(sunLight.position.x, sunLight.position.z);
        }
        const near = cullCardsForLight(casters, sunLight.position, Infinity,
                                       sunCards.capacity);
        const count = packCardInstances(near, cardAtlas, cardPack, sunCards.capacity);
        writeCardBindings(sunCards, cardPack, count);

        // Spread over the ground around the player, along the sun. The rays that
        // matter are the ones that graze a card, so the sample box is sized to the
        // sprites rather than to the map.
        const L = sunLight.position.clone().normalize();
        const angular = sunAngular();
        const slope = angular ? Math.tan(angular / 2) : 0.06;
        const samples = [];
        if (line) {
          // A WALK down-sun from the player instead of a scatter. Random points
          // report whether the two agree on average; a line reports WHERE they
          // stop agreeing, which is what a visible edge needs.
          for (let i = 0; i < n; i++) {
            const d = (i / (n - 1)) * 24;
            samples.push({
              from: { x: sprites.player.mesh.position.x - L.x * d, y: 0.8,
                      z: sprites.player.mesh.position.z - L.z * d },
              dir: { x: L.x, y: L.y, z: L.z }, maxDist: 12
            });
          }
        } else {
          for (let i = 0; i < n; i++) {
            samples.push({
              from: { x: sprites.player.mesh.position.x + (Math.random() - 0.5) * 12,
                      y: 0.75 + Math.random() * 2,
                      z: sprites.player.mesh.position.z + (Math.random() - 0.5) * 12 },
              dir: { x: L.x, y: L.y, z: L.z },
              maxDist: 12
            });
          }
        }

        const gpu = await runGPUCards(renderer, sunCards, samples, slope);
        if (line) {
          const rows = samples.map((sm, i) => {
            const cpu = cardsVisibility(near, sm.from, sm.dir, 12, slope);
            return `${((i / (n - 1)) * 24).toFixed(1).padStart(5)}m cpu ` +
                   `${cpu.toFixed(3)} gpu ${gpu[i].toFixed(3)}` +
                   (Math.abs(cpu - gpu[i]) > 0.1 ? '  <-- DIVERGE' : '');
          });
          return rows.join(String.fromCharCode(10));
        }
        let worst = 0, worstAt = -1, sum = 0, shadowed = 0;
        for (let i = 0; i < n; i++) {
          const cpu = cardsVisibility(near, samples[i].from, samples[i].dir, 12, slope);
          if (cpu < 0.99) shadowed++;
          const e = Math.abs(cpu - gpu[i]);
          sum += e;
          if (e > worst) { worst = e; worstAt = i; }
        }
        const s0 = samples[worstAt];
        return `${count} cards, ${n} rays, ${shadowed} shadowed on CPU. ` +
               `mean |diff| ${(sum / n).toFixed(4)}, worst ${worst.toFixed(4)} ` +
               `at (${s0.from.x.toFixed(2)}, ${s0.from.y.toFixed(2)}, ` +
               `${s0.from.z.toFixed(2)}) cpu ${cardsVisibility(near, s0.from, s0.dir, 12, slope).toFixed(3)} ` +
               `gpu ${gpu[worstAt].toFixed(3)}. A worst above ~0.05 means the ` +
               `shader and the reference disagree, not that either is noisy.`;
      }
    },
    cards: {
      help: 'analytic sprite shadows - each light gets the cutout turned to face IT',
      usage: 'bxb.cards()  toggles',
      run: () => {
        cardsOn = !cardsOn;
        // Build on demand rather than waiting for a frame. The resolve normally
        // rides the render loop, which does not run while the window is hidden -
        // so asking about the cards from a paused tab would report them as still
        // loading forever, which is a property of the tab and not of the cards.
        if (cardsOn) updateCards();
        // Whether the pass has a card loop in it at all is baked into the kernel,
        // so this is a material rebuild rather than a uniform write - the same
        // trade the torch makes, and for the same reason: off has to be free.
        if (shadowsOn()) { rebuild(); }
        if (!cardsOn) return 'sprite cards off - nothing but terrain casts';
        if (!cardsReady) {
          // Say WHY, not just that it has not happened. "Still loading" covers a
          // texture that genuinely has not arrived and one whose image is never
          // going to be readable, and those need opposite fixes.
          const st = pendingCards.map(p => {
            const t = p.getTexture();
            const im = t && t.image;
            return `${p.key}:${p.error ? 'FAILED ' + p.error :
                    !t ? 'no-texture' :
                    !im ? 'no-image' :
                    !im.width ? 'width-0' : 'ready-but-unbuilt'}`;
          }).join(' ');
          return `cards on - no atlas. pending ${pendingCards.length}, ` +
                 `built ${cardTypes.length}. ${st || '(nothing pending - the ' +
                 'build ran and then something after it failed)'}`;
        }
        const cov = cardTypes.map(t =>
          `${t.key} ${t.card.cols}x${t.card.rows} ${(t.coverage * 100).toFixed(0)}% opaque`
        ).join(', ');
        return `cards on - [${cov}] in a ` +
               `${cardAtlas.width}x${cardAtlas.height} atlas, ${lastCardCount} ` +
               `casting this frame, cap ${MAX_CARDS} per light. Each light tests ` +
               `the cutout turned to face IT, so a torch and the sun disagree ` +
               `about which way Bob is side-on - which is the point, and what a ` +
               `single baked field cannot do. Analytic, so there is no voxel size ` +
               `and no cascade seam.`;
      }
    },
    cardface: {
      help: 'how the sprite cutouts are turned - toward the sun, or locked to a world plane',
      usage: 'bxb.cardface()  cycles sun -> x -> z  |  bxb.cardface("sun")',
      run: (mode = null) => {
        const order = { sun: 'x', x: 'z', z: 'sun' };
        cardMode = mode && order[mode] !== undefined ? mode : order[cardMode];
        // Nothing to invalidate by hand: the facing is part of the remembered box,
        // so sameBox sees it change and every cutout rewrites itself next frame.
        if (cardMode === 'sun') {
          const f = cardFacing();
          return `cutouts face the SUN - card normal along its azimuth ` +
                 `(${(-f.uz).toFixed(2)}, ${f.ux.toFixed(2)}). The full ` +
                 `silhouette always faces the light, so it cannot thin to a line. ` +
                 `bxb.cardface() to lock it to a world plane and see the ` +
                 `difference; move the sun with bxb.light() and the cutouts turn ` +
                 `with it.`;
        }
        return `cutouts LOCKED to the ${cardMode === 'x' ? 'XY' : 'ZY'} plane. ` +
               `Swing the sun round with bxb.light() and watch the shadow thin ` +
               `toward a line as it goes edge-on - that degeneracy is exactly what ` +
               `the sun-facing mode exists to remove.`;
      }
    },
  };

  return {
    name: 'cards',
    options: () => ({
      cards: cardsReady && cardsOn ? sunCards : null,
      lightCards: cardsReady && cardsOn ? lightCards : null,
      viewCards: cardsReady && cardsOn ? viewCards : null
    }),
    // The sun's and the view's cards, and this frame's casters.
    update: updateCards,
    // The proxies written into the cascades (shadows on, after the re-origin).
    updateOccluders,
    get ready() { return cardsReady; },
    get sunBindings() { return sunCards; },
    // Every binding, for the console's live references.
    bindings: () => ({ sunCards, lightCards, viewCards }),
    // The card types built so far (the GI bounces them), and this frame's casters.
    types: cardTypes,
    casters: () => frameCasters,
    // A sprite mesh's normal map, from its card (null until it has resolved).
    normalFor: m => (m === sprites.player.mesh ? bobCard : m === sprites.enemy.mesh ? evilCard : treeCard).normalTex,
    // The point lights' slices (render/features/lights.js): a light's cards
    // packed into the shared buffer, then committed.
    lightSlices: {
      active: () => !!(frameCasters && lightCards && cardsReady && cardsOn),
      pack: (position, radius, used, room) => packCardsFor(frameCasters, position, radius,
                                                           false, cardPack.subarray(used * 16), room),
      commit: used => writeCardBindings(lightCards, cardPack, used)
    },
    commands,
    settings: { title: 'Sprites', settings: [
      { key: 'cards', label: 'Card shadows', type: 'toggle',
        get: () => cardsOn, set: v => flip(cardsOn, v, () => commands.cards.run()) },
      { key: 'cardface', label: 'Cutout facing', type: 'select', options: ['sun', 'x', 'z'],
        get: () => cardMode, set: v => commands.cardface.run(v) },
      { key: 'proxies', label: 'Baked proxies', type: 'toggle',
        help: 'sprites voxelised into the field - off by default, kept for GI / reflections',
        get: () => proxiesOn, set: v => flip(proxiesOn, v, () => commands.proxies.run()) }
    ]}
  };
}
