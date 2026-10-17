// --- Point lights: the torch, placed lamps, model lights, and the light list ---
//
// Every light is a plain record { name, position, level, colour, sourceRadius }.
// Each frame the live ones (core/lights.js liveLights) are packed into the
// light list the shading reads, with the budgets deciding which get a shadow
// march, sprite-card shadows, marches at reflection hits and mirror light.
//
// ctx:
//   scene        lamp markers go in it
//   holder       { sprite, facingRight() }: who holds the torch, and which way
//                the art faces (the torch leads on that side)
//   shadowsOn()  whether the marched shading is on (for the messages)
//   cards        a light's sprite cards (the cards live with the sun's):
//                active() whether there are any this frame, pack(light, used,
//                room) packs one light's at slot used and returns how many,
//                commit(used) uploads them
//   mirrorRects() the mirrors this frame, for the mirror-light budget
import * as THREE from 'three/webgpu';
import { TORCH_COLOUR, TORCH_LEVEL, TORCH_HEIGHT, TORCH_FORWARD, TORCH_SIDE, MAX_LIGHT_LEVEL,
         LIGHT_SOURCE_RADIUS, clampLevel, lightRadiusMetres, pointPenumbraMetres, MAX_LIGHTS,
         LIGHT_FLOATS, POINT_CARD_CAPACITY, liveLights, packLight, parseColour,
         DEFAULT_SHADOW_BUDGET, DEFAULT_LIGHT_CUTOFF, pickShadowed, easeShadowWeight,
         DEFAULT_CARD_BUDGET, DEFAULT_HIT_SHADOW_BUDGET, DEFAULT_MIRROR_LIGHT_BUDGET } from '../../core/lights.js';
import { pickMirrored } from '../../core/mirrors.js';
import { WorldModels } from '../../core/world.js';
import { createLightBindings, writeLightBindings } from '../../gpu/pointlights.js';
import { MAX_CARDS } from '../../gpu/cards.js';
import { flip } from '../engine.js';

// The lights of placed models (world.js WorldModels): one per lightSource
// cube, at its centre, as big as the cube - its penumbra is the cube's. Its
// level from the cube's name (lightSource_12), else MODEL_LIGHT_LEVEL. The
// colour is the cube's own pixels (render.js lightColourOf), read when its
// texture has decoded; warm white until then.
const MODEL_LIGHT_LEVEL = 10;
const MODEL_LIGHT_FALLBACK = 0xffe2b0;

// A few colours to cycle through when a lamp is dropped without one, picked to
// be tellable apart where they overlap.
const LAMP_COLOURS = [0xff8a3d, 0x4da6ff, 0x7dff6a, 0xff5ad1, 0xfff1c9];

export function createLights({ scene, holder, shadowsOn, cards, mirrorRects }) {
  // --- The torch: a dynamic point light on the player ---
  //
  // The torch is row 0 of the light list whenever it is lit, and nothing more
  // special than that: off means it is left out of the list, which costs the
  // kernel nothing, and on is one more row. No material rebuild either way.
  let torchOn = false;
  let torchLevel = TORCH_LEVEL;
  // How big the flame is, in metres. A uniform rather than a kernel constant: it
  // is the knob the penumbra is found by eye with, so a sweep has to be a uniform
  // write rather than a material rebuild.
  let torchSourceRadius = LIGHT_SOURCE_RADIUS;
  // How far out in front of the holder, and to their side, the flame is held, in
  // metres. Plain numbers rather than uniforms: they are read on the CPU when the
  // position is written each frame, so there is nothing on the GPU to rebuild when
  // either changes.
  let torchOffset = TORCH_FORWARD;
  let torchSide = TORCH_SIDE;
  const torchLight = {
    name: 'torch', position: new THREE.Vector3(), level: 0,
    colour: TORCH_COLOUR, sourceRadius: LIGHT_SOURCE_RADIUS
  };

  // Off is level 0, which liveLights drops before the list reaches the GPU.
  function syncTorchLight() {
    torchLight.level = torchOn ? clampLevel(torchLevel) : 0;
    torchLight.sourceRadius = torchSourceRadius;
    return torchLight;
  }

  // --- Placed lights: everything that is not the torch ---
  //
  // Plain records, the same shape as torchLight. Each carries a small unlit marker
  // so it can be found in the scene; the marker is display only - it is not in the
  // field and casts nothing.
  const placedLights = [];
  // The level Alt+L and a bare bxb.lamp() drop at - set from the settings panel.
  let lampLevel = 10;

  const modelLights = [];
  // Profiling only (bxb.gpuprofile): the model lights left out.
  let modelLightsOn = true;
  function syncModelLights() {
    modelLights.length = 0;
    for (const m of WorldModels) {
      for (const l of m.lights) {
        modelLights.push({
          name: m.name, level: l.level ?? MODEL_LIGHT_LEVEL, sourceRadius: l.radius,
          position: new THREE.Vector3(...l.position),
          get colour() { return m.model.lightColour ?? MODEL_LIGHT_FALLBACK; }
        });
      }
    }
  }
  syncModelLights();

  let lightBindings = null;
  let lightPack = null;
  let lastLightCount = 0;
  let lastLightCards = 0;
  let lastShadowedCount = 0;
  let lastCardLights = 0;
  let lastLightTime = 0;
  // Perf settings. The budget is CPU-side - it decides which rows get a shadow
  // weight - and the cutoff is a uniform on the bindings, so neither rebuilds.
  let shadowBudget = DEFAULT_SHADOW_BUDGET;
  // Of those, how many test sprite cards too. The rest still march the field.
  let cardBudget = DEFAULT_CARD_BUDGET;
  // Of those, how many march at secondary hits, and how many throw mirror light
  // (lights.js DEFAULT_HIT_SHADOW_BUDGET). Uniform data, no rebuild.
  let hitShadowBudget = DEFAULT_HIT_SHADOW_BUDGET;
  let mirrorLightBudget = DEFAULT_MIRROR_LIGHT_BUDGET;
  // Per light-list row: whether it throws mirror light (the mirrors read it).
  let lightMirrorRows = [];

  // Every light there is: the torch, placed lamps, then the placed models' -
  // those off inside bxb.gpuprofile, to price them.
  function allLights() {
    return modelLightsOn ? [syncTorchLight(), ...placedLights, ...modelLights]
                         : [syncTorchLight(), ...placedLights];
  }

  function ensureLightBindings() {
    if (lightBindings) return lightBindings;
    lightBindings = createLightBindings(MAX_LIGHTS);
    lightBindings.cutoff.value = DEFAULT_LIGHT_CUTOFF;
    lightPack = new Float32Array(MAX_LIGHTS * LIGHT_FLOATS);
    return lightBindings;
  }

  // The list, packed and uploaded, with each light's cards packed into its slice
  // of the shared card buffer. Once per frame, after the torch has moved.
  function updateLights() {
    if (!lightBindings) return;
    const focus = holder.sprite.position;
    const live = liveLights(allLights(), MAX_LIGHTS, focus);
    const withCards = cards.active();
    // The shadow budget: which lights get a march, ranked by what they give the
    // player's surroundings. Each light's weight eases toward in or out, so a
    // light crossing the budget fades its shadow rather than popping it.
    const now = performance.now();
    const dt = lastLightTime ? Math.min(0.1, (now - lastLightTime) / 1000) : 0;
    lastLightTime = now;
    const shadowed = pickShadowed(live, focus, shadowBudget);
    // The card budget ranks by the same score, so it is a subset of the shadowed
    // set whenever it is the smaller of the two.
    const carded = pickShadowed(live, focus, Math.min(cardBudget, shadowBudget));
    const hitShadowed = pickShadowed(live, focus, Math.min(hitShadowBudget, shadowBudget));
    const mirrored = pickMirrored(live, mirrorRects() || [], focus, mirrorLightBudget);
    // The hit-shadow and mirror budgets fade as the shadow budget does.
    for (const l of live) {
      l.hitWeight = l.hitWeight === undefined ? (hitShadowed.has(l) ? 1 : 0)
                  : easeShadowWeight(l.hitWeight, hitShadowed.has(l), dt);
      l.mirrorWeight = l.mirrorWeight === undefined ? (mirrored.has(l) ? 1 : 0)
                     : easeShadowWeight(l.mirrorWeight, mirrored.has(l), dt);
    }
    lightMirrorRows = live.map(l => l.mirrorWeight > 0);
    let used = 0, marching = 0, cardLights = 0;
    for (let i = 0; i < live.length; i++) {
      const l = live[i];
      const start = used;
      let count = 0;
      // A new light starts at its target rather than fading in from nothing.
      l.shadowWeight = l.shadowWeight === undefined ? (shadowed.has(l) ? 1 : 0)
                     : easeShadowWeight(l.shadowWeight, shadowed.has(l), dt);
      l.cardWeight = l.cardWeight === undefined ? (carded.has(l) ? 1 : 0)
                   : easeShadowWeight(l.cardWeight, carded.has(l), dt);
      if (l.shadowWeight > 0) marching++;
      // Cards only for a light that marches AND is in the card budget - they are
      // only ever read by its march, and weighted out past the budget.
      if (withCards && l.shadowWeight > 0 && l.cardWeight > 0) {
        cardLights++;
        const room = Math.min(MAX_CARDS, POINT_CARD_CAPACITY - used);
        if (room > 0) {
          count = cards.pack(l, used, room);
          used += count;
        }
      }
      packLight(lightPack, i, l, start, count, l.shadowWeight, l.cardWeight, l.hitWeight, l.mirrorWeight);
    }
    if (withCards) cards.commit(used);
    writeLightBindings(lightBindings, lightPack, live.length);
    lastLightCount = live.length;
    lastLightCards = used;
    lastShadowedCount = marching;
    lastCardLights = cardLights;
  }

  function lampMarker(colour) {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.08, 8, 6),
                             new THREE.MeshBasicMaterial({ color: colour }));
    m.raycast = () => {};   // never picked by the hover / click raycasts
    return m;
  }

  function addLamp({ level = lampLevel, colour = null, position = null,
                     sourceRadius = LIGHT_SOURCE_RADIUS } = {}) {
    const c = colour === null
      ? LAMP_COLOURS[placedLights.length % LAMP_COLOURS.length]
      : parseColour(colour);
    const pos = position ? new THREE.Vector3(position.x, position.y, position.z)
                         : heldLightPosition(new THREE.Vector3());
    const lamp = { name: 'lamp', position: pos, level: clampLevel(level),
                   colour: c, sourceRadius, marker: lampMarker(c) };
    lamp.marker.position.copy(pos);
    scene.add(lamp.marker);
    placedLights.push(lamp);
    return lamp;
  }

  function removeLamp(lamp) {
    const i = placedLights.indexOf(lamp);
    if (i < 0) return;
    placedLights.splice(i, 1);
    scene.remove(lamp.marker);
    lamp.marker.geometry.dispose();
    lamp.marker.material.dispose();
  }

  function describeLight(l, i) {
    const p = l.position;
    return `${i}: ${l.name} L${l.level} #${parseColour(l.colour).toString(16).padStart(6, '0')} ` +
           `at (${p.x.toFixed(1)}, ${p.y.toFixed(1)}, ${p.z.toFixed(1)}), ` +
           `${lightRadiusMetres(l.level).toFixed(1)} m reach`;
  }

  function listLights() {
    const all = allLights();
    const live = new Set(liveLights(all, MAX_LIGHTS, holder.sprite.position));
    const rows = all.map((l, i) => describeLight(l, i) + (live.has(l) ? '' : '  (not shaded)'));
    return `${live.size}/${MAX_LIGHTS} lights shaded, ${lastShadowedCount} with ` +
           `shadows (budget ${shadowBudget}), ${lastCardLights} with sprite shadows ` +
           `(budget ${cardBudget}), ${lastLightCards} point-light ` +
           `cards this frame (cap ${MAX_CARDS} per light, ${POINT_CARD_CAPACITY} ` +
           `shared)\n` + rows.join('\n');
  }

  // The flame follows the sprite every frame. Nothing else has to happen: the
  // distance field is geometry, so a light that moves invalidates none of it.
  //
  // Held OUT IN FRONT rather than at the sprite's centre. The character is a
  // silhouette in the distance field now, so a flame at its own position is inside
  // solid geometry - the marcher hits an occluder on the first step and the torch
  // shadows itself. TORCH_FORWARD and TORCH_HEIGHT are what carry it clear.
  const torchForward = new THREE.Vector3();

  function updateTorchPosition() {
    if (!torchOn) return;
    heldLightPosition(torchLight.position);
  }

  // Where a light held by Bob sits right now. The torch follows this every frame;
  // a dropped lamp stays where it was when dropped.
  function heldLightPosition(out) {
    const sprite = holder.sprite;
    // The sprite's own +Z in world space, which is where it is facing - it is
    // yawed to meet the camera every frame, so this is "toward the viewer" without
    // having to re-derive it from the camera's heading. Reading the orientation the
    // sprite actually has, rather than the one it was asked for, keeps the flame
    // from lagging a frame behind on a camera spin.
    sprite.getWorldDirection(torchForward);
    // Flattened. getWorldDirection carries the character's pitch lean as well as
    // its yaw, and letting that through would bob the flame up and down as the
    // camera tilts - the offset is meant to be horizontal.
    torchForward.y = 0;
    if (torchForward.lengthSq() < 1e-8) torchForward.set(0, 0, 1);
    else torchForward.normalize();

    // The sprite's own right, in world space: X = up cross Z for a right-handed
    // basis, which for a flattened forward is (fz, -fx). Taken from the forward
    // vector rather than from the camera's heading for the same reason forward
    // itself is - one source of truth for where the sprite is actually pointing.
    const rightX = torchForward.z, rightZ = -torchForward.x;
    // Bob mirrors when he turns, so the hand holding the torch swaps with him.
    //
    // facingRight() is true when the art is mirrored to face RIGHT (movement.js
    // faceSprite), and the torch LEADS - it goes to the side he is facing, so
    // facing right puts it on screen-right. Which side reads as correct is a call
    // about the art rather than about the geometry, so torchSide may be negative
    // and bxb.torchside() flips it without a rebuild.
    const side = torchSide * (holder.facingRight() ? 1 : -1);

    const ox = torchForward.x * torchOffset + rightX * side;
    const oz = torchForward.z * torchOffset + rightZ * side;

    return out.set(
      sprite.position.x + ox,
      sprite.position.y + TORCH_HEIGHT,
      sprite.position.z + oz);
  }

  // Toggling or re-levelling the torch is a light-list write - see syncTorchLight.
  // Off leaves it out of the list, which costs nothing; on costs one more march,
  // capped at the light's own radius.
  function toggleTorch(level = null, flame = null, forward = null) {
    if (flame !== null) torchSourceRadius = Math.max(0, flame);
    if (forward !== null) torchOffset = Math.max(0, forward);
    if (level !== null) torchLevel = clampLevel(level);
    // A flame-size or reach sweep on a lit torch is a CPU-side write - the flame
    // radius is a uniform, the offset is read when the position is set - so
    // neither must be mistaken for a toggle.
    const sizeOnly = level === null && (flame !== null || forward !== null);
    if (!sizeOnly) torchOn = level === null ? !torchOn : torchLevel > 0;
    syncTorchLight();
    updateTorchPosition();
    if (!torchOn) return 'torch off';
    if (!shadowsOn()) return `torch level ${torchLevel} - run bxb.shadows() to see it`;
    // What to expect, so the softening can be checked rather than admired: a
    // penumbra that does NOT shrink as the torch backs away is the sun's cone
    // left in by mistake, not a flame that is too small.
    const at = (h, d) => pointPenumbraMetres(h, d, torchSourceRadius).toFixed(2);
    // Named explicitly because a flame that vanishes when the offset is wound to
    // zero is not a broken light - it is the light sitting inside the character's
    // own silhouette and being occluded by it.
    const right = holder.facingRight();
    const held = `held ${torchOffset.toFixed(2)} m in front and ` +
                 `${torchSide.toFixed(2)} m to Bob's ` +
                 `${(torchSide < 0) === right ? 'left' : 'right'}` +
                 ` while he faces ${right ? 'right' : 'left'}` +
                 ` (bxb.torchside() flips which), ` +
                 `${TORCH_HEIGHT.toFixed(2)} m up`;
    return `torch on - ${held}, level ${torchLevel}/${MAX_LIGHT_LEVEL}, reaching ` +
           `${lightRadiusMetres(torchLevel).toFixed(1)} m, ` +
           `#${TORCH_COLOUR.toString(16).toUpperCase()}, ` +
           `${(torchSourceRadius * 2).toFixed(2)} m flame. Soft by CONE TRACE, the ` +
           `same marcher the sun uses - the only difference is that its cone opens ` +
           `toward the light instead of staying a fixed angle, so an occluder ` +
           `softens MORE the nearer it sits to the flame and LESS as the torch ` +
           `backs away. Expect a penumbra of ${at(1, 2)} m from an occluder 1 m ` +
           `out with the torch at 2 m, ${at(1, 6)} m with it at 6 m. Contact stays ` +
           `hard either way. bxb.torch(${torchLevel}, ${(torchSourceRadius * 2).toFixed(2)}) ` +
           `to resize the flame.`;
  }

  // Alt+L: a lamp where Bob's torch would be; the message for the console.
  function dropLamp() {
    const l = addLamp();
    return `lamp dropped - ${describeLight(l, placedLights.length)}` +
           (shadowsOn() ? '' : ' - bxb.shadows() to see it');
  }
  function clearLamps() {
    for (const l of [...placedLights]) removeLamp(l);
  }

  const budget = n => Math.max(0, Math.min(MAX_LIGHTS, Math.round(n)));
  const commands = {
    torchside: {
      help: 'which side of Bob the torch is held, and how far - negative swaps sides',
      usage: 'bxb.torchside()  flips  |  bxb.torchside(0.3)  |  bxb.torchside(-0.3)',
      run: (m = null) => {
        // Sign only, or sign and distance. Flipping is the common case - which
        // side reads right depends on the sprite art, not on the geometry, so it
        // is settled by eye rather than derived.
        torchSide = m === null ? -torchSide : m;
        if (!torchOn) return `torch side ${torchSide.toFixed(2)} m - bxb.torch() to light it`;
        const right = holder.facingRight();
        const facing = right ? 'right' : 'left';
        const at = (torchSide < 0) === right ? 'left' : 'right';
        return `torch ${Math.abs(torchSide).toFixed(2)} m to screen-${at} ` +
               `while Bob faces ${facing}. Turn him round and it should swap; ` +
               `if it does not move at all the sprite is not mirroring.`;
      }
    },
    torch: {
      help: 'toggle the torch, set its level (0-15), flame size, or how far out it is held',
      usage: 'bxb.torch()  |  bxb.torch(15)  |  bxb.torch(0) off  |  ' +
             'bxb.torch(12, 0.5) flame diameter in m  |  bxb.torch(12, 0.5, 0.45) reach in m',
      run: (level = null, flameDiameter = null, forward = null) =>
        toggleTorch(level, flameDiameter === null ? null : flameDiameter / 2, forward)
    },
    lamp: {
      help: 'drop a point light where Bob holds the torch (Alt+L). Level 1-15, any colour',
      usage: "bxb.lamp()  |  bxb.lamp(12)  |  bxb.lamp(12, '#ff8800')  |  " +
             "bxb.lamp(12, '#ff8800', x, y, z) at world metres",
      run: (level = lampLevel, colour = null, x = null, y = null, z = null) => {
        const position = x === null ? null : { x, y: y ?? 1.5, z: z ?? 0 };
        const l = addLamp({ level, colour, position });
        return `lamp ${describeLight(l, placedLights.length)}` +
               (shadowsOn() ? '' : ' - bxb.shadows() to see it');
      }
    },
    lamps: {
      help: 'list every light, or clear/remove placed lamps',
      usage: "bxb.lamps()  |  bxb.lamps('clear')  |  bxb.lamps('pop')  |  bxb.lamps(i, level) re-level",
      run: (op = null, level = null) => {
        if (op === 'clear') { clearLamps(); return 'lamps cleared'; }
        if (op === 'pop') { if (placedLights.length) removeLamp(placedLights.at(-1)); return listLights(); }
        // Row 0 is the torch, so placed lamp i is row i.
        if (typeof op === 'number' && level !== null) {
          const l = placedLights[op - 1];
          if (!l) return `no lamp ${op} - rows start at 1, row 0 is the torch`;
          l.level = clampLevel(level);
        }
        return listLights();
      }
    },
    shadowbudget: {
      help: 'how many point lights get a shadow march per frame; the rest light unshadowed',
      usage: 'bxb.shadowbudget(12)  |  bxb.shadowbudget(16) every light',
      run: (n = DEFAULT_SHADOW_BUDGET) => {
        shadowBudget = budget(n);
        return `shadow budget ${shadowBudget} - the ${shadowBudget} lights that ` +
               `give Bob's surroundings the most get a march, the rest light ` +
               `unshadowed. Changes fade over a quarter second.`;
      }
    },
    cardbudget: {
      help: 'how many point lights also cast sprite (card) shadows; the rest shadow terrain only',
      usage: 'bxb.cardbudget(6)  |  bxb.cardbudget(16) every shadowed light',
      run: (n = DEFAULT_CARD_BUDGET) => {
        cardBudget = budget(n);
        return `card budget ${cardBudget} - the ${cardBudget} most important ` +
               `shadowed lights test sprite cards. The others still march the ` +
               `field, so walls still block them; only sprites stop casting.`;
      }
    },
    hitshadowbudget: {
      help: 'of the shadowed lights, how many also march at reflection / through-glass hits',
      usage: 'bxb.hitshadowbudget(1)  |  bxb.hitshadowbudget(16) every shadowed light',
      run: (n = DEFAULT_HIT_SHADOW_BUDGET) => {
        hitShadowBudget = budget(n);
        return `hit-shadow budget ${hitShadowBudget} - in reflections and through glass, ` +
               `the ${hitShadowBudget} most important lights march; the rest light what they show unshadowed.`;
      }
    },
    mirrorlightbudget: {
      help: 'how many point lights throw mirror light (reflected light off mirrors)',
      usage: 'bxb.mirrorlightbudget(1)  |  bxb.mirrorlightbudget(16) every light',
      run: (n = DEFAULT_MIRROR_LIGHT_BUDGET) => {
        mirrorLightBudget = budget(n);
        return `mirror-light budget ${mirrorLightBudget} - the ${mirrorLightBudget} most important ` +
               `lights glare off mirrors; the sun always does.`;
      }
    },
    lightcutoff: {
      help: 'the faint outer band of each light that is not marched (0 = none)',
      usage: 'bxb.lightcutoff(1/100)  |  bxb.lightcutoff(0)',
      run: (c = DEFAULT_LIGHT_CUTOFF) => {
        ensureLightBindings().cutoff.value = Math.max(0, Math.min(0.5, Number(c) || 0));
        return `light cutoff ${lightBindings.cutoff.value.toFixed(4)} - contribution ` +
               `below this is dropped and the rest rescaled, so the edge stays smooth`;
      }
    }
  };

  return {
    name: 'lights',
    options: () => ({ lights: ensureLightBindings() }),
    // The light list's bindings, made on first use (the LPV reads them too).
    bindings: ensureLightBindings,
    update: updateLights,
    // After Bob has moved, before the list is packed.
    moveTorch: updateTorchPosition,
    toggleTorch,
    get torchOn() { return torchOn; },
    // Per light-list row: whether it throws mirror light this frame.
    get mirrorRows() { return lightMirrorRows; },
    // After a world edit: the placed models' lights read again.
    syncModelLights,
    get modelLightCount() { return modelLights.length; },
    get modelLightsOn() { return modelLightsOn; },
    set modelLightsOn(on) { modelLightsOn = !!on; },
    dropLamp, clearLamps,
    commands,
    settings: { title: 'Torch & lamps', settings: [
      { key: 'torch', label: 'Torch (T)', type: 'toggle',
        get: () => torchOn, set: v => flip(torchOn, v, () => toggleTorch()) },
      { key: 'torchlevel', label: 'Torch level', type: 'range', min: 1, max: MAX_LIGHT_LEVEL, step: 1,
        get: () => torchLevel,
        // Re-levelling an unlit torch must not light it, which toggleTorch(level) would.
        set: v => { torchLevel = clampLevel(v); syncTorchLight(); } },
      { key: 'flame', label: 'Flame size (m)', type: 'range', min: 0, max: 1, step: 0.05,
        get: () => torchSourceRadius * 2,
        set: v => { torchSourceRadius = v / 2; syncTorchLight(); } },
      { key: 'forward', label: 'Held forward (m)', type: 'range', min: 0, max: 1, step: 0.05,
        get: () => torchOffset, set: v => { torchOffset = v; updateTorchPosition(); } },
      { key: 'side', label: 'Held side (m)', type: 'range', min: -0.6, max: 0.6, step: 0.05,
        get: () => torchSide, set: v => { torchSide = v; updateTorchPosition(); } },
      { key: 'lamplevel', label: 'New lamp level', type: 'range', min: 1, max: MAX_LIGHT_LEVEL, step: 1,
        get: () => lampLevel, set: v => { lampLevel = clampLevel(v); } },
      { key: 'droplamp', label: 'Drop lamp (Alt+L)', type: 'action', button: 'Drop',
        run: () => commands.lamp.run() },
      { key: 'poplamp', label: 'Remove last lamp', type: 'action', button: 'Remove',
        run: () => commands.lamps.run('pop') },
      { key: 'clearlamps', label: 'Clear lamps', type: 'action', button: 'Clear',
        run: () => commands.lamps.run('clear') },
      { key: 'listlights', label: 'List lights', type: 'action', button: 'List',
        run: () => { const t = commands.lamps.run(); console.log(t); return t; } }
    ]},
    // The light budgets, for the Perf section.
    perfSettings: [
      { key: 'budget', label: 'Shadowed lights', type: 'range', min: 0, max: MAX_LIGHTS, step: 1,
        help: 'how many point lights get a shadow march; the rest light unshadowed',
        get: () => shadowBudget, set: v => commands.shadowbudget.run(v) },
      { key: 'cardbudget', label: 'Sprite-shadow lights', type: 'range', min: 0, max: MAX_LIGHTS, step: 1,
        help: 'of the shadowed lights, how many also cast sprite (card) shadows - walls still block the rest',
        get: () => cardBudget, set: v => commands.cardbudget.run(v) },
      { key: 'hitbudget', label: 'Lights shadowed in reflections', type: 'range', min: 0, max: MAX_LIGHTS, step: 1,
        help: 'of the shadowed lights, how many march at reflection and through-glass hits - the rest light those unshadowed',
        get: () => hitShadowBudget, set: v => commands.hitshadowbudget.run(v) },
      { key: 'mirrorbudget', label: 'Mirror-light lights', type: 'range', min: 0, max: MAX_LIGHTS, step: 1,
        help: 'how many point lights glare off mirrors (the sun always does)',
        get: () => mirrorLightBudget, set: v => commands.mirrorlightbudget.run(v) },
      { key: 'cutoff', label: 'Light cutoff', type: 'range', min: 0, max: 0.1, step: 0.005,
        help: 'faint outer band of each light that is not marched',
        get: () => (lightBindings ? lightBindings.cutoff.value : DEFAULT_LIGHT_CUTOFF),
        format: v => (v > 0 ? '1/' + Math.round(1 / v) : 'off'),
        set: v => commands.lightcutoff.run(v) }
    ]
  };
}
