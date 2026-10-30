// --- Generated trees (Research/Procedural_Trees.md, Research/Oak_Growth.md) ---
//
// Branch nodes on the BoxGrid (half-metre cells, 4 texels), each hanging from
// a face neighbour, grown by a species' kit:
//
// 'oak' - an oak's architecture (Research/Oak_Growth.md) grown the way
// Dynamic Trees' limbs look: straight axis runs, turning only at a joint. A
// bare trunk to the fork; two to four scaffold limbs leaving it around the
// compass, each out runs and up runs in turn - steep at the trunk, flattening
// outward, as an oak's limbs do; off their joints shorter branches, and off
// theirs shorter again (orders), each turned toward the most room left in
// the crown, so the tips spread over the dome. No branch touches another's
// cells: limbs never graze or tangle. A tip ends in a cluster of buds - short
// twigs, where the leaves clump. Everything stays inside the crown envelope
// (a dome, flatter underneath). Age sizes the tree: full at species.pulses, a
// little more past it.
//
// 'pulses' - Dynamic Trees' growth: each pulse is a signal sent up from the
// root with an energy - steps it may take. At every branch cell it picks its
// next direction by weighted chance: up (upProbability, never below
// lowestBranch otherwise), straight on, and toward existing branch (its radius
// + 2) or air (1) - so growth follows the limbs already there. Where it steps
// into air, one new branch cell grows and the pulse ends. Age is how many
// pulses it has had.
//
// Thickness from the tips down - the pipe rule:
//   r = clamp(sqrt(r_child^2 + sum r_side^2) + taper, r, maxRadius)
// in texels; a tip is twigRadius (one texel wide).
//
// Leaves: Dynamic Trees' hydration automaton. Twig cells emit 5; an air cell
// takes the first rule (H, C, R) where at least C of its six neighbours are AT
// H, repeated until it settles. Then cut to the crown envelope, its edge
// ragged by chance, and every cell not joined through leaves to the wood
// dropped - no leaf floats.
//
// Then as boxes on the texel grid: each branch cell a cube of its width,
// joined to its parent by a box of its own width, and straight runs of one
// width merged; the leaf cells merged into as few boxes as cover them.
//
// Pure and seeded, in exact arithmetic (no transcendentals - core/rng.js): the
// same species, seed and age give the same tree on every machine.
import { createRng } from './rng.js';

export const CELL = 4;   // texels a BoxGrid cell

// Species. Lengths in cells unless said; radii in texels; [lo, hi] a seeded
// range.
export const TREE_SPECIES = {
  oak: {
    kit: 'oak',
    pulses: 160,                // the age at full size
    height: [18, 24], fork: [4, 7], scaffolds: [2, 4],
    crown: [0.5, 0.62],         // the dome's radius, a share of the height
    // A limb's share of steps up, at the trunk and out at 0.75 of the crown's
    // radius: the scaffolds, the leader, their branches.
    limbRise: [0.7, 0.12], leaderRise: [0.85, 0.35], branchRise: [0.4, 0.05],
    run: [2, 3],                // cells an out run takes
    orders: 3,                  // orders of branch off the scaffolds
    side: [0.85, 0.7, 0.5],     // the chance of a branch off a joint, by order
    childScale: 0.65,           // a branch's length: its parent's steps left, times this
    shell: 0.85,                // branches end where the envelope's measure passes this
    buds: 0.8,                  // the chance of each of a tip's four bud twigs
    hollow: 0.3,                // the crown is filled out from this envelope measure to its shell
    taper: 0.12, twigRadius: 0.5, maxRadius: 3.5,
    // Dynamic Trees' deciduous solver: [H, C, R].
    leafRules: [[5, 1, 4], [4, 2, 3], [3, 2, 2], [4, 1, 1], [3, 1, 1], [2, 1, 1]],
    textures: { bark: 'terrain_oakBark', top: 'terrain_oakBark_top', leaves: 'terrain_oakLeaves' },
    // The leaves' summer ramp, dark to light (the noise's four levels map onto
    // it): the grass's hue and step (terrain_grass.png #388800 #409000
    // #489800), a shade under it - a canopy reads as part of the same green.
    ramp: [[0x30, 0x78, 0x00], [0x38, 0x80, 0x00], [0x40, 0x88, 0x00], [0x48, 0x90, 0x00]]
  }
};

const DIRS = [[0, 1, 0], [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0]];
const UP = 0, DOWN = 5;
const key = (x, y, z) => `${x},${y},${z}`;
// A twig one texel wide: no wood box, its cell leaf instead (treeBoxes,
// growLeaves) - hidden in the leaves anyway.
const hidden = n => Math.round(2 * n.r) < 2;

// The crown envelope: a dome about height cy, radius R across, reaching `up`
// above and `down` below. <= 1 inside.
const envelope = (R, cy, up, down) => ({
  R, cy, up, down,
  measure(x, y, z) {
    const dy = y - cy, semi = dy >= 0 ? up : down;
    return (x * x + z * z) / (R * R) + (dy * dy) / (semi * semi);
  }
});

// Grows a tree: { nodes: Map key -> { at: [x, y, z], r, parent }, leaves:
// Map key -> hydration, fork: the height below which the trunk is bare,
// envelope }, cells as [x, y, z] from the root cell (0, 0, 0, the first above
// the ground).
export function growTree(species, seed, age = species.pulses) {
  const rng = createRng('tree', seed);
  const grown = species.kit === 'oak' ? growOak(species, rng, age) : growPulses(species, rng, age);
  thicken(species, grown.nodes);
  const leaves = growLeaves(species, rng, grown.nodes, grown.envelope);
  return { ...grown, leaves };
}

// The oak kit (see the top).
function growOak(species, rng, age) {
  const sc = age <= species.pulses ? Math.max(0.2, age / species.pulses) : 1 + 0.3 * (1 - species.pulses / age);
  const between = ([lo, hi]) => lo + rng.float() * (hi - lo);
  const H = Math.max(5, Math.round(between(species.height) * sc));
  const fork = Math.max(1, Math.min(H - 4, Math.round(between(species.fork) * sc)));
  const R = Math.max(1.5, H * between(species.crown));
  const base = fork + 1, top = H - 1, cy = base + (top - base) * 0.4;
  const env = envelope(R, cy, top - cy, cy - base);

  const nodes = new Map(), owner = new Map();
  const add = (at, parent, id) => {
    const n = { at, r: species.twigRadius, parent: parent ? parent.at : null };
    nodes.set(key(...at), n);
    owner.set(key(...at), id);
    return n;
  };
  // Free for branch `id`: no node there, and no other branch's cell touching
  // it (by a face, an edge or a corner) but `spare`'s - its parent's, for its
  // first steps off it (-1: any).
  const clear = (at, id, spare) => {
    if (nodes.has(key(...at))) return false;
    if (spare === -1) return true;
    for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
      const o = owner.get(key(at[0] + a, at[1] + b, at[2] + c));
      if (o !== undefined && o !== id && o !== spare) return false;
    }
    return true;
  };
  // In the crown: the dome above its middle; below, its underside counted
  // at 0.6 - a limb leaving the fork is low in the dome but well inside it.
  const inCrown = ([x, y, z]) => {
    if (y > top) return false;
    if (y < base) return true;
    const dy = y - cy;
    return dy >= 0 ? env.measure(x, y, z) <= species.shell
      : (x * x + z * z) / (R * R) + 0.6 * (dy * dy) / (env.down * env.down) <= species.shell;
  };
  // Where branches end, or are about to grow: a new branch heads for room.
  const ends = [];
  const room = p => ends.reduce((m, e) => {
    const dx = p[0] - e[0], dy = p[1] - e[1], dz = p[2] - e[2];
    return Math.min(m, dx * dx + dy * dy + dz * dz);
  }, Infinity);

  // The trunk.
  let n = add([0, 0, 0], null, 0);
  for (let y = 1; y <= fork; y++) n = add([0, y, 0], n, 0);
  const forkNode = n;

  // A branch: from node `from`, heading (hx, hz) outward, an out run then an
  // up run, over and over - the up run rise / (1 - rise) of the out run,
  // rise going from b.rise[0] at the trunk to b.rise[1] at 0.75 R out.
  // Out runs follow the heading by turns of axis (a line in runs, not in
  // single steps). It ends at `budget` steps, the crown's shell or another
  // branch; at its tip a cluster of buds - short twigs - and off its joints
  // its own branches, the next order, to grow after (breadth first).
  const queue = [];
  let ids = 1;
  const grow = b => {
    const l1 = Math.abs(b.hx) + Math.abs(b.hz) || 1, ux = b.hx / l1, uz = b.hz / l1;
    let at = b.from, steps = 0, outs = 0, px = 0, pz = 0, upAcc = 0, ended = false, out = [ux >= 0 ? 1 : -1, 0, 0];
    const joints = [];
    const step = d => {
      const next = [at.at[0] + d[0], at.at[1] + d[1], at.at[2] + d[2]];
      if (!inCrown(next) || !clear(next, b.id, steps < 2 ? b.spare : null)) return false;
      at = add(next, at, b.id);
      steps++;
      return true;
    };
    if (b.lead) for (let i = 0; i < b.lead && !ended; i++) ended = !step([0, 1, 0]);
    while (!ended && steps < b.budget) {
      const r = Math.sqrt(at.at[0] * at.at[0] + at.at[2] * at.at[2]);
      const rise = b.rise[0] + (b.rise[1] - b.rise[0]) * Math.min(1, r / (0.75 * R));
      const len = b.firstRun && !outs ? b.firstRun : rng.range(...species.run);
      const dx = (outs + len) * ux - px, dz = (outs + len) * uz - pz;
      out = Math.abs(dx) >= Math.abs(dz) ? [Math.sign(dx) || 1, 0, 0] : [0, 0, Math.sign(dz) || 1];
      for (let i = 0; i < len && !ended; i++) {
        if (!step(out)) { ended = true; break; }
        outs++; px += out[0]; pz += out[2];
      }
      if (ended) break;
      upAcc += len * rise / (1 - rise);
      const upLen = Math.floor(upAcc);
      upAcc -= upLen;
      for (let i = 0; i < upLen && !ended; i++) ended = !step([0, 1, 0]);
      if (!ended) joints.push({ node: at, steps, out });
    }
    ends.push(at.at);
    // The buds: up, on and to the sides, a twig of a cell or two each.
    const side = [out[2], 0, out[0]];
    for (const d of [[0, 1, 0], out, side, side.map(v => -v)]) {
      if (!rng.chance(species.buds)) continue;
      let tip = at;
      for (let i = rng.range(1, 2); i > 0; i--) {
        const next = [tip.at[0] + d[0], tip.at[1] + d[1], tip.at[2] + d[2]];
        if (!inCrown(next) || !clear(next, b.id, null)) break;
        tip = add(next, tip, b.id);
      }
    }
    if (b.order >= species.orders) return;
    // Off the joints (not the tip's), turned 60 or 90 degrees from the
    // branch's heading to the side with the more room, never back in.
    const S3 = Math.sqrt(3) / 2;
    for (const j of joints.slice(0, -1)) {
      if (!rng.chance(species.side[b.order])) continue;
      const left = b.budget - j.steps;
      const len = Math.max(2, Math.round(left * species.childScale));
      let best = null;
      for (const [c, sn] of [[0.5, S3], [0.5, -S3], [0, 1], [0, -1]]) {
        const hx = b.hx * c - b.hz * sn, hz = b.hx * sn + b.hz * c;
        const [x, y, z] = j.node.at;
        if (x * hx + z * hz < -0.1 * (Math.abs(x) + Math.abs(z))) continue;
        const reach = Math.min(len, 0.5 * R);
        const probe = [x + hx * reach, y + reach * species.branchRise[0], z + hz * reach];
        const score = room(probe) - (env.measure(...probe) > 1 ? 1e6 : 0);
        if (!best || score > best.score) best = { hx, hz, probe, score };
      }
      if (!best) continue;
      ends.push(best.probe);
      queue.push({ from: j.node, hx: best.hx, hz: best.hz, order: b.order + 1, budget: len,
                   rise: species.branchRise, id: ids++, spare: b.id, lead: 0 });
    }
  };

  // The scaffolds: around the compass from a seeded heading, a quarter turn
  // of jitter each, the first the leader (steeper, on up from the trunk).
  // They leave over a fork zone, a cell apart up the trunk, the leader from
  // its top: all leaving flat from one cell piled their thick first runs
  // into a shelf round the fork (2026-10-30). The rest leave with a short
  // out run, then climb.
  const count = Math.max(1, Math.round(rng.range(...species.scaffolds) * Math.min(1, sc + 0.2)));
  const turn = { 1: [1, 0], 2: [-1, 0], 3: [-0.5, Math.sqrt(3) / 2], 4: [0, 1] }[Math.min(count, 4)];
  const zone = [forkNode];
  while (zone.length < count && zone[zone.length - 1].at[1] < top - 3) {
    const t = zone[zone.length - 1];
    zone.push(add([0, t.at[1] + 1, 0], t, 0));
  }
  let hx, hz, hl;
  do { hx = rng.float() * 2 - 1; hz = rng.float() * 2 - 1; hl = hx * hx + hz * hz; } while (hl > 1 || hl < 0.05);
  for (let k = 0; k < count; k++) {
    const leader = k === 0 && count > 1;
    const from = leader ? zone[zone.length - 1] : zone[Math.min(k - (count > 1 ? 1 : 0), zone.length - 1)];
    queue.push({ from, hx: hx + (rng.float() - 0.5) * 0.4, hz: hz + (rng.float() - 0.5) * 0.4,
                 order: 0, budget: 3 * H, rise: leader ? species.leaderRise : species.limbRise,
                 id: ids++, spare: leader ? 0 : -1, lead: leader ? 2 : 0, firstRun: leader ? 0 : 2 });
    // The next heading: a turn on, from the heading before jitter.
    [hx, hz] = [hx * turn[0] - hz * turn[1], hx * turn[1] + hz * turn[0]];
  }
  for (let q = 0; q < queue.length; q++) grow(queue[q]);
  return { nodes, fork, envelope: env };
}

// The pulse kit (see the top).
function growPulses(species, rng, pulses) {
  const nodes = new Map();
  nodes.set(key(0, 0, 0), { at: [0, 0, 0], r: species.twigRadius, parent: null });

  // One pulse from node n, coming in along `dir`, `steps` taken.
  function signal(n, dir, steps, energy) {
    if (energy <= 0) return;
    const [x, y, z] = n.at;
    let next = UP;
    if (steps + 1 > species.lowestBranch) {
      const w = DIRS.map((d, i) => {
        if (i === DOWN) return 0;
        const from = dir !== null && d.every((v, a) => v === -DIRS[dir][a]);
        if (from) return 0;
        const cx = x + d[0], cy = y + d[1], cz = z + d[2];
        const there = nodes.get(key(cx, cy, cz));
        if (there) return there.r + 2;   // along an existing limb
        // New growth seeks light: away from crowding (branch cells round the
        // cell it would grow into), and sideways outward from the trunk - so
        // limbs spread over the crown instead of clumping at its middle.
        let crowd = 0;
        for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) {
          if ((a || b || c) && nodes.has(key(cx + a, cy + b, cz + c))) crowd++;
        }
        let p = 1;
        if (i === UP) p += species.upProbability;
        if (i === dir) p += 1;
        if (d[1] === 0 && d[0] * x + d[2] * z > 0) p *= 2;
        return p / (1 + crowd);
      });
      const total = w.reduce((a, b) => a + b, 0);
      let pick = rng.float() * total;
      next = w.findIndex(v => (pick -= v) < 0);
      if (next < 0) next = UP;
    }
    const d = DIRS[next], at = [x + d[0], y + d[1], z + d[2]];
    const child = nodes.get(key(...at));
    if (!child) {
      if (at[1] >= 0) nodes.set(key(...at), { at, r: species.twigRadius, parent: n.at });
    } else if (child.parent && child.parent.join() === n.at.join()) {
      signal(child, next, steps + 1, energy - 1);
    }                                       // into another limb: no loops
  }
  const root = nodes.get(key(0, 0, 0));
  for (let p = 0; p < pulses; p++) signal(root, null, 0, species.energy);

  // The envelope: from the lowest branch to above the top, as wide as the
  // branches reach.
  let top = 0, reach = 0;
  for (const n of nodes.values()) {
    top = Math.max(top, n.at[1]);
    reach = Math.max(reach, Math.sqrt(n.at[0] * n.at[0] + n.at[2] * n.at[2]));
  }
  const cy = (species.lowestBranch + top + 2) / 2, ry = Math.max(1.5, (top + 2 - species.lowestBranch) / 2);
  return { nodes, fork: species.lowestBranch - 1, envelope: envelope(Math.max(2, reach + 1.5), cy, ry, ry) };
}

// The pipe rule, from the tips down: children are always added after their
// parents, so the nodes in reverse are tips first.
function thicken(species, nodes) {
  const list = [...nodes.values()];
  const area = new Map();
  for (let i = list.length - 1; i >= 0; i--) {
    const n = list[i], a = area.get(n);
    if (a !== undefined) n.r = Math.min(species.maxRadius, Math.max(n.r, Math.sqrt(a) + species.taper));
    if (n.parent) {
      const p = nodes.get(key(...n.parent));
      area.set(p, (area.get(p) || 0) + n.r * n.r);
    }
  }
}

// Leaves (see the top): the automaton round the twigs, cut to the envelope,
// joined to the wood - on a dense grid round the tree. A hidden twig's own
// cell is leaf (a hydration of 5), so the twig leaves no hole.
function growLeaves(species, rng, nodes, env) {
  const M = 4;   // the automaton reaches 3 past a twig
  let lo = [0, 0, 0], hi = [0, 0, 0];
  for (const n of nodes.values()) for (let a = 0; a < 3; a++) {
    lo[a] = Math.min(lo[a], n.at[a]); hi[a] = Math.max(hi[a], n.at[a]);
  }
  lo = lo.map((v, a) => (a === 1 ? 0 : v - M)); hi = hi.map(v => v + M);
  const sx = hi[0] - lo[0] + 1, sy = hi[1] - lo[1] + 1, sz = hi[2] - lo[2] + 1;
  const at = (x, y, z) => ((y - lo[1]) * sz + (z - lo[2])) * sx + (x - lo[0]);
  const STEP = [sz * sx, 1, -1, sx, -sx, -sz * sx];   // DIRS as index steps
  // Wood: 1 shown, 2 a hidden twig; the twigs (r < 1) emit 5.
  const wood = new Uint8Array(sx * sy * sz), source = new Uint8Array(sx * sy * sz);
  for (const n of nodes.values()) {
    const i = at(...n.at);
    wood[i] = hidden(n) ? 2 : 1;
    if (n.r < 1) source[i] = 1;
  }
  // The cells the automaton may fill: air within reach of a twig, inside the
  // grid's one-cell rim (so every neighbour is in it).
  const near = [], seen = new Uint8Array(sx * sy * sz);
  for (const n of nodes.values()) {
    if (n.r >= 1) continue;
    for (let dx = -3; dx <= 3; dx++) for (let dy = -2; dy <= 3; dy++) for (let dz = -3; dz <= 3; dz++) {
      const x = n.at[0] + dx, y = n.at[1] + dy, z = n.at[2] + dz;
      if (y < 1 || y >= hi[1] || x <= lo[0] || x >= hi[0] || z <= lo[2] || z >= hi[2]) continue;
      const i = at(x, y, z);
      if (!seen[i] && !wood[i]) { seen[i] = 1; near.push(i); }
    }
  }
  let leaf = new Uint8Array(sx * sy * sz), next = new Uint8Array(sx * sy * sz);
  const counts = new Uint8Array(8);
  for (let pass = 0; pass < 8; pass++) {
    let changed = false;
    for (const i of near) {
      counts.fill(0);
      for (const d of STEP) { const j = i + d; counts[source[j] ? 5 : wood[j] ? 0 : leaf[j]]++; }
      let v = 0;
      for (const [h, c, r] of species.leafRules) if (counts[h] >= c) { v = r; break; }
      next[i] = v;
      if (v !== leaf[i]) changed = true;
    }
    [leaf, next] = [next, leaf];
    if (!changed) break;
  }
  for (let i = 0; i < wood.length; i++) if (wood[i] === 2) leaf[i] = 5;
  // The envelope, a little past it where the twigs reach its shell, ragged.
  const xyz = i => [i % sx + lo[0], Math.floor(i / (sx * sz)) + lo[1], Math.floor(i / sx) % sz + lo[2]];
  for (const i of near) {
    if (!leaf[i]) continue;
    if (env.measure(...xyz(i)) > 1.1 + 0.25 * rng.float() || (leaf[i] === 1 && rng.float() < 0.3)) leaf[i] = 0;
  }
  // The crown filled: its outer part (the envelope's measure from `hollow` to
  // 1, ragged) wherever a branch is within 3 cells. The twigs' clumps alone
  // left gaps between the limbs - a sparse crown of lobes (2026-10-30).
  if (species.hollow !== undefined) {
    const floor = env.cy - env.down;
    for (const n of nodes.values()) {
      if (n.at[1] < floor) continue;
      for (let dx = -3; dx <= 3; dx++) for (let dy = -3; dy <= 3; dy++) for (let dz = -3; dz <= 3; dz++) {
        if (dx * dx + dy * dy + dz * dz > 9) continue;
        const x = n.at[0] + dx, y = n.at[1] + dy, z = n.at[2] + dz;
        if (y < 1 || y >= hi[1] || x <= lo[0] || x >= hi[0] || z <= lo[2] || z >= hi[2]) continue;
        const i = at(x, y, z);
        if (wood[i] || leaf[i]) continue;
        const m = env.measure(x, y, z);
        if (m < species.hollow || m > 1 + 0.15 * rng.float()) continue;
        leaf[i] = 2;
        if (!seen[i]) { seen[i] = 1; near.push(i); }
      }
    }
  }
  // Joined: flood from the leaves touching shown wood.
  const joined = new Uint8Array(sx * sy * sz), queue = [];
  const cells = near.concat([...nodes.values()].filter(hidden).map(n => at(...n.at)));
  for (const i of cells) {
    if (leaf[i] && STEP.some(d => wood[i + d] === 1)) { joined[i] = 1; queue.push(i); }
  }
  while (queue.length) {
    const i = queue.pop();
    for (const d of STEP) { const j = i + d; if (leaf[j] && !joined[j]) { joined[j] = 1; queue.push(j); } }
  }
  const leaves = new Map();
  for (const i of cells) if (joined[i]) leaves.set(key(...xyz(i)), leaf[i]);
  return leaves;
}

// A box [x0, y0, z0, x1, y1, z1] in texels from the root cell's floor centre
// (the model's origin: the middle of the top of the block it stands on).
const cellCentre = ([x, y, z]) => [x * CELL, y * CELL + CELL / 2, z * CELL];
const around = (c, w) => { const lo = c - Math.floor(w / 2); return [lo, lo + w]; };

// Merges boxes that share a cross-section and touch along the third axis,
// and drops boxes inside others.
function mergeBoxes(boxes) {
  let list = boxes.map(b => b.slice());
  let merged = true;
  while (merged) {
    merged = false;
    list = list.filter((b, i) => !list.some((o, j) => j !== i && o[0] <= b[0] && o[1] <= b[1] && o[2] <= b[2] &&
      o[3] >= b[3] && o[4] >= b[4] && o[5] >= b[5] && (j < i || o.some((v, a) => v !== b[a]))));
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const a = list[i], b = list[j];
        for (let ax = 0; ax < 3; ax++) {
          const o1 = (ax + 1) % 3, o2 = (ax + 2) % 3;
          if (a[o1] !== b[o1] || a[o1 + 3] !== b[o1 + 3] || a[o2] !== b[o2] || a[o2 + 3] !== b[o2 + 3]) continue;
          if (a[ax + 3] < b[ax] || b[ax + 3] < a[ax]) continue;
          a[ax] = Math.min(a[ax], b[ax]); a[ax + 3] = Math.max(a[ax + 3], b[ax + 3]);
          list.splice(j--, 1);
          merged = true;
          break;
        }
      }
    }
  }
  return list;
}

// Cells [x, y, z] as boxes [x0, y0, z0, x1, y1, z1] in cells, greedily: from
// each cell not yet covered, a run along x, grown along z while the next row
// is all free cells, then along y while the next layer is.
function mergeCells(cells) {
  const free = new Set(cells.map(c => key(...c)));
  const boxes = [];
  const sorted = cells.slice().sort((a, b) => a[1] - b[1] || a[2] - b[2] || a[0] - b[0]);
  const row = (x0, x1, y, z) => { for (let x = x0; x < x1; x++) if (!free.has(key(x, y, z))) return false; return true; };
  const take = (x0, x1, y0, y1, z0, z1) => {
    for (let y = y0; y < y1; y++) for (let z = z0; z < z1; z++) for (let x = x0; x < x1; x++) free.delete(key(x, y, z));
  };
  for (const [x, y, z] of sorted) {
    if (!free.has(key(x, y, z))) continue;
    let x1 = x + 1, z1 = z + 1, y1 = y + 1;
    while (free.has(key(x1, y, z))) x1++;
    while (row(x, x1, y, z1)) z1++;
    for (;;) {
      let full = true;
      for (let zz = z; zz < z1 && full; zz++) full = row(x, x1, y1, zz);
      if (!full) break;
      y1++;
    }
    take(x, x1, y, y1, z, z1);
    boxes.push([x, y, z, x1, y1, z1]);
  }
  return boxes;
}

// The boxes as model cubes (core/bbmodel.js's form), every face wrapped onto
// its tile (bbmodel.js transformCube): bark (tile 0) on the wood's sides, end
// grain (tile 1) on its tops and bottoms, leaves (tile 2) all round. Image
// rows run down, so v runs against y on the sides. The trunk's root box first.
const FACE_AXES = [[2, 1], [2, 1], [0, 2], [0, 2], [0, 1], [0, 1]];   // as bbmodel.js
export function treeCubes({ wood, leaves }) {
  const cube = (b, tileOf) => ({
    name: '', light: false, lightLevel: null, rects: [], min: b.slice(0, 3), max: b.slice(3),
    faces: FACE_AXES.map(([u, v], d) => ({
      u: { axis: u, o: 12 * tileOf(d), d: 1 }, v: { axis: v, o: 0, d: v === 1 ? -1 : 1 }, wrap: true }))
  });
  const root = wood.reduce((best, b) => (b[1] <= 0 && (!best || b[4] > best[4]) ? b : best), null);
  const woodSorted = root ? [root, ...wood.filter(b => b !== root)] : wood;
  return [...woodSorted.map(b => cube(b, d => (d === 2 || d === 3 ? 1 : 0))),
          ...leaves.map(b => cube(b, () => 2))];
}

// The tree as boxes: { wood: [box], leaves: [box] } (see above).
export function treeBoxes({ nodes, leaves }) {
  const width = n => Math.max(1, Math.round(2 * n.r));
  const wood = [];
  for (const n of nodes.values()) {
    if (hidden(n) && n.parent) continue;
    const c = cellCentre(n.at), w = width(n);
    const cube = [0, 1, 2].map(a => around(c[a], w));
    // Down to the ground for the root.
    if (!n.parent) cube[1][0] = 0;
    wood.push([cube[0][0], cube[1][0], cube[2][0], cube[0][1], cube[1][1], cube[2][1]]);
    if (n.parent) {
      const p = cellCentre(n.parent);
      const lo = c.map((v, a) => Math.min(v, p[a])), hi = c.map((v, a) => Math.max(v, p[a]));
      const span = [0, 1, 2].map(a => (lo[a] === hi[a] ? around(c[a], w) : [lo[a], hi[a]]));
      wood.push([span[0][0], span[1][0], span[2][0], span[0][1], span[1][1], span[2][1]]);
    }
  }
  const woodBoxes = mergeBoxes(wood);
  // Leaf cells clear of the wood, merged greedily: runs along x, then
  // stacked along z, then along y.
  const inWood = c => {
    const lo = [c[0] * CELL - CELL / 2, c[1] * CELL, c[2] * CELL - CELL / 2];
    return woodBoxes.some(b => b[0] < lo[0] + CELL && b[3] > lo[0] && b[1] < lo[1] + CELL && b[4] > lo[1] &&
      b[2] < lo[2] + CELL && b[5] > lo[2]);
  };
  const cells = [...leaves.keys()].map(k => k.split(',').map(Number)).filter(c => !inWood(c));
  const leafBoxes = mergeCells(cells).map(([x0, y0, z0, x1, y1, z1]) =>
    [x0 * CELL - CELL / 2, y0 * CELL, z0 * CELL - CELL / 2, x1 * CELL - CELL / 2, y1 * CELL, z1 * CELL - CELL / 2]);
  // cells: those drawn (shownLeafCells); all: every leaf cell - what shades (core/leaves.js).
  return { wood: woodBoxes, leaves: leafBoxes, leafCells: cells.length, cells: shownLeafCells(cells, inWood), all: cells };
}

// The leaf cells drawn (gpu/leaves.js): the crown's shell, LEAF_SHELL cells
// deep from the air - its faces are cut out, so the eye gets a layer or two
// in and finds the branches; past the shell the crown is bare inside, as an
// oak's is. Each with its height in the crown (0 at its lowest leaf cell, 1
// at its highest: the albedo darkens toward the bottom) and which of its
// faces are drawn, a bit each in LEAF_FACES order: a face to air, a face to
// the bare inside, and between two shell cells one face, the lower cell's
// + face (never two on one plane: they would fight). A face on wood is not
// drawn. As [x, y, z, height, faces] in cells.
// +x, -x, +y, -y, +z, -z: three's BoxGeometry's face order.
export const LEAF_FACES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
export const LEAF_SHELL = 3;
function shownLeafCells(cells, inWood) {
  const leaf = new Set(cells.map(c => key(...c)));
  const solid = (x, y, z) => leaf.has(key(x, y, z)) || inWood([x, y, z]);
  let lo = Infinity, hi = -Infinity;
  for (const c of cells) { lo = Math.min(lo, c[1]); hi = Math.max(hi, c[1]); }
  // The shell: cells open to air, then each layer in from them.
  const shell = new Set();
  let front = cells.filter(([x, y, z]) => LEAF_FACES.some(d => !solid(x + d[0], y + d[1], z + d[2])));
  for (const c of front) shell.add(key(...c));
  for (let k = 1; k < LEAF_SHELL && front.length; k++) {
    const next = [];
    for (const [x, y, z] of front) for (const d of LEAF_FACES) {
      const n = [x + d[0], y + d[1], z + d[2]], nk = key(...n);
      if (leaf.has(nk) && !shell.has(nk)) { shell.add(nk); next.push(n); }
    }
    front = next;
  }
  const out = [];
  for (const [x, y, z] of cells) {
    if (!shell.has(key(x, y, z))) continue;
    let faces = 0;
    LEAF_FACES.forEach((d, f) => {
      const n = [x + d[0], y + d[1], z + d[2]], nk = key(...n);
      if (!solid(...n) || (leaf.has(nk) && (!shell.has(nk) || f % 2 === 0))) faces |= 1 << f;
    });
    if (faces) out.push([x, y, z, (y - lo) / Math.max(1, hi - lo), faces]);
  }
  return out;
}
