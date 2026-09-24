import { loadWorldMirror } from './world.js';
import { createBoxGridAt, scrollForHandoff } from './boxgrid.js';

// --- The field worker ---
//
// Keeps a mirror of every cascade and does the re-origin bake here, off the
// main thread. Only the strips that scrolled in go back; the main grid applies
// them with applyHandoff. Messages are handled in order, so a reset queued
// after a scroll always wins.
//
// The bake reads World for which blocks exist and which are glass, so the
// mirror's World holds each key with its material (world.js worldMirrorEntries).
const grids = [];

self.onmessage = ({ data: m }) => {
  if (m.type === 'world') {
    loadWorldMirror(m.blocks);
  } else if (m.type === 'reset') {
    grids[m.level] = createBoxGridAt(m.x, m.z, null, m.level);
  } else if (m.type === 'scroll') {
    const out = scrollForHandoff(grids[m.level], m.x, m.z);
    self.postMessage({ ...out, level: m.level, gen: m.gen }, [out.data.buffer]);
  }
};
