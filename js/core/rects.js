// --- Rectangles on a face: what is left of one when others cover it ---
//
// Shared by the buried-face masks (world.js exposedFaceMask, bbmodel.js
// modelFaceMasks) and the mirror rectangles (mirrors.js). Pure, no imports.

// rect less cover, both [u0, u1, v0, v1]: up to four rectangles - the strips
// either side along u, then above and below the cover between them.
export function subtractRect(r, cv) {
  const u0 = Math.max(r[0], cv[0]), u1 = Math.min(r[1], cv[1]);
  const v0 = Math.max(r[2], cv[2]), v1 = Math.min(r[3], cv[3]);
  if (u0 >= u1 || v0 >= v1) return [r];
  return [[r[0], u0, r[2], r[3]], [u1, r[1], r[2], r[3]], [u0, u1, r[2], v0], [u0, u1, v1, r[3]]]
    .filter(p => p[1] > p[0] && p[3] > p[2]);
}
// Whether covers (each [u0, u1, v0, v1]) leave nothing of rect.
export function rectCovered(rect, covers) {
  let rest = [rect];
  for (const c of covers) {
    rest = rest.flatMap(r => subtractRect(r, c));
    if (!rest.length) return true;
  }
  return false;
}
