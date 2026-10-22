// --- BoxGrid: the world's 3 x 3 x 3 subdivision ---
//
// Every block divides into BOX_CELLS^3 boxes of BOX_METRES - half a metre,
// four texels. A block's centre is a box centre (its middle box), so box i
// along an axis is centred on i x BOX_METRES and block b holds boxes 3b - 1 ..
// 3b + 1. The grid the world's finer systems share:
//   - spell smoke's cubes (core/smoke.js)
//   - basic mode's lighting (planned)
//   - fluid physics (planned)
//
// Not the field: the signed distance field the rays march (core/field.js) is
// a separate grid, 12.5 cm voxels in three cascades. (Until 2026-10-22 that
// field was called the boxGrid.)
import { BLOCK_METRES } from './world.js';

export const BOX_CELLS = 3;

export const BOX_METRES = BLOCK_METRES / BOX_CELLS;

// The box holding world coordinate v (metres), along one axis.
export const boxIndex = v => Math.round(v / BOX_METRES);

// The centre of box i, in metres.
export const boxCentre = i => i * BOX_METRES;

// The block holding box i.
export const blockOfBox = i => Math.round(i / BOX_CELLS);
