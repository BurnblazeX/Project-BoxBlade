import { file, section, ok } from './lib/harness.mjs';
import { REGION_CHUNKS, regionOf, chunkSlot, regionChunks, regionFile } from '../js/core/region.js';

file('region.test.mjs - the storage grid (World plan §0.2a)');

section('chunks to regions');
{
  ok('32 chunks a side', REGION_CHUNKS, 32);
  ok('chunk 0, 0 is region 0, 0', regionOf(0, 0).join(), '0,0');
  ok('the last chunk of region 0', regionOf(31, 31).join(), '0,0');
  ok('the next region starts at 32', regionOf(32, 0).join(), '1,0');
  ok('negative chunks round down', regionOf(-1, -33).join(), '-1,-2');
  ok('slots run x first', [chunkSlot(0, 0), chunkSlot(1, 0), chunkSlot(0, 1), chunkSlot(31, 31)].join(), '0,1,32,1023');
  ok('a negative chunk has a slot in its region', chunkSlot(-1, -1), 1023);
  ok('a region covers its chunks', regionChunks(1, -1).join(), '32,63,-32,-1');
  // Every chunk of a region lands in it, each in its own slot.
  const slots = new Set();
  const [cx0, cx1, cz0, cz1] = regionChunks(-2, 3);
  let inside = true;
  for (let cx = cx0; cx <= cx1; cx++) for (let cz = cz0; cz <= cz1; cz++) {
    slots.add(chunkSlot(cx, cz));
    if (regionOf(cx, cz).join() !== '-2,3') inside = false;
  }
  ok('... all of them, in REGION_CHUNKS² slots', [inside, slots.size].join(), `true,${REGION_CHUNKS ** 2}`);
  ok('file names as Minecraft names them', regionFile(-2, 3), 'r.-2.3.bxr');
}
