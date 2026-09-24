import { file, section, ok, truthy, falsy } from './lib/harness.mjs';
import { keyPart, createSetCache } from '../js/shadowsets.js';

file('shadowsets.test.mjs - kept shadow material sets');

section('the key');
const uni = { value: 1 };            // stands in for a uniform: a class instance
class Binding {}
const tex = new Binding(), tex2 = new Binding();
const opts = (over = {}) => ({
  rays: 1, cone: false, cards: null, lights: tex, albedo: () => 0,
  gi: { levels: [{ tex, extent: 36 }], strength: uni }, ...over
});
ok('same settings, same key', keyPart(opts()), keyPart(opts()));
truthy('a flag changes it', keyPart(opts()) !== keyPart(opts({ cone: true })));
truthy('a number changes it', keyPart(opts()) !== keyPart(opts({ rays: 16 })));
truthy('cards arriving change it', keyPart(opts()) !== keyPart(opts({ cards: tex2 })));
truthy('a different binding object changes it', keyPart(opts()) !== keyPart(opts({ lights: tex2 })));
ok('a fresh closure does not (the albedo sampler)', keyPart(opts({ albedo: () => 1 })), keyPart(opts()));
ok('a fresh plain wrapper around the same bindings does not (giBinding)',
   keyPart(opts({ gi: { levels: [{ tex, extent: 36 }], strength: uni } })), keyPart(opts()));
truthy('but its contents do', keyPart(opts()) !==
       keyPart(opts({ gi: { levels: [{ tex: tex2, extent: 36 }], strength: uni } })));
ok('key order does not matter', keyPart({ a: 1, b: 2 }), keyPart({ b: 2, a: 1 }));
truthy('null and undefined differ from false', keyPart(null) !== keyPart(false) &&
       keyPart(undefined) !== keyPart(false));

section('the cache');
const c = createSetCache(2);
const A = { compiled: true }, B = { compiled: true }, C = { compiled: true }, D = { compiled: false };
c.set('a', A); c.set('b', B); c.set('c', C);
ok('holds more than keep until evicted', c.size, 3);
let out = c.evict(C);
ok('evicts the oldest', out[0], A);
ok('down to keep', c.size, 2);
c.get('b');                          // refresh b
c.set('d', D);
out = c.evict(null);
ok('a get refreshes: c is now the oldest', out[0], C);
falsy('b survived', out.includes(B));
c.set('e', { compiled: true });
out = c.evict(B);
falsy('the active set is never evicted', out.includes(B));
falsy('nor one still compiling', out.includes(D));
ok('nor the newest', c.has('e'), true);
ok('so the cache may run over keep', c.size, 3);
out = c.drain(B);
truthy('drain empties all but the active', c.size === 1 && c.has('b'));
ok('and returns the rest', out.length, 2);
c.delete('b');
ok('delete removes it', c.size, 0);

section('shader text stability');
{
  const { unstableNames, hashText } = await import('../js/shadowsets.js');
  ok('finds an id-named buffer', unstableNames('var<uniform> NodeBuffer_412 : X; NodeBuffer_412.value')[0], 'NodeBuffer_412');
  ok('once per name', unstableNames('NodeBuffer_1 NodeBuffer_1 NodeBuffer_2').length, 2);
  ok('named buffers pass', unstableNames('var<uniform> bxbLights : bxbLightsStruct;').length, 0);
  ok('hash is stable', hashText('fn main() {}'), hashText('fn main() {}'));
  truthy('and changes with the text', hashText('NodeBuffer_1') !== hashText('NodeBuffer_2'));
  ok('eight hex digits', hashText('').length, 8);
}

section('why a build missed');
{
  const { diffKeyParts } = await import('../js/shadowsets.js');
  ok('names the parts that differ', diffKeyParts({ cone: 'true', rays: '1' }, { cone: 'false', rays: '1' }).join(), 'cone');
  ok('and ones only one side has', diffKeyParts({ a: '1' }, { a: '1', b: '2' }).join(), 'b');
  ok('nothing when equal', diffKeyParts({ a: '1' }, { a: '1' }).length, 0);
}
