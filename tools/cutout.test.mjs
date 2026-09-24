import { file, section, ok, truthy, falsy } from './lib/harness.mjs';
import { alphaIsBinary, cutoutOpaque, wantsTransparent, registerCutout,
         syncCutout, classifyTextureAlpha, cutoutsEnabled, setCutoutsEnabled } from '../js/cutout.js';

file('cutout.test.mjs - binary-alpha sprites drawn opaque');

const rgba = (...alphas) => {
  const a = new Uint8ClampedArray(alphas.length * 4);
  alphas.forEach((v, i) => { a[i * 4 + 3] = v; });
  return a;
};
const tex = (alphaBinary) => ({ userData: alphaBinary === undefined ? {} : { alphaBinary } });
const mat = (map, alphaTest = 0.5) => ({ map, alphaTest, transparent: true, needsUpdate: false, userData: {} });

section('alpha classification');
truthy('all 0 and 255 is binary', alphaIsBinary(rgba(0, 255, 255, 0)));
truthy('fully opaque is binary', alphaIsBinary(rgba(255, 255)));
falsy('one partial texel is not (glass, soft edges)', alphaIsBinary(rgba(0, 255, 128)));
falsy('1 is partial', alphaIsBinary(rgba(1)));
falsy('254 is partial', alphaIsBinary(rgba(254)));
// Headless there is no document: an unreadable image must fall back to the old
// answer, transparent.
const unreadable = { image: { width: 4, height: 4 }, userData: {} };
classifyTextureAlpha(unreadable);
ok('an unreadable image is not binary', unreadable.userData.alphaBinary, false);

section('which materials may draw opaque');
truthy('binary alpha with an alpha test', cutoutOpaque(mat(tex(true))));
falsy('partial alpha stays transparent', cutoutOpaque(mat(tex(false))));
falsy('unclassified (still loading) stays transparent', cutoutOpaque(mat(tex(undefined))));
falsy('no alpha test: the clear texels would draw', cutoutOpaque(mat(tex(true), 0)));
falsy('no map', cutoutOpaque(mat(null)));
truthy('on top (the clip fix) is always transparent', wantsTransparent(mat(tex(true)), true));

section('register, sync, toggle');
const t = tex(true);
const m = registerCutout(mat(t));
falsy('a binary cutout goes opaque on register', m.transparent);
truthy('and rebuilds its pipeline', m.needsUpdate);
m.needsUpdate = false;
syncCutout(m);
falsy('no rebuild when nothing changed', m.needsUpdate);

let onTop = false;
const c = registerCutout(mat(tex(true)), () => onTop);
falsy('a character clear of walls is opaque', c.transparent);
onTop = true; syncCutout(c);
truthy('clipping, it is transparent (drawn last)', c.transparent);
onTop = false; syncCutout(c);
falsy('clear again, opaque again', c.transparent);

const glass = registerCutout(mat(tex(false)));
truthy('partial alpha is left transparent', glass.transparent);
falsy('and untouched', glass.needsUpdate);

setCutoutsEnabled(false);
falsy('toggle reports off', cutoutsEnabled());
truthy('off: everything transparent, as before', m.transparent && c.transparent);
setCutoutsEnabled(true);
truthy('toggle reports on', cutoutsEnabled());
falsy('on again: opaque', m.transparent || c.transparent);
truthy('glass never changes', glass.transparent);
