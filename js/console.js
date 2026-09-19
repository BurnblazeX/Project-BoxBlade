// --- bxb debug console ---
//
// Installs a `bxb` namespace on window so debug actions can be driven from the
// browser console without adding keybinds for everything. Commands are supplied
// by the caller (main.js owns the state); this module only holds the registry,
// help text and dispatch, which keeps it testable without a DOM.

export function createConsole(commands) {
  const api = {};

  for (const [name, def] of Object.entries(commands)) {
    const fn = (...args) => def.run(...args);
    fn.help = def.help;
    fn.usage = def.usage || `bxb.${name}()`;
    api[name] = fn;
  }

  api.help = () => {
    const lines = ['BoxBlade debug console', ''];
    const width = Math.max(...Object.keys(commands).map(n => (commands[n].usage || `bxb.${n}()`).length));
    for (const [name, def] of Object.entries(commands)) {
      lines.push(`  ${(def.usage || `bxb.${name}()`).padEnd(width)}  ${def.help}`);
    }
    lines.push('', '  bxb.help()  this list');
    const text = lines.join('\n');
    console.log(text);
    return text;
  };
  api.help.help = 'this list';

  return api;
}

export function installConsole(api, target = globalThis) {
  target.bxb = api;
  console.log('bxb debug console ready - type bxb.help()');
  return api;
}
