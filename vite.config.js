import { defineConfig } from 'vite';
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { decodeRegion } from './js/core/region.js';

// Levels (the editor, js/tools/editor.js) live in levels/ at the repo root:
// <name>.bxr, a region in level format v2 (js/core/region.js), or a v1
// <name>.json, which still loads. Under the dev server they are read at
// /levels/<name>.<ext> and written by POST /__levels?name=<name>, the .bxr
// bytes as the body; a build copies them into dist/levels/, so the deployed
// game loads them and cannot write them. Not in public/, whose changes the dev
// server may answer with a reload - mid-edit.
const LEVELS_DIR = path.resolve('levels');
const LEVEL_NAME = /^[A-Za-z0-9_-]{1,64}$/;
const LEVEL_TYPES = { bxr: 'application/octet-stream', json: 'application/json' };
function levelsPlugin() {
  return {
    name: 'boxblade-levels',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = new URL(req.url, 'http://local');
        const get = url.pathname.match(/^\/levels\/([A-Za-z0-9_-]{1,64})\.(bxr|json)$/);
        if (req.method === 'GET' && get) {
          const file = path.join(LEVELS_DIR, `${get[1]}.${get[2]}`);
          if (!fs.existsSync(file)) { res.statusCode = 404; return res.end(); }
          // Sent as stored: a .bxr is gzip already, and carries no
          // Content-Encoding, so the game inflates it itself.
          res.setHeader('Content-Type', LEVEL_TYPES[get[2]]);
          res.setHeader('Cache-Control', 'no-store');
          return res.end(fs.readFileSync(file));
        }
        if (url.pathname !== '/__levels') return next();
        const name = url.searchParams.get('name') || 'dev_level';
        if (req.method !== 'POST' || !LEVEL_NAME.test(name)) { res.statusCode = 400; return res.end('bad request'); }
        const parts = [];
        let size = 0;
        req.on('data', c => { parts.push(c); size += c.length; if (size > 64e6) req.destroy(); });
        req.on('end', async () => {
          try {
            const bytes = new Uint8Array(Buffer.concat(parts));
            await decodeRegion(bytes);   // refuses anything that is not a region
            fs.mkdirSync(LEVELS_DIR, { recursive: true });
            fs.writeFileSync(path.join(LEVELS_DIR, name + '.bxr'), bytes);
            res.end('saved');
          } catch (err) {
            res.statusCode = 400;
            res.end(String(err.message || err));
          }
        });
      });
    },
    generateBundle() {
      if (!fs.existsSync(LEVELS_DIR)) return;
      for (const f of fs.readdirSync(LEVELS_DIR)) {
        if (!/\.(bxr|json)$/.test(f)) continue;
        this.emitFile({ type: 'asset', fileName: 'levels/' + f, source: fs.readFileSync(path.join(LEVELS_DIR, f)) });
      }
    }
  };
}

// The commit the build came from, for the version stamp (js/version.js).
const git = args => {
  try {
    return execSync(`git ${args}`, { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return '';
  }
};

// GitHub Pages serves the site from /Project-BoxBlade/, not the domain root, so
// a build prefixes every path with it. Dev stays at /.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/Project-BoxBlade/' : '/',
  define: {
    __BUILD_COMMIT__: JSON.stringify(git('rev-parse --short HEAD') || 'unknown'),
    // Tracked files changed since that commit: the build is not exactly it.
    __BUILD_DIRTY__: JSON.stringify(git('status --porcelain --untracked-files=no') !== ''),
    __BUILD_TIME__: JSON.stringify(new Date().toISOString())
  },
  server: {
    port: 3000,
    open: true
  },
  plugins: [levelsPlugin()]
}));
