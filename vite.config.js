import { defineConfig } from 'vite';

// GitHub Pages serves the site from /Project-BoxBlade/, not the domain root, so
// a build prefixes every path with it. Dev stays at /.
export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/Project-BoxBlade/' : '/',
  server: {
    port: 3000,
    open: true
  }
}));
