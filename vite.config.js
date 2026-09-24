import { defineConfig } from 'vite';
import { execSync } from 'node:child_process';

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
  }
}));
