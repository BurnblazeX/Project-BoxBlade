// --- Version and build stamp ---
//
// The version follows the FortHex convention, A##//#a:
//   A/B/C/R   Alpha, Beta, Candidate, Release
//   ##        major
//   //#       minor
//   a         hotfix, a = 1 (omitted when there is none)
// e.g. B28//3d is Beta, major 28, minor 3, hotfix 4.
export const VERSION = 'A03//6b';

// The loaded bundle's content hash, from its file name: Vite names the built
// entry index-<hash>.js, and the hash changes with every byte of the code. So
// it is a checksum of exactly what this tab is running. null when unbuilt (dev).
export function bundleHash(url) {
  const m = /-([A-Za-z0-9_-]{6,})\.js(?:[?#].*)?$/.exec(url || '');
  return m ? m[1] : null;
}

// What the corner shows: version, commit (+ when built from uncommitted
// changes), and the bundle checksum, or "dev" under the dev server.
export function buildLabel({ version, commit, dirty, bundle }) {
  return `${version}  ${commit || 'unknown'}${dirty ? '+' : ''}  ${bundle ? '#' + bundle : 'dev'}`;
}
