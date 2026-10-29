# tools

Headless smoke tests. No WebGPU device, no DOM — modules that only import
Three.js are fine, anything constructing a renderer is not.

```bash
npm test            # everything
npm test boxgrid    # only files matching "boxgrid"
```

One `*.test.mjs` per major feature. Tests are plain top-level code; importing
the file runs it. Assertions come from `lib/harness.mjs` (`ok`, `near`,
`truthy`, `falsy`, `inRange`, `throws`, `note`). Exit code is non-zero on
failure.

Two tests guard the structure rather than a feature:

- `boundaries.test.mjs` — which layer (`js/core`, `game`, `render`, `gpu`,
  `tools`) may import which, that core loads headless, and line budgets that
  may only come down.
- `shaders.test.mjs` — every shader the engine builds, generated as WGSL
  without a GPU (`lib/shaders.mjs`) and compared byte for byte with
  `snapshots/shaders.json`. A change meant to alter shaders is accepted with
  `node tools/run-tests.mjs shaders --update`; diff `.shaders/accepted`
  against `.shaders/current` to see what moved.

What can't be covered here — actual rendering, input — goes in the
in-browser checklist instead.

Regions (`js/core/region.js`, level format v2) are binary `.bxr` files in
`levels/`; `node tools/region.mjs info|dump|pack <file> [out]` shows one, dumps
it to readable JSON (a row per block) for reading or diffing, and packs any
level — a dump, or an old v1 `.json` — back into a `.bxr`.
