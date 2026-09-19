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

What can't be covered here — shaders, actual rendering, input — goes in the
in-browser checklist instead.
