# Diffpack and Rolldown comparison

`oracle/benchmark.mjs` generates the same broad TypeScript module graph used by
the Diffpack scaling benchmark and measures Diffpack against pinned Rolldown
1.2.0.

Both implementations use one entry, four imports per generated module, one
output chunk, disabled tree shaking, no minification, and no source maps. Corpus
generation and filesystem-watch detection latency are excluded. Fresh results
include discovery, parsing, transformation, linking/reachability, rendering,
and writing. Edit results include transformation through writing the updated
single chunk.

Run it with:

```console
cd oracle
npm ci
npm run bench -- 10000 4 5
```

## Apple M2 Max results

The 10,000-module values are medians of five fresh builds. The 50,000-module
run is a single measurement and should be repeated before treating small
differences as significant.

| Modules | Method | Fresh build | Content edit | Entry dependency removal | Final output |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | Diffpack | 179 ms | 6.79 ms | 6.73 ms | 4.81 MB |
| 10,000 | Rolldown 1.2.0 | 174 ms | 166 ms | 203 ms | 1.56 MB |
| 50,000 | Diffpack | 1.09 s | 32.6 ms | 34.6 ms | 24.3 MB |
| 50,000 | Rolldown 1.2.0 | 1.06 s | 1.21 s | 1.23 s | 7.83 MB |

Fresh builds are now within approximately 3% of Rolldown at both scales.
Diffpack is approximately 24–30 times faster on the 10,000-module edits and
35–37 times faster on the single 50,000-module edit measurements. The direct
production path does not construct a Differential manifest: it builds a dense
persistent reachability index, resolves exact relative files through a cached
fast path, and uses deterministic integer module IDs in the emitted runtime.

Rolldown's scope-hoisted output remains substantially smaller. Diffpack keeps
one factory and specifier map per module, so its current output is about three
times larger even after compacting runtime IDs. Scope hoisting or a more compact
development transport is therefore the next major output-side target.

This benchmark is intentionally narrow. It does not show that Diffpack is a
faster production bundler: Diffpack does not yet perform tree shaking, source
maps, minification, CSS processing, or code splitting. It does show a concrete
competitive target:

1. Make fresh builds materially faster, not merely equal, by interning paths and
   edges directly into the dense graph during discovery and reducing tiny-file
   I/O overhead.
2. Preserve the current module-granular edit path while adding symbol inclusion.
3. Add more realistic source sizes and application fixtures before optimizing
   against synthetic tiny modules alone.
