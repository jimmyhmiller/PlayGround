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

The 10,000-module values are medians of five runs. The 50,000-module values are
medians of three runs.

| Modules | Method | Fresh build | Content edit | Entry dependency removal | Final output |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | Diffpack | 133 ms | 4.85 ms | 5.74 ms | 4.81 MB |
| 10,000 | Rolldown 1.2.0 | 148 ms | 148 ms | 192 ms | 1.56 MB |
| 50,000 | Diffpack | 787 ms | 30.8 ms | 33.5 ms | 24.3 MB |
| 50,000 | Rolldown 1.2.0 | 1.11 s | 1.11 s | 1.19 s | 7.83 MB |

Fresh builds are now approximately 10% faster than Rolldown at 10,000 modules
and 29% faster at 50,000 modules. Diffpack is approximately 26–31 times faster
on the 10,000-module edits and 33–36 times faster at 50,000 modules. The direct
production path does not construct a Differential manifest: discovery interns
each path once, module state stores dense integer edges, reachability consumes
that adjacency directly, and a two-level directory/specifier cache avoids
repeated path hashing and allocation during resolution.

Rolldown's scope-hoisted output remains substantially smaller. Diffpack keeps
one factory and specifier map per module, so its current output is about three
times larger even after compacting runtime IDs. Scope hoisting or a more compact
development transport is therefore the next major output-side target.

This benchmark is intentionally narrow. It does not show that Diffpack is a
faster production bundler: Diffpack does not yet perform tree shaking, source
maps, minification, CSS processing, or code splitting. It does show a concrete
competitive target:

1. Replace the current transform-then-reparse frontend with a single AST
   lowering pipeline; parsing and tiny-file I/O now dominate fresh builds.
2. Preserve the current module-granular edit path while adding symbol inclusion.
3. Add more realistic source sizes and application fixtures before optimizing
   against synthetic tiny modules alone.
