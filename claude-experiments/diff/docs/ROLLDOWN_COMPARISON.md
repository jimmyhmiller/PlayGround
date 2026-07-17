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
npm run bench -- 10000 4 3
```

## Apple M2 Max results

The 10,000-module values are medians of three fresh builds. The 50,000-module
run is a single measurement and should be repeated before treating small
differences as significant.

| Modules | Method | Fresh build | Content edit | Entry dependency removal | Final output |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | Diffpack | 550 ms | 12.1 ms | 11.9 ms | 11.2 MB |
| 10,000 | Rolldown 1.2.0 | 167 ms | 175 ms | 217 ms | 1.59 MB |
| 50,000 | Diffpack | 2.79 s | 74.0 ms | 82.1 ms | — |
| 50,000 | Rolldown 1.2.0 | 1.02 s | 1.10 s | 1.17 s | — |

Rolldown is approximately 2.7–3.3 times faster on fresh builds. Diffpack is
approximately 14–18 times faster on these warm edits. The reasons are
architectural: Rolldown's scope-hoisted production linker is highly optimized
for complete builds, while Diffpack keeps module factories and performs a
specialized graph repair before rewriting a simple cached-shape runtime chunk.
At 10,000 modules Diffpack currently writes about seven times as many bytes,
because every module retains an independent factory, path mapping, and runtime
metadata instead of being scope-hoisted. Despite that disadvantage, its warm
edit remains faster in this workload; output size is nevertheless a clear
production-quality gap.

This benchmark is intentionally narrow. It does not show that Diffpack is a
faster production bundler: Diffpack does not yet perform tree shaking, source
maps, minification, CSS processing, or code splitting. It does show a concrete
competitive target:

1. Reduce cold parse/transform/resolve time by roughly threefold.
2. Preserve the current module-granular edit path while adding symbol inclusion.
3. Add more realistic source sizes and application fixtures before optimizing
   against synthetic tiny modules alone.
