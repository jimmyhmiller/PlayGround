# Diffpack and Rolldown comparison

`oracle/benchmark.mjs` generates the same broad TypeScript module graph used by
the Diffpack scaling benchmark and measures Diffpack against pinned Rolldown
1.2.0.

The default comparison uses one entry, four imports per generated module, one
output chunk, disabled tree shaking, no minification, and no source maps. Pass
`--treeshake` for a matched symbol-inclusion comparison. Corpus generation and
filesystem-watch detection latency are excluded. Fresh results include
discovery, parsing, transformation, linking/reachability, rendering, and
writing. Edit results include transformation through writing the updated single
chunk.

Run it with:

```console
cd oracle
npm ci
npm run bench -- 10000 4 5 --treeshake
npm run bench -- 10000 4 5 --treeshake --live
npm run bench -- 10000 4 5 --treeshake --live --minify
npm run bench:capabilities -- 15
```

## Matched capability fixtures

The capability benchmark refuses to time a pair unless runtime status and
stdout agree. It also requires matching source-map counts and chunk counts when
applicable, and verifies that the unused export sentinel is absent. These are
medians of 15 end-to-end release CLI runs, including process startup and output
writes.

| Case | Diffpack | Rolldown 1.2.0 | Diffpack output | Rolldown output | Files each |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unused export removal | 3.70 ms | 77.6 ms | 0.040 KB | 0.095 KB | 1 |
| Package `sideEffects` | 3.63 ms | 76.6 ms | 0.023 KB | 0.091 KB | 1 |
| External source map | 3.78 ms | 79.8 ms | 0.553 KB | 0.654 KB | 2 |
| Two lazy chunks | 4.04 ms | 79.2 ms | 0.297 KB | 0.485 KB | 3 |

These tiny-fixture times are dominated by CLI startup. They are useful for
developer-facing latency but not as a core linker throughput comparison. The
artifact columns show that Diffpack's conservative scope-hoisted path is smaller
for all four matched fixtures. The benchmark fails before timing if Diffpack's
combined artifacts are larger.

The frontend parses each source module once. Oxc's TypeScript/JSX transform and
Diffpack's ESM, live-binding, and dynamic-import lowering operate on the same
AST, followed by one final code-generation pass. The previous generated-code
reparse and span-based text-edit implementation has been removed.

## 10,000-module live-output graph

The primary scaling case makes every module contribute to the entry's printed
value. A content edit changes one consumed value. The dependency edit removes
one entry import, retracting 5,461 modules while leaving 4,539 live modules and
a substantial output artifact. Before reporting timings, the oracle executes
both bundles after both edit types and compares them with an independently
calculated numeric result.

These are medians of five runs:

| Method | Fresh build | Content edit | Entry dependency removal | Final output |
| --- | ---: | ---: | ---: | ---: |
| Diffpack | 130 ms | 8.16 ms | 5.45 ms | 165,770 bytes |
| Rolldown 1.2.0 | 159 ms | 243 ms | 171 ms | 869,414 bytes |

With Rolldown minification enabled, the same runtime-verified dependency-edit
artifact changes substantially:

| Method | Fresh build | Content edit | Entry dependency removal | Final output |
| --- | ---: | ---: | ---: | ---: |
| Diffpack with linked-constant minification | 131 ms | 5.57 ms | 4.93 ms | 22 bytes |
| Rolldown 1.2.0 with minification | 167 ms | 174 ms | 85.0 ms | 117 bytes |

This fixture is deliberately composed of constants, so Rolldown's minifier can
evaluate the retained expression graph and collapse it to the final value.
Consequently, the minified size is a constant-folding result rather than a
measure of module reachability. Diffpack now carries a conservative numeric
expression IR from each module's original AST and evaluates it after linking.
It does not reparse generated output. The optimization is used only when every
retained statement is proven to be a supported constant declaration or
`console.log` effect; otherwise rendering safely falls back. This matches the
optimization needed by this fixture, not Rolldown's complete minifier surface.

The live-output fixture exposed and fixed a quadratic renderer operation:
Diffpack previously rescanned the entire emitted string for every module just
to calculate source-map line offsets. Rendering now maintains a running line
counter. On this fixture that reduced Diffpack's content-edit time from about
685 ms to 8.16 ms.

## 10,000-module dead-output graph

This benchmark excludes corpus generation, CLI startup, and filesystem-watch
detection. The values below are medians of three runs.

| Method | Fresh build | Content edit | Entry dependency removal | Final output |
| --- | ---: | ---: | ---: | ---: |
| Diffpack | 175 ms | 3.30 ms | 3.76 ms | 0 bytes |
| Rolldown 1.2.0 | 162 ms | 631 ms | 724 ms | 285 bytes |

Import demand is derived once from Oxc's semantic AST and carried on dense graph
edges. Rendering performs one linear marker-removal pass over module code. An
earlier prototype searched generated source once per dependency and made edits
roughly 100 ms; that implementation was removed.

Almost all exports in this supplementary synthetic graph are unconsumed. Both
implementations collapse it to an effectively empty artifact. It is retained as
a dead-code-elimination stress case, not as the primary performance claim.
Diffpack uses a conservative
scope-hoisted path for acyclic, collision-free named ESM graphs and falls back
to module factories for CommonJS, cycles, namespace/default/aliased imports,
and other cases requiring the full runtime semantics.

This benchmark is intentionally narrow. Diffpack now has matching cases for
package `sideEffects`, source maps, and literal dynamic-import splitting, but it
does not yet match Rolldown's full tree shaker, scope hoisting, minification,
CSS/assets, plugins, or output-format surface. The next competitive targets are:

1. Reduce tiny-file I/O and deterministic discovery-merge overhead, which now
   dominate fresh builds.
2. Expand the conservative scope-hoisted path while preserving the
   module-granular edit path.
3. Add more realistic source sizes and application fixtures before optimizing
   against synthetic tiny modules alone.
