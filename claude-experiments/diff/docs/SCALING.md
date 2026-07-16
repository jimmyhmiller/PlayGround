# Scaling results

These measurements establish the PoC's current order of growth. They are not
yet production-bundler benchmarks: parsing, filesystem traversal, package
resolution, transformation, code generation, and source maps are not included.

## Method

- Release build, one Timely worker
- Apple M2 Max with 64 GiB RAM
- Broad dependency tree with fanout 8, keeping fixed-point depth logarithmic
- Four imports per module; shared back-edges introduce sharing and cycles
- One initial fact batch followed by a two-fact content edit and a leaf deletion
- No complete graph snapshots are constructed for edit revisions
- Output revisions contain delta counts, not cloned reachable manifests
- Times are captured inside the process; peak RSS is from `/usr/bin/time -l`

Run it with:

```console
cargo build --release
/usr/bin/time -l target/release/diffpack scale 100000 8 4
```

## Current delta-based path

| Modules | Edges | Initial dataflow | Edit input | Edit dataflow | Edit output | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 399,990 | 221.9 ms | 0.002 ms | 0.216 ms | <0.001 ms | 268 MB |
| 250,000 | 999,990 | 582.0 ms | 0.007 ms | 0.200 ms | 0.001 ms | 676 MB |
| 500,000 | 1,999,990 | 1,309.5 ms | 0.002 ms | 0.248 ms | 0.001 ms | 1,366 MB |

Small-edit latency is effectively independent of total graph size in this
topology. Initial load and memory remain approximately linear in the number of
facts, as expected.

## Before and after at 100,000 modules

| Measurement | Snapshot path | Delta path |
| --- | ---: | ---: |
| Dependency edges | 399,990 | 399,990 |
| Input handling for one edit | 76.1 ms | 0.002 ms |
| Differential handling for one edit | 0.21 ms | 0.216 ms |
| Output handling for one edit | 13.2 ms | <0.001 ms |
| Peak RSS | about 600 MB | 268 MB |

The old benchmark retained three complete graphs, compared global `BTreeSet`s,
cloned the full reachable output after each revision, and retained initial
output rows merely to count them. The new path sends weighted facts directly:

```text
(module, old_content_hash) -1
(module, new_content_hash) +1
```

Output additions and retractions update integer counts directly. A full
manifest is only materialized by the compatibility API used by the small demo.

## Remaining work

The scalable revision boundary is fixed, but several production concerns remain:

1. Connect a file watcher to `DeltaRevision` through a persistent per-module
   fact store. `scan_graph` remains appropriate for initial discovery only.
2. Intern paths and specifiers into compact numeric IDs. Strings are still
   copied into input batches and Differential arrangements.
3. Instrument arrangement sizes and allocated bytes per relation.
4. Add Node package resolution, export conditions, and symlink policy.
5. Benchmark high-fanout edits, entry removal, large SCCs, per-symbol liveness,
   multiple build targets, and multiple entry points.

Per-symbol and build-variant relations are especially important. Naïvely
materializing `(entry, module, symbol, target)` can create a multiplicative
state explosion even though module-level reachability scales linearly.
