# End-to-end bundler scaling

This benchmark measures the implemented bundler pipeline, not only the module
graph. It creates a real TypeScript project on disk, resolves and transforms all
modules, loads the graph into a persistent Differential session, emits one
chunk, changes one module, and emits again.

## Workload

- Apple M2 Max, 64 GiB RAM
- Release build; Rayon front-end workers and Timely dataflow workers reported below
- Broad dependency tree with four imports per module
- Small TypeScript modules containing type annotations
- One content-only edit that preserves the dependency edges
- Dynamic chunking, tree shaking, source maps, and minification disabled
- Corpus generation is measured separately and excluded from build totals

Run it with:

```console
cargo build --release
/usr/bin/time -l target/release/diffpack bundle-scale 100000 4
```

## Initial build

The filesystem-heavy phases varied between runs, especially at 50,000 and
100,000 small files. The table gives the observed ranges where repeated runs
were available.

| Modules | Imports | Output | Parse + transform + resolve | Differential load | Emit chunk | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 4,000 | 1.1 MB | 131 ms | 16 ms | 4 ms | 23 MB |
| 10,000 | 40,000 | 10.6 MB | 1.8–2.4 s | 126–129 ms | 26–73 ms | 108–109 MB |
| 50,000 | 200,000 | 52.5 MB | 7.7–13.6 s | 0.88–1.16 s | 159–216 ms | 539–540 MB |
| 100,000 | 400,000 | 104.7 MB | 37.4–44.7 s | 1.31–2.72 s | 0.44–3.35 s | 991–1,011 MB |

The 100,000-module build is not competitive with a production incremental
bundler yet. Initial parsing/transformation/resolution is the dominant CPU and
filesystem cost. Differential's initial recursive load is material but smaller.

## Multi-threaded pipeline

Discovery proceeds in dependency frontiers. Files in each frontier are read,
transformed with Oxc, and resolved in parallel on Rayon; generated module and
source-map fragments are also rendered in parallel and sorted before assembly
to keep output deterministic. On the 12-logical-CPU test machine:

| Modules | Front-end workers | Transform + resolve | Emit chunk |
| ---: | ---: | ---: | ---: |
| 10,000 | 1 | 755 ms | 23.6 ms |
| 10,000 | 12 | 521 ms | 13.4 ms |
| 50,000 | 1 | 3.64 s | 115 ms |
| 50,000 | 12 | 2.92 s | 51.3 ms |

The gains are intentionally below core-count scaling: this synthetic corpus has
tiny modules, so filesystem metadata, path resolution, frontier barriers, and
allocator contention dominate much of the work. Larger source files give Oxc
more CPU work per scheduled task.

The persistent Differential session now supports multiple Timely workers via
`DIFFPACK_DATAFLOW_THREADS`. It defaults to one after measuring the recursive
reachability workload: at 10,000 modules, one, two, and four workers took about
72 ms, 76 ms, and 70 ms respectively, while 12 workers took 5.0 s in one run.
At 50,000 modules, one worker took 422 ms and four took 510 ms. Cross-worker
exchange and recursive-progress coordination outweigh parallelism for these
small records, so parallel Timely execution is available but not forced.

## Direct baseline without Differential

`bundle-scale-direct` runs the same parallel corpus generation, Oxc transforms,
resolution, edit, and complete chunk writes without constructing a
Timely/Differential dataflow. It recomputes reachability from the entry after
every revision using a conventional Rayon-parallel frontier traversal.

The dependency-edit variants remove one import from the entry module, forcing
both implementations to identify and retract the same unreachable region:

| Modules | Mode | Initial reachable | Final reachable | Initial reachability | Dependency-edit reachability |
| ---: | --- | ---: | ---: | ---: | ---: |
| 10,000 | Parallel traversal | 8,944 | 8,859 | 26.7 ms | 53.8 ms |
| 10,000 | Differential | 8,944 | 8,859 | 77.6 ms | 88.9 ms |
| 50,000 | Parallel traversal | 44,300 | 44,007 | 184 ms | 299 ms |
| 50,000 | Differential | 44,300 | 44,007 | 488 ms | 642 ms |

The outputs agree, but the conventional full recomputation wins at these sizes.
The current Differential formulation pays arrangement, iteration, consolidation,
and progress-tracking costs that outweigh maintaining the small output delta.
This is stronger evidence than the earlier content-only baseline: Differential
does not yet earn back its machinery even when reachability actually changes.

The ordinary dependency/specifier maps still exist because resolving imports
and generating executable `require` mappings is part of doing the same bundling
work; what is absent from the direct mode is the maintained dataflow graph.

## One-module edit

These are the current measurements after removing whole-resolver-cache
invalidation from ordinary source edits:

| Modules | Changed artifacts | Transform + resolve | Differential update | Rewrite chunk | Approximate edit total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 1 | 0.40 ms | 0.18 ms | 2.1 ms | 2.7 ms |
| 50,000 | 1 | 0.85 ms | 1.46 ms | 196 ms | 198 ms |
| 100,000 | 1 | 0.57 ms | 1.31 ms | 741 ms | 743 ms |

The important result is that transformation and Differential propagation stay
small. The edit curve is dominated by serializing and writing the complete
single chunk. At 100,000 modules, more than 99% of measured edit latency is
emission.

## Interpretation

The current design scales well as an incremental control plane but poorly as a
monolithic development output format:

1. Initial discovery is parallel within dependency frontiers, but many small
   filesystem and resolver operations still limit CPU scaling.
2. Paths and specifiers remain strings, contributing to the roughly 1 GiB RSS
   at 100,000 modules.
3. Every edit regenerates a 105 MB JavaScript file even when one artifact
   changed. Development output should publish module artifacts independently or
   send an HMR delta.
4. Production chunk emission should cache unchanged chunk fragments and only
   relink chunks affected by membership or runtime changes.
5. This workload uses tiny modules. Larger real modules increase transformation
   and output costs, while tree shaking could reduce the emitted set.

The next performance milestone should therefore be module-granular development
serving/HMR and cached chunk fragments. Optimizing the sub-millisecond
Differential edit itself would not materially improve current end-to-end
latency.
