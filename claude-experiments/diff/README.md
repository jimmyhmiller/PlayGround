# diffpack

`diffpack` is a proof of concept for maintaining a JavaScript module graph with
[Oxc](https://github.com/oxc-project/oxc) and
[Differential Dataflow](https://github.com/TimelyDataflow/differential-dataflow).

It is intentionally not a production bundler yet. Oxc parses JavaScript,
TypeScript, JSX, and TSX and extracts static imports, re-exports, and literal
dynamic imports. A small resolver turns relative imports into module edges.
Differential Dataflow maintains the transitive reachability fixed point as
entry, edge, and module-version facts are inserted and retracted.

Run the three-revision demonstration:

```console
cargo run -- demo
```

Or inspect one entry graph:

```console
cargo run -- build fixtures/incremental/01-initial/entry.js
```

Build and execute an actual single-chunk bundle:

```console
cargo run -- bundle fixtures/bundle/entry.ts dist/bundle.js
node dist/bundle.js
```

Keep the dataflow and module-fact store alive while watching for edits:

```console
cargo run -- watch fixtures/bundle/entry.ts dist/bundle.js
```

The watch path reparses and transforms only changed or newly discovered modules,
sends weighted module/edge updates to a persistent Differential session, and
rewrites the deterministic single output chunk after the revision frontier is
complete.

Run the test suite:

```console
cargo test
```

Run the behavioral compatibility oracle. It executes the same tagged fixtures
through Diffpack and a pinned Rolldown reference, then compares both with an
explicit expected result:

```console
cd oracle
npm ci
npm test
```

The Rust test suite separately verifies that an incremental edit produces the
same reachable set, emitted bytes, and runtime behavior as a clean rebuild.
The current compatibility target and oracle contract are documented in
[docs/ORACLE.md](docs/ORACLE.md).

Run a release-mode scale test with 100,000 modules, a fanout of eight, and four
imports per module:

```console
cargo build --release
/usr/bin/time -l target/release/diffpack scale 100000 8 4
```

The CSV output separates weighted input updates, Differential propagation, and
output-delta handling. Peak resident memory is reported by `/usr/bin/time` on
macOS. The benchmark applies one initial fact batch, changes one module's
content hash with two updates, then retracts one leaf module and its edges.

Recorded results and methodology are in [docs/SCALING.md](docs/SCALING.md).

Run the full on-disk bundler benchmark, including Oxc transformation, resolution,
Differential loading, and chunk emission:

```console
cargo build --release
/usr/bin/time -l target/release/diffpack bundle-scale 10000 4
```

Run the same build and edit without constructing a Differential dataflow. This
mode uses a persistent dense integer graph and incrementally repairs a
reachability spanning tree:

```console
/usr/bin/time -l target/release/diffpack bundle-scale-direct 10000 4
```

Compare both approaches on an edit that removes an entry import and makes part
of the module graph unreachable:

```console
target/release/diffpack bundle-scale-direct-deps 10000 4
target/release/diffpack bundle-scale-deps 10000 4
```

Module discovery, Oxc transformation, dependency resolution, benchmark corpus
generation, and chunk rendering run on Rayon's worker pool. It defaults to the
machine's available parallelism; set `RAYON_NUM_THREADS` to control it, including
`RAYON_NUM_THREADS=1` for a single-thread baseline. The persistent Differential
session supports multiple Timely workers via `DIFFPACK_DATAFLOW_THREADS`; it
defaults to one because this recursive module-graph workload benchmarks faster
without cross-worker coordination.

End-to-end results are in
[docs/BUNDLER_SCALING.md](docs/BUNDLER_SCALING.md).

## Current boundary

- Relative imports and packages are resolved with `oxc_resolver`, including
  package `exports`, extension aliases, and `node_modules` traversal.
- Oxc strips TypeScript and lowers JSX before module linking.
- Static ESM, CommonJS `require`, JSON, re-exports, and literal dynamic imports
  are supported by the single-chunk runtime.
- Source and AST payloads do not enter the dataflow; tuples contain owned module
  IDs and content hashes.
- Logical time is a monotonically increasing revision number. Inputs advance
  after each revision, allowing Differential's traces to compact old history.
- Dynamic imports are currently folded into the single chunk rather than split.
- Imported bindings are lowered to runtime property reads at module evaluation;
  full ESM live-import semantics and top-level await are not implemented yet.
- CSS, assets, source maps, tree shaking, minification, and production chunk
  splitting are not implemented.

The next compatibility target is complete single-chunk ESM linking semantics.
After that, the next architectural increment is symbol-level linking/tree
shaking followed by a real chunk graph for dynamic imports and shared
dependencies.
