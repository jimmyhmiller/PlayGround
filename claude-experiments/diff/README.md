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

Run the test suite:

```console
cargo test
```

## Current boundary

- Only relative imports are resolved. Bare package imports produce diagnostics.
- Source and AST payloads do not enter the dataflow; tuples contain owned module
  IDs and content hashes.
- Logical time is a monotonically increasing revision number. Inputs advance
  after each revision, allowing Differential's traces to compact old history.
- The output is a deterministic reachable-module manifest, not executable code.

The next useful increment is a persistent watch loop plus Oxc transform/codegen
artifacts keyed by `(module, content hash, build context)`.
