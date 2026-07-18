# diffpack

`diffpack` is an experimental JavaScript bundler built with
[Oxc](https://github.com/oxc-project/oxc). It maintains a persistent dense module
graph and incrementally repairs entry reachability after edits.

It is not a production bundler yet. The production checklist for a TanStack
Start target is in
[docs/TANSTACK_PRODUCTION_CHECKLIST.md](docs/TANSTACK_PRODUCTION_CHECKLIST.md).
Current implementation progress is tracked in
[docs/TANSTACK_IMPLEMENTATION_STATUS.md](docs/TANSTACK_IMPLEMENTATION_STATUS.md).

## Bundle

```console
cargo run -- bundle fixtures/bundle/entry.ts dist/bundle.js
node dist/bundle.js
```

Optional flags:

```console
cargo run -- bundle fixtures/bundle/entry.ts dist/bundle.js --sourcemap
cargo run -- bundle fixtures/bundle/entry.ts dist/bundle.js --minify
```

## Watch

```console
cargo run -- watch fixtures/bundle/entry.ts dist/bundle.js
```

The watch path parses and transforms only changed or newly discovered modules,
applies dependency-edge changes to the persistent dense reachability index, and
emits the updated reachable graph.

## Visualize

Generate a self-contained interactive linker visualization:

```console
cargo run -- visualize fixtures/bundle/entry.ts target/diffpack-graph.html
open target/diffpack-graph.html
```

Generate the 10,000-module live scaling graph after its entry dependency is
removed:

```console
cargo run --release -- visualize-scale 10000 4 target/diffpack-large-graph.html
```

The visualization shows dense module IDs, import demands, dynamic edges,
reachability, direct effects, flat-link eligibility, declarations, exports,
pruned imports, and the conservative constant-folding IR. It reads the actual
cached linker records and does not parse the project again.

## Correctness oracle

```console
cargo test
cd oracle
npm ci
npm test
npm run parity:strict
```

The oracle executes tagged fixtures through Diffpack and pinned Rolldown,
compares explicit runtime expectations, and checks clean versus incremental
behavior. See [docs/ORACLE.md](docs/ORACLE.md).

## Performance comparison

```console
cd oracle
npm run bench -- 10000 4 5 --treeshake --live
npm run bench -- 10000 4 5 --treeshake --live --minify
```

See [docs/ROLLDOWN_COMPARISON.md](docs/ROLLDOWN_COMPARISON.md) for methodology
and current results.

The standalone scaling commands are:

```console
cargo build --release
target/release/diffpack bundle-scale-direct 10000 4
target/release/diffpack bundle-scale-direct-deps 10000 4
target/release/diffpack bundle-scale-direct-live 10000 4
target/release/diffpack bundle-scale-direct-live-deps 10000 4
```

See [docs/BUNDLER_SCALING.md](docs/BUNDLER_SCALING.md).

## Current boundary

- Package resolution uses `oxc_resolver`, including package exports, extension
  aliases, and `node_modules` traversal.
- Oxc strips TypeScript and lowers JSX before linking.
- Static ESM, basic CommonJS, JSON, re-exports, source maps, literal dynamic
  imports, basic code splitting, conservative tree shaking, and package
  `sideEffects: false` are implemented.
- A conservative flat path scope-hoists safe modules and falls back to module
  factories for unsupported cases.
- The minifier currently performs a narrow linked-constant optimization; a
  general combined-AST Oxc minification path is still required.
- CSS, general assets, plugin hosting, multiple environments, SSR, and HMR are
  not implemented.
