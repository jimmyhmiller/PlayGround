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

## Web app build (HTML entry)

```console
diffpack build <project-root> [--vite] [--out-dir <dir>] [--no-minify] [--sourcemap]
```

Bundles an HTML-rooted web app: `index.html` is the entry, its
`<script type="module" src>` starts the graph, and the built document is
rewritten Vite-style (script + extracted stylesheet injected into `<head>`).
`--vite` opts in to Vite conventions as a bundle — `vite.config` `define`/`base`
evaluation, the `.env`/`VITE_*` file stack, `import.meta.env`, and the `public/`
passthrough. Without the flag none of those apply; Vite behavior is never
implicit. The pinned `create-vite` React app in
`integration/vite-react-reference` builds with this command and passes its
acceptance + headless-browser gates (see its `acceptance.mjs` /
`browser-check.mjs`).

## Conformance and competitive benchmarks

- `conformance/` runs 48 executable ESM/CJS-semantics fixtures against Node
  ground truth, comparing diffpack with pinned Rolldown and esbuild
  (`docs/CONFORMANCE.md`).
- `bench/` measures cold/incremental/memory/size against esbuild, Rolldown,
  rspack, and Vite on tiny and realistic corpora plus the real TanStack app
  (`docs/COMPETITIVE_BENCHMARKS.md`).

## Current boundary

- Package resolution uses `oxc_resolver`, including package exports, extension
  aliases, tsconfig `paths`, and `node_modules` traversal; Node built-ins are
  externals.
- Oxc strips TypeScript and lowers JSX; production minification (compression +
  mangling) runs per chunk with composed source maps.
- Static ESM, CommonJS interop, JSON, re-exports, live bindings, literal
  dynamic imports, code splitting, conservative tree sha­king, package
  `sideEffects`, and export-level dead-module elimination are implemented.
  Import order is execution order (and CSS cascade order).
- CSS: global stylesheets, CSS Modules (scoping, `composes`, `:global`),
  `@import` inlining with media wrapping, `url()` asset rewriting, and a native
  Tailwind v4 engine.
- `--format esm` output supports top-level `await` (single-chunk) and
  `import.meta`; CommonJS output refuses both with module-naming errors.
- TanStack Start builds natively end to end (13/13 gates, SSR + server
  functions + route splitting + manifests); `diffpack dev` is a long-lived dev
  server with client HMR, React Fast Refresh, and in-process server hot reload.
- Not yet: JS plugin hosting (deliberate — popular plugins are reimplemented
  natively), multiple HTML entries, content-hashed JS chunk names, shared
  vendor chunking, CSS modules in `?url` verbatim copies, a non-root `base`
  for asset URLs, and the Vite Environment API / module runner.
