# Crate extraction workflow

This document governs the migration from the original monolithic `diffpack`
crate to the workspace crates under `crates/`. The migration is complete only
when the root package is a CLI/composition layer and no integration crate
depends on it.

The extraction is complete. The boundary-hardening and public-extension work
that follows it is tracked in
[`CORE_ARCHITECTURE_NEXT_STEPS.md`](CORE_ARCHITECTURE_NEXT_STEPS.md).

## Target dependency direction

```text
                         diffpack (CLI only)
                                  |
          +-----------------------+-----------------------+
          |                       |                       |
    diffpack-next          diffpack-tanstack        diffpack-web
          |                       |                       |
          +-----------------------+-----------+-----------+
                                  |           |
                              diffpack-web    |
                                  |           |
                                  +-----------+
                                              |
                                  diffpack-vite-compat
                                              |
                                  diffpack-default-loader
                                              |
                                       diffpack-core
```

An integration may depend directly on any lower layer it needs. Next and
TanStack may consume the framework-neutral browser/dev primitives in
`diffpack-web`; framework integrations must not depend on one another.

## Ownership rules

### `diffpack-core`

- Module identity and graph records.
- Parsing and dependency facts.
- Graph deltas and incremental reachability.
- Transform/linker IR and generic JavaScript lowering.
- Tree shaking and side-effect consumption (not package discovery).
- Chunk planning, rendering contracts, source-map composition, and diagnostics.
- Provider interfaces and immutable provider ordering.

It must not know about filesystems, Node subprocesses, HTML, CSS conventions,
Vite, Next, React, RSC, TanStack, Tailwind, or dev-server protocols.

### `diffpack-default-loader`

- Filesystem and Node/package resolution.
- `package.json` metadata, including `sideEffects` discovery.
- JavaScript/TypeScript/JSON source loading.
- CSS, preprocessors, assets, workers, WebAssembly, and built-in query loaders.
- Default implementations of core provider interfaces.

### `diffpack-web`

- HTML entries and rewriting.
- Browser output layouts.
- HMR protocol, browser runtime, and framework-neutral dev-server primitives.

### `diffpack-vite-compat`

- Vite config evaluation and aliases.
- `define`, `import.meta.env`, `import.meta.glob`, manifests, and Vite defaults.
- Any eventual Vite/Rollup plugin bridge.

### `diffpack-next`

- App and Pages Router discovery/adapters.
- RSC directives, references, manifests, and runtimes.
- Next images, fonts, metadata, `styled-jsx`, and Next output conventions.

### `diffpack-tanstack`

- TanStack config conventions and environment specialization.
- Route-tree generation and splitting.
- Server functions, manifests, and TanStack runtimes.

## Migration loop

Every slice follows the same loop:

1. Select one cohesive capability and inspect all inbound/outbound references.
2. Put shared data types in the lowest valid crate before moving behavior.
3. Move the implementation and its tests together.
4. Leave a root compatibility re-export only when existing callers still need it.
5. Replace `crate::` references with explicit lower-layer crate dependencies.
6. Run formatting, focused tests, workspace compilation, and diff checks.
7. Remove the compatibility re-export as soon as all callers have migrated.
8. Record newly discovered coupling here or in `crates/README.md`.

Do not create empty placeholder APIs to make a dependency diagram look finished.
A crate boundary counts only when it owns implementation or a stable contract.

## Validation gates

The checks are executable locally and in
`.github/workflows/crate-extraction.yml`. Run after every slice:

```sh
./scripts/check-extraction.sh slice <changed-crate>
```

Run after each phase:

```sh
./scripts/check-extraction.sh phase
```

Run the complete suite before declaring the extraction finished (or select the
`final` input when manually dispatching the GitHub Actions workflow):

```sh
./scripts/check-extraction.sh final
```

The known Tailwind corpus test requires the external corpus inputs to be present;
absence of those inputs must be reported separately from extraction regressions.

`check-crate-boundaries.sh` derives workspace dependencies from Cargo metadata
and rejects upward or sibling dependencies. This is intentionally independent
of source layout, so moving a file cannot silently invert the architecture.

Before declaring the reorganization complete:

```sh
cargo tree -p diffpack-core
cargo tree -p diffpack-default-loader
cargo tree -p diffpack-web
cargo tree -p diffpack-vite-compat
cargo tree -p diffpack-next
cargo tree -p diffpack-tanstack
rg 'pub use diffpack::' crates
rg 'crate::(next_|rsc|route_|server_fn|vite_)' crates/diffpack-core
```

The final two searches must return no matches. `diffpack-core` and
`diffpack-default-loader` must not depend on the root `diffpack` package.

## External provider checkpoint

`diffpack-core::ProviderPipeline` is an immutable, cheaply cloned ordered hook
chain. The live bundler entry point
`Bundler::discover_direct_with_config_and_providers` now executes provider
resolution, loading, and transforms during both parallel cold discovery and
serial incremental rebuilds. Built-in virtual/query loaders retain priority.

Provider-resolved JavaScript and TypeScript are supported. Transform-emitted
assets flow into the reachable module records and the ordinary deduplicated
asset output pass. Provider-declared externals are retained as runtime external
edges and never enter module discovery. Extra watch files join incremental
invalidation. Incoming provider source maps still return an explicit error until
map composition moves behind the extracted rendering contract; they must never be
silently discarded.

## Phase order

1. Core records: diagnostics, graph delta, persistent graph, reachability.
2. Core compilation: transform IR, linker, tree shaking, chunks, rendering/maps.
3. Default providers: resolver, loader dispatch, CSS/assets/preprocessors.
4. Web layer: HTML, browser emit, HMR, dev-server primitives.
5. Vite compatibility.
6. Next integration.
7. TanStack integration.
8. Dependency reversal and root CLI reduction.
9. Compatibility-re-export removal and complete verification.

## Definition of done

- Root `src/lib.rs` contains no implementation modules; it may be removed if the
  CLI needs no library facade.
- Root `src/main.rs` composes public APIs from the workspace crates.
- No crate under `crates/` depends on the root `diffpack` package.
- No framework name or behavior appears in `diffpack-core`.
- Incremental rebuild, graph, chunking, source-map, conformance, web, Next, and
  TanStack tests retain their prior behavior.
- External tools can supply ordered resolve/load/transform providers without
  graph internals depending on the tool.
