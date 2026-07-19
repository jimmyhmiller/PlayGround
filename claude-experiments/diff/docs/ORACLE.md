# Bundler oracle and compatibility target

The oracle deliberately operates at module and fixture granularity. It does not
introduce a fine-grained incremental runtime into the bundler.

## Authorities

Every behavioral case has three results:

1. An explicit expected exit status, standard output, and standard error. This
   is the semantic contract.
2. The result of executing a Diffpack bundle.
3. The result of executing a bundle produced by pinned Rolldown 1.2.0.

All three must agree. Generated JavaScript is not compared because two correct
bundlers can emit structurally different programs. Rolldown is a second
implementation, not the ultimate authority: if Rolldown and the explicit
expectation disagree, the case fails instead of teaching Diffpack the reference
implementation's behavior.

Run all behavioral cases:

```console
cd oracle
npm ci
npm test
```

Filter by a case name or tag:

```console
npm test -- esm
npm test -- dynamic-import
node runner.mjs --list
```

The runner builds Diffpack once, creates all outputs in a temporary directory,
executes them with Node, and removes the temporary directory afterward.

## Incremental equivalence

`cargo test --test oracle_incremental` applies an edit sequence to a persistent
`Bundler` and `DirectReachability` session. After an import removal and a source
edit it performs a clean build of the same project and requires:

- identical reachable module sets;
- identical emitted bundle bytes;
- identical runtime output.

This is the primary oracle for incremental correctness. External bundlers do
not need to implement the same incremental algorithm.

## Current target: correct single-chunk applications

The first target is intentionally narrower than Rolldown or Turbopack. Diffpack
must correctly bundle application entry points into one executable chunk for:

- static ESM imports and exports;
- TypeScript syntax removal;
- re-exports and default exports;
- CommonJS default interop;
- JSON modules;
- literal dynamic imports folded into the single chunk;
- package `exports` resolution;
- deterministic module side-effect order;
- content edits, dependency additions, dependency removals, and detached cycles.

The baseline is green only when every fixture tagged `baseline` passes and the
incremental equivalence test passes.

The current target explicitly excludes source maps, minification, CSS, assets,
plugins, top-level await, multiple output formats, and production code
splitting. Those exclusions keep the next milestone small enough to finish
without prematurely designing their incremental machinery.

## ESM linking target

The second target extends the baseline with:

- live imported bindings;
- namespace objects;
- ambiguous star re-exports;
- circular ESM initialization and temporal dead zones;
- mixed ESM/CommonJS cycles;
- side-effect-only modules and package `sideEffects` metadata;
- useful syntax and resolution errors.

These cases are tagged `esm-linking`. Imported identifiers are lowered to live
namespace property reads rather than copied values. Export getters are installed
before module evaluation so cycles observe initialized bindings lazily and
preserve temporal-dead-zone failures. Star re-exports track collisions and
remove ambiguous names. Syntax and resolution diagnostics fail the bundle
command instead of emitting a known-invalid artifact.

The target is green only when all `baseline` and `esm-linking` cases pass.

## Production-parity gates

`npm run parity` compares production behaviors separately from the behavioral
baseline. `npm run parity:strict` makes any missing capability fail CI. The
gates cover unused exported-symbol removal, package `sideEffects` metadata,
source-map emission, and dynamic-import code splitting.

All four gates currently pass. Import demand is collected from the semantic AST
and stored on dependency edges; the renderer strips locally unused pure
declarations and respects package `sideEffects: false`. Source maps contain
mapped source paths and contents. Literal dynamic imports emit lazy CommonJS
chunks that share one runtime registry and module cache.

The gate compares observable runtime output, unused-symbol removal, source-map
artifacts, and chunk counts. It does not require byte-identical JavaScript:
Diffpack uses module factories while Rolldown scope-hoists, so emitted structure
may differ. The matched benchmark additionally requires Diffpack's total
artifact bytes to be no larger than Rolldown's. Tree shaking is deliberately
conservative rather than a complete statement-level implementation.

`npm run bench:capabilities` first runs these pairwise equivalence checks and
only then records release-CLI build times and artifact sizes.

## Adding a case

Create a self-contained directory under `oracle/fixtures`, then add its entry,
tags, and expected output to `oracle/cases.json`. Prefer observable behavior
over snapshots of generated code. A new feature is supported only when its
fixture passes in a release-independent run and, when edits are relevant, has a
fresh-versus-incremental equivalence case.
