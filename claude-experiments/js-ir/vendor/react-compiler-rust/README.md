# Vendored React Compiler HIR (Rust port)

A **subset** of the React Compiler's Rust port, vendored so we can build and test
a real `Cfg` backend over the actual React HIR (not a mock) — see
`docs/HIR_COMPARISON.md` and `crates/jsir-jslir/src/cfg/`.

## Provenance

- Upstream: `facebook/react`, branch `pr-36173` (the React Compiler Rust port).
- Commit: `0dc7f2e55e16e0b462288e3dfecbcaf0eafc1ba5`.
- Path upstream: `compiler/crates/`.

## What's here (and what isn't)

Only the two crates needed for the HIR type definitions + CFG edge semantics:

- `react_compiler_hir` — `HIR`, `BasicBlock`, `Terminal`, `Instruction`,
  `Place`, `Identifier`, `Effect`, `each_terminal_successor`, dominators, etc.
- `react_compiler_diagnostics` — its only internal dependency (source locations,
  diagnostics; serde-only).

The rest of the port (lowering, SSA, inference, reactive-scopes, the oxc/swc
front ends, the CLI) is **not** vendored — we only need the HIR data model and
its control-flow accessors to demonstrate the shared `Cfg` interface. Build deps
are light: `indexmap`, `serde`, `serde_json` (all already in the workspace lock).

## How it's used

`crates/jsir-jslir` takes `react_compiler_hir` as an **optional** dependency
behind the `react-hir` feature. `cfg/react.rs` implements `jsir_jslir::cfg::Cfg`
for `react_compiler_hir::HIR`, and `tests/cross_backend.rs` runs the *same*
generic CFG algorithms (dominators / RPO / reachability) on a real React HIR and
on a JSLIR CFG:

```sh
cargo test -p jsir-jslir --features react-hir
```

## Updating

Re-copy the two crate directories from the upstream commit above. These are
upstream sources under their original license (`facebook/react`, MIT) — see the
upstream repository for license terms.
