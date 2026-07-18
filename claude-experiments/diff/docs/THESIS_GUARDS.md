# Thesis guards: speed, incrementality, low memory

Updated: 2026-07-17

Diffpack's advantage is a build that is **incremental** (a leaf edit re-does only
what changed), **fast**, and **low memory**. Deep integrations (the plugin host,
CSS/asset pipeline, SSR) are exactly the changes that can quietly erode this, so
these properties are guarded by asserted tests, not just benchmarks. Keep every
guard below green when adding such work; if a change genuinely needs to move a
threshold, move it deliberately with a measured justification.

## What is measured

Allocation is counted deterministically by a tracking global allocator
(`src/memory.rs`, installed in `src/lib.rs`). Unlike wall clock or OS RSS,
allocation counts are reproducible, so they make non-flaky assertions. Relaxed
atomics keep the overhead negligible and uniform, so speed numbers stay
representative.

`diffpack::memory::snapshot()` reports live / peak / cumulative bytes and
allocation count; `reset_peak()` rebases the high-water mark.

## Asserted guards

- **Incrementality (transform)** — `bundle_benchmark::thesis_guards::a_leaf_edit_retransforms_exactly_one_module`
  (unit test, parallelism-safe): editing one leaf of a 600-module graph must
  re-transform exactly **1** module.
- **Incrementality (emit)** — `bundle_benchmark::thesis_guards::a_leaf_edit_rerenders_exactly_one_chunk_with_a_bounded_cache`
  (unit test, parallelism-safe): editing one leaf must re-render exactly **1**
  chunk (not the whole bundle), and the per-chunk render cache must stay bounded
  to the live chunk set across 200 edits (no per-edit revision leak). The reused
  chunks' byte-parity against a clean build is proven by `tests/oracle_incremental.rs`
  (`incremental_emit_reuses_every_unchanged_chunk_and_matches_a_clean_build`).
- **Low memory** — `tests/thesis_memory.rs::the_incremental_graph_stays_low_memory`
  (isolated test binary, so process-wide counters are clean):
  - resident graph `< 16_000` bytes/module (measured ~3.5 KB);
  - build peak `< 24_000` bytes/module (measured ~3.4 KB, i.e. the build does not
    hoard transient ASTs);
  - 200 edits to one module grow retained memory by `< 256 KB` (measured ~0.2 KB
    — edits do not accumulate revisions);
  - after dropping the bundler, residual `<` 1/4 of the graph's resident cost
    (measured ~2-3% — teardown releases the graph).
- **Determinism** — `tests/oracle_incremental.rs`: an incremental build after
  structural edits is byte-identical to a clean rebuild — both the single-file
  bundle after structural edits AND the full multi-chunk output tree after a leaf
  edit (the one re-rendered chunk plus every cache-reused chunk).

Thresholds carry 3-5x headroom over measured values: generous enough not to
flake, tight enough to trip on an order-of-magnitude regression (e.g. starting to
retain full ASTs, or leaking a revision per edit).

## Benchmarks (measured, not asserted)

Run these to see current numbers and to calibrate thresholds after a deliberate
change. `N` = module count, `I` = imports/module, `E` = edit count.

```console
cargo run --release -- bundle-scale-direct N I            # clean build phase timings
cargo run --release -- bundle-scale-direct-live N I       # + runtime-verified via node
cargo run --release -- bundle-scale-direct-live-deps N I  # incremental dependency-edit timings
cargo run --release -- bundle-scale-memory N I E          # peak/retained/per-edit-growth memory
```

`bundle-scale-memory` prints, per run: build peak MB, retained MB, bytes/module,
max modules re-transformed by any single edit, retained growth across `E` edits,
and residual after teardown. This is the calibration source for the memory guard
thresholds above.

## When adding the plugin host / deep integration

The plugin host is the highest-risk addition: hosting JS transforms can retain
ASTs/sources across edits or force full rebuilds. Before merging such work:

1. `cargo test --release` (all guards green, no threshold relaxed to pass).
2. `cargo run --release -- bundle-scale-memory 2000 4 200` and confirm
   bytes/module and per-edit growth have not regressed.
3. If a threshold must change, record the new measured baseline and the reason
   here.
