# Competitive benchmark harness

Benchmarks diffpack against esbuild, rolldown (pinned 1.2.0, same as
`oracle/`), rspack, and vite on shared, runtime-verified corpora, plus the
real TanStack Start app in `integration/tanstack-start-reference`.

Methodology and measured results: [docs/COMPETITIVE_BENCHMARKS.md](../docs/COMPETITIVE_BENCHMARKS.md).

## Setup

```console
# from the repository root
cargo build --release
cd bench
npm ci

# the real-app benchmark also needs the fixture's dependencies:
cd ../integration/tanstack-start-reference
npm install --no-save   # npm ci currently fails: the pinned lockfile is out of sync (missing lru-cache@11.5.2)
```

`/usr/bin/time` (GNU time, for `-v`) must be installed.

## Run

```console
cd bench
node run.mjs                 # everything: all corpora + the real app (~20 min)
node run.mjs --skip-app      # synthetic corpora only
node run.mjs --only-app      # real app only
```

Options (defaults in parentheses):

- `--profiles tiny,realistic` — module-size profiles (both)
- `--sizes 1000,10000` — module counts (both)
- `--tools diffpack,esbuild,rolldown,rspack,vite` — subset of tools
- `--cold-runs 5` — timed cold builds per pair (plus 1 untimed warmup)
- `--edits 5` — timed incremental edits per pair (plus `--warmup-edits 2`)
- `--app-runs 5` — timed runs per real-app case
- `--imports 4` — imports per module
- `--keep-work` — keep the temporary corpus/output directory
- `--skip-app` / `--only-app`

Results accumulate into `results/results.json` (re-running a subset updates
only those cells), and a summary table is printed at the end.

## Layout

- `run.mjs` — orchestrator: corpus generation, cold timing (fresh process per
  run, caches deleted before every run), peak RSS via `/usr/bin/time -v`,
  output sizes, runtime verification, real-app cases, results/summary.
- `gen.mjs` — corpus generator adapted from `oracle/benchmark.mjs`; both
  profiles share the oracle's exact dependency graph and runtime-verified
  value semantics.
- `util.mjs` — timing/verification/size helpers.
- `tools/cold-*.mjs` — fresh-process cold-build drivers (rolldown, rspack,
  vite; esbuild uses its native CLI binary, diffpack its release binary).
- `tools/incr-*.mjs` — incremental drivers: esbuild `context.rebuild()`,
  rolldown `watch` with `incrementalBuild: true` (as in `oracle/benchmark.mjs`),
  rspack `compiler.watch`, diffpack `diffpack watch`. Vite is skipped
  (dev-server HMR is a different axis; `vite build` has no rebuild API).

## Guarantees

- A (bundler, corpus) pair is timed only after its emitted bundle executes
  under node and prints the independently computed expected value; every timed
  cold run and every incremental rebuild is re-verified. Disagreement excludes
  the pair with a recorded error instead of a number.
- No estimated numbers: everything in `results/results.json` comes from a run
  on this machine.
