# gc-rust benchmark suite — gc-rust vs Rust, Go, JVM

Four classic benchmarks, one algorithm each, implemented in all four languages
and **verified to produce identical numeric output** before timing. The point is
an honest, apples-to-apples measurement of gc-rust's codegen and garbage
collector against mature toolchains.

## Layout

```
suite/<bench>/<bench>.rs    Rust   (taken from a public benchmark suite, credited)
suite/<bench>/<bench>.go    Go     (")
suite/<bench>/<bench>.java  Java   (")
suite/<bench>/<bench>.gcr   gc-rust (faithful port of the same algorithm)
```

Benchmarks: `nbody`, `spectralnorm`, `fannkuchredux`, `binarytrees`.

## Credits — competitor sources are NOT written by us

The Rust / Go / Java programs are taken **verbatim** (single-threaded,
standard-library variants) from public benchmark suites, with their original
attribution headers preserved in each file:

- **Programming-Language-Benchmarks** (https://github.com/hanabi1224/Programming-Language-Benchmarks),
  MIT-licensed — source of most of the Rust/Go/Java files here.
- **The Computer Language Benchmarks Game**
  (https://salsa.debian.org/benchmarksgame-team/benchmarksgame/), BSD-3-Clause —
  source of the single-threaded Java spectral-norm / fannkuch-redux and the
  underlying algorithms (n-body by Christoph Bauer; fannkuch-redux by Oleg
  Mazurov; spectral-norm; binary-trees).

The `.gcr` files are faithful gc-rust ports of those same algorithms, written for
this comparison.

## Fairness notes

- **Single-threaded.** Only single-threaded, std-only variants are used, so the
  numbers reflect core codegen + GC, not thread-scaling.
- **One documented edit:** Go's fannkuch-redux is goroutine-parallel in every
  published version, so it is pinned to one OS thread via `GOMAXPROCS(1)`
  (a one-line change, commented in the file). No algorithm was modified.
- **gc-rust args are hardcoded** in each `.gcr` to match the CLI `N` the harness
  passes to the other three languages (gc-rust `main()` takes no argv).
- **JVM times include process startup + partial HotSpot warmup** — the real cost
  of running Java.

## Running

```
python3 bench/run_suite.py     # compile all, verify outputs match, hyperfine -> bench/results.json
python3 bench/gen_report.py     # bench/results.json -> bench/report.html (interactive)
```
