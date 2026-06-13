# Benchmarks: funct vs CPython

A few small programs run in both funct and Python to get a rough sense of where
funct's bytecode VM stands. **funct is young and unoptimized**; CPython is
decades of tuned C. The goal here is a sanity check and to find hotspots, not to
win.

## Setup

- funct: `cargo build --release`, run via `./target/release/funct run <f>.ft`
- Python: 3.14.2, `python3 <f>.py`
- Timing: [`hyperfine`](https://github.com/sharkdp/hyperfine), 5 runs + 1 warmup,
  on the same machine (Apple Silicon). Times are whole-process wall clock
  (interpreter startup included).

Run them all:

```bash
cargo build --release
for b in fib loops collatz list; do
  hyperfine -N -w 1 "./target/release/funct run bench/ft/$b.ft" "python3 bench/py/$b.py"
done
```

Each pair computes the same result (verified equal) — see `ft/` and `py/`.

## Results

Three engines, same program, identical result verified for each row. Reported as
**relative speed** (ratios are thermal-invariant — the machine throttles under
sustained load, so back-to-back ratios are far more stable than absolute ms).

| Benchmark | What it stresses | funct vs rhai 1.25 | funct vs Python 3.14 |
|---|---|---|---|
| `fib(30)` | non-tail recursion, calls | **2.18× faster** | 3.0× slower |
| `loops` (2000×2000) | tight `for` loops, mutable counters | **1.13× faster** | 2.4× slower |
| `collatz` (1..100k) | `while` loops, mutable locals, `/` `%` | **1.24× faster** | 2.7× slower |
| `list` (n=10k) | `map`/`filter`/`fold` building a list | 1.66× slower | **1.2× faster** |
| startup (`println(1)`) | boot + prelude compile | ~on par | **9.5× faster** |

**Headline: funct now beats rhai (the language it replaces) on every compute
benchmark except array building, and is ~2.4–3× off CPython 3.14** — a mature
computed-goto interpreter with adaptive opcode specialization. For a young
switch-based bytecode VM that's a strong place to be. It started this round
*behind* rhai on the iterative benchmarks; a series of interpreter optimizations
(below) flipped that.

## Interpreter optimizations (this round)

Profiled with macOS `sample`; each change verified against the full test suite.

1. **`let mut` escape analysis.** Only closures-captured mutables get an
   `Arc<Mutex>` cell; the rest are plain stack slots. `collatz` 1.6× faster,
   `loops` 1.3× faster. (See `compiler.rs::lambda_captured_names`.)
2. **Fast dispatch loop.** `run()` skipped per-instruction `current_line()` +
   breakpoint bookkeeping for the common (`StopWhen::Never`) case, and dropped a
   redundant `step()` dispatch layer (`run_fast`).
3. **`Instr` is `Copy`.** Moved the `MakeClosure` capture `Vec` into a side
   table on `FnProto`, so the per-instruction instruction read is a register
   copy, not a heap-touching clone. Cut `step_inner` self-time ~17%.
4. **Shrank `Value` 72 → 40 bytes.** `List`/`Tuple` hold the 64-byte
   `imbl::Vector` behind an `Arc` so the enum stays small — `Value` is
   moved/dropped/cloned on every stack op. Cut `push`/`drop`/`clone` self-time
   ~44%.

Remaining ceiling: `step_inner` dispatch is ~57% of runtime — inherent to a
switch interpreter. Further gains would need superinstructions, a register VM,
or computed-goto-style threading (bigger architectural changes).

## Earlier findings

**1. `let mut` escape analysis (DONE — big win for iterative code).**
Originally *every* `let mut` lived in a shared `Cell` (`Arc<Mutex>`) so a closure
could capture it, costing a mutex lock per read/write. Tight loops hammer mutable
counters (`total`, `i`, `j`, `x`, `c`), so they paid that cost every iteration;
`fib` uses only immutable params (no cells), which is why funct was already
fastest there. The compiler now runs an escape analysis (`lambda_captured_names`
in `compiler.rs`): only a `let mut` actually captured by a nested closure gets a
`Cell`; the rest are plain stack slots (direct load/store). Captured mutables
still share correctly (closure tests pass). Effect:

| benchmark | before | after |
|---|---|---|
| `collatz` | 4.67 s | **2.88 s** (1.6× faster) |
| `loops` | 865 ms | **672 ms** (1.3× faster) |

That closed most of the gap to rhai on iterative code.

**2. Persistent vectors fixed list-building (was O(n²)).**
Lists/records now use `imbl::Vector`/`OrdMap` (structural sharing). Before this,
`push` cloned the whole list so `map`/`filter` were quadratic — the `list`
benchmark was 845 ms (≈65× slower than rhai). It is now 29 ms (≈2× rhai), and
indexing did not regress (64-element list = single chunk → still ~O(1)).

**3. `for` beats hand-rolled `while`.**
`for i in 0..n` runs ~1.9× faster than `while i < n { ...; i = i + 1 }` — `for`
avoids the per-iteration compare + increment + reassign as separate bytecode.

## Files

```
bench/ft/           fib.ft  loops.ft  collatz.ft  list.ft     # funct
bench/rhai/         fib.rhai  loops.rhai  collatz.rhai  list.rhai
bench/py/           fib.py  loops.py  collatz.py  list.py      # python (idiomatic)
                    list_fair.py                               # python immutable-append
bench/rhai-runner/  minimal `rhai-run <file>` harness (rhai 1.25)
```
