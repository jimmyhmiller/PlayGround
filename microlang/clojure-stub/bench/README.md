# The microclj performance suite

> **Running the tests: use `--release`.** `cargo test` defaults to debug, and
> this runtime is ~10x slower unoptimized — the correctness suites take **7
> minutes** in debug and **26 seconds** in release:
>
>     cargo test --release -p clojure-stub --features jit \
>         --test run --test jit --test seq_oracle
>
> The GC suites are deliberately not in that loop (they need the verify heap and
> are slow by design). Run them for changes to GC rooting, the call convention,
> or `apply`/frames:
>
>     MICROLANG_GC_VERIFY=1 cargo test --release -p clojure-stub --features jit --test gc_generational
>     cargo test --release -p clojure-stub --features jit --test gc_stress_library -- --ignored
>
> And while iterating, prefer a smoke `.clj` run straight against
> `target/release/microclj` (~0.5s) — it can also be diffed against real
> Clojure with `clojure -M`.


    cargo build --release --features jit -p clojure-stub --bin microclj
    clojure-stub/bench/bench.sh              # the real thing (~3s warmup/workload)
    clojure-stub/bench/bench.sh --quick      # for iterating; NOT publishable
    clojure-stub/bench/bench.sh --out DIR    # keep the raw TSVs

Requires `clojure` on `PATH`. Goal: **microclj consistently at or under real
JVM Clojure**, measured honestly.

## The design, and the bugs that motivated each rule

This replaces a set of ad-hoc scratchpad shell scripts. Those scripts produced
a table that was wrong in both directions, and every rule below is a defence
against one of the specific ways it was wrong.

**One file, both runtimes.** `suite.clj` is executed byte-for-byte by microclj
and by JVM Clojure. The predecessor kept the workloads in a bash associative
array *and* re-typed them into a `.clj` oracle; two copies drift and you end up
comparing two different programs. This is also why `System/nanoTime` exists in
the dialect (see `host_jvm.clj`) — it is `%nanos`, which the runtime always
had and the host layer had simply never exposed.

**Every workload returns a checksum, and both runtimes must agree.** This is
the load-bearing rule. A recorded predecessor run has microclj doing
`vecbuild` in 0.16ms against Clojure's 27ms — not a 170x win, but a workload
that never ran. (Real answer: ~54ms.) `compare.clj` **refuses to print a
table** when checksums disagree, because a ratio between two different
computations is not a measurement. Checksums must also be cheap: an `nth`
probe into a lazy seq re-walks it and measures the checksum instead of the
workload. For lazy workloads `count` is the checksum — it forces the seq,
which is exactly the work, and an unforced-seq bug shows up as a wrong count.

**Time-budgeted warmup, not a fixed number of calls.** The predecessor warmed
up with two calls. Measured here: a cold JVM runs `vecbuild` at 31ms and a warm
one at 10.7ms, so timing from cold flatters microclj by ~2.5x. Both tiers JIT;
both get the budget. This is why the old table's JVM column was systematically
pessimistic.

**Median of many samples, plus an explicit spread.** A single timed call cannot
distinguish a regression from the scheduler. `spread` = (p95-min)/median; over
15% the table marks the row `NOISY` rather than quietly presenting it as
evidence. In practice microclj sits at 2.6–8% and the noise is on the JVM side,
where GC pauses dominate the allocation-heavy workloads. **GC pauses are not
filtered out** — they are real cost, and `spread` is how they are disclosed.
`samples` is 25 because the medians of the pause-heavy workloads wobble at 15.

**Timing is in-process.** It cannot be wall-clock of the process: microclj
spends ~340ms loading its prelude, which would swamp every workload.

**`--quick` is the same harness with smaller budgets**, never a different one,
and it verifies it actually patched the knobs — a harness that silently runs
settings other than the ones it reports is worse than no harness.

## Reading the output

`ratio` is microclj median / JVM median; lower is better, under 1.00x means we
win. `geomean` tracks the overall gap; `worst` is the tail that "consistently"
in the goal refers to.

## Caveat: `raw-loop` measures the autovectorizer

JVM Clojure runs `raw-loop`'s 5M iterations of `inc`/`+`/`<` in ~1.9ms, about
one cycle per iteration. That is C2 unrolling and vectorizing the reduction,
not a language-level difference. Treat that row as a known outlier: closing it
is a "build an autovectorizer" project, unrelated to closing `apply` or
`transduce`.
