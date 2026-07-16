# programs — Scheme programs compiled by the GC metaprogram, checked against Chez

Each `.scm` here is a real Scheme program compiled straight to a native binary through
the metaprogram — `scm2coil.py` maps it onto the `Val` runtime (`../sval.coil`) and the
transparent-GC transform (`../gcauto2.coil`) inserts the collector. `./run.sh` compiles
every program and diffs its output against **Chez Scheme running the identical source**.

## Result: 9/9 match Chez exactly

| program        | computes                        | output      |
|----------------|---------------------------------|-------------|
| fact           | 12!                             | 479001600   |
| ackermann      | Ackermann(3,6)                  | 509         |
| tak            | Takeuchi tak(18,12,6)           | 7           |
| sum-range      | sum 1..1000 over a cons list    | 500500      |
| gcd            | subtractive gcd(1071,462)       | 21          |
| evenodd        | mutual recursion, evn(100000)   | 1           |
| reverse        | reverse a list (accumulator)    | (1 2 3 4 5 6) |
| append-len     | length of two appended ranges   | 50          |
| fib-direct     | fib(30) compiled (not interpreted) | 832040   |

These exercise deep recursion, non-tail recursion, heavy consing (`sum-range` builds a
1000-element list under GC), mutual recursion, and list printing — all reclaimed by the
metaprogram's collector, with no pointers or rooting in the source.

## Timing: total vs compute

Each compiled program reads a monotonic clock (`clock_gettime`) around the actual
evaluation and prints `[time] compute_ns=…` to stderr, so we can separate the pure
**compute** from process **startup/teardown** that a wall-clock also sees. `./run.sh`
prints the compute time next to each result; `total` below is `hyperfine` wall-clock.

| program     | ours total | ours compute | chez total |
|-------------|-----------:|-------------:|-----------:|
| fact        | 1.3 ms     | 0.007 ms     | 34.6 ms    |
| gcd         | 1.2 ms     | 0.005 ms     | 34.8 ms    |
| reverse     | 1.3 ms     | 0.014 ms     | 34.9 ms    |
| append-len  | 1.3 ms     | 0.012 ms     | 35.1 ms    |
| evenodd     | 1.3 ms     | 0.053 ms     | 35.7 ms    |
| sum-range   | 1.4 ms     | 0.030 ms     | 35.4 ms    |
| ackermann   | 2.0 ms     | 0.656 ms     | 35.4 ms    |
| tak         | 1.6 ms     | 0.135 ms     | 35.6 ms    |
| fib-direct  | 6.0 ms     | 4.21 ms      | 40.0 ms    |

Two things fall out. (1) Our binaries are AOT natives: ~1.3 ms of process startup plus the
compute, versus Chez's ~35 ms `--script` startup floor — so most small-program wall-clock
on both sides is startup, not work. (2) On compute, the generated code is genuine fast
native: fib(30) runs in 4.2 ms (and fib compiled directly, 6 ms wall, vs ~330 ms through
the metacircular evaluator, is the ~55× compile-vs-interpret gap on the same source). A
compute-dominated run where work dwarfs startup — fib(33) — puts our native ~19.5 ms
against Chez's ~21.8 ms (startup subtracted): level with Chez's own compiler.

Caveat: our `compute` is run-only (compilation happened ahead of time); Chez's `--script`
compiles at load, so its wall-clock also carries compile time. The comparison is native
AOT vs compile-and-go, which is why our startup is tiny.

## Frontend scope

`scm2coil` is FIRST-ORDER: top-level named functions, `if`/`begin`/`quote`, binary
`+ - * < =`, and `cons/car/cdr/eq?/null?/pair?/number?/symbol?/not`. It does not take
functions as values (no `(map f xs)`) or `let`/`lambda` expressions — the full language
with closures is what the metacircular evaluator in `../meta` provides. Programs here stay
in that first-order subset; each is checked against Chez by `./run.sh`.
