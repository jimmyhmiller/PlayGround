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

## Timing: real compute, not startup

Each program runs its core an iteration count calibrated to ~200 ms of real work (a
tail-recursive `bench` driver threading the result), so the benchmark measures
*computation*, not the ~1 ms of process startup that dominated trivial one-shot runs.
Each program reads a monotonic clock (`clock_gettime`) around the evaluation and prints
`[time] compute_ns=…`; `./run.sh` shows it next to each result. The iteration driver
only works because of the TCO fix in `../gcauto2.coil` — before it, a tail-recursive
*allocating* loop leaked GC roots and blew up the heap (see that commit).

Compute vs compute (ours = internal timer, load-independent; chez = wall minus its ~35 ms
`--script` startup floor):

| program      | ours   | chez   | faster    | notes |
|--------------|-------:|-------:|-----------|-------|
| fact         | 192 ms | 221 ms | ours 1.2× | tight arithmetic loop |
| gcd          | 198 ms | 215 ms | ours 1.1× | tight arithmetic loop |
| fib-direct   | 228 ms | 247 ms | ours 1.1× | tree recursion, no alloc |
| evenodd      | 196 ms | 438 ms | ours 2.2× | deep mutual tail recursion |
| tak          | 193 ms | 157 ms | chez 1.2× | deep recursion |
| binary-trees | 176 ms | 127 ms | chez 1.4× | **GC stress** (53 collections) |
| ackermann    | 199 ms |  80 ms | chez 2.5× | very deep recursion |
| reverse      | 206 ms |  71 ms | chez 2.8× | **GC stress** (80 collections) |
| sum-range    | 204 ms |  51 ms | chez 4.0× | **GC stress** (23 collections) |
| append-len   | 202 ms |  62 ms | chez 3.3× | **GC stress** (39 collections) |

The honest split: on **compute-bound** code (arithmetic loops, non-allocating recursion)
the generated native is competitive with or faster than Chez. On **allocation-bound**
code — the GC stressors that build and discard millions of cons cells — Chez is 1.4–4×
faster, because it has a generational, moving collector over unboxed values, while ours is
a non-moving mark-sweep over boxed `(ptr Val)` cells. That gap is the value representation
the transparent-GC transform requires (uniform rooting), not a codegen deficit; it is the
same reason the metacircular evaluator — where allocation is *mixed* with heavy non-alloc
dispatch — stays ~1.15× (allocation is a smaller share there, so GC quality matters less).

Startup, for reference, is where AOT still wins outright: our binaries start in ~1.3 ms
vs Chez's ~35 ms `--script` floor. But that is not what these numbers measure — this is
real work.

## Frontend scope

`scm2coil` is FIRST-ORDER: top-level named functions, `if`/`begin`/`quote`, binary
`+ - * < =`, and `cons/car/cdr/eq?/null?/pair?/number?/symbol?/not`. It does not take
functions as values (no `(map f xs)`) or `let`/`lambda` expressions — the full language
with closures is what the metacircular evaluator in `../meta` provides. Programs here stay
in that first-order subset; each is checked against Chez by `./run.sh`.
