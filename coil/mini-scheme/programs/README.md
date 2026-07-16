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

## Speed: on par with Chez's compiler on compute, far faster on startup

Our programs are AOT native binaries (~1.5 ms startup); `chez --script` has a ~35 ms
startup floor. So short programs look lopsided (tak 1.4 ms vs 36 ms = ~23×). On a
compute-dominated run that dwarfs startup, the generated code is roughly level with Chez:

| fib(33)        | wall-clock (min) | minus ~startup |
|----------------|------------------|----------------|
| our native     | 21.0 ms          | ~19.5 ms       |
| chez --script  | 56.8 ms          | ~21.8 ms       |

So the Scheme the metaprogram compiles is genuine native code, competitive with Chez's
own compiler — and `fib(30)` compiled directly (6 ms) vs run through the metacircular
evaluator (~330 ms) shows the ~55× gap between compiling and interpreting the same fib.

## Frontend scope

`scm2coil` is FIRST-ORDER: top-level named functions, `if`/`begin`/`quote`, binary
`+ - * < =`, and `cons/car/cdr/eq?/null?/pair?/number?/symbol?/not`. It does not take
functions as values (no `(map f xs)`) or `let`/`lambda` expressions — the full language
with closures is what the metacircular evaluator in `../meta` provides. Programs here stay
in that first-order subset; each is checked against Chez by `./run.sh`.
