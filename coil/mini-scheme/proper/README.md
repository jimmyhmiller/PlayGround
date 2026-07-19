# proper — the same interpreter written PROPERLY in idiomatic native Coil

`scheme-proper.coil` is the metacircular evaluator written the way a Coil programmer
actually would: a **sum-typed** Scheme value (`defsum Val`) with **unboxed integers**
(only pairs/closures heap-allocate), `match` for dispatch, an arena allocator, and
the special-form symbol ids **interned once** (not per dispatch).

## Result: parity with Chez

Benchmark, all interpreting the SAME target **(fib 30)**, min-of-15:

| host running the interpreter                 | fib(30) | vs Chez |
|----------------------------------------------|---------|---------|
| **proper native Coil** (unboxed sum + arena) | **285 ms** | **~1.0×** |
| Chez (native compiler)                       | 297 ms  | 1×      |
| Petite Chez (interpreter)                    | 4.39 s  | ~15×    |
| boxed (scheme metaprogram + GC)              | 9.38 s  | ~31×    |

A hand-written tree-walking interpreter in idiomatic Coil runs **at parity with Chez
Scheme's optimizing native compiler**, and ~15× faster than Petite (Chez's interpreter).

## How we got here — and the answer to "is Coil slow?"

The first cut of this file was 970 ms (~3.2× off Chez). A profile (`sample`) showed
**~60% of the time in `intern`**: my `symeq` re-interned the strings `"quote"`/`"if"`/
`"lambda"`/`"define"` — a linear scan of the symbol table — on *every* expression, up
to four times each. The Scheme version never does this: `'quote` is interned once at
read time and `eq?` on symbols is O(1). Caching the four ids (interned once at setup,
integer-compared at dispatch) cut runtime 3× and closed the entire gap.

So the ~3.2× was **not** Coil being slow, and it was **not** the nature of the test —
it was a missing optimization in the interpreter *code* (redundant interning). Once
the interpreter does what the algorithm intends, Coil's LLVM codegen is competitive
with Chez's on this workload.

Residual language-level differences are minor here: our `Val` is 16 bytes (Chez uses
8-byte tagged words) and a few small helpers don't fully inline — but at parity, they
clearly cost little. And the earlier boxed version's ~31× was the value representation
(377M allocations from boxing every integer), which the metaprogram requires only so
the transparent-GC transform can root every value uniformly.

Caveat: this uses an arena (never frees — a batch interpreter); a production version
would collect. The unboxed-int representation is the win regardless.

    ../../coil build scheme-proper.coil -o /tmp/sp && /tmp/sp
