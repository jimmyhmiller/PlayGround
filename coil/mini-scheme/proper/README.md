# proper — the same interpreter written PROPERLY in idiomatic native Coil

`scheme-proper.coil` is the same metacircular evaluator, but written the way a Coil
programmer actually would: a **sum-typed** Scheme value (`defsum Val`) with **unboxed
integers** (only pairs and closures touch the heap), `match` for dispatch, and an
arena allocator. No per-integer boxing, no GC — a batch tree-walking interpreter.

Contrast with the two earlier versions, which represent every value — including every
integer — as a heap-allocated GC-managed `Val` so the transparent-GC metaprogram can
root them uniformly.

Benchmark, all interpreting the SAME target **(fib 30)**:

| host running the interpreter                 | fib(30) | vs Chez | allocations |
|----------------------------------------------|---------|---------|-------------|
| Chez (native compiler)                       | 305 ms  | 1×      | —           |
| **proper native Coil** (unboxed sum + arena) | 970 ms  | **3.2×**| 21.5 M      |
| Petite Chez (interpreter)                    | 4.46 s  | 14.6×   | —           |
| boxed (scheme metaprogram + GC)              | 9.38 s  | 30.8×   | 377 M       |

What this shows:

- **Proper idiomatic Coil is 3.2× off Chez's native compiler — and beats Petite**, one
  of the fastest Scheme *interpreters*, by 4.6×. That is a strong result for a
  hand-written tree-walker.
- **The whole ~10× gap between "proper" and "boxed" is the value representation.**
  Unboxed integers cut allocations from 377 M to 21.5 M (17×). The boxing exists in
  the metaprogram version only so the transparent-GC transform can root every value
  uniformly as a `(ptr Val)` — that uniformity is what makes the GC automatic, and
  it is exactly what costs the 10×.
- **The remaining 3.2× to Chez is optimization** (register allocation, inlining,
  native fixnum arithmetic), not the interpreter design.

Caveat: this version uses an arena (it never frees — a batch interpreter). A
production version would collect; the point here is the value representation, which
is the dominant factor. GC would bound memory at some overhead, but the unboxed-int
win — the 10× — would remain.

    ../../coil build scheme-proper.coil -o /tmp/sp && /tmp/sp
