# native — the same little Scheme interpreter, HAND-WRITTEN in native Coil

`schemenative.coil` is the same first-order metacircular evaluator as
`../meta/evalcore.scm`, mirrored function-for-function, but written directly in
native Coil over the `Val` runtime (with the transparent-GC transform). It embeds
the same fib target and runs it.

The point is the comparison against the *transpiled* version (`../meta/evalcore.coil`,
generated from the Scheme source by our scheme metaprogram). If the two run at the
same speed, the metaprogram + transpiler add no overhead over hand-writing.

Benchmark, all interpreting **(fib 30)** — 377M allocations, GC bounded at 20 000 live:

| host running the same interpreter        | fib(30) | vs Chez |
|------------------------------------------|---------|---------|
| Chez (native compiler)                   | 307 ms  | 1×      |
| Petite Chez (interpreter)                | 4.4 s   | ~14×    |
| transpiled from evalcore.scm (metaprogram)| 10.5 s | ~34×    |
| **hand-written native Coil**             | 10.1 s  | ~33×    |

**Hand-written native Coil ≈ transpiled-from-Scheme** (within run-to-run noise) — so
our "scheme mode" (write Scheme, compile it via the GC metaprogram) produces code
indistinguishable from writing the interpreter by hand in Coil. The whole ~33× gap
to Chez is value representation and optimization (unboxed fixnums, inlining, register
allocation) — not the metaprogram, the transpiler, or the transparent GC (which is
free: the collector keeps the working set at 20 000 objects while 377M are allocated).

    ../../coil build schemenative.coil -o /tmp/sn && /tmp/sn      # run it
