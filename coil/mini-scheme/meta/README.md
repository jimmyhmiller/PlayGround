# meta — the SAME metacircular Scheme evaluator, run by Chez and by our Coil scheme metaprogram

This is the fair comparison. `evalcore.scm` is a first-order metacircular Scheme
evaluator (a tree-walking `seval`/`sapply` over an assoc-list environment, closures
and primitives as tagged data). It is **portable Scheme** — Chez and Petite run it
directly — and it is written in the exact subset our Coil "scheme mode" compiles.

- **`sval.coil` + `gcauto2.coil`** are the *scheme metaprogram*: the GC-managed `Val`
  runtime (cons/car/…/arithmetic as Val ops) and the transparent-GC transform. This
  is the metaprogram that "lets us do scheme" — it adds precise GC and the Val APIs,
  nothing else.
- **`scm2coil.py`** is a thin syntactic frontend: it maps the Scheme subset onto the
  Val runtime (a number → `(mk-int n)`, `(if c t e)` → `(if (truthy c') …)`, `(+ a b)`
  → `(s-add a' b')`, a function → a `defn` over `Val`, …). The generated Coil is then
  compiled to native by the GC metaprogram. (It could equally be a Coil `(meta …)`;
  the substance — GC + Val APIs — is the Coil metaprogram.)

So both Chez and our system run **the same evaluator source** over the same target
program. `./run.sh bench` reproduces the numbers below (target: `(fib 30)`):

| host running evalcore.scm                     | fib(30) via the interpreter | vs Chez |
|-----------------------------------------------|-----------------------------|---------|
| Chez (native compiler)                        | 306 ms                      | 1×      |
| Petite Chez (interpreter)                     | 4.4 s                       | ~14×    |
| **our Coil** (scheme metaprogram → native+GC) | 9.5 s                       | **~31×**|

What the numbers say:

- **The transparent metaprogram GC is effectively free.** Our evaluator makes
  **377 million allocations** (every integer is boxed). With the collector on it runs
  in 9.2 s holding 20 000 live; with collection DISABLED it runs *slower* (9.8 s),
  because 377M un-freed objects (~9 GB) thrash malloc. The precise mark-sweep the
  transform inserted pays for itself.
- **The ~31× gap to Chez is compiler quality, not the GC or the metaprogram.** Chez
  is an optimizing native compiler (unboxed fixnums, register allocation, inlining);
  our frontend boxes every value and runs zero optimization passes. ~31× off Chez —
  and *faster to start up and within ~2× of Petite's optimized interpreter* — is a
  good showing for a naive box-everything compilation whose only cleverness is that a
  metaprogram wrote the garbage collector.
