# mini-scheme — a metacircular Scheme evaluator under transparent metaprogram GC

A working Scheme `eval`/`apply`, written in the **transparent-GC dialect**: every
value is `Val`, and the source contains **no pointers, no allocation calls, and no
rooting**. A metaprogram transform (`gcauto2.coil`) inserts the entire precise
garbage collector — heap references, allocation, and root management — automatically.

- **`sval.coil`** — the runtime a Scheme program assumes: a GC-managed tagged `Val`
  (int / pair / symbol / bool / closure / primitive), a precise mark-sweep heap with
  a tag-dispatched tracer, `cons`/`car`/`cdr`/…, symbol interning, and the mutable
  global environment (a permanent GC root). Hand-written, like a language's runtime.
- **`gcauto2.coil`** — the transform. It rewrites the managed type `Val` → `(ptr Val)`,
  frames each function, roots managed parameters, and A-normalizes managed call
  arguments AND let-bindings into rooted temporaries — so a value from one allocation
  survives the next. The evaluator author writes none of this.
- **`scheme.coil`** — the evaluator itself, in transparent style: `seval`/`apply-fn`,
  environments (local assoc list + mutable global for recursion), `quote`/`if`/
  `lambda`/`define`/`begin`, and the primitives. Reads a program from stdin, using the
  Coil compiler's own reader to parse it, and prints the result.

## Run it / compare to Chez

    ./run-all.sh          # builds, runs every tests/*.scm through mini-scheme AND chez

Every result is identical to Chez Scheme. The headline is `fib 25`: **4,248,846 cells
allocated, 214 collections, 20,000 peak live** — the transparent GC reclaims millions
of allocations under real pressure, and the answer (`75025`) matches Chez exactly.

## Benchmark (vs Chez Scheme)

`./bench.sh` (needs hyperfine, `chez`, `petite`). mini-scheme is a naive tree-walking
interpreter; Chez is a production native-code compiler; Petite is Chez's interpreter.
Numbers below are startup-subtracted compute on **fib(30)** (2.7M calls, **47M heap
allocations** in mini-scheme):

| implementation                    | startup | fib(30) compute | vs mini-scheme  |
|-----------------------------------|---------|-----------------|-----------------|
| mini-scheme (tree-walker + GC)    | 1.8 ms  | ~2700 ms        | 1×              |
| Petite Chez (interpreter)         | 24 ms   | ~33 ms          | ~82× faster     |
| Chez (native compiler)            | 36 ms   | ~4 ms           | ~630× faster    |

Two honest takeaways:

- **The transparent metaprogram GC costs ~18% of runtime.** A no-collection build runs
  fib(30) in 2.22 s vs 2.70 s with the collector on (47M allocations, 2381 collections,
  20 000 peak live). The precise mark-sweep + shadow-stack the transform inserted is
  cheap; the ~82× gap to Petite is the *interpreter design* (linear env lookups, boxed
  integers, consed argument lists, tree-walking), not the GC or the metaprogram.
- **mini-scheme starts up fastest** — it's a native binary; the Chez runtimes pay
  20–36 ms of init before running anything.
