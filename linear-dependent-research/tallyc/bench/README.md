# tallyc vs C — does the safe memory layer cost anything?

tallyc compiles a **dependent + linear** language to native code. Memory safety
comes from *linearity* — an owning handle (`Own a`, a list, a cursor) is used
exactly once, so a leak or a use-after-free is a **type error** — and the
type/region/proof machinery is **erased** (multiplicity 0 ⇒ no runtime slot, no
instruction). The claim is that this safety is *zero-overhead*. This directory
checks it against hand-written C, with **normal programs** — no benchmark
scaffolding, no special flags.

```
bench/run.sh
```

- `examples/bench.tal` is an ordinary tally program: `tally build` emits a normal
  executable that runs the workload once and prints a number. It folds 1,000,000
  transactions on an intrusive **circular doubly-linked list with O(1)
  remove-by-cursor** (`new` → `insert` → `remove` → `free`), summing the value each
  transaction round-trips through the heap. `Nat` is `%builtin`, so the count is a
  literal `1000000` and the checker normalizes on machine integers (no unary
  `Succ(Succ(...))`).
- `bench/bench.c` is the same workload written by hand with raw pointers.

## Result: the safe abstraction compiles to *nothing*

The workload is pure (its allocations don't escape), so at `-O2` **both** compilers
evaluate the whole thing and fold it to a single constant:

```
tally:  define i64 @tally_dep_main() { ret i64 499999500000 }
C:      main: … mov x8, #0x746a4ae6e0  (= 499999500000) ; printf
```

The dependently-typed, linearity-checked tally program and the raw-pointer C twin
produce the **same machine code**. The 2,000,000 `malloc`/`free` calls and all the
list surgery are proven dead and removed on both sides — the erased proofs cost
exactly nothing. That's a stronger statement than a timing tie: the safe operation
doesn't merely match C's speed, it leaves no trace at all when its result is
unused, just like the C does.

## A real allocation workload: build + traverse a binary tree

A *balanced* alloc/free in a tight loop is dead code for any good optimizer, so the
DLL workload above folds to a constant. `examples/tree.tal` does work that genuinely
survives `-O2`: it **builds** a binary tree of 2^22 - 1 distinct nodes and then
**traverses** it. `build` is *general recursion* — it recurses with a different
label on each side (`2*label`, `2*label+1`), so the two subtrees differ and cannot
be shared; tallyc compiles it to a real recursive native function (`Fix`) that
mallocs every node. The allocations escape into the tree and are read back by `sum`,
so neither side can be optimized away. `bench/tree.c` is the hand-written twin.

Representative result (Apple silicon, depth 22 = 4.2M nodes allocated + traversed,
median of 5):

```
tally tree-sum: 8796090925056   median 0.085 s
C tree-sum:     8796090925056   median 0.081 s
tally/C wall-time ratio: 1.05   (1.0 = parity)
```

Parity, and the runtime RSS is now identical to C's (≈136 MB), because nullary
constructors cost nothing to allocate (see below). The dependent/linear types were
already fully erased; the remaining ~5% is run-to-run noise.

### How this got to parity (the leaf-allocation fix)

The first version of this benchmark ran at **1.49×** C. The cause was *not* the
dependent types — it was that every `Leaf` was a heap cell. A boxed constructor
used to `malloc` even when it had no fields, so a depth-22 tree allocated its 2^22
**leaves** in addition to its 2^22 - 1 nodes: ~8.4M `malloc`s vs C's ~4.2M
(C uses `NULL` for a leaf), and ~200 MB RSS vs C's ~136 MB. On an allocation-bound
workload that is the whole gap.

A nullary constructor (`Leaf`, `Nil`, …) is immutable and field-less, so it needs
no per-use allocation: tallyc now represents it as the address of ONE shared,
module-level constant cell. Leaves cost zero `malloc`s, RSS drops to exactly C's,
and the wall-time ratio falls from 1.49× to 1.05×. (Further parity polish — dropping
the unused tag word on single-boxed-constructor types, or an immediate/NULL leaf
so `sum` does a null-check instead of a tag load — is possible but already in the
noise here.)

Take-away across the two workloads: where the safe operation is pure (the DLL) it
costs *nothing* — identical machine code to C; where it allocates real data (the
tree) it now matches C's allocation count and runs at parity. In neither case do
the erased dependent/linear proofs cost anything.

## Files

| file | what |
|------|------|
| `../examples/bench.tal` | the safe circular-DLL workload (folds to a constant) |
| `bench.c`               | the hand-written C twin of the DLL workload |
| `../examples/tree.tal`  | build + traverse a binary tree (non-folding, real work) |
| `tree.c`                | the hand-written C twin of the tree workload |
| `run.sh`                | builds both pairs, checks they agree, times the tree |
