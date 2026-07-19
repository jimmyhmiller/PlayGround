# The differentiator demo — a safe manual-memory linked structure in surface Tally

**STATUS: ✅ the TYPE-LEVEL demo is DONE + RUNS.** The linear-dependent type system
enforces manual-memory obligations soundly, end-to-end, executing natively with ZERO GC.
This is the milestone the whole foundation (gate-2 + CBV-let + (a) variance-aware
positivity + (b) eliminator-join, all independently signed off) was built for: it proves
the thesis — Idris-power dependent types + C-level manual memory + total safety, in ONE
language.

## What runs

A 2-node OWNED linked list `[1, 2]`, built on the heap, traversed, freed, and summed —
type-checked memory-safe and running natively to `3`:

```
%builtin Nat Nat
enum Nat { Zero : Nat, Succ : Nat -> Nat }
enum Opt (a : Type) { none : Opt a, some : a -> Opt a }
struct Node { head : Nat, tail : Opt (Own Node) }     -- recursion via the (a)-verified positivity

add : Nat -> Nat -> Nat
fn add(m, n) { match m { Zero => n, Succ(k) => Succ(add(k, n)) } }

main : Nat
fn main() {
  match unbox(alloc(Node(Succ(Zero), some(alloc(Node(Succ(Succ(Zero)), none)))))) {
    Node(h1, t1) => match t1 {
      none     => Zero,                                   -- dead (t1 is some)
      some(o2) => match unbox(o2) { Node(h2, t2) => match t2 {
        none     => add(h1, h2),                          -- LIVE: 1 + 2 = 3
        some(o3) => match free(o3) { U => Zero }          -- dead; still must free o3
      } }
    }
  }
}
```

- **box** = `alloc : {0 a} -> a -> Own a` — allocate a cell, store the payload.
- **unbox** = `{0 a} -> (1 o : Own a) -> a` — deref-and-CONSUME: load the pointee, FREE
  the cell, MOVE the contents out. The sound primitive for a consuming free-traversal of
  a linear-FIELD structure (a non-consuming read-back would DUPLICATE the linear `tail` —
  that needs the Phase-C view-unfold, deferred). The linear `o` is consumed exactly once;
  its linear `tail` is moved to the caller and threaded onward by the eliminator (bound at
  `1` via gate-2's use-site rebind, freed once down the chain).
- **free** = `{0 a} -> (1 o : Own a) -> Unit` — free a cell whose contents need no extraction.

## Why it proves the thesis

Every `Own` is consumed EXACTLY ONCE on EVERY path — and that obligation is the safety
guarantee, enforced by the verified type system, NOT by a GC or a runtime check:

- DROP an owned binder (e.g. the dead `some(o3)` arm returns `Zero` without `free(o3)`) ⇒
  LEAK ⇒ `0 ⋢ 1` ⇒ REJECTED.
- USE an owned binder twice (e.g. `unbox(o2)` then `free(o2)`) ⇒ DOUBLE-FREE ⇒ `ω ⋢ 1` ⇒
  REJECTED.
- The dead match arms must themselves discharge their linear binders (free their owns) —
  the eliminator-JOIN (b) requires every arm to consume the same resources.

Tests: `differentiator_demo_owned_linked_list_runs_natively` (dep_codegen, native run = 3);
`differentiator_demo_linked_list_is_type_safe_and_red_teamed` (rust_surface: safe accepts,
leak rejected, double-free rejected).

## The two completions (immediate next, both real — not optional)

1. **Phase B — inline value layout (byte-level zero-leak).** The goal is "as low as C",
   which means the bytes ARE reclaimed. Today the all-boxed codegen DOUBLE-boxes
   `Own`-of-struct (an `Own` cell pointing at a separately-boxed `Node` struct cell);
   `unbox` frees the `Own` cell, but the inner struct cell is not freed by the current
   backend — a backend-level leak (the TYPE level is fully safe; the codegen isn't yet
   leak-free). Phase B stores the struct INLINE in the `Own` cell (one box ⇒ `unbox`
   reclaims everything), making the run actually-as-low-as-C. This is the SECOND half of
   the demo: the type-level safety (the hard, novel part) is proven; Phase B is the
   codegen completion.

2. **Recursive traversal of arbitrary-length structures — the linear-accumulator fold.**
   A fuel-driven `sumFree(fuel, l)` (the natural arbitrary-length traversal) is a 1a′
   accumulator fold whose ACCUMULATOR is LINEAR (`l : Opt (Own Node)`). 1a′ v1 builds the
   accumulator motive/IH at multiplicity `ω`, so a linear accumulator fails
   (`the let-bound variable at multiplicity 1 is used ω times`). Extending the 1a′
   accumulator lowering to carry the accumulator's actual quantity (`1` for a linear acc,
   threaded once through the IH) unlocks recursive build/traverse/free of lists/trees of
   any length. (Alternatively, well-founded recursion via 1b once `natWf` lands.) The
   straight-line demo above sidesteps this by unrolling a fixed length.
