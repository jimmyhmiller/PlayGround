# 00 — Design goals, non-goals, and the hard tensions

This document fixes *what we are trying to build* and, more importantly, names the
tensions between the goals and states the bet we are making to resolve each one.
The later documents cash out those bets formally.

## 1. Goals

- **G1 — Full dependent types.** Types may depend on terms. We want Π/Σ, equality,
  inductive families, and the ability to write specifications and proofs.
- **G2 — Substructural (linear/affine) types.** Every value has a *multiplicity*
  controlling how often it may be used. At minimum we want `0` (erased), `1`
  (linear, exactly once), `ω` (unrestricted). Likely also `≤1` (affine, droppable)
  and some notion of *borrowed/shared-read*.
- **G3 — C-level expressiveness.** Raw pointers, pointer arithmetic, manual
  `malloc`/`free`, in-place/destructive update, explicit data layout, stack vs.
  heap, no mandatory garbage collector, predictable performance.
- **G4 — Total memory & resource safety.** No use-after-free, double-free,
  out-of-bounds, uninitialized reads, null derefs, leaks of linear resources, or
  data races — *enforced statically*, with zero runtime tax for the safety that
  was proven at compile time.
- **G5 — Programmable type system (the "Shen" goal).** Typing rules are data.
  The checker is, in part, a logic/constraint engine. Users can introduce new
  judgments and rules (e.g. a session-typing discipline, a units-of-measure
  system, a custom effect system) without forking the compiler.

## 2. Non-goals (for now)

- A surface syntax. We work at the level of core calculi and judgments.
- An implementation, optimizer, or backend.
- Automatic memory management. GC may be offered as a *library* (a region with
  `ω` semantics), never imposed.
- Decidable, push-button proof automation for the *full* logic. We accept that
  proof obligations may need user assistance; we only insist the *checking* of a
  finished proof is decidable (see Tension T3).

## 3. The hard tensions

### T1 — "Used in a type" vs. "used once"

If a value `x` is **linear** (must be used exactly once) but also appears in a
**type** (e.g. `Vec n A`, where `n` is a value), hasn't it been "used" by the
type? Naively, dependent + linear seems contradictory.

**Bet:** Quantitative Type Theory. Annotate every variable use with a multiplicity
from a semiring, and make *type-level* use count as multiplicity **`0`**. A
`0`-use consumes nothing, so `x` may appear in types freely and *still* be spent
linearly at runtime. The same `0` doubles as the **erasure** marker: anything that
is only ever used at `0` has no runtime representation. This is the keystone — it
makes dependent typing both *compatible with* linearity and *free at runtime*.
Formalized in `docs/01-qtt-core.md`.

### T2 — C-level memory vs. memory safety

Manual `free`, strong (type-changing) in-place update, and raw pointers are
exactly the operations that make C unsafe. We want them anyway.

**Bet:** Split **aliasing** from **permission** (L3 / ATS views).

- A **pointer** `Ptr ℓ` names a static location `ℓ`. Pointers are ordinary
  `ω`-values: copy them freely, store them anywhere, alias to your heart's content.
- A **view / capability** `A @ ℓ` is a *linear* proposition asserting "a value of
  type `A` currently lives at `ℓ`". You cannot dereference `Ptr ℓ` without
  presenting `A @ ℓ`.

Because the view is linear, at most one party holds the right to access `ℓ` at a
time, which makes the dangerous operations sound:

- **Strong update** `write : Ptr ℓ → (A @ ℓ) → B → (B @ ℓ)` changes the *type*
  stored at `ℓ`. Safe because no stale `A @ ℓ` can survive — there was only one.
- **Free** `free : Ptr ℓ → (A @ ℓ) → 1` *consumes* the view. The pointer value may
  still exist and be copied, but it is now inert: with no `_ @ ℓ` in scope, it can
  never be dereferenced again. Use-after-free and double-free become
  *untypeable*, not *unchecked-at-runtime*. Formalized in `docs/02-memory-views.md`.

### T3 — Full dependency vs. decidable checking

Unrestricted dependent type checking requires deciding term equality, which is
undecidable in general (and even where decidable, can be wildly expensive). C-like
code should not pay theorem-proving costs to compile.

**Bet:** *Stratify*, ATS-style.

- A **static** level: a small, deliberately **decidable** index/constraint
  language (e.g. linear arithmetic over integers and locations, finite maps of
  views) discharged by a decision procedure. Sizes, bounds, location aliasing,
  and the like live here and check fast.
- A **dynamic** level: the actual programs. Dependency on *static* indices is the
  common case and stays decidable.
- Full term-level proofs are available but *explicit*: you write the proof term,
  and *checking* it is decidable even if *finding* it was not. We never ask the
  compiler to search for proofs in the undecidable fragment without a budget.

This is also where the **Shen** goal lands: the rule layer above the decidable
core is programmable (Horn clauses over judgments), with an explicit
fuel/termination story so "programmable" never means "the compiler might loop."
Formalized in `docs/03-programmable-checker.md`.

### T4 — Linearity vs. ergonomics

Pure linearity is painful: you cannot even read a value twice. Rust is usable
because of *borrowing* — temporary, scoped, non-consuming access.

**Bet:** Enrich the semiring and/or add a borrowing judgment. Options under
consideration (see `docs/01` §6 and `docs/02` §5):

- A richer rig with **fractional permissions** (Boyland) for shared read access:
  a view `A @ ℓ` at fraction `q ∈ (0,1]` permits reads; only `q = 1` permits
  writes; fractions split and recombine.
- A **borrow** as a second-class, region-scoped capability that is *returned* at
  end of scope (Rust's `&`/`&mut` reconstructed as a view that must be given back).

We will likely want both: fractions for the logic, borrows for the surface
ergonomics. This tension is the least settled and is flagged throughout.

### T5 — Programmable rules vs. soundness

If users can add typing rules, they can add *unsound* ones.

**Bet:** A two-tier rule system. The **kernel** rules (QTT + the view discipline)
are fixed and are what the soundness proof is about. User rules are
**conservative extensions** that must *elaborate down to* kernel derivations — a
user rule is sugar/automation that produces a core derivation the kernel
re-checks. Think "tactics that must produce a checkable proof term," not "new
axioms." So the programmable layer can be arbitrarily clever (or buggy) without
threatening safety: a bad user rule fails to elaborate; it cannot forge a kernel
derivation. Detailed in `docs/03-programmable-checker.md` §4.

## 4. Success criteria for the *research* (not the language)

We will consider the formal core "good enough to start building" when we have:

1. A core calculus **λ-Tally** with: rig-annotated dependent contexts, the 0/1
   modal judgment, Π/Σ, and the view/location primitives.
2. A **type-erasure** statement: the `0`-fragment compiles away; runtime terms
   are the multiplicity-`>0` skeleton.
3. A **type-safety** statement (progress + preservation) for the dynamic
   fragment, *including* the memory primitives, ideally via a syntactic
   (logical-relations or progress/preservation) argument.
4. A **memory-safety** corollary: well-typed programs never deref a freed/strongly
   -updated-away location; linear resources are neither duplicated nor dropped.
5. A precise **stratification + elaboration** story making checking decidable for
   the static fragment and the kernel, with user rules as elaborators.

`docs/04-roadmap.md` turns these into concrete proof obligations and an order to
attack them.
