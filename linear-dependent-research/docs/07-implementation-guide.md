# 07 — Implementing λ-Tally: a guide

This document is for someone who wants to **build** λ-Tally — a dependent +
linear (quantitative) type system for memory-safe low-level code — rather than
study the metatheory. It tells you what to build, in what order, which design
decisions are load-bearing, what is already proven (so you can lean on it), and
the concrete pitfalls we hit. It is written against the artifacts in this repo;
file references are to things you can run today.

If you read nothing else, read §1 (the thesis), §3 (the rig — get it right
first), §6 (the soundness obligation you must not break), and §9 (pitfalls).

---

## 1. What you are building, in one paragraph

A single calculus with **two axes that are usually separate**: types may depend
on terms (Π/Σ, a universe), and every variable carries a **multiplicity** drawn
from a rig `R = {0, 1, ω}` that says how many times it may be used (`0` =
erased/compile-time-only, `1` = exactly once/linear, `ω` = unrestricted). The
linear fragment is what buys you **memory safety without a garbage collector**:
a heap capability has type `Cp` and multiplicity `1`, so the type system forces
each allocation to be freed exactly once — no double-free, no use-after-free, no
leak. The headline result (proved, see §2) is **well-typed ⟹ memory-safe**.

The core bet: *the pointer/owner split of L3-style memory languages is just two
multiplicities in a dependent context.* An unrestricted pointer is `ω`; a linear
capability/view is `1`; erased ghost/proof state is `0`. You do not need a
separate ownership system bolted onto types — the multiplicity rig **is** the
ownership system.

---

## 2. What is already proved (your foundation)

Do not re-derive these. They are machine-checked Agda (`agda/`, checks with
`LC_ALL=C.UTF-8 agda <File>.agda`, tested on Agda 2.6.3, no stdlib):

| Result | File | What it gives you |
|---|---|---|
| Rig laws + usage order `⊑` | `Rig.agda` | the semiring is associative/commutative/distributive; `x ⊑ 1# ⟹ x ≡ 1#` (exactly-once) and `x ⊑ 0# ⟹ x ≡ 0#` (erasure) |
| Contexts as a module over the rig | `Context.agda` | `+ᶜ`, `π ·ᶜ`, `𝟘` and their laws — the algebra every context-splitting rule uses |
| Type safety of the **linear core** | `Syntax/Semantics/Progress/Renaming/Substitution/Preservation.agda` | progress + preservation for the simply-typed quantitative calculus; the quantitative substitution lemma is the crux |
| **Dependent** calculus inhabited | `Dependent.agda` | Π/Σ with multiplicities, a universe, the 0-fragment (type formation at multiplicity 0); `dep-id`, `dep-sigma` check |
| **Confluence** (diamond) | `Confluence.agda` | parallel reduction + Takahashi complete development; the new ingredient dependent conversion needs |
| **Memory safety** of a heap machine | `MemSafety.agda` | a store-typing invariant that is *inductive* (preservation) and *implies safe access*; unbounded, no SMT |
| **well-typed ⟹ memory-safe** | `CombinedSound.agda` | the combine: a closed well-typed program runs with `rok ≡ true` (no double-free/UAF) and **no leak**; no postulates |

Bounded/automated evidence (faster to iterate against while prototyping) lives in
`prototype/*.rkt` (Rosette: bounded-exhaustive memory safety, resource
soundness) and `prototype/*.py` (a runnable QTT checker and an L3-style memory
model). The assurance ladder and what each rung buys you is `docs/05-soundness.md`.

The honest gaps (see §8): full **dependent** type safety (Church–Rosser strip
lemma → Π-injectivity → dependent substitution → preservation) is in progress;
soundness of the fully general capability-**moving** `let` is sketched but not
mechanized (the proved version uses an allocate-bind form, `nu`, that sidesteps
environment aliasing — §6).

---

## 3. Get the rig right first (it is load-bearing)

Everything keys off `R = {0, 1, ω}` with these operations (`Rig.agda`):

```
+ : 0+y=y, 1+1=ω, 1+ω=ω, ω+_=ω      (addition: "uses accumulate")
* : 0*_=0, 1*y=y, ω*1=ω, ω*ω=ω      (scaling: "uses under a binder")
⊑ : reflexive, and x ⊑ ω only       (usage order)
```

The single most important fact, and the one that makes the whole memory-safety
argument go through:

> `⊑` makes `0` and `1` **incomparable**. The only thing `⊑ 1` is `1` itself.
> Therefore `1 + 1 = ω ⋢ 1`.

When a binder declares a variable linear (budget `1`), the rule checks the
variable's *computed* usage `σ ⊑ 1`, which forces `σ ≡ 1` — used **exactly**
once, not at most once. A capability used twice computes to `ω`, which fails
`⊑ 1`, so the double-free is **rejected at type-check time**. This is not a
side condition; it is the mechanism. If you implement `⊑` as a numeric `≤`
(so `0 ≤ 1`), you get *affine* (droppable) capabilities and you will leak.
Decide deliberately:

- **Strict linear** (`0 ⋢ 1`): exactly-once. Frees are mandatory; no leaks. This
  is what `CombinedSound.agda` proves.
- **Affine** (`0 ⊑ 1`): at-most-once. You permit dropping a capability, which
  means you either leak or need an implicit `free`-on-drop. If you want this,
  add a separate `≤1`/affine multiplicity rather than weakening `⊑`, so the rig
  stays a lattice you control.

**Contexts are a left module over the rig** (`Context.agda`). A usage context
`γ : Vec R n` records, per variable, how much the term used it. Two operations
do all the work:

- `γ +ᶜ δ` (pointwise `+`): combine the usages of two subterms evaluated in the
  same environment (function and argument; the two halves of a pair; the two
  statements of a sequence).
- `π ·ᶜ γ` (pointwise `*`): scale a subterm's usage by the multiplicity at which
  it is consumed (the argument of an application used `π` times).

Implement these and their laws (identity, commutativity, associativity,
distributivity) before anything else. Every typing rule below is phrased in
terms of them, and every metatheoretic step is an instance of a module law.

---

## 4. The typing judgment

Use the **leftover / usage-as-output** discipline rather than splitting contexts
guess-first. The judgment is

```
Φ ⊢[ γ ] t ⦂ A
```

read: in type context `Φ` (the variables' types), term `t` has type `A`, and
**`γ` is the per-variable usage it incurred**. `Φ` is an input; `γ` is computed
as an *output* of checking `t`. This is far easier to implement than the
"declaratively split the context into Γ₁, Γ₂" presentation, because you never
guess the split — you check each subterm, get its `γ`, and combine with `+ᶜ`.

Representative rules (linear core, `Syntax.agda`; memory version,
`Combined.agda` / `CombinedSound.agda`):

```
var:   Φ ⊢[ only i ] var i ⦂ Φ[i]          -- "only i" = 1 at i, 0 elsewhere
app:   Φ ⊢[ γ ] f ⦂ (π A → B)   Φ ⊢[ δ ] a ⦂ A
       ──────────────────────────────────────────
       Φ ⊢[ γ +ᶜ (π ·ᶜ δ) ] f a ⦂ B
lam:   Φ,A ⊢[ γ , σ ] b ⦂ B     σ ⊑ π        -- bound var used within budget π
       ──────────────────────────────────────
       Φ ⊢[ γ ] λb ⦂ (π A → B)
```

`only i` is the unit usage context (the `1`-at-`i` vector). The `σ ⊑ π` premise
is where linearity is enforced.

### Bidirectional algorithm

The declarative rules are not directly an algorithm (where does `γ` split? what
is the type of a redex?). Implement **bidirectional** type checking:

- `check : Φ → Tm → Ty → Maybe (Ctx)` — given an expected type, check, returning
  the usage context.
- `infer : Φ → Tm → Maybe (Ty × Ctx)` — synthesize a type and usage.

Both return the usage context as output. Combine sub-results with `+ᶜ` / `·ᶜ`.
At a binder with budget `π`, recurse into the body with the variable appended,
read off the head entry `σ` of the returned usage, and check `σ ⊑ π`; the usage
you propagate upward is the *tail* (the body's usage of the outer variables).
The Python `prototype/qtt_checker.py` is a working, readable reference for this
loop (it predates the dependent layer but the multiplicity bookkeeping is the
same).

For the **dependent** layer add a definitional-equality (conversion) check at the
`check`/`infer` boundary: when `infer` gives `A` and you expected `B`, succeed if
`A ≡βη B`. Conversion needs normalization; see §7.

### The 0-fragment (erasure) — do not skip this

Type *formation* and the typing of types happens at multiplicity `0`. When you
check that `A` is a well-formed type (or check the domain of a Π, or a type-level
argument), you do so in the **0-scaled** context, so anything used only in types
contributes `0` to runtime usage and is erased. `Dependent.agda` implements this
as a mode/flag on the judgment. Practically: a dependency in a *type* must not
make you think a value was used at runtime. Get this wrong and either erasure is
unsound (you erase something you needed) or linearity is too strict (a value
"used" only in a type annotation is reported as consumed).

**Erasure in the native backend, and how we keep it honest.** `tallyc`'s
dependent backend (`tallyc/src/dep_codegen.rs`, behind `--features llvm`) lowers
a checked `dep::Term` to LLVM and JIT-runs it. Erasure there is concrete: a
multiplicity-`0` constructor argument gets **no heap slot**, and a `Π[0]`
function argument is **never compiled** (the application β-reducer reads each
binder's multiplicity off the head's `Π` telescope and binds `None` for the
erased ones, so an index/proof never becomes an instruction). The reading-`None`
path is a hard error, so the kernel's guarantee — a checked term never reads an
erased variable at runtime — is enforced rather than assumed. This is checked
empirically by reading the emitted IR (`emit_ir`): in the test
`vec_ir_has_zero_overhead`, a length-indexed `Vec`'s `Cons` cell is `malloc(24)`
(tag + element + tail), *not* 32 bytes, and the fold over it is handed no length;
in `dll_ir_has_zero_overhead`, the ghost region/cursor identity appears nowhere
in the IR and `alloc`/`free` are direct libc calls. These are evidence for —
not a proof of — the erasure theorem T1.2 (`docs/04` Phase 1).

---

## 5. The memory model

Two routes; pick based on how much you want in the type system vs. as a library.

**Capabilities as linear values (recommended, this is what's proved).** Add a
type `Cp` (a capability/view to one live heap cell) that is *always* used
linearly, and primitives that are ordinary typed functions:

```
new   : Cp                       -- allocate, return the capability   (your alloc)
use/free : Cp → Un               -- consume the capability            (your free)
read  : Cp → (A × Cp)            -- read returns a fresh capability (the view is threaded)
write : A → Cp → Cp              -- strong update returns the updated view
```

The trick that makes reads/writes safe: a capability is **consumed and a new one
returned**, so you can never use a stale view — the old one is gone (linearity).
This is the L3 "the view is the proof the cell is live" idea; `read`/`write`
thread the view. `MemSafety.agda` models the full set (alloc/read/write/resize/
free/mkclaim/staleread) including **strong update** (resize changes the stored
length) and proves the store-typing invariant inductive.

**The operational picture you must preserve** (`Combined.agda`/`CombinedSound.agda`):
a heap is `(nxt, liv)` — a fresh-address counter and a liveness map. `new`
allocates at `nxt` (fresh, strictly increasing) and sets it live; `free` sets it
dead and the evaluator's `ok` flag goes false iff you free a dead/non-capability.
The theorem is that for well-typed programs `ok` stays true and the final
liveness map is empty.

---

## 6. The soundness obligation (do not break this)

If you change the type system, this is the property to preserve, and the proof
strategy tells you *which invariants your implementation's rules must maintain*.
The argument in `CombinedSound.agda` is a **separation/frame argument keyed on
the rig**:

1. The usage context `γ` of a subterm determines exactly which live heap cells
   it **owns** (the cells held by its `1`-usage capability variables). Function
   `ownedχ ρ γ` computes this owned set from the environment and usage.
2. **Linearity ⟹ disjointness.** When two subterms run in sequence with combined
   usage `γ +ᶜ δ`, no cell is owned by both: if it were, that variable's usage
   would be `1 + 1 = ω`, which `⊑ 1` rejects. (`Disj` in the proof.)
3. **The frame is preserved.** Because a subterm only frees cells it owns and
   allocates fresh cells (above `nxt`), the other subterm's owned cells are
   untouched. This is the separation-logic frame rule, discharged by freshness +
   disjointness rather than by a logic.
4. Therefore every `use`/`free` lands on a live cell (`onOwned`/`onFresh` of the
   `Spec` record), `ok` stays true, and a closed `Un` program with empty usage
   leaves no live cell.

What this means for your implementation: **owned sets must stay disjoint across
every context combination (`+ᶜ`), and allocation must be fresh.** Any rule you
add that lets two `1`-usage variables alias the same cell, or that reuses
addresses, breaks the proof — and the bug it permits is a real double-free.

**The one aliasing subtlety** (and the proved scope): a `let x = e₁ in e₂` whose
`e₁` is a bare capability variable *moves* a capability — now two environment
slots hold the same address, even though linearity keeps only one of them at
usage `1`. The mechanized proof avoids this by using an allocate-and-bind form
`nu e` (the bound capability is always *freshly allocated*, so environment
addresses are globally distinct and disjointness is free). The fully general
moving `let` is sound too, but the proof needs a *usage-relative* injectivity
invariant (distinct `1`-usage slots have distinct addresses, maintained because
the moved-from slot drops to usage `0`). This is the noted next step; if you
implement general `let`, carry that invariant. See the comment block in
`CombinedSound.agda` and the roadmap in `agda/README.md`.

---

## 7. Conversion, normalization, and the universe

The simply-typed core needs no conversion. The moment types depend on terms you
need to decide `A ≡ B` up to β (and usually η). Two practical points:

- **`Type : Type` is fine for a prototype, fatal for consistency.** The current
  `Dependent.agda` uses `⋆ : ⋆`, which makes the system logically inconsistent
  (every type inhabited) and non-normalizing in principle. It is perfectly
  usable for *type-safety* and for writing programs; it just is not a logic.
  When you want proofs to mean something, replace `⋆ : ⋆` with a universe
  hierarchy `⋆ᵢ : ⋆ᵢ₊₁`. Budget this; it touches every type-formation rule.
- **Implement conversion via normalization-by-evaluation (NbE)**, not naive
  rewriting. Confluence (`Confluence.agda`) guarantees normal forms are unique,
  so NbE is sound; but you still want it for speed and for closing under η. The
  metatheory you need to trust NbE — confluence's diamond property — is the part
  already done; Church–Rosser (the strip lemma) and Π-injectivity-up-to-conversion
  are the remaining standard boilerplate (roadmap).

Multiplicities and conversion interact only mildly: conversion compares the
*type* structure; multiplicity annotations on Π/Σ are part of that structure and
must match (or be related by the rig order if you allow subusaging). Keep it
simple first: require multiplicities to match exactly under conversion.

---

## 8. Suggested build order (staging)

Each stage is runnable and testable on its own. This mirrors how the artifacts
were built and de-risks the hard parts early.

1. **Rig + contexts.** `R`, `+`, `*`, `⊑`, `+ᶜ`, `·ᶜ`, `𝟘`, laws. Unit-test the
   laws. (≈ `Rig.agda` + `Context.agda`.)
2. **Simply-typed linear core + bidirectional checker.** `var`/`lam`/`app`,
   usage-as-output, the `σ ⊑ π` check. Worked examples: a linear identity, a
   `K` combinator with an *erased* (`0`) argument. (≈ `qtt_checker.py`,
   `Syntax.agda`.)
3. **Operational semantics + type safety tests.** CBV evaluator; property-test
   progress/preservation (Redex-style random, or Rosette bounded-exhaustive
   before you attempt a proof). (≈ `Semantics/Progress/Preservation.agda`,
   `memory-model.rkt`.)
4. **Memory primitives as linear functions.** `Cp`, `new`, `free`, threaded
   `read`/`write`. Get the *closed* soundness theorem (well-typed ⟹ no
   double-free/UAF, no leak) — bounded first (Rosette `resource-soundness-
   rosette.rkt`), then the unbounded version. (≈ `Combined.agda` →
   `CombinedSound.agda`.) **This is the milestone that proves the idea works.**
5. **Dependent layer.** Π/Σ with multiplicities, universe, 0-fragment, NbE
   conversion. Re-establish type safety over the dependent syntax. (≈
   `Dependent.agda`, `DepSubst.agda`, `Confluence.agda`; remaining: dependent
   substitution lemma + preservation.)
6. **Fold memory into the dependent kernel** — the full λ-Tally: capabilities
   indexed by length/permission, dependent strong update, etc.
7. **Surface concerns:** inference/elaboration of multiplicities (so users don't
   annotate every `1`), borrows/shared-read (an extra multiplicity or a fractional
   permission), error messages that explain *which* use made a count exceed `1`.

Milestone 4 is the make-or-break; if it works (it does, for the `nu` fragment),
the design is sound. Everything after is engineering and standard dependent-type
metatheory.

---

## 9. Pitfalls we actually hit (read this before you start)

- **`⊑` as numeric `≤`.** Makes `0 ⊑ 1`, silently turning linear into affine.
  Leaks. Keep `0` and `1` incomparable (§3).
- **Forgetting the 0-fragment.** Type-level uses must not count as runtime uses.
  Check types in the `0`-scaled context (§4).
- **Context splitting by guessing.** Don't. Use usage-as-output and `+ᶜ`/`·ᶜ`
  (§4). The "leftover typing" framing collapses a class of bugs.
- **Aliasing via capability-moving `let`.** Breaks the disjointness the safety
  proof relies on unless you carry usage-relative injectivity (§6). Start with
  allocate-and-bind (`nu`) where addresses are fresh by construction.
- **Reusing heap addresses.** Freshness (strictly increasing `nxt`) is what makes
  a new allocation provably distinct from every owned cell. A free-list that
  recycles addresses needs a generation/tag or you reintroduce ABA-style
  use-after-free that the simple proof can't see.
- **`Type : Type` shipped as a logic.** Fine for type safety, unsound as a proof
  assistant. Plan the universe hierarchy (§7).
- **Naive conversion that loops.** Without normalization you can diverge on
  ill-typed or (with `Type:Type`) even well-typed input. Use NbE; rely on
  confluence for uniqueness.
- **Mechanization ergonomics (if you verify in Agda):** match on the *term* in
  the `var` case so the `App` index unifies; case-split guard booleans so
  operations reduce (`MemSafety.agda` step-helpers); functions stuck on an
  abstract `Val`/`Bool` won't reduce — case the scrutinee or use `with … in eq`;
  declare a fixity for your tuple constructor `_,_` or `a , b , c` won't parse.
  These cost hours each; they're documented inline in the Agda files.

---

## 10. Artifact map

| You want… | Look at |
|---|---|
| the design goals and the hard tensions | `docs/00-design-goals.md` |
| the QTT core on paper | `docs/01-qtt-core.md` |
| the pointer/view = two-multiplicities thesis | `docs/02-memory-views.md` |
| a runnable checker to copy | `prototype/qtt_checker.py` |
| the native backend (checked `dep::Term` → LLVM, erasing) | `tallyc/src/dep_codegen.rs` (`tally run <file>`, `--features llvm`) |
| a runnable memory model | `prototype/lambda_tally_memory.py`, `prototype/memory-model.rkt` |
| bounded verification harnesses | `prototype/*-rosette.rkt` |
| the proved metatheory + module table | `agda/` and `agda/README.md` |
| how certain we are, rung by rung | `docs/05-soundness.md`, `docs/06-rosette-metatheory.md` |
| **the proof you must not break** | `agda/CombinedSound.agda` |

The short version: build the rig, then the linear core with usage-as-output,
then memory primitives as linear functions, and check the closed-program safety
property at every step. The multiplicities are not decoration — they are the
ownership discipline, and `1 + 1 = ω ⋢ 1` is the whole game.
