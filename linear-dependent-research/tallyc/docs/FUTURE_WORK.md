# tallyc — Future Work / Vision

*A language as low-level as C, with complete and explicit control over allocation,
that is nonetheless 100% memory-safe, expresses dependent types as freely as Idris,
and lets you carve out a **total** subset whose programs are provably terminating
and total.*

This is a **future-work document**: it describes the language tallyc is aiming to
become, not what exists today. It is a north star to steer toward. §12 is the
roadmap from the current implementation; §13 is the honest list of what is unsolved.
Everything here is a *proposal* — subject to change as the hard parts (especially
the view layer and the totality checker) are actually built.

---

## 0. Thesis

There are three traditions that have never been fully unified in one language:

1. **C / Zig** — total control over memory layout and allocation, zero overhead,
   no runtime. But no safety: use-after-free, out-of-bounds, data races, leaks,
   uninitialized reads are all on you.
2. **Idris / Agda / Lean** — full dependent types; you can state and prove
   arbitrary properties, and a *total* program is a genuine mathematical proof.
   But the memory model is a garbage collector you don't control.
3. **Rust / ATS** — ownership types make manual memory safe. Rust stops short of
   dependent types; ATS has them but in a constrained, separate proof layer and is
   famously hard to use.

The bet of tallyc is that **one mechanism — Quantitative Type Theory (QTT) — is
enough to fuse all three.** A single per-variable *quantity* (`0`, `1`, or `ω`)
simultaneously gives you:

- **erasure** (`0`): proofs, indices, and types vanish at runtime — dependent
  types cost nothing;
- **linearity / ownership** (`1`): a value used exactly once — the basis of
  safe manual `malloc`/`free`, in-place mutation, and data-race freedom;
- **ordinary values** (`ω`): copy freely.

On top of QTT, layout is explicit (ADTs are values you place, à la Zig), and a
**totality discipline** stratifies the language into a terminating, total core
(usable in types and proofs) and a general-recursion layer (runtime glue). You get
C's machine model, Rust's safety, Idris's expressiveness, and a checkable
"this fragment is total" guarantee — with no garbage collector, ever.

---

## 1. Goals and non-goals

### Goals

- **G1 — C-level performance and control.** Explicit allocation, explicit layout,
  stack/inline/heap/arena placement, no hidden allocation, no runtime, no GC.
  Emitted code is competitive with hand-written C, instruction for instruction.
- **G2 — Total memory safety, statically.** Outside explicitly `unsafe` blocks, a
  well-typed program cannot: use-after-free, double-free, leak a linear resource,
  read uninitialized memory, index out of bounds, type-confuse memory, or race on
  shared mutable state.
- **G3 — Full dependent types.** Π, Σ, indexed inductive families, propositional
  equality, dependent pattern matching, universe polymorphism — Idris-level
  expressiveness. You can index data by values and prove properties about programs.
- **G4 — A certified-total subset.** A function (or module) can be checked
  `total`: it provably terminates, covers all cases, and lives over strictly
  positive data. Total functions are mathematical functions: usable in types, in
  proofs, and trustworthy as specifications.
- **G5 — Zero-cost abstraction, proven.** Everything in G2–G4 that is `0`-quantity
  is erased; the proof obligation is discharged at compile time and the runtime is
  pure data movement. This is checkable in the emitted IR.
- **G6 — A small trusted core.** Elaboration is untrusted; a small kernel
  re-type-checks the fully elaborated term. Bugs in inference cannot produce an
  unsound program.

### Non-goals

- **Not a proof assistant first.** Totality and proofs are *available and
  checkable*, not *mandatory*. You can write a partial `main` that loops forever
  and does I/O. The total fragment is opt-in (and is what types are built from).
- **Not garbage-collected, ever.** There is no tracing GC and no implicit
  reference counting. Reclamation is the linear type system's job.
- **Not automatically verified.** tallyc proves *type and memory safety* and
  *whatever you state in types*. It is not an SMT-backed deductive verifier like
  F\*; there is no solver in the trusted loop. Decidable checking is preferred to
  automation.

---

## 2. The core idea: one quantity system

Every binder carries a **quantity** π drawn from the semiring `{0, 1, ω}`:

| π   | meaning            | runtime                          | example                     |
|-----|--------------------|----------------------------------|-----------------------------|
| `0` | erased / static    | **none** — gone at compile time  | a length index, a proof, a type |
| `1` | linear / owned     | a value used **exactly once**    | an `Own T`, a mutation token |
| `ω` | unrestricted       | an ordinary copyable value       | an `Int`, a shared `&T`      |

Usage is tracked with the QTT accounting rules: using a variable in two branches
*adds* its usages (`1 + 1 = ω`, which exceeds `1` ⇒ a linearity error); passing a
variable to a `π`-binder *scales* the usage by `π` (so a `0`-argument's body usage
is multiplied by `0` — it is never actually demanded). The type checker computes,
for each binder, the quantity at which it was used, and rejects the program if that
exceeds the declared quantity.

This one mechanism is the whole spine:

- **`0` ⇒ erasure ⇒ dependent types are free.** An index `{0 n : Nat}` is solved
  by unification and never compiled. (tallyc already does this and proves it in the
  emitted IR.)
- **`1` ⇒ ownership ⇒ safe manual memory.** An `Own T` taken at `1` must be
  consumed exactly once. Dropping it is a *leak* (`0 ⋢ 1`); using it twice is a
  *use-after-free* (`ω ⋢ 1`). `free` is "the function that consumes the `1`."
- **`ω` ⇒ normal programming.** Plain data is `ω` and behaves like any language.

The same `Own a → Unit` "consume exactly once" type is `free`, an arena handoff, a
move, or a destructor — they differ only in what they *do*, not in how the checker
accounts for them.

---

## 3. The type system

### 3.1 Universes

`Type 0 : Type 1 : Type 2 : …`, cumulative, with universe polymorphism and
inference (Idris-style). A genuine hierarchy is required for the totality guarantee
to mean anything: `Type : Type` (which the current prototype has) makes the logic
inconsistent via Girard's paradox, so a *total* program could "prove" `False`. The
hierarchy is the price of G4.

### 3.2 The dependent core

- **Π types** `(x : A) → B x` with a quantity: `(π x : A) → B`.
- **Σ types** `(x : A) × B x`, including linear pairs `(1 x : A) × B` (the basis of
  "an address *and* its ownership token").
- **Inductive families** indexed by values:
  `Vec : (A : Type) → Nat → Type`, `Fin : Nat → Type`, etc. Strictly positive
  (§4.3). Each family generates a dependent eliminator; surface `match` lowers to
  it.
- **Propositional equality** `x = y`, with `refl`, `J`, and `subst`/transport. The
  engine of "the types say this program is correct."
- **Dependent pattern matching** with coverage checking (§4.2), `with`-abstraction,
  and absurd patterns (impossible cases discharged by constructor disjointness).

### 3.3 Definitional equality

Type-checking compares types up to βδιη-normalization (normalization by
evaluation — already the kernel's design). For this to be **decidable**, every
computation that can appear in a type must terminate — which is exactly why
type-relevant code must be total (§4).

---

## 4. Totality and the total subset

This is the feature that makes "we can know our program is total" real.

### 4.1 What total means

A function is **total** when it is:

1. **Covering** — every well-typed input matches some clause (no missing cases);
2. **Terminating** — it returns on every input (no infinite recursion); and
3. **Positive** — defined over strictly positive data (§4.3), so its induction is
   sound.

A total function is a *mathematical function* `A → B`: it always yields a value,
and `f x` denotes that value. Such functions can be used inside types, normalized
during checking, and trusted as specifications.

### 4.2 The checker

- **Coverage** is decided by the dependent pattern-match compiler: split the scrutinee
  on its constructors, recursively cover each, and discharge impossible index
  combinations as absurd. A `partial` annotation opts out.
- **Termination** is accepted via, in increasing power:
  - *structural recursion* — recursive calls on a syntactic sub-component of a
    matched argument (the common case; tallyc's "simple fold" detector is a
    primitive version of this);
  - *lexicographic / multi-argument* descent;
  - *well-founded recursion* — a recursive call justified by an `Acc`-style
    accessibility proof on a well-founded relation (lets you write, e.g.,
    quicksort or gcd and still be total);
  - optionally *sized types* for higher-order/coinductive cases.
- **Positivity** (§4.3) is a syntactic check on datatype declarations.

The compiler reports the totality status of every definition; `%total` on a
definition or module turns "not total" into an error, so you can *demand* a
certificate for a subset.

### 4.3 Strict positivity

An inductive family may only recurse in *strictly positive* positions — never to
the left of an arrow. `data Bad = MkBad (Bad → Bad)` is rejected, because it would
let you build a non-terminating term in the total fragment and break soundness.
(tallyc's current occurrence check is the seed of this.)

### 4.4 The total / partial boundary, and why it is exactly the type/value boundary

The key design move — already prototyped — is that **the total fragment is
precisely what may appear in types.**

- **Total code** (structural/well-founded recursion, eliminators) *reduces* during
  type-checking. It can index data, appear in equalities, and serve as a spec.
- **General recursion** is provided by an explicit `Fix` (a partial fixpoint). The
  kernel treats `Fix` **opaquely**: it never unfolds it during type-checking. So a
  potentially-non-terminating function cannot be forced by the checker, which keeps
  type-checking decidable *even though the language is Turing-complete at the value
  level*.

The consequence is a clean stratification with no separate proof language:

```
            may appear in types          runtime only
            (TOTAL, reduces)             (PARTIAL, opaque to checker)
            ────────────────             ───────────────────────────
            eliminators / folds          Fix (general recursion)
            structural recursion         I/O, FFI
            well-founded recursion        unbounded loops
            proofs, indices              effectful glue
```

You write your data invariants, decreasing measures, and proofs in the total
fragment; you write your event loop and allocator plumbing in the partial
fragment; and the boundary is enforced by "does it reduce in a type." `assert_total`
and `believe_me`-style escape hatches exist but are tracked and surfaced.

### 4.5 What this buys

- A library can export a `total` certificate: its callers know those functions
  terminate and cover all cases.
- Proof-carrying code: a value of type `Sorted xs` is a *proof* that `xs` is
  sorted, erased at runtime — the dependent payoff, available but never forced.
- You can choose your point on the spectrum per module: a parser kernel can be
  total; the I/O driver around it need not be.

---

## 5. The memory model — as low as C, fully manual, 100% safe

The departure from ML/Idris: **constructors do not allocate.** Constructing an ADT
makes a *value* with a known layout; *placing* that value in memory is a separate,
explicit step you control. This is the Zig/C model, made safe by quantities.

### 5.1 ADTs are values with layout

A `struct`/`enum` has a C-compatible layout: size, alignment, field offsets, and
for sums a tag plus a union of payloads. A value lives wherever you put it — a
register, a stack local, a field of another value, a slot in an array — with **no
allocation**. Small values pass in registers; large ones pass by (compiler-managed)
pointer. Layout is observable and controllable (`#[repr]`-style attributes, niche
optimizations, explicit tag types).

```
struct Point { x : I64, y : I64 }          -- a flat 16-byte value, never boxed
enum Shape { Circle(I64) | Rect(I64, I64) } -- tag + union, flat
```

### 5.2 Recursion names its own indirection

A recursive type cannot be inline-infinite, so — as in C — you write the
indirection explicitly. There is no implicit box:

```
struct Node { value : I64, left : Opt (Own Node), right : Opt (Own Node) }
```

`Own Node` is a **linear owned pointer** to a heap `Node`; `Opt (Own Node)` is a
nullable owned pointer (the niche/null-pointer optimization makes `none` literally
the null pointer — a leaf costs nothing). This is exactly `struct node { long
value; node *left, *right; }` with `NULL` leaves, but the `Own` makes ownership
linear and checked.

### 5.3 Allocation is explicit and pluggable

```
fn build(a : Allocator, d : I64, label : I64) -> Opt (Own Node @ a) {
    match d {
        0 => none,
        _ => some(box(a, Node {                 -- `box` = the only thing that allocates
                 value: label,
                 left:  build(a, d-1, label+label),
                 right: build(a, d-1, label+label+1),
             })),
    }
}
```

- `box(a, v)` allocates `sizeof v` from allocator `a`, moves `v` in, returns a
  linear `Own` whose lifetime is tied to `a` by the region index `@ a`.
- **Allocators are first-class** (Zig-style): `malloc`/`free`, a **bump/arena**
  (free the whole region at once — often *faster than C*), a fixed pool, or inline
  stack storage. The region index in the type prevents a pointer from escaping its
  allocator; freeing the region is sound because no live pointer into it can
  survive. (tallyc already has region indices in its linear DLL.)
- `free(o)` consumes an `Own`; for arena-allocated values, the region's bulk free
  consumes them all at once and the checker proves none are used afterward.

### 5.4 Borrowing: shared and unique references

Threading an owned token through every read is verbose. Borrows recover ergonomics
without copying:

- `&T` — a **shared**, read-only borrow (`ω`: copyable), valid for a lexical/region
  scope. Many `&T` may coexist; no mutation through them.
- `&mut T` — a **unique**, mutable borrow (`1`: linear), exclusive for its scope;
  in-place mutation through it is safe because uniqueness rules out aliasing.

Borrows are modeled as scoped (fractional or region-indexed) **views** that are
returned when the scope ends — the lender regains full ownership. This is Rust's
discipline expressed in the same QTT + region machinery, not a separate borrow
checker bolted on.

### 5.5 Views and separation logic — in-place mutation and linked structures

Underneath ownership is a **view** layer (the ATS/Low\*/Steel idea):

- A **points-to view** `p ↦ v` is a linear, *erased* proposition: "address `p`
  currently holds `v`." Owning it grants read/write to that cell.
- `Own T = ∃ (p : Ptr). p ↦ (v : T)` — an address bundled with its exclusive view.
- Views compose with a **separating conjunction** `⊗`: `(p ↦ a) ⊗ (q ↦ b)` asserts
  *disjoint* ownership of two cells. Disjointness is what makes aliasing-free
  mutation and parallelism sound.
- **Recursive views** describe linked structures:
  `List p = (p = null) ∨ ∃ q. (p ↦ Cons q) ⊗ List q`. Pattern-matching a list view
  *unfolds* it into the head cell ⊗ the tail list, so you can free or mutate the
  head and recurse on the tail with full safety.

The address `p` is `ω` (copy it freely); the *view* is `1` (linear). This is the
"L3 address/permission split" tallyc was originally built around — generalized from
one canned data structure to arbitrary, user-defined heap layouts.

### 5.6 Initialization typestate — no uninitialized reads

Raw allocation yields *uninitialized* memory. The view tracks this as typestate:

```
alloc  : (n : Bytes) → ∃ p. p ↦ Raw n          -- uninitialized
write  : (1 _ : p ↦ Raw) → (v : T) → p ↦ Init T -- now initialized
read   : (& (p ↦ Init T)) → T                   -- only Init can be read
free   : (1 _ : p ↦ Raw n) → ()                 -- free needs raw (drop runs first)
```

A `Raw` cell cannot be read; only `write` turns it `Init`. Reading uninitialized
memory is a *type error*. `box` is `alloc` then `write`, so the common path never
exposes `Raw`.

### 5.7 Bounds safety via dependent indexing

Arrays and slices are length-indexed; indices are `Fin n` (a natural `< n`):

```
get : {0 n : Nat} → & (Array n T) → Fin n → T   -- no Fin n exists ⇒ no OOB
```

Out-of-bounds is *unrepresentable*, not *checked-at-runtime*. The `n` and the `Fin`
proof are `0`-quantity and erase: the emitted code is a bare indexed load, no bounds
branch (unless you *choose* a runtime-checked index, which produces an `Option`).

### 5.8 Concurrency

Data-race freedom falls out of linearity: shared state is either `&T` (read-only,
freely shared) or `&mut T`/`Own T` (unique, not shared). A thread spawn that
captures a `1`-resource moves it; two threads cannot both hold a `&mut` to the same
cell because the view's separating conjunction forbids overlap. Higher-level
concurrency (channels, locks) is built as linear-typed libraries.

---

## 6. What "100% safe" means — precisely

"Safe" is a theorem about the fragment outside `unsafe`. A well-typed safe program
enjoys **type soundness** (progress + preservation) *and* the **memory-safety
invariant**: at every step,

- every pointer dereferenced is to live, owned, *initialized*, correctly-typed
  memory;
- every index is in bounds;
- every linear resource is consumed exactly once on every path.

Concretely, these bug classes are *unrepresentable*, each by a named mechanism:

| bug class                | ruled out by                              |
|--------------------------|-------------------------------------------|
| use-after-free           | linearity — `free` consumes the only token |
| double-free              | linearity — token already consumed         |
| memory leak              | linearity — a `1` token must be consumed   |
| uninitialized read       | initialization typestate (`Raw`/`Init`)    |
| out-of-bounds access     | dependent indexing (`Fin n`)               |
| type confusion           | typed pointers + monomorphization          |
| data race                | linearity + separation (`⊗` disjointness)  |
| dangling stack reference | region/scope indices on borrows            |
| (in the total subset) non-termination, partiality | totality checking (§4) |

### The trusted computing base

Honesty about what you must trust:

- **The kernel** — a small QTT type checker that *re-checks the fully elaborated
  term*. Elaboration, inference, and unification are **untrusted**: a bug there
  yields a rejected program or a re-check failure, never an unsound accepted one.
  (tallyc already re-checks.)
- **The erasure pass and codegen** — must preserve the operational meaning and
  drop exactly the `0`-quantity data.
- **LLVM, the system allocator, the OS.**
- **`unsafe` blocks and FFI** — the explicit escape hatch (for implementing
  primitives, calling C). Safety is "100% outside `unsafe`," and `unsafe` is
  syntactically marked and greppable. Primitives like `box`/`free`/the allocators
  are implemented in `unsafe` and *audited once*, then exposed with safe types — the
  same model as Rust's `std` and ATS's `$UNSAFE`.

---

## 7. Erasure and compilation

### 7.1 The erasure theorem

Everything of quantity `0` — types, indices, proofs, views, region tags, `Fin`
witnesses — has **no runtime representation**: it is never allocated, stored,
loaded, or branched on. The dependent and ownership apparatus is a *compile-time
bookkeeping* that is deleted before codegen. This is stated as a theorem and is
**checkable in the emitted IR** (tallyc already asserts, in its test suite, that a
length-indexed `Vec` never materializes its index, and that region/cursor machinery
leaves no trace).

### 7.2 Lowering

- Source → elaborate → **kernel re-check** → **erase** → monomorphize → LLVM.
- Monomorphization: polymorphic and dependent functions are specialized per
  (erased) layout, as in Rust/C++ — no runtime dictionaries, no boxing of generics.
- Value ADTs lower to LLVM aggregates with the declared layout; `Own`/borrows to
  raw pointers; `box`/`free` to allocator calls; eliminators to `switch` + field
  loads (and, for `Nat`-like and structural folds, to native loops).
- The zero-overhead guarantee: the safe, dependently-typed program and the
  hand-written C twin compile to equivalent machine code. (Demonstrated today for
  the linear DLL — both fold to identical code — and, after the nullary-constructor
  fix, for tree build/traverse at parity.)

---

## 8. Effects, I/O, and the outside world

The total core is pure. Observable effects live in the partial fragment:

- **I/O and the OS** are partial and effectful; a small algebraic-effects or
  linear-world-passing layer sequences them. (tallyc's sibling language already has
  algebraic effects + continuations; the same design transfers.)
- **Divergence** is an effect: a `Fix`-using function is marked partial.
- **FFI** crosses into `unsafe`; C signatures are imported with hand-written,
  audited safe wrappers (often expressing ownership transfer with `Own`).

The point: effects do not pollute the total core, and the type of a function tells
you which fragment it lives in.

---

## 9. Surface language and ergonomics

A language this expressive is only usable if the ceremony is bounded:

- **Implicit, inferred arguments** for indices and proofs (`{0 n : Nat}` solved by
  unification) — you rarely write indices by hand.
- **Quantity inference** with sensible defaults (`ω` unless annotated), so most code
  reads like an ordinary functional/imperative language; you reach for `1`
  explicitly when you mean ownership.
- **Borrows and `do`-notation** to hide view-threading: a `&mut`-block reads like
  imperative mutation, desugaring to linear view plumbing.
- **Proof ergonomics** — holes, auto/`search` for trivial proofs, and a tactic
  language for the rest; most array code needs only `Fin`/`auto`, not manual proof.
- **Editor-driven development** — type-of-hole, case-split, and totality status
  surfaced interactively (the Idris/Agda workflow).

The aspiration: *ordinary code looks ordinary*; the dependent and linear machinery
appears only where you ask for a guarantee.

---

## 10. Standard library shape

- **`core`** — total: `Nat`/`Int`/`Bool`, `Vec`/`Fin`, `Eq`/`Dec`, `Acc`
  (well-founded recursion), the proof combinators.
- **`mem`** — the memory layer: `Ptr`, views, `Own`, `&`/`&mut`, `Opt`,
  `Array`/`Slice`, allocators (`malloc`, `Arena`, `Pool`, `Stack`). Safe API over an
  audited `unsafe` core.
- **`alloc`-generic collections** — `Vec`/`HashMap`/`List`/trees parameterized by an
  allocator, each a worked example of safe manual memory.
- **`io`** — partial: files, sockets, the event loop.

The collections double as the proof that the model is usable: a `Vec<T>` you can
`push`/`pop` with no leaks, no UAF, bounds-safe indexing, and your choice of
allocator — written *in the language*, not as compiler intrinsics.

---

## 11. Where it sits (prior art)

| system | dependent types | linear/ownership | manual memory / C-level | totality | GC |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Idris 2** | full | QTT (0/1/ω) | no | yes | yes |
| **Agda / Lean 4** | full | partial / no | no (Lean: RC) | yes | yes |
| **ATS** | yes (constrained) | yes (views/vtypes) | yes | not a focus | optional |
| **F\* / Low\* / Steel** | full + SMT | separation logic | yes (Low\*→C) | yes | no (Low\*) |
| **Rust** | no (const generics) | yes (affine + borrows) | yes | no | no |
| **Zig** | comptime only | no | yes | no | no |
| **tallyc (target)** | **full** | **QTT (0/1/ω)** | **yes, explicit** | **yes, opt-in** | **never** |

tallyc's intended niche: **Idris 2's exact type theory (QTT) and totality, with
ATS/Low\*'s manual-memory safety, Zig's explicit allocators, a small decidable
kernel (no SMT in the trusted loop), and full erasure to native code.** The closest
existing point is F\*/Low\*/Steel — but that is an SMT-backed verifier with a large
trusted base; tallyc trades automation for a small kernel and decidable checking.

---

## 12. Implementation roadmap (from today's tallyc)

Today tallyc has: QTT `0/1/ω`, dependent Π/Σ/indexed families/`Eq`, `%builtin Nat`,
eliminators + general recursion (`Fix`, opaque to the checker), erasure proven in
IR, an LLVM backend, linear `Own`/region/DLL primitives, nullary-constructor
unboxing, no GC, and a small re-checking kernel. The path to the design above:

- **Phase A — explicit allocation of ADTs.** Typed `Own T`, `box`/`free`, `Opt`
  with the null-pointer optimization, recursion via `Opt (Own T)`. Decouple
  construction from allocation. *(Gets the Zig/C tree, C-identical, no implicit
  malloc. Mostly backend; weeks.)*
- **Phase B — value layouts.** ADTs as flat values: `sizeof`/alignment/offsets,
  by-value vs by-pointer passing, embedding, niche optimizations. *(The big
  representation rewrite — from "everything is a tagged i64 pointer" to real
  layouts.)*
- **Phase C — views and borrows.** Points-to views, `⊗`, `&`/`&mut`, region-scoped
  borrows, initialization typestate. *(Safe in-place mutation and linked structures
  without canned intrinsics.)*
- **Phase D — allocators.** First-class `Allocator`, `Arena`/`Pool`, region indices
  tying lifetimes to allocators (generalize the existing DLL regions).
- **Phase E — a real totality checker.** Coverage via the pattern-match compiler;
  termination via structural + well-founded recursion (`Acc`); strict positivity;
  `%total` certificates. *(Upgrades the current "simple fold" heuristic into a real
  checker.)* **E1 (termination + `%total`) is implemented.** A dedicated
  `src/totality.rs` runs a structural-descent / size-change analysis over the
  recursive call graph (self + mutual via SCC), POSITIVELY verifying that every
  recursive call decreases on a strict-subterm pattern binder; `%total` is a
  CERTIFICATE — a `%total` fn the checker can't certify is a hard error
  (annotation ≠ proof), partiality is contagious, and per-fn status is reported.
  The trusted base does NOT grow: a `Total` verdict drives lowering to a kernel
  ELIMINATOR (re-checked total-by-construction); a `Partial` one to an opaque
  `Fix` (or an honest hard error), and `full ⊑ structural` monotonicity makes
  "a `%total` fn can never lower to a `Fix`" airtight. Accumulator-style and
  mutual recursion are honestly declined (Phase E2/E3), never mislabeled total.
  *Still to do:* E2 (the real coverage / pattern-match compiler — nested + absurd
  cases), E3 (well-founded `Acc` recursion), E4 (tighten strict positivity).
- **Phase F — universes.** Replace `Type : Type` with a cumulative hierarchy +
  universe polymorphism. *(Required before the totality guarantee is sound.)*
  **Status: the hierarchy is implemented.** `Type i : Type (i+1)`, predicative
  `Π`/`Σ` at `max`, one-directional cumulativity (conversion stays strict so the
  hierarchy cannot collapse), and the key soundness side-condition — a datatype's
  universe must be ≥ every constructor argument's level — which rejects the
  self-quantifying datatype at the heart of Girard's/Hurkens' paradox. Large
  elimination targets any universe (the motive's level is inferred and genuinely
  type-checked). *Not yet done:* universe **polymorphism** in definitions; until
  it lands the surface defaults every `Type` to `Type 0` (a sound sublanguage),
  and definitions needing a higher universe are written at the kernel level.
- **Phase G — ergonomics and effects.** Borrow/`do` sugar, proof automation, the
  effect/IO layer, the standard library.

Phases A–D make it a safe systems language; E–F make the totality guarantee real;
G makes it usable.

---

## 13. Open problems and honest risks

- **~~`Type : Type` today~~ — RESOLVED (Phase F).** The kernel now has a
  predicative, cumulative universe hierarchy (`Type i : Type (i+1)`), and a
  datatype can no longer sit in a universe small enough to quantify over itself
  (the predicativity side-condition that blocks Girard/Hurkens). A `total`
  program can no longer inhabit `False` via the universe loophole. The
  side-condition is `level(field/param/index) > universe` (strict), **not** `≥`:
  a family deliberately may store *values* of, and quantify over a *parameter*
  in, its own universe — `Vec (A : Type 0) : Type 0` does not lift to `Type 1`.
  Only storing a *universe itself* is the Girard retract, and only that is
  rejected; storing the parameter-type itself is caught by the same field check.
  *Remaining gap:* universe **polymorphism** is not yet implemented, so the
  surface is restricted to `Type 0` (positive multi-level datatypes are reachable
  only from the kernel/tests); this is a sound restriction, not an unsoundness.
- **Linearity × dependency interaction.** Combining quantities with dependent
  pattern matching and views has subtle corners (e.g. how a `1`-resource's *type*
  may depend on a value it is used with). QTT handles the basics; the view layer
  (Phase C) is where the genuine research risk is — this is the part F\*/Steel
  needed years and an SMT solver for, and tallyc wants to do it decidably.
- **Termination expressiveness vs convenience.** A purely structural checker rejects
  too much; well-founded recursion is powerful but demands the programmer supply a
  measure. Finding the ergonomic sweet spot is open.
- **Borrow ergonomics without a bespoke borrow checker.** Expressing Rust-grade
  borrowing purely in QTT + regions, with good inference, is unproven at scale.
- **Erasure soundness across the whole feature set.** Each new construct must
  preserve "0 ⇒ no runtime trace." This is currently maintained by IR-level tests;
  it should become a stated and (ideally) mechanized invariant.
- **Compile-time cost of dependent + monomorphized code.** Full dependent types plus
  monomorphization can be slow to compile; staging and caching matter.

None of these is known to be impossible — each has been solved *separately* by some
system in §11. The bet is that QTT is the substrate on which they compose into one
small, fast, safe, native language with no GC.

---

*tallyc: C's machine, Idris's types, Rust's safety, no garbage collector, and a
button that says "this is total."*
