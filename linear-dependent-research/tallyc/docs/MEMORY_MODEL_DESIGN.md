# tallyc memory model — design (for review)

*The differentiator: a language as low-level as C, with complete manual control over
allocation, that is nonetheless 100% memory-safe — using ONE mechanism (QTT's per-
variable quantity `0/1/ω`) for erasure, ownership, and borrowing, with NO garbage
collector and NO reference counting, ever.* This document is the design to review
before any kernel/implementation work; it synthesizes three layers (allocation,
borrows/views, surface) into one coherent model, grounded in tallyc as it exists
today, and resolves the cross-layer seams. Companion: `FUTURE_WORK.md` §5 (the
vision), §6 (the safety theorem + TCB), §7 (erasure).

## 0. The one-quantity spine (already true in tallyc)

The kernel's only resource rule is `σ.leq(π)` over the semiring `{0,1,ω}`
(`src/mult.rs`): using a binder adds usages (`1 + 1 = ω`), passing it to a `π`-binder
scales by `π`. From this single rule:

- **`0` ⇒ erasure ⇒ dependent types are free.** A `{0 n : Nat}` index, a proof, a
  region tag, a view witness is scaled to `0` and never reaches codegen
  (`compile_postulate` drops `Mult::Zero` args). Proven IR-checkable today (a length-
  indexed `Vec` never materializes its index; the DLL leaves no region trace).
- **`1` ⇒ ownership ⇒ safe manual memory.** `ω ⋢ 1` rejects using a linear value
  twice (use-after-free / double-free); `0 ⋢ 1` rejects dropping it (leak). `0` and
  `1` are deliberately *incomparable*.
- **`ω` ⇒ ordinary values.** Plain data copies freely.

The whole model is this rule applied to memory. Nothing below grows the trusted
kernel: the primitives are `postulate`s with audited `unsafe` native lowerings, and
the kernel re-checks every USE against their `0/1`-typed signatures.

## 1. Layer A — Allocation & ownership

- **Construction is not allocation.** A `struct`/`enum` constructor builds a *value*
  with known size/align/offsets (Zig-style); it never touches the allocator. `box` is
  the only thing that allocates. (tallyc already compiles a nullary constructor — a
  leaf/`none` — to a shared constant with *zero* malloc.)
- **`Own T` = a linear owned pointer, defined as a kernel linear pair** (NOT a new
  opaque postulate):

  ```
  Own T  ≜  (0 p : Ptr) × (1 _ : p ↦ T)      -- erased address  +  its exclusive 1-view
  free   : {0 a : Type} -> (1 o : Own a) -> Unit
  ```

  Reusing the existing `Σ[1]` makes free/move/arena-handoff/destructor all the *one*
  type `(1 o : Own a) → R` — indistinguishable to the checker (each "consumes the 1"),
  differing only in what the body does.
- **`Opt (Own T)` is the null niche** — `none` is literally the null pointer (the
  zero-malloc nullary path), so recursion via `struct Node { value : I64, left :
  Opt (Own Node), right : Opt (Own Node) }` is exactly the C struct with `NULL` leaves.
- **First-class allocators with region indices `@ r`** (`{0 r : Region}`, erased),
  generalizing the existing DLL regions: `malloc`, bump/`Arena`, fixed `Pool`,
  inline/stack.

## 2. Layer B — Borrows, regions & views (the research-risk layer, done decidably)

- **Points-to views**, `0`/`1`-quantity propositions (no runtime content): `p ↦ v`
  (the right to read/write that cell), separating conjunction `(p↦a) ⊗ (q↦b)`
  (DISJOINT ownership — the soundness core of aliasing-free mutation & parallelism),
  and recursive views for linked heaps (pattern-matching a view UNFOLDS head-cell ⊗
  tail by constructor disjointness — the same auto-generated eliminator the kernel
  builds today, no solver).
- **`&[r] T` shared (ω, copyable)** vs **`&mut[r] T` unique (1, linear)**. v1 uses
  **read-back** primitives (`get`/`set` return the borrow token), threading exactly
  like the existing DLL `Cursor r`; the kernel's "scrutinee consumed once" rule does
  the accounting — `&mut`-as-linear-state, not as magic. (Fractional permissions are a
  later option, not needed for v1.)
- **Borrows are region-scoped views returned at scope end:** `borrow_mut` splits an
  `Own` into a `&mut[r]` + a linear `Loan r a` the lender keeps; `return_mut` reunites
  them into a full `Own`. Dangling references are impossible because `r` is a
  `0`-erased kernel name: a borrow escaping its scope would reference an out-of-scope
  `r` and not type-check — the *same* mechanism that rejects cross-region DLL splices
  today. This is Rust's discipline in QTT + regions, **with no bespoke borrow checker**.
- **Initialization typestate:** `alloc → p ↦ Raw`; `write : Raw → Init`; `read` needs
  `Init`; `free` needs `Raw` (drop first). Reading uninitialized memory is a *type
  error*; `box = alloc ; write` so the common path never exposes `Raw`.
- **Concurrency** falls out: shared = `&T` (ω, shareable), unique = `&mut`/`Own` (1);
  `⊗` disjointness forbids two threads holding `&mut` to the same cell.

**Decidability (no SMT in the trusted loop).** F\*/Steel need a solver because
permissions carry arbitrary heap predicates discharged by entailment. tallyc
restricts views to: linear `Σ`/`⊗` of points-to atoms, strictly-positive recursive
families, and region equality. View entailment then reduces to three *syntactic, no-
search* checks the kernel already runs — (i) rig usage counting, (ii) inductive
unfold by constructor-head disjointness, (iii) region-name unification. We trade
Steel's automatic frame inference for decidability: non-trivial framing must be
*written*, not solved. That is the accepted tradeoff (FUTURE_WORK §13).

## 3. Layer C — Surface & ergonomics

- **Quantity inference, ω-default.** Unannotated = `ω`; write `1` for ownership, `0`
  (usually via implicits) for erasure. The elaborator never *raises* a budget — it
  computes usage; the kernel checks `σ.leq(π)`. Ordinary code reads ordinary.
- **Implicit indices + `Fin n`.** `{0 n : Nat}` solved by unification, written once;
  `Fin n` bounds-safety erases to a *bare indexed load, no bounds branch*.
- **No surface lambdas (decision).** Adding general closures invites hidden allocation
  (a closure is a `box`ed environment) — violating "`box` is the only alloc," and the
  capture-by-linear-value quantity inference is the hard, undecidable-flavored corner.
  Higher-order memory ops take **named top-level fns** (no capture, no alloc); if a
  real closure need arises, the answer is an *explicit, visibly-`box`ed* `Closure`
  type, never an implicit one.
- **`&mut` / `do` / `arena` are pure sugar over the Layer-B view threading**, e.g.

  ```
  &mut node { node.value = node.value + 1 }
  -- desugars to:  let v0 = borrow_mut(node); let v1 = set_field(v0,.value, get_field(v0,.value)+1); let node = return_mut(v1);
  ```

  Each step is a `let` that lowers to a `(1 v : …)` binder, so skipping `return_mut`
  is a `0 ⋢ 1` *leak* error and reusing `v0` after `v1` is an `ω ⋢ 1` *aliasing*
  error — the sugar reads imperative, the desugaring is checked linear plumbing, and
  the views erase to bare stores.
- **Total/partial boundary** is the existing `%total`/`%partial` + `totality.rs`:
  total code reduces in types; `Fix`/IO/unbounded loops are partial and opaque.

## 4. Cross-layer decisions resolved (the seams)

1. **`Own T` is the kernel linear `Σ` pair**, not a fresh postulate — so view-
   splitting, move, and free-as-the-consuming-fn all fall out of existing typing.
   (Surface hides the existential address `p`.)
2. **Two allocator disciplines, both in QTT+regions** — resolving the A/B tension over
   "is each pointer linear, or the region?":
   - **Individual (malloc):** each `Own T` is its own `1`; freed individually by
     `free`. The leak check forces every `Own` to be consumed.
   - **Region/arena:** the **region capability is the single `1`**; pointers are
     `&[r]`/region-scoped views (`ω` within the scope), and `release : (1 _ :
     Allocator r) → Unit` bulk-frees. Soundness is by SCOPE, not per-pointer
     consumption: after `release`, `r` is out of scope, so no `@ r` pointer can be
     used (type error) — no scan needed. Individual `free` is NOT offered for arena
     pointers (you can't double-free what you can't name post-release).
   This gives arena ergonomics (one bulk free, often faster than C) AND malloc safety,
   from the same machinery.
3. **Read-back `&mut`** (token-returning `get`/`set`) for v1 — reuses the linear-pair
   eliminator verbatim; fractions deferred.
4. **Erasure is the hard gate, tested per-primitive.** Every new `0`-quantity
   construct (`↦`, `⊗`, `Region`, `Loan`, `Raw`/`Init`, `&`/`&mut`, `borrow_mut`/
   `return_mut`/`split`/`join`) must ship with an IR-trace test proving it leaves
   ZERO trace (an `&mut` block → bare load/store; `borrow_mut`/`return_mut` → nothing,
   identity on the address). This is the existing "erasure proven in IR" discipline
   extended to each new primitive — non-negotiable and checkable.

## 5. ⚠ VERIFIED soundness findings — three reachable double-free channels, all CLOSED

The design+review process found (and red-teamed, not asserted) THREE reachable ways a
linear-typed BINDER defaulting to `ω` laundered a double-free. Root cause is uniform: a
binder of a linear type must default to `1`, not `ω`. All three are now fixed
(fail-toward-linearity), each red-teamed to the 1a′ bar (double-free `ω⋢1` REJECTED,
leak `0⋢1` REJECTED, single-use ACCEPTED); both suites stay green.

1. **`let`-bound linear value** (`rust_surface.rs` `elab_let`). `let o = alloc(Zero);
   free(o); free(o)` was ACCEPTED. FIX: a `let` whose bound type `contains_linear`
   binds at `1` (commit `bbbb26974`). Test: `let_linearity_rejects_double_free_and_leak`.
2. **`Own` struct/enum FIELD** (`reject_linear_fields`). `struct Box { p : Own Nat }`
   parses today; a field-hidden `Own` laundered a double-free on TWO channels — `let`
   (a struct with an Own field reads non-linear to `contains_linear`) and no-`let`
   `match` (fields stored at `Mult::Omega`). FIX: reject linear fields AT DECLARATION
   (closes both + enum variants; makes `contains_linear` transitively sufficient).
   Test: `linear_fields_rejected_at_declaration`. **Lifted in Phase A** (field-aware
   linearity — §7).
3. **`Own` function PARAMETER** (`elab_ty` arrow). `fn f(x : Own Nat) { free(x);
   free(x) }` (param defaulted `ω`) was ACCEPTED. FIX: a linear-domain arrow binder at
   the default `ω` elaborates at `1`. Test: `linear_param_defaults_to_one_no_double_free`.

**Remaining (documented, NOT a double-free):** a linear value passed to an
ABSTRACT-typed param (`{0 a} -> a -> …`) stays `ω` (`a` is a `Var` — can't be seen as
linear), so it can LEAK (drop without free), not double-free. This is the §13
linearity×polymorphism corner; the sound fix is real surface linear params + linearity
polymorphism (Phase A). It is NOT papered over by weakening linearity. The CONCRETE
reachable channels (let / field / concrete param) are all closed.

## 6. Non-negotiables — consolidated judgment

| Non-negotiable | Verdict | How |
|---|---|---|
| No GC, no refcount, no temp leak | ✅ | reclamation = the `1`-consume obligation; no refcount slot, no tracing pass |
| rig `0/1/ω` preserved (`ω⋢1` use-twice, `0⋢1` leak) | ✅ | exact accounting holds; the three §5 linear-binder-defaulting-to-ω channels (let / field / param) are all closed + red-teamed; only the §13 abstract-param leak remains (documented, Phase A) |
| erasure leaves ZERO IR trace, checkable | ✅ (target) | only `Ptr` (an `i64`) + real data survive; all views/regions/proofs vanish — per-primitive IR tests (§4.4) |
| C/Zig-low-level, `box` the only alloc | ✅ | constructors don't allocate; arena `box` = one add; tree compiles to its C twin |
| small kernel, decidable, no SMT | ✅ | primitives audited `unsafe` + kernel re-check; entailment = rig + ctor-disjointness + region-unification |

## 7. Build roadmap (refines FUTURE_WORK §12 A–D)

0. **The LINEARITY FOUNDATION — DONE (gate-2 SIGNED OFF; CBV-let under its own review).**
   The whole field-hidden double-free/UAF/leak CLASS is closed by ONE convergent
   mechanism (replacing the earlier whack-a-mole forbids):
   - **Use-site instantiation-aware linearity** (gate-2, SIGNED OFF — reviewer ran 32
     red-team programs, no 5th instance): `type_is_linear` is field-aware (recurses
     ctor field defs, seen-guarded) AND instantiation-aware (types substituted by
     `elim_method_telescope`); every binding site (let / param / match-field via
     `rebind_linear_fields`) binds an instantiated-linear value at `1`, so the
     unchanged kernel enforces `ω⋢1`/`0⋢1`. The forbid is LIFTED. The §13 abstract-param
     corner is confirmed LEAK-only (a concrete `Own` is always bound at 1, so feeding it
     to an ω-param is rejected `ω⋢1`).
   - **CBV `let`** (`Term::Let`, under review; positivity-of-`Let` red-team passes —
     `occurs` recurses ty/e/body): usage `U_e ⊕ U_body` (e counted once), enabling
     multi-owner sequencing without over-counting.
1. **Phase A — explicit allocation:** `Own T` as the `Σ[1]` pair, `box`/`free`, `Opt`
   null-niche, recursion via `Opt (Own T)`; the malloc allocator.
   **DAY-ONE GATING INVARIANT:** the per-primitive erasure **IR-trace test** (§4.4).
   **TWO PREREQUISITES for recursive-`Own` structures** (linked list / tree — the
   bread-and-butter; surfaced by the gate-2 reviewer, both SAFE-direction over-rejections
   today, both BLOCK box/free on linked structures):
   (a) **POSITIVITY must recognize `Own`/`↦` as a POSITIVE type former.** Today
   `struct Node { next : Opt (Own Node) }` is rejected by STRICT-POSITIVITY (`Node`
   under `Own`), not linearity — `strictly_positive` treats `Own Node`'s `Node` like any
   occurrence. But `Own T` is a POINTER: a pointer to `T` is a POSITIVE occurrence (like
   a constructor field, NOT a function-arrow domain). So `strictly_positive`/`occurs`
   must treat `Own`/`↦` as a positive (transparent) wrapper — else linked structures
   can't be declared. (Corrects the earlier doc claim "enables `Opt(Own Node)`
   recursion" — gate-2 lifted the LINEARITY forbid, but POSITIVITY still blocks it.)
   (b) **The ELIMINATOR must JOIN (max) branch usages, not SUM them.** A linear value
   freed exactly-once-per-arm of a `match` is over-rejected `ω⋢1` because the eliminator
   SUMS method-branch uses; only ONE branch runs, so once-per-branch = once → the join
   (max) of branch usages is correct. Same family as the CBV-let fix (usage accounting
   too conservative); its DUAL (let = sequencing/sum-fix; match = branching/join-fix).
   Real box/free code (match a linked structure, free per-branch) hits this.
2. **Phase B — value layouts:** real size/align/offsets, by-value vs by-pointer,
   niche opts (the representation rewrite from "everything is a tagged i64").
3. **Phase C — views & borrows: FIRST SLICE DONE (v2.1).** Views (`PtsTo`, `Loan`)
   are ZERO-WIDTH linear values (IR-trace-tested: a view program is bare
   loads/stores); `&mut` read-back borrows (`borrow`/`restore`) compile to
   identity on the address; take/refill typestate is the size-carrying `Hole a`;
   the misuse space (drop, double-use, free-under-borrow via loan stranding,
   linear-payload duplication via the `vtake`+`vwrite` idiom) is closed by the
   ordinary rig — red-teamed. *Remaining:* recursive views that ERASE (unbounded
   linked structures under views), shared `&` fractions, `&mut { … }` sugar.
4. **Phase D — allocators:** first-class `Arena`/`Pool`, region capabilities + bulk
   `release`, the two-discipline model (§4.2).

## 8. Open risks (honest)

- **Borrow-ergonomics inference at scale** (which view, returned where) — the genuine
  §13 research risk; unproven beyond single-region drivers. Mitigate: read-back v1 +
  region pinned by an in-scope `Loan`, stress-tested before scaling.
- **Decidable framing cost:** non-trivial frames must be written, not solved — the
  accepted price for no-SMT.
- **Region inference for multi-region nests** — believed decidable (each `r` pinned by
  a `Loan`/`Allocator` in scope), must be proven out.
- **`let`-quantity inference fail-safe:** default `let` to `1` and relax to `ω` only
  when the type is provably copyable — fail toward linearity, never away.
