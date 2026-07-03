# Phase Safety — perfect safety: completing the surface, hardening the core

*tallyc's thesis is that one mechanism (QTT `0/1/ω`) delivers C's machine model,
Idris's types, and Rust's safety with no GC. The performance pillar is proven
pointwise (DLL, tree, quicksort, byte/float sweeps, intrusive list, pool DLL all
at measured C parity) and the layout is now fully packed. This document is the
plan for the SAFETY pillar: prove there is no current hole, harden the two
completeness corners, and then extend what can be **safely expressed** —
shared borrows, initialization typestate, and the one genuinely-missing pillar,
data-race-free concurrency.*

---

## 0. Where safety actually stands (audited, not assumed)

The honest current invariant. **tallyc is memory-safe today for everything it can
express.** Verified by construction and by probe:

| bug class | ruled out by | status |
|---|---|---|
| use-after-free / double-free | linearity — `free`/`unbox` consume the only `1` token | enforced, kernel re-checked |
| memory leak | linearity — a `1` token must be consumed on every path | enforced |
| out-of-bounds | dependent `Fin`/`Lt` index, erased | enforced, IR-tested (no bounds branch) |
| uninitialized read | **no raw-alloc primitive is exposed** — `alloc` stores a value, `anew` fills | enforced by absence |
| type confusion | typed pointers + monomorphization / per-instantiation layout | enforced |
| non-exhaustive match | the elaborator REJECTS a missing case (probe: `missing a case for Blue`, `match on Nat is missing the zero case`) | enforced for flat matches |
| **data race** | **there is no concurrency primitive at all** — no `spawn`, no shared mutable cell across threads | **not a hole: the capability is absent** |

Two distinctions this plan is built on:

- A **hole** is a program you can write today that is unsafe. Audit finding:
  there is **no known hole**. The single `build_unreachable()` default in a boxed
  switch is reached only for a tag the kernel has proven impossible (an absurd /
  refuted arm), and every reachable arm is covered or the program is rejected.
- A **missing feature** is a capability that does not exist, so there is nothing
  to be unsafe with. Concurrency, shared `&`, and general recursive views are
  missing features, not holes.

So "perfect safety" here means two things, in order: (1) keep the no-hole
property airtight as the language grows (Phase 0 + the completeness items), and
(2) grow the set of things you can express **without** ever introducing a hole
(the new safe features).

### 0.1 The non-negotiable principle — the trusted base never grows silently

Every safety feature below is either:

- **derived in the language** (a linear-typed library over existing primitives), or
- a **minimal, audited `unsafe` primitive** exposed with a safe type that the
  **kernel re-checks structurally**. Elaboration is untrusted; a bug in a new
  feature yields a rejected or re-check-failing program, never an unsound accepted
  one.

No new construct may (a) let a `0`-quantity thing leave a runtime trace, (b) let a
`1`-resource be used other than exactly once, or (c) reach a `build_unreachable()`
on a tag the kernel has not proven impossible. Each phase's **gate** tests exactly
these three invariants for the construct it adds.

---

## Phase 0 — Safety audit (prove no current hole before adding anything)

Deliverables, each a checked-in test or doc, no feature work:

- **0.1 `unreachable` ledger.** Enumerate every `build_unreachable()` /
  `unreachable` emission in `dep_codegen.rs` (boxed `Case`/`Elim` defaults, absurd
  arms, refuted convoy arms) and pin, with an adversarial corpus, that each is
  guarded by kernel-checked exhaustiveness or a proven-empty index. Adversarial
  inputs: non-exhaustive flat matches (already rejected), nested/deep patterns,
  multi-scrutinee matches, mixed absurd+reachable arms, `%partial` bodies. **Any
  non-exhaustive match that COMPILES is a P0 bug**, fixed before Phase A.
- **0.2 Erasure invariant, extended.** The existing IR tests assert `0 ⇒ no
  runtime trace` for `Vec` indices and region/cursor machinery. Extend the same
  assertions to the constructs added since: packed-scalar leaves, views
  (`PtsTo`/`Loan` are already width-0), the null niche, and `dlt` proofs. One
  test per construct: the erased witness never becomes an instruction.
- **0.3 Trusted-base ledger (`docs/TRUSTED_BASE.md`).** One table: every
  kernel-opaque primitive (`alloc`, `free`, `unbox`, `anew`/`aget`/`aset`/`afree`,
  `borrow`/`restore`, the pool ops, `dlt`, the scalar ops/casts, `%foreign`,
  `putc`/`getc`) with the safe type it is exposed at and the one-line argument for
  why its safe type is sound. This is the honest TCB, and the checklist every
  future primitive must join.

**Exit gate for Phase 0:** the adversarial coverage corpus is green (all bad
matches rejected, all good ones compile and run), the erasure assertions pass, and
the TCB ledger exists. Only then does feature work start.

---

## Part A — complete the safety surface

### A1. Real coverage / the pattern-match compiler (E2) — *safety-critical, first*

**Why first.** A coverage gap is a *memory-safety* hole: an unhandled tag reaching
its arm hits `unreachable` = UB. Flat exhaustiveness is enforced today; the gap is
**nested** patterns (`match p { Cons(Cons(...), ...) }`), **multiple scrutinees**,
and **absurd discharge** beyond the current single-index v1 scope
(`try_absurd_match` / the convoy `absurd_ctor` classifier in `rust_surface.rs`).

**Design.** A standard dependent coverage checker: split the first scrutinee on its
constructors, recurse on the residual matrix for each, and discharge a column whose
index makes every constructor impossible (extend the v1 "single decidably-empty
index" to any index the unifier proves empty). Lower to nested kernel
`Case`/`Elim`, which is re-checked. The elaborator stays untrusted; a coverage bug
becomes a rejected program.

**Gate.** An adversarial corpus of nested/multi-scrutinee non-exhaustive matches is
*all* rejected; the valid nested versions compile, run, and kernel-re-check; and no
new `unreachable` is reachable on a non-absurd tag (checked against the 0.1
ledger).

### A2. Shared `&` read-only borrows (duplicable views)

**What's missing.** Only the unique, linear `&mut` read-back borrow exists
(`borrow`/`restore`, Phase C slice 1). There is no way to hand out many concurrent
read-only references.

**Design.** A shared borrow is a **duplicable (ω), read-only, scoped view**: split
an `Own T` into an ω address plus a `0`-width, ω, read-only permission
`Shared p a`; the permission may be freely copied, grants `read` only, and the
lender regains full ownership when the borrow scope closes. This reuses the exact
Phase C machinery (address is ω, view is erased) with two changes: the view is
duplicable and read-only, and `restore` waits for scope end rather than a single
consumer. No fractions needed for read-only sharing; a later refinement can add
fractional permissions if a shared-then-reunify-to-unique pattern is required.

**Gate.** Many `&` coexist and read the same cell; a `&mut` cannot coexist with any
live `&` (a linearity error); `free`-under-shared-borrow is a compile error; and
the emitted code is a bare load through the ω address (identity — no refcount, no
trace of the permission), asserted in IR.

### A3. Full initialization typestate — the `Raw`/`Init` story

**What exists.** The size-carrying `Hole a` typestate (take-then-refill) and the
fact that no raw-alloc primitive is exposed, so uninitialized reads are impossible
today. **What's missing.** The *general* story that lets a library allocate raw and
initialize field-by-field while the checker forbids reading the uninitialized part.

**Design.** Expose the audited primitives
`ralloc : (n : Bytes) -> ∃ p. p ↦ Raw n`,
`write : (1 _ : p ↦ Raw) -> (v : T) -> p ↦ Init T`,
`read : & (p ↦ Init T) -> T`,
`free : (1 _ : p ↦ Raw n) -> Unit`.
A `Raw` cell is **unreadable by type** — only `write` turns it `Init`. `alloc`
becomes `ralloc` then `write` sugar, so the common path never exposes `Raw`. Each
is one audited `unsafe` primitive with a safe type; the typestate is `0`-width and
erased.

**Gate.** Reading a `Raw` cell is a type error; `free` on an `Init` cell requires
dropping to `Raw` first (destructor discipline); the desugared `alloc` produces IR
identical to today's; the typestate leaves no runtime trace.

### A4. Concurrency as a linear-typed library — the one genuinely-missing pillar

**What's missing.** Everything. There is no `spawn`. Data-race freedom is a
*theorem waiting for a feature*, not an implemented guarantee.

**Design.** The minimal audited primitive is a `spawn` that **moves** a linear
environment into a fresh OS thread and a `join` that recovers its linear result:

```
spawn : {0 a} -> (1 work : Own Env -o Own a) -> (1 env : Own Env) -> Handle a
join  : {0 a} -> (1 h : Handle a) -> Own a
```

Because `env` is consumed at `1`, it is *moved*, not shared — the parent cannot
touch it afterward. Shared state crosses only as `& T` (read-only, A2); unique
state (`&mut` / `Own`) is moved and never aliased. On top of `spawn`/`join`:

- **structured parallelism** `par : (& A) -> (& A) -> ...` over provably-disjoint
  slices (the disjointness is the separating conjunction the view layer already
  encodes: two `&mut` into non-overlapping array halves);
- **channels / locks** as ordinary linear-typed libraries (a channel endpoint is a
  `1` resource; a lock hands out a scoped `&mut`).

**The theorem this earns (G2, concurrency case).** Two threads cannot both hold a
`&mut`/`Own` to the same cell, because the view's separating conjunction forbids
overlap and `spawn` consumes its moved state linearly; `&` is read-only. So a
well-typed concurrent program is data-race-free by the same accounting that makes
single-threaded mutation safe.

**Gate.** Sharing a `&mut` (or a second use of a moved `Own Env`) across `spawn` is
a compile error; a structured `par` over disjoint array halves computes the right
answer AND the emitted binary is **ThreadSanitizer-clean**; a deliberately-racy
version is unwritable in safe code (only via `%foreign`/`unsafe`). This is the
milestone that makes "Rust's safety" true for the concurrent case.

**Risk.** A4 is the largest and most research-adjacent item here. The
`spawn`/`join`/`par`-over-disjoint-slices core is tractable and rests on machinery
that exists. A fully general channel/actor separation story is harder and can be
staged after the structured-parallelism core lands.

---

## Part B — harden the total / dependent core

The dependent guarantees (bounds safety, proof-carrying invariants) are only as
trustworthy as the total fragment they are stated in. Three completeness items.

### B1. Well-founded recursion (E3)

**Gap.** The totality checker (`src/totality.rs`) certifies structural and
accumulator-fold descent (E1 / Phase 1a′). Genuinely-total algorithms whose
descent is *measured* rather than structural — quicksort, gcd, binary search —
run today at C speed but only as `%partial` (uncertified).

**Design.** `Acc`-accessibility recursion: a recursive call is justified by an
`Acc`-proof on a well-founded relation, the proof `0`-erased. Add `Acc`/`WfRec` to
the total core; the termination checker accepts a call guarded by an accessibility
witness; lowering is the existing `Fix` (runtime-identical) but the definition now
carries a `%total` certificate the kernel accepts.

**Gate.** `qsort` and `gcd` are certified `%total`; their runtime is unchanged
(still C parity); a non-decreasing well-founded call is rejected (annotation is not
proof).

### B2. Universe polymorphism (F)

**Gap.** The universe hierarchy is sound (`Type i : Type (i+1)`, predicative,
cumulative, Girard/Hurkens retract rejected), but the surface pins every `Type` to
`Type 0`. Multi-level positive datatypes are reachable only from the kernel.

**Design.** Level variables in definitions (`def id {l} (A : Type l) ...`), inferred
where possible; conversion stays strict so the hierarchy cannot collapse.

**Gate.** A level-polymorphic `id` / `Vec` type-checks at several levels in one
program; the self-quantifying (Girard) datatype is still rejected.

### B3. Strict positivity (E4) + mutual recursion in totality

**Gap.** Positivity is the seed occurrence check; the totality checker declines
mutual recursion conservatively (any back-edge ⇒ `Partial`).

**Design.** Full strict-positivity for nested and parameterized datatypes
(recursion only in strictly-positive positions, never left of an arrow); size-change
/ lexicographic descent over the mutual-recursion SCC so mutually-recursive total
functions are certified.

**Gate.** A known-bad positive/negative datatype corpus is correctly split;
mutually-recursive `even`/`odd` is certified `%total`; a mutual non-descent is
rejected.

---

## Sequencing

Safety-first: prove no hole, close the one latent-hole class, then extend.

```
0  (audit)                         — no feature work; prove the no-hole property
A1 (coverage E2)                   — the only place a real hole could live
A3 (Raw/Init typestate)            — general safe raw allocation
A2 (shared & borrows)              — read-only sharing, prerequisite for par
B1 (well-founded recursion E3)     — independent; earns %total for qsort/gcd
A4 (concurrency)                   — the missing pillar; the big one
B2, B3 (universe poly, positivity/mutual) — soundness completeness
```

Phases before A4 are finishing work, each with a clear gate. A4 and, further out,
general recursive views (the erasing inductive heap predicates behind an
in-language O(1)-remove DLL, tracked in `PHASE_C2_RECURSIVE_VIEWS.md`) are the
research-hard frontier; everything else is reachable with the machinery that exists.

---

## Open risks (honest)

- **Concurrency separation at scale (A4).** The structured-parallelism core is
  sound by the existing linearity + disjointness accounting; a general channel /
  actor story with separation is unproven at scale (the part F\*/Steel needed SMT
  for). Stage the general case after the core.
- **Coverage completeness (A1).** Getting nested + absurd coverage *provably*
  complete (no reachable `unreachable`) is the highest-value correctness work;
  the risk is a subtle nested/indexed case slipping through. Mitigated by the 0.1
  adversarial ledger being the acceptance gate.
- **Borrow ergonomics without a bespoke borrow checker (A2).** Expressing shared +
  unique borrows purely in QTT + regions with good inference is unproven at scale;
  read-only sharing (no fractions) is the tractable first slice.
- **Erasure soundness across every new construct.** Each feature must preserve
  `0 ⇒ no runtime trace`; maintained by the 0.2 IR assertions, ideally promoted to
  a mechanized invariant later.

*Perfect safety is not a single feature; it is the invariant that the no-hole
property survives every extension. This plan closes the two completeness corners,
adds the missing safe capabilities one audited primitive at a time, and gates each
on the three unbreakable rules: erase every `0`, consume every `1` once, and never
reach an unproven `unreachable`.*
