# 06 — Verifying the metatheory in Rosette: plan, results, and ceiling

Goal: push Rosette/Z3 as far as it goes toward *writing down and checking the
metatheory*, so we de-risk the core idea — linear mutable views inside a
(lightly) dependent setting give memory safety + strong update — **before**
investing in the Agda/Rocq proofs (`docs/05`). This file tracks what is
reachable here, what is not, and the staged path.

## What Rosette can and cannot reach

Rosette is automated and **bounded**: it reduces a property to SMT and either
verifies it (within a finite domain) or returns a concrete counterexample. So:

| Metatheorem | Reachable in Rosette? | How |
|---|---|---|
| **Memory safety** (no UAF/OOB) | **yes, and now UNBOUNDED in trace length** | inductive store-typing invariant (below) |
| **Preservation + progress** (operational) | yes | invariant is inductive + implies safety |
| **Resource soundness** (1 = used once; 0 erases) | yes (planned E2) | usage ledger as an invariant |
| **Algorithmic = declarative typing** | partly (bounded term size) | symbolic terms up to depth d |
| **Normalization** (every term reduces) | **no** | needs induction over all terms |
| **Logical consistency** (`⊥` uninhabited) | **no** | needs the above |
| **Decidability of conversion** | **no** (only bounded sanity) | higher-order, undecidable |

The bottom three are *fundamentally* proof-assistant work. Rosette's job is to
make us confident the design is right — and to find bugs cheaply — so the proof
effort is aimed at something that actually holds.

## The key technique: an inductive invariant beats bounded length

Our first Rosette results (`docs/05`) said "safe for every program **up to length
k**". `memory-safety-rosette.rkt` now does better. We exhibit a store-typing
invariant `Inv(state)` and ask Z3 to prove:

- **base**: the initial state satisfies `Inv` (concrete: holds);
- **(P) one-step preservation + safety**: from an **arbitrary** state satisfying
  `Inv`, **any single operation** (a) performs no unsafe access and (b) yields a
  state still satisfying `Inv`.

If (P) holds, then by induction `Inv` — and therefore safety — holds at **every
reachable state, for programs of any length**. The *state* stays finite (so Z3
copes), but the *trace length is unbounded*. Array lengths are unbounded naturals
too (we dropped the length cap for this check). Locations are independent (each
op writes only its own location's fields), so the argument is uniform in the
number of locations.

`Inv`: *holding a live linear view of a location implies the location is alive and
the view's claimed length equals the actual length.* That store-typing
consistency is exactly what makes a SOUND read in-bounds.

## Result so far (E0–E1)

- **E0 — abstract memory transition system, bounded.** SOUND verified for all
  programs up to length 16; BROKEN counterexamples (UAF and the dependent
  out-of-bounds witness `alloc len=1; mkclaim; resize len=0; staleread idx=0`).
- **E1 — the metatheorem.** For the SOUND discipline (which mirrors the Stage-C
  primitive multiplicities: `read/write/free` require the multiplicity-`1` view),
  **`Inv` is an inductive invariant and implies safety → memory safety holds for
  programs of ANY length, arrays of ANY size**, in seconds. BROKEN breaks `Inv`,
  and the bounded check supplies a concrete *reachable* unsafe trace.

This is a real proof of the operational memory-safety theorem **for the model**.
The load-bearing fact, now established rather than asserted, is that the
multiplicity-`1` requirement on views is what removes the stale-claim hazard — in
both its erased-dependent-reference and its aliasing forms (one `scap` slot
models both).

## The honest ceiling (what the model still abstracts)

- **Locations are a finite set; lengths/indices stand in for dependent proofs.**
  The bounds check `i < n` represents a dependent `i < n` proof; it is not the
  real dependent-conversion checker.
- **No real terms or closures yet** — it is a transition system, not the λ-Tally
  reduction relation. The dreaded symbolic-substitution problem is sidestepped by
  not having substitution; E3 reintroduces functions via an *environment*
  machine, not substitution, to stay tractable.
- **The discipline is encoded as operational guards**, not *derived* from the
  Stage-A/C typing rules. Closing that gap (E4) is what connects the static
  checker to this dynamic guarantee.
- **Z3 + Rosette + model fidelity** remain trusted. Bounded checks refute; the
  inductive result proves-for-the-model, not for the real calculus.

## Plan to keep going

- [x] **E0** abstract memory model, bounded-exhaustive safety.
- [x] **E1** inductive store-typing invariant → unbounded operational safety.
- [x] **E2** **erasure soundness** as a non-interference theorem
      (`memory-safety-rosette.rkt`): two states agreeing on the runtime-relevant
      fields but differing arbitrarily on the erased (multiplicity-`0`) fields
      stay in agreement and have identical safety after any operation. So the
      `0`-fragment is *computationally irrelevant* — the formal licence to erase
      it. SOUND: holds. BROKEN (a read may consult the erased claim): fails.
- [x] **E3 (core)** **environment machine** with binding
      (`resource-soundness-rosette.rkt`): a real term language with `let`
      evaluated by an environment evaluator (no substitution — the technical
      point of E3) plus a resource heap. *First-class `λ`/application with
      closures and function types is the remaining extension*; the
      substitution-free binding core that E3 was about is done.
- [x] **E4** **static ⟹ dynamic** (`resource-soundness-rosette.rkt`): a real
      leftover-context **multiplicity type checker** plus the evaluator, with Z3
      proving the resource-soundness theorem — *accepts(e) ⟹ run(e) leaks
      nothing and never double-frees / uses-after-free* — for **every well-typed
      closed program up to depth 4**. A BROKEN checker that keeps the type checks
      but drops “used exactly once” is shown to accept a leaking program
      (`seq (use new) (let _=new in unit)`). So linearity is precisely what turns
      a type checker into a memory-safety guarantee.
- [x] **E5** **inter-location dependency** (`interloc-dependency-rosette.rkt`):
      a dependent capability whose read-bound on one location is the *value
      stored at another* location. The lambda-Tally fix — forming the dependency
      **freezes** both cells (consumes their linear views) so neither can be
      mutated while it is live — is verified as an **inductive invariant
      (unbounded length)** under SOUND; BROKEN (no freeze) breaks it and a
      bounded search exhibits a concrete unsafe trace (`alloc; alloc; mkdep;
      free arr; depread`). The hardest case — “types depend on values; strong
      update changes values”, now across two cells — holds. (Remaining: first-
      class functions in E3, larger bounds, unifying the Rosette models.)
- [ ] **Handoff** the unbounded/inductive-over-terms results (normalization,
      consistency, decidable conversion, full dependent Π/Σ metatheory) to Agda
      (`docs/05`), cribbing GraD and graded-Agda.

### Where this leaves "does the idea work?"

The operational metatheory is now machine-checked: memory safety is **unbounded**
(inductive invariant, E1), erasure is sound (E2), and the **static multiplicity
checker is proved to deliver memory safety** on every well-typed program up to a
depth bound (E4) — with the linearity rule shown to be load-bearing. That is
strong, concrete evidence that the core idea is sound. What remains is breadth
(first-class functions, inter-location dependency, larger bounds) and the
pure-type-theory properties (normalization, consistency, decidable conversion)
that are fundamentally proof-assistant work — the Agda handoff in `docs/05`.
