# 05 — Soundness, and how we get evidence for it

How do we become *certain* λ-Tally is correct? We don't, fully — no one does.
What we can do is shrink the trusted base, state honestly what is proved vs.
assumed vs. conjectured, and stress-test the rest from several independent
directions. This document records the distinctions, the plan, and the first
concrete evidence.

## "Sound" is not one claim

We had been using one word for several separable properties:

| Claim | Meaning for λ-Tally | Status |
|---|---|---|
| **Type safety** (progress + preservation) | well-typed ⟹ does not get stuck (no use-after-free at runtime) | tested (Redex) + bounded-verified (Rosette, below) |
| **Logical consistency** | not every type is inhabited; `⊥` has no closed proof | **deliberately false** today — prototype uses `Type : Type` |
| **Normalization / decidable checking** | the checker always terminates; conversion can't loop | not guaranteed (`Type : Type`); needs a universe hierarchy |
| **Resource correctness** | a `1` really means "used once at runtime"; `0` really erases | asserted; GraD proves the analogue |
| **Algorithmic soundness + completeness** | the *checker* accepts exactly what the *declarative rules* do | **no declarative rules exist yet** to check against |

The last row is the deepest gap: right now the Python checker *is* its own
specification, so it cannot be verified against anything. Writing the
declarative system on paper (roadmap M1.1) is the prerequisite for every
positive correctness claim.

## The ladder of assurance

```
(a) examples / unit tests        ← the prototypes' test suites
(b) property-based fuzzing        ← Redex redex-check (random)
(b+) bounded-exhaustive checking  ← Rosette + Z3 (all programs up to size k)
(c) pen-and-paper metatheory      ← what a POPL paper does
(d) mechanized proof (Agda/Rocq)  ← high assurance for the *calculus*
(e) verified implementation       ← checker proven against the mechanized spec
```

Each rung covers the blind spots below it. (a)–(b+) can only *refute* soundness
(find a counterexample), never *establish* it. The realistic target is **(d) for
the declarative calculus + (b+) continuously for the implementation**, with
differential testing as a sanity layer.

## Does any existing system already do what we want?

No single one — and the gap is precisely our contribution and our risk. Two
well-developed halves exist:

- **The quantitative/graded dependent kernel** (+ mechanized metatheory + proven
  erasure): QTT (Atkey; Idris 2 in production), **GraD** (Choudhury–Eades–
  Eisenberg–Weirich: graded dependent types with a heap-based *resource*-
  soundness theorem), and the Agda-formalized **graded modal dependent type
  theory** (Abel–Danielsson–Eriksson: normalization, consistency, decidability,
  erasure). This half is essentially *borrowable* — we should adopt one of these
  as the proof skeleton rather than reinvent it.
- **Linear views over mutable memory** (strong update, alloc/free, safety): **L3**
  (not dependent), **ATS** (dependent, but a *separate* linear layer, not a
  unified semiring), **F\*/Low\*/Steel** (achieves the goal via *separation
  logic*, not multiplicities).

The one cell **nobody fills**: L3/ATS-style linear mutable views living *natively*
as multiplicity-`1` entries in a QTT/graded dependent context, with a mechanized
end-to-end *memory*-safety theorem. That bridge is λ-Tally's bet.

It is not unfilled by accident. The hard interaction: **dependent types let types
mention values; strong update lets a value's type change.** If a dependent type
elsewhere references a location whose contents' type just mutated, soundness is
on the line. ATS and Steel sidestep this by separation; we confront it. So the
bridge is where soundness either lands or breaks — and the first thing to probe.

## First concrete evidence: bounded memory-safety (Stage D)

`prototype/memory-safety-rosette.rkt` attacks exactly that crux with SMT-backed
bounded-exhaustive checking (rung b+). The model is genuinely **dependent**:
locations hold *length-indexed arrays*, a view carries the length, and an element
access requires a static bounds check (`i < n`) — our stand-in for a dependent
proof. `resize` is a **strong update that changes the dependent index** `n`, so
the failure mode is a real **out-of-bounds access** (or use-after-free), not mere
tag confusion. A key economy: an *erased* (multiplicity-0) dependent proof
"`l : Array n`" and a *duplicated* linear view (aliasing) are the **same** hazard
here — "a length claim that outlives the mutation" — so one secondary-capability
slot models both. Two disciplines:

- **SOUND**: a read requires a live **linear view** (mirrors the Stage-C primitive
  types, where `read/write/free` take `View A l` at multiplicity `1`). A stale
  claim cannot authorise a read.
- **BROKEN**: a stale claim alone may authorise a read ("what if we mis-graded the
  view, or let an erased proof be used at runtime").

Result (Z3, in seconds):

- **SOUND — verified safe for *every* program up to length 16** (≈10²²·k operation
  sequences over the symbolic alphabet), *including* the `alloc; mkclaim; resize;
  staleread` strong-update-under-a-dependent-reference programs. Bounded evidence
  that linearity removes the hazard.
- **BROKEN — counterexamples at length 4**: `alloc l len=1; mkclaim l; free l;
  staleread l` (use-after-free), and with `free` disabled, `alloc l len=1;
  mkclaim l (claims len 1); resize l len=0; staleread l idx=0` — a genuine
  **out-of-bounds** read: the stale length claim outlived the `resize`.

The contrast is the content: under the sound rule, `resize`/`free` consume the
linear view, so the later read's guard fails — the stale claim cannot authorise
it. **The multiplicity-`1` requirement on views is exactly what is load-bearing**,
and it defends against the erased-reference *and* the aliasing form of the hazard
at once. This validates the design choice baked into the Stage-C prelude.

Scope/trust (honest): this is *bounded* (no induction → refutes, doesn't prove
for all sizes), and its trusted base is Z3 + Rosette's symbolic VM + the model
faithfully reflecting the real rules. It is a strong bug-finder and design
validator, not a foundational certificate. The model is still abstract: a finite
set of locations and lengths, with bounds-checks standing in for full dependent
proofs — not the real conversion checker.

**Update:** the model now goes beyond bounded length. An *inductive store-typing
invariant* is machine-checked to be preserved by every operation from any
well-formed state, so memory safety holds for programs of **any length** under
the sound discipline — a proof of the operational safety theorem *for the model*.
See `docs/06` for the Rosette metatheory plan, the inductive technique, the
reachable-vs-proof-assistant split, and the staged path (resource ledger,
functions, connecting the static checker, inter-location dependency).

## Plan (extends roadmap Phase 1–3)

1. **Write the declarative system on paper** (M1.1) — nothing can be "correct"
   until there is an independent spec. Replace `Type : Type` with a universe.
2. **State each theorem with explicit hypotheses**, including the *conditional*
   nature of the Stage-C guarantee (the postulated primitives are axioms; their
   realizability is an obligation, witnessed operationally by Stage B/D).
3. **Hunt counterexamples cheaply first** — extend Rosette (more ops, real type
   indices, larger bounds) and Redex before attempting proofs.
4. **Differential-test** the Python checker against **Idris 2** (production QTT)
   and **Agda**'s quantities on shared examples; disagreement = bug.
5. **Discharge the realizability obligation**: prove/fuzz that the postulated
   primitive types are *validated* by the heap semantics (a logical-relations
   argument) — this is what turns Stage C's conditional safety into safety.
6. **Mechanize** the declarative calculus + metatheory in **Agda** (closest prior
   art) or **Rocq** (if we want the MetaCoq-style verified-implementation
   endgame), cribbing structure from GraD and graded-Agda.

## The irreducible residue

Even full mechanization rests on three things a proof assistant cannot check for
you: the **kernel's own correctness**, whether the **theorem statement says what
you meant**, and the **consistency assumption** Gödel forbids discharging
internally (MetaCoq assumes strong normalization for exactly this reason).
"Certain" means a tiny, well-understood trusted base with everything above it
stress-tested from independent angles — not zero doubt.

> References for the systems named here are in the previous chat research and
> must be verified into `notes/bibliography.md` (D0.1) before they are relied on.
