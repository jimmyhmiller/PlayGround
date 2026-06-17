# Prototype: executable, semi-formal models

Two small, runnable models that make the formalisms in `../docs` concrete enough
to argue with. They are deliberately split along the same fault line as the
theory: the *static* multiplicity discipline vs. the *operational* memory story.

| File | Tool | Models | Corresponds to |
|------|------|--------|----------------|
| `qtt_checker.py`   | Python 3 | the QTT **type checker** (multiplicities, dependent Pi/Sigma, NbE conversion, erasure) | `docs/01-qtt-core.md` |
| `memory-model.rkt` | PLT Redex (Racket) | the **operational** memory semantics (heap + new/read/write/free) and a linear type system that rejects use-after-free / double-free / leaks | `docs/02-memory-views.md` |

Why two tools? The thing you most need to *understand* (how linear and dependent
typing cohabit, and how erasure works) is a **type-checking** phenomenon — best
felt by running check/infer and watching the usage ledger. The thing you most
need to *stress-test* (that the C-level memory ops can't go wrong) is an
**operational** phenomenon — best caught by an executable reduction relation and
property testing. See `../docs/04-roadmap.md` for how these feed the eventual
mechanized proofs, and the "Open: unification" note below.

## Stage A — `qtt_checker.py`

```
python3 qtt_checker.py
```

A bidirectional, "resourced" type checker for the QTT core (de Bruijn indices,
NbE for definitional equality). check/infer return a **usage vector**; contexts
combine with the rig operations `+` and `*`. Type positions are checked at
multiplicity `0`, which is exactly how "type-level use is free" / erasure is
realized.

The test suite is the documentation. It shows, among others:

- **the cohabitation demo** — a *linear* identity `(A :^0 Type) -> (x :^1 A) -> A`
  type-checks: `A` appears in the types (of `x` and of the result) yet is spent
  at multiplicity `0`, while `x` is linear and used exactly once;
- the contrast where `A` is declared **linear** but used only in types → rejected
  (type-uses don't discharge a linear obligation);
- duplicate-a-linear, drop-a-linear, and use-an-**erased**-value-at-runtime all
  rejected, with the specific reason printed;
- a linear `swap` over a multiplicative pair (`docs/02`'s "pointer ⊗ view" shape).

Exit code is `0` iff every example's accept/reject expectation is met.

Not modeled here (on purpose): the heap and the memory primitives (that's Stage
B), a universe hierarchy (uses `Type : Type`), and fully dependent let-pair.

## Stage B — `memory-model.rkt`

```
racket memory-model.rkt
```

A PLT Redex model. Two halves:

1. **Operational semantics** `(H e) --> (H e)`: a heap `H` plus call-by-value
   evaluation contexts and the four primitives. `read`/`write`/`free` only fire
   when the location is live, so a capability for a freed location is
   **operationally stuck** — the hazard is real and visible.
2. **A linear type system** (leftover-context formulation, so it's algorithmic):
   pointers are unrestricted, **capabilities are linear**. Using a cap twice
   fails (it's already consumed); never using it leaves a non-empty linear
   leftover (a leak). Hence:
   - `good` / `good-simple` (alloc … free) — accepted, and run to `(empty-heap,
     unit)` (no leak);
   - `bad-leak` (never free) — rejected;
   - `bad-double-free` — rejected;
   - `bad-use-after-free` — rejected.

The file ends with a `redex-check` to illustrate the property-based
"Run Your Research" methodology (fuzzing a metatheorem for counterexamples).

> Requires only the non-GUI Redex core (`redex/reduction-semantics`); it does
> **not** `(require redex)`, which would pull in GUI libs.

### Scope of the Redex model

To stay small it uses an **opaque** `Cap` (no location in the type) and `Unit`
payloads. That already buys use-after-free / double-free / leak safety, because
those follow from **linearity alone**. The richer **location-indexed** capability
`Cap l T` — needed for *strong update* (type-changing in-place write) and
wrong-pointer safety — is the documented next step (`docs/02` §3, `docs/04`
Phase 2).

## Open: unifying the two

Stage A has the dependent multiplicity kernel but no heap; Stage B has the heap
and linear capabilities but no dependency. The research goal (`docs/04` Phase 2 &
4) is the single system: the Stage-B views become `0`/`1`-annotated entries in a
Stage-A QTT context, and `alloc/read/write/free` become kernel primitives typed
in λ-Tally. These two files are the scaffolding you grow toward that merge.

## Reproducing

- Python ≥ 3.8 (tested on 3.11). No dependencies.
- Racket ≥ 8 with `redex-lib` (tested on 8.10). No GUI required.
