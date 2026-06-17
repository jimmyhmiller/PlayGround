# Prototype: executable, semi-formal models

Three small, runnable models that make the formalisms in `../docs` concrete
enough to argue with. Stages A and B are deliberately split along the theory's
fault line — the *static* multiplicity discipline vs. the *operational* memory
story — and Stage C joins them.

| File | Tool | Models | Corresponds to |
|------|------|--------|----------------|
| `qtt_checker.py`         | Python 3 | the QTT **type checker** (multiplicities, dependent Pi/Sigma, NbE conversion, erasure) | `docs/01-qtt-core.md` |
| `memory-model.rkt`       | PLT Redex (Racket) | the **operational** memory semantics (heap + new/read/write/free) and a linear type system that rejects use-after-free / double-free / leaks | `docs/02-memory-views.md` |
| `lambda_tally_memory.py` | Python 3 | the **unification**: memory primitives as QTT functions, so multiplicities alone give memory safety + strong update | `docs/04` Phases 2 & 4 |

Why split A and B across two tools? The thing you most need to *understand*
(how linear and dependent typing cohabit, and how erasure works) is a
**type-checking** phenomenon — best felt by running check/infer and watching the
usage ledger. The thing you most need to *stress-test* (that the C-level memory
ops can't go wrong) is an **operational** phenomenon — best caught by an
executable reduction relation and property testing. Stage C then shows the two
disciplines are really one. See `../docs/04-roadmap.md` for how these feed the
eventual mechanized proofs.

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

## Stage C — `lambda_tally_memory.py`

```
python3 lambda_tally_memory.py
```

The unification. Stage A had the dependent multiplicity kernel but no heap;
Stage B had the heap and linear capabilities but no dependency. Stage C imports
Stage A's checker **unchanged** and adds the memory layer purely by *postulating
constants* — no new typing rules:

```
Loc  : Type                   -- locations: erased        (used at multiplicity 0)
Ptr  : Loc -> Type            -- pointers : unrestricted   (multiplicity w)
View : Type -> Loc -> Type    -- A @ l   : the LINEAR view (multiplicity 1)

alloc : (A:^0 Type)(i:^1 A) -> Sigma(l:^0 Loc). Sigma(p:^w Ptr l). View A l
read  : (l:^0 Loc)(A:^0 Type)(p:^w Ptr l)(v:^1 View A l) -> Sigma(_:^w A). View A l
write : (l:^0 Loc)(A:^0 Type)(B:^0 Type)(p:^w Ptr l)(v:^1 View A l)(n:^1 B)
                                                        -> Sigma(_:^w Unit). View B l
free  : (l:^0 Loc)(A:^0 Type)(p:^w Ptr l)(v:^1 View A l) -> Unit
```

The whole point: **the pointer/view split of L3/ATS is just two multiplicities
in a dependent tensor.** `p` is bound at `w` (copy it freely); the view `v` at
`1` (linear). Stage A's existing usage accounting then delivers, with no extra
machinery:

- `alloc … free`, and `alloc … read … free` — accepted;
- **strong update** — `write` returns `View B l` from `View A l`: the type stored
  at `l` changes (`Unit → Byte`) and `free`s at the new type. Sound *because* the
  view is linear, so no stale `View A l` can survive;
- **leak** (never free) — rejected: the linear view is dropped;
- **double-free** and **use-after-free** — rejected: the linear view is used
  twice.

This is the *static* half of the unified safety story; the *operational* half
(that the discipline keeps the heap consistent at runtime) is `memory-model.rkt`.
A simplification: `read` hands back its payload at `w` rather than tracking
copyability, and locations are abstract variables (location equality is
syntactic, no solver). Both are noted in the file and are the next things to
sharpen on the way to the mechanized development (`docs/04`).

## Reproducing

- Python ≥ 3.8 (tested on 3.11). No dependencies.
- Racket ≥ 8 with `redex-lib` (tested on 8.10). No GUI required.
