# Agda development — the machine-checked metatheory

This is the start of the proof-assistant phase (rung **(d)** on the `docs/05`
assurance ladder, the Agda handoff in `docs/06`). Where the Rosette models give
*bounded, automated evidence*, the Agda development gives *unbounded,
machine-checked proof*: theorems that hold for **all** terms, by induction.

Strategy (per `docs/05`): build bottom-up, cribbing structure from the
mechanized graded/quantitative developments (GraD; Abel–Danielsson–Eriksson's
graded-Agda), and reproduce — as real proofs — the properties the Rosette work
checked boundedly (resource soundness, erasure), then go beyond Rosette's ceiling
(normalization, consistency).

## Modules (each type-checks with `agda <File>.agda`)

| File | Contents | Status |
|------|----------|--------|
| `Rig.agda`     | the multiplicity rig `R = {0,1,ω}`: `+`, `*`, the usage order `⊑`, and **proven** semiring + order laws; `⊑1→≡1` / `⊑0→≡0` formalize "1 = used exactly once", "0 = erased". | ✅ checks |
| `Context.agda` | usage contexts as a **left module over the rig**: `+ᶜ`, `π ·ᶜ`, `𝟘`, with the module laws (identity, commutativity, associativity, distributivity) proven by lifting the rig laws pointwise — exactly the algebra every context-splitting step in the metatheory uses. | ✅ checks |

Self-contained: `Rig.agda` uses a tiny inline prelude; `Context.agda` only
imports `Rig`. No standard library needed — `agda Rig.agda && agda Context.agda`
from this directory checks everything. (Tested on Agda 2.6.3.)

## Roadmap

- [x] **M1 Rig** — the semiring of multiplicities, laws, and usage order.
- [x] **M2 Context** — contexts as a module over the rig.
- [ ] **M3 Syntax + typing** — an intrinsically (quantitatively) typed linear
      λ-calculus: terms indexed by type and usage context, so linearity holds
      *by construction*. Uses `+ᶜ`/`·ᶜ` from M2 in the App/binder rules.
- [ ] **M4 Operational semantics** — small-step (or an environment evaluator) +
      a resource heap, mirroring the Rosette model.
- [ ] **M5 Resource soundness / type safety** — the *unbounded* proof of what
      `resource-soundness-rosette.rkt` checked to depth 4: well-typed ⟹ no
      use-after-free / double-free / leak. Plus erasure soundness (M-erasure),
      the unbounded form of the Rosette E2 result.
- [ ] **Beyond Rosette** — normalization and logical consistency for the
      dependent core (replacing `Type:Type` with a universe), and ultimately the
      memory primitives (`alloc/read/write/free`) typed in the quantitative
      kernel — the full λ-Tally.

Each milestone names a *theorem* (a checked `.agda` file), not just code.
