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
| `Rig.agda`      | the multiplicity rig `R = {0,1,ω}`: `+`, `*`, the usage order `⊑`, and **proven** semiring + order laws; `⊑1→≡1` / `⊑0→≡0` formalize "1 = used exactly once", "0 = erased". | ✅ checks |
| `Context.agda`  | usage contexts as a **left module over the rig**: `+ᶜ`, `π ·ᶜ`, `𝟘`, with the module laws (identity, commutativity, associativity, distributivity) proven by lifting the rig laws pointwise — exactly the algebra every context-splitting step in the metatheory uses. | ✅ checks |
| `Syntax.agda`   | a de Bruijn calculus with the **quantitative typing judgment** `Φ ⊢[ Γ ] t ⦂ A`: the var rule uses `only`, application uses `Γ +ᶜ (π ·ᶜ Δ)`, the lambda checks the bound variable's usage against its budget with `⊑`. Worked derivations: a **linear identity** and a **K combinator with an erased (0#) argument** — linearity and erasure in one system, type-checked. | ✅ checks |
| `Semantics.agda`| call-by-value **operational semantics**: de Bruijn renaming/substitution, values, `_⟶_` (β + ξ rules) and its closure `_⟶*_`, with worked single- and multi-step reductions. | ✅ checks |
| `Progress.agda` | **progress** (half of type safety): every well-typed closed term is a value or steps. Holds for all such terms — the unbounded form of the Rosette "no stuck states" check. | ✅ checks |

Self-contained: `Rig.agda` uses a tiny inline prelude; `Context.agda` only
imports `Rig`. No standard library needed — `agda Rig.agda && agda Context.agda`
from this directory checks everything. (Tested on Agda 2.6.3.)

## Roadmap

- [x] **M1 Rig** — the semiring of multiplicities, laws, and usage order.
- [x] **M2 Context** — contexts as a module over the rig.
- [x] **M3 Syntax + typing** — a de Bruijn calculus with the quantitative typing
      judgment `Φ ⊢[ Γ ] t ⦂ A`, using `+ᶜ`/`·ᶜ`/`⊑` in the rules; linear and
      erased example derivations check.
- [x] **M4 Operational semantics** — call-by-value `_⟶_` with de Bruijn
      substitution; example reductions check.
- [x] **M5a Progress** — proven (`Progress.agda`).
- [ ] **M5b Preservation** — the remaining, genuinely hard piece. Attempting it
      surfaced two precise obstacles, both known and addressed in the literature
      (GraD; graded-Agda), confirming it is tractable but not a one-sitting job:
    1. **The quantitative substitution lemma** needs the *general* de Bruijn
       renaming + simultaneous-substitution infrastructure, because going under
       a binder turns a single substitution into `exts σ` (so single-variable
       induction does not close). The usage bookkeeping is a `Γ`-weighted sum of
       the substitution's per-variable usage contexts — the Wood–Atkey "matrix"
       view — which the `Context.agda` module laws (`·ᶜ-distrib`, `·ᶜ-assoc`,
       `+ᶜ-comm/assoc`) are exactly what discharge.
    2. **A subusaging design point.** Our lambda rule already uses `σ ⊑ π`, but
       the `var`/`app` rules pin the usage context *exactly*. After β the reduct
       is typeable at usage `Γ +ᶜ (σ ·ᶜ Δ) ⊑ Γ +ᶜ (π ·ᶜ Δ)`; closing the `⊑`
       gap (only the `ω` budget creates it — for `{0,1}` budgets `⊑1→≡1`/`⊑0→≡0`
       force equality, so the linear/erased fragment that matters for memory is
       already exact) needs the leaf rules to be subusaging-aware, the standard
       QTT reformulation.
- [ ] **M5 Resource soundness / type safety** — the *unbounded* proof of what
      `resource-soundness-rosette.rkt` checked to depth 4: well-typed ⟹ no
      use-after-free / double-free / leak. Plus erasure soundness (M-erasure),
      the unbounded form of the Rosette E2 result.
- [ ] **Beyond Rosette** — normalization and logical consistency for the
      dependent core (replacing `Type:Type` with a universe), and ultimately the
      memory primitives (`alloc/read/write/free`) typed in the quantitative
      kernel — the full λ-Tally.

Each milestone names a *theorem* (a checked `.agda` file), not just code.
