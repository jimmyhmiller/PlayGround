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
| `Progress.agda` | **progress** (half of type safety): every well-typed closed term is a value or steps. | ✅ checks |
| `Renaming.agda` | the weakening lemma: inserting an unused (`0#`) variable anywhere preserves typing (the infrastructure that pushes derivations under binders). | ✅ checks |
| `Substitution.agda` | the **quantitative substitution lemma**: a typed substitution applied to a well-typed term stays well-typed, at usage `mvp Γ Δs` (the Γ-weighted sum). The crux lemma. | ✅ checks |
| `Preservation.agda` | **preservation** (β via the substitution lemma; congruence cases recurse) and its multi-step closure. **Progress + preservation = type safety.** | ✅ checks |

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
- [x] **M5b Preservation** — **proven** (`Preservation.agda`), single- and
      multi-step. The quantitative substitution lemma (`Substitution.agda`) and
      the weakening lemma (`Renaming.agda`) are the supporting infrastructure.
      The `ω`-fragment subusaging gap noted earlier is sidestepped by reporting
      the reduct's usage context existentially (a reduct may use fewer resources,
      which is exactly right).
- [x] **M5 TYPE SAFETY** — progress + preservation, for all well-typed terms.
      **This is the unbounded, machine-checked counterpart of the Rosette E4
      resource-soundness result** (which was bounded to depth 4). The make-or-
      break milestone for the linear core: done.

### Remaining (the dependent layer and beyond)

- [ ] Dependent types: extend the calculus to full `Π`/`Σ` with a universe
      hierarchy (replacing the implicit `Type` story), re-proving the above.
- [ ] The memory primitives (`alloc/read/write/free`) typed in the quantitative
      kernel, with the resource-safety theorem — the full λ-Tally.
- [ ] Normalization and logical consistency (genuinely beyond Rosette's reach).

Each milestone names a *theorem* (a checked `.agda` file), not just code.
