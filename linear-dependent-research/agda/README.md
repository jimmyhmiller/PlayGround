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
| `Dependent.agda` | the **dependent** quantitative calculus: a universe, dependent `Π`/`Σ` with multiplicities, λ, app, pair, definitional conversion, and the **0-fragment** (type formation does not track usage). The `dep-id` derivation `λA.λx.x : Π(A:⁰⋆).Π(x:¹A).A` machine-checks the linear+dependent+erasure unification; `dep-sigma` a dependent `Σ`. | ✅ checks |
| `DepSubst.agda` | the **substitution algebra** for the dependent syntax (`ren`/`sub` compose; capstone `sub-comm`). Foundation for confluence and the dependent substitution lemma. | ✅ checks |
| `Confluence.agda` | **confluence** via Takahashi: parallel reduction closed under ren/sub, the complete development, the triangle lemma, and the **diamond property**. The new ingredient for dependent canonical forms. | ✅ checks |
| `MemSafety.agda` | the **memory-safety kernel**: the heap machine (alloc/read/write/resize/free/mkclaim/staleread incl. strong update), a store-typing **invariant**, and machine-checked **preservation** (invariant is inductive) + **safe access** (invariant ⟹ reads are in-bounds of live memory), giving safety for programs of **any length**. The Agda port of the Rosette E1 result — unbounded, **no SMT in the trusted base**. | ✅ checks |
| `Combined.agda` | the **combined** linear+memory calculus in one system: terms `var \| tt \| new \| use \| lt \| sq`, types `Un \| Cp` (a linear capability), the quantitative typing judgment, and a structural big-step heap evaluator whose `ok?` flag flips on a double-free / use-after-free. The "both in one system" artifact. | ✅ checks |
| `CombinedSound.agda` | **SOUNDNESS: well-typed ⟹ memory-safe.** For the canonical memory idiom (`var \| tt \| use \| nu \| sq`), a machine-checked proof that a **closed, well-typed program runs with NO error** (`rok ≡ true`: no double-free / use-after-free) **and leaves NO leak** (final heap has no live cell). A separation/frame argument keyed on the rig: the usage context tracks which live cells each subterm *owns*, and linearity (`1# + 1# = ω ⋢ 1#`) forces owned sets disjoint across a `sq` split. The unbounded, machine-checked counterpart of the Rosette E4 resource-soundness result — **no SMT, no postulates**. | ✅ checks |

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

- [x] **M6 Dependent calculus** — `Dependent.agda`: universe, dependent `Π`/`Σ`
      with multiplicities, λ/app/pair, definitional conversion, and the QTT
      0-fragment, **type-checked and inhabited** (`dep-id`, `dep-sigma`). The
      linear+dependent+erasure unification is now machine-checked in a genuinely
      dependent type theory, not just the simply-typed core.

### Remaining (the dependent metatheory and beyond)

- [~] **Dependent type safety** (progress + preservation for `Dependent.agda`).
      In progress, bottom-up:
    - [x] substitution algebra (`DepSubst.agda`);
    - [x] confluence's **diamond property** (`Confluence.agda`) — the hard new
          ingredient;
    - [ ] Church–Rosser (strip lemma from the diamond) and the convertibility
          characterisation (`Π`-injectivity, `⋆ ≇ Π` — standard boilerplate on
          the diamond);
    - [ ] the dependent substitution lemma (substitution into types; structurally
          the Module-5b proof on `Tm`, using `DepSubst`);
    - [ ] dependent preservation + progress (using `Π`-injectivity to invert the
          conversion rule).
- [~] The memory primitives in a **proven kernel**:
    - [x] operational memory safety (`MemSafety.agda`) — the store-typing
          invariant is inductive and implies safe access, unbounded, no SMT (the
          Agda port of Rosette E1; preservation + safe-access for the heap
          machine, strong update included);
    - [x] wire it to the term-level linear type checker (`well-typed ⟹ memory
          safe`): `CombinedSound.agda` proves a closed well-typed program of the
          combined calculus runs without error (no double-free / use-after-free)
          and leaks nothing — the resource-soundness link Rosette E4 showed
          bounded, now an **unbounded, machine-checked, postulate-free** theorem.
          (Proven for the `nu`/`use`/`sq` memory idiom; extending to the fully
          general `lt` needs the usage-relative injectivity invariant, sketched
          in the module — the only remaining gap is the cap-*moving* `lt`.)
    - [ ] fold the primitives into the **dependent** kernel — the full λ-Tally.
- [ ] **Consistency / normalization** (genuinely beyond Rosette's reach), and a
      universe hierarchy replacing `⋆ : ⋆`.

Each milestone names a *theorem* (a checked `.agda` file), not just code.
