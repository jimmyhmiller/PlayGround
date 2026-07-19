# Glossary

Terminology used consistently across the docs. Kept short; the docs are the
authoritative source.

- **Rig** — a *ri**N**g without **N**egation* (a semiring): `(R, +, ·, 0, 1)` with
  commutative `+`, distributive `·`, annihilating `0`. The set multiplicities come
  from. Baseline `R = {0, 1, ω}`. (`docs/01` §1.)

- **Multiplicity** — an element `ρ ∈ R` annotating a variable's *budget*: how many
  times it may be used. `0` erased, `1` linear (exactly once), `ω` unrestricted.

- **0-fragment / erased** — anything only ever used at multiplicity `0`. Lives at
  the type level; **has no runtime representation**. The reason dependent types
  cost nothing here. (`docs/01` §5.2.)

- **σ (the modal index)** — outer `{0,1}` index on the judgment `Γ ⊢ M :^σ A`:
  `1` = term will run (resources count), `0` = erased/type-formation position
  (budgets forced to `0`). (`docs/01` §3.)

- **Context scaling / addition** — `π · Γ` multiplies all budgets by `π`; `Γ₁ + Γ₂`
  adds them pointwise. The algebra that makes linearity compositional. (`docs/01`
  §2.)

- **QTT** — Quantitative Type Theory (McBride 2016, Atkey 2018): the dependent +
  linear type theory we use as the kernel. (`docs/01`.)

- **λ-Tally** — our working name for the core calculus = QTT specialized to our rig
  + the view primitives. (`docs/01`, `docs/02`.)

- **Location `ℓ`** — a *static* (multiplicity-`0`, erased) name for a piece of
  memory. Sort `Loc`. (`docs/02` §1.)

- **Pointer `Ptr ℓ`** — an ordinary runtime value (multiplicity `ω`, freely
  copyable) naming location `ℓ`. **Grants no access by itself.** (`docs/02` §1.)

- **View / capability `A @ ℓ`** — a **linear** (multiplicity-`1`) proposition: "a
  value of type `A` currently lives at `ℓ`". The *permission* to dereference `ℓ`.
  ATS calls these *views*; L3 calls them *capabilities*. (`docs/02` §2.)

- **Separating conjunction `∗`** — `V ∗ W`: the views `V` and `W` govern
  **disjoint** memory. From separation logic; forbids aliasing-unsoundness.

- **Strong update** — an in-place write that *changes the type* stored at a
  location (`A @ ℓ → B @ ℓ`). Sound here because the view is linear ⇒ no stale
  alias. (`docs/02` §3.)

- **Take vs. read** — `read` copies a value out and keeps the view (needs copyable
  `A`); `take` *moves* a (possibly linear) value out, leaving the slot `Hole`-typed
  until refilled. (`docs/02` §3.)

- **Region** — a lifetime-scoped batch of allocations freed together; a linear
  `Region r` capability. An **ω-region you never free = opt-in GC**. (`docs/02` §4.)

- **Fractional permission `q`** — a rational `q ∈ (0,1]` on a view: `q<1` ⇒
  read-only/shareable, `q=1` ⇒ writable/freeable. Split and recombine. The
  shared-read ergonomics knob (UNSETTLED). (`docs/02` §5.1.)

- **Borrow** — a view lent for a lexical scope and statically guaranteed returned;
  Rust's `&`/`&mut` reconstructed as sugar over views + fractions + lifetimes.
  (`docs/02` §5.2.)

- **Stratification** — splitting checking into a decidable **static index domain**
  (A), a fixed **kernel** (B), and a **programmable rule layer** (C). From ATS.
  (`docs/03` §1.)

- **Kernel** — the fixed, trusted core (λ-Tally + views) the soundness theorems are
  about. The TCB together with the blessed decision procedures. (`docs/03`.)

- **Rules-as-data / programmable layer** — user-declared judgments + Horn-clause
  rules run by a logic engine (the "Shen" goal), **fenced** so they must
  *elaborate to a kernel derivation* rather than assert new axioms. (`docs/03`
  §2,§4.)

- **Elaboration fence / conservativity** — the guarantee that anything accepted via
  user rules has a kernel derivation, so kernel safety transfers. A bad user rule
  fails to elaborate; it cannot forge soundness. (`docs/03` §4, `docs/04` T4.1.)

- **TCB** — Trusted Computing Base: kernel (B) + blessed static decision procedures
  (A). The programmable layer (C) is **outside** it.
