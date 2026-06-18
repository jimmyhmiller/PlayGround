# 09 — Totality, and partial evaluation (staging)

Two things λ-Tally needs to spec out before it is a real dependent language, and
they are linked: **totality** (functions terminate and cover all cases) and
**partial evaluation** (compile-time reduction / staging). They are linked
because partial evaluation is only sound when reduction *terminates* and is
*confluent* — which is exactly what totality buys.

This doc fixes the design. It builds on the QTT core (`tallyc/src/dep.rs`:
NbE + multiplicities), confluence (`agda/Confluence.agda`), and the assurance
ladder (`docs/05`).

---

## 1. Totality

### 1.1 Why we cannot avoid it

A dependent language type-checks by **normalizing** (conversion calls `eval`).
So non-termination is not just a runtime bug — it breaks the *checker* and the
*logic*:

1. **Decidable type-checking.** Conversion (`a ≡ b`) reduces both sides to normal
   form. If reduction can diverge, type-checking can hang. (We saw this risk
   noted for `Type : Type` in `docs/07` §7.)
2. **Type-level computation.** `Vec (2+2)` must reduce to `Vec 4`. If the
   function computing an index loops, the *type* is meaningless.
3. **Logical soundness.** A non-terminating term `loop : ⊥` inhabits every type,
   so any proof becomes worthless. Partial matches do the same: a `head` that
   "can't happen" on `[]` is a hole in the logic.

So totality is a property of the **whole** language used in types/proofs — not
just a nicety.

### 1.2 The two obligations

- **Coverage** — every pattern match is exhaustive. With dependent matching,
  *impossible* cases are discharged by unification (e.g. `Vec 0` has no `cons`,
  so a match on a non-empty vector of statically-zero length needs no case).
- **Termination** — recursion is well-founded.

### 1.3 The load-bearing design choice: eliminators, not raw recursion

There are two ways to provide recursion:

- **General `fix` / pattern-matching + recursive calls** (Idris/Agda surface):
  ergonomic, but you must then *check* termination (a syntactic guardedness /
  structural-descent analysis) and coverage separately. These checkers are
  subtle and are where real proof assistants have had soundness bugs.
- **Eliminators / recursors** (Coq's `_rect`, our `agda/` development): each
  inductive family comes with *one* eliminator that recurses structurally by
  construction. **Totality and coverage are then automatic** — there is no way
  to write a non-covering or non-terminating function, because the only
  recursion available is the eliminator's.

**Decision for the λ-Tally kernel: eliminators are primitive; totality is by
construction.** The kernel exposes datatypes + their dependent eliminators only.
The *surface* language may offer Idris-style dependent pattern matching for
ergonomics, but it **elaborates down to eliminators** (à la Agda→core, Lean→`rec`),
and the elaborator runs a structural-recursion + coverage check whose *output is
an eliminator term the kernel re-checks*. The trusted base stays tiny: if the
elaborator is buggy, the kernel rejects its output.

This is the same move that keeps the kernel honest in Coq/Lean: clever, possibly
unsound elaboration on top of a small, total-by-construction core.

### 1.4 Datatypes: strict positivity

Inductive families must be **strictly positive** (the type being defined may not
occur to the left of an arrow in a constructor argument). Without it you can
encode `loop`/`⊥`. The datatype checker enforces this when a family is declared.

### 1.5 Interaction with multiplicities (the part specific to us)

Erasure does **not** remove the totality obligation. A multiplicity-`0` proof is
erased at runtime, but it is still **normalized during conversion** — so it must
still terminate. Totality applies to the 0-fragment too. (Linearity is
orthogonal: using something once says nothing about whether it terminates.)

### 1.6 Partiality as a deliberate, quarantined opt-out

C-level code genuinely has partial functions (an event loop, a `while(true)`,
hardware that may not respond). We keep them, *quarantined*:

- a `partial` function is **runtime-only**: it may be used in the `1`/`ω`
  (runtime) fragment, but is **barred from the 0-fragment** — it cannot appear
  in a type, an index, or a proof, and the checker will not reduce it during
  conversion.
- This is Idris's `partial`/`%default partial` discipline. It preserves the
  logic (no partial function ever participates in a proof) while keeping the
  language honest about real partial computation.

So totality is the default *for the type/proof world*; partiality is allowed but
fenced off from it. This matches the project's split: the 0-fragment is the
logic; the runtime fragment is the (optionally partial) program.

---

## 2. Partial evaluation / staging

### 2.1 The key observation: QTT multiplicities already give a staging discipline

A two-stage (compile-time / run-time) language marks which computations are
*static* (known now) vs *dynamic* (deferred to runtime). **We already have that
mark: the multiplicity.**

- `0` = erased = **static**: types, indices, proofs — computed at compile time,
  gone at runtime. This *is* the static stage.
- `1` / `ω` = **dynamic**: the residual program that runs.

So λ-Tally has a built-in notion of "what is known at compile time," for free,
from the resource discipline we already built.

### 2.2 NbE *is* the partial evaluator

Normalization by evaluation — the engine in `dep.rs` (`eval` to semantic values,
`quote` back) — is exactly partial evaluation:

- it **reduces** all the redexes it can (type-level computation, applications of
  known functions, erased arguments) — the *static* part;
- it **residualizes** the rest as *neutrals* (`VNeu`) — the *dynamic* part it
  can't reduce yet.

We did not build a separate partial evaluator; the dependent type-checker's
normalizer is one. The **erasure pass** that lowers to LLVM is then just "PE that
discards the 0-fragment and emits the residual runtime program."

### 2.3 What dependency buys: free specialization

Because indices and types are `0` (static), specializing code to them is free:

- a `Vec n` routine where `n` is a statically-known index can be **unrolled /
  specialized** at compile time — `n` is already being normalized away;
- generic (`Π[0]`-polymorphic) code can be **monomorphized** the same way;
- bounds (`Fin n`) known statically collapse to constants.

This is the "dependent types enable staging" point: the information you put in
types is, by construction, available to the compile-time stage.

### 2.4 Going further: explicit staging

The 0-fragment gives *implicit* staging (types/indices). For *runtime* values we
want sometimes to force at compile time (loop unrolling, table generation), add
an explicit two-level annotation — `static e` / quote-splice à la MetaML / Typed
Template Haskell / Idris elaborator-reflection. The normalizer already does the
work; `static` just says "reduce this dynamic subterm now and residualize the
result." (Design it so `static` cannot smuggle a `partial` function into
compile time — same quarantine as §1.6.)

### 2.5 Why PE is sound here — and the link back to totality

Partial evaluation must (a) terminate and (b) give the *same* answer as running
the program. Both rest on properties we have or have proved:

- **Termination** of the static stage = **totality** (§1). PE of the 0-fragment
  can't loop precisely because the 0-fragment is total.
- **Same answer / determinism** = **confluence**: the normal form is unique, so
  it doesn't matter which redexes PE contracts first. We proved the diamond
  property in `agda/Confluence.agda`.
- **Erasure correctness**: the 0-fragment doesn't affect runtime behaviour (the
  erasure theorem — the bounded version is the Rosette E-series; the linear-core
  version is in the Agda type-safety result).

So: **totality + confluence + erasure ⟹ partial evaluation is a sound,
terminating, semantics-preserving compile-time pass.** The two features are one
story.

---

## 3. The path to Idris-like data (Vec, Fin, Σ, …) and the merge

What exists today (`tallyc/src/dep.rs`): the QTT core with `Π`, `Σ` (dependent
pairs — `Pair`/`fst`/`snd`), a universe, `Eq`/`refl`, and `Nat`. To reach
Idris-style `Vec`/`Fin`/general datatypes:

1. **General inductive families + dependent eliminators** in the kernel
   (strictly positive; eliminator typing computed from the constructors). By
   §1.3 this gives totality + coverage for free. Then `Nat`, `Vec n`, `Fin n`,
   `Σ`, `Eq` are *library definitions*, not built-ins.
2. **A surface syntax + elaborator**: Idris-style implicit arguments (the `0`
   multiplicity already marks erased/implicit indices), dependent pattern
   matching elaborated to eliminators (§1.3), and unification for index solving
   (we already do a baby version for the `Vec<n>`/tag implicits in `check.rs`).
3. **The merge**: make this dependent QTT core the *single* front-end. The
   current low-level layer becomes definitions/postulates *inside* it —
   `Ptr`/`Own`/`View` as primitives with QTT signatures (the postulates already
   sketched in `prototype/lambda_tally_memory.py`), `Vec<n>` as the library
   `Vec`. Capabilities become **indexed by propositions**, so proofs constrain
   memory operations (the dependent+linear payoff).
4. **Codegen = erasure = PE** (§2.2): normalize/erase the 0-fragment, emit the
   residual via LLVM (the inkwell backend we have).
5. **Universe hierarchy** to replace `Type : Type` once we want consistency
   (`docs/07` §7).

Item 1 (inductive families + eliminators) is the gate to everything Idris-like
and is the recommended next build; it is also what makes the totality story
concrete (the eliminator is the only recursion, so it is total by construction).
