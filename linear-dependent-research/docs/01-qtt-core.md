# 01 — Quantitative Type Theory, the core calculus (λ-Tally)

This is the keystone. It is the type theory that is **both** dependent **and**
linear, in which the dependent layer is **free at runtime**. We follow McBride
(*I Got Plenty o' Nuttin'*, 2016) and Atkey (*Syntax and Semantics of Quantitative
Type Theory*, LICS 2018), which is the theory shipped in Idris 2. Everything here
is the standard QTT machinery, specialized and annotated for our purposes; the
memory/view extensions are layered on in `docs/02`.

> Notation health warning: exact side-conditions on the rules below follow
> Atkey 2018. Where this doc paraphrases, the canonical statement is the paper;
> see `notes/bibliography.md`. The *structure* (rig-annotated contexts, the 0/1
> modal index, scaling/addition of contexts, the zeroing variable rule, the
> erasure result) is what matters and is reproduced faithfully.

## 1. The resource semiring (rig)

Multiplicities are drawn from a **rig** `R` = a *semiring without negation*:

- a set `R` with two operations `+` and `·`,
- `(R, +, 0)` a commutative monoid (addition, identity `0`),
- `(R, ·, 1)` a monoid (multiplication, identity `1`),
- `·` distributes over `+` on both sides,
- `0` annihilates: `0 · x = x · 0 = 0`.

Intuition: `+` combines *parallel* demands on the same variable (used in branch A
*and* branch... no — used in subterm A *and* subterm B); `·` composes *nested*
demand (a function used `π` times whose body uses its argument `ρ` times demands
`π · ρ` of the argument). `0` = no demand, `1` = unit demand.

### The starter rig: zero–one–many

`R = {0, 1, ω}` with:

```
+ | 0  1  ω        · | 0  1  ω
--+---------       --+---------
0 | 0  1  ω        0 | 0  0  0
1 | 1  ω  ω        1 | 0  1  ω
ω | ω  ω  ω        ω | 0  ω  ω
```

Read:
- **`0`** — *erased*. Used only in types / at compile time. **No runtime presence.**
- **`1`** — *linear*. Used exactly once at runtime.
- **`ω`** — *unrestricted*. Used any number of times (including zero).

Note `1 + 1 = ω`: using something twice lands you in "many". This rig is exact
enough to express linearity yet collapses everything past one use into `ω`.

We keep the rig **abstract** on purpose. Other useful instances:
- `{0, 1}` with `1 + 1` undefined → *strict linearity* (no weakening at all).
- The affine rig adding `≤1` (droppable-but-not-duplicable).
- Fractional/borrow extensions (§6, and `docs/02` §5).
The metatheory is parametric in `R`; choosing `R` tunes the discipline.

## 2. Multiplicity-annotated contexts

A context binds variables with both a type **and** a multiplicity:

```
Γ ::= ⋄ | Γ, x :^ρ A          (ρ ∈ R)
```

`x :^ρ A` reads "x is available with budget ρ, at type A". Two algebraic
operations on contexts (defined only when shapes/types match):

- **Scaling** `π · Γ`: multiply every annotation by `π`.
  `π · (Γ, x :^ρ A) = (π · Γ), x :^{π·ρ} A`.
- **Addition** `Γ₁ + Γ₂`: same variables and types, add the annotations
  pointwise. `(Γ, x :^ρ A) + (Γ, x :^σ A) = (Γ + Γ), x :^{ρ+σ} A`.

`0 · Γ` zeroes every budget — "a context where nothing may be used at runtime".
This is the *type-formation* context: types are checked where all runtime budgets
are `0`.

## 3. The judgment: a modal 0/1 index

The typing judgment carries an outer index `σ ∈ {0, 1}` (a sub-rig of `R`):

```
Γ ⊢ M :^σ A
```

- `σ = 1` ("present"): we are typing a term that **will run**; resource
  accounting in `Γ` is meaningful.
- `σ = 0` ("absent"/contemplation): we are typing a term in an **erased**
  position — a type, an index, a proof that is only inspected at compile time.
  At `σ = 0`, the rules force all budgets used to be `0`.

This single index is what lets *the same syntax* serve as both types and terms
while keeping the runtime/erased boundary crisp. The slogan: **whatever is only
ever needed at `σ = 0` is erased.**

## 4. Representative rules

Universe `U` (à la Russell or Tarski — orthogonal to our concerns; assume some
universe hierarchy `U₀ : U₁ : ...`).

### Variable (zeroing)

```
──────────────────────────────────────
0·Γ , x :^σ A , 0·Γ′  ⊢  x :^σ A
```

Using `x` spends exactly `σ` of `x` and **`0` of everything else** (the
surrounding context is zeroed). At `σ = 0` even the use of `x` costs `0`: that is
how a variable can appear in a type without being consumed.

### Π-formation (function types)

A function type records the multiplicity `π` its argument will be used with:

```
0·Γ ⊢ A :^0 U        0·Γ , x :^0 A ⊢ B :^0 U
──────────────────────────────────────────────
0·Γ ⊢ (x :^π A) → B :^0 U
```

Types are formed in a zeroed (`σ = 0`) context — forming a type uses nothing at
runtime. The bound `x` is available at multiplicity `0` inside `B`, so `B` may
*mention* `x` (dependency!) without "using" it.

### λ-introduction

```
Γ , x :^{σ·π} A ⊢ M :^σ B
─────────────────────────────────────
Γ ⊢ λ(x :^π A). M :^σ (x :^π A) → B
```

To build a function used at outer relevance `σ` whose argument has declared
multiplicity `π`, the body must be well-typed with `x` budgeted at `σ · π`. (At
`σ = 0` everything is `0`; at `σ = 1` the body gets `x` at exactly `π`.)

### Application

```
Γ ⊢ M :^σ (x :^π A) → B        Δ ⊢ N :^{σ·π?} A      (see note)
──────────────────────────────────────────────────────────────
Γ + (π · Δ) ⊢ M N :^σ B[N/x]
```

The caller's resources are `Γ` (for the function) **plus** `π · Δ` (the argument's
resources, scaled by how many times the function uses it). If `π = ω`, the
argument's resources are scaled to `ω` (it may be used freely inside). If `π = 0`,
`π · Δ = 0`, so an argument that is only used at the *type* level contributes **no**
runtime resource — this is the formal version of "erased arguments are free".

> Note on the argument's index: the argument is checked relevantly when it will
> actually be run (`σ = 1, π > 0`) and at `0` otherwise; Atkey's presentation
> handles this uniformly. The substitution `B[N/x]` is ordinary dependent
> substitution. Pin to the paper before mechanizing.

### Σ-types / tensor

Two flavors matter:

- **Multiplicative pair** `(x :^π A) ⊗ B` — a dependent tensor whose components'
  resources *add*. Eliminated by `let (x, y) = M in N`, which makes *both*
  components available. This is the linear Σ; it is how we package "a pointer
  **and** its view" (see `docs/02`).
- **Additive/with pair** `A & B` — offers *either* projection but you must choose;
  resources are *shared* (max/`+`-idempotent), not added. Useful for "a value that
  could be observed two incompatible ways, but only one will be".

```
Γ ⊢ M :^σ A      Δ ⊢ N :^σ B[M/x]
───────────────────────────────────────────
Γ + Δ ⊢ (M, N) :^σ (x :^π A) ⊗ B        (⊗-intro, resources add)
```

```
Γ ⊢ M :^σ (x :^π A) ⊗ B      Δ , x :^{σ·π} A , y :^σ B ⊢ N :^σ C
────────────────────────────────────────────────────────────────
Γ + Δ ⊢ let (x,y) = M in N :^σ C[M/ (x,y)]          (⊗-elim)
```

This is enough core to be going on with; identity types, inductive families, and
`Bool`/`Nat`/`Fin`/`Vec` are added the usual way, each constructor/eliminator
threaded with multiplicities. `Vec : (n :^0 Nat) → (A :^0 U) → U` — the length and
element type are `0` (erased indices), the *elements* are relevant.

## 5. Why this gives us what we want

### 5.1 Linearity *and* dependency coexist

A linear `x :^1 A` can appear in a type `B` via a `0`-use (the Π/Σ formation rules
run at `σ = 0`). Its single runtime use is still tracked separately. No conflict:
**type-level use and runtime use are different ledgers**, and `0`-uses are on the
free ledger.

### 5.2 Erasure (the runtime-cost theorem)

The property we are really buying:

> **Erasure (informal).** Define `|M|` = the term `M` with every `σ = 0`
> subterm (all type annotations, all `0`-multiplicity arguments, all indices and
> proofs used only at the type level) deleted. If `Γ ⊢ M :^1 A`, then `M` and
> `|M|` have the same runtime behavior, and `|M|` mentions no `0`-budget variable.

Consequences:
- `Vec n A` and a bare array have the **same** runtime representation; `n` is
  erased.
- A proof `p : x = y` passed to satisfy a precondition is erased — **proof-carrying
  code with no runtime proof**.
- Dependent typing therefore costs **zero** bytes and **zero** cycles, which is
  precondition #1 for "as low level as C". (Atkey proves the semantic version:
  the `0`-fragment is interpreted in a degenerate/erased way.)

### 5.3 The same `0` is the foundation of the memory model

In `docs/02`, *locations* `ℓ` are `0`-multiplicity (compile-time) indices, while
*views* `A @ ℓ` are `1`-multiplicity (linear) propositions. The 0/1 split we
already have is exactly the split between "the static name of a piece of memory"
and "the linear permission to touch it." We get the memory discipline by
*instantiating* QTT, not by bolting on a second system.

## 6. Open knobs (the parts still being chosen)

- **Which rig?** `{0,1,ω}` is the baseline. We almost certainly want affine `≤1`
  (so values can be *dropped*, i.e. RAII-style automatic destructors that *consume*
  the value). We are evaluating a rig that includes fractional permissions
  `q ∈ (0,1] ⊆ ℚ` for shared reads (Boyland), so the *same* multiplicity machinery
  carries borrow information. Risk: fractions complicate the decision procedure
  and the `+`/`·` tables. (T4 in `docs/00`.)
- **Subusaging / order on `R`.** A preorder `ρ ⊑ ρ′` ("having more budget than
  needed is fine") buys ergonomics (weakening) but interacts with strict
  linearity. Decide per-rig.
- **Definitional equality.** What conversion does the `σ = 0` checker use? This is
  the decidability fault line; see T3 and `docs/03`.
- **Universe story.** Cumulativity vs. not; not load-bearing for the linear/memory
  content, so deferred.

## 7. One worked micro-example (informal)

A safe "consume a file handle exactly once" signature:

```
open  : (path :^ω String) → File              -- File is a linear (1) resource type
read  : (h :^1 File) → (String ⊗ File)         -- consumes h, returns a fresh h
close : (h :^1 File) → 1                        -- consumes h, returns unit; no handle survives
```

Because `File` values are linear (`1`), the checker rejects "use after close"
(no `File` in scope after `close`) and "forget to close" (a leftover `1`-budget
`File` is an unspent linear resource = error), with **no runtime tracking**: the
discipline is entirely in the `+`/`·` accounting at `σ = 1`, erased at runtime.
The memory primitives in `docs/02` are this same pattern with `File` replaced by
the view `A @ ℓ`.
