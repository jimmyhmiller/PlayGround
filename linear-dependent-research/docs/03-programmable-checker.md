# 03 — The programmable, stratified checker (the "Shen" goal)

Two things must be true at once:

- **Decidability where it counts.** C-like code must compile without the compiler
  doing open-ended theorem proving. (Tension T3.)
- **Programmability.** Users can extend the type system with new judgments and
  rules — a session-typing discipline, units of measure, a custom effect/region
  system — *without forking the compiler* and *without being able to break
  soundness*. (Goal G5, Tension T5.)

Shen achieves programmability by making its type system a **sequent calculus whose
rules are user-supplied Horn clauses**, executed by an embedded logic engine; type
checking is (deliberately) Turing-complete. ATS achieves decidable dependency by
**stratifying** into a small decidable *static* index language plus a separate
*dynamic* program language. Tally wants **both**: ATS's stratification for the
decidable core, Shen's rules-as-data for the extensible layer — fenced so the
extensions cannot forge unsound derivations.

## 1. Three strata

```
   ┌─────────────────────────────────────────────────────────┐
   │  (C) PROGRAMMABLE RULE LAYER  — user judgments & rules    │  extensible, may
   │      (Horn clauses over judgments; elaborators/tactics)   │  be undecidable,
   │      MUST elaborate down to (B). Cannot add axioms.       │  fuel-bounded
   ├─────────────────────────────────────────────────────────┤
   │  (B) KERNEL  — λ-Tally + views (docs/01, docs/02)         │  fixed; the
   │      The trusted core. Soundness theorem is ABOUT this.   │  soundness target
   ├─────────────────────────────────────────────────────────┤
   │  (A) STATIC INDEX / CONSTRAINT DOMAIN                      │  small, DECIDABLE
   │      ints, Fin, locations, view-disjointness, fractions   │  decision procedure
   └─────────────────────────────────────────────────────────┘
```

### (A) The static domain — deliberately decidable

The indices that appear in everyday types — array lengths, `Fin n` bounds,
location (in)equality, the disjointness of `∗`-separated views, permission
fractions — live in a constraint language chosen to be **decidable and cheap**:

- linear integer arithmetic (Presburger) for sizes/bounds,
- equality/disequality over location names (a finite congruence-closure problem),
- a small lattice/heap-fragment for view disjointness (separation constraints),
- rational `+`/`≤` for fractional permissions (§5.1 of `docs/02`).

These are dischargeable by SMT-style decision procedures. ATS's experience is the
existence proof that "dependent types over a *constrained* index domain" compiles
fast and predictably. **The vast majority of safety obligations land here** and
never touch the term-level prover.

### (B) The kernel — the trusted core

λ-Tally (`docs/01`) + the view discipline (`docs/02`). Fixed. Small. The
type-safety, erasure, and memory-safety theorems are statements **about the
kernel**. Kernel checking is decidable: conversion/definitional equality uses a
terminating normalization (the design constraint on the kernel's equality is
exactly "stay decidable"; full undecidable equational reasoning is pushed up into
explicit proof terms in stratum C that the kernel only *re-checks*).

### (C) The programmable layer — rules as data

This is the Shen-shaped part. A user can declare:

- **new judgment forms** (e.g. `Γ ⊢ p : Session S` for a session-typed channel),
- **inference rules** for them, written as **Horn clauses over judgments**:
  `J₀  ⇐  J₁ , … , Jₙ , C` where `Jᵢ` are (object-level) judgments and `C` is a
  side-constraint in the static domain (A),
- **elaborations**: how a use of the new judgment **compiles to** a kernel (B)
  derivation.

The engine that runs these clauses is a logic/constraint solver (Prolog-/
datalog-/λProlog-shaped), exactly as in Shen. The difference from Shen — and the
thing that keeps us sound — is §4.

## 2. Judgments as a first-class datatype

Concretely, represent a judgment as data and the checker as a *relation* over it:

```
Judgment ::= HasType Ctx Term Mult Type      -- the kernel judgment Γ ⊢ M :^σ A
           | Discharge Ctx Constraint         -- the static solver (A)
           | <user-declared forms…>

Rule     ::= Judgment  ⇐  [Judgment]  ⊗  [Constraint]   -- a Horn clause
```

A **rulebase** is a set of `Rule`s. Checking a program = searching for a
derivation of its top-level `Judgment` against the rulebase, delegating
`Constraint`s to the decision procedure (A) and ultimately bottoming out in kernel
(B) rules. The kernel rules are themselves just the *built-in*, *immutable*
clauses of the rulebase; user rules are *added* clauses — but only of the
restricted, elaborating kind in §4.

This is the precise sense in which "the type system is a program you can edit":
the rulebase is data, and the checker is an interpreter for it.

## 3. The decidability / termination story

Stratum (A) is decidable by construction. Stratum (B) is decidable because the
kernel's conversion is terminating. Stratum (C) is where Turing-completeness
sneaks in (Horn-clause search can diverge), so:

- Every user-rule search runs under a **fuel budget**; exhaustion is a *compile
  error*, never a hang and never an unsoundness.
- We prefer rulebases that are **terminating by a checkable criterion** (e.g.
  structural decrease on a subject, stratified/negation-free datalog, a measure
  the user supplies). The fuel budget is the backstop for the rest.
- Result: the *programmable* layer may be as expressive as Shen's (you can encode
  hard checks), but a finished compile is always a *finite, replayable*
  derivation. "Programmable" never costs us "the compiler might loop."

So the spectrum from "fast, fully decidable" to "expressive, fuel-bounded" is a
**dial the programmer turns**, not a fixed point of the language: stay in (A)/(B)
for C-like code and instant compiles; reach into (C) for bespoke disciplines.

## 4. Soundness fence: extensions elaborate, they do not axiomatize

The danger (Tension T5): a user adds rule `R` and now `0 = 1` is derivable.
The fence:

> **A user rule may not be trusted as an axiom. It must be an *elaborator*: a
> procedure that, whenever it claims `Γ ⊢ M :^σ A`, produces a **kernel
> derivation** of `Γ ⊢ M′ :^σ A` that the kernel re-checks independently.**

So user rules are **conservative**: they can be arbitrarily clever, arbitrarily
buggy, or arbitrarily slow, and the worst that happens is they **fail to produce a
kernel derivation** (compile error). They cannot mint a derivation the kernel
would reject. This is precisely the LCF/"tactics vs. axioms" architecture, and the
proof-irrelevant-by-erasure (`docs/01` §5.2) story makes the re-checked proof
terms cost nothing at runtime.

Two flavors of extension, both fenced:

- **Derived (proof-producing).** The rule emits a kernel proof term. Always safe;
  the kernel is the judge. (Most extensions.)
- **Reflected (solver-backed).** The rule defers an obligation to a stratum-(A)
  decision procedure that is *itself* in the trusted base (like an SMT core).
  Adding a *new* trusted solver enlarges the TCB and needs the same scrutiny as
  changing the kernel — so this is privileged, not open to arbitrary users.

The TCB is therefore: kernel (B) + the blessed decision procedures (A). The entire
programmable layer (C) is **outside** the TCB. That is the whole point.

## 5. What this looks like in use (sketch)

A user wants binary session types for channels. They:

1. Declare a judgment `Γ ⊢ c chan S` (`S` a session protocol descriptor — an
   index in stratum A, since protocols are finite static data).
2. Give Horn-clause rules for `send`/`recv`/`close` that step `S` and thread a
   **linear** channel view (reusing the view machinery of `docs/02` — a channel is
   a linear resource, so this is *free*: linearity is already in the kernel).
3. Provide elaborations: `send` elaborates to a kernel `write`-like primitive on
   the channel's view; protocol progress is a stratum-A constraint.

Nothing about the kernel changed. Session-type safety is *derived* from
linearity + the static protocol index + an elaboration the kernel re-checks. If
their rules are buggy, programs using them fail to elaborate — the rest of the
language is unaffected. This is the Shen promise (extend the type system) with a
soundness guarantee Shen does not make.

## 6. Open questions

- **Exact static domain.** How much do we put in (A) before compile times suffer?
  Presburger + congruence closure + a separation fragment is the candidate; the
  separation/disjointness fragment is the least standard and needs design.
- **Conversion in the kernel.** Precisely which definitional equality keeps (B)
  decidable while being expressive enough that everyday code does not constantly
  drop into explicit proofs. (This is *the* classic dependent-type tuning problem.)
- **Elaboration trust surface.** The re-checking story is clean for proof-producing
  rules; for solver-backed rules we need a crisp account of what counts as a
  blessed solver and how its results enter the kernel (proof certificates? trusted
  oracle?). Certificate-producing solvers are the safer answer.
- **How "Shen" do we go on syntax?** Shen also gives you a Lisp and reader macros.
  Whether the *term* language is as malleable as the *rule* language is a separate
  decision from the type theory and is deferred.
