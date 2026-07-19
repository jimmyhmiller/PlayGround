# Absurd-case discharge (index-refinement coverage) — plan, for the maximal bar

**STATUS: SCOPED (design-first, soundness-critical) — the dogfood's dependent `eval` needs
it (total `Fin n → Vec Nat n → Nat` lookup). The Leader steered: build it; cardinal =
discharge ONLY genuinely-impossible cases, NEVER a reachable one (a wrong discharge = a
coverage hole = a runtime hole = unsoundness).**

## What's needed (surfaced by the dependent interpreter)

A total dependent lookup `lookup : Fin n → Vec Nat n → Nat` cannot be written today: every
formulation hits a `match` whose missing case is ABSURD but the coverage checker still
demands it:
- match `i` first → "missing a case for `Nil`" (`i = FZ` refines `n = Succ n'`, so
  `env : Vec Nat (Succ n')` — `Nil : Vec a Zero` is impossible, but not discharged);
- match `env` first → "missing a case for `FZ`" (`env = Nil` refines `n = Zero`, so
  `i : Fin Zero` is uninhabited — `match i {}` should be a sound-empty match, but `FZ` is
  demanded).

So the gap is **PARTIAL absurd discharge under INDEX REFINEMENT**: matching one indexed value
refines a shared index, which makes a *constructor of another value* (or of the same family
at the refined index) provably impossible. The existing `try_absurd_match` only handles the
ALL-absurd (empty) match, single-index, and uses a Nat sentinel.

## The for-all-T sentinel (the PREREQUISITE — the existing code's own caveat)

`try_absurd_match`'s Nat sentinel makes the kernel backstop **T-dependent**: a mis-classified
REACHABLE case would be silently accepted when the result type `T = Nat`. SOUND today only
because v1 fires solely when EVERY constructor is absurd (no reachable case to mis-classify).
PARTIAL discharge HAS reachable cases beside absurd ones, so the Nat sentinel is unsound here.

Fix = **EX FALSO**, which is T-INDEPENDENT and self-checking:
- a fresh uninhabited `enum Void {}` (no constructors) + `exfalso : {0 a : Type} -> Void -> a`
  (the Void eliminator, zero methods — `Elim("Void", λ_.a, [], v)`).
- an absurd case's method is `exfalso(contradiction) : T` where `contradiction : Void` is
  derived from the GENUINE index contradiction (NO-CONFUSION: distinct constructors are
  unequal — e.g. from `Zero = Succ n'`, transport `refl` along the discriminator
  `λx. NatCase x Unit (λ_. Void)` to get `Void`).
- WHY for-all-T + sound: `exfalso` yields ANY `T` (T-independent — no sentinel-vs-T backstop
  hole); and it type-checks ONLY if a real `Void` is derivable, i.e. ONLY if the index
  contradiction is genuine. If a REACHABLE case were mis-classified absurd, NO `Void` is
  derivable → the discharge FAILS TO TYPE-CHECK (the kernel re-check is the backstop, now
  T-independent). That is exactly the Leader's cardinal made structural.

## UPDATE — verify-before-building DE-RISKED this: NO J, NO general partial-discharge needed

Probing a J-free path: derive the `Void` from the EXISTING all-absurd discharge instead of
no-confusion/J. `fzv : Fin Zero -> Void` via `match i {}` (Fin Zero is empty — all ctors absurd
at index Zero) TYPE-CHECKS today (the all-absurd discharge, with `T = Void`). Then lookup is:

```
fn lookup(i, env) {
  match env {
    Nil           => exfalso(fzv(i)),                 -- env=Nil ⇒ n=Zero ⇒ i : Fin Zero ⇒ Void
    Cons(v, rest) => match i { FZ => v, FS(j) => lookup(j, rest) },  -- env=Cons ⇒ n=Succ k ⇒ i : Fin (Succ k)
  }
}
```

This needs NEITHER a `J`/`Eq`-eliminator (the kernel has `Eq`/`Refl` but NO eliminator —
confirmed) NOR a general partial-discharge: the `Void` is built from `fzv` (existing
all-absurd discharge) + `exfalso` (validated). **The ONLY missing piece is INDEX REFINEMENT:**
matching `env = Nil` must refine `n := Zero` in that arm so `i : Fin n` becomes `i : Fin Zero`
(making `fzv(i)` type-check), and `env = Cons(...)` must refine `n := Succ k` so `i : Fin
(Succ k)` (a normal `match i`). Today the elaborator does NOT refine the sibling's index and
PANICS on `lookup` (a bug — must become a clean error or, better, type-check via refinement).

So the build collapses to ONE substantive piece (plus a panic→clean-error fix): **index
refinement** — unify the scrutinee's index with each constructor's result index and substitute
into the arm context's types. That is the standard dependent-pattern-matching mechanism, far
smaller than the J + no-confusion + general-partial-discharge chain. (J / general partial
discharge remain future general capabilities, but the dogfood's dependent eval does NOT need
them — exactly the friction-driven minimalism.)

## The three pieces (original general plan — superseded for the dogfood by index-refinement)

1. **`Void` + `exfalso` + no-confusion** (`disjoint : C₁ … = C₂ … → Void` for distinct-head
   constructor indices) — the T-independent discharge primitive. Prelude-level.
2. **PARTIAL discharge in coverage** (`elab_case` / `elab_nested_match`): for each MISSING
   constructor case, test if it is ABSURD at the scrutinee's actual index (its result index
   decidably disjoint from the scrutinee's — reuse the `ctor_head` disjointness, which is
   already conservative/sound: KNOWN-different heads only, else "reachable"). If absurd,
   SYNTHESIZE its method via `exfalso`(no-confusion); if reachable, keep "missing a case".
3. **INDEX REFINEMENT across matches**: matching an indexed scrutinee must REFINE the shared
   index in the arm context (e.g. `i = FZ` ⇒ `n := Succ n'`), so a subsequent `match env`
   sees `env : Vec Nat (Succ n')` and can discharge `Nil`. This is dependent pattern-matching
   with index unification — the substantive part.

## SOUNDNESS CARDINAL + red-team (maximal bar — like the kernel terms)

- Discharge ONLY a constructor whose index PROVABLY cannot unify with the scrutinee's
  (decidable head-disjointness; conservative — unknown ⇒ NOT discharged ⇒ ordinary coverage).
- **Red-team: a program that SHOULD reach a "discharged" case must be UN-CONSTRUCTABLE.** I
  will write reachable-looking near-misses and confirm they are NOT discharged (the case is
  still demanded) — only the genuinely-impossible ones discharge. A wrongly-discharged
  reachable case = a value that hits a missing method = a runtime hole; the `exfalso`
  construction makes this a TYPE error (no `Void` derivable), and the red-team confirms it.
- The for-all-T `exfalso` removes the Nat-sentinel T-dependence (verify: the discharge works
  for a non-`Nat` result type, and a deliberately-wrong discharge is rejected for EVERY `T`).

## Payoff

Total `lookup`, then the dependent `eval : Vec Nat d → Own (Expr d) → Nat` RUNS end-to-end —
the complete dependent-half demo (scope-safety by typing AND the dependent interpreter
executing). Build order: `Void`/`exfalso`/no-confusion → partial discharge (reuse the
disjointness classifier) → index refinement → the running dependent interpreter. Bring each
for the maximal bar.
