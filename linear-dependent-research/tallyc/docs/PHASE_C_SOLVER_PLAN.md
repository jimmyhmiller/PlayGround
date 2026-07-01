# Phase (A) — the decidable static constraint domain (stratum A of `docs/03`)

This is the first slice of Track B: the small, **decidable** index/constraint
domain that sits *underneath* the trusted kernel (`docs/03` stratum A). It is in
the TCB, so every rule it adds must be a *sound* decision — it may never equate two
semantically-distinct terms. It is deliberately explicit-first: it decides facts
the checker needs (definitional equality of indices, later bounds), it does **not**
do proof search / hole-filling. Inference is out of scope for now, by design.

## Where it plugs in

The kernel's definitional equality is `conv(a,b) = quote(a) == quote(b)` — purely
structural on NbE normal forms (`src/dep.rs`). So closed arithmetic folds
(`2 + 3 ↦ 5`), but **open** index arithmetic does not: `n + m` and `m + n` quote to
different terms, as do `n + 0` and `n`. Everyday length-indexed code
(`Vec (m + n)` vs `Vec (n + m)`) therefore fails to type-check with no way out
short of hand-written rewrite proofs. Closing that is the job of stratum A.

## Slice 1 — linear-Nat equality (DONE)

`src/solver.rs`: a canonical form for the linear-arithmetic fragment of `Term`.

- **Domain.** `Zero`, `NatLit`, `Suc`, `Add` are the arithmetic constructors;
  every other `Term` is an opaque **atom** (its internals are still canonicalized
  recursively, so `f (n + 0)` and `f n` share an atom). A term canonicalizes to a
  `LinNat` = `const : u64` + a sorted multiset of `coeff · atom`.
- **`canon : Term -> Term`** rewrites each maximal arithmetic subterm to a
  canonical shape and leaves all other structure intact.
- **Wiring.** `conv` becomes `canon(quote a) == canon(quote b)`.
- **Soundness.** Every identity used — commutativity, associativity, `+0`, and
  `Suc n = n + 1` — is *true* for `Nat` `+`, and canonicalization only fires on
  genuinely arithmetic nodes. So `conv` gets strictly coarser only by *valid*
  equalities: it never accepts a program the intended semantics rejects. It is
  also *complete* for the linear fragment (equal linear combinations ⇒ identical
  canonical form). This is exactly the decidable (A)-domain of `docs/03`.

## Slice 2 — surface: `+` in type/index position (DONE)

The surface parsed `+` only in term position; `Eq Nat (n + m) (m + n)` was a parse
error. Added `Ty::Add`, an infix parse in index position, and its elaboration to
`Term::Add`, so the equalities Slice 1 decides are actually writable.

## Slice 3 — inequality decision (DONE)

`m ≤ n` / `m < n` decided over the same linear-Nat normal form
(`solver::diff_witness`), exposed EXPLICITLY and proof-producing:

- **Propositions.** `Le a b` / `Lt a b` are built-in surface type-constructors
  encoded existentially — `Le a b := Σ (0 d : Nat). Eq Nat (a + d) b`,
  `Lt a b := Σ (0 d : Nat). Eq Nat (Succ a + d) b`. No new kernel primitive: it is
  `Σ` + the identity type, and its equation is decided by Slice 1's `canon`.
- **Discharge.** `le(a, b)` / `lt(a, b)` run the decision procedure, compute the
  witness `d = b − a` (or `b − (a+1)`), and emit the proof `(d, refl)`. This is the
  LCF discipline of `docs/03` §4: the solver is an *elaborator* whose output the
  kernel re-checks; a solver bug can only fail to compile, never mint an unsound
  proof. An unprovable bound is a hard error (no silent stub).
- **Strictly more than the inductive `LT`.** Because the witness is a linear
  combination, OPEN bounds over variables (`n ≤ n + m`) are dischargeable — the
  inductive `LT` of `proofs.rs.tal` cannot build those (you cannot case-split a
  variable). `examples/bounds_dec.tal` gates an array read on a decided `i < n` and
  shows the out-of-bounds variant rejected.
- **Built-ins yield to user names.** `Le`/`Lt`/`le`/`lt` resolve only as a fallback
  after datatypes/constructors/defs, so a user `enum Lt` or `fn lt` still wins.
- *Known limitation:* the implicit solver does not yet infer a `get`'s `{0 i}`/`{0
  n}` THROUGH the Σ-encoded proof, so those bounds are passed explicitly for now
  (consistent with the explicit-first scope).

## Next slices (not yet built)

- **Location (in)equality** — finite congruence closure over `Ptr ℓ` names (needed
  by the view layer, `docs/02`).
- **View disjointness** and **fractional permissions** — the least-standard
  fragments; design-first when the view layer proper begins.
