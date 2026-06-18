# 10 — Surface syntax: ML types, Rust terms

This doc fixes the **surface language** for λ-Tally: the concrete syntax you
actually write, and how it elaborates to the kernel (`tallyc/src/dep.rs`: the
QTT core — dependent types + linearity + general inductive families + dependent
eliminators). It is a *design spec*; the kernel and a first ML/Agda-flavored
surface (`tallyc/src/surface.rs`, v0.9) already exist, and this defines the
v1.0 surface that supersedes the latter.

The guiding principle, settled in design discussion:

> **Types and signatures are ML/Idris-flavored; terms and definitions are
> Rust-flavored.**

Concretely:

- **No angle brackets.** Type application is by **juxtaposition** (`Vec a n`),
  not `Vec<a, n>`. For a dependent language where types and values intermix,
  `<>` fights the parser (the `>>` / comparison ambiguity, turbofish) and does
  not carry value indices gracefully. Juxtaposition is both cleaner and *easier*
  to parse.
- **Rust terms.** `fn`, `enum`, `struct`, `match`, `let`, `{}` blocks, method
  calls. A programmer who knows Rust reads the *bodies* with no new concepts.

## Two decisions locked in

1. **Signature / body glue: a standalone ML signature line above a Rust body.**
   The type is written once, in full ML style; the `fn` only names the runtime
   arguments. This keeps the dependent type pristine and keeps implicits and
   multiplicities out of the term syntax.

   ```rust
   add : Nat -> Nat -> Nat
   fn add(m, n) {
       match m {
           Zero    => n,
           Succ(k) => Succ(add(k, n)),
       }
   }
   ```

   (The inline alternative — `fn append(xs: Vec a m, …) -> Vec a (m+n)` with ML
   type expressions in Rust binder slots — is rejected as primary because it
   handles implicits and multiplicities clumsily. It may return as optional
   sugar for simple non-dependent functions.)

2. **Linearity by *type*, with `0`/`1` overrides.** A type is declared linear
   (a `resource` / an owning capability like `Own`/`Cursor`); binders of a
   linear type default to multiplicity `1` (must be used exactly once), binders
   of an unrestricted type default to `ω`. You only annotate to override. This
   matches the Rust mental model ("owning a non-`Copy` value = you must move or
   destroy it") and our v0.2 `Own<S>`. Pure-QTT per-binder annotations
   (`(1 f : File)`) remain available for when you want them.

---

## 1. The type language (ML-style)

```
type   ::= arrow
arrow  ::= app  ('->' arrow)?            -- right-associative function arrow
         | '(' binder ')' '->' arrow     -- dependent explicit domain
         | '{' binder '}' '->' arrow     -- implicit domain (erased + inferred)
app    ::= atom atom*                    -- application by juxtaposition
atom   ::= NAME | 'Type' | '(' type ')'
binder ::= MULT? NAME+ ':' type          -- e.g.  0 a   |   1 x   |   m n : Nat
MULT   ::= '0' | '1'                     -- omitted ⇒ unrestricted (ω) / by-type
```

Examples:

```
Nat -> Nat -> Nat
(n : Nat) -> Fin n -> Nat
{0 a : Type} -> {0 m n : Nat} -> Vec a m -> Vec a n -> Vec a (m + n)
(1 f : File) -> Unit
```

- `a -> b` is the non-dependent arrow (sugar for `(_ : a) -> b`).
- `(x : A) -> B` binds `x` in `B` (a Π-type). Group like Idris: `(m n : Nat) -> …`.
- `{x : A} -> B` is an **implicit**: erased (multiplicity `0`) *and* solved by
  unification at the use site — the dependent/erased indices you don't want to
  write. `{0 a : Type}` is the common case; the `0` is implied by the braces and
  may be omitted (`{a : Type}`).
- `Type` is the universe (`Type : Type` today; a hierarchy later — `docs/07` §7).
- Type variables are lowercase (`a`), type/family names CamelCase (`Vec`).

`+`, and numeric literals on `Nat`, are ordinary total definitions usable in
types precisely because they are total (`docs/09` §1).

---

## 2. Multiplicities, erasure, the 0-fragment

The kernel rig is `{0, 1, ω}` (`tallyc/src/mult.rs`). In the surface:

| where | default | meaning |
|---|---|---|
| `{…}` implicit binder | `0` | erased; computed at compile time, gone at runtime |
| explicit binder of a **linear** type | `1` | must be used exactly once (no implicit drop) |
| explicit binder of an unrestricted type | `ω` | use freely |
| `0 x` / `1 x` annotation | as written | override |

The erased (`0`) fragment is the static / compile-time stage (the
types-indices-proofs world); `1`/`ω` is the residual runtime program (`docs/09`
§2: multiplicities *are* the staging discipline, NbE is the partial evaluator).

---

## 3. Data: `enum` and `struct`

### 3.1 `enum` — inductive families (incl. GADTs)

An `enum` is a (possibly indexed) inductive family. Parameters and indices form
an ML telescope after the name; each variant is given by **its type** (an ML
arrow chain ending in the family applied to its parameters and result indices —
the GADT form):

```rust
enum Nat {
    Zero : Nat,
    Succ : Nat -> Nat,
}

enum Vec (a : Type) (n : Nat) {
    Nil  : Vec a 0,
    Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (k + 1),
}

enum Fin (n : Nat) {
    FZ : {0 k : Nat} -> Fin (k + 1),
    FS : {0 k : Nat} -> Fin k -> Fin (k + 1),
}
```

Sugar for the common, non-indexed case (a plain Rust enum): `Zero` ≡ `Zero : D`,
and tuple/record variants `Cons(a, Vec a k)` desugar to the arrow type with the
result `D params` filled in.

Families must be **strictly positive** (the checker enforces it —
`tallyc/src/dep.rs::check_signature`); the family may not occur to the left of
an arrow in a constructor argument.

### 3.2 `struct` — records

A `struct` is a single-constructor data declaration (a labelled product; Σ under
the hood). It may carry parameters/indices too:

```rust
struct Pair (a : Type) (b : Type) { fst : a, snd : b }

struct Node (r : Region) {
    value : Int,
    next  : Ptr (Node r),
    prev  : Ptr (Node r),
}
```

A field type must be **copyable** (never a linear `Own`) — you cannot fabricate
a capability by reading memory (the L3 invariant; `docs/02`, and
`tallyc/src/check.rs`).

---

## 4. Functions: signature + body

A definition is an ML **signature** line plus a Rust `fn`. The `fn` binds the
**explicit** runtime arguments positionally to the signature's explicit domains;
implicits (`{…}`) are not listed (they are inferred), but may be brought into
scope with a `{…}` prefix when a body needs to name them:

```rust
append : {0 a : Type} -> {0 m n : Nat} -> Vec a m -> Vec a n -> Vec a (m + n)
fn append(xs, ys) {
    match xs {
        Nil        => ys,
        Cons(h, t) => Cons(h, append(t, ys)),
    }
}

-- naming an implicit explicitly, when needed:
fn append{a, m, n}(xs, ys) { … }
```

A bare `fn` with no separate signature is allowed only when the type is fully
inferable (no dependency) — otherwise the signature is required.

---

## 5. Pattern matching, recursion, totality

`match` + ordinary recursive calls is the surface; the **dependent eliminator**
is the compiled form. This is the ergonomic core of the language: you write Rust,
the elaborator produces a total-by-construction eliminator term that the kernel
re-checks.

```rust
fn add(m, n) {
    match m {
        Zero    => n,
        Succ(k) => Succ(add(k, n)),   -- structural recursive call
    }
}
```

The elaborator must, for a `match` on a value of inductive family `D`:

1. **Coverage** — every constructor of `D` has an arm (impossible cases under a
   refined index are discharged automatically; see below).
2. **Motive inference** — synthesize the eliminator's motive `P` from the `fn`'s
   declared return type (and the matched value).
3. **Index refinement** — in each arm, unify the scrutinee's index with the
   constructor's result index, learning equations (e.g. matching a `Vec a m`,
   the `Cons` arm learns `m = k + 1`; the `Nil` arm learns `m = 0`). This is the
   GADT part.
4. **Recursion → induction hypothesis** — a structural recursive call on a
   sub-component of the matched value (`add(k, …)` where `Succ(k)` is the
   pattern) is translated to the eliminator's IH argument for that recursive
   field. Calls that are *not* structurally decreasing are rejected (unless in a
   `partial fn`).

Totality is the default. Genuinely partial code (event loops, hardware waits) is
quarantined behind `partial fn`: runtime-only, barred from the `0`-fragment, and
never reduced during type conversion (`docs/09` §1.6).

---

## 6. Linearity and the memory layer

Linearity is *by type* (decision 2). A `resource` (or an owning capability) is
linear: its binders must be consumed exactly once, with **no implicit drop** —
forgetting one is a leak error; using one twice is a use-after-move error.

```rust
resource File          -- linear: must be consumed

close : File -> Unit   -- the `File` binder is linear by type ⇒ must be used once
fn use_file(f) { close(f) }
--   omit close(f)      ⇒ error: `f` leaked (linear value dropped)
--   close(f); read(f)  ⇒ error: use after move
```

The memory primitives reuse Rust's surface, reinterpreted (`docs/02`, the v0.2–
v0.5 implementation in `tallyc/src/check.rs`):

- **Lifetimes are regions** (Tofte–Talpin): `Node r`, `List r`, `Cursor r` are
  parameterized by a region `r : Region` (usually an implicit `{0 r : Region}`).
- **`Own t`** — a linear owning capability (zero-sized, erased at runtime); the
  right to read/write/free, held exactly once.
- **`Ptr t`** — a bare, copyable address carrying *no* permission (cannot be
  dereferenced without a matching capability). `next`/`prev` are `Ptr`s, so
  aliasing them is trivially sound.
- A **`Cursor r`** is a linear token into a region — the Vale `LinearKey` model
  (`docs/08`): consuming it on `remove` makes double-remove / use-after-remove
  type errors, and the region `r` stops cross-list removal.

The headline application — the intrusive doubly-linked list with **O(1)
remove**, fully safe, no GC — in the v1.0 surface:

```rust
struct Node (r : Region) { value : Int, next : Ptr (Node r), prev : Ptr (Node r) }

resource List   (r : Region)    -- a linear handle to the region
resource Cursor (r : Region)    -- a linear token for one node

new    : {0 r : Region} -> List r
insert : {0 r : Region} -> List r -> Int -> (Cursor r, List r)
remove : {0 r : Region} -> List r -> (1 c : Cursor r) -> (Int, List r)
free   : {0 r : Region} -> List r -> Unit

fn remove(list, c) { ... }   -- c consumed ⇒ O(1), no double-remove, no leak
```

---

## 7. Propositions, equality, proofs

Propositional equality is a type. In type position write `a == b` (or `Eq t a b`
when the type must be explicit); runtime decidable equality is a separate
`Eq`-style operation, so the two never clash. Proof obligations ride in `where`
clauses (like Rust trait bounds, but propositional); proofs are erased terms
(`refl`, lemmas), living in the `0`-fragment.

```rust
-- a proof discharged by the eliminator's own computation:
two_plus_two : (Nat.add 2 2) == 4
fn two_plus_two() { refl }

-- a where-obligation:
reverse : {0 a : Type} -> {0 n : Nat} -> Vec a n -> Vec a n
fn reverse(xs) where (n + 0) == n { ... }
```

---

## 8. Elaboration target (→ the kernel)

The surface elaborates to `tallyc/src/dep.rs` (the kernel re-checks everything —
elaboration is untrusted, so a buggy front-end cannot pass an ill-typed program):

| surface | kernel |
|---|---|
| `enum` / `struct` | `DataDecl` (params, indices, `Constructor`s) in the `Signature` |
| variant `C : telescope -> D params idxs` | `Constructor { args, idxs }` (decompose the Π-prefix) |
| `(x : A) -> B`, `{x : A} -> B` | `Term::Pi(π, A, B)` with `π = ω`/`0` |
| `fn` + signature | a checked def: a `Lam`-term of the signature type |
| `a b` | `Term::App` (or `Term::Data`/`Term::Constr` when the head is a family/ctor) |
| `match` | `Term::Elim(D, motive, methods, scrut)` — §5 |
| `a == b`, `refl` | `Term::Eq`, `Term::Refl` |
| implicit `{…}` arg | inserted erased argument, solved by unification |

The `enum`/`struct` → `Signature` and signature/`fn` → def paths, plus the type
language, are a direct extension of what v0.9 (`surface.rs`) already does.

---

## 9. Worked examples

```rust
enum Nat { Zero : Nat, Succ : Nat -> Nat }

add : Nat -> Nat -> Nat
fn add(m, n) {
    match m { Zero => n, Succ(k) => Succ(add(k, n)) }
}

enum Vec (a : Type) (n : Nat) {
    Nil  : Vec a 0,
    Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (k + 1),
}

append : {0 a : Type} -> {0 m n : Nat} -> Vec a m -> Vec a n -> Vec a (m + n)
fn append(xs, ys) {
    match xs {
        Nil        => ys,
        Cons(h, t) => Cons(h, append(t, ys)),
    }
}

enum Fin (n : Nat) {
    FZ : {0 k : Nat} -> Fin (k + 1),
    FS : {0 k : Nat} -> Fin k -> Fin (k + 1),
}

fin_to_nat : {0 n : Nat} -> Fin n -> Nat
fn fin_to_nat(i) {
    match i { FZ => Zero, FS(j) => Succ(fin_to_nat(j)) }
}
```

The same three families and functions are checked today through the v0.9 surface
and the kernel tests (`tallyc/src/dep/tests.rs`, `tallyc/src/surface/tests.rs`);
this is the Rust-flavored skin over them.

---

## 10. Implementation roadmap

- **Phase 1 — the front end (cheap; mostly a reskin).** Lexer + parser for the
  Rust+ML grammar above; elaborate the type language, `enum`/`struct` →
  `Signature`, signature + `fn` → defs. `match` compiles to `Elim` for the
  direct cases: one level of constructor patterns, motive from the return type,
  and structural self-recursion on the matched argument's fields → the IH. No
  implicit *inference* yet (write implicits explicitly, or fall back to the v0.9
  `elim` form where needed).
- **Phase 2 — full dependent pattern matching.** Nested/deep patterns, full
  coverage with absurd-case elimination, unification-based **index refinement**
  (the GADT learning), and **implicit argument inference** (extend the baby
  unifier already in `tallyc/src/check.rs`). This is the hard, high-value piece;
  it is orthogonal to the syntax and unchanged by dropping `<>`.
- **Phase 3 — the merge.** Fold the low-level memory layer (`Own`/`Ptr`/
  regions/cursors, today in `check.rs`) into this one front end, with
  capabilities **indexed by propositions** so proofs constrain memory operations
  (`docs/07` §6, `docs/09` §3) — the dependent + linear payoff.

---

## 11. Open questions / deferred

- **`==` spelling** — propositional `==` in type position vs a decidable `Eq`
  trait/op at the term level; resolved by position, but the lexical overlap may
  warrant a distinct token.
- **Linear `Drop`** — whether a linear `resource` may declare an automatic
  destructor (Vale's "Higher RAII" whitelisted destroyers, `docs/08`) or must
  always be consumed explicitly.
- **Universe hierarchy** — `Type : Type` is fine for a language, not for a logic
  (`docs/07` §7); needed before the proof fragment is trusted.
- **`where` proof search** — how much is automated (a tactic/decision procedure)
  vs supplied by hand.
