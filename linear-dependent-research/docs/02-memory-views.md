# 02 — The memory model: locations, pointers, views, strong update, free

This is where "as low level as C, but 100% safe" is actually paid for. We extend
λ-Tally (`docs/01`) with a memory discipline taken from **L3** (Ahmed, Fluet,
Morrisett — *A Linear Language with Locations*) and **ATS**'s *views*. The whole
trick is one idea, stated three ways:

> **Separate the *alias* from the *permission*.** A pointer is a freely-copyable
> *name* for a location. The *right to dereference* that location is a separate,
> **linear** object. Aliasing is cheap and unrestricted; access is single-threaded
> and tracked. Every dangerous C operation becomes safe because the permission is
> linear and there is therefore never a stale one.

This reuses QTT directly: *locations are `0`-multiplicity (erased) indices*;
*views are `1`-multiplicity (linear) propositions*. We add **no** new
substructural mechanism — we instantiate the one we already have.

## 1. Locations and pointers

- A **location** `ℓ` is a *static* index (sort `Loc`), multiplicity `0`. It is a
  compile-time name; it is **erased** and has no runtime footprint of its own.
- A **pointer** `Ptr ℓ` is an ordinary runtime value (multiplicity `ω` — copy it,
  store it, alias it freely) that carries the static location `ℓ` in its type.
  `Ptr` is *just an address*; holding one grants **no** access rights.

```
Ptr : (ℓ :^0 Loc) → U          -- pointers are unrestricted runtime values
```

Two pointers to the same place are *definitionally* the same location index; the
solver (see `docs/03`) reasons about location (in)equality.

## 2. Views (linear capabilities / separation assertions)

A **view** is a *linear proposition* describing the current contents of memory.
Views are the runtime-erased (`σ = 0` at the *logic* level) but linearly-tracked
(`1` at the *resource* level) heart of the system. The grammar of the view algebra:

```
V, W ::= A @ ℓ            -- "a value of type A is stored at location ℓ"
       | emp              -- "no memory" (unit of ∗)
       | V ∗ W            -- separating conjunction: V and W on DISJOINT memory
       | ∃ (ℓ :^0 Loc). V -- existential over a location (the address is abstract)
       | V ⊕ W            -- disjunction (e.g. tagged-union payloads)
       | ...              -- dataviews: user-defined linear memory shapes (ATS)
```

`∗` is **separating** conjunction (separation logic): `V ∗ W` asserts `V` and `W`
govern *non-overlapping* regions. This is exactly what forbids aliasing-induced
unsoundness. Views compose by `∗` and are consumed/produced linearly, so the rig
multiplicity on a view is `1`.

> Connection: a view `A @ ℓ` is the points-to `ℓ ↦ v : A` of separation logic;
> `∗` is the frame-able separating conjunction; threading views through the
> typing judgment *is* a Hoare/separation logic embedded in the type system. ATS
> calls these *views* (linear) and *props* (non-linear) and lets users declare
> their own via `dataview`/`dataprop`.

## 3. The primitives

All four are the *only* trusted memory operations; everything else is derived and
checked. Signatures (multiplicities shown; `1` = linear, `ω` = unrestricted,
`0` = erased index):

```
alloc : (A :^0 U) → (init :^1 A)
        → ∃(ℓ :^0 Loc). (Ptr ℓ  ⊗  A @ ℓ)
-- allocate, store init, hand back the pointer (ω) PLUS the linear view (1).

read  : (ℓ :^0 Loc) → (p :^ω Ptr ℓ) → (A @ ℓ)
        → (A  ⊗  A @ ℓ)                                   -- A must be ω (copyable)
-- borrow-read a COPYABLE value: returns the value and gives the view back.

take  : (ℓ :^0 Loc) → (p :^ω Ptr ℓ) → (A @ ℓ)
        → (A  ⊗  Hole @ ℓ)
-- move-read a LINEAR value out, leaving the slot typed `Hole` (must be refilled).

write : (ℓ :^0 Loc) → (p :^ω Ptr ℓ) → (B :^0 U)
        → (old : A @ ℓ) → (new :^1 B) → (B @ ℓ)
-- STRONG update: consumes `A @ ℓ`, stores `new`, returns `B @ ℓ`. Type at ℓ
-- may CHANGE from A to B.

free  : (ℓ :^0 Loc) → (p :^ω Ptr ℓ) → (A @ ℓ) → 1
-- consume the view; ℓ is now ungoverned. The pointer value may survive but is inert.
```

### Why each dangerous thing is now safe

- **Use-after-free.** `free` consumes `A @ ℓ`. After it, *no* `_ @ ℓ` exists in the
  context. `read`/`write`/`take`/`free` all *demand* a `_ @ ℓ`. So any access after
  `free` simply does not type-check. The `Ptr ℓ` you still hold is a dead name.
  Use-after-free is **untypeable**, not caught-at-runtime.
- **Double-free.** `free` needs `A @ ℓ`; it is consumed once; a second `free`
  needs another `A @ ℓ` that nobody can produce. Untypeable.
- **Strong update is sound.** `write` changes the stored type `A → B`. In C this
  is a footgun because other aliases still think it is an `A`. Here there is
  exactly **one** view (it is linear); after `write` it is `B @ ℓ`; no stale
  `A @ ℓ` can exist to be misused. So we get **type-changing in-place mutation**
  (e.g. initialize-in-place, in-place variant changes, arena bump-and-retag)
  *safely* — something Rust cannot express directly.
- **Out-of-bounds / null.** Bounds are dependent-type obligations on the *index*
  level (e.g. `read_at : (i :^0 Fin n) → ...` with a proof `i < n`), discharged by
  the static solver (`docs/03`). `Ptr ℓ` is non-null by construction; nullable
  pointers are `Maybe (Ptr ℓ)` or an explicit `NullablePtr` whose view-elimination
  forces a null check.
- **Leaks of linear resources.** A leftover `1`-budget view at the end of a scope
  is an unspent linear resource → type error. You must `free` it (or hand it back).
  Memory leaks of *tracked* memory are **untypeable**. (Untracked/`ω` arena memory
  is a deliberate, separate choice — see §4.)

### Move vs. borrow read

`read` requires `A : ω` (the value is copyable, so we can hand a copy out and keep
the view). `take` works for *any* `A` including linear ones: it *moves* the value
out, retyping the slot as `Hole` (an empty, must-be-filled cell). You then `write`
something back before you are allowed to `free` or let the view escape — this is
how you safely move a linear value through memory (e.g. `swap`, `replace`).

## 4. Regions (and where GC would live)

Pure per-location `free` is precise but tedious. **Regions** batch it:

- A region `r` is created with `newregion : 1 → ∃r. Region r`, a linear capability.
- Allocation *into* a region, `allocIn r`, produces views whose lifetime is tied
  to `r`. The region's view algebra tracks the *set* of live locations.
- `freeregion : Region r → 1` frees the whole region at once, consuming all views
  derived from it (the type system requires they have all been returned to `r`).

This is the Tofte–Talpin region idea, made linear. A **GC'd heap** is then *just a
region you never free*, whose allocations hand out `ω` (unrestricted, non-linear)
references — i.e. you opt *into* GC, per-region, as a library, by choosing `ω`
views with no `free`. "No mandatory GC" = the default region is linear; "GC when
you want it" = an `ω` region. This is the knob that lets the *same* language be
both an arena allocator and a managed runtime.

## 5. Borrowing and shared reads (the ergonomics layer) — UNSETTLED

Strict single-ownership of views is too rigid for everyday reading/aliasing. Two
complementary mechanisms, mirroring T4 in `docs/00`:

### 5.1 Fractional permissions (Boyland)

Generalize `A @ ℓ` to `A @ ℓ` carrying a fraction `q ∈ (0,1] ⊆ ℚ`:

- `q = 1`: full permission — may `write`/`free`.
- `0 < q < 1`: read-only — may `read`, may **not** `write`/`free`.
- Splitting: `A @[1] ℓ  ⟺  A @[½] ℓ  ∗  A @[½] ℓ` (and any `q = q₁ + q₂`).
- Recombination back to `1` re-grants write/free.

This lets many readers share, then merge back to a unique writer — *data-race
freedom for reads* drops out: a `write` needs `q = 1`, which is incompatible with
any outstanding read fraction existing. Fractions can be carried as `0`-multiplicity
(erased) static rationals, so they cost nothing at runtime. Cost: the static
solver must handle rational `+` constraints.

### 5.2 Second-class borrows (Rust-style)

A **borrow** is a view lent for a lexical scope and *statically guaranteed to be
returned*. Model: a scoped operator `borrow : (A @ ℓ) → (∀ lifetime α. (A @[α] ℓ)
→ R) → (R ⊗ A @ ℓ)` — you get a region-bounded fractional/branded view `A @[α] ℓ`
usable only within the continuation, and the full `A @ ℓ` comes back after. This
reconstructs `&`/`&mut` as *sugar* over views + fractions + lifetimes-as-regions.

**Status:** This is the least-settled part of the design. The open question is
whether fractions alone suffice or whether we need first-class lifetimes/brands as
well, and how much of this can be *user-defined* (the `docs/03` programmable layer)
versus kernel. Flagged loudly so we do not pretend it is solved.

## 6. What "expressing as much as C" concretely buys us here

| C idiom | Tally rendering | Safety mechanism |
|---|---|---|
| `malloc`/`free` | `alloc` / `free` | linear view consumed by `free` |
| `*p = x;` (same type) | `write` with `A = B` | linear view, no stale alias |
| `*p = x;` (retype, e.g. union) | `write` with `A ≠ B` (strong update) | linear view ⇒ no stale `A @ ℓ` |
| pointer aliasing | copy `Ptr ℓ` (`ω`) freely | `Ptr` grants no access; view does |
| `p[i]` | `read`/`write` + `Fin n` index proof | bounds discharged statically |
| arena / bump allocator | a `Region r` | batch `free`, views tied to `r` |
| `struct` layout, in-place init | view `∗`-composition + `take`/`write` | disjointness via `∗` |
| `realloc` / move | `take` then `write`/`alloc` | `Hole`-typed slot must be refilled |
| GC | an `ω` region you never free | opt-in, library-level |
| restrict / no-alias | unique (`q=1`) view | uniqueness is the default |
| `volatile`, MMIO | a primitive view family (future) | trusted, kernel-blessed leaf |

The bottom rows (`volatile`/MMIO, FFI to actual C) are the *trusted boundary*: a
small set of kernel-blessed primitive views, exactly like `alloc`/`read`/`write`/
`free` are trusted. Everything above them is derived and machine-checked.

## 7. Proof obligations this section creates (for `docs/04`)

1. **Heap typing & store soundness.** A runtime heap `H` (locations → values) is
   *well-typed* w.r.t. the multiset of live views. State and prove the invariant
   that the linear views in the context exactly partition the live heap (the `∗`
   structure ⇒ disjointness).
2. **Preservation across the four primitives.** Each of `alloc/read/take/write/
   free` maps a well-typed (context, heap) pair to a well-typed one.
3. **Memory-safety corollary.** In a well-typed configuration, every executed
   `read/write/take/free ℓ` has a live, correctly-typed cell at `ℓ`; no execution
   dereferences a freed or never-allocated location.
4. **No-leak corollary.** A program whose final context has no leftover `1`-budget
   views frees everything it allocates (for the linear/non-region fragment).
5. **Strong-update soundness lemma.** Because views are linear, `0·1` aliasing of
   `_ @ ℓ` is impossible; formalize "there is at most one view per live location"
   and use it to justify type-change.

These are the load-bearing theorems. `docs/04` orders them.
