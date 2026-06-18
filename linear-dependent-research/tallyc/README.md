# `tallyc` — the λ-Tally compiler (Rust + inkwell/LLVM)

The real-compiler successor to the Python POC (`../poc/`). Same idea, native
codegen: the L3 address/permission split, a linear/permission checker (the novel
part), and an **erasing** lowering to LLVM IR — so permissions/regions/ghosts
cost nothing at runtime.

```
src/lexer  → parser → ast → check        (frontend: pure Rust, no deps)
                              │ erase
                              ▼
                            codegen (inkwell, behind `--features llvm`)
```

## Status (v0.4 — dependent indices + the 0-fragment)

- **Type-level dependency.** New types `Nat` and `Vec<n>` (a LINEAR
  length-indexed vector). The index `n` is a Nat term in normal form
  (`k` | `m` | `m + k`) with structural definitional equality.
- **Length arithmetic in the operations' types**, so safety is decided by the
  index: `vnew : Vec<0>`, `vpush : Vec<n> → Int → Vec<n+1>`, `vhead`/`vtail`
  require a statically non-empty `Vec<n+1>`, `vfree` requires `Vec<0>`.
  Pop-from-empty and free-of-nonempty are **type errors**.
- **The 0-fragment.** Multiplicity-0 parameters are ERASED and IMPLICIT: not
  passed at the call site, but solved by unifying argument types against
  parameter types (index unification). E.g.
  `fn push_zero(0 n: Nat, v: Vec<n>) -> Vec<n+1> { vpush(v, 0) }` — `n` is
  inferred per call; using an erased index at runtime is a type error.
- Backend: vectors lower to a linked stack; the length is never materialised
  (erased). `cargo test --features llvm`: 8 tests incl. `dependent_vec_runs`.
  Example: `examples/vec.tal`. (Matches `../agda/Dependent.agda`.)

## Status (v0.7 — the FUSED dependent+linear core: QTT, via NbE)

The Route-B core (`src/dep.rs`): a **Quantitative Type Theory** checker —
dependent types and linearity in one calculus. One syntax for terms *and* types,
a universe, `Π`/`λ`/application, a base `Nat`/`+`, and an identity type
`Eq A a b` with `refl`. Definitional equality is decided by **normalization by
evaluation**; and every `Π` carries a **multiplicity** `Π[π]` with the
bidirectional checker threading a **usage context** (combined by the rig's `+`
and `·`), so linearity is enforced *under dependency*:

```
λA. λx. x  :  Π[0](A:Type). Π[1](x:A). A     -- the polymorphic LINEAR identity
```
`A` is **erased** (used 0× at runtime — the 0-fragment — it appears only in the
type); `x` is **linear** (used exactly once). Misuse is caught by the rig:
using a `Π[1]` argument twice fails `ω ⋢ 1`, dropping it fails `0 ⋢ 1`, while a
`Π[ω]` argument may be used freely. And **proofs are terms checked by
computation**, now living in the 0-fragment:

```
refl : Eq Nat (2 + 2) 4        -- checks: 2+2 and 4 share a normal form
refl : Eq Nat (2 + 2) 5        -- REJECTED
```

This is the dependent + linear + erasure unification — the same as
`agda/Dependent.agda`'s `dep-id`, and a port of `../prototype/qtt_checker.py`.

**The dependent eliminator (induction), totality by construction.** `Nat` now
has its constructors `zero`/`suc` and the **dependent eliminator** `natElim`:

```
natElim P z s n : P n
  P : Nat → Type     z : P 0     s : Π(k:Nat). P k → P (suc k)
```

`natElim` is the *only* recursion, so every function written with it is **total
by construction** (docs/09 §1.3) — there is no `fix`, no way to write a
non-terminating or non-covering term. `add` is then a *definition*, not a
built-in: `add = λm.λn. natElim (λ_.Nat) n (λk.λr. suc r) m`. It **computes** by
the eliminator's ι-rules under NbE (`add 2 3 ↝ 5`), and proofs are discharged by
that user-defined computation:

```
refl : Eq Nat (add 2 3) 5        -- checks, via natElim's own reduction
```

`cargo test`: 12 tests (incl. `polymorphic_linear_identity`,
`linearity_is_enforced_under_dependency`, `natelim_is_a_total_recursor`,
`proof_by_user_defined_computation`).

Standalone for now (its own `Term` syntax). `Nat`+`natElim` is the template;
remaining Route-B steps: **(1)** **general inductive families** — generalise
the hardcoded `natElim` to *any* strictly-positive family with an eliminator
*typed from its constructors*, so `Vec n`/`Fin n`/`lseg` become user
declarations (this is the gate to everything Idris-like, and what makes the
totality story general); **(2)** a surface parser for the dependent syntax;
**(3)** merge with the low-level layer so `Own`/`Vec<n>` are definitions in the
calculus and capabilities are **indexed by propositions** (so proofs constrain
the memory operations); **(4)** a universe hierarchy for logical consistency
(`Type : Type` today is fine for a language, not for a logic —
`../docs/07-implementation-guide.md` §7).

## Status (v0.5 — linear cursors: the intrusive DLL with O(1) remove)

The headline application, **sound and native**. A `Cursor<'L>` is a LINEAR token
(the Vale `LinearKey` model) tagged with a type-level list identity `'L`. Pairs
(`Pair<A,B>`, `let (x, y) = e;`) thread results.

- `lnew : Lst<'L>` (fresh tag); `linsert : Lst<'L> → Int → Pair<Cursor<'L>, Lst<'L>>`;
  `lremove : Lst<'L> → Cursor<'L> → Pair<Int, Lst<'L>>`; `lfree : Lst<'L> → Unit`.
- **O(1) remove by handle** — the operation safe Rust cannot express — and it is
  sound: the cursor is linear, so `lremove` consumes it ⇒ **double-remove and
  use-after-remove are type errors**; the tag stops **cross-list** removal; an
  un-removed cursor (= a node) **leaks** ⇒ nothing is forgotten.
- Native: lowers to a real **circular sentinel doubly-linked list**, so
  insert/remove are branch-free pointer surgery. `cargo test --features llvm`:
  10 tests incl. `linear_cursor_dll_runs` (insert 3, remove the middle → 20).
  Example: `examples/dll.tal`. (Background: `../docs/08-prior-art-vale.md`.)

The trade-off (vs a *copyable* cursor) is that a linear cursor can't be freely
duplicated. A copyable-cursor + O(1)-remove needs per-node liveness tracking —
a symbolic region checker (what `../poc/tally_dll.py` does, sound for closed
programs) or region polymorphism + generational references (Vale's runtime
model). Those remain available as a separate "bounded-verification" mode.

## Status (v0.3 — functions + multiplicities)

- **Quantitative type system.** A multiplicity rig `{0,1,ω}` (`mult.rs`, the one
  from `../agda/Rig.agda`) and a **usage-context checker**: every binding has a
  multiplicity *budget*; the checker counts actual *usage* and requires
  `usage ⊑ budget`. Borrows (field reads, `addr`, write bases) cost 0; only
  moves/consumes spend budget.
- **Functions thread linear capabilities across call boundaries.** At a call,
  an argument's usage is *scaled* by the parameter's budget (the rig `·`). So:
  a function can take ownership (`fn consume(c: Own<C>)` must free or return it
  — leaking it is a type error), return ownership (`fn make() -> Own<C>`), and
  passing the same `Own` to two parameters, or to a `ω` parameter, is rejected.
- **Parameter multiplicities.** `fn f(1 n: Int)` (use exactly once), `0 n: Int`
  (erased), default `ω` for copyable / `1` for `Own`. Demonstrated end to end:
  `make(42)`/`consume(a)` type-checks **and JIT-compiles to native code → 42**.
- Examples: `examples/functions.tal` (accept), `examples/bad_leak_fn.tal`
  (reject: a linear capability leaks inside a function).

Next: the region/cursor discipline for the intrusive doubly-linked list with
O(1) remove, built on functions; then richer types / dependent indices.

## Status (v0.2 — type-directed)

- **Type system:** `struct` declarations and the L3 split *as types* —
  `Own<S>` (a LINEAR owning capability, must be used exactly once) vs `Ptr<S>`
  (an UNRESTRICTED, copyable bare address that **cannot** be dereferenced). The
  checker is type-directed and **structurally linear**; memory safety falls out
  of the type discipline. Key typing rules:
    - a struct field's type must be **copyable** (never `Own`) — you cannot
      fabricate a capability by reading memory;
    - `alloc S {..} : Own<S>`; `addr(x)` borrows `x:Own<S>` to a copyable
      `Ptr<S>`; `e.f` requires the base to be `Own<S>` (a `Ptr<S>` base is
      rejected — no capability);
    - an `Own` must be consumed once (free / move); dropping, re-using, or
      leaving it live at end of scope is an error.
  Rejects double-free, use-after-free, use-after-move, leaks, deref of a bare
  `Ptr`, type mismatches, missing/unknown fields, and linear struct fields. The
  discipline matches `../agda/CombinedSound.agda`.
- **Backend:** `codegen.rs` lowers a *checked* program to `tally_main() -> i64`
  and JITs it (types are erased: cells are `malloc`'d blocks, `free` is libc
  `free`). End-to-end test compiles
  `struct C{val:Int} let a=alloc C{val:0}; a.val=42; let r=a.val; free a; r;`
  to native code and gets `42`.
- **CLI / examples:** `tally check examples/twonodes.tal` (accept),
  `examples/bad_deref.tal` (reject, with a precise error).

Next: **functions with multiplicity-annotated parameters** (the usage-context
algebra, threading linear capabilities across calls), then the region/cursor
discipline for the intrusive doubly-linked list with O(1) remove.

## Build & test

Frontend only (no LLVM needed):

```
cargo test          # lexer/parser/checker tests
cargo run -- check <file.tal>
```

With the LLVM backend:

```
cargo test --features llvm
```

### LLVM backend prerequisites

Needs **LLVM 18** dev libraries. On this Ubuntu sandbox they are installed with:

```
apt-get install -y llvm-18-dev libpolly-18-dev libzstd-dev
export LLVM_SYS_181_PREFIX=$(llvm-config-18 --prefix)
```

The cloud sandbox ships `libLLVM.so` + `llvm-config-18` but not the `-dev`
headers/static libs, so add the three `apt` packages above (e.g. in a repo
setup script) before `cargo build --features llvm`. `inkwell` is pinned to the
`llvm18-0` feature.
