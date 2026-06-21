# `tallyc` — the λ-Tally compiler (Rust + inkwell/LLVM)

A dependent + linear (quantitative) language with one surface, one kernel, and an
**erasing** lowering to LLVM IR — so types, indices, regions, and proofs cost
nothing at runtime. The novel part is the linear/permission checker; the rest is a
small, ordinary dependent-type compiler.

> **Where this is headed:** [`docs/FUTURE_WORK.md`](docs/FUTURE_WORK.md) — the
> north-star vision: as low-level as C with complete control over allocation, 100%
> memory-safe, Idris-level dependent types, and an opt-in **total** subset whose
> programs are provably terminating. (Aspirational; this README is what exists.)

```
rust_surface   lex → parse → elaborate            (the surface language)
     │  emits a checked dep::Term
     ▼
   dep          NbE kernel: eval / quote / conv + bidirectional QTT checker
     │  erase types / indices / proofs (multiplicity 0)
     ▼
dep_codegen     LLVM lowering (inkwell — Tally's native backend, always on)
```

CLI: `tally check <f>` (type-check), `tally run <f>` (check + JIT-run `main`),
`tally build <f> [-o out] [-O2]` (check + AOT to a native executable).

**`Nat` is a normal inductive, optimized like Idris 2.** Write the ordinary
`enum Nat { Zero, Succ }` and opt it into the packed machine-integer
representation with a pragma:

```
%builtin Nat Nat
enum Nat { Zero : Nat, Succ : Nat -> Nat }
fn main() { mul(1000, 1000) }   // literals, `+`, and `match`→native loop;
                                // normalizes on integers, never a unary blow-up
```

The pragma is validated (the type must be Nat-shaped), and without it an
`enum Nat` stays an ordinary unary datatype. Numeric literals and `+` are sugar at
the packed type.

**The memory layer is a built-in prelude.** `Own`, `alloc`, and `free` are
provided automatically (and given a real `malloc`/`free` lowering), so a program
needs no boilerplate:

```
fn main() { free(alloc(Zero)) }   // alloc + free, checked by linearity
```

## Status (v1.6 — one Rust surface, `%builtin Nat`, a memory prelude, AOT vs C)

The compiler was cut down to a single surface (`rust_surface`) over the QTT kernel
(`dep`) and the LLVM backend (`dep_codegen`); the old low-level/ML frontends are
gone. `tally build` is a real AOT compiler: it type-checks, lowers `main` to LLVM,
runs the standard `-O` pipeline, emits an object (`<out>.o` + `<out>.o.ll`), and
links a normal native executable with `cc` (run `main` once, print its result).

**Does the safe memory layer cost anything? No — measured against C** as *normal
programs* (see `bench/`, `bench/README.md`):

`examples/bench.tal` folds 1,000,000 transactions on the intrusive circular DLL
with O(1) remove-by-cursor (`new`/`insert`/`remove`/`free`), summing the value each
round-trips through the heap — written with `%builtin Nat` literals, no
scaffolding. `bench/bench.c` is the hand-written twin. At `-O2` **both** fold the
entire pure workload (2,000,000 `malloc`/`free` + all the list surgery) to the same
constant:

```
tally:  define i64 @tally_dep_main() { ret i64 499999500000 }
C:      main: … mov x8, #0x746a4ae6e0  (= 499999500000) ; printf
```

Same machine code: the dependently-typed, linearity-checked program and the
raw-pointer C are identical after erasure. (A workload whose allocations *escape*
keeps the `malloc`/`free` on both sides and still matches C — see the bench notes.)

**General recursion (`Fix`).** Recursion is a fold (an eliminator) when every
recursive call passes its non-scrutinee arguments through unchanged — those stay
total and reducible in types (so `mul(1000,1000)` never overflows). When a
recursive call *varies* a non-scrutinee argument, it is **general recursion**: it
lowers to a real recursive native function. So a binary tree with distinct
subtrees actually allocates 2^d nodes:

```
build : Nat -> Nat -> Tree
fn build(d, label) {
    match d {
        Zero    => Leaf,
        Succ(k) => Node(build(k, label + label), label, build(k, label + label + 1)),
    }
}
```

The kernel treats a `Fix` *opaquely* (it never unfolds it during type-checking),
so this stays decidable and compiles in milliseconds while doing all its work at
runtime. `bench/tree.c` is its twin: tally builds + traverses a 4.2M-node tree at
**parity with C** (ratio ≈ 1.05, identical RSS). Getting there fixed a real cost —
nullary constructors (`Leaf`/`Nil`) used to `malloc` a cell each, doubling the
allocations on a tree; they are now shared module-level constants (zero allocation),
matching C's NULL-for-leaf (see `bench/README.md`).

`cargo test`: **53 tests**, including `%builtin Nat` (literals,
`+`, `match`→native loop, no overflow at `mul(1000,1000)`), general recursion
(`Fix`) building distinct trees, the boxed-eliminator binder-order fix, the
elaborator regression, the memory prelude, and `aot_*_executable` (link + run).

## Status (v1.4 — GENERAL boxed datatypes run natively; the memory layer is real libc; erasure proven in the IR)

The native backend (`src/dep_codegen.rs`) is no longer
limited to `Nat`. **Every checked datatype now compiles and runs end to end:**

- **`Nat`-like families stay UNBOXED `i64`** (one nullary + one single-recursive
  constructor); their eliminator is a native counting loop.
- **Every other inductive family is a BOXED heap cell**: a constructor `malloc`s
  a block of `i64` slots — slot 0 is an integer constructor TAG, the remaining
  slots hold its **non-erased** arguments in declaration order. Its eliminator
  becomes a recursive native function that switches on the tag and recurses on
  the recursive fields (one induction hypothesis per recursive argument). So
  `Vec` and `Fin` run as real boxed data, not via the type checker's evaluator.
- **The linear memory layer runs on real libc `malloc`/`free`.** `alloc`/`free`
  lower to direct `@malloc`/`@free` calls; the intrusive **circular sentinel
  doubly-linked list** (`new`/`insert`/`remove`) is branch-free pointer surgery
  with an O(1) unlink-and-`free` by cursor — the same representation the older
  non-dependent backend (`src/codegen.rs`) used, now driven from the dependent
  core. The cursor IS the node pointer; the region/list identity is ghost.

```
$ tally run examples/vec_sum.run.tal   →  60      # boxed Vec, folded natively
$ tally run examples/fin_run.run.tal   →  2       # boxed Fin, native eliminator
$ tally run examples/dll.rs.tal        →  1       # insert, O(1) remove, free
$ tally run examples/memory.rs.tal     →  0       # alloc then free (libc)
```

**Erasure is the zero-overhead guarantee, and it is now *proven in the emitted
LLVM IR*** (`emit_ir`, exercised by `vec_ir_has_zero_overhead` and
`dll_ir_has_zero_overhead` in `src/dep_codegen.rs`). The tests read the actual
generated IR and assert:

- The Vec constructor's length index `{0 k : Nat}` is **never materialized**:
  each `Cons` cell is `malloc(24)` = 3 slots (tag, element, tail) — *not* 32
  bytes, which a stored index would require — and no length value is threaded
  into the eliminator (`vsum` over a 3-element vec contains no `store i64 3`, and
  its helper takes only the scrutinee, never a separate `n`). This last point was
  a real erasure leak the IR inspection surfaced and we fixed: the application
  β-reducer now reads each binder's multiplicity off the head function's `Π`
  telescope and **drops `Π[0]` arguments entirely** (never compiled, never
  stored), instead of compiling every argument.
- The DLL's ghost region machinery (`Region`/`R0`/`Cursor`) leaves **no trace**
  in the runtime IR — no symbol, no cell, no value. Every heap cell is 24 bytes
  (node/sentinel/linear-pair); there is no region cell or cursor-identity cell.
  `alloc`/`free` are direct `call ptr @malloc` / `call void @free`, with exactly
  two frees (the removed node, then the freed list). The only heap traffic is the
  actual data.

`cargo test`: **59 tests** (the 50 v1.3 tests plus boxed
`dependent_vec_sum_runs`/`dependent_vec_length_runs`/`dependent_vec_head_runs`/
`fin_to_nat_runs`/`fin_zero_runs`, the linear `linear_alloc_free_runs`/
`dependent_dll_remove_runs`, and the two IR zero-overhead tests).

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
  (erased). `cargo test`: 8 tests incl. `dependent_vec_runs`.
  Example: `examples/vec.tal`. (Matches `../agda/Dependent.agda`.)

## Status (v1.3 — END TO END: dependent source → native code → run)

The dependent front end and the native backend are now connected. `tally run
<file>` type-checks a program through the QTT kernel and
**compiles `main` to native code via LLVM, then JIT-executes it** — no
intermediate normalization, so the recursion runs in machine code:

```
$ tally run examples/nat.run.tal
examples/nat.run.tal: type-checks, compiled to native, ran → 14
```

`src/dep_codegen.rs` lowers a checked `dep::Term`: types/indices/proofs are
**erased** (they never reach a runtime position), applications are β-reduced, and
**each dependent eliminator becomes a real LLVM loop** (`elim z s n` ↦
`acc = z; for k in 0..n { acc = s k acc }`). So `add`/`mul`, defined by `match`
(which compiles to the eliminator), run as native loops — `(3+4)*2 = 14` is
computed by generated machine code, not by the type checker's evaluator.

Scope at v1.3 (first slice): `Nat`-like datatypes as `i64`. **General (boxed)
datatypes and linking the memory postulates to libc are now done — see v1.4
above.** `cargo test`: 50 tests at v1.3 (incl.
`nat_add_runs_natively`, `nat_mul_runs_natively`).

## Status (v1.2 — proofs-as-capabilities, and the intrusive DLL, in the core)

Phase 3 continues — two payoffs that need *both* dependency and linearity:

**Capabilities indexed by propositions** (`examples/proofs.rs.tal`). A memory op
can demand a compile-time *proof* — the thing neither Rust nor a plain linear
type system can express. With `LT m n` an inductive proposition (`m < n`):

```rust
postulate get : {0 a : Type} -> {0 n : Nat} -> {0 i : Nat}
             -> (0 _ : LT i n) -> Arr a n -> a       -- reading index i needs i < n

read1 : {0 a : Type} -> Arr a (Succ (Succ (Succ Zero))) -> a
fn read1(arr) { get(p13, arr) }                       -- p13 : LT 1 3 discharges the bound
```

The proof is erased (`0`), so it's free at runtime; and an out-of-bounds read
*cannot be written* because `LT 3 3` has no proof.

**The intrusive doubly-linked list with O(1) remove** (`examples/dll.rs.tal`) —
the headline application, in the core. `List r`/`Cursor r` are linear handles
into a ghost region `r`; `let (a, b) = e;` threads the linear state (it compiles
to a single eliminator on a *linear pair*):

```rust
client : {0 r : Region} -> (1 l0 : List r) -> List r
fn client(l0) {
    let (c, l1) = insert(l0, Succ(Zero));
    let (v, l2) = remove(c, l1);            -- O(1) remove by handle
    l2
}
```

The region index makes a cursor usable only on its own list, and linearity makes
the handles impossible to leak or use twice — so use-after-remove (`ω ⋢ 1`),
leaks (`0 ⋢ 1`), and cross-list removal (`s ≠ r`) are all type errors. This
needed one kernel fix: the eliminator's usage rule now consumes the scrutinee
*once* and fires a method `ω` times only for a *recursive* family — so a linear
pair can be destructured while a recursive eliminator stays conservative. 43 lib
tests.

## Status (v1.1 — the memory layer in the dependent+linear core, via postulates)

Phase 3 (the merge) has begun: the L3 memory primitives now live **inside** the
dependent+linear calculus as `postulate`s — opaque typed constants checked by
the same QTT core as everything else. Memory safety is not a separate analysis;
it falls out of **linearity**.

```rust
postulate Own   : Type -> Type
postulate alloc : {0 a : Type} -> a -> Own a
postulate free  : {0 a : Type} -> (1 o : Own a) -> Unit

main : Unit
fn main() { free(alloc(Zero)) }            -- accepted; `a` inferred at both calls
```

An owning `Own a` is taken at multiplicity `1` (exactly once), so the QTT checker
rejects the unsafe programs with no bespoke memory analysis:

| program | verdict | why |
|---|---|---|
| `fn consume(o) { free(o) }` | ✓ accepted | `o` used once |
| `fn leak(o) { U }` | ✗ rejected | `o` dropped — leak (`0 ⋢ 1`) |
| `fn dbl(o) { MkPair(free(o), free(o)) }` | ✗ rejected | use-after-free (`ω ⋢ 1`) |

The kernel re-checks the elaborated term (postulates added as `Term::Const` +
`Neutral::NConst`, looked up in the `Signature`), so the discipline isn't trusted
to the front end. Example: `examples/memory.rs.tal`. Next: capabilities
**indexed by propositions** (so a proof can be required to read/write), and
regions/`Ptr`/cursors (`docs/10` §10, the O(1) intrusive DLL).

## Status (v1.0 — Rust-flavored surface: `fn`/`enum`/`match`, ML types, implicits)

`src/rust_surface.rs` implements `../docs/10-surface-syntax.md`: the language
whose *types/signatures are ML-flavored* (juxtaposition `Vec a n`, arrow types,
no angle brackets) and whose *terms are Rust-flavored* (`fn`, `enum`, `struct`,
`match`). Run it with `tally lang <file>`:

```rust
enum Vec (a : Type) : Nat -> Type {
    Nil  : Vec a Zero,
    Cons : {0 k : Nat} -> a -> Vec a k -> Vec a (Succ k),
}

append : {0 a : Type} -> {0 m : Nat} -> {0 n : Nat} -> Vec a m -> Vec a n -> Vec a (add m n)
fn append(xs, ys) {
    match xs {
        Nil        => ys,
        Cons(h, t) => Cons(h, append(t, ys)),   // a, k inferred; recursion ↦ IH
    }
}
```

```
$ tally lang examples/vec.rs.tal
ok: examples/vec.rs.tal elaborates and type-checks (2 datatype(s), 3 def(s))
main = Cons Nat (Succ Zero) (Succ Zero) (Cons Nat Zero Zero (Nil Nat))
```

Two headline results:

- **`match` compiles to the kernel's dependent eliminator** (Phase 1). The
  motive is inferred from the `fn`'s declared return type (incl. the dependent
  case — `append`'s `Vec a (add m n)` becomes `λ m. λ _. Vec a (add m n)`),
  coverage is checked, and each *structural recursive call* is rewritten to the
  eliminator's induction hypothesis. So you write Rust-looking `fn`/`match` with
  ordinary recursion and get a **total-by-construction** term the kernel
  re-checks — elaboration is untrusted.
- **Implicit `{..}` arguments** (Phase 2). A brace binder is erased and
  *inferred*; datatype parameters are always inferred at constructor use sites.
  So `Cons(h, t)` and `FS(FZ)` carry no type or index arguments — they are
  solved by matching the constructor's result type against the expected type
  (first-order unification with holes). Above, `a`, `m`, `n`, and every per-cons
  length `k` are written nowhere.

Scope: `match` is one level of constructor patterns on a parameter; recursion
must be structural (non-structural is rejected). Implicit *constructor* and
*datatype* arguments are solved wherever an expected type is known (the common
case — `match` arm bodies and checked positions). `enum` (incl. GADT-style
indexed families), `struct` (records), `Eq`/`refl` proofs, and strict
positivity all work. Example: `examples/vec.rs.tal`. `cargo test`: 9
`rust_surface` tests (Nat/add, implicit indexed Vec/append, implicit Fin, a
struct, a proof, and the coverage/recursion/parse rejections).

**Implicit function-call arguments** are also solved: `append(xs, ys)` and
`fin_to_nat(x)` infer the callee's implicits from the *arguments'* types (and the
expected result type), via a small typing context built from the eliminator
method telescope. So `append(xs, ys)` at a call site writes neither the element
type nor either length.

Not yet: full nested/deep pattern matching and a real occurs/scope check in the
unifier. Next is the **merge** with the low-level memory layer (`docs/10` §10).

## Status (v0.9 — a SURFACE language: parser + elaborator)

You no longer write the kernel's de Bruijn `Term`s by hand. `src/surface.rs` is a
**named surface syntax** with a parser and an elaborator that compiles down to the
`dep.rs` kernel (which still does all type-checking). You write `data`
declarations, `def`s, `fun`/`->`/application, and a `match`-style `elim`; run it
with `tally dep <file>`:

```
data N where
  | z : N
  | s : N -> N

def add : (m : N) -> (n : N) -> N
  = fun m n => elim m to (fun _ => N) {
      | z      => n ;
      | s k ih => s ih
    }

data Vec (A : Type) : N -> Type where
  | nil  : Vec A z
  | cons : (0 k : N) -> A -> Vec A k -> Vec A (s k)

def append : (0 A : Type) -> (0 m : N) -> (0 n : N)
           -> Vec A m -> Vec A n -> Vec A (add m n)
  = fun A m n xs ys => elim xs to (fun k _ => Vec A (add k n)) {
      | nil           => ys ;
      | cons k h t ih => cons A (add k n) h ih
    }
```

```
$ tally dep examples/vec.dep.tal
ok: examples/vec.dep.tal elaborates and type-checks (2 datatype(s), 3 def(s))
main = cons N (s z) (s z) (cons N z (s (s z)) (nil N))    -- [1,2] : Vec N 2
```

- **Multiplicities** use Idris's QTT notation: `(0 x : A)` erased, `(1 x : A)`
  linear, plain `(x : A)` unrestricted; `data` params/indices default to erased.
- **`elim … to MOTIVE { | ctor binders => body … }`** is the eliminator with
  named binders: each branch names the constructor's arguments *and* its
  induction hypotheses (one per recursive argument). It elaborates directly to
  the kernel's dependent eliminator — recursion is *only* via `elim`, so every
  `def` is total by construction.
- Elaboration is purely syntactic (name resolution + eliminator assembly); the
  kernel re-checks everything, so a buggy front-end cannot make an ill-typed
  program pass. Strict-positivity violations, length mismatches, and ill-typed
  defs are all rejected. Examples: `examples/vec.dep.tal`, `examples/fin.dep.tal`.
- `cargo test`: surface tests for `N`/`add`, indexed `Vec`/`append`, `Fin`/
  `fin2nat`, a `refl` proof by computation, and the rejections (23 lib tests total).

Still pending: Idris-style dependent *pattern matching* (auto-elaborated to
eliminators, so you don't name IHs by hand), then the merge with the low-level
memory layer (`Own`/`Vec<n>` as declarations, capabilities indexed by
propositions), and a universe hierarchy.

## Status (v0.8 — GENERAL inductive families + dependent eliminators)

The QTT core (`src/dep.rs`) now has **general, user-declared inductive
families**, the gate to everything Idris-like. A `Signature` holds
strictly-positive datatype declarations (parameters, indices, constructors —
each constructor a telescope ending in `D params idxs`), and **every family
gets a dependent eliminator whose type is *computed from its constructors*** —
with an induction hypothesis inserted for each recursive argument. The
eliminator is the only recursion, so anything defined with it is **total by
construction** (docs/09 §1.3).

`Nat`, `Vec n`, and `Fin n` are now **ordinary declarations**, not built-ins:

```
data N    where z | s N
data Vec (A:Type) : (Nat) where vnil : Vec A 0
                                vcons : Π(n).A → Vec A n → Vec A (suc n)
data Fin : (Nat)  where fz : Fin (suc n) | fs : Fin n → Fin (suc n)
```

and are exercised through the *one* generic `elim`:

```
add    = λm.λn. elim[N]   (λ_.N)        n  (λk.λih. s ih)              m   -- 2+3 ↝ 5
append = λA.…   elim[Vec] (λk._.Vec A (add k n)) ys (λk h t ih. vcons A (add k n) h ih) xs
fin2nat= λn.λi. elim[Fin] (λm._.Nat)    (λm.0) (λm prev ih. suc ih)    i
```

`append Nat 2 1 [10,20] [30] : Vec Nat 3 ↝ [10,20,30]` — the length index `add
m n` is tracked in the type and only checks because the eliminator makes
`add` reduce definitionally. Non-strictly-positive declarations (e.g.
`mk : (Bad → Bad) → Bad`) are **rejected**. The mechanism: a `Signature`
threaded through NbE; `Data`/`Constr`/`Elim` terms; the eliminator type built
by de Bruijn `shift`/`subst` over the constructor telescopes; a generic ι-rule
that reads recursion off the constructor's declared argument types. `cargo
test`: 16 lib tests (incl. `nat_as_a_user_datatype_with_generic_elim`,
`append_via_generic_elim_tracks_length_and_computes`,
`fin_is_a_recursive_indexed_family`, `strict_positivity_is_enforced`).

Remaining Route-B steps: **(1)** a surface parser + elaborator (Idris-style
dependent pattern matching → eliminators; implicit/erased args via `Π[0]`);
**(2)** merge with the low-level layer so `Own`/`Vec<n>` are declarations and
capabilities are **indexed by propositions**; **(3)** a universe hierarchy for
logical consistency. (The hand-written built-in `Nat` with literals and `+` is
kept as a convenient index type; built-in `Vec` has been removed in favour of
the user-declared one.)

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
The dependent eliminator (induction) and length-indexed vectors that this
version introduced as *hand-written* built-ins are, as of **v0.8 above**,
subsumed by general user-declared inductive families — `Nat`/`Vec`/`Fin` are now
declarations exercised through one generic `elim`. (Built-in `Nat` with literals
and `+` is kept as a convenient index type.)

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
  insert/remove are branch-free pointer surgery. `cargo test`:
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

LLVM is Tally's native backend and is **always compiled in** — there is no LLVM-free
mode and no feature flag. A plain `cargo` invocation builds the whole compiler and runs
the whole suite (frontend + native backend together):

```
cargo test                  # the ONE suite — frontend + native backend (118 tests)
cargo run -- check <file.tal>   # type-check (dependent + linear, no leaks/use-after-free)
cargo run -- run   <file.tal>   # type-check + JIT-compile main to native, run it
cargo run -- build <file.tal>   # type-check + AOT-compile to a native executable
```

### LLVM prerequisite

Needs **LLVM 18** (inkwell is pinned to `llvm18-0`). `.cargo/config.toml` points
`LLVM_SYS_181_PREFIX` at the Homebrew `llvm@18` (`/opt/homebrew/opt/llvm@18`), so on the
dev machine `cargo build` / `cargo test` Just Works with no flags and no manual env.

- **macOS:** `brew install llvm@18` (the config's stable symlink path resolves to it).
- **Elsewhere** (e.g. a Linux CI/sandbox): install LLVM 18 dev libs and export the prefix
  — an externally-set `LLVM_SYS_181_PREFIX` OVERRIDES the config (it is non-forcing), e.g.

  ```
  apt-get install -y llvm-18-dev libpolly-18-dev libzstd-dev
  export LLVM_SYS_181_PREFIX=$(llvm-config-18 --prefix)
  ```
