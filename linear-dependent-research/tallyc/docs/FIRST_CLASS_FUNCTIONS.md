# First-Class Functions — closures are *data*, and you choose the representation

Status: **P1 implemented** (the callable-function-value primitive — §2, §11);
remaining phases are proposal (open points in §13). See §11 for what runs today.

## 0. The requirement

> **Full freedom and choice, but very very explicit.**

Unpacked into what the compiler may and may not do:

- **Freedom** — every closure representation is available: no capture, flat value
  captures, heap captures, region/pool captures, borrowed captures, hand-rolled
  defunctionalisation. You are never boxed into one strategy.
- **Explicit** — *you* choose the representation; it is written in the **type**; and
  every allocation is a call *you* make. The compiler never picks a representation,
  never inserts a `malloc`, never inserts a `free`, never does escape analysis,
  never GCs or refcounts. If a closure lands on the heap, that word is in your
  source.

These two together rule out most languages' answers (§9, §12) and point at exactly
one design, which turns out to need almost no new language.

## 1. The thesis (and why it's basically already true)

tallyc already has a **complete, explicit vocabulary for where data lives**: flat
values by value, `Own e` (heap, linear, `free`d by hand), `Pool`/`Region` (arena),
`Slice`/borrows (a view into memory you don't own), and plain `enum`s (tagged
unions). A **closure is just an environment plus closed code** — and an environment
is *data*. So a closure is *data*, and it inherits that whole vocabulary for free.

There is nothing to invent at the type level. This all type-checks **today**:

```
struct FlatClo (e : Type) (a : Type) (b : Type) { env : e, code : e -> a -> b }

applyFlat : {0 e : Type} -> {0 a : Type} -> {0 b : Type} -> FlatClo e a b -> a -> b
fn applyFlat(c, x) { match c { FlatClo(env, code) => code(env, x) } }   -- ✓ total

addK : Nat -> Nat -> Nat
fn addK(k, x) { x + k }
add5 : FlatClo Nat Nat Nat
fn add5() { FlatClo(5, addK) }                                          -- ✓ total

fn main() { applyFlat(add5, 10) }                                       -- ✓ checks → 15
```

The existential heap closure type-checks too (at `Type 1`, the universe cost of
hiding the environment type — §3):

```
boxed enum OwnClo (a : Type) (b : Type) : Type 1 {
  MkOwnClo : {0 e : Type} -> (1 env : Own e) -> (code : e -> a -> b) -> OwnClo a b
}                                                                        -- ✓ checks
```

The **only** thing that does not work yet is *running* `code(env, x)` — calling a
value of function type in native code. That single primitive (§2) is the entire
implementation gap. Everything else in this document is library and sugar.

This is the payoff of the requirement: because we refuse a built-in closure
abstraction, closures reduce to "data + one primitive," and the data half is done.

## 2. The one primitive: a closed function is a callable value

A **closed** function — a top-level `fn`, or a lambda lifted so it captures nothing
— is a value whose representation is a single code address (`i64`). The new
capability is: **hold one as a value, and apply a value of function type** (an
indirect `call`). No allocation, ever. Two forms, both zero-alloc:

```
apply : {0 a : Type} -> {0 b : Type} -> (a -> b) -> a -> b
fn apply(f, x) { f(x) }        -- f is a code pointer; f(x) is an indirect call
```

This bottoms out the whole design: every closure's `code` field is a closed
function value, so once *this* is callable, `applyFlat`, `OwnClo`, defunctionalised
tables, etc. all run. There is no regress — captures live in the *environment* data,
never in the code pointer.

## 3. The representation family (the "full freedom" catalog)

Each row is an ordinary tallyc type you (or the stdlib) define — no privileged
default. You pick a row by naming its type; the type states the allocation.

| Representation | Type (library-level) | Env lives in | Heap? | Freed by | Dispatch |
|---|---|---|---|---|---|
| **bare fn** | `a -> b` | — (no captures) | no | — | (in)direct call |
| **flat** | `struct FlatClo e a b { env : e, code : e->a->b }` | inline value (regs/stack) | **no** | — (it's a value) | destructure + call |
| **owned** | `boxed enum OwnClo a b : Type 1 { … Own e … }` | heap | **yes — your `alloc`** | you (`freeC`) | unpack + call |
| **region** | env is an `RPtr r` into a `Pool r`/`Region` | arena | yes — region | region release | `pget` + call |
| **borrowed** | env is a `Slice`/`SRead` view (linear loan) | someone else's memory | no | lender, when loan ends | read + call |
| **defunctionalised** | your `enum Fns { … }` + `fn applyFns(f, x){ match f … }` | inline tag union | **no** | — (it's a value) | `match` on tag |

Notes that matter:

- **flat** is the default you'll reach for: zero heap, C-struct-by-value, and — via
  monomorphisation (§5) — the callback in `map(inc, xs)` compiles to inlined code
  with *no closure value at all*.
- **owned** is the only row that heap-allocates, and it does so because *you wrote
  `alloc`*. It is **linear** (holds `Own e`) so leak / double-free / use-after-free
  are compile errors, and you `freeC` it once. It costs a universe (`Type 1`)
  because it hides the environment type — that is the honest price of a uniform,
  heterogeneous closure (`List (OwnClo a b)`), and you only pay it when you ask for
  one.
- **defunctionalised** is Roc/Reynolds "lambda sets" done **by hand**: an `enum` of
  your closures + an `apply` `match`. It is already how tallyc's own interpreters
  are written. No boxing, no vtable, dispatch is a switch. If you want R3, you write
  R3 — explicitly, and only where you want it.

You can define more (a closure whose env is a `Vec`, a bump-pointer, an mmap'd
region…). The language does not enumerate these; the **memory vocabulary** does,
and it is open.

## 4. The allocation contract, exactly

| You write | Representation | Heap alloc? | Who allocated |
|---|---|---|---|
| a top-level `fn` as a value | code ptr | no | — |
| a capture-free `\x. e` | code ptr | no | — |
| `FlatClo(env, code)` / capturing `\[flat] x. e` | flat value | **no** | — |
| capture where a field is an `Own` you made | flat value holding a ptr | no *new* alloc | you, earlier |
| `box … : OwnClo a b` / `\[own] x. e` | 2-word `{code, envptr}` | **yes** | **you**, at that `alloc` |
| a `Pool`/region-backed closure | arena slot | yes | you, at `palloc` |

There is no row where the compiler allocates for you. That table *is* the design.

## 5. Genericity without traits (tallyc has none — and doesn't need them here)

A higher-order function that should work over *any* closure has two honest,
explicit shapes:

**(a) Monomorphise over the environment.** Write the HOF over a concrete closure
type with the env as an erased parameter; tallyc specialises per env shape (it
already monomorphises level- and mult-polymorphism this way), so the closure is flat
and the whole thing inlines — zero cost.

```
map : {0 e : Type} -> {0 a : Type} -> {0 b : Type}
    -> FlatClo e a b -> List a -> List b
fn map(f, xs) { match xs { Nil => Nil, Cons(h,t) => Cons(applyFlat(f,h), map(f,t)) } }
```

**(b) Dictionary-passing** — be generic over the representation `f` by taking its
`apply` explicitly. `apply` is itself a closed function value (the §2 primitive), so
there is no magic and no regress:

```
map : {0 f : Type} -> {0 a : Type} -> {0 b : Type}
    -> (apply : f -> a -> b) -> f -> List a -> List b
fn map(apply, f, xs) {
  match xs { Nil => Nil, Cons(h,t) => Cons(apply(f, h), map(apply, f, t)) }
}
```

This is exactly what a typeclass would desugar to — except **you** pass the
dictionary, so it is visible and there is no resolution magic. (If we ever add
traits, `Apply f a b` becomes sugar for (b). Not required.)

The `map`/`filter`/`foldr` in the shipped stdlib become form (a) or (b); when the
function argument is statically known, monomorphisation erases the closure entirely
(the fast path that makes them *run*, not just check).

## 6. Linearity & multiplicity (read the call-count off the type)

Because a closure is a value holding its captures, QTT already gives us the whole
story:

- capture only **ω** values ⇒ an **ω closure**: reusable, callable any number of
  times (`map`'s callback: `(w f : …)`).
- capture a **linear** value ⇒ a **one-shot closure**: calling it consumes the
  capture, so it is callable **exactly once** — this is `spawn`'s `(1 work : (1 x :
  e) -> a)` and Rust's `FnOnce`. Handing a one-shot closure to `filter` (which reuses
  its predicate across the recursion) is a *type error*, correctly.
- argument-multiplicity and capture-multiplicity are independent, exactly as `lmap`
  already shows (`(w f : (1 x : a) -> b)`).

So "can I call this twice / does calling it free something" are properties of the
closure's *type*, checked — not conventions.

## 7. Lambda sugar (optional) — and it must name its representation

Bare `fn` + the §3 library already give first-class functions; sugar is ergonomics.
The rule that keeps it honest: **a capturing lambda must name the representation it
compiles to.** No un-annotated capturing lambda has a default — that would be an
implicit allocation choice.

```
\x. x + k              -- captures k ⇒ ERROR: choose a representation (\[flat]/\[own]/…)
\[flat] x. x + k       -- ⇒ FlatClo Nat Nat Nat over env=(k)     — no heap
\[own]  x. x + k       -- ⇒ OwnClo  Nat Nat     over Own (k)     — your alloc, linear
\x. x + 1              -- captures nothing ⇒ a bare code pointer  — fine, no tag needed
```

Desugaring is standard lambda-lifting: compute free vars, lift the body to a
top-level `fn lam$k(env, x) { let (…) = env; body }`, and emit the chosen
constructor (`FlatClo((k), lam$k)` / the `OwnClo` `box`, etc.). The allocation is
whatever the named representation says — visible, typed, yours.

(Alternative if even tagged sugar feels too implicit: ship **no** lambda sugar, keep
only §3's explicit constructors. That is the most-explicit end and is a one-line
decision in §13.)

## 8. Lowering

1. **The primitive (§2):** a term for "closed function as value" → its LLVM function
   address; application of a function-typed value → indirect `call`. The single new
   backend capability. (Lambda-lifting, if sugar is added, is a surface pass that
   produces only closed functions, so it reduces to this.)
2. **flat closures:** already just a `struct` — reuse the existing by-value
   flat-struct ABI; apply = destructure + §2 call. *No new codegen.*
3. **owned / region / borrowed:** already just `Own`/`Pool`/`Slice` data — reuse the
   existing memory codegen; apply = unpack + §2 call. *No new codegen.*
4. **defunctionalised:** already just `enum` + `match`. *No new codegen.*
5. **monomorphisation over env types / statically-known callbacks:** extend the
   existing surface monomorphisation passes so a HOF specialises per closure shape /
   inlines a known callback — the zero-cost fast path.

So the plan is: **build one primitive, reuse everything else.** That is the whole
appeal of "closures are data."

## 9. What tallyc deliberately refuses (and why)

Named so the omission is a decision, not an oversight:

- **No `dyn`/auto-boxing** (Rust `dyn`, C++ `std::function`): would let a closure
  become heap-uniform without you writing it. If you want uniform, you write `OwnClo`.
- **No escape analysis** (Swift, Go): the compiler deciding stack-vs-heap is the
  compiler choosing your allocation. Refused.
- **No GC / refcount environments** (OCaml, Haskell, Koka, Swift ARC): hidden frees.
  Refused. Environments are linear (`Own`), region-scoped, or values.
- **No implicit closure at all** for a capturing lambda without a named
  representation (§7): no silent default.

## 10. Totality & dependent functions

- **Totality:** calling a closure calls an unknown function; a closure used in the
  **total** fragment must be built from **total** code. After lifting, `code` is a
  top-level function checked for totality normally — no new machinery. Non-total
  code stays fine in `%partial`.
- **Dependent closures** (argument/return type depends on the value) are out of
  scope for the first cut; first-class functions here are simple `a -> b`, which is
  all `map`/`filter`/`foldr` need. Dependent first-class functions remain a
  definition-level construct — future work.

## 11. Phasing

- **P1 — the primitive (§2). ✅ DONE.** Closed function values are callable in
  native code: a top-level `fn` (plain `λ` or `%partial` `Fix`) used as a value
  materialises to a real LLVM function and yields its code address (`fn_value` /
  `build_closed_fn` in `dep_codegen`); applying a function-typed variable emits an
  indirect call (`compile_indirect_call`), flattening each argument to its `i64`
  components exactly as `build_fix` does for parameters. This makes the shipped
  stdlib's `map`/`filter`/`foldr` **run** with real function arguments, and the
  whole representation family of §3 executes — see `examples/first_class.tal`
  (→ 60) and `examples/stdlib.tal`. No lambda syntax yet, so P1 function values
  are top-level `fn`s and closure `code` fields (both closed).
  - *Known P1 limits, both surfaced as CLEAR errors, never silent miscompiles:*
    (a) a function value with a **multi-word return** is rejected (`build_closed_fn`
    guards `ret_width == 1`) — return a scalar/pointer or apply directly; (b)
    **partial application** of a `fn` is rejected at elaboration (exact-arity check),
    so currying/partial closures are a later phase, not a codegen hazard.
- **P2 — flat closures as blessed stdlib types (§3 flat).** `FlatClo` + `applyFlat`
  in the prelude; capturing works with zero heap.
- **P3 — owned/region/borrowed closures (§3). ✅ (owned) DONE.** The existential
  heap closure `OwnClo a b : Type 1` runs, including a heterogeneous `List` of
  closures with different capture shapes folded over a seed, each freed once — see
  `examples/closures_owned.tal` (→ 31). Two things this pinned down:
  (1) **existential-constructor inference** — `MkOwnClo`'s hidden `{0 e}` is now
  recovered from the explicit arguments' types (a pre-pass in `solve_ctor`
  mirroring `solve_fn_call`), fixing `cannot infer implicit argument of MkOwnClo`;
  (2) **representation** — the `code` field takes its environment BY POINTER
  (`(1 o : Own e) -> a -> b`), so the existential `e` crosses a generic caller's
  abstract boundary as a uniform word (tallyc never boxes an abstract-typed value
  by value), and each concrete `code` unboxes it. Linear discipline: the linear
  consumption must be `let`-sequenced (`let v = unbox…; code…`), not nested in the
  call. **Region variant ✅ (by composition):** a flat closure is a value, so the
  existing `Pool`/`Region` vocabulary stores closures in an arena and frees them
  all at once with `prelease` — no new closure machinery, the "closures are data"
  thesis. `examples/closures_region.tal` → 27 (two closures arena-allocated,
  applied, the whole region released). The **borrowed** variant composes the same
  way from the `borrow`/`SRead` API, but the closure is incidental to the borrow
  ceremony, so it is a documented pattern rather than a headline example.
- **P4 — monomorphisation of HOFs over closures (§5/§8-5). ✅ DONE (both shapes).**
  A statically-known function argument is inlined into its recursive HOF — **no
  indirect call, no function-value pointer** in the hot loop (proven with the new
  `tally ir`).
  - *`%partial` (`Fix`) HOFs.* The App-arm `Fix`-head handling specialises via the
    static-argument transform (PE's `try_specialise_static_param`, verbatim-check):
    the callback is substituted in, the parameter dropped, the recursion tied to a
    fresh residual `Fix`. `applyN(inc, 5, 0)` → a 2-param `tally_fix` with `inc`
    inlined as `add i64, 1`.
  - *Total (eliminator-based) HOFs — the stdlib `map`/`filter`/`foldr`.* A binder
    bound to a static function is carried as a `Slot::StaticFn(term, ptr)` in the
    runtime env (it rides WITH the env, so scoping is automatic and it survives
    capture into an eliminator helper via `rebind_live`); at an application site the
    callback is inlined from its term instead of calling the captured pointer.
    `map`/`filter`/`foldr` chained lower to eliminator loops of pure inlined
    arithmetic — `add i64` for `dbl`, `add i64` for the fold — with **0 indirect
    calls**. The materialised code pointer becomes an unused capture that `-O2`
    deletes. A DYNAMIC closure value (a `code` field, a runtime callback) stays a
    `Slot::Val` and takes the P1 indirect path, unchanged.
- **P1.5 — multiplicity-polymorphic closures. ✅ DONE.** A datatype may take a
  `(m : Mult)` parameter (`struct FlatClo (m : Mult) (e)(a)(b) { code : e -> (m x
  : a) -> b }`); it is monomorphised to `FlatClo$1`/`FlatClo$w` in the same pass
  as mult-poly *functions* (a function's own `m` flows into a datatype ref, so
  `mangle_ty` resolves it at instantiation), and the user-written base
  constructor `FlatClo`/`match FlatClo(…)` resolves to the instance via the
  expected/scrutinee type (`resolve_mono_ctor`, gated by a `poly_ctor_base` set
  so `match` desugaring accepts the base name). So flat/owned closures get the
  multiplicity-polymorphism the defunctionalised representation already had — ONE
  `applyC` serves both an ω callback and a linear (arg-consuming) one; passing
  the linear closure at ω is a type error. `examples/closures_linearity.tal` → 156.
- **P5 — lambda sugar with named representation (§7), if adopted.**

Each phase is usable on its own; P1 alone makes the current stdlib run.

### Validation — C parity

`examples/hof_bench.tal` passes a degree-3 polynomial kernel as a **first-class
function value** to a `%partial` higher-order XOR-fold, over a stdin-controlled
(so not constant-folded) run count. `tally ir` shows **0 indirect calls** — P4
specialises the callback into the loop — and against the hand-written
`bench/hof_bench.c` (callback inlined by hand): identical output, **tally 0.12s
vs C 0.12s over 251M iterations** (arm64, `-O2`, best of 5). A higher-order
function over a first-class function runs at hand-written-C speed.

## 12. Prior art, mapped to this design's axis (does it *force* a representation, or *offer the family*?)

- **C / Zig** — offer only R5 (fn ptr + explicit context); no closures. Maximally
  explicit, zero freedom of representation. We keep their explicitness and *add* the
  family.
- **Rust** — `fn`/`impl Fn` (our flat + monomorphise) and `Box<dyn Fn>` (our owned) —
  but `dyn` auto-boxes and `move`/capture defaults are compiler-driven. We take the
  representations, drop the automatic ones.
- **C++** — value lambdas (our flat) + `std::function` (auto-boxed R4). Explicit
  capture list, but SBO/heap is hidden. We keep explicit capture, refuse hidden heap.
- **Roc / MLton** — R3 lambda sets / whole-program defunctionalisation, automatic.
  We make R3 available **by hand** (the defunctionalised row) instead of forcing it
  globally, so it composes with separate compilation and dependent types.
- **Swift / Go** — escape analysis picks heap. Refused (§9).
- **OCaml / Haskell / Koka / Lean** — boxed, GC/RC. Refused for environments (§9).
- **ATS** — the closest ancestor: a linear+dependent C-level language with `clo`
  (flat), `cloptr` (linear heap closure, explicitly freed), and template functions.
  Our flat / owned / monomorphised rows are ATS's three, recovered as *data* rather
  than three built-in closure kinds.

The distinguishing move: other languages ship one or two closure representations as
*language features*; tallyc ships **zero** as features and **all** of them as data,
because it already has the memory vocabulary to express them and the requirement is
that you choose.

## 13. Remaining decisions

1. **Lambda sugar (§7): tagged `\[flat]`/`\[own]` sugar, or no sugar at all
   (constructors only)?** Sugar is ergonomic and still explicit (the tag names the
   allocation); no-sugar is the most explicit possible and ships sooner.
2. **Which representations are *blessed* in the prelude** vs left for users to
   define? (Recommend: `FlatClo`+`applyFlat` and `OwnClo`+`box`/`callC`/`freeC`;
   region/borrowed/defunctionalised documented as patterns.)
3. **HOF genericity default (§5):** monomorphise-over-env (form a, zero-cost, ties
   the HOF to a closure type + code duplication) vs dictionary-passing (form b, one
   code copy, an explicit `apply` argument). Could ship both; which is the one the
   stdlib `map`/`filter`/`foldr` use?
4. **`OwnClo` at `Type 1`:** accept the universe bump, or add a non-existential
   uniform encoding (e.g. env as an opaque byte buffer with an erased size) to keep
   it in `Type 0`? (The `Type 1` version works today; the byte-buffer version is more
   C-like but less safe.)
