# Compile-time evaluation (`comptime`)

`(comptime E)` runs `E` in the **real language** during compilation and splices
the resulting literal. The same `defn`s, the same `=`/arithmetic, the same `match`
— compiled and executed during the build instead of being lowered into the output.
This is "the whole language available at compile time": runtime code becomes usable
at compile time by *running* it.

```lisp
(defn fact [(n i64)] (-> i64) (if (icmp-le n 0) 1 (imul n (fact (isub n 1)))))
(defn main [] (-> i64) (comptime (fact 5)))     ; main compiles to `ret i64 120`
```

It also dissolves the "two `=`" question: inside `comptime`, the `Eq` trait's `=`
is the same impl the runtime uses — literally the same `=`.

```lisp
(comptime (if (= 7 7) 100 0))   ; the runtime Eq trait, evaluated at compile time
```

## How it works

- The parser produces one `EComptime` node. The checker type-checks the inner
  expression (so the form has its type) but does **not** evaluate it yet.
- After every function is checked, `comptime.coil`'s `fold-expr` walks the elaborated
  program and replaces each `EComptime` node with the literal its inner expression
  evaluates to. Because it runs post-check, a `comptime` form can call any `defn`,
  recursively. Mono/codegen never see an `EComptime` node.
- The evaluator is the **compiled engine** (`comptime_eval.coil`): it recovers the
  site's checked type, builds a minimal closure sub-program of everything `E` calls
  plus a synthetic `(defn coil.ct.thunk [] (-> T) E)` exported as `coil_ct_thunk`,
  monomorphizes and builds it, runs the entry, and reads the result back — scalars out
  of the return register, aggregates through a write-through pointer thunk walked by
  the natural C layout. `build-value` turns that into a literal. There is no
  interpreter (`docs/INTERP_DELETION.md`).
- ⚠ Because the thunk is real compiled code, a runaway comptime computation is
  **unbounded**, and deep self-recursion is not tail-call-optimized on this path, so
  it crashes rather than reporting. Division by zero **is** explicitly guarded
  (integer only — a comptime float division to infinity is legal).

## Supported

- scalars: `int`/`bool`/`float` literals, arithmetic + comparison + `inot`, `cast`.
- control flow: `if`, `let` (immutable **and** mutable), `do`, `match`,
  `loop`/`break`/`continue`.
- the `=` trait (it lowers to an ordinary impl call) — so one `=` at both phases.
- calls to any `defn`, including recursion and generics.
- **memory:** mutable locals, `zeroed`/`alloc`, `load`/`store!`, `field`/`index`
  places, struct/array/sum aggregates, and passing aggregates **across function
  calls**. It is real memory in the compiler's process, not a model of it.

**Aggregate results:** a `comptime` form may return any aggregate — a
**struct** (incl. nested), a **sum**, or an **array**. The value-builder
synthesizes the elaborated expression that reconstructs it: a struct/array becomes
`(let [t (alloc-stack T)] (store! (field/index t …) v)… (load t))` (an immutable
`t` holding a real `(ptr T)`); a sum becomes a variant call. The classic use — a
**compile-time lookup table** — works: build an array with a loop in `comptime`,
index it at runtime.

**Static-asserts can run real code:** `(static-assert (comptime (= (check) 42)) …)`
folds its condition at compile time, so an assertion can call any `defn`.

**The computation is unrestricted** — generic calls, FFI/`extern` (a comptime
`(strlen c"hello!")` calls libc), function pointers, strings, allocators and
collections, and `sizeof`/`alignof`/`offsetof` all work.

What is refused — as a clear located error, never a miscompile — is the **result**,
which has to become a literal:

- a **pointer** (a compile-time address is meaningless in the built program), and
- an aggregate that is a **generic instance** (`(Option i64)`, `(Pair i64 i64)`):
  "cannot be materialized". Return a plain struct/sum, or the scalar you need.

**Computed `const`s:** a `const`'s value is any expression. A bare literal inlines
as before; a scalar/sum computation is evaluated at compile time —
`(const FACT5 (fact 5))`, `(const DOUBLE (* BASE 2))`.

**Aggregate consts = static data tables.** A `const` whose type is a struct or
array is evaluated once and emitted as a **constant global**; references become a
pointer to it. So a compile-time lookup table is real static data:

```lisp
(const SQUARES (squares))         ; => @const.SQUARES = private constant [8 x i64] [0,1,4,9,…]
(load (index SQUARES 5))          ; reads 25 straight from the global, at runtime
```

⚠ A **sum-typed `const`** is not supported: it does not fall back to rebuilding at use
sites, it aborts the build with `UNIMPLEMENTED: codegen: unknown static const <name>`.
Use `(comptime …)` at the use site, or a struct/array const.

## Compile-time reflection

Compile-time code can introspect a type's structure — so reflection isn't
macro-only. These forms take a *type* and fold to a literal (like `sizeof`), usable
in `comptime`/`const`/`static-assert`/ordinary code:

- `(field-count T)` → `i64` (struct fields)
- `(variant-count T)` → `i64` (sum variants)
- `(struct? T)` / `(sum? T)` / `(int? T)` / `(float? T)` / `(ptr? T)` / `(array? T)` → `bool`

Per-field reflection (the index is a compile-time value — a literal or a
`comptime`/loop variable):

- `(field-name T i)` → the i-th field's name, as a comptime **string** (`(slice u8)`)
- `(field-type-kind T i)` → its type's kind tag (`i64`: 0 int, 1 float, 2 bool,
  3 struct, 4 sum, 5 ptr, 6 array, 7 slice, 8 other)

```lisp
(const NF (field-count Point))                 ; a compile-time constant
(static-assert (struct? Point) "must be a struct")
(comptime (* (field-count Point) (variant-count Shape)))

; THE PAYOFF — a runtime field-metadata table, generated at compile time:
(defstruct FieldDesc [(name (slice u8)) (kind i64)])
(const FIELDS
  (comptime (let [(mut t) (zeroed (array FieldDesc 3)) (mut i) 0]
    (loop (if (>= (load i) (field-count Mix)) (break)
      (do (store! (field (index (mut t) (load i)) name) (field-name Mix (load i)))
          (store! (field (index (mut t) (load i)) kind) (field-type-kind Mix (load i)))
          (store! i (+ (load i) 1)))))
    (load t))))
; => @const.FIELDS = constant [3 x %FieldDesc] [ {{"a",1},0}, {{"b",1},1}, … ]
```

## Roadmap

- field *types* as first-class comptime `Type` values (recurse into a field's type),
  and reflecting a generic type parameter (resolved at mono).
- materializing a **generic-instance aggregate** as a comptime result.
- a **bound on runaway comptime** (there is none today), and TCO
  on the comptime thunk path.
