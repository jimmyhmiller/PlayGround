# Compile-time evaluation (`comptime`)

`(comptime E)` runs `E` in the **real language** during compilation and splices
the resulting literal. The same `defn`s, the same `=`/arithmetic, the same `match`
— executed by interpretation instead of being lowered to machine code. This is the
bridge toward "the whole language available at compile time": runtime code becomes
usable at compile time by *running* it.

```lisp
(defn fact [(n i64)] (-> i64) (if (icmp-le n 0) 1 (imul n (fact (isub n 1)))))
(defn main [] (-> i64) (comptime (fact 5)))     ; main compiles to `ret i64 120`
```

It also dissolves the "two `=`" question: inside `comptime`, the `Eq` trait's `=`
runs by interpretation — it is literally the same `=` as runtime.

```lisp
(comptime (if (= 7 7) 100 0))   ; the runtime Eq trait, evaluated at compile time
```

## How it works

- The parser produces `ExprKind::Comptime`. The checker type-checks the inner
  expression (so the form has its type) but does **not** evaluate it yet.
- After every function is checked, `comptime::fold_program` walks the elaborated
  program and replaces each `Comptime` node with the literal its inner expression
  evaluates to. Because it runs post-check, a `comptime` form can call any `defn`,
  recursively. Mono/codegen never see a `Comptime` node.
- The evaluator (`src/comptime.rs`) is a tree-walker over the typed `Expr` with a
  fuel budget (a runaway loop errors instead of hanging the compiler).

## Supported (Stages 1 + 1b)

- scalars: `int`/`bool`/`float` literals, arithmetic + comparison + `inot`, `cast`.
- control flow: `if`, `let` (immutable **and** mutable), `do`, `match`,
  `loop`/`break`/`continue`.
- the `=` trait (it lowers to an ordinary impl call) — so one `=` at both phases.
- calls to monomorphic `defn`s, including recursion.
- **memory model (1b):** mutable locals, `zeroed`/`alloc`, `load`/`store!`,
  `field`/`index` places, struct/array/sum aggregates, and passing aggregates
  **across function calls** (by reference). Modelled with reference-counted cells;
  aggregate values are references into them, deep-copied where the language copies.

**Aggregate results (1c/1d):** a `comptime` form may return any aggregate — a
**struct** (incl. nested), a **sum**, or an **array**. The value-builder
synthesizes the elaborated expression that reconstructs it: a struct/array becomes
`(let [t (alloc-stack T)] (store! (field/index t …) v)… (load t))` (an immutable
`t` holding a real `(ptr T)`); a sum becomes a variant call. The classic use — a
**compile-time lookup table** — works: build an array with a loop in `comptime`,
index it at runtime.

**Static-asserts can run real code:** `(static-assert (comptime (= (check) 42)) …)`
folds its condition by interpretation, so an assertion can call any `defn`.

Not supported *yet* — each raises a clear error rather than miscompiling:

- generic calls, FFI/`extern`, `llvm-ir`, function pointers, strings.
- `sizeof`/`alignof`/`offsetof` (those need LLVM target layout, only available in
  codegen — a `comptime` form can't compute them).

A fuel budget bounds runaway loops/recursion.

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

(Sums in statics aren't supported yet — a sum const is rebuilt at use sites.)

## Compile-time reflection (Stage 2)

Compile-time code can introspect a type's structure — so reflection isn't
macro-only. These forms take a *type* and are evaluated by the comptime
interpreter (folded to a literal, like `sizeof`), usable in `comptime`/`const`/
`static-assert`/ordinary code:

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

- field *types* as comptime `Type` values (recurse into a field's type), and
  reflecting a generic type parameter (resolved at mono).
- comptime string operations (compare/concat) — needs `=`/ops over `(slice u8)`.
- `sizeof`/`alignof`/`offsetof` at comptime (a layout module independent of LLVM).
- **3** — staged macros: run code generation in the runtime language too (the big
  rearchitecture that unifies the macro language with the runtime).
