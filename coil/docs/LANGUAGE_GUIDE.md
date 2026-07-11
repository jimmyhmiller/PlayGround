# The Coil Language — a guide for agents

A dense, practical reference for writing correct Coil. Coil is a low-level,
Lisp-syntax, ahead-of-time language: s-expressions, a macro system, and a type
system where **calling convention** and **allocation** are first-class. It emits
a native object and links with the system `cc`; the `wasm32-unknown-unknown`
target instead writes a WebAssembly module directly. Read this end to end before
writing Coil; most mistakes come from the gotchas marked ⚠.

The compiler is self-contained: the prelude and standard library are bundled
inside it, so `coil` and every `(import "…")` below work from any directory.

## Build & run

    coil run   file.coil                 # build + run a single file
    coil build file.coil -o out          # build a native executable
    coil build file.coil --target wasm32-unknown-unknown -o out.wasm
    coil run                             # build+run the ./Coil.toml project
    coil run -- arg1 arg2                # forward args to the program
    coil build file.coil -lm             # link a library (-l<name>)
    coil repl                            # interactive session
    coil fmt   file.coil                 # print formatted source (--write / --check)

`main`'s `i64` return is the process exit code. There is no JIT. A file that is
imported must start with `(module NAME)`. `Coil.toml`:

    [package]
    name  = "app"
    entry = "main.coil"      # default src/main.coil
    [link]
    libs = ["m"]             # -> -lm

The Wasm target produces an instantiable module. The compiler performs the final
Wasm-object conversion itself; this target does not invoke a linker process, and
native libraries cannot be linked into the module. It is otherwise a full target
for **the browser** — enough to build interactive pages in Coil (see `web/`):

- **Exports.** `main` and its linear `memory` are always exported. Every
  `(export-c [f :as "name"])` function is also exported, so JS can call into Coil:
  `instance.exports.name(args)`. Args/results are wasm scalars (`i32`/`i64`/floats).
- **Host imports.** An `extern … :cc c` that is *declared but never defined* becomes
  a wasm import `env.<name>` — this is how Coil calls out to JS (DOM, `console`,
  fetch, …). Pass a string as a `(ptr u8)`+`i32` pair via `(slice-data s)` /
  `(slice-len s)`; the host reads it from linear memory as UTF-8.
- **Self-contained.** The finalizer resolves the linker-provided `__memory_base`
  and `GOT.mem.*` globals to concrete addresses, so string literals and
  `alloc-static` global state work with no JS-side plumbing. The only imports a
  module has are the host functions it actually calls.
- **`externref`.** A built-in opaque type: a wasm reference to a host (JS) value,
  held directly by the runtime and GC-managed. Use it in `extern` signatures and as
  params/returns/`let`-locals to pass JS values to and from Coil without a handle
  table — `(extern js_get :cc c [externref (ptr u8) i32] (-> externref))`. ⚠ An
  `externref` lives only in wasm locals/args; it **cannot** be stored in linear
  memory (no struct field, array, `(mut …)` slot, or `(ptr externref)`). To persist
  one across calls, hand it to a host retain-table and keep the returned `i32`
  index. Transient `externref`s are collected automatically — nothing to free.

`web/coil-runtime.js` instantiates a module and wires the `env.dom_*` imports from
`web/dom.coil` to a real `document`; `web/counter.coil` is a worked interactive
example (build with `--target wasm32-unknown-unknown`).

## Modules & imports

    (module app)                         ; every importable file starts with this
    (import "other.coil" :use *)         ; bring all exported names in, unqualified
    (import "other.coil" :use [a b])     ; specific names
    (import "other.coil" :as x)          ; qualified: x.name
    (export foo bar)                     ; optional; omitted = everything visible

Paths resolve relative to the **importing file's own directory**; bare stdlib
names (`alloc.coil`, `str.coil`, …) resolve to the bundled library from anywhere.
⚠ `extern` declarations are NOT deduped across modules — declare each libc
extern in ONE module and `:use *` it, or two importers colliding will fail to link.

## The two operator tiers

- **Metal ops**, any width: `iadd isub imul idiv irem`, `icmp-eq icmp-ne icmp-lt
  icmp-le icmp-gt icmp-ge`, `iand ior ixor ishl ishr`, `udiv urem` (unsigned),
  `fadd fsub fmul fdiv`, `fcmp-eq fcmp-ne fcmp-lt fcmp-le fcmp-gt fcmp-ge`.
- **Clean prelude operators**: `+ - * / %`, `= != < <= > >=`, `& | ^ << >>`.
  Implemented on `i64` (all of them) and `bool` (`=` / `!=`). `f64` has `+ - * /`
  and `< <= > >=` but **deliberately no `Eq`** — like Rust, because `NaN != NaN`
  breaks reflexivity; use `fcmp-eq` / `fcmp-ne` for float equality. For `u8`/`u32`
  and other widths, use the metal ops. Prefer clean operators on `i64`/`bool`;
  drop to metal for other widths.

## Numbers, bool, casts

Int types `i8 i16 i32 i64 u8 u32 u64 …` (arbitrary width, real signedness).
Floats `f32 f64`. `bool` is real (`true`/`false`). Literals infer width from
context; hex `0x1F`, binary `0b1010`, octal `0o17`, underscores `1_000`.

`(cast T x)` converts: `(cast i64 f)` truncates f64→i64 (numeric), `(cast f64 i)`
converts int→float, `(cast (ptr T) x)` reinterprets pointers, `(cast i64 p)` is a
pointer's address. ⚠ `cast` between f64 and i64 is a **numeric conversion, not a
bit reinterpret**. For a bitcast (e.g. NaN-boxing) round-trip through memory:
`(let [p (alloc-stack i64)] (store! p bits) (load (cast (ptr f64) p)))` — LLVM at
-O3 folds this to a register move.

## Control flow

Core: `if`, `do`, `let`, `loop`/`break`/`continue`. Macros (no import needed —
reexported from core): `when unless cond case case-by while for and or not`.

    (if cond then else)      ; ⚠ BOTH branches required, and they must have the
                             ;    SAME type (the whole if yields a value).
    (do a b c)               ; sequence, yields last
    (let [x e (mut y) e0] …) ; bindings; (mut y) is a mutable stack cell
    (loop … (break) … (break v) … (continue))
    (cond t1 e1 t2 e2 … else)      ; lone trailing = else
    (case x k1 e1 k2 e2 … default) ; x evaluated once; a dense integer case
                                   ;   compiles to a JUMP TABLE

⚠ `if` needs matching branch types. For effect-only conditionals write
`(if c (do …effects… 0) 0)` so both sides are `i64`. `store!` returns the stored
value's type, so `(if c (store! p ptr) 0)` mismatches — wrap: `(do (store! p ptr) 0)`.

There is no `return`. Structure with `if`, or use `(block :b … (return-from :b v))`.
Self-tail-recursion is constant-stack (guaranteed `musttail`).

## Structs

    (defstruct Point [(x i64) (y i64)])
    (defstruct Rect  [(lo Point) (hi Point) (data (ptr u8)) (buf (array u8 64))])

- `(field p name)` → a `(ptr FieldType)` (a place); then `load`/`store!`.
  Requires `p : (ptr Struct)`. Nested: `(field (field s lo) x)`. Array field
  element: `(index (field s buf) i)`.
- `(load place)` reads, `(store! place v)` writes.
- `(zeroed T)` = a zero value; `(sizeof T)`, `(alignof T)`, `(offsetof S f)` are
  compile-time.
- Passing: `(p Point)` = **immutable ref** (a `store!` through it won't type-check);
  `(mut Point)` = **mutable ref**, pass a place with `(mut place)`; `(ptr Point)` =
  raw pointer (metal / FFI / allocators). A `let` of struct/array type is a stack place.
- ⚠ `(field rvalue name)` fails — `field` needs a place (a pointer), not a value.
  Load into a place first, or take its address.

**Struct "inheritance" (C-style):** embed a header struct as the first field and
cast pointers — the header is at offset 0, so `(cast (ptr Sub) hdrptr)` and
`(cast (ptr Hdr) subptr)` are the same address.

## Sum types (tagged unions)

    (defsum Value (VBool [(b bool)]) (VNil) (VNumber [(n f64)]) (VObj [(o (ptr Obj))]))
    (defsum Option [T] (None) (Some [(val T)]))   ; generic

    (match v
      (VBool [b] …) (VNil [] …) (VNumber [n] …) (VObj [o] …))   ; must be exhaustive
    (Some 42)  (None)  (VNumber 1.5)             ; construct

Stored by value (tag + payload). Fine inside structs and generic collections.
Recursive sums need a `(ptr …)` child. `_` is a wildcard binder.

**Choose `defsum` for a closed set of mutually exclusive shapes.** This is the
default for `Option`/`Result`, state machines, protocol messages, syntax trees,
and compiler type representations: adding a variant makes every non-exhaustive
`match` a compile error. Prefer several small domain sums over one giant
all-purpose node type. For a recursive syntax tree, keep recursive children
behind pointers:

    (defsum Expr
      (IntLit [(value i64)])
      (Add [(left (ptr Expr)) (right (ptr Expr))]))

    (defstruct LocatedExpr [(line i64) (col i64) (expr Expr)])

Use a `defstruct` with an integer `kind` tag only when a uniform,
representation-sensitive record is intentional: for example, a hot token
stream, bytecode instruction, FFI record, or an externally prescribed layout.
Do not use it merely to avoid writing a sum; it permits invalid combinations of
fields and does not make new cases visible to the type checker.

## Pointers, memory, allocation

Three allocation *operations*, each yields `(ptr T)`:

- `(alloc-stack T)` → `alloca`, this frame. ⚠ **NEVER call `alloc-stack` inside a
  loop that runs many times** — alloca isn't freed until the function returns, so
  it leaks the C stack per iteration and eventually segfaults. Hoist it into a
  `let` outside the loop and reuse the slot.
- `(alloc-static T)` → one global cell per call site (see Globals).
- `(alloc-heap T)` → `malloc` (pair with `free`).

`(index p i)` → `(ptr T)` at element i (pointer arithmetic, scaled by `sizeof T`);
`(index p -1)` is p−1. Null: `(cast (ptr T) 0)`; null test `(= (cast i64 p) 0)`.

**Allocator API** (`lib/alloc.coil`, thread a `(ptr Allocator)`):

    (malloc-allocator)                 ; stable global libc allocator
    (arena-allocator cap)              ; bump allocator
    (create [T] a)                     ; -> (Option (ptr T))
    (alloc-slice [T] a n)              ; -> (Option (ptr T)) array of n
    (destroy [T] a p)                  ; free one T
    (unwrap-ptr [T] optbox)            ; (Option (ptr T)) -> (ptr T), null on OOM
    (raw-alloc a size align)           ; -> (Option (ptr i8))
    (raw-resize a p oldsz newsz align) ; realloc
    (raw-free a p size align)
    ; idiom: (let [p (unwrap-ptr [T] (create [T] a))] (store! p …) p)

## Collections (bundled)

**ArrayList** (`lib/arraylist.coil`): `(al-new [T] a)`, `(al-len [T] l)`,
`(al-get [T] l i)`, `(al-set! [T] (mut l) i v)`, `(al-push! [T] (mut l) v)`,
`(al-pop! [T] (mut l))`, `(al-free! [T] (mut l))`. Mutators take `(mut …)`.
**HashMap** (`lib/hashmap.coil`): `(hm-new [K V] a ops)`, `(hm-new-scalar [K V] a)`,
`(hm-get [K V] m k)` → `(Option V)`, `(hm-put! [K V] (mut m) k v)`,
`(hm-remove! [K V] (mut m) k)`. For string keys use `(str-keyops)` from `lib/str.coil`.
Type args `[T]` come right after the name; usually inferable, so often omittable.

## Strings & bytes

`"…"` has type `(slice u8)` (UTF-8 bytes, static storage). `c"…"` has type
`(ptr i8)` (NUL-terminated C string, for FFI/`printf`). ⚠ Don't pass `"…"` to a
`(ptr i8)` param or `c"…"` to a `(slice u8)` param.

`(slice T)` is a fat pointer `{data, len}`. `(slice-data s)`, `(slice-len s)`,
`(slice-get s i)`, `(subslice s lo hi)`, `(slice-new [T] ptr n)`. String helpers
(`lib/str.coil`): `(str-len s)`, `(char-at s i)`, `(str-eq a b)`, `(str-hash s)`,
`(substr s lo hi)`, `(str-concat a x y)`.

## Character literals

`\a` `\Z` `\0` are that byte's value (an integer literal). Delimiters/quotes work:
`\(` `\)` `\{` `\}` `\"` `\;` `\.` `\,` `\*`. Named: `\space`=32 `\newline`=10
`\tab`=9 `\return`=13 `\nul`=0 `\backspace`=8 `\formfeed`=12. Hex: `\u41`=65.
They are plain `i64` literals — use with metal/clean ops after casting the byte:
`(= (cast i64 (load p)) \a)`.

## Functions & function pointers

    (defn name [(a T) (b U)] (-> R) body…)   ; last expr is the return value
    (defn id [T] [(x T)] (-> T) x)            ; generic: [T] before the arg list
    (defn f [(p (mut Rect))] (-> i64) …)      ; mutable-ref param
    (defn main [(argc i32) (argv (ptr (ptr i8)))] (-> i64) …)   ; CLI entry

**Function pointers** (native callbacks, dispatch tables):
`(fnptr c [ArgTs…] Ret)` is the type (`c` = C convention); `(fnptr-of fn)` takes a
function's address; `(call-ptr fp args…)` calls indirectly. A normal `defn` can be
taken as a `(fnptr c …)` and called via `call-ptr`; aggregate (struct/sum) returns
cross the call correctly. Forward references within a file resolve (mutual
recursion is fine) — define in any order.

## Global mutable state

There is **no top-level mutable variable**. Use `alloc-static` inside a zero-arg
accessor — it returns the same global cell every call:

    (defn counter [] (-> (ptr i64)) (alloc-static i64))
    (store! (counter) (+ (load (counter)) 1))
    ; for a global struct singleton (like a VM):
    (defstruct VM [(x i64) …])
    (defn vm [] (-> (ptr VM)) (alloc-static VM))   ; (load (field (vm) x)) …

`(const NAME VALUE)` / `(const NAME TYPE VALUE)` — compile-time immutable bindings.
The value is ANY expression, run at compile time: `(const OP_RETURN 0)`, `(const
FACT5 (fact 5))`. An aggregate const (struct/array) is evaluated once and emitted as
a static global (a compile-time lookup table): `(const SQUARES (build-squares))`.

## Compile-time: comptime, macros, reflection

The whole language runs at compile time — one language, two phases. No separate
macro dialect.

**`(comptime E)`** evaluates `E` during compilation and splices the literal result:
`(comptime (fact 5))` compiles to the constant `120` (no call in the output). It runs
real Coil — arithmetic, `if`/`let`/`loop`, `match`, any monomorphic `defn`
(recursively), mutable locals, and memory (`alloc`/`load`/`store!`/`field`/`index`).
It may return a scalar, struct, sum, or array; build a table with a loop and index it
at runtime. ⚠ Not at comptime: generics, FFI/`extern`, `sizeof`/`alignof` (codegen
needs LLVM layout). Each raises a clear error, never a miscompile.

**Macros are ordinary functions** `[Code…] (-> Code)` — detected by type, no
`defmacro`. `Code` is a first-class value: quote a form with `` `FORM ``, splice a
value in with `~E`, splice a list's elements with `~@E`. `(gensym)` gives a fresh
symbol so macro temporaries don't capture. `&` before the last param makes it
variadic (soaks up the rest as one Code list). Calls expand inline, outside-in:

    (defn when [(c Code) (body Code)] (-> Code) `(if ~c (do ~@body) 0))
    (when (< x 10) (println "small"))     ; → (if (< x 10) (do (println …)) 0)

**`(meta (gen …))`** runs a generator at compile time and splices its result as new
top-level forms; later code may depend on what it generates.

**Reflection** — introspect a type by name at comptime (fold to literals):
`(field-count T)`, `(variant-count T)`, `(struct? T)`/`(sum? T)`/`(int? T)`/`(float?
T)`/`(ptr? T)`/`(array? T)`, `(field-name T i)`, `(field-type-kind T i)`,
`(field-type-name T i)`, `(field-index T "name")`. Inside a macro (where a type
arrives as a Code symbol) use the `code-*` family: `code-field-count`/`-name`/`-kind`
/`-type`, `code-variant-sum`/`-count`/`-name`/`-fields`, and trait reflection
`code-trait-method-count`/`-name`/`-arity`/`-param-type`/`-ret-type` (for generating
vtables). Take Code apart with `code-count`/`code-nth`/`code-rest`/`code-sym`
/`code-list?`/`code-sym?`/`code-int?`. This makes `derive` (lib/derive.coil:
eq/hash/keyops) a pure library, not a compiler builtin.

## I/O & FFI

    (extern printf   :cc c [(ptr i8) ...] (-> i32))     ; ... = variadic
    (extern snprintf :cc c [(ptr i8) i64 (ptr i8) ...] (-> i32))
    (extern write    :cc c [i64 (ptr i8) i64] (-> i64)) ; fd 1=stdout 2=stderr
    (extern exit     :cc c [i32] (-> void))

`(printf c"%d\n" 42)`. Floats cross the C ABI correctly; structs pass/return by
value with the real C ABI. To call a Coil fn from C (e.g. `qsort` comparator) pass
`(fnptr-of f)`. Ambient `print`/`println` (over stdout) need no import.
`lib/io.coil`/`lib/fmt.coil` give a `(ptr Writer)` API: `(stdout)`, `(stderr)`,
`(print-str w s)`, `(fmt w "n={d} s={s} f={f}\n" a b c)`. ⚠ `{f}` is a fixed
6-digit display, NOT C `%g`; for exact float formatting call libc `snprintf` with
`c"%g"`. `coil cimport header.h` auto-generates bindings from a real C header.

## Reserved-name gotchas ⚠

`call` and `block` are builtins/macros — don't name a `defn` `call` or `block`
(you'll get "call target: expected symbol" / "macro arity mismatch"). Avoid `type`
as a struct field name. When in doubt, prefix your name (`p-call`, `vm-call`).

## Bundled standard library

These modules ship inside the compiler — `(import "NAME.coil" :use *)` works from
anywhere, no path or install step:
`alloc` (allocators), `arraylist`, `hashmap`, `slice`, `str`, `mem`, `io`, `fmt`,
`print`, `result` (Option/Result), `control` (case/cond/while/for/…), `match`,
`try`, `thread`, `atomic`, `simd`, `closure`, `derive`, `mmio`, `sexp`. The common
ones are summarized above; import a module and call its functions directly.
