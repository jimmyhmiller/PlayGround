# Coil (working name)

A low-level, Lisp-macro language where **calling convention** and **allocation**
are part of the **type system** — assembly-level control over where values live
across calls and where data lives in memory, with higher-level constructs built
up through macros. Backend: **raw LLVM** (via [inkwell]).

Core bet: a *closure* is not a primitive — it's
`(code-pointer-with-a-convention, environment-with-an-allocation)`. Make the
type system talk about conventions and allocations, and closures, vtables,
coroutines, and FFI trampolines become library code written in macros.

Coil is **ahead-of-time, no JIT**: it emits a native object file and links an
executable with the system `cc` — no runtime dependency on LLVM, and the exotic
`:shim` conventions become ordinary relocations the assembler/linker resolves.
Macros run during compilation in a tree-walking interpreter (no JIT, no LLVM);
`coil build` fully expands every macro before emitting any code.

- Full design: [`docs/DESIGN.md`](docs/DESIGN.md)

## Status

**M0–M4 + a user-defined macro system** — an s-expression front end that
AOT-compiles to native objects through LLVM. Calling conventions are part of the
type system (including ones LLVM's CC enum cannot express); allocation is about
explicit control (Zig-style allocators), not safety; and the higher-level
surface is grown with **real Lisp macros** — a compile-time interpreter, not
template substitution.

- M0: reader → core AST → LLVM codegen → JIT. `i64`, `let`, arithmetic,
  comparisons, `if`, direct + recursive calls.
- M1: `defcc` conventions attached to functions; the `:native` lowering sets the
  real LLVM calling convention on the function **and every call site**; the
  checker rejects mismatched arity, unbound vars, and (for shims) missing return
  registers or too few argument registers.
- M2: `:shim` lowering for exotic register layouts. A shim function becomes a
  `ccc` `__impl` body plus a **`naked` trampoline** that marshals the
  convention's registers into the SysV registers; call sites use
  register-constrained inline asm so each argument is genuinely pinned to its
  register. Verified through JIT, including recursion through the exotic ABI.
- M3: allocation. Pointers are **region-less** — `(ptr T)` is just a pointer
  (à la Zig/C). *Where* memory comes from is an **operation**, not part of the
  type: `(alloc-stack T)` → `alloca`, `(alloc-static T)` → a global,
  `(alloc-heap T)` → `malloc`; all yield `(ptr T)`. No ownership, borrows, or
  lifetimes — control over allocation comes from allocators (below).
- Allocators (Zig-style): an allocator is an explicit **value** — a vtable
  struct of function pointers — threaded through, so a function that allocates
  *takes* an `Allocator` and its type shows it. `lib/alloc.coil` ships a
  `malloc`-backed and an `arena` (bump) allocator; the same code runs under
  either by swapping the value. The compiler has no allocator concept; it's a
  library (structs + fnptrs + `extern`).
- Macros: `defmacro` / `def` run in a compile-time Lisp (quasiquote `` ` ``,
  unquote `~`, splicing `~@`, `gensym`, list/symbol builtins, recursion,
  helper functions). Macros can compute, recurse, and emit whole top-level
  definitions, and they compose with conventions and allocators. The target is a
  compile-time value (`target-arch`, `target-os`, `target-pointer-width`), so a
  macro can branch per architecture — e.g. select a `defcc`. **Automatic
  hygiene**: a template symbol ending in `#` (e.g. `tmp#`) auto-gensyms, so
  macro temporaries can't capture caller bindings. Inspect with `coil expand`.
- Modules: `(include "path")` splices another file's macros and definitions in
  (resolved from the working directory, with an include guard). `lib/closure.coil`
  is the `defclosure` macro shipped as a prelude — `(include "lib/closure.coil")`
  then `(defclosure NAME [captures] [params] RET body...)`.
- C interop: `(extern name :cc c [types] (-> ret))` declares a foreign
  function's convention + signature; calls are type-checked and the symbol is
  resolved at link time (libc, etc.). At the extern boundary any pointer matches
  a pointer parameter (`void*`-style). So programs can do I/O — e.g.
  `putchar`/`write`/`puts`.
- C types: **arbitrary-width integers** `iN`/`uN` for any `N` (Zig-style: `u2`,
  `i7`, `u23`, `i64`) with real signed/unsigned semantics (sext vs zext, sdiv vs
  udiv, signed vs unsigned compares; mixing widths or signedness is a type
  error). Typed pointers `(ptr TYPE)` (so
  `(ptr (ptr i8))` is `char**`), pointer indexing `(index p i)`, width
  `(cast :iN e)` (and ptr reinterpret), and `(sizeof TYPE)`. `main` may take
  `(argc :i32) (argv (ptr (ptr i8)))`, so programs read their command line.
- Structs & arrays: `(defstruct Name [(field :type) ...])` and `(array T N)`.
  A field/element is reached as a pointer via `(field p name)` / `(index p i)`,
  then `load`/`store!`. Structs nest by value (or self-reference by pointer);
  allocate any type with `(alloc-stack/static/heap TYPE)`.
- Layout control (the dual of calling conventions): `(defstruct Name :layout
  c | packed | (align N) | explicit ...)`. `explicit` places each field at a
  per-field `:at` offset (gaps = padding, overlap = a union), realized as a byte
  blob. `:layout bits` packs sub-byte fields (`(f :bits N)`) into a backing
  integer, accessed by value as `uN` via `(get p f)` / `(set! p f v)`.
  Compile-time `(sizeof T)`, `(alignof T)`, `(offsetof Struct field)` and
  `(static-assert COND "msg")` pin a layout down — a wrong size/offset is a
  compile error. (Per-field endianness sketched in [`docs/LAYOUT.md`](docs/LAYOUT.md).)
- Generics (monomorphization): `(defn id [T] [(x T)] (-> T) x)`, generic
  structs `(defstruct Pair [A B] ...)` used as `(Pair i64 i64)`, explicit type
  args `(id [i64] x)`. Generic **sum types** too: `(defsum Option [T] (None)
  (Some [(val T)]))` with `(match o (None [] ...) (Some [v] ...))`.
- Function pointers & closures: `(fnptr CC [types] ret)` type, `(fnptr-of name)`
  for a function's address, and indirect `(call-ptr fp args...)` (honoring the
  convention). Closures are **not** a language primitive — a closure is a struct
  of `{ code pointer, environment pointer }`. `closure.coil` shows heterogeneous
  heap closures (different captures, one type, one generic `apply`); `defclosure`
  (`lib/closure.coil`) is a userland macro that generates the whole closure
  (env struct + code + new/call/free) from one line — closures as a library.

Not yet: the `adapt` macro (general convention-to-convention trampolines),
type-argument inference (generics/sums need explicit type args today), per-field
endianness, an IO `Writer` capability, a per-arch shim backend. See the design
docs.

## What it looks like

```lisp
(defcc fast2 :params [rax rdx] :ret rax
  :clobber [rax rdx rcx] :preserve [rbx rbp] :native fast)

(defn add :cc fast2 [(a :i64) (b :i64)] (-> :i64) (iadd a b))
(defn main [] (-> :i64) (add 20 22))
```

`add` compiles to a `fastcc` LLVM function and is invoked with `call fastcc`.

## Build & run

Needs Rust and **LLVM 18** with `llvm-config` on `PATH`. On Debian/Ubuntu the
dev pieces inkwell/llvm-sys link against:

```sh
apt-get install -y llvm-18-dev libpolly-18-dev libzstd-dev zlib1g-dev
```

Then:

```sh
cargo test                                     # 95 tests (build + run native exes)

# AOT: compile + link a native executable, then run it (exit code = result)
cargo run -- run   examples/allocators.coil; echo $?                       # => 42 (Zig-style allocators)
cargo run -- run   examples/closure-lib.coil; echo $?                      # => 42 (uses (include ...))
cargo run -- run   examples/per-arch.coil; echo $?                         # => 42
cargo run -- run   examples/closure.coil; echo $?                          # => 42
cargo run -- run   examples/allocation.coil; echo $?                       # => 42
cargo run -- run   examples/extern.coil                                    # prints 12345
cargo run -- build examples/args.coil -o /tmp/args && /tmp/args a b c      # echoes argv

# Inspect the pipeline
cargo run -- emit-obj examples/shim.coil -o /tmp/shim.o   # native object file
cargo run -- emit-ir  examples/shim.coil                  # LLVM IR (trampoline + inline asm)
cargo run -- expand   examples/macros.coil                # program after macro expansion
```

There is no `eval`/JIT: the only way to run a program is to AOT-compile it.
`main` (i64, no args) is the process entry; its return value is the exit code
(low 8 bits).

[inkwell]: https://github.com/TheDan64/inkwell
