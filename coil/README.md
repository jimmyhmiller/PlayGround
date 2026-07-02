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

Coil's only loop is **self-tail-recursion**, and it is emitted as an explicit
LLVM `musttail` call — a frame-reusing jump guaranteed by the backend at *any*
optimization level, so stack-safety does not depend on the optimizer (a
100M-deep recursion runs in constant stack even at `-O0`). The compiled output
is then **fully optimized**: every object runs LLVM's `-O3` pipeline (mem2reg,
inlining, GVN, loop optimizations, tail-call elimination). On matched compute
benchmarks this lands Coil at **parity with `cc -O3`** (within measurement
noise, same LLVM backend and opt tier) — see [`bench/`](bench/README.md).
(`emit-ir` deliberately shows the *un*optimized IR Coil generates, which is what
the struct-ABI tests diff against clang.)

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
  register. The trampoline + call-site asm is emitted per target architecture
  (x86-64 and AArch64); verified by AOT-compiling and running native
  executables, including recursion through the exotic ABI, on both arches.
- M3: allocation. Pointers are **region-less** — `(ptr T)` is just a pointer
  (à la Zig/C). *Where* memory comes from is an **operation**, not part of the
  type: `(alloc-stack T)` → `alloca`, `(alloc-static T)` → a global,
  `(alloc-heap T)` → `malloc`; all yield `(ptr T)`. No ownership, borrows, or
  lifetimes — control over allocation comes from allocators (below).
- Allocators (Zig-style): an allocator is an explicit **value** — a vtable
  struct of function pointers — threaded through, so a function that allocates
  *takes* an `Allocator` and its type shows it. The interface is alignment-aware
  (`alloc(len,align)` / `resize(ptr,old,new,align)` / `free(ptr,len,align)`) and
  signals failure with a **sum type** (`(Option (ptr i8))`, `None` = OOM), never
  a null sentinel. `lib/alloc.coil` ships a `malloc`-backed and an `arena` (bump)
  allocator plus a typed generic layer — `(create [T] a)`, `(destroy a p)` (`T`
  inferred), `(alloc-slice [T] a n)` — that sizes/aligns itself with
  `sizeof`/`alignof`. The same code runs under either allocator by swapping the
  value; the compiler has no allocator concept (structs + fnptrs + `extern`).
- IO (Zig-style): a `Writer`/`Reader` is likewise an explicit capability value
  `{ fn-ptr, ctx }` threaded in, so doing IO shows up in a function's type — no
  ambient stdout. `lib/io.coil` ships `write-all`/`write-byte`/`print-int`/
  `print-str` over a `Writer`, with `stdout`/`stderr`, a `null` sink, and a
  fixed-buffer (in-memory) sink; `Reader` mirrors it (`stdin`, `read-some`).
  Errors are a sum type (`(Result :i64 IoError)`). It composes with allocation —
  the same `Writer` interface formats into an allocator-provided buffer or a file
  descriptor (see `examples/io.coil`).
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
  a pointer parameter (`void*`-style). A trailing `...` marks a **variadic**
  function, so `printf`/`snprintf`/`scanf` and friends are callable
  (`(extern printf :cc c [(ptr i8) ...] (-> i32))`). Scalars, pointers, and
  floats cross the boundary correctly, and **structs pass/return by value** with
  the real C ABI — small structs are coerced into registers and large ones go
  indirect (`byval`/`sret`), matching what clang emits on both **System V
  AMD64** and **AArch64 AAPCS64** (see "Struct-by-value C ABI" below). So
  programs can do I/O — e.g. `putchar`/`write`/`printf` — and call libc
  functions that take or return structs by value (e.g. `div`/`ldiv`).
  `coil cimport <header.h>` auto-generates these bindings from a real C header
  by walking **clang's JSON AST** (clang does the parse — Coil never hand-rolls a
  C grammar): functions, scalars, pointers, simple structs, **typedefs**
  (resolved through clang's desugaring — `size_t`→`u64`, typedef'd structs),
  **enums** (the type → its integer width; the constants → `const` defs), and
  object-like **`#define`** constants (a `clang -dM` pass). The cardinal rule:
  unmappable constructs (unions, bitfields, `long double`, function-pointer
  params) are **refused with a clear note**, never emitted as a silent-wrong
  binding. `--link-flag`/`-l` on `build`/`run` links any C library or object.
- Scalars: **floats** `f32`/`f64` (literals `3.14`/`1e9`, `fadd`/`fsub`/`fmul`/
  `fdiv`/`frem`, ordered compares `fcmp-lt…`, and `cast` among int↔float and
  f32↔f64); a real **`bool`** (comparisons return it; `true`/`false`; short-
  circuiting `and`/`or`/`not`); and **bitwise/shift** ops on integers
  (`iand`/`ior`/`ixor`/`ishl`/`ishr` — arithmetic shift for signed, logical for
  unsigned — and `inot`).
- C types: **arbitrary-width integers** `iN`/`uN` for any `N` (Zig-style: `u2`,
  `i7`, `u23`, `i64`) with real signed/unsigned semantics (sext vs zext, sdiv vs
  udiv, signed vs unsigned compares; mixing widths or signedness is a type
  error). Integer **literals infer their width** from context (bidirectional
  elaboration): a bare `42` adopts the `iN`/`uN` it meets — the other operand of
  an op, the other `if`/`match` branch, a `store!`/return/call/field target — so
  `(store! p 42)` into a `(ptr u8)` and `(iadd x 1)` with `x : u8` need no
  `(cast :u8 …)`; a literal that doesn't fit its inferred type is a compile
  error. Typed pointers `(ptr TYPE)` (so
  `(ptr (ptr i8))` is `char**`), pointer indexing `(index p i)`, and `(sizeof
  TYPE)`. `cast` converts among ints and pointers in every direction: int↔int
  width change, ptr↔ptr reinterpret, and **int↔ptr** (`(cast (ptr i8) 0)` is
  null, `(cast :i64 p)` is an address — for null tests, MMIO, tagged pointers,
  packing an fd into a vtable `ctx`). **String literals** `"…"` lower to a
  private NUL-terminated `[N x i8]` and have type `(ptr i8)` (C-string
  compatible). `main` may take `(argc :i32) (argv (ptr (ptr i8)))`, so programs
  read their command line.
- Constants: `(const NAME VALUE)` / `(const NAME TYPE VALUE)` — a named scalar
  constant (int/float/bool). A reference elaborates to the literal **inline**
  (resolved during checking, zero runtime cost): an *untyped* const re-enters
  width inference exactly like writing the literal (so it slots into any
  `iN`/`uN` context), a *typed* const pins the width and fit-checks at
  definition. Distinct from `def` (the compile-time macro binding); consts live
  in a flat global namespace and are shadowed by locals. C enum constants and
  object-like `#define`s lower to these (see C interop).
- Structs & arrays: `(defstruct Name [(field :type) ...])` and `(array T N)`.
  A field/element is reached as a pointer via `(field p name)` / `(index p i)`,
  then `load`/`store!`. Structs nest by value (or self-reference by pointer);
  allocate any type with `(alloc-stack/static/heap TYPE)`.
- References & mutability (the everyday tier over `ptr`): a bare struct
  parameter `(p Point)` is an **immutable reference** — read its fields, but a
  `store!` through it is a compile error; `(mut Point)` is a **mutable
  reference**, opt-in. A `let` of a struct/array value (e.g.
  `(let [(mut v) (zeroed Vec)] …)`) is a **stack place** of that value type — no
  `(ptr …)`, no `alloc-stack` — and `(mut place)` borrows a place mutably at a
  call site (`(push (mut v) x)`), the one marker that points where a write can
  escape. It's **const-correctness, not borrow-checking** (no lifetimes/aliasing
  analysis) — purely "is this handle allowed to write." References are the same
  machine representation as `ptr`; the checker erases them to pointers after the
  mutability check, so codegen is unchanged and `ptr` stays the metal tier for
  allocators/FFI/arithmetic.
- Layout control (the dual of calling conventions): `(defstruct Name :layout
  c | packed | (align N) | explicit ...)`. `explicit` places each field at a
  per-field `:at` offset (gaps = padding, overlap = a union), realized as a byte
  blob. `:layout bits` packs sub-byte fields (`(f :bits N)`) into a backing
  integer, accessed by value as `uN` via `(get p f)` / `(set! p f v)`.
  Compile-time `(sizeof T)`, `(alignof T)`, `(offsetof Struct field)` and
  `(static-assert COND "msg")` pin a layout down — a wrong size/offset is a
  compile error. (Per-field endianness sketched in [`docs/LAYOUT.md`](docs/LAYOUT.md).)
- Generics (monomorphization): `(defn id [T] [(x T)] (-> T) x)`, generic
  structs `(defstruct Pair [A B] ...)` used as `(Pair i64 i64)`. Generic **sum
  types** too: `(defsum Option [T] (None) (Some [(val T)]))` with
  `(match o (None [] ...) (Some [v] ...))`. **Type arguments are inferred**
  (bidirectional checking): from the value arguments — `(id 42)`, `(pair-sum p)`,
  `(Some 42)` — by unifying declared parameter types against actual ones (even
  nested, e.g. `(ptr (Pair T T))`); *and* from the expected type pushed in from
  context — a bare `(None)`, or `(Ok v)`/`(Err e)` whose `Result` error type
  appears in no field, adopt their type from the function's return type, a
  `store!`/branch/argument target, or an enclosing constructor's field. Explicit
  `[i64 …]` is still accepted, and is needed only when nothing pins a parameter
  down (e.g. a bare `(None)` in a plain `let` with no expected type). The
  inferred arguments are filled in before monomorphization, which stays a pure
  specializer.
- Function pointers & closures: `(fnptr CC [types] ret)` type, `(fnptr-of name)`
  for a function's address, and indirect `(call-ptr fp args...)` (honoring the
  convention). Closures are **not** a language primitive — a closure is a struct
  of `{ code pointer, environment pointer }`. `closure.coil` shows heterogeneous
  heap closures (different captures, one type, one generic `apply`); `defclosure`
  (`lib/closure.coil`) is a userland macro that generates the whole closure
  (env struct + code + new/call/free) from one line — closures as a library.

Per-arch shim lowering is done: `:shim` trampolines and register-constrained
shim call sites are emitted for both **x86-64** (AT&T `call`, `%rsp` prologue,
`rdi…r9`/`rax`) and **AArch64** (`bl`, `x0…x7`/`x0`, `lr`-clobber), selected
from the target triple. A `defcc` that names registers absent on the target
(e.g. an x86 `defcc` compiled for arm64) is a clear hard error, not miscompiled
asm.

Not yet: the `adapt` macro (general convention-to-convention trampolines) and
per-field endianness. See the design docs.

## Struct-by-value C ABI

When a struct crosses the C boundary by value — an `extern`/`c`-convention
parameter or return, in either direction — Coil lowers it with the real C ABI
rather than passing a pointer. `src/abi.rs` classifies each struct for the target
and produces exactly the LLVM-level coercion clang emits:

- **System V AMD64.** Structs ≤ 16 bytes are split into eightbytes, each
  byte-classified (INTEGER vs SSE) and merged per the SysV field-walk rules, then
  coerced to the matching register slot (`iN` for integer eightbytes; `float` /
  `double` / `<2 x float>` for SSE). A two-eightbyte *return* is wrapped in a
  `{T0, T1}` literal struct. Structs > 16 bytes go indirect — `byval(T)` for an
  argument, an `sret(T)` hidden pointer for a return.
- **AArch64 AAPCS64.** A Homogeneous Floating-point Aggregate (1–4 members of one
  FP type) passes as `[N x fT]` and returns as the struct type. Other composites
  ≤ 16 bytes pack into x-registers (`i64` / `[2 x i64]` for an arg; the natural
  `iN` width for a small return). Composites > 16 bytes go indirect.

This is **verified two ways**: the emitted IR's `declare`/`define` lines are
diffed against what `clang -arch <a> -S -emit-llvm` produces for the equivalent C
(`tests/struct_abi.rs::emits_abi_{x86_64_sysv,aarch64_aapcs64}`), and real
programs are linked against C — calling libc `div`/`ldiv`, a C `<=16B`/`>16B`
round-trip helper, and a C caller of Coil functions that *return* structs — and
run, natively and (for the SysV path on an arm64 host) cross-compiled and run
under Rosetta. An unclassifiable shape is a hard error, never a silent pointer.

## Raw LLVM IR & SIMD

Coil exposes LLVM's instruction set directly, the way Mojo reaches into MLIR —
one general primitive, no per-opcode compiler support. `(llvm-ir RESULT
[operands…] "BODY")` drops a snippet of LLVM IR into an `alwaysinline` helper:
`$ret`/`$tN` expand to the result/operand LLVM type strings and `$N` to the
operand SSA names, `declare` lines are hoisted to module scope, and the helper
is linked in and **inlined away by `-O3` to zero overhead**. The form is
**type-checked at the boundary** — it has exactly its declared `RESULT` type, so
it composes with the rest of the type system — while the LLVM verifier checks the
body. Every instruction and intrinsic (present or future) is reachable this way.

The one supporting *type* is `(vec T N)` — an LLVM `<N x T>` SIMD vector (`load`/
`store!` and the existing arithmetic work on it lane-wise). On top, **SIMD is a
macro library** (`lib/simd.coil`): `vec4f`, `splat4f`, `vadd4f`/`vmul4f`, `vfma4f`
(via `@llvm.fma`), `reduce-add4f`, and a `dot4f` — each a one-line macro over
`(llvm-ir …)`. So it's *explicit* SIMD (you choose the width, not the
auto-vectorizer) that still optimizes: `examples/simd.coil`'s dot product lowers
to NEON `ldr q`, `fmul.4s`, and a horizontal reduce (`coil emit-ir` shows the
`<4 x float>` ops pre-inline). Because Coil now hosts arbitrary LLVM IR, a C
function compiled with `clang -emit-llvm` can be pasted into an `(llvm-ir …)` and
runs identically (`tests/llvm_ir.rs` embeds the murmur3 finalizer). See
`tests/llvm_ir.rs`.

```lisp
(include "lib/simd.coil")
(defn main [] (-> :i64)                         ; dot([10,10,10,12],[1,1,1,1]) = 42
  (cast :i64 (dot4f (vec4f 10 10 10 12) (splat4f 1))))
```

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
cargo test                                     # 155 tests (build + run native exes)

# AOT: compile + link a native executable, then run it (exit code = result)
cargo run -- run   examples/everything.coil; echo $?                       # => 42 (every feature, 24 self-checks)
cargo run -- run   examples/allocators.coil; echo $?                       # => 42 (Zig-style allocators)
cargo run -- run   examples/closure-lib.coil; echo $?                      # => 42 (uses (include ...))
cargo run -- run   examples/per-arch.coil; echo $?                         # => 42
cargo run -- run   examples/closure.coil; echo $?                          # => 42
cargo run -- run   examples/allocation.coil; echo $?                       # => 42
cargo run -- run   examples/inference.coil; echo $?                        # => 42 (literal inference)
cargo run -- run   examples/extern.coil                                    # prints 12345
cargo run -- run   examples/io.coil                                        # prints answer=42 (Writer capability)
cargo run -- run   examples/structabi.coil; echo $?                        # => 92 (struct-by-value C ABI: libc div)
cargo run -- run   examples/simd.coil; echo $?                             # => 42 (explicit SIMD: NEON fmul.4s via macros)
cargo run -- run   examples/references.coil; echo $?                       # => 42 (mut refs + let stack locals)
cargo run -- build examples/args.coil -o /tmp/args && /tmp/args a b c      # echoes argv

# Debug info: -g emits DWARF (function-granularity line tables + a .dSYM on macOS)
# so lldb/gdb can set breakpoints by function and show file:line in backtraces.
cargo run -- build examples/fib.coil -o /tmp/fib -g && lldb /tmp/fib        # see docs/DEBUGINFO_DWARF.md

# Benchmark the optimized output against C (clang -O3) on matched programs
bench/run.sh                                              # => bench/RESULTS.md (≈ cc -O3)

# Inspect the pipeline
cargo run -- emit-obj examples/shim.coil -o /tmp/shim.o   # native object file
cargo run -- emit-ir  examples/shim.coil                  # LLVM IR (trampoline + inline asm)
cargo run -- expand   examples/macros.coil                # program after macro expansion

# Cross-compile for a non-host target with --target <triple> (any command).
# The triple drives both the IR/ABI lowering and the linker's -arch; on macOS a
# cross binary runs under Rosetta, so `run` works end-to-end.
cargo run -- run     examples/per-arch.coil --target x86_64-apple-macosx11.0.0; echo $?  # => 42
cargo run -- build   examples/per-arch.coil -o /tmp/pa --target x86_64-apple-macosx11.0.0
cargo run -- emit-ir examples/per-arch.coil --target x86_64-apple-macosx11.0.0   # x86-64 SysV lowering
```

There is no `eval`/JIT: the only way to run a program is to AOT-compile it.
`main` (i64, no args) is the process entry; its return value is the exit code
(low 8 bits).

## REPL (and Emacs)

`coil repl` is an interactive session over the same AOT pipeline — no
interpreter, no JIT, nothing that can drift from the compiled language. The
session accumulates your top-level definitions as source; every other form is
an expression, compiled (session definitions + a generated entry that prints
the value) and run:

```
$ cargo run -- repl
coil> (defn square [(x i64)] (-> i64) (imul x x))
#'repl/square
coil> (square 12)
144
coil> (def n (square 3))          ; bind a VALUE for later inputs
n = 9
coil> (iadd n 33)
42
```

Definitions redefine by name (send an edited `defn` again to replace it).
Evaluation is *live* by default — hot code loading, evcxr-style: each eval
compiles to a dylib that is `dlopen`ed into the session process, so it's still
the real compiled code, but the process persists. `(def name EXPR)` stores the
value in a heap cell that later inputs read back, fully typed; malloc'd memory
and FFI state (an open window, a socket) survive across inputs too. A binding
is an immutable snapshot — rebind with another `def`; persistent *mutable*
state is an explicit pointer, Coil's normal discipline:

```
coil> (def counter (alloc-heap i64))
coil> (store! counter (iadd (load counter) 1))
```

The trade: a crashing eval takes the session down (it *is* the process).
`coil repl --isolate` gives the old behavior instead — every expression runs
in a fresh process; crash-proof, nothing persists.

Values print by their *inferred* type (the real checker runs over a probe):
scalars, quoted strings, `[…]` slices/arrays, `(Name :field v …)` structs,
`(Variant …)` sums — recursively; a type with no honest rendering (function
pointers, SIMD vectors) prints as `#<its-type>`. Commands: `:help`, `:type
EXPR`, `:session`, `:load file.coil`, `:reset`, `:quit`. Externs that need
libraries take the same link flags as `run`: `coil repl -lSDL2`.

The REPL is wire-compatible with Emacs' stock `inferior-lisp`; `emacs/coil-mode.el`
packages that up (a `lisp-mode` derivative for `.coil` plus `M-x run-coil`):

```elisp
(add-to-list 'load-path "/path/to/coil/emacs")
(require 'coil-mode)
;; (setq coil-program "/path/to/coil/target/release/coil") ; if not on PATH
```

Then from any `.coil` buffer: `C-x C-e` sends the sexp before point, `C-M-x`
sends (and redefines) the enclosing definition, `C-c C-l` `:load`s the file,
`C-c C-z` jumps to the REPL.

[inkwell]: https://github.com/TheDan64/inkwell
