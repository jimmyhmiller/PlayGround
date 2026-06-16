# Coil (working name)

A low-level, Lisp-macro language where **calling convention** and **allocation**
are part of the **type system** — assembly-level control over where values live
across calls and where data lives in memory, with higher-level constructs built
up through macros. Backend: **raw LLVM** (via [inkwell]).

Core bet: a *closure* is not a primitive — it's
`(code-pointer-with-a-convention, environment-with-an-allocation)`. Make the
type system talk about conventions and allocations, and closures, vtables,
coroutines, and FFI trampolines become library code written in macros.

- Full design: [`docs/DESIGN.md`](docs/DESIGN.md)

## Status

**M0–M3 + a user-defined macro system** — an s-expression front end that
JIT-compiles through LLVM, where **both** calling convention and allocation are
part of the type system (conventions LLVM's CC enum cannot express, and pointers
whose region is part of their type), and the higher-level surface is grown with
**real Lisp macros** — a compile-time interpreter, not template substitution.

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
- M3: allocation as types. A pointer's **region** is part of its type:
  `(ptr frame)` → `alloca`, `(ptr static)` → a global, `(ptr heap)` →
  `malloc`/`free`. The checker enforces region soundness — `frame` pointers may
  not cross a function boundary, and `free` only accepts a `heap` pointer.
- Macros: `defmacro` / `def` run in a compile-time Lisp (quasiquote `` ` ``,
  unquote `~`, splicing `~@`, `gensym`, list/symbol builtins, recursion,
  helper functions). Macros can compute, recurse, and emit whole top-level
  definitions, and they compose with conventions and regions. Inspect any
  expansion with `coil --expand`.

Not yet: the `adapt` macro (general convention-to-convention trampolines),
closures derived from (convention × allocation) (M4), richer pointee types, a
macro standard library. See the design doc.

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
cargo test                                  # 21 end-to-end tests
cargo run -- examples/fib.coil              # => 55
cargo run -- examples/conventions.coil      # => 42  (native fastcc)
cargo run -- examples/shim.coil             # => 42  (exotic rax/rdx convention)
cargo run -- examples/allocation.coil       # => 42  (frame/static/heap regions)
cargo run -- examples/macros.coil           # => 41  (user-defined macros)
cargo run -- --expand examples/macros.coil  # show macro expansion
cargo run -- --emit-ir examples/shim.coil   # show the trampoline + inline asm
```

[inkwell]: https://github.com/TheDan64/inkwell
