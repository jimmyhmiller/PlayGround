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

**M0 + M1 implemented** — an s-expression front end that JIT-compiles through
LLVM, with calling conventions as first-class data that actually drive codegen.

- M0: reader → core AST → LLVM codegen → JIT. `i64`, `let`, arithmetic,
  comparisons, `if`, direct + recursive calls.
- M1: `defcc` conventions attached to functions; the `:native` lowering sets the
  real LLVM calling convention on the function **and every call site**; the
  checker rejects mismatched arity, unbound vars, and conventions with no
  lowering yet (the `:shim` path is M2).

Not yet: allocation/region types (M3), adapters/trampolines + `:shim` lowering
(M2), macros, multiple types. See the roadmap in the design doc.

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
cargo test                                  # 6 end-to-end tests
cargo run -- examples/fib.coil              # => 55
cargo run -- examples/conventions.coil      # => 42
cargo run -- --emit-ir examples/conventions.coil   # show the LLVM IR
```

[inkwell]: https://github.com/TheDan64/inkwell
