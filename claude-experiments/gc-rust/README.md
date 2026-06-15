# gc-rust

A fast, **monomorphized**, garbage-collected systems language with Rust-like
syntax and OCaml-weight ergonomics. Rust's nice parts — `struct`/`enum`/`match`/
traits/generics, value types, explicit integer widths and signedness — without
the machinery that exists only to serve borrow-checking. No lifetimes, no
borrows, no `Box`. A precise copying GC owns memory; generics monomorphize so
value types stay flat and fast.

> Status: **real and running.** The full pipeline — lex → parse → resolve →
> typecheck → **monomorphize** → core IR → LLVM → JIT — executes today for:
> scalars + arithmetic (signed/unsigned), `if`/`while`/`loop`/`match`, recursion,
> **heap structs + enums with a real copying GC** (`examples/shapes.gcr`), and
> **monomorphized generics** (`fn dup<T>` becomes separate native functions per
> concrete type — no boxing). The GC relocates live objects under stress with
> precise roots, all from compiler-generated frame code. Traits, closures,
> `Vec`/`String`/stdlib, and the benchmark suite are next. See `PLAN.md`.

## Try it

```
cargo run --bin gcr -- run examples/fib.gcr      # prints 2178309
cargo run --bin gcr -- parse examples/types.gcr  # item summary
cargo run --bin gcr -- check examples/types.gcr  # resolve + report
cargo test                                        # GC + frontend + codegen
```

## What's here

| Piece | File(s) | State |
|---|---|---|
| Garbage collector | `src/gc/` | reused from ai-lang, self-contained, 92 tests green |
| Codegen↔GC ABI | `src/runtime.rs` | trimmed; precise shadow-stack frames + safepoints |
| Lexer | `src/lexer.rs` | done |
| Parser → surface AST | `src/parser.rs`, `src/ast.rs` | done (fns, structs, enums, traits, impls, generics, closures, match) |
| Name resolution | `src/resolve.rs` | done |
| Types + checking | `src/types.rs`, `src/lower.rs` | scalar subset done; heap/generics in progress |
| Monomorphic core IR | `src/core.rs` | frozen contract (`docs/core-ir.md`) |
| LLVM codegen | `src/codegen.rs` | scalar subset runs; GC frames next |

## Design

- `docs/language.md` — the surface language (frozen v0).
- `docs/core-ir.md` — the monomorphic IR between the middle-end and codegen.
- `docs/gc.md` — the borrowed collector and the exact codegen↔GC contract.
- `PLAN.md` — the build plan and phases.

## The one borrowed thing: the GC

Only the garbage collector is reused (from the ai-lang project): a precise,
semi-space **copying** collector with a shadow-stack root system and a
cooperative safepoint protocol that integrates with LLVM-compiled code. It
benchmarks at or above malloc/free Rust on allocation-heavy workloads. Nothing
else from that project is used — gc-rust is its own language.
