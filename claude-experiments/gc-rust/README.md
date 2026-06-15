# gc-rust

A fast, **monomorphized**, garbage-collected systems language with Rust-like
syntax and OCaml-weight ergonomics. Rust's nice parts — `struct`/`enum`/`match`/
traits/generics, value types, explicit integer widths and signedness — without
the machinery that exists only to serve borrow-checking. No lifetimes, no
borrows, no `Box`. A precise copying GC owns memory; generics monomorphize so
value types stay flat and fast.

> Status: **real, running, and fast — and it builds standalone native
> executables.** `gcr build app.gcr -o app` emits a native object via LLVM and
> links it with the GC runtime (a 24 MB LLVM-free static lib) into a standalone
> binary. `gcr run` JITs. The pipeline — lex → parse → resolve → typecheck →
> **monomorphize** → core IR → LLVM (O2) → JIT/AOT — handles: scalars
> (signed/unsigned), floats + `sqrt`, `if`/`while`/`loop`/`for`/`match`,
> recursion, **heap structs + enums + a real copying GC**, **monomorphized
> generics** (no boxing), **value types fully flat** (`value struct`/`value
> enum` + tuples, passed in registers), **methods + trait dispatch with checked
> bounds**, `Option`/`Result` + **`?`**, **closures**, **arrays + a growable
> `Vec`**, and **exhaustiveness-checked `match`**. The GC relocates live objects
> under stress with precise roots, and the GC-stress suite stays green under
> aggressive LLVM optimization (the optimizer respects the root invariants).
> See `docs/tour.md` for a guided tour.
>
> **The headline benchmarks** (bit-identical checksums vs Rust; gc-rust time
> *includes* compile+JIT):
>
> | benchmark | Rust `-O` | **gc-rust** (O2 JIT) | |
> |---|---|---|---|
> | nbody, 50M f64+sqrt iters | 0.76s | **0.54s** | faster |
> | nbody w/ `value struct Vec3` **by value**, 5M iters | 0.245s | **0.085s** | **~2.9× faster** |
> | binary_trees (40× depth-16, 5.2M heap allocs) | 0.077s | 0.115s | 1.5× (GC vs malloc) |
>
> The `value struct Vec3` version is the thesis in one line: a value type passed
> by value through `add`/`scale`/`dot` compiles to a flat 3×f64 aggregate in
> registers — zero-cost abstraction, no boxing, no heap on the hot path, with a
> precise GC available but untouched. ai-lang's equivalent was 9.5× *slower*
> than Rust because it boxed generics; gc-rust monomorphizes + flattens value
> types, so it matches or beats Rust.
>
> Also working: **closures** (captures + indirect call), **arrays**
> (`array_new`/`get`/`set`/`len`, GC-traced for reference elements), a
> **growable `Vec`** written in gc-rust itself (`examples/vec.gcr` — reallocates
> and copies on grow, GC-managed), and `print_int`/`print_float`. String ops and
> tuples are next. See `PLAN.md`.

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
