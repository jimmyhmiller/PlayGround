# gc-rust

A fast, **monomorphized**, garbage-collected systems language with Rust-like
syntax and OCaml-weight ergonomics. Rust's nice parts — `struct`/`enum`/`match`/
traits/generics, value types, explicit integer widths and signedness — without
the machinery that exists only to serve borrow-checking. No lifetimes, no
borrows, no `Box`. A precise copying GC owns memory; generics monomorphize so
value types stay flat and fast.

**Immutable by default — for *all* values.** Bindings, struct fields, array
elements, function parameters, and method receivers are all immutable unless
declared `mut`. Even without ownership, you keep real control over mutation:
`let mut x` to rebind, `mut self`/`mut p` to mutate a receiver or parameter, and
the rule is *transitive* — mutating `p.x` or `a[i]` requires the access path's
root binding to be `mut`. See [`docs/mutability.md`](docs/mutability.md).

> Status: **a usable language with a generational GC, built end to end.**
> `gcr run` JITs; `gcr build app.gcr -o app` emits a native object via LLVM and
> statically links the LLVM-free GC runtime into a standalone binary. A project
> can be a directory with a `gcr.toml`. The pipeline is lex → parse → resolve →
> typecheck → **monomorphize** → core IR → LLVM (O2) → JIT/AOT.
>
> **The language today:** scalars (signed/unsigned ints, floats + `sqrt`),
> `if`/`while`/`loop`/`for`, recursion; **heap structs + enums** on a precise
> **generational GC**; **monomorphized generics** (no boxing); **value types
> fully flat** (`value struct`/`value enum` + tuples, passed in registers);
> **methods + trait dispatch with checked bounds**; user-implementable `Eq`/`Ord`
> traits; `Option`/`Result` + **`?`**; **closures** (captures + indirect call);
> **complete `match`** (enum variants, integer/string/bool/char literals,
> scalar matching, named bindings, **guards**, and guards combined with variant
> patterns), exhaustiveness-checked; **UTF-8 `String`s** with a string stdlib
> (`split`/`index_of`/`pad`/`repeat`/`reverse`/…) plus a **Unicode code-point
> API** (`str_chars`/`char_to_str`/`str_len_chars`/`char_at`/`str_sub_chars` —
> character-aware length, indexing, and boundary-safe slicing over the byte
> storage); **`Vec<T>`** with a broad API
> (`first`/`get_opt`/`concat`/`slice`/`take`/`drop`/`swap`/`find`/`position`),
> a generic **`HashMap<K, V>`** (any `K: Eq + Hash`) plus the string-keyed
> **`MapStr<V>`**, and **functional iterators** (`vec_map`/`filter`/`fold`/…),
> all written in gc-rust itself; a **`Display` trait** with generic `print`/
> `print_line`/`str_of` and newline-free `print`; a **module system**
> (`mod`/`use`/`pub`, multi-file projects); **wrapping integer overflow** +
> opt-in checked arithmetic; and **immutable-by-default** mutability enforced
> through the access path. See `docs/tour.md` for a guided tour.
>
> **Benchmarks vs Rust** — both compiled to native code in **release / `-O3`**;
> every benchmark produces a **bit-identical checksum** to its Rust twin
> (verified). gc-rust is AOT (`gcr build`). Best of 9 runs:
>
> | benchmark | what it stresses | Rust `-O3` | **gc-rust (AOT)** | ratio |
> |---|---|---|---|---|
> | fib(38) | recursion + int arithmetic | 66 ms | 74 ms | 1.12× |
> | nbody (5M) | f64 + `sqrt`, scalar | 58 ms | **57 ms** | **0.98×** |
> | nbody_vec3 (5M) | `value struct Vec3` by value | 69 ms | 75 ms | 1.08× |
> | mandelbrot (800×500) | f64 hot loop | 33 ms | 41 ms | 1.24× |
> | binary_trees | **GC** alloc + trace | 54 ms | 151 ms | 2.79× |
>
> **The honest read:** compute-bound code is at parity with Rust (nbody scalar
> even edges it out) — the monomorphized value types compile to flat register
> aggregates with no abstraction penalty, on the same LLVM backend. The one
> outlier is `binary_trees` at 2.8×: that is the cost of a *garbage collector vs.
> Rust's malloc/free*, on the single workload that is pure allocation + tracing.
> 2.8× for a relocating generational GC against zero-overhead manual memory is
> the expected price of automatic memory management. The `value struct Vec3`
> result is the thesis in one line: a value type passed by value through
> `add`/`scale`/`dot` is a flat 3×f64 aggregate in registers — zero-cost
> abstraction, no boxing, no heap on the hot path. (ai-lang's equivalent was
> ~9.5× *slower* than Rust because it boxed generics.)

## Try it

```
cargo run --bin gcr -- run examples/fib.gcr           # prints 2178309
cargo run --bin gcr -- run examples/stdlib.gcr        # Vec/MapStr/iterators tour
cargo run --bin gcr -- run examples/match.gcr         # literals, guards, enum patterns
cargo run --bin gcr -- run examples/project           # a gcr.toml project (multi-file)
cargo run --bin gcr -- run examples/binary_trees.gcr --gc-stress  # collect every alloc
cargo run --bin gcr -- check examples/types.gcr       # full typecheck + diagnostics
cargo run --bin gcr -- emit mono examples/stdlib.gcr  # monomorphization table (JSON)
cargo run --bin gcr -- emit layout examples/stdlib.gcr # GC heap object shapes (JSON)
cargo run --bin gcr -- emit reflect examples/shapes.gcr # runtime reflection metadata (JSON)
GCR_HEAP_DUMP=1 cargo run --bin gcr -- run examples/shapes.gcr # render the live heap graph (docs/reflection.md)
# emit stages: tokens | ast | core | layout | reflect | mono | llvm — structured IR for tooling/agents
cargo run --bin gcr -- eval examples/fib.gcr          # scratchpad: run + report {value,type} JSON
printf '1 + 2 * 3\n' > /tmp/x.gcr && cargo run --bin gcr -- eval /tmp/x.gcr  # bare expr → main synthesized
cargo test                                             # frontend + codegen + GC
./scripts/run_examples.sh                              # every example, normal + --gc-stress
```

**AOT (`gcr build`)** statically links the GC runtime, so it needs the
`gcrust-rt` *staticlib*, which a plain `cargo build` doesn't produce. Build the
compiler + runtime together with the provided alias, then build native binaries:

```
cargo gcr-release                                      # builds gcr + the release GC runtime
./target/release/gcr build examples/fib.gcr -o fib     # standalone native binary
./fib; echo $?                                         # 2178309 & 0xFF = 5
```

(`cargo gcr-build` does the same for a debug build. `$GCRUST_RUNTIME_LIB`
overrides the linked runtime if you need a specific one.)

A **project** is a directory with a `gcr.toml`:

```toml
[package]
name = "calculator"
version = "0.1.0"
entry = "src/main.gcr"
```

`gcr run <dir>` reads the manifest and builds the entry file (which may `mod`
sibling files). Integer arithmetic **wraps** on overflow (defined, like
Java/Go/OCaml — see `docs/overflow.md`); `checked_add_i64`/`checked_mul_i64`
return `Option` for opt-in detection.

## What's here

| Piece | File(s) | State |
|---|---|---|
| Garbage collector | `crates/gcrust-rt/src/gc/` | generational (nursery + minor/major GC + write barrier); semi-space major collector; 92 tests green |
| Codegen↔GC ABI | `crates/gcrust-rt/src/runtime.rs` | precise shadow-stack frames + safepoints; `ai_gc_*` allocation + write-barrier externs |
| Lexer | `src/lexer.rs` | done |
| Parser → surface AST | `src/parser.rs`, `src/ast.rs` | done (fns, structs, enums, traits, impls, generics, closures, match, modules); recursion-depth bounded |
| Name resolution + modules | `src/resolve.rs` | done (fully-qualified names, `pub` visibility, `use` aliases) |
| Types + checking + mono | `src/types.rs`, `src/lower.rs` | done (bidirectional check, generic instantiation, exhaustiveness, multi-error reporting) |
| Monomorphic core IR | `src/core.rs` | done (`docs/core-ir.md`) |
| LLVM codegen (JIT + AOT) | `src/codegen.rs` | done (GC frames, safepoints, value types, closures, strings, write barriers) |
| Standard library | `src/prelude.gcr` | `Option`/`Result`, `Vec`, `MapStr`, iterators, `String` ops, `Eq`/`Ord` — written in gc-rust |
| Project manifest | `src/manifest.rs` | `gcr.toml` (`[package]` name/version/entry) |

Test counts: 156 frontend/codegen unit tests + 92 GC tests + module, AOT,
fuzz, generational, and ABI integration suites.

## Design

- `docs/language.md` — the surface language (frozen v0).
- `docs/core-ir.md` — the monomorphic IR between the middle-end and codegen.
- `docs/gc.md` — the borrowed collector and the exact codegen↔GC contract.
- `docs/mutability.md` — immutable-by-default rules (`mut` at every binder).
- `docs/modules.md` — the module system (`mod`/`use`/`pub`, multi-file).
- `docs/overflow.md` — integer overflow semantics (wrapping) + checked helpers.
- `PLAN_PROD.md` — the production build plan and phases.

## The one borrowed thing: the GC

Only the garbage collector is reused (from the ai-lang project): a precise,
**generational** collector — a bump-allocated nursery (young generation)
scavenged by cheap **minor** collections that promote survivors to a semi-space
**copying** tenured (old) generation, with a card-table **write barrier** so a
minor GC finds old→young pointers without scanning all of tenured. It uses a
shadow-stack root system and a cooperative safepoint protocol that integrates
with LLVM-compiled code, and relocates live objects under stress with precise
roots. Set `GCR_GC_STATS=1` to print minor/major collection counts (e.g.
`binary_trees` runs 12 minor + 0 major). `--gc-stress` runs the single-
generation semi-space collector with collect-on-every-allocation as the
strongest relocation-correctness check. Nothing else from ai-lang is used —
gc-rust is its own language. See [`docs/gc.md`](docs/gc.md).
