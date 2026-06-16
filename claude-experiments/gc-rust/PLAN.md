# gc-rust — a fast, monomorphized, GC'd systems language

> **Historical document.** This was the *initial* build plan (bring-up of the
> compiler + GC). It has been fully executed and then some; the language is now
> a working, usable system. For the production roadmap that followed (and is also
> complete), see `PLAN_PROD.md`; for the current state, see the README and
> `docs/`.

## What this is (and is NOT)

**gc-rust is a brand-new language.** It is *not* ai-lang and shares none of
ai-lang's semantics: no content-addressing, no codebase store, no `def`, no
`resolve`/`codec`/`edit`, no Unison anything.

The language we want:

- **Rust-like surface syntax**, OCaml-weight feel (lighter than Java, no
  ceremony), `fn`/`struct`/`enum`/`match`/traits/generics.
- **Fast. Monomorphized.** Generics specialize per concrete type (true Rust
  codegen model). Value types stored inline, no boxing for concrete types.
- **Control over representation**: `u8/u16/u32/u64/i*`, unsigned types, value
  types stored flat, explicit box/unbox distinction where it matters.
- **GC instead of borrow-checking.** No lifetimes, no borrows, no `Box`
  ceremony — the GC owns memory. We keep all the *good* parts of Rust syntax
  that exist for reasons *other* than memory management.
- **Traits monomorphic** (static dispatch by default; `dyn` later, optional).
- **AOT compiled via LLVM/Inkwell** (also JIT-capable, same backend).
- **NOT** copying ai-lang's async/distribution/`at()`/`state`/`serve` story.

## The ONE thing we copy: the GC strategy + its LLVM integration

ai-lang's `src/gc/` is a **fully self-contained module** (verified: zero
`crate::` imports outside `gc::`). It is a production-grade, tested,
benchmarked GC library. We lift it as-is and adapt only what's needed.

What we copy from ai-lang:

| File | What it is | Action |
|---|---|---|
| `src/gc/*.rs` (15 files) | semi-space copying collector, headers, type_info, scan, alloc (bump + atomic + `AllocWindow`), **precise root frames** (`roots.rs` shadow-stack), **safepoint/thread protocol** (`thread.rs`), SATB + card-table barriers (dormant, for future concurrent/gen GC), statemap tracing | **Copy verbatim**, then trim ai-lang-specific bits |
| `runtime.rs` `Thread`/`Frame`/`FrameOrigin`/`AllocWindow` ABI + `thread_offsets`/`frame_offsets` | the fixed-offset structs JIT'd code reads; the **contract** between codegen and GC | **Copy the GC-essential fields, redesign the rest** |
| codegen patterns: `emit_prologue`, `finalize_frame_zeroing`, `emit_safepoint_poll`, root-slot spill/reload, `FrameOrigin` emission | how LLVM IR participates in precise GC | **Re-implement** (our codegen is new, but follows these proven patterns exactly) |

What we explicitly DROP from ai-lang's runtime/codegen:

- `code_table` / closure `code_hash` dispatch (content-addressed) → gone.
- `boxed_int_ti`, `atom_ti`, `prim_array_ti` uniform-boxed machinery → gone;
  we monomorphize, so there's no uniform-boxed-Int.
- Result dual-register ABI is a *good* idea we may re-derive later, but it's
  not GC and not v1-critical.

### Why this GC is the right one to copy

- **Precise, not conservative.** Roots are an explicit shadow-stack of frames
  (`Frame { parent, origin, [roots; N] }`); `FrameOrigin.num_roots` tells the
  collector exactly how many slots to scan. No stack scanning, no false
  retention. This is the Fil-C / "shadow stack" model and it composes with
  LLVM cleanly (no statepoint intrinsics needed).
- **Safepoint handlers done right** (your core requirement): every loop back-edge
  emits a poll of `thread.state` (volatile byte); when GC requests a stop, the
  mutator traps into `ai_gc_pollcheck_slow`, parks at a safepoint with its
  frame chain published, GC runs, mutator resumes. Re-entering the runtime
  (any `ai_gc_alloc_*`, any call into Rust) spills live roots to frame slots
  first, so a collection mid-call relocates them correctly. This is the
  "whenever you're interrupting with the runtime, it's all good" guarantee.
- **Relocation-safe interface.** Allocation can move objects (copying
  collector); the spill/reload-around-allocation discipline + opaque-pointer
  GEPs make every codegen site correct under relocation.
- **Proven fast.** ai-lang's `binary_trees` (the GC-stress benchmark) runs at
  **0.54× Rust / 1.02× Go** — i.e. *faster than malloc/free Rust*, on par with
  Go's generational GC. The semi-space bump-allocator + cheap precise roots is
  why.

## Architecture of the new compiler

```
source (.gcr files, conventional modules)
  │  lexer  → tokens
  │  parser → surface AST (spans, names, full type syntax)
  │  name resolution + module system (files-as-truth, `mod`/`use`)
  │  type inference + checking (Hindley-Milner-ish + traits, explicit prim types)
  │  MONOMORPHIZATION (specialize generics + trait impls per concrete type)
  │  lowering → typed core IR (monomorphic, explicit reprs, explicit box/unbox)
  │  codegen → LLVM IR (Inkwell) with precise GC frames + safepoints
  └→ AOT object/exe  (or JIT via ExecutionEngine — same module)
        │
        └── links against the copied GC runtime (libgcrt: src/gc/ + runtime ABI)
```

## Representation model (the "fast fast fast" part)

- **Primitives unboxed, inline, native width**: `i8..i64`, `u8..u64`, `f32`,
  `f64`, `bool`, `char`. Unsigned is a real distinction (affects div/shift/cmp).
- **`struct` is a value type by default** — stored inline in its container, no
  header, no indirection (like Rust). Lives on the GC heap only when boxed into
  a reference or when it contains/escapes as a reference.
- **Reference types** (heap-allocated, GC-traced) get the object header
  (`Compact` 8B or `Full` 16B) + a `TypeInfo` describing field layout so the
  scanner knows which slots are pointers.
- **`enum`**: monomorphic tagged union; niche/tag optimization later; payload
  inline when value-typed.
- **Generics**: monomorphized → each instantiation is its own concrete type
  with its own `TypeInfo`. `Vec<Vec3>` stores `Vec3`s flat. **This is what
  closes ai-lang's nbody gap** (its 9.5× slowdown was uniform-boxed generics;
  we don't have those).
- **Traits**: static dispatch, monomorphized impls. `impl Trait` / generic
  bounds resolve at monomorphization. `dyn Trait` (vtable) is a later opt-in.
- **Box/unbox where it matters** (your JVM-ish ask): a `ref T` / `Gc<T>` type
  for explicit heap indirection + sharing/cycles; value types stay flat unless
  you ask for a reference. Arrays of value-typed elements are flat (unboxed),
  arrays of references are pointer arrays (traced).

## Build phases (designed for parallel agent execution — "ultra plan")

### Phase 0 — Substrate (the GC copy). *Sequential, foundational.*
1. `cargo init`; `Cargo.toml` with `inkwell` (llvm21-1) + `blake3` (for any
   incidental hashing) — but NOT ai-lang's FFI deps.
2. Copy `ai-lang/src/gc/` → `gc-rust/src/gc/` verbatim. Confirm it compiles
   standalone (it has its own `tests.rs` — run them: GC must be green before
   anything else).
3. Author `src/runtime.rs`: the **trimmed** `Thread`/`Frame`/`FrameOrigin`/
   `AllocWindow` ABI (GC-essential fields only) + the `ai_gc_*` extern fns
   (`alloc_fixed`, `alloc_varlen`, `pollcheck_slow`, `gc_collect`) that
   codegen will call. Offset asserts must hold.
4. A tiny hand-written LLVM-IR smoke test (or `inkwell` unit) that allocates,
   roots, forces a collection, and survives — proves the ABI end-to-end before
   we have a parser.

### Phase 1 — Frontend. *Parallelizable across agents.*
- **1a** lexer + tokens.
- **1b** surface AST + parser (Rust-like grammar: `fn`, `let`, `struct`,
  `enum`, `impl`, `trait`, `match`, generics `<T>`, `where`, closures `|x|`,
  literals incl. typed integer suffixes `1u32`).
- **1c** module system + name resolution (`mod`, `use`, paths, files-as-truth).
- Deliverable: parse all of a `examples/` corpus into a clean AST.

### Phase 2 — Types & monomorphization. *Sequential-ish, the hard core.*
- **2a** type representation + inference/checking (explicit primitive widths,
  signedness, trait bounds). 
- **2b** trait resolution (impl selection, coherence-lite).
- **2c** **monomorphization pass**: collect instantiations from `main`'s
  reachable graph, specialize generic fns + trait methods + types, produce a
  monomorphic core IR with concrete `TypeInfo` per reachable type.
- **2d** lowering to typed core IR (explicit reprs, box/unbox decisions,
  closure capture classification ptr-vs-scalar for the GC).

### Phase 3 — Codegen + GC integration. *The payoff. Built against Phase 0 ABI.*
- **3a** LLVM module setup, type lowering (struct/enum → LLVM aggregates, refs
  → ptr), function ABI.
- **3b** **GC frames**: `emit_prologue` / `finalize_frame_zeroing` /
  `emit_epilogue`, `FrameOrigin` globals, root-slot spill/reload around every
  allocation and every runtime call (port ai-lang's exact discipline).
- **3c** **safepoints**: `emit_safepoint_poll` at loop back-edges; wire
  `ai_gc_pollcheck_slow`.
- **3d** expr/stmt lowering: arithmetic (signed/unsigned correct),
  calls, `match` lowering (tag switch + payload extract), struct/enum
  construction + field access, closures (lifted fns + capture env objects),
  arrays (flat value arrays + ref arrays, both traced correctly).
- **3e** AOT driver (emit object, link) + JIT driver (ExecutionEngine), `main`.

### Phase 4 — Make it a *real* language (production-ready from day one).
*Heavily parallelizable — one agent per stdlib area.*
- Core stdlib (monomorphic, written in gc-rust where possible): `Option`,
  `Result`, `Vec<T>`, `String`, `HashMap<K,V>`, slices, iterators, `?`
  operator, `print`/formatting.
- A real test suite (unit + golden-output examples) + the 5 benchmark programs
  ported (`fib`, `loop_mix`, `mandelbrot`, `nbody`, `binary_trees`) so we can
  prove the perf story vs Rust/Go — **nbody is the headline**: monomorphized
  value-type arrays should put us near 1× Rust, not 9.5×.
- GC stress mode (port ai-lang's `gc_every_alloc`) wired into CI.
- Docs: a `README` with the language tour + a `docs/gc.md` explaining the
  borrowed collector + root-frame contract.

## Parallelization strategy ("ultra plan", build in parallel)

- Phase 0 is the gate — it must land first and be green (GC tests + ABI smoke
  test) before fan-out.
- Then Phases 1, and the *design* of 2 and 3, run as parallel agents against
  shared, frozen interfaces:
  - Surface AST type (1b) is the contract between lexer/parser/resolver.
  - Core IR type (2d) is the contract between the type/mono pipeline and
    codegen — freeze its shape early so Phase 3 agents can build against a
    stub.
- Phase 4 stdlib areas are independent → one agent each.
- Each agent works to a typed interface + tests; integration happens at the
  frozen boundaries.

## Definition of done for v1

- `gc-rust build examples/nbody.gcr && ./nbody` runs, GC-clean under stress,
  and the benchmark table shows nbody within ~1.2× of Rust (vs ai-lang's
  9.5×) — proving monomorphized value types + the borrowed GC together.
- All 5 benchmarks + the example corpus compile and produce correct output.
- A usable stdlib (Vec/String/HashMap/Option/Result/iterators) and the GC
  passes its stress suite.

## Open design questions to settle as we build (not blocking the plan)

- Exact integer-overflow semantics (wrapping vs checked vs UB-free trap).
- `Gc<T>` explicit-reference syntax & whether cycles need anything beyond the
  copying collector (semi-space handles cycles for free — a plus).
- Closures: do we monomorphize closure types or use a uniform fn-ptr+env? (lean
  monomorphic for hot paths, fn-ptr for stored/dynamic.)
- When (if) to turn on the dormant generational/concurrent GC paths we're
  inheriting.
