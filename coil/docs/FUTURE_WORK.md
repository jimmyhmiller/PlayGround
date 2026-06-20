# Coil — Future Work: the road to a Zig/C competitor

This document is an honest, comprehensive map of what Coil would need to become a language you'd actually reach for instead of C or Zig. It is deliberately broad: it lists far more than any one person will build soon, organized so the high-leverage work is easy to find. It also argues a thesis — **don't clone Zig; lean into the things Coil can do that Zig and C cannot** — and then enumerates the table-stakes work needed to make those strengths usable.

---

## 1. Where Coil stands today

The hard compiler core is done and is genuinely competitive:

- **Native AOT, fully optimized.** Emits an object, links with `cc`, runs LLVM's `-O3` pipeline. On matched compute benchmarks it sits at parity with `cc -O3` (`bench/`).
- **Calling convention is a type.** `defcc` with a `:native` lowering (rides an LLVM CC) or a `:shim` lowering (naked trampoline + register-constrained call sites), emitted per architecture (x86-64, AArch64). Nothing else exposes this.
- **Allocation is a value, not a keyword.** Region-less pointers (`(ptr T)`); `alloc-stack/static/heap`; Zig-style allocators threaded as vtable values (`lib/alloc.coil`); an IO capability the same way (`lib/io.coil`).
- **Layout is a type** (the dual of calling convention): `:c`/`:packed`/`(align N)`/`:explicit`/`:bits`, with `sizeof`/`alignof`/`offsetof`/`static-assert`.
- **Real metaprogramming.** A compile-time Lisp (`defmacro`, quasiquote, gensym hygiene) that can emit whole top-level definitions and branch on the target.
- **A module system** with namespaced functions/types/sums/conventions, file-relative `import` (`:as`/`:use`), `(export …)` visibility, and **proper cross-module macro hygiene** (references resolve in the macro's defining module, including the macro-generated, second-order case).
- **Raw LLVM IR as a first-class escape hatch.** `(llvm-ir …)` inlines arbitrary LLVM IR with zero overhead; on top of it, SIMD (`(vec T N)` + `lib/simd.coil`) is a *library*. A consequence we proved: a C function from `clang -emit-llvm` pastes straight into Coil and runs identically.
- **C interop**: `extern` (incl. variadic), and struct-by-value across the real C ABI (SysV AMD64 + AArch64 AAPCS64), verified against clang.
- **Generics** by monomorphization with inference; **references & mutability** as const-correctness (no borrow checker); arbitrary-width integers; floats; bool; bitwise.
- 162 tests; a benchmark harness; a kitchen-sink example.

What this means: Coil is a sharp *core*, not yet a *tool*. The gap to "Zig/C competitor" is almost entirely livability, library, safety-in-debug, tooling, and reach — not deep compiler theory.

---

## 2. Strategy: the moat — lean into what only Coil can do

Before the table-stakes list, the strategic point. Zig is excellent; out-copying it on its own terms is a losing race. Coil has five capabilities that are unusual or unique, and the roadmap should make *those* first-class, because they're the reason someone would switch.

1. **Calling-convention-as-type.** Hand-rolled ABIs, syscall conventions, interrupt/exception handlers, naked functions, JIT trampolines, register-pinned hot paths, FFI to anything. This is the killer feature for **kernels, embedded, runtimes, and interop**. No mainstream systems language gives you this in the source language.
2. **Raw LLVM IR + C embedding.** Coil can host arbitrary LLVM IR and therefore host C (proven). That opens *being a C compiler* (`coil cc`), **mixing C and Coil in one module with cross-language inlining**, and reaching every LLVM instruction/intrinsic without compiler changes. This is a genuine moat versus Zig's `@cImport`.
3. **Allocation-as-values** (already Zig's best idea) plus **layout-as-types** (the dual). Together: explicit, verified control over where data lives *and* how it's shaped — ideal for wire formats, MMIO, protocols, and zero-copy parsing.
4. **A real Lisp macro system** that emits definitions and DSLs. Many "language features" become libraries (closures, SIMD, and — on the roadmap — `defer`, iterators, async, even safety checks). The language can grow without the compiler growing.
5. **Descriptive, not safe — by choice.** Coil deliberately has no borrow checker. The Zig-style answer (runtime safety checks in debug, off in release) fits Coil perfectly and is far less work than a borrow checker.

The roadmap below is in two halves: **table stakes** (the unglamorous work that makes any of this usable) and **lean-in** (the differentiators). Ship table stakes first; they unblock everything, including dogfooding the differentiators.

---

## 3. Table stakes — make it livable

### 3.1 Diagnostics (the single highest-leverage work)

Today a type error says `in 'main': arithmetic on different types (f64 vs i64)` with **no line, column, or source snippet**: the tokenizer tracks no positions. Everything else is more painful until this exists.

- **Source spans** end to end: byte offsets in the tokenizer → spans on `Sexp` → spans on the AST → spans in errors. Macro-generated nodes carry the expansion's provenance (the macro call site + the template location).
- **Rich rendering**: `file:line:col`, the offending source line, a caret/underline, and the existing context-frame "stack trace" (function → argument → macro expansion → import) with a location per frame.
- **Multi-error reporting** (don't stop at the first) and basic recovery.
- This also unlocks the LSP (§5) nearly for free.

### 3.2 Debug info (DWARF)

Emit line tables + variable/type info via inkwell's `DIBuilder`, so `lldb`/`gdb`, breakpoints, stepping, and crash backtraces work. Coil already emits native objects; this is the missing metadata. Big multiplier for real debugging.

### 3.3 Core control flow & error ergonomics

TCO already turns self-tail-recursion into a loop, so most of this is library/macro work:

- **Loops**: `while`, `for` (over ranges and slices), `loop`/`break`/`continue`, `inline for` (comptime-unrolled). At least as macros; consider blessing `while`/`for` for nicer diagnostics.
- **`defer` / `errdefer`** for deterministic cleanup (scope guards) — the design already calls this out.
- **`?` / `try`** over `Option`/`Result` to remove the match-pyramid; consider Zig-style **error sets** as a `defsum` convention.
- **`switch`** with exhaustiveness, ranges, and payload captures over sums/ints.
- A few classics: `cond`, `->`/`->>`, `when`/`unless`, `match` guards.

### 3.4 The everyday type tier

- **Slices**: a `(slice T)` fat pointer `{ ptr, len }` — the single biggest ergonomics/safety win. Indexing, iteration, sub-slicing, `len`. Optional bounds checking in debug (§4). String literals become `(slice u8)`/a string type.
- **A real string type** (length-carrying, UTF-8 aware) — today only C-strings (`(ptr i8)`).
- **Optionals as a niche**: `?T` with null-pointer/niche optimization (`?*T` is pointer-sized), exhaustive unwrap.
- **Sentinel-terminated arrays/pointers** (`[*:0]u8`-style) for C interop.
- **Tuples / anonymous structs**, **enums with explicit values**.
- **`comptime`-known array lengths** and array/slice literals.

### 3.5 A real standard library

This is the largest *volume* of work and most of it is library code now that generics + allocators + the primitives exist. Stage it:

- **mem**: copy/set/eql/move, alignment helpers, `@memcpy`/`@memset` intrinsics.
- **Collections (allocator-aware)**: `ArrayList` (dynamic array — the `vector.coil` example is the seed), `HashMap`/`ArrayHashMap`, `StringHashMap`, sets, sorted maps, ring buffers, priority queue, intrusive lists.
- **Formatting & IO**: a `std.fmt`-style `format`/`print` (positional + named, width/precision), buffered readers/writers, files, `stdin/stdout/stderr` (basic versions exist).
- **OS**: filesystem, process, env, `args` (exists), time/clock, paths, `mmap`.
- **Math**: libm bindings or LLVM intrinsics (`sqrt`, trig, `fma`), `min/max/clamp`, checked/saturating/wrapping arithmetic, big integers, PRNG.
- **Encoding**: UTF-8, JSON, base64, hex (a SIMD JSON parser is a natural showcase — see §6).
- **Algorithms**: sort, search, partition.

### 3.6 In-language testing

Zig-style `test "name" { … }` blocks compiled and run by `coil test`, with an `assert`/`expect`, per-test isolation, and a summary. Coil currently tests from Rust; in-language tests are needed to dogfood the stdlib and for users.

---

## 4. Safety in debug, speed in release (the Zig promise — cheap for Coil)

Coil chose *descriptive, not memory-safe*. The right safety story is therefore **opt-in runtime checks compiled out of release builds**, not a borrow checker. This is the feature that makes Zig feel safe without a GC, and it's mostly codegen + a build-mode flag.

- **Build modes**: `Debug`, `ReleaseSafe`, `ReleaseFast`, `ReleaseSmall` (map to opt level + which checks are emitted).
- **Checks** (debug/safe only): slice/array **bounds**, integer **overflow** (LLVM `*.with.overflow` intrinsics → trap), **null/optional** unwrap, **alignment** on casts, division by zero, unreachable, optionally **shift-amount** and enum-tag validity.
- **A GeneralPurposeAllocator** with leak detection, double-free, and use-after-free poisoning — a debug allocator that turns the allocator-as-value design into a safety tool.
- **Sanitizer integration**: thread/emit metadata so LLVM ASan/UBSan/TSan can be enabled (`-fsanitize=…` through the link step).
- **`assert`/`@panic`** with a message + backtrace (needs DWARF, §3.2).

This bundle is high-leverage and far less work than a borrow checker, and it directly answers "but is it safe?"

---

## 5. Tooling & ecosystem

- **`coil fmt`** — a formatter (cheap given the reader; huge for adoption and diffs).
- **LSP** — diagnostics (free once spans exist), hover types, go-to-def/find-refs (the module resolver already computes this), completion, rename.
- **Tree-sitter / TextMate grammar** for editor highlighting.
- **A REPL-ish `coil eval`** — Coil is AOT-only, but a compile-and-run loop (or a JIT mode via inkwell's execution engine, used only for the REPL) would help learning and scripting.
- **Docs**: a "learn Coil in 30 minutes" tutorial, a complete language reference, a stdlib reference (doc-comments → generated docs).

---

## 6. Lean-in #1 — C interoperability *dominance*

Zig's adoption is driven by `zig cc` and `@cImport`. Coil can match and then exceed this, because it already hosts LLVM IR and can embed C (we proved it).

- **`coil cc`** — drive clang/lld to compile C (and link), bundled like `zig cc`: an instant cross-compiler and drop-in C toolchain.
- **Mix C and Coil in one module** — compile C to IR, `link_in_module` it (the `(llvm-ir …)` machinery already does this), so the optimizer **inlines across the language boundary**. This is a capability `@cImport` doesn't have.
- **`cimport`** — parse C headers via libclang and auto-generate `extern` declarations + struct/enum/typedef bindings. The single biggest unlock for using the C ecosystem.
- **Export C headers** for Coil APIs so C/other languages can call Coil.
- **A C-source-as-a-module on-ramp**: a debug-info-guided lifter could raise C → more idiomatic Coil incrementally (a research-y but high-value migration story).

---

## 7. Lean-in #2 — embedded, freestanding, and bare metal

Coil's "no runtime, AOT, conventions-as-types, layout-as-types, raw IR" is *tailor-made* for systems where C and assembly dominate. This is a market where the convention/layout features are not a curiosity but the whole point.

- **Freestanding targets**: `--target …-none`, no libc, custom `_start`, `panic`/`unreachable` without an OS.
- **Linker control**: custom linker scripts, `--section`/placement attributes, `--no-std`, static-only links, `-nostartfiles`.
- **Interrupt/exception handlers** and **naked functions** as first-class (the `:shim`/naked machinery already exists; expose it cleanly + an `interrupt` calling convention).
- **MMIO**: typed volatile loads/stores (raw IR can already do `volatile`/`atomic`; give it a typed surface), `:explicit`/`:bits` layouts for device registers (already supported), `int↔ptr` for fixed addresses (already supported).
- **TLS** (thread-local storage), weak symbols, `comptime`-selected memory maps.
- **More targets**: RISC-V (32/64), ARM32/Thumb, Xtensa, AVR, WASM (`wasm32-freestanding`), and **Windows x64** (PE/COFF + the MS x64 ABI — needed for desktop reach).

A concrete, compelling demo: a tiny ARM Cortex-M blink/UART program with the device registers as `:bits` layouts and the vector table as an array of shim-convention handlers — pure Coil, no C, no assembly files.

---

## 8. Lean-in #3 — metaprogramming, comptime, and "features as libraries"

- **`adapt`** (the deferred macro): general convention-to-convention trampolines synthesized from two `defcc` descriptions — the last piece of the calling-convention story, and the foundation for FFI shims and JIT glue.
- **A comptime story that bridges macros and types.** Coil's macros are a separate Lisp; Zig's comptime is type-level computation in the *same* language. Decide and build the bridge: comptime evaluation of Coil functions, comptime type construction, and comptime values flowing into types (`Array(T, comptime_len)`). This is what makes generic data structures and reflection ergonomic.
- **Reflection / `@typeInfo`** — introspect struct fields, sum variants, function signatures at comptime; enables serialization, formatting, ORMs, and generic printing without boilerplate.
- **Interfaces / structural constraints** — bless the allocator/`Writer` vtable pattern, or add comptime structural "does T have method m" checks, so generic code documents its requirements.
- **Closures, iterators, `defer`, error sets, even bounds-checked slices** can largely be *library* macros over the core — keep the compiler small.

---

## 9. Lean-in #4 — concurrency & async (a natural fit for the thesis)

Coil's design explicitly says coroutines and async frames are points in the *(calling-convention × allocation)* space. That makes concurrency a place where Coil's core pays off rather than bolts on.

- **Typed atomics** (`atomicrmw`/`cmpxchg` already reachable via raw IR — give them a typed surface), `volatile`, memory orderings, fences.
- **Threads**: spawn/join, thread-local storage, the GeneralPurposeAllocator made thread-safe.
- **Synchronization**: mutex, condvar, once, RW-lock, channels.
- **Async/coroutines as a library**: stackless coroutines built from a custom calling convention (frame as a heap/arena allocation, resume/suspend as convention-aware calls) — the thesis realized. This is a research-grade but on-brand differentiator.

---

## 10. Backend, ABI, and codegen depth

- **Finish the calling-convention story**: `:shim` function *pointers*, aggregate-by-value across `:shim` conventions, and a safe parallel-move register scheduler in trampolines (currently assumes non-colliding source/dest).
- **More ABIs**: Windows x64, ARM32 AAPCS, RISC-V, WASM, plus unsigned-typed externs and struct/array literals across the boundary (deferred items).
- **Per-field endianness** in `:explicit` layouts (sketched in `LAYOUT.md`) — wire formats by value.
- **Typed inline assembly** surface (beyond the shim internals) and `naked`.
- **LTO** (already whole-program), **PGO**, section/symbol attributes, `inline`/`noinline`/`cold`/`hot`, function `align`.
- **A SIMD standard library**: per-arch widths via `target-arch`, the full op set (shuffles, masks, `tbl`/`pshufb`, `clmul`, `compressstore`, reductions), and a `stream … carry` looping construct for data-parallel kernels (a la the `simd-lang` JSON parser) — all macros over `(llvm-ir …)`.

---

## 11. Build system, packages, and scale

- **`build.coil`** — a build system written in Coil itself (Zig's `build.zig` is a major draw): targets, steps, options, install, cross-compile, run, test.
- **A package manager** — fetch/pin/version dependencies, a lockfile, content addressing.
- **Project conventions** — multi-file layout, a manifest, standard `src/`/`tests/`.
- **Compile speed at scale** — the honest risk (see §12). Options: cache per-module expansion/IR, a fast non-LLVM debug backend (Zig's approach for fast iteration), and incremental relinking.

---

## 12. The hard problems (be honest about these)

- **Compile speed with whole-program monomorphization + a tree-walking macro interpreter.** Fine now, a real risk at 100k+ LOC. Mitigations: a custom fast backend for debug builds (bypass LLVM), caching, parallel codegen, and possibly a bytecode comptime VM instead of the AST interpreter. This is the single biggest threat to "competitor" status and deserves early prototyping.
- **Incremental compilation** is genuinely hard under whole-program mono. A per-module IR cache + on-demand instantiation is the likely path; true incremental may require giving up some whole-program optimization.
- **Safety without a borrow checker.** The debug-mode-checks plan (§4) covers spatial/temporal *detection*; it does not *prevent* use-after-free at compile time. That's a deliberate, defensible stance (it's C/Zig's stance too) — but it should be stated, not papered over.
- **Macro hygiene's last edge.** The current system is robust through the second-order case; full Racket-style scope-set hygiene is more than Coil likely needs, but the residual (a reference to a macro-generated def that appears *after* the use site) should be documented and, if it ever bites, addressed with a proper expansion-environment model.
- **A `comptime` model that unifies with the macro Lisp** is a real design project, not a feature — it touches the type system, the evaluator, and generics.
- **Stdlib surface area.** "Batteries included" is hundreds of files. It needs sustained effort and an API-design throughline (allocator-aware, error-set-based, slice-first).

---

## 13. A phased critical path

A pragmatic ordering that keeps the tree shippable and dogfoodable at every step:

1. **Phase 1 — Livable.** Source spans + caret diagnostics → DWARF → loops/`defer`/`?` → slices + strings → stdlib core (ArrayList, HashMap, fmt/print, mem) → `test` blocks + `coil test`. *Then dogfood by writing a real program (a JSON parser, a small interpreter, a CLI tool) and fix what hurts.*
2. **Phase 2 — Safe & allocator-first.** Build modes + debug safety checks + a GeneralPurposeAllocator with leak/UAF detection + sanitizer hooks.
3. **Phase 3 — C dominance.** `coil cc`, mixed C/Coil modules with cross-language inlining, `cimport`, export-C-headers, Windows x64 ABI.
4. **Phase 4 — Build & packages.** `build.coil`, a package manager, project conventions; start attacking compile speed.
5. **Phase 5 — Tooling.** `coil fmt`, LSP, grammar, tutorial + reference.
6. **Phase 6 — Lean-in depth.** `adapt`, embedded/freestanding + an MCU demo, comptime/reflection, SIMD stdlib + a `stream` construct, concurrency (atomics/threads/channels), async-as-a-library, more targets.
7. **Phase 7 — Self-host.** Rewrite the compiler in Coil. The ultimate credibility and dogfooding test; the `lang` bootstrap sibling shows the path.

The first dozen items in Phase 1 are what turn Coil from an impressive core into a tool you'd actually use. Everything in §6–§10 is what would make people *choose* it over C or Zig once it's usable.

---

## 14. The smallest next step

If only one thing is built next: **source locations in diagnostics** (§3.1). It is the highest leverage per unit effort, it makes building everything else (and dogfooding) far less painful, and it nearly gives the LSP away. After that, **a real `for`/`while` + `defer` + slices**, then **a stdlib `ArrayList`/`HashMap`/`print`**, then **write a real program in Coil** and let the friction set the rest of the agenda.
