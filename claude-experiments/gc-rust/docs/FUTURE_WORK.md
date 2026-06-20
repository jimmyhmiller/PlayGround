# Future Work: Making the gc-rust GC Production-Ready

This is an honest engineering roadmap for taking gc-rust's garbage collector from "works for the test suite and self-hosting workload" to "production-ready." It is grounded in the current implementation, names real files/symbols, and flags what is *verified* versus what still needs an *audit*. Items are prioritized: P0 is soundness (nothing else matters until these are closed), then completeness, performance, the self-host split, observability, and testing.

## Current state (honest snapshot)

The collector is a precise, monomorphizing, generational, moving GC living in the LLVM-free `gcrust-rt` crate. New objects allocate in a young nursery (256 MB) collected by cheap minor GCs; survivors promote to a tenured generation (semi-space, 256 MB per space). Roots are precise: every `Ref` local is a frame root slot, every value-with-ref local is rooted via indirect frame roots, and heap objects are traced by a per-`type_id` `TypeInfo` shape table (with `interior_ptrs` for references embedded in flattened `#[value]` fields). A generational write barrier marks card tables on old→young pointer stores.

Recent hardening (see `docs/reflection.md`): an ANF pass (`src/anf.rs`) ensures no GC value is ever stranded in a register across a safepoint; `gen_alloc` and `gen_make_closure` reload GC operands after the allocation safepoint; value types containing references work everywhere (structs and enums, locals and heap fields); and `--gc-stress` now genuinely collects on *every* allocation and is sound single- and multi-threaded.

What is **not** production-ready: precise-rooting completeness has not been exhaustively audited, one documented soundness hole remains (generic enum int/pointer ambiguity), the generational write barrier misses interior references, allocation is a runtime call on every object (no inline fast path), the self-hosted compiler's output links a *different, non-collecting* runtime, and the concurrent collector path is unverified. Details below.

---

## P0 — Soundness (must close before any production claim)

A moving precise GC is either sound or it is a memory-corruption generator. These are the items that can corrupt the heap.

- **Audit every allocation/safepoint site for the stale-register reload pattern.** The bug class fixed this cycle: code evaluates a GC operand into a register, then allocates (a safepoint that relocates the operand's rooted slot), then stores the *stale* register. Found and fixed in `gen_alloc` and `gen_make_closure` (both reload via `repr_relocates` after the alloc). NOT yet audited: the string primitives (`StrConcat`/`StrSubstring`/`StrFromNum`/`StrFromChar`/`ReadFile`/`type_name_of`) which evaluate `Ref` operands then call an allocating runtime fn; the array ops (`ArrayNew`/`ArraySet`); and any `RuntimeCall` whose callee allocates. The runtime fns likely read their operands before allocating (so they may be safe), but this must be *proven*, site by site, not assumed. This is the single highest-priority task.

- **Fix the generic-enum int/pointer ambiguity (the documented residual unsoundness).** `gcr_runtime_main` in `crates/gcrust-rt/src/runtime.rs` explicitly notes: "a generic enum payload is statically int-or-pointer, so a rare value can be mis-moved" — and the production nursery is sized at 256 MB specifically to avoid collecting during the self-host workload so this never triggers. That is a workaround, not a fix. Root cause: a monomorphic enum payload slot whose static repr is "int-or-pointer" (a value that the GC must decide whether to trace) is ambiguous. Options: ensure monomorphization assigns a concrete `Ref`-vs-scalar repr to every payload slot (so no slot is ever ambiguous); or carry a per-slot trace bit; or use tagged values. Until fixed, the GC is unsound for any program that collects while such a value is live.

- **Generational write barrier for interior references.** `SetField` only emits `emit_write_barrier` for `FieldLoc::Ptr` fields (`src/codegen.rs` ~line 973). Mutating a value-with-ref field (`FieldLoc::ValueAt`) in a *tenured* object writes references into the raw region without marking the card — so a minor GC won't scan that card and the young pointee can be reclaimed while still referenced. Fix: for a `ValueAt` store into a possibly-tenured object, emit a barrier for each interior reference offset (the `interior_ptrs` of that value), or mark the whole object's card. Low effort, real soundness hole.

- **Safepoint coverage audit.** Polls are emitted at loop headers (`emit_safepoint_poll`, `src/codegen.rs` ~line 2424), which covers the common case. Verify the complete set: every loop back-edge (not just the header), function entry/calls for STW liveness, and any unbounded computation. A long-running, allocation-free loop on one thread that never polls will block a stop-the-world collection on another thread indefinitely — a liveness bug that becomes a hang, not corruption, but is still a correctness defect.

- **FFI pinning.** `as_c_bytes` copies a `String`/array's bytes to the (non-moving) native stack for the duration of an extern call, which is correct. Verify there is no path where a *bare* GC reference is handed to C and then a collection moves it (e.g. a struct pointer passed to a callback that re-enters managed code and allocates). If such paths exist, they need explicit pinning or a copy.

- **Continuous stress in CI.** `--gc-stress` is now real and the whole suite passes under it. Wire it so the entire test corpus *and* the examples run under `--gc-stress` on every CI run, not just on demand. Cheap insurance against regressions in any of the above.

---

## P1 — Collector completeness & robustness

These don't corrupt memory but block real workloads.

- **Concurrent/incremental major GC: verify or finish.** The heap carries the machinery for concurrent collection (`GcPhase::Copying`, a global `SATBQueue`, snapshot-at-the-beginning barriers). It is unclear whether this path is sound, complete, and exercised, or experimental scaffolding (the default major GC may be stop-the-world). Determine the status; either finish and verify it (with stress + the soundness audit above applied to the concurrent barriers) or remove it to avoid a false sense of capability. Concurrent or at least incremental collection is required for any latency-sensitive workload.

- **Stop-the-world pause bounds.** Today a major collection is a full STW copy of the tenured generation. Measure pause times on realistic heaps and set a target. If STW pauses are unacceptable, this drives the concurrent/incremental work above.

- **Dynamic heap sizing and OOM policy.** Sizes are hard-coded (256 MB nursery, 256 MB × 2 tenured). Production needs: configurable initial/max heap, grow-on-pressure and shrink-when-idle heuristics, and a graceful out-of-memory path. Currently OOM `std::process::abort()`s (`alloc_with_published_frame`). At minimum, surface a recoverable error or a controlled crash with diagnostics.

- **Tenured-generation memory overhead.** The tenured generation is a copying semi-space, so it reserves 2× the live old-gen size. For large heaps that doubling is expensive. Consider a mark-compact or mark-sweep tenured collector to halve the footprint, at the cost of collector complexity.

- **Large-object handling and fragmentation.** Objects too big for the nursery go straight to tenured (`alloc_with_published_frame`). Verify behavior for very large objects and for varlen objects (arrays/strings) near space limits; consider a dedicated large-object space.

- **Weak references and finalization.** Not present. If the language needs weak maps, caches, or resource finalizers (e.g. closing FDs held by GC'd objects), these need first-class GC support (weak roots, finalizer queues, resurrection semantics).

---

## P2 — Performance

The collector can be correct and still too slow to ship.

- **Inline bump allocation in generated code.** Every allocation is currently a call to `ai_gc_alloc_fixed`/`ai_gc_alloc_varlen` (the `alloc_window` fast path is kept *closed* — limit 0 — for the generational and stress heaps, so the JIT/AOT code always defers to the runtime). The single biggest allocation-throughput win is emitting an inline bump-pointer fast path (compare-and-bump against the nursery limit, call the runtime only on overflow). This is standard and large-impact.

- **Inline write-barrier fast path.** The card-marking barrier goes through `ai_gc_write_barrier`. Inline the common-case check (is the store old→young?) so most stores cost a few instructions, not a call.

- **Thread-local allocation buffers (TLABs).** Multithreaded allocation currently contends on a shared atomic bump pointer (`AtomicBumpAllocator`). Per-thread TLABs remove that contention and are a prerequisite for allocation-heavy multithreaded scaling.

- **Parallel collection.** The collector appears single-threaded. Parallelizing the copy/scan across worker threads cuts pause times on multi-core machines.

- **Root-set minimization.** The ANF pass conservatively let-binds *every* non-atomic GC subexpression, so functions get more root slots than strictly necessary (each scanned at every GC). A liveness-aware pass that only roots values actually live across a safepoint would shrink frames and speed root scanning. Correctness-first today; optimize once stable.

- **Generational tuning.** Nursery size, promotion policy (promote-on-first-survival vs aging), and minor/major ratio should be measured and tuned against real allocation profiles rather than fixed.

---

## P3 — The two-runtime split (self-host unification)

This is a structural issue worth its own section.

The JIT and `gcr build` AOT path use the real precise moving generational GC. But the **self-hosted compiler's output** links `runtime/runtime.c` (`gcr_rt.c`), a non-moving bump allocator that **never frees** — self-hosted binaries effectively leak for their whole run. They get away with it because the compiler is a short-lived batch process. This is not acceptable for general production binaries.

Unifying on the real GC requires the bootstrap codegen (`compiler/codegen.lang`) to emit what the real GC needs: precise frame/root information and safepoint polls in the code it generates — exactly the rooting discipline the Rust-hosted `src/codegen.rs` already implements (frames, `interior_ptrs`, ANF, reload-after-safepoint). This is substantial: the self-hosted codegen would need to gain the entire precise-rooting machinery. Until then, "production binaries" really means "AOT-from-the-Rust-compiler binaries," and the self-hosted toolchain is a forcing function, not a production target.

---

## P4 — Observability & tooling

Largely in place; a few gaps.

- **Have:** heap snapshot dumps (`GCR_HEAP_DUMP=1` text, `=json` structured with per-type histogram, root set, and dominator-based retained sizes — `crates/gcrust-rt/src/gc/dump.rs`); GC statistics (`GCR_GC_STATS=1`); a statemap tracer for GC visualization (`StatemapTracer`); and full runtime reflection metadata.

- **Need:** allocation-site profiling (which call sites allocate the most), pause-time histograms and a GC log suitable for offline analysis, automatic leak/growth detection across collections, and a live (not just program-end) heap explorer driven from a safepoint. The JSON snapshot is already a clean substrate for an interactive viewer (e.g. a jim widget).

---

## P5 — Testing & verification

- **GC fuzzing.** Generate random programs (varied object graphs, sharing, cycles, threads, value-with-ref nesting) and run them under `--gc-stress`, diffing results against a no-GC reference. This is how the remaining stale-register and ambiguity bugs will be found.

- **A targeted test for the int/pointer ambiguity** once its fix is designed — currently there is no test that forces a collection while an ambiguous generic-enum payload value is live.

- **Multithreaded stress matrix.** `threads_stress` now collects for real; expand it to cover more spawn/join/atom/channel patterns and higher thread counts, under both generational and semi-space heaps.

- **Property/differential testing** of the collector primitives (the relocation + rooting invariants) beyond the current unit tests in `crates/gcrust-rt/src/gc/tests.rs`.

---

## Suggested order

1. P0 stale-register audit (finish what this cycle started) and the interior-ref write barrier (small, real holes).
2. P0 generic enum int/pointer ambiguity (the one known soundness hole that needs design).
3. P0 safepoint coverage audit + continuous `--gc-stress` in CI.
4. P2 inline allocation + inline write barrier (the performance unlock).
5. P1 decide the concurrent-GC story + dynamic heap sizing/OOM.
6. P3 self-host unification (large, structural).
7. P5 GC fuzzing throughout (it accelerates everything above).
