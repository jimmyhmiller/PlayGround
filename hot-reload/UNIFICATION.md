# Unifying the three tiers

The goal: **one runtime, one heap, one GC, one step semantics.** "Interpreted vs
JIT" and "single- vs multi-threaded" become configuration, not separate
codebases; every feature works in every configuration; live editing is a
safepoint operation, exactly like GC.

This doc tracks the refactor. See `RUNTIME_DESIGN.md` for the semantics.

## Semantics duplication map (the footguns)

Every row is a place the same idea is implemented more than once and can drift.

| Concern | Interpreter (`runtime.rs`) | JIT (`src/jit.rs`) | Shared (`mt.rs`) | Status |
|---|---|---|---|---|
| Instruction semantics | ~~`execute` match~~ → `InterpMachine` | `define_step` codegen | ~~`run_actor` match~~ → `MtMachine` | **DONE** — one `exec::step_instruction` over the `Machine` trait; JIT still compiles (checked by fuzzer) |
| Object model | ~~`BTreeMap<Object>`~~ | (reads via externs) | ~~`Mutex<BTreeMap<ObjCell>>`~~ | **DONE** — one `heap::Heap` |
| Migration barrier | ~~`migrate`~~ | via `lt_get_field`→`jit_get_field` | ~~`read_field`~~ | **DONE** — `Heap::migrate` (concurrency-safe) |
| Soundness (`value_ok`) | ~~on `Runtime`~~ | via `expect_value` | ~~reimplemented~~ | **DONE** — `Heap::value_ok` |
| Allocation | ~~`alloc`/`jit_new`~~ | via `lt_new` | ~~`alloc`~~ | **DONE** — `Heap::new_object`/`alloc` |
| GC roots + sweep | `collect_garbage_with_roots` | driver hands slot roots | `collect` + safepoint parking | sweep+trace unified on `Heap` (`retain`/`child_refs`); root-*gathering* still per-tier |
| Frame/slot layout | one shared `Frame` | `RawFrame`/`RawSlot` (tag+i64) | one shared `Frame` | **DONE for the managed tiers** — interp + Shared share `Frame` (holds any `Value`, incl `Foreign`) + `frame_roots`; the JIT's C-ABI `RawSlot` widening is coupled to JIT-FFI and lands in Phase 5 (widen there, where it's exercised) |
| Effects (`Emit`) | `jit_emit` | via `lt_emit` | `Shared::emit` | one path each; step calls `m.emit` |

The `Machine` trait (`exec.rs`) is the seam: `InterpMachine` (over `&mut Runtime`
+ an actor) and `MtMachine` (over a thread-local frame stack + `&Shared`) are
the only two implementations, and `step_instruction` is written once against it.
Tier feature gaps are now expressed as the `Machine` answering `Unsupported`
(FFI/globals on the concurrent tier; message passing on the interpreter), which
`step_instruction` turns into a clear trap — the exact seam Phase 6 removes.

## Tier feature gaps (behavior depends on executor — a footgun)

| Feature | Interp | JIT | Shared |
|---|:---:|:---:|:---:|
| FFI (`CallForeign`) + globals (`LoadGlobal`) | ✅ | traps | traps |
| `Send`/`Recv` | traps | traps | ✅ |
| Multicore + STW GC | — | — | ✅ |
| Live `eval` while running | ✅ | ✅ (recompile) | frozen |

## The regression net (Phase 0 — DONE)

`tests/differential_fuzz.rs`:
- `jit_matches_interpreter_on_random_programs` — interp ↔ JIT on the full
  edit-at-yield scenario (600 seeds).
- `shared_matches_interpreter_on_random_programs` — interp ↔ Shared on the
  steady-state (no-edit) run to completion (600 seeds).

Shared cannot take mid-run edits yet, so the two comparisons split the scenario
between them; after Phase 4 the Shared comparison extends to the edit scenario.

## Phases

- [x] **Phase 0** — three-way differential net + this duplication map.
- [x] **Phase 1** — one `Heap` (objects + `ObjCell` + migration + `value_ok` + alloc).
- [x] **Phase 2** — one step function (`exec::step_instruction` over `Machine`); `run_actor` match and `execute` both deleted.
- [x] **Phase 3** — one managed `Frame` (interp + Shared) + one `frame_roots`. (JIT `RawSlot` widening deferred to Phase 5, where JIT-FFI exercises it.)
- [x] **Phase 4** — live edit on the concurrent runtime. Install logic is now one `impl World` path used by both tiers; `Shared.world` is an `RwLock` a worker reads per step and an editor writes between steps; `Shared::install_*` are live. Proven by `tests/live_concurrent.rs` (a worker thread's function hot-swapped between two of its calls, deterministic via message-passing handshake). *(Note: edits use the world `RwLock`, not the GC safepoint — simpler and sufficient; a tight-loop worker could in theory writer-starve on a platform with reader-preferring `RwLock`, a later fairness tweak.)*
- [x] **Phase 5 (JIT under threads) — DONE.** Worker threads execute the same
  compiled `step` functions over the thread-safe `Shared` runtime, respecting
  the LLVM/Miri boundary: `livetype-core` still never links LLVM; the `livetype`
  crate owns the threads + compiled code and calls *into* core's `Shared`. One
  extern set now serves both executors via a `JitHost` enum (`Single(&mut
  Runtime)` / `Concurrent(&Shared)`), so the JIT and interpreter still can't
  drift on alloc/migration/effects. JIT workers hit GC safepoints (publishing
  native-frame roots), so stop-the-world collection pauses them too. Proven by
  `tests/jit_threads.rs` (loops, calls, concurrent shared-heap alloc, and STW GC
  firing while JIT threads churn). *Remaining Phase-5 optimization:* version-
  cached recompile (still recompiles per run; a mid-run edit that bumps a version
  the driver hasn't compiled surfaces a clear error — no silent stall), gated by
  the inkwell self-referential-lifetime issue.
- [~] **Phase 6** — *managed tiers done.* FFI + globals now run on the
  concurrent tier (`tests/live_concurrent.rs::ffi_and_globals_run_on_the_concurrent_tier`),
  so the interpreter and concurrent tiers are feature-reaching for FFI/globals.
  The remaining silos are JIT-only or unreachable-from-source (see below).

## What actually remains, and the real constraints

Phases 1–5 delivered the unification: one heap, one step semantics, one managed
frame, one install path, live-edit across threads, and the JIT executing on the
concurrent tier's threads (the LLVM/Miri boundary was honored by re-pointing the
externs at `Shared` via `JitHost` and keeping the threaded driver in the
`livetype` crate — core still links no LLVM). Phase 6 removed the
practically-important silo (FFI/globals on the concurrent tier). What's left is
bounded and lower-urgency:

1. **Version-cached recompile.** The JIT still recompiles the world per run;
   caching a compiled module across runs fights `Compiled<'ctx>`'s borrow of its
   `Context`. Needs an ownership shim (`ouroboros`) or a leaked per-generation
   `Context`. Consequence today: live-edit *combined with* JIT-threads needs a
   recompile on the version bump; the driver returns a clear error if it meets an
   uncompiled version rather than stalling.
2. **JIT FFI/globals/message passing** (reachable only by hand-built IR, not from
   the surface language): needs `RawSlot` widened to a tagged two-word slot (to
   carry `Foreign{kind,ptr}`) plus `lt_call_foreign` / `lt_load_global` externs.
   The `step_instruction` seam already traps these clearly; wiring is mechanical
   once the slot is widened.
3. **Interpreter message passing** (`Send`/`Recv`): needs a parked-actor
   scheduler state + deadlock detection in `run()`, and there's no `send`/`recv`
   surface syntax to reach it — inert from source today.

## Hard problems to solve along the way

- **Safepoint-aware native calls**: a long FFI call can't hit a safepoint and
  would stall STW GC; the worker must declare itself parked-for-GC before the
  call (JNI-style) and re-check on return.
- **FFI retaining managed objects**: a global-ref pin table (JNI `GlobalRef`
  analog) for objects C keeps past a call; non-moving GC keeps the pointer valid,
  the pin supplies liveness.
- **Single-threaded perf under a locked heap**: start correct (mutex), optimize
  to atomic `Arc` swap / thread-local fast paths later.
- **Recompile cache invalidation** on Broken→repaired version transitions.
- **Edit/GC ordering** at a shared safepoint: drain edits → collect → release.
