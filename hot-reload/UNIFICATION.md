# Unifying the three tiers

The goal: **one runtime, one heap, one GC, one step semantics.** "Interpreted vs
JIT" and "single- vs multi-threaded" become configuration, not separate
codebases; every feature works in every configuration; live editing is a
safepoint operation, exactly like GC.

**STATUS: DONE — including the endgame (Phase 8).** There is now exactly ONE
executor in the codebase: `Engine` (`livetype-core/src/engine.rs`), a tiered
actor loop over the one thread-safe runtime `Shared`. The interpreter is its
cold tier; compiled code (behind the `TierSource` trait) is its hot tier; a
worker thread is the same loop on another thread. `Runtime`, `InterpMachine`,
`MtMachine`, `Shared::run_actor`, `JitActor`/`drive`/`run_interleaved`,
`run_jit_threads`, and `Tiered` are all deleted. See "The final architecture"
at the bottom.

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
  firing while JIT threads churn). **Version-cached recompile — DONE:** `JitCode`
  caches compiled code by world epoch and recompiles on demand when a live edit
  advances it (leaking one engine per edit generation — a documented
  research-prototype tradeoff around the inkwell self-referential-lifetime
  issue), so a program can be **live-edited while it runs on the JIT threads**
  (`live_edit_a_program_running_on_a_jit_thread`: a JIT worker's `tick()`
  hot-swapped 1→2 mid-loop, then a breaking edit stops it).
- [x] **Phase 7 — auto-tiering (interp → JIT promotion).** The two engines are
  no longer a manual choice. `Tiered` (src/jit.rs) starts every function
  interpreted, counts calls per `(func, version)`, and promotes a hot one to the
  JIT; a single actor's stack freely mixes interpreted (`Value` registers) and
  JIT (`RawSlot`) frames, marshalling at the call/return boundaries. Both engines
  are reused verbatim — the interpreter's `step_instruction` (via a
  `TieredMachine`) and the JIT's compiled `step` — with `push_callee` /
  `deliver_return` as the shared tier-and-marshal seam. Proven by
  `tests/tiered.rs`: a function called in a loop is promoted mid-run, result
  identical to the pure interpreter; FFI/globals survive promotion.
  **Hot-reload composes with tiering** (`tests/tiered_hotreload.rs`, via
  `Tiered::{install_function,install_schema,install_migration,resume}`): editing
  a *promoted (JIT)* function is picked up on the next run; a breaking edit
  **traps a JIT caller** rather than running stale native code (soundness holds
  through the JIT because effects go through the runtime externs), and
  trap → repair → `resume()` completes with a JIT frame on the stack; a live
  schema migration is transparent to a JIT-compiled reader (its `GetField` uses
  the same migration barrier as the interpreter). *Remaining:* on-stack
  replacement (promoting a hot *loop* in a running frame, not just at the next
  call) and applying tiering on the concurrent workers.

- [x] **Phase 6 — the surface language's features now run on every tier.** FFI +
  globals run on the concurrent tier
  (`tests/live_concurrent.rs::ffi_and_globals_run_on_the_concurrent_tier`) *and*
  on the JIT: `RawSlot` now tag-encodes a `Foreign` handle (kind in the tag's
  high bits, pointer in the payload — no wider frame layout), and the JIT lowers
  `CallForeign`/`LoadGlobal` to `lt_call_foreign`/`lt_load_global` externs routed
  through the shared `JitHost`, so a foreign program runs identically on the
  interpreter and the JIT, single-threaded or across threads
  (`tests/jit_ffi.rs`, incl. a `Foreign` handle round-tripping through JIT
  slots). The one remaining trap is `Send`/`Recv` on the interpreter and JIT —
  message passing has **no surface syntax** (reachable only by hand-built IR),
  so it stays concurrent-tier-only, documented, inert from source.

- [x] **Phase 8 — ONE executor (the endgame).** All the remaining top-level
  drivers collapsed into a single `Engine` in `livetype-core`:
  - `Shared` is THE runtime: world (RwLock), heap, globals, FFI registry,
    mailboxes, preemptive STW GC. `Shared::new()` starts empty; there is no
    "setup `Runtime` then freeze" step — installs are always live.
  - `Engine` is THE executor: one actor loop over a mixed stack of interpreted
    `Frame`s and native `JitFrame`s (`RawSlot` registers), promotion by a
    per-`(func, version)` activation counter at the ONE frame-push site
    (`push_callee` — spawn, interpreted calls, and native calls all land
    there), marshaling at call/return boundaries, safepoints every turn.
  - The LLVM/Miri boundary moved to the `TierSource` trait: "give me a native
    step address for this function version, if you have one." The raw frame
    representation and the runtime externs (`lt_new`, `lt_get_field`, `lt_emit`,
    `lt_call_foreign`, `lt_load_global` over a `NativeHost(&Shared)`) live in
    core (`native.rs`) — they are LLVM-free. The `livetype` crate is now ONLY
    the compiler: codegen + `JitCode` (the epoch-cached compiled-address store),
    which implements `TierSource`. `NoJit` is the oracle/Miri configuration.
  - Trap-and-repair is engine-level: a paused actor keeps its mixed-tier stack;
    `resume`/`resume_with`/`pause_expected`/`thaw` work whichever frame kind is
    on top. There is no auto-resume on install anymore — hosts resume
    explicitly, one code path for every tier.
  - Deleted: `Runtime` (the struct and its executor), `InterpMachine`,
    `MtMachine`, `Shared::run_actor`/`run_threads`, `JitActor`, `drive`,
    `run_interleaved`, `resume_with` (JIT flavor), `handle_call*`,
    `run_jit_threads`, `Tiered`, `TieredMachine`, `JitHost::Single` — every
    parallel implementation of "run an actor" is gone. `exec::step_instruction`
    now has exactly one `Machine` impl (`EngineMachine`).
  - The **tier gap table above is now empty**: FFI, globals, live edit, STW GC,
    and message passing all work on the one engine (message passing simply
    means the frame stays on the cold tier — the codegen skips nothing today
    because Send/Recv have no surface syntax, but a function containing them
    would just never promote).
  - The differential fuzzer compares *configurations* of the one engine
    (never-promote oracle vs always-promote vs auto-tiering, plus a
    worker-thread run) — 600 seeds each, plus the whole ported suite (61
    tests). `cargo +nightly miri test -p livetype-core` is clean.

## The final architecture

```
livetype-core (LLVM-free, Miri-gated)
  model/verify/heap        IR, CFG verifier, the one Heap (Mutex<Arc<Body>> cells)
  runtime.rs               World installs (the one install path), ResumePlan
  mt.rs        Shared      THE runtime: world lock, heap, globals, FFI, mailboxes, STW GC
  native.rs                RawSlot/RawFrame/externs — the C-ABI contract, no LLVM
  exec.rs                  step_instruction over Machine (ONE impl: EngineMachine)
  engine.rs    Engine      THE executor: tiered actor loop + TierSource seam + NoJit
  frontend/    Session     source → IR, evals onto an Engine

livetype (links LLVM 21)
  jit.rs       compile     codegen for step fns
               JitCode     epoch-cached addresses; impl TierSource
               jit_engine  Engine::new(JitCode, threshold) — the full system
```

Configurations of the one engine: `Engine::interp()` (oracle, Miri),
`jit_engine(0)` (always native), `jit_engine(n)` (auto-tiering),
`engine.run_threads(..)` (same loop, more threads).

## Performance (2026-07, `livetype-bench`, release, per work unit)

| bench | interp | jit(0) | native Rust |
|---|---|---|---|
| loop_sum (arith iter) | 38 ns | **3 ns** | 0.3 ns |
| call_add (call) | 144 ns | **65 ns** | 1.3 ns |
| fib_25 (rec call) | 135 ns | **69 ns** | 0.9 ns |
| alloc_read (alloc+field) | 343 ns | **227 ns** | — |
| yield_loop (safe point) | 47 ns | **7 ns** | 0.3 ns |

How (`engine.rs` module docs have the details):
- **Interp batching**: up to `INTERP_BATCH` (64) instructions under ONE world
  read guard with the current function resolved once per frame change (no
  per-instruction lock/lookup/clone). A batch ends at a `Yield`, block, stop,
  or tier switch, so edits + STW GC stay at most one bounded batch away;
  host-driven `Engine::step` keeps exact per-instruction granularity.
- **Lock-free native turns**: a `JitFrame` caches its compiled entry address
  and declared result type at push. Sound because a function *version* is
  immutable and compiled engines are never torn down — an address never goes
  stale. `Shared::code_epoch` (an atomic mirror of `world.epoch`) invalidates
  the per-actor `TierSource` map snapshot only when an edit actually lands.
- **Slot-level boundaries**: native→native calls/returns copy `RawSlot`s
  directly with the soundness check done at the slot level (`slot_ok` mirrors
  `value_ok`; a reference still asks the heap for its nominal type) — no
  `Value` round trip, no intermediate register vec.
- **Tier decision fast paths**: thresholds 0/`u64::MAX` skip the counter lock;
  a per-actor `hot_local` set (hotness is monotonic per version) skips it once
  promoted.
- **Heap**: object table is a `HashMap`; `new_object` no longer clones the
  schema (it was cloning every field name String per allocation!);
  `get_field`'s current-schema fast path reads under the body lock with one
  table lookup, entering the migration barrier only when actually stale.
- **Codegen**: extern-call marshaling allocas are hoisted to the entry block —
  per-site allocas in a loop grew the native stack every iteration (a
  200k-iteration allocating loop overflowed the stack before this).

## What remains (optimizations, not architecture)

- Reclaim leaked JIT engines once no frame pins an old version.
- A fairer world lock so a tight-loop worker can't writer-starve a live edit
  (batching already bounds reader hold times).
- Allocation is ~227 ns: every object is two `Arc`s + a per-body `BTreeMap`
  under the table lock. Going much lower means an arena/bump heap under the
  same non-moving-handle semantics — a real project.
- OSR: promote a hot *loop* in a running frame (today promotion is at frame
  push). This is also why a mixed interp-caller→native-callee boundary
  (`tiered` call_add: 177 ns) can't collapse further — the caller never
  promotes mid-run.

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
