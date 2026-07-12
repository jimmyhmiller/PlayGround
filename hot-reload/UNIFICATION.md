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
| Frame/slot layout | `Frame{registers:Vec<Option<Value>>}` | `RawFrame`/`RawSlot` (tag+i64) | `MtFrame{regs:Vec<Option<Value>>}` | **Phase 3** — abstracted behind `Machine` for now; still 3 layouts |
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
- [ ] **Phase 1** — one `Heap` (objects + `ObjCell` + migration + `value_ok` + alloc).
- [ ] **Phase 2** — one step function; delete `run_actor`.
- [ ] **Phase 3** — unified frame/slot (flat, GC-scannable, holds any `Value`).
- [ ] **Phase 4** — live edit on the concurrent runtime (edit = safepoint op).
- [ ] **Phase 5** — JIT under threads + version-cached recompile.
- [ ] **Phase 6** — delete the trap-gated feature silos.

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
