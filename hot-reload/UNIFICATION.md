# Unifying the three tiers

The goal: **one runtime, one heap, one GC, one step semantics.** "Interpreted vs
JIT" and "single- vs multi-threaded" become configuration, not separate
codebases; every feature works in every configuration; live editing is a
safepoint operation, exactly like GC.

This doc tracks the refactor. See `RUNTIME_DESIGN.md` for the semantics.

## Semantics duplication map (the footguns)

Every row is a place the same idea is implemented more than once and can drift.

| Concern | Interpreter (`runtime.rs`) | JIT (`src/jit.rs`) | Shared (`mt.rs`) | Target |
|---|---|---|---|---|
| Instruction semantics | `execute` match | `define_step` codegen | `run_actor` match | interp = spec; JIT compiles; **delete `run_actor`** |
| Object model | `BTreeMap<ObjectId, Object>` (`Object{body:Arc<Body>}`) | (reads via externs) | `Mutex<BTreeMap<ObjectId, Arc<ObjCell>>>` (`ObjCell{body:Mutex<Arc<Body>>}`) | one `Heap` |
| Migration barrier | `migrate` | via `lt_get_field`→`jit_get_field` | `read_field` inline | one impl on `Heap` |
| Soundness (`value_ok`/`expect_value`) | on `Runtime` | shared via `expect_value` | reimplemented on `Shared` | one impl on `Heap` |
| Allocation | `alloc`/`jit_new` | via `lt_new`→`jit_new` | `alloc` inline | one impl on `Heap` |
| GC roots + sweep | `collect_garbage_with_roots` | driver hands slot roots | `collect` + safepoint parking | one collector |
| Frame/slot layout | `Frame{registers:Vec<Option<Value>>}` | `RawFrame`/`RawSlot` (tag+i64) | `MtFrame{regs:Vec<Option<Value>>}` | one flat GC-scannable slot |
| Effects (`Emit`) | `jit_emit` | via `lt_emit`→`jit_emit` | `output` mutex inline | one impl |

Already single-sourced (keep): `jit_new`/`jit_get_field`/`jit_emit` shared by
interp + JIT; `expect_value`/`value_ok` shared by interp + JIT;
`operand_type_error` reconstructs JIT traps to match the interpreter.

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
