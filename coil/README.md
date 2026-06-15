# coil

A low-level Lisp that exposes MLIR as a first-class thing. Fresh implementation
of the design in [`../mlir-lisp-design/`](../mlir-lisp-design/)
(DESIGN → SPEC → KERNEL → **AOT** → prelude).

**coil is an ahead-of-time compiler, not an interpreter** (see
[`../mlir-lisp-design/AOT.md`](../mlir-lisp-design/AOT.md)). A program is
compiled to native code through MLIR → LLVM; nothing runs inside coil. Codegen
(`emit`) is an ordinary symbol-table-directed walk that *builds* IR — not
program execution. Compile-time macros are **staged** (compiled, then called,
proc-macro style), never tree-walk-interpreted.

This is a clean start — **not** built on `lispier`. The structural decision that
shapes everything: the MLIR layer sits behind a `Backend` trait
(`src/backend.rs`), so the compiler front-half — reader, `Val`, emit — builds and
is tested **without MLIR/LLVM installed**, using a `RecordingBackend` that logs
builder calls. The real `melior`-backed implementation lands behind the `mlir`
cargo feature.

## Status

| Component | State |
|---|---|
| Reader (SPEC §1) | ✅ implemented + tested |
| `Val` (syntax cases) | ✅ |
| Printer / round-trip | ✅ |
| `Backend` trait + `NullBackend` | ✅ (KERNEL §4 codegen surface) |
| `RecordingBackend` (test harness) | ✅ |
| **`emit`: core forms → MLIR builder calls** | ✅ op-calls, `let`, `do`, `region`, `block`, types, attrs, SSA threading |
| Expander: surface sugar (`defn`, `(: …)`, control flow) | ⏳ next |
| `MeliorBackend` (`--features mlir`) | ⏳ |
| Staged macros (proc-macro model) | ⏳ |

## Build & test

```sh
cargo test            # core, no MLIR needed
cargo run -- read examples/add.coil
```

## Layout

```
src/
  value.rs     Val — the shared value universe (syntax cases now; MLIR cases later)
  reader.rs    text → Val   (recursive descent; SPEC §1)
  printer.rs   Val → text   (structural round-trip)
  backend.rs   the MLIR boundary: Backend trait + NullBackend (KERNEL §4)
  recording.rs RecordingBackend — logs builder calls; tests emit without MLIR
  emit.rs      core forms → Backend calls — the lisp→MLIR mapping (AOT codegen)
  lib.rs / main.rs
tests/
  reader.rs   reader + round-trip tests
  emit.rs     mapping tests against RecordingBackend
examples/
  add.coil
```

## Roadmap (next increments)

The AOT spine is built bottom-up (AOT.md §"build order"):

1. **`emit`** — core forms → MLIR builder calls. ✅ done (this increment).
2. **MeliorBackend** (`--features mlir`): implement `Backend` against real MLIR;
   compile a hand-written core-form program to an object file end to end.
3. **Expander**: a fixed, structural expander for the surface sugar (`defn`,
   op-call normalization, `(: v t)`, control flow) — `Val → Val`, no user
   computation yet.
4. **Staged macros**: compile-and-call user `defmacro`s (proc-macro model);
   scope-set hygiene (ELABORATION §5).
5. **Passes + driver**: pass-pipeline values, verify with source-span
   diagnostics, lower to LLVM, emit object/executable.
