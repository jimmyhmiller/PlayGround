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
| **`emit`: core forms → MLIR builder calls** | ✅ op-calls, `let`, `do`, `region`, `block`, types, attrs, `(: v t)`, SSA threading |
| **`expand`: bootstrap surface sugar → core forms** | ✅ `defn`, `if`→`scf.if`, `when`, `cond`, `->`; implicit returns |
| `MeliorBackend` (`--features mlir`) | ⏳ (needs an MLIR toolchain) |
| Staged user macros (proc-macro model) | ⏳ |

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
  expand.rs    surface sugar → core forms (bootstrap expander; AOT.md phase 0)
  emit.rs      core forms → Backend calls — the lisp→MLIR mapping (AOT codegen)
  lib.rs / main.rs
tests/
  reader.rs   reader + round-trip tests
  expand.rs   sugar→core tests + read→expand→emit integration
  emit.rs     mapping tests against RecordingBackend
examples/
  add.coil
```

```sh
coil expand examples/add.coil   # show surface → core-form lowering
```

## Roadmap (next increments)

The AOT spine is built bottom-up (AOT.md §"build order"):

1. **`emit`** — core forms → MLIR builder calls. ✅ done.
2. **`expand`** — bootstrap surface sugar → core forms. ✅ done.
3. **MeliorBackend** (`--features mlir`): implement `Backend` against real MLIR;
   compile a hand-written core-form program to an object file end to end.
   *(Blocked here: this environment has LLVM but no MLIR dev libs/headers.)*
4. **Staged macros**: compile-and-call user `defmacro`s (proc-macro model);
   scope-set hygiene (ELABORATION §5).
5. **Passes + driver**: pass-pipeline values, verify with source-span
   diagnostics, lower to LLVM, emit object/executable.

Stages 1–2 are a working, MLIR-free front end: `read → expand → emit` turns
surface coil into a recorded sequence of MLIR builder calls, all under test.
