# coil

A low-level Lisp that exposes MLIR as a first-class thing. Fresh implementation
of the design in [`../mlir-lisp-design/`](../mlir-lisp-design/)
(DESIGN → SPEC → KERNEL → ELABORATION → prelude).

This is a clean start — **not** built on `lispier`. The one structural decision
that shapes everything: the MLIR layer sits behind a `Backend` trait
(`src/backend.rs`), so the core — reader, `Val`, evaluator, elaboration — builds
and is tested **without MLIR/LLVM installed**. The real `melior`-backed
implementation will land behind the `mlir` cargo feature.

## Status

| Component | State |
|---|---|
| Reader (SPEC §1) | ✅ implemented + tested |
| `Val` (syntax cases) | ✅ |
| Printer / round-trip | ✅ |
| `Backend` trait + `NullBackend` | ✅ skeleton (KERNEL §4 surface) |
| Single-pass elaborator (ELABORATION) | ⏳ next |
| Macros / hygiene | ⏳ |
| `MeliorBackend` (`--features mlir`) | ⏳ |

## Build & test

```sh
cargo test            # core, no MLIR needed
cargo run -- read examples/add.coil
```

## Layout

```
src/
  value.rs    Val — the shared value universe (syntax cases now; MLIR cases later)
  reader.rs   text → Val   (recursive descent; SPEC §1)
  printer.rs  Val → text   (structural round-trip)
  backend.rs  the MLIR boundary: Backend trait + NullBackend stub (KERNEL §4)
  lib.rs / main.rs
tests/
  reader.rs   reader + round-trip tests
examples/
  add.coil
```

## Roadmap (next increments)

1. **Evaluator core**: `env`, closures, and the special forms (`quote`, `if`,
   `let`, `fn`, `do`, `def`, `quasiquote`) — KERNEL §3 — over `NullBackend`.
2. **Macros**: `defmacro`, fixpoint expansion, scope-set hygiene (ELABORATION §5).
3. **Single-pass elaboration**: `elab` with a live builder; op-call sugar;
   `build` / `with-scratch` (ELABORATION §1–3) — still against a backend trait,
   so testable with a fake builder.
4. **MeliorBackend**: wire `Backend` to real MLIR; run `examples/` end to end.
