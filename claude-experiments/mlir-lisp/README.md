# MLIR-Lisp: Self-Contained Meta-Circular Compiler ✨

A **working** meta-circular MLIR compiler where programs are defined in Lisp and JIT executed.

## Quick Start

```bash
cargo run --bin mlir-lisp examples/hello.lisp
# ✨ Execution result: 230
```

## Write Your First Program

```lisp
;; test.lisp
(import lisp-core)

(defn main [] i32
  (+ (* 10 20) 30))
```

```bash
cargo run --bin mlir-lisp test.lisp
# ✨ Execution result: 230
```

## What Works

✅ Lisp → MLIR → LLVM → JIT → Execute
✅ Import system
✅ Real compilation
✅ Actual execution

Try it now!
