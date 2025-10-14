# Working CLI Demo! 🎉

## It Actually Works!

```bash
$ cargo run --bin mlir-lisp examples/hello.lisp
```

## The File (examples/hello.lisp)

```lisp
;; Import the core dialect
(import lisp-core)

;; Define main function that will be executed
(defn main [] i32
  (+ (* 10 20) 30))
```

## What Happens

```
Reading file: examples/hello.lisp
Generated MLIR:
module {
  func.func @main() -> i32 {
    %c10_i32 = arith.constant 10 : i32
    %c20_i32 = arith.constant 20 : i32
    %0 = arith.muli %c10_i32, %c20_i32 : i32
    %c30_i32 = arith.constant 30 : i32
    %1 = arith.addi %0, %c30_i32 : i32
    return %1 : i32
  }
}

Lowering to LLVM...
JIT compiling and executing...
✨ Execution result: 230
✅ Program executed successfully!
```

## Try It Now!

```bash
# Run the example
cargo run --bin mlir-lisp examples/hello.lisp

# Create your own file
cat > my-program.lisp <<'EOF'
(import lisp-core)

(defn main [] i32
  (* 42 2))
EOF

# Run it!
cargo run --bin mlir-lisp my-program.lisp
# Result: 84
```

## It's Real!

✅ Loads Lisp files
✅ Imports work
✅ Compiles to MLIR
✅ Lowers to LLVM
✅ JIT executes
✅ Returns the result

**The meta-circular compiler is WORKING!** 🚀
