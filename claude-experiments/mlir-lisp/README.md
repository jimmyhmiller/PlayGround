# MLIR-Lisp

A minimalist Lisp that compiles directly to MLIR (Multi-Level Intermediate Representation).

## Status

**Phase 3 Complete**: JIT Execution Working! 🎉🚀
- ✅ S-expression parser using nom
- ✅ AST types (Symbol, Keyword, String, Integer, Float, List, Vector, Map)
- ✅ All parser tests passing
- ✅ Design document with syntax examples
- ✅ MLIR context management (melior integration)
- ✅ Operation emitter that generates valid MLIR
- ✅ Operand support (SSA values work!)
- ✅ JIT compilation via MLIR ExecutionEngine
- ✅ **End-to-end: Lisp → MLIR → LLVM → Native Code → Execution**

## Quick Start

```bash
# Run tests
cargo test

# Build the compiler
cargo build --release

# Run a .lisp file
cargo run --release examples/simple.lisp
cargo run --release examples/add.lisp

# Or use the binary directly
./target/release/mlir-lisp examples/simple.lisp

# Run without arguments to see the built-in example
cargo run --release
```

**Example Output:**
```
Parsed 4 expressions

Generated MLIR:
module {
  func.func @main() -> i32 {
    %c10_i32 = arith.constant 10 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = arith.addi %c10_i32, %c32_i32 : i32
    return %0 : i32
  }
}

Lowering to LLVM...
After LLVM lowering:
module {
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(10 : i32) : i32
    %1 = llvm.mlir.constant(32 : i32) : i32
    %2 = llvm.mlir.constant(42 : i32) : i32
    llvm.return %2 : i32
  }
}

JIT compiling and executing...
✨ Execution result: 42
Expected: 42 (10 + 32)
✅ Success! Lisp → MLIR → LLVM → JIT → Executed!
```

Note: LLVM optimized the addition at compile time!

## What We've Built So Far

### Parser
The parser (`src/parser.rs`) can parse:
- **Symbols**: `foo`, `arith.addi`, `%result`, `+`, `-`
- **Keywords**: `:value`, `:sym-name`, `:type`
- **Integers**: `42`, `-10`
- **Floats**: `3.14`, `-2.5`
- **Strings**: `"hello world"`
- **Lists**: `(+ 1 2)`
- **Vectors**: `[1 2 3]`
- **Maps**: `{:value 10 :type i32}`
- **Comments**: `; This is a comment`

### Example Code (Currently Working!)

```lisp
;; Add two numbers and return the result
;; This actually compiles and runs!

(op arith.constant
    :attrs {:value 10}
    :results [i32]
    :as %ten)

(op arith.constant
    :attrs {:value 32}
    :results [i32]
    :as %thirty_two)

(op arith.addi
    :operands [%ten %thirty_two]
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
```

This gets parsed, compiled to MLIR, lowered to LLVM, JIT compiled, and executed - returning 42!

## Next Steps

See `DESIGN.md` for the full vision and syntax examples.

**What Works Now**:
1. ✅ MLIR integration with LLVM 20
2. ✅ Operation emitter (attrs, results, operands all working)
3. ✅ SSA value tracking and usage
4. ✅ JIT compilation and execution
5. ✅ Basic integer types (i8, i16, i32, i64)

**Next Steps**:
1. More arithmetic operations (sub, mul, div)
2. Core special forms (let, do, block, module)
3. Macro system
4. Type/attribute helpers
5. Standard library macros
6. File interpreter / REPL

## Dependencies

Requires **LLVM 20** and the melior Rust bindings.

**Ubuntu 24.04:**
```bash
sudo apt install llvm-20-dev libmlir-20-dev mlir-20-tools
sudo update-alternatives --install /usr/bin/llvm-config llvm-config /usr/bin/llvm-config-20 100
```

## Philosophy

- **MLIR as the primitive**: Everything compiles to MLIR operations
- **Minimal core**: Only essential features built-in
- **Macros for abstraction**: Build your own language on top
- **Dual syntax**: Support both builder-style and text-format-style
- **No magic**: All operations are transparent MLIR code
