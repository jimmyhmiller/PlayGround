# Lispier Examples

All examples in this directory now work with the Rust implementation.

## Example Files

### Basic Operations
- `simple.lisp` - Basic module with arith.addi
- `addi_chain.lisp` - Let bindings with arith operations

### Function Definitions
- `add.lisp` - Simple function returning a sum
- `multiply.lisp` - Function with multiplication
- `subtract.lisp` - Function with subtraction
- `float_add.lisp` - Float addition with f32
- `f64_precision.lisp` - f64 precision arithmetic
- `i64_large.lisp` - i64 large number arithmetic
- `nested_ops.lisp` - Nested operations
- `type_inference.lisp` - Type inference for operations
- `variables.lisp` - Variable bindings with def

### Namespace Aliases
- `arithmetic.lsp` - Functions with namespace aliases (arith :as a, func :as f)

### Control Flow
- `control_flow.lsp` - Conditional branching with cf.cond_br and block successors

### Structured Control Flow (SCF)
- `scf_loops.lsp` - SCF dialect with scf.for loops and arith.select

### Memory Operations
- `memory.lsp` - Memref dialect with alloc, store, load, dealloc

### Simple Entry Point
- `hello.lsp` - Simple main function

## What Works

The current implementation supports:
- Module generation
- Operation generation with regions
- Let expressions with variable bindings
- Type annotations
- Symbol table with scoping (including parent scope access for nested regions)
- Namespace aliases via require-dialect
- Block arguments and block successors for control flow
- arith dialect operations (addi, subi, muli, addf, cmpi, select, etc.)
- func dialect (func.func, func.return, func.call)
- cf dialect (cf.br, cf.cond_br with block successors)
- scf dialect (scf.for with iter_args, scf.yield)
- memref dialect (memref.alloc, memref.store, memref.load, memref.dealloc)
- Constant generation with type inference

## Example Usage

```bash
# Show AST
cargo run -- show-ast examples/simple.lisp

# Show generated MLIR IR
cargo run -- show-ir examples/simple.lisp

# JIT compile and run (for examples with main function)
cargo run -- run examples/add.lisp
```
