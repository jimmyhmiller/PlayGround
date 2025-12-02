# Quick Clojure PoC

A minimal Clojure implementation in Rust that compiles directly to ARM64 machine code.

## Project Goals

Build a Clojure clone supporting the [Clojure special forms](https://clojure.org/reference/special_forms) with ARM64 compilation. The project follows a multi-stage bootstrap strategy to eventually use ClojureScript's native persistent data structure implementations.

## Bootstrap Strategy

**The Problem**: Need persistent data structures to implement a reader, but need a reader to define persistent data structures.

**The Solution**: Multi-stage bootstrap using temporary Rust libraries, then compiling native ClojureScript data structures.

## Current Status: Stage 1 Complete - JIT COMPILATION WORKING! ✓

### Stage 0: Foundation (COMPLETED)

**Goal**: Read S-expressions using Rust-backed persistent structures

**Implemented**:
- ✅ Clojure/EDN reader using `clojure-reader` crate
- ✅ Value representation with im-rs persistent data structures
- ✅ Simple REPL (read-print only, no evaluation yet)

### Stage 1: JIT Compiler to ARM64 (COMPLETED)

**Goal**: Compile Clojure to native ARM64 machine code

**Implemented**:
- ✅ Clojure AST representation
- ✅ Analyzer (Value → AST conversion with special form recognition)
- ✅ **Direct ARM64 JIT compiler** (no intermediate VM!)
- ✅ Arithmetic compilation: `+`, `-`, `*`
- ✅ Literal compilation: integers, booleans
- ✅ Nested expression compilation
- ✅ Working REPL with **native code execution**

**Compiles and Executes as ARM64 Machine Code**:
```clojure
=> 42
42
=> (+ 1 2)
3
=> (* 2 3)
6
=> (+ (* 2 3) 4)
10
```

**This is REAL compilation** - generates ARM64 instructions (MOV, ADD, SUB, MUL) and executes them in mmap'd executable memory!

## Running

```bash
# Run REPL
cargo run

# Run tests
cargo test
```

## Project Structure

```
quick-clojure-poc/
├── src/
│   ├── value.rs         # Clojure value representation
│   ├── reader.rs        # EDN → Value conversion
│   ├── clojure_ast.rs   # AST representation
│   ├── eval.rs          # Tree-walking interpreter
│   ├── compiler.rs      # AST → IR compiler (for future ARM64 compilation)
│   ├── main.rs          # REPL
│   └── beagle/          # Reused components from Beagle (for future use)
│       ├── gc/          # Garbage collectors (3 implementations)
│       ├── ir.rs        # Intermediate representation
│       ├── types.rs     # Type system with 3-bit tagging
│       ├── code_memory.rs  # Executable memory management
│       └── machine_code/   # ARM64 code generation
└── tests/
    ├── stage0_test.clj  # Test cases for reader
    └── stage1_test.clj  # Test cases for evaluation
```

## Next Steps: Stage 2

**Goal**: Add more language features

**TODO**:
- [ ] Implement `fn` - function definition
- [ ] Implement `let` - local bindings
- [ ] Implement `loop`/`recur` - iteration
- [ ] Add sequence operations: `first`, `rest`, `cons`, `conj`, `nth`
- [ ] Add data structure constructors: `vector`, `hash-map`, `list`
- [ ] Namespace system

**Then**: Stage 3 - Compile to ARM64 using Beagle's backend!

## Dependencies

- **im** (15.1): Temporary persistent data structures (will be replaced in Stage 3)
- **clojure-reader** (0.4): Clojure/EDN parser

## Architecture

```
Stage 0 (Completed):
Source → clojure-reader → Value (im-rs backed)
                          ↓
                        Print

Stage 1 (Completed):
Source → Reader → Analyzer → AST → Eval → Result
                  ↓
           Special form recognition

Stage 2 (Next):
Add fn, let, loop/recur

Stage 3 (Future):
Source → Reader → Analyzer → AST → IR → ARM64
                                    ↓
                                Beagle backend
```

## Key Insight

The bootstrap problem is solved by:
1. Using im-rs crate as temporary scaffolding
2. Building a working compiler with im-rs
3. Using that compiler to compile ClojureScript's native data structures (Stage 3)
4. Switching to compiled native structures and re-bootstrapping

This mirrors how ClojureScript uses JavaScript arrays/objects temporarily before transitioning to its own persistent structures.

## Credits

This project reuses significant components from [Beagle](https://github.com/jimmyhmiller/beagle):
- Garbage collection implementations
- ARM64 code generation
- Intermediate representation
- Type system with pointer tagging
