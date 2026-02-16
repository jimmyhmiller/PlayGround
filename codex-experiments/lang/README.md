# Lang - Self-Hosting Compiler

Lang is a statically-typed systems programming language with a **self-hosting compiler**. The project includes both a Rust-based compiler and a bootstrap compiler written entirely in Lang itself.

## Status: Self-Hosting ✓

The Lang compiler is **fully self-hosting** and can compile itself through multiple bootstrap stages (T2 verified). The compiler pipeline implements:

- **Lexer** - Tokenization with keyword recognition
- **Parser** - Recursive descent parser producing AST
- **Qualifier** - Path resolution and module qualification
- **Resolver** - Symbol resolution across modules
- **Typechecker** - Static type checking with inference
- **Codegen** - LLVM IR generation and AOT compilation

## Language Features

- **Static typing** with type inference
- **Algebraic data types** - enums with pattern matching
- **Structs** with named fields
- **Generics** - parametric polymorphism
- **Pattern matching** - exhaustive match expressions
- **First-class functions** - closures and higher-order functions
- **Mutable variables** - explicit `mut` keyword
- **Module system** - multi-file compilation with use imports
- **Foreign function interface** - extern declarations for C interop
- **Memory safety** - GC-allocated reference semantics

## Quick Example

```lang
enum State { Start, Working, Done }
struct Counter { value: I64 }

fn main() -> I64 {
    let mut c: Counter = Counter { value: 0 };
    let mut s: State = State::Start();

    let mut i: I64 = 0;
    while i < 5 {
        c.value = c.value + 1;
        i = i + 1;
    };

    s = State::Working();
    match s {
        State::Start => 0,
        State::Working => c.value,
        State::Done => -1
    }
}
```

## Building

### Prerequisites

- Rust (latest stable)
- LLVM 19.x (`brew install llvm@19` on macOS)
- Clang

### Build the Rust Compiler

```bash
cargo build --release
```

The compiler binary will be at `target/release/langc`.

## Using the Compiler

The `langc` compiler has four modes:

### 1. Type Check Only
```bash
# Just parse and type check (no code generation)
cargo run --release -- check examples/generics_test.lang
# Output: "ok" if successful
```

### 2. Run (Compile and Execute)
```bash
# Compile to temporary executable and run immediately
cargo run --release -- run examples/generics_test.lang
# Output: 42

# Pass arguments to your program (after --)
cargo run --release -- run examples/generics_test.lang -- arg1 arg2
```

This mode:
- Compiles your `.lang` files to LLVM IR
- Links with the runtime
- Executes the result
- Cleans up temporary files

### 3. Build (Create Executable)
```bash
# Build a standalone executable
cargo run --release -- build examples/generics_test.lang
# Creates: build/generics_test (executable)

# Run the executable
./build/generics_test
# Output: 42
```

The compiler produces **complete executables** with all necessary runtime libraries linked in. No separate linking step needed!

### 4. Bootstrap (Self-Hosting Mode)
```bash
# Build the bootstrap compiler (compiler written in Lang)
cargo run --release -- build compiler/main.lang
# Creates: build/main (the bootstrap compiler executable)

# Use it to compile programs
./build/main compiler/main.lang
# Creates: build/output (executable)

# Or use the automated bootstrap mode
cargo run --release -- bootstrap
```

This demonstrates **complete self-hosting** - the Lang compiler (written entirely in Lang) compiles itself into a working executable!

The bootstrap compiler produces **complete executables**, not object files. It's a fully functional compiler.

## Multi-File Compilation

The compiler automatically discovers and compiles all modules:

```lang
// main.lang
use mymodule::foo;

fn main() -> I64 {
    foo()
}
```

```lang
// mymodule.lang
fn foo() -> I64 { 42 }
```

```bash
cargo run --release -- run main.lang
# Automatically finds and compiles mymodule.lang
```

The compiler searches for modules in:
1. The same directory as the importing file
2. The `stdlib/` directory

## Example Programs

The `examples/` directory contains several test programs:

- **`generics_test.lang`** ✓ - Demonstrates generic functions (working)
- **`break_continue_test.lang`** ✓ - Loop control flow (working)
- **`tree_sum.lang`** - Recursive tree data structure
- **`binarytrees.lang`** - Benchmark program
- **`state_machine.lang`** - Enum-based state machine (needs updating)

Try them out:
```bash
cargo run --release -- run examples/generics_test.lang
cargo run --release -- run examples/break_continue_test.lang
```

## Project Structure

```
├── src/                    # Rust-based compiler implementation
│   ├── main.rs            # CLI entry point and multi-file compilation
│   ├── lexer.rs           # Tokenizer
│   ├── parser.rs          # Parser (AST construction)
│   ├── qualify.rs         # Path qualification pass
│   ├── resolve.rs         # Symbol resolution
│   ├── typecheck.rs       # Type checker
│   ├── codegen.rs         # LLVM IR codegen
│   └── runtime.rs         # Runtime support functions
│
├── compiler/               # Self-hosting compiler (written in Lang)
│   ├── main.lang          # Bootstrap entry point (291 lines)
│   ├── lexer.lang         # Lexer (661 lines)
│   ├── parser.lang        # Parser (1259 lines)
│   ├── ast.lang           # AST definitions (352 lines)
│   ├── qualify.lang       # Qualification (589 lines)
│   ├── resolve.lang       # Resolution (677 lines)
│   ├── typecheck.lang     # Type checking (1462 lines)
│   └── codegen.lang       # LLVM codegen (2497 lines)
│
├── stdlib/                 # Standard library (written in Lang)
│   ├── io.lang            # File I/O, stdio, process control
│   ├── string.lang        # String operations
│   └── vec.lang           # Dynamic arrays/vectors
│
├── runtime/                # C runtime support
│   └── runtime.c          # Low-level runtime (GC, intrinsics)
│
├── examples/               # Example programs
│   ├── state_machine.lang # Enums and pattern matching
│   ├── tree_sum.lang      # Recursive data structures
│   ├── binarytrees.lang   # Benchmark
│   ├── generics_test.lang # Generic functions
│   └── break_continue_test.lang
│
└── docs/                   # Design documentation
```

## Compiler Pipeline

```
Source (.lang files)
  ↓
Lexer → Tokens
  ↓
Parser → AST
  ↓
Qualifier → Qualified paths
  ↓
Resolver → Symbol table
  ↓
Typechecker → Type-checked AST
  ↓
Codegen → LLVM IR
  ↓
AOT Compilation → Object files
  ↓
Linker → Executable
```

## Implementation Highlights

### Multi-File Compilation

The compiler discovers and compiles all modules starting from entry files, following `use` declarations to build the complete module graph.

### AOT vs JIT

The compiler uses **Ahead-of-Time (AOT) compilation** exclusively:
- Generates LLVM IR for all modules
- Produces object files via LLVM
- Links with `liblang_runtime.a` and system libraries
- Avoids JIT due to LLVM FastISel issues with complex pattern matching

### Memory Model

All structs and enums are **GC-allocated pointers** with reference semantics:
- No explicit memory management needed
- Uniform representation simplifies codegen
- Vectors use `RawPointer<I8>` internally with typed access functions

### Type System

- **Primitives**: `I64`, `Bool`, `String`, `RawPointer<T>`
- **Structs**: Named fields, reference semantics
- **Enums**: Tagged unions with pattern matching
- **Generics**: Monomorphization (planned)
- **Functions**: First-class values with closure support

## Standard Library

The standard library is written entirely in Lang:

- **`io`** - File I/O (`read_file`, `write_file`), stdio (`print_str`), process control (`exit_process`, `arg_str`)
- **`string`** - String operations (`string_len`, `string_eq`, `string_concat`, `string_slice`, `string_from_i64`, `parse_i64`)
- **`vec`** - Dynamic vectors with typed access functions

## Runtime Support

C runtime (`runtime/runtime.c`) provides:
- Pointer intrinsics for vector operations
- Command-line argument access
- File system operations
- Process control

Rust runtime (`src/runtime.rs`) exports:
- Vector operations
- String operations
- Memory allocation wrappers

## Known Limitations

See `~/.claude/projects/-Users-jimmyhmiller-Documents-Code-PlayGround/memory/bootstrap-lessons.md` for detailed notes on:
- LLVM FastISel crashes (workaround: use AOT)
- Extern function deduplication requirements
- Method call limitations (no chained field access on call results)
- Keyword restrictions as identifiers
- Variable shadowing in nested patterns

## Bootstrap Verification

The self-hosting compiler has been verified through **T2 bootstrap**:
1. Rust compiler compiles Lang bootstrap compiler (T0)
2. T0 compiles itself to produce T1
3. T1 compiles itself to produce T2
4. Binary comparison confirms T1 ≡ T2 ✓

## Contributing

This is an experimental project exploring self-hosting compiler design. The codebase demonstrates:
- How to build a complete compiler pipeline from scratch
- Self-hosting bootstrap techniques
- LLVM IR generation from a high-level IR
- Multi-pass compilation architecture

## License

MIT (or specify your license)
