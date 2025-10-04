# Project Capabilities (January 2025)

## Language Features

### Literals & Basic Types
- **Integer literals**: generic `Int` and sized (`U8`, `U16`, `U32`, `U64`, `I8`, `I16`, `I32`, `I64`)
- **Float literals**: generic `Float` and sized (`F32`, `F64`)
- **String literals**: `String` type with full string support
- **Boolean literals**: `true`, `false` with `Bool` type
- **Nil**: `nil` with `Nil` type
- **Symbols & Keywords**: full symbol and keyword support
- **Collections**: lists, vectors `[...]`, maps (basic support)

### Type Definitions
- **Struct definitions**: `(def Point (: Type) (Struct [x Int] [y Int]))`
  - Named struct types with typed fields
  - Struct constructors: `(Point 10 20)` creates struct instances
  - Struct field access: `(. struct-expr field-name)` reads field value
  - Works with struct values, pointers to structs, and extern types
  - Nested struct support (structs as field types)
  - Empty structs allowed
- **Enum definitions**: `(def Color (: Type) (Enum Red Green Blue))`
  - Variant access via qualified names: `Color/Red`, `Color/Blue`
  - Type-safe enum usage with exhaustive checking

### Functions
- **Function definitions**: `(def f (: (-> [Int] Int)) (fn [x] (+ x 1)))`
- **Function types**: `(-> [ParamTypes...] ReturnType)`
- **Multi-parameter functions**: supports arbitrary parameter count
- **Recursive functions**: fully supported with two-pass type checking
- **Mutual recursion**: forward references enable mutually recursive functions
- **Higher-order functions**: functions as values (basic support, closures limited)

### Control Flow
- **Conditional expressions**: `(if condition then-expr else-expr)`
  - Branch type unification (both branches must have compatible types)
  - Nested conditionals supported
- **Loop constructs**:
  - `(while condition body*)` - standard while loops
  - `(c-for [var (: Type) init] test step body*)` - C-style for loops

### Variables & Bindings
- **Global definitions**: `(def name (: Type) value)`
- **Let bindings**: `(let [x (: Int) 42] body)`
  - Multiple bindings in sequence
  - Sequential dependencies (later bindings can use earlier ones)
  - Scoped environments with proper shadowing
  - Nested let expressions
- **Mutable variables**: `(set! var value)` for mutation

### Operators
- **Arithmetic**: `+ - * / %`
  - Numeric type merging (int/float promotion)
  - Multiple operand support for `+` and `*`
  - Type-safe operations with proper coercion
- **Comparison**: `< > <= >= = !=`
  - Returns `Bool` type
  - Works with numeric types (int and float)
- **Boolean/Logical**: `and`, `or`, `not`
  - Short-circuit evaluation
  - Returns `Bool` type

### Memory Management (Pointers)
- **Pointer types**: `(Pointer T)` for any type `T`
- **Allocation**: `(allocate Type value)` - heap allocation
- **Deallocation**: `(deallocate ptr)` - manual memory management
- **Dereferencing**: `(dereference ptr)` - read through pointer
- **Writing**: `(pointer-write! ptr value)` - write through pointer
- **Address-of**: `(address-of value)` - get pointer to value
- **Null pointers**: `pointer-null` - null pointer literal
- **Pointer equality**: `(pointer-equal? ptr1 ptr2)` - compare pointers
- **Struct field access**:
  - `(pointer-field-read ptr field)` - read struct field through pointer
  - `(pointer-field-write! ptr field value)` - write struct field through pointer
- **Nested pointers**: `(Pointer (Pointer T))` - arbitrary nesting supported

### Arrays (Fixed-Size)
- **Array types**: `(Array ElementType Size)` - fixed-size arrays
  - Example: `(Array Int 10)` - array of 10 integers
  - Multi-dimensional: `(Array (Array Float 3) 2)` - 2x3 matrix
- **Array creation**:
  - `(array Type Size)` - uninitialized array
  - `(array Type Size InitValue)` - initialized array
- **Array operations**:
  - `(array-ref arr index)` - read element at index
  - `(array-set! arr index value)` - write element at index
  - `(array-length arr)` - get compile-time size
  - `(array-ptr arr index)` - get pointer to element
- **Heap arrays** (pointer-based):
  - `(allocate-array Type Size)` - allocate array on heap
  - `(allocate-array Type Size InitValue)` - allocate with initialization
  - `(deallocate-array ptr)` - free heap array
  - `(pointer-index-read ptr index)` - read from pointer array
  - `(pointer-index-write! ptr index value)` - write to pointer array
- **Type safety**: All array operations are type-checked
- **Integration**: Arrays work with structs, pointers, and functions

### Namespaces
- Basic namespace support with `(ns ...)`
- Qualified name resolution for enum variants

## Type System

### Type Checking
- **Bidirectional type checker**: combines synthesis and checking modes
- **Two-pass type checking**: enables forward references and mutual recursion
- **Subtyping**: sized integers/floats are subtypes of generic Int/Float
- **Numeric literal polymorphism**: int/float literals adapt to expected type
- **Type inference**: synthesizes types where annotations not required
- **Type annotations**: `(: Type)` syntax for explicit typing

### Error Reporting
- Detailed error messages with:
  - Expression index in source
  - Error kind (TypeMismatch, UnboundVariable, etc.)
  - Offending form displayed for debugging
- Comprehensive error types:
  - `TypeMismatch` - incompatible types
  - `UnboundVariable` - undefined variable reference
  - `ArgumentCountMismatch` - wrong number of function arguments
  - `CannotSynthesize` - unable to infer type
  - `InvalidTypeAnnotation` - malformed type annotation

## Code Generation

### C Backend
- **Type-safe C emission**: validates all code through type checker before generation
- **Typed output**:
  - Correct primitive widths (`uint8_t`, `int32_t`, `float`, `double`)
  - Required headers automatically included (`<stdint.h>`, `<stdbool.h>`, etc.)
  - Pointer types mapped to C pointers
  - Struct definitions with proper C syntax

### Compilation Modes
- **Executable mode**:
  - Emits `int main()` wrapper
  - Compiles to standalone binary
  - `--run` flag to build and execute
- **Bundle mode**:
  - Emits `lisp_main()` entry point
  - Builds macOS `.bundle` (dynamic library)
  - `--bundle --run` loads and invokes dynamically
  - Enables REPL integration

## Tooling & CLI

### Compiler
- **Usage**: `zig run src/simple_c_compiler.zig -- <file>`
- **Flags**:
  - `--run` - compile and execute immediately
  - `--bundle` - build as dynamic library
  - `--bundle --run` - build and dynamically load/execute

### REPL
- **Interactive environment** with live evaluation
- **Implementation approach**:
  - NO runtime state - recompiles from scratch each input
  - `definitions_map` tracks all `(def ...)` forms by name
  - Concatenates ALL previous definitions + new input
  - Recompiles to C → builds `.bundle` → loads and executes
- **Capabilities**:
  - Function/struct/enum redefinitions (replaces in map, recompiles)
  - No incremental compilation
  - Each evaluation starts fresh
- **Limitations**:
  - Previous computation results are lost between evaluations
  - **NOT suitable for stateful programs** (counters, accumulators, etc.)
  - Future work needed for proper runtime state management

## Test Coverage

### Test Suite
- **Main test runner**: `zig test src/test_all.zig`
- **184 test cases** covering all major features
- **Test categories**:
  - Lexer and parser tests
  - Reader tests (S-expression parsing)
  - Type checker comprehensive tests (positive & negative cases)
  - Backend code generation tests
  - Forward reference and mutual recursion tests
  - Pointer operation tests
  - Struct and enum tests
  - Struct field access tests (`(. struct field)` syntax)
  - **Array tests** (23 tests):
    - Stack array type checking and code generation
    - Multi-dimensional arrays
    - Heap array allocation (`allocate-array`, `deallocate-array`)
    - Pointer indexing (`pointer-index-read`, `pointer-index-write!`)
    - Array pointer operations (`array-ptr`)
    - Arrays in struct fields
  - Function application tests
  - Control flow tests

### Example Programs
- **Fibonacci**: typed recursive implementation with U32
- **Type showcase**: demonstrates all type system features
- Both executable and bundle compilation modes tested


# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
MUST use the scratch/ folder for all ad-hoc testing files. Never create test files in the project root.
