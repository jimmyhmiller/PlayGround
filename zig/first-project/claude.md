# Project Capabilities (September 2024)

## Language Features
- Literals: integers (generic + sized), floats (F32/F64), strings, nil
- Symbols, keywords, namespaces `(ns …)`
- Lists, vectors, maps
- Struct and enum definitions: `(def Point (: Type) (Struct [x Int] [y Int]))`
- Function definitions `(def … (fn …))`, higher-order functions, function types `(-> …)`
- Forward references via two-pass type checking
- `let` bindings with type annotations and scoped environments
- Arithmetic `+ - * / %` with numeric-type merging (ints/floats)
- Comparison operators `< > <= >= = !=` returning `Bool`
- Conditional expressions `(if …)` with branch type unification
- Recursive functions (verified with typed Fibonacci example)

## Type System
- Bidirectional checker with subtype support for sized ints/floats
- Numeric literal polymorphism (int literals adapt to expected numeric type)
- Type annotation parsing for variables, functions, structs, enums
- Detailed error reporting (expression index, error kind, offending form)

## Code Generation
- Simple C backend that validates via the type checker before emission
- Typed C output (correct primitive widths, required headers)
- Executable mode: emits `int main()` and optionally runs compiled binary
- Bundle mode: emits `lisp_main` entry point, builds macOS `.bundle`, dynamic load support

## Tooling / CLI
- `zig run src/simple_c_compiler.zig -- <file>` → compile to C
- `--run` builds/executes the generated program
- `--bundle` builds a macOS bundle; `--bundle --run` loads and invokes `lisp_main`
- Type-check failures print a concise diagnostic with the problematic form

## REPL Implementation Details
- The REPL currently has **NO runtime state** - it recompiles from scratch on every input
- Redefinition mechanism: tracks all `(def ...)` forms by name in `definitions_map`
- On each input, concatenates ALL previous definitions + new input, recompiles to C, compiles to `.bundle`, loads and executes
- This means:
  - Function/struct/enum redefinitions work by replacing the definition in the map and recompiling everything
  - No incremental compilation or state preservation between evaluations
  - Each evaluation starts fresh - previous computation results are lost
  - **WARNING**: This approach will NOT support proper stateful programs (e.g., counters, accumulators)
  - Future work needed: proper runtime state management if stateful REPL is desired

## Test Coverage
- `zig test src/test_all.zig` covers lexer, parser, reader, type checker, backend, simple C compiler
- Backend tests include arithmetic, `if`, structs/enums, forward references, lets, etc.
- Sample typed Fibonacci demonstrates U32 recursion through both executable and bundle workflows


# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
MUST use the scratch/ folder for all ad-hoc testing files. Never create test files in the project root.
