# Lisp0 Build 0.1.0 Release

**Release Date:** October 18, 2025

---

## What's Included

This release contains:

1. **`bin/lisp0`** - Standalone compiler binary (728KB, optimized)
2. **`docs/LANGUAGE_REFERENCE.md`** - Comprehensive language documentation
3. **`examples/`** - 10 working example programs demonstrating all major features

---

## Quick Start

### Compile and Run a Program

```bash
./bin/lisp0 examples/01_hello_world.lisp --run
```

### Compile Only

```bash
./bin/lisp0 examples/02_fibonacci.lisp
./examples/02_fibonacci
```

### Available Flags

- `--run` - Compile and execute immediately
- `--bundle` - Build as dynamic library (macOS .bundle)
- `--bundle --run` - Build and dynamically load/execute

---

## Example Programs

All examples are fully tested and working:

| # | File | Demonstrates |
|---|------|--------------|
| 01 | `01_hello_world.lisp` | Basic program structure, C FFI, printf |
| 02 | `02_fibonacci.lisp` | Recursion, conditionals, typed integers |
| 03 | `03_structs_and_points.lisp` | Struct definitions, field access, nested structs |
| 04 | `04_pointers_and_memory.lisp` | Heap allocation, pointers, dereferencing |
| 05 | `05_arrays.lisp` | Heap-allocated arrays, pointer indexing |
| 06 | `06_loops_and_iteration.lisp` | while loops, c-for loops, mutation |
| 07 | `07_higher_order_functions.lisp` | Functions as values, composition |
| 08 | `08_macros.lisp` | Macro definitions, syntax-quote, gensym |
| 09 | `09_linked_list.lisp` | Recursive data structures, pointers |
| 10 | `10_enums_and_pattern_matching.lisp` | Enums, pattern matching with if |

### Running All Examples

```bash
for f in examples/*.lisp; do
  echo "=== Running $f ==="
  ./bin/lisp0 "$f" --run
  echo
done
```

---

## Language Features

### Core Features
- ✅ Static type checking with bidirectional type inference
- ✅ First-class functions with recursion and mutual recursion
- ✅ Structs and enums (algebraic data types)
- ✅ Manual memory management (pointers, allocate, deallocate)
- ✅ Heap-allocated arrays
- ✅ C FFI (include headers, declare/extern functions)
- ✅ Compile-time macros with hygiene support (gensym)
- ✅ Control flow (if, while, c-for)
- ✅ Let bindings with scoping
- ✅ Mutation (set!)

### Type System
- Integer types: `Int`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`
- Float types: `Float`, `F32`, `F64`
- Boolean: `Bool`
- Strings: `String`
- Pointers: `(Pointer T)`
- Arrays: `(Array T N)` (heap-allocated via `allocate-array`)
- Structs: `(Struct [field Type] ...)`
- Enums: `(Enum Variant1 Variant2 ...)`
- Functions: `(-> [ParamTypes...] ReturnType)`

---

## Known Limitations

### Bugs (See BUGS.md)

1. **Arrays in local let bindings** (ID: `concerned-overcooked-crawdad`)
   - Stack arrays `(Array T N)` fail in local `let` bindings
   - **Workaround:** Use `allocate-array` which returns `(Pointer T)`

2. **Unhelpful error messages** (ID: `fortunate-sociable-mastodon`)
   - "ERROR writing return expression: UnsupportedExpression" lacks context
   - No source location, no hint about the actual problem

3. **deallocate in nested contexts**
   - `deallocate` may fail with code generation errors in certain expression positions
   - **Workaround:** Avoid deallocate in deeply nested expressions

### Missing Features
- No closures (higher-order functions work, but can't capture environment)
- No variadic macros (`& rest` syntax)
- No garbage collection (manual memory management only)
- No pattern matching (must use if/else chains)
- No module system beyond basic namespaces

---

## Documentation

See `docs/LANGUAGE_REFERENCE.md` for complete language documentation including:

- All supported types and literals
- Function definitions and recursion
- Struct and enum syntax
- Memory management with pointers
- C FFI best practices
- Macro system details
- Control flow constructs
- Error messages reference

---

## System Requirements

- **macOS** (tested on Darwin 25.0.0)
- **Zig compiler** (for building from source)
- **C compiler** (clang/gcc for compiling generated C code)

---

## Example: Hello World

```lisp
(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Hello, World!\n"))
    0))
```

Compile and run:
```bash
./bin/lisp0 examples/01_hello_world.lisp --run
```

---

## Example: Fibonacci

```lisp
(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

(def fib (: (-> [U32] U32))
  (fn [n]
    (if (<= n 1)
        n
        (+ (fib (- n 1))
           (fib (- n 2))))))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "fib(10) = %u\n") (fib 10))
    0))
```

---

## Contributing

Found a bug? Please report it using the bug tracker:

```bash
bug-tracker list  # View all bugs
```

Or check `BUGS.md` in the project root.

---

## License

See main project repository for license information.

---

**Happy Hacking!** 🚀
