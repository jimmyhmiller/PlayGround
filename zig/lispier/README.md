# Lispier

A complete Zig implementation of a reader and parser for the Lispier syntax (Lispy syntax for MLIR).

## Features

- **Complete Reader**: Converts source code into lists, vectors, maps, and symbols with full namespace tracking
- **Parser**: Transforms reader data structures into a well-typed AST suitable for MLIR code generation
- **Namespace System**: Built-in support for dialect imports with three notation styles:
  - Fully qualified: `(require-dialect arith)` → `arith.addi`
  - Aliased: `(require-dialect [arith :as a])` → `a/addi`
  - Unqualified: `(use-dialect arith)` → `addi`
- **MLIR Integration**: Uses c-mlir-wrapper to validate operations against loaded dialects
- **C API**: All reader types are exposed via C API for FFI interop
- **Extensive Tests**: Comprehensive test coverage for tokenizer, reader, parser, and integration

## Project Structure

```
lispier/
├── src/
│   ├── main.zig              # Main API and compiler
│   ├── repl.zig              # Interactive REPL
│   ├── tokenizer.zig         # Lexical analysis
│   ├── reader_types.zig      # Value types (List, Vector, Map, Symbol)
│   ├── reader.zig            # Reader with namespace tracking
│   ├── ast.zig               # AST node types
│   ├── parser.zig            # Parser (reader values → AST)
│   └── mlir_integration.zig  # MLIR dialect validation
├── tests/
│   └── integration_test.zig  # Integration tests
├── build.zig                 # Build configuration
└── SYNTAX.md                 # Complete syntax specification
```

## Building

### Prerequisites

1. **Zig** (0.11.0 or later)
2. **MLIR** installed with C API headers
3. **c-mlir-wrapper** library installed

### Install c-mlir-wrapper

```bash
cd ~/Documents/Code/PlayGround/claude-experiments/c-mlir-wrapper/
mkdir build && cd build
cmake ..
make
sudo make install
```

### Build Lispier

```bash
zig build
```

### Run Tests

```bash
zig build test
```

### Run REPL

```bash
zig build run  # Currently disabled
```

### Use Development Tools

```bash
# Show tokenization and reader output
zig build show-reader -- '(+ 1 2 3)'

# Show AST structure
zig build show-ast -- '(def x (+ 1 2))'
```

See [TOOLS.md](TOOLS.md) for complete documentation on development tools.

## Usage

### As a Library

```zig
const std = @import("std");
const lispier = @import("lispier");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var compiler = try lispier.Compiler.init(allocator);
    defer compiler.deinit();

    const source =
        \\(require-dialect arith)
        \\(arith.addi 1 2)
    ;

    var result = try compiler.compile(source);
    defer result.deinit(allocator);

    if (result.is_valid) {
        std.debug.print("Compiled successfully!\n", .{});
        // Access AST: result.nodes
    } else {
        std.debug.print("Validation errors:\n", .{});
        for (result.validation_errors) |err| {
            std.debug.print("  - {s}\n", .{err.message});
        }
    }
}
```

### REPL

```bash
$ zig build run
Lispier REPL v0.1.0
Type expressions or :quit to exit

> :help
Available commands:
  :help               - Show this help
  :quit, :q           - Exit the REPL
  :load-dialect NAME  - Load an MLIR dialect
  :dialects           - List loaded dialects
  :ops DIALECT        - List operations in a dialect

> :load-dialect arith
Loaded dialect: arith

> :ops arith
Operations in arith:
  - arith.addi
  - arith.subi
  - arith.muli
  ...

> (require-dialect arith)
✓ Compiled successfully
  Tokens: 4
  Values: 1
  AST Nodes: 1

> (arith.addi 1 2)
✓ Compiled successfully
  Tokens: 6
  Values: 1
  AST Nodes: 1
  Node 0: Operation
    Operation: arith.addi
    Operands: 2
    Regions: 0
```

## Architecture

### Reader Level

The reader implements **namespace-aware symbol resolution** from the start. When you parse:

```lisp
(require-dialect [arith :as a])
(a/addi 1 2)
```

The symbol `a/addi` is resolved to the `arith` namespace during reading, creating a `Symbol` with:
- `name`: "addi"
- `namespace`: pointer to "arith" namespace
- `uses_alias`: true

### Parser Level

The parser converts reader values into a typed AST:

```
Value (List)                →  Node (Operation)
  Symbol "arith.addi"       →    name: "addi"
  Number 1                  →    namespace: "arith"
  Number 2                  →    operands: [Literal(1), Literal(2)]
```

### MLIR Integration

The `DialectRegistry` uses c-mlir-wrapper to:
1. Load MLIR dialects dynamically
2. Validate that operations exist in their declared dialects
3. Enumerate available operations

## Key Components

### Tokenizer

Handles all syntax elements:
- Delimiters: `()`, `[]`, `{}`
- Symbols with complex types: `memref<128x128xf32>`
- Numbers: `42`, `3.14`, `1.5e-3`, `-10`
- Strings: `"hello\n"`
- Keywords: `:foo`, `:bar-baz`
- Block labels: `^bb1`, `^loop`
- Comments: `; comment`

### Reader

Converts tokens to values with namespace tracking:
- Lists: `(a b c)`
- Vectors: `[1 2 3]`
- Maps: `{:key value}`
- Symbols: `arith.addi`, `a/addi`, `addi` (with namespace info)
- Literals: numbers, strings, booleans, nil

Special forms processed during reading:
- `(require-dialect name)` or `(require-dialect [name :as alias])`
- `(use-dialect name)`

### Parser

Transforms reader values into AST nodes:
- **Operations**: MLIR operations with namespace, attributes, operands, regions
- **Regions**: Containers for blocks
- **Blocks**: Labeled blocks with arguments and operations
- **Bindings**: `def` and `let` forms
- **Type Annotations**: `(: value type)`
- **Literals**: Pass-through values

### AST Validator

Uses MLIR introspection to:
- Check that dialects exist
- Verify operations are defined in their dialects
- Collect validation errors

## C API

All reader types are exported for C FFI:

```c
// Create values
Value* lispier_value_create_list(void);
Value* lispier_value_create_number(double num);
Value* lispier_value_create_string(const char* str);

// Manipulate lists
bool lispier_value_list_append(Value* list, Value* item);

// Access data
double lispier_value_get_number(Value* val);
const char* lispier_value_get_string(Value* val);

// Cleanup
void lispier_value_destroy(Value* val);
```

## Examples

### Simple Arithmetic

```lisp
(require-dialect arith)
(def x 42)
(def y 10)
(def sum (arith.addi x y))
```

### Function with Blocks

```lisp
(require-dialect [func :as f] [arith :as a])

(f/func {:sym_name "add" :function_type (-> [i64 i64] [i64])}
  (do
    (block [(: arg0 i64) (: arg1 i64)]
      (def result (a/addi arg0 arg1))
      (f/return result))))
```

### Control Flow

```lisp
(require-dialect cf arith)

(do
  (block [(: n i64)]
    (cf.br {:successors [^loop]} n))

  (block ^loop [(: iter i64)]
    (def is_zero (arith.cmpi {:predicate "eq"} iter 0))
    (cf.cond_br {:successors [^done ^continue]
                 :operand_segment_sizes [1 0 1]}
                is_zero iter))

  (block ^continue [(: val i64)]
    (def next (arith.subi val 1))
    (cf.br {:successors [^loop]} next))

  (block ^done
    (arith.constant {:value 0})))
```

### Mixed Notations

```lisp
(require-dialect arith [func :as f])
(use-dialect memref)

(def buffer (alloc))          ; unqualified (from use-dialect)
(def sum (arith.addi 1 2))    ; fully qualified
(f/return sum)                 ; aliased
```

## Testing

The project includes extensive tests at multiple levels:

### Unit Tests

Each module has inline tests:
- `tokenizer.zig`: 10+ tests for lexical analysis
- `reader.zig`: Tests for namespace resolution
- `parser.zig`: Tests for AST generation
- `mlir_integration.zig`: Tests for dialect loading

### Integration Tests

`tests/integration_test.zig` includes:
- Simple arithmetic
- Functions with blocks
- Let bindings
- Control flow
- Type annotations
- Attribute parsing
- Invalid dialect/operation detection
- Mixed notation styles
- Destructuring
- Block labels and arguments

Run all tests via the build script (ensures MLIR/LLVM link flags and include paths are set):
```bash
zig build test             # unit tests
zig build integration-test # integration tests
```

## Next Steps

Potential extensions:

1. **MLIR Code Generation**: Convert AST to actual MLIR operations via C API
2. **Type Inference**: Implement type propagation and inference
3. **Macro System**: Add compile-time code generation
4. **Error Recovery**: Better error messages with source locations
5. **Standard Library**: Common patterns and utilities in Lispier syntax
6. **LSP Server**: Language Server Protocol for IDE support

## License

See parent repository for license information.

## Related

- [SYNTAX.md](SYNTAX.md) - Complete syntax specification
- [c-mlir-wrapper](https://github.com/jimmyhmiller/c-mlir-wrapper) - MLIR introspection C API
- [MLIR](https://mlir.llvm.org/) - Multi-Level Intermediate Representation
