# MLIR Parser for Zig

A grammar-driven recursive descent parser for MLIR (Multi-Level Intermediate Representation) written in Zig.

## Features

- **MLIR Parser**: Parse MLIR generic format to AST
- **MLIR Printer**: Print AST back to MLIR format (roundtrip support)
- **Lisp Converter**: Convert MLIR to S-expression Lisp format
- Grammar-driven implementation following official MLIR specification

## Quick Start

### Installation

Build and install all binaries globally:

```bash
./install.sh
```

This will:
- Build the project with `zig build`
- Install `mlir-to-lisp`, `mlir_parser`, and `debug_printer` to `~/.local/bin`
- Create symlinks so you can use them from anywhere

Make sure `~/.local/bin` is in your PATH:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Usage

```bash
# Build everything
zig build

# Run tests
zig build test

# Convert MLIR to Lisp S-expressions (after installation)
mlir-to-lisp myfile.mlir

# Or run without installation
zig build mlir-to-lisp -- myfile.mlir

# Get help
mlir-to-lisp --help
```

## Project Structure

```
mlir-parser/
├── grammar.ebnf              # Official MLIR grammar (source of truth)
├── CLAUDE.md                 # Developer guide with rules and patterns
├── README.md                 # This file
├── build.zig                 # Build configuration
├── src/
│   ├── lexer.zig            # Tokenization
│   ├── ast.zig              # AST node definitions
│   ├── parser.zig           # Recursive descent parser
│   ├── printer.zig          # MLIR printer (roundtrip support)
│   ├── lisp_printer.zig     # Lisp S-expression converter
│   ├── root.zig             # Public API exports
│   ├── main.zig             # Main executable
│   ├── mlir_to_lisp.zig     # MLIR to Lisp converter CLI
│   └── debug_printer.zig    # Debug utility
├── test/
│   ├── basic_test.zig
│   ├── integration_test.zig
│   ├── roundtrip_test.zig
│   ├── lisp_printer_test.zig
│   └── ...
├── test_data/
│   └── examples/            # Test MLIR files (generic format)
├── scripts/
│   ├── convert_to_generic.sh    # Convert MLIR to generic format
│   └── validate_examples.sh     # Validate MLIR files
├── docs/                    # Documentation and reports
└── archive/                 # Old test files and examples
```

## Tools

### MLIR to Lisp Converter

Convert MLIR (generic format) to S-expression Lisp format:

```bash
# Convert a file
zig build mlir-to-lisp -- fibonacci.mlir

# Example output
(mlir
  (operation
    (name builtin.module)
    (regions
      (region
        (block
          (arguments [])
          (operation
            (name func.func)
            (attributes {:function_type (!function (inputs i32) (results i32))
                        :sym_name @fibonacci})
            ...))))))
```

### Convert to Generic Format

Before parsing, convert MLIR to generic format:

```bash
./scripts/convert_to_generic.sh
```

This uses `mlir-opt --mlir-print-op-generic` to convert all `.mlir` files in `test_data/examples/` to the generic format required by the parser.

## Grammar-Driven Approach

**Every parser and printer function MUST have a grammar comment.** This is the core principle of this project.

Example:
```zig
// Grammar: type ::= type-alias | dialect-type | builtin-type
pub fn parseType(self: *Parser) ParseError!ast.Type {
    // Implementation follows the grammar rule exactly
    ...
}
```

See `CLAUDE.md` for complete development guidelines.

## Testing

All test files in `test_data/examples/` are automatically discovered and tested:

```bash
# Run all tests
zig build test

# Tests include:
# - Lexer tests
# - Parser tests
# - Roundtrip tests (parse → print → parse)
# - Lisp printer tests
# - Integration tests
```

## API Usage

```zig
const std = @import("std");
const mlir = @import("mlir_parser");

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const source = "%0 = \"arith.constant\"() <{value = 42 : i32}> : () -> i32";

    // Parse MLIR
    var module = try mlir.parse(allocator, source);
    defer module.deinit();

    // Print back to MLIR
    const mlir_output = try mlir.print(allocator, module);
    defer allocator.free(mlir_output);

    // Convert to Lisp
    const lisp_output = try mlir.printLisp(allocator, module);
    defer allocator.free(lisp_output);
}
```

## What Works

### ✅ Complete Features
- Lexer with all token types
- AST node definitions
- Type parsing (integer, float, index, function, dialect, tensor, memref, vector, complex, tuple)
- Operation parsing (generic operations)
- Block and region parsing
- Attributes and properties
- Successors (control flow)
- MLIR printer (roundtrip support)
- **Lisp S-expression converter**

### ✅ Lisp Format Highlights
- Converts MLIR to clean S-expression format
- Function types: `(!function (inputs i32) (results i32))`
- Typed literals: `(: 42 i32)`
- Symbol references: `@fibonacci`
- Keyword attributes: `:value`, `:sym_name`
- Full nested region and block support

## Testing Strategy

1. **Simple examples first**: Start with constants
2. **Validate with mlir-opt**: Every test case validated
3. **Generic format only**: All tests use generic format
4. **Incremental complexity**: Operations → blocks → regions
5. **Roundtrip testing**: Parse → print → parse stability
6. **Real-world examples**: Test with actual MLIR from LLVM

## Documentation

- `CLAUDE.md` - Complete developer guide with Zig 0.15.1 API notes
- `docs/LISP_CONVERTER_PLAN.md` - Lisp converter implementation plan
- `docs/ROUNDTRIP_TEST_REPORT.md` - Roundtrip testing results
- `docs/TEST_STATUS.md` - Test status summary
- `docs/VERIFICATION_SUMMARY.md` - Verification report

## References

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin/)
- Grammar file: `grammar.ebnf` (in this repository)
- Development guide: `CLAUDE.md` (in this repository)

---

**Key Philosophy**: This parser is built to be maintainable and correct by strictly following the grammar. When in doubt, consult `grammar.ebnf` and the MLIR Language Reference.
