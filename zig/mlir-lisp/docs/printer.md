# Printer Module

The printer module (`src/printer.zig`) converts parsed MLIR AST structures back into s-expression format, enabling round-trip transformations.

## Purpose

The printer ensures that data parsed into structured AST can be serialized back to text without losing information. This is crucial for:

- **Round-trip validation**: Verifying that parsing preserves all information
- **Code generation**: Outputting MLIR in s-expression format
- **Debugging**: Inspecting parsed structures in human-readable form
- **Transformation pipelines**: Serializing modified AST back to source

## Features

### Complete Coverage

The printer handles all MLIR constructs:

- **Operations**: Name, result bindings, types, operands, attributes
- **Regions**: Nested region structures
- **Blocks**: Labels, arguments, and contained operations
- **Successors**: Control flow targets with operands
- **Attributes**: Key-value pairs with various value types
- **Types and Attributes**: Prefixed expressions (`!type` and `#attr`)
- **Values**: All reader value types including literals, identifiers, collections

### Indentation

Output is properly indented for readability:
- 2-space indentation (configurable)
- Nested structures are indented appropriately
- Operations, regions, and blocks are formatted hierarchically

### Error Handling

Uses explicit `PrintError` error set for clear error propagation.

## Usage

```zig
const Printer = @import("mlir_lisp").Printer;

// Create printer
var printer = Printer.init(allocator);
defer printer.deinit();

// Print a module
try printer.printModule(&module);

// Get the output
const output = printer.getOutput();
std.debug.print("{s}\n", .{output});
```

## Round-Trip Example

```zig
// Input
const input =
    \\(mlir
    \\  (operation
    \\    (name arith.constant)
    \\    (result-bindings [%c0])
    \\    (result-types !i32)
    \\    (attributes { :value (#int 42) })))
;

// Parse
var tok = Tokenizer.init(allocator, input);
var reader = try Reader.init(allocator, &tok);
var value = try reader.read();
defer value.deinit(allocator);

var parser = Parser.init(allocator);
var module = try parser.parseModule(value);
defer module.deinit();

// Print
var printer = Printer.init(allocator);
defer printer.deinit();
try printer.printModule(&module);

const output = printer.getOutput();
// Output is semantically equivalent to input
```

## Implementation Details

### Design Decisions

1. **Writer Pattern**: Uses `ArrayList(u8).writer()` for efficient buffer writing
2. **Explicit Errors**: Uses `PrintError` to avoid inferred error set issues
3. **Recursive Printing**: Handles nested structures through recursive function calls
4. **Value Preservation**: Directly prints wrapped `Value` objects from the reader

### Output Format

The printer produces output that:
- Is valid input for the tokenizer/reader/parser
- Preserves all semantic information
- May differ in whitespace from original input
- Uses consistent formatting rules

### Testing

Comprehensive tests in `test/printer_test.zig` verify:
- Simple operations with attributes
- Functions with regions and blocks
- Control flow with successors
- Empty structures (empty blocks, no arguments)
- Full round-trip: input → parse → print → parse → verify

### Demo Programs

- `zig build test-printer`: Simple constant example
- `zig build test-complex-printer`: Complex function with nested structures

Both demos show:
1. Original input
2. Parsed structure details
3. Printed output
4. Re-parsed verification
5. Round-trip confirmation

## See Also

- [Parser](./parser.md) - Converts reader Values to typed AST
- [Reader](./reader.md) - Converts tokens to s-expression Values
- [Grammar](./grammar.md) - Complete language specification
