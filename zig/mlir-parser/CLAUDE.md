# MLIR Parser Project - Developer Guide

## Project Overview

This project implements a **recursive descent parser for MLIR** (Multi-Level Intermediate Representation) written in Zig. The parser strictly follows the official MLIR grammar specification and produces an Abstract Syntax Tree (AST).

## Core Principles

### 1. Grammar-Driven Development

Every parser function **must** directly correspond to a production rule in `grammar.ebnf`. This ensures:
- The parser structure mirrors the grammar
- Code is self-documenting
- Easier to verify correctness against the spec

**Example:**
```zig
// Grammar: operation ::= op-result-list? (generic-operation | custom-operation) trailing-location?
fn parseOperation() !Operation {
    // Implementation follows the grammar rule exactly
    const result_list = try self.parseOpResultList() orelse null;
    const op = try self.parseGenericOperation(); // or parseCustomOperation()
    const location = try self.parseTrailingLocation() orelse null;
    return Operation{ .results = result_list, .operation = op, .location = location };
}
```

### 2. Mandatory Grammar Comments

**Every parsing function MUST include a comment showing the exact EBNF rule it implements.**

**Every printing function MUST include a comment showing the exact EBNF rule it implements.**

Format:
```zig
// Grammar: <production-name> ::= <rule-definition>
fn parseFunctionName() !ReturnType { ... }

// Grammar: <production-name> ::= <rule-definition>
fn printFunctionName() !void { ... }
```

This makes it easy to:
- Understand what each function is parsing/printing
- Verify the implementation matches the grammar
- Review changes for grammar compliance
- Help others understand the parser/printer structure
- Ensure parser and printer stay in sync

### 3. Test-Driven Development

We follow a strict progression:

1. **Start Simple**: Create the simplest possible MLIR example
   ```mlir
   %0 = arith.constant 42 : i32
   ```

2. **Validate with mlir-opt**: Every test case MUST pass validation
   ```bash
   mlir-opt --verify-diagnostics examples/01_simple_constant.mlir
   ```

3. **Incrementally Add Complexity**: Only after simple cases work
   ```mlir
   %0 = arith.constant 42 : i32
   %1 = arith.constant 13 : i32
   %2 = arith.addi %0, %1 : i32
   ```

4. **Find Real-World Examples**: Search MLIR test suites online
   - LLVM project MLIR tests
   - Dialect-specific examples
   - Complex control flow examples

### 4. Roundtrip Testing

**The printer enables roundtrip validation**: `parse → print → parse → print`, with stable output.

Every parser feature MUST have a corresponding printer implementation. The test progression is:

1. **Parse** the source code to an AST
2. **Print** the AST back to text
3. **Parse** the printed text to a new AST
4. **Print** the new AST again
5. **Verify** that step 2 and step 4 produce identical output

This ensures:
- Parser correctly captures all information
- Printer correctly represents all AST nodes
- No information is lost in the roundtrip
- Output is stable and canonical

**Example roundtrip test:**
```zig
const source = "%0 = \"arith.constant\"() <{value = 42 : i32}> : () -> i32";

// Parse → Print → Parse → Print
var module1 = try parseSource(allocator, source);
const printed1 = try printModule(allocator, module1);
var module2 = try parseSource(allocator, printed1);
const printed2 = try printModule(allocator, module2);

// printed1 and printed2 must be identical
try testing.expectEqualStrings(printed1, printed2);
```

**When adding new features:**
1. Implement parser function with grammar comment
2. Add corresponding printer function with same grammar comment
3. Create roundtrip test that exercises the feature
4. Validate with `mlir-opt` if possible

### 5. Grammar Reference

The complete MLIR grammar is in `grammar.ebnf`. Key sections:

- **Lines 5-15**: Lexical tokens (identifiers, literals, operators)
- **Line 19**: Top-level structure (operations, aliases)
- **Lines 23-33**: Identifiers and value references
- **Lines 37-50**: Operations (generic and custom)
- **Lines 54-62**: Blocks and labels
- **Lines 66-67**: Regions
- **Lines 71-105**: Type system
- **Lines 108-129**: Attributes

## Implementation Strategy

### Phase 1: Foundation
- **Lexer** (`src/lexer.zig`): Tokenize MLIR source
  - Each token type maps to grammar lines 5-15
  - Comment each token variant with grammar reference
- **AST** (`src/ast.zig`): Node types for all grammar productions
  - Each struct tagged with its grammar rule
- **Printer** (`src/printer.zig`): Convert AST back to MLIR text
  - Each print function mirrors a parse function
  - Grammar comments match parser
- **Simple Examples**: Validate basics work

### Phase 2: Core Parser & Printer
- **Type Parsing & Printing**: Builtin types (i32, f64, index), function types, dialect types
- **Operation Parsing & Printing**: Generic operations with results, operands, attributes
- **Test**: Arithmetic operations, constants
- **Roundtrip Test**: Ensure parse→print→parse stability

### Phase 3: Control Flow
- **Block Parsing & Printing**: Block labels, arguments, successors
- **Region Parsing & Printing**: Nested regions with entry blocks
- **Test**: Basic blocks with branching
- **Roundtrip Test**: Complex control flow

### Phase 4: Advanced Features
- **Attributes Parsing & Printing**: Dictionary attributes, dialect attributes
- **Aliases Parsing & Printing**: Type aliases and attribute aliases
- **Test**: Complex dialect-specific code
- **Roundtrip Test**: All attribute and alias types

### Phase 5: Real-World Validation
- Find complex MLIR examples online
- Test tensor operations, memref operations
- Test LLVM dialect IR
- Ensure all pass `mlir-opt` validation
- **Roundtrip Test**: Real-world MLIR code

## Project Structure

```
mlir-parser/
├── grammar.ebnf              # Official MLIR grammar (source of truth)
├── CLAUDE.md                 # This file
├── src/
│   ├── lexer.zig            # Tokenization (grammar lines 5-15, 23-33)
│   ├── ast.zig              # AST node definitions
│   ├── parser.zig           # Recursive descent parser (main logic)
│   ├── printer.zig          # AST to MLIR text printer (roundtrip support)
│   ├── root.zig             # Public API exports
│   └── main.zig             # CLI tool
├── examples/
│   ├── 01_simple_constant.mlir
│   ├── 02_simple_add.mlir
│   ├── 03_basic_block.mlir
│   ├── 04_region.mlir
│   └── ... (progressively more complex)
└── test/
    ├── basic_test.zig
    ├── integration_test.zig
    ├── roundtrip_test.zig   # Roundtrip validation tests
    └── ...
```

## Development Workflow

### For New Features

1. **Identify the grammar rule** in `grammar.ebnf`
2. **Write the simplest test case** that uses that rule
3. **Validate with mlir-opt** to ensure correctness
4. **Implement the parser function** with grammar comment
5. **Test and iterate** until the parser handles the case
6. **Add more complex examples** and repeat

### For Bug Fixes

1. **Create a minimal MLIR example** that triggers the bug
2. **Validate it with mlir-opt** to confirm it's valid MLIR
3. **Identify which grammar rule** is being misparsed
4. **Fix the parser function** for that rule
5. **Verify all existing tests** still pass

### Code Review Checklist

- [ ] Every new parsing function has a grammar comment
- [ ] Every new printing function has a grammar comment
- [ ] Grammar comments match the actual EBNF rule
- [ ] Implementation follows the grammar structure
- [ ] Parser and printer functions are in sync (same grammar rules)
- [ ] Test cases included and validated with mlir-opt
- [ ] Roundtrip test added for new features
- [ ] Error messages are clear and reference grammar when possible

## Common Patterns

### Parser Patterns

#### Optional Elements (grammar: `expr?`)
```zig
// Grammar: trailing-location ::= `loc` `(` location `)`?
fn parseTrailingLocation() !?Location {
    if (!self.match(.kw_loc)) return null;
    self.expect(.lparen);
    const loc = try self.parseLocation();
    self.expect(.rparen);
    return loc;
}
```

#### Repetition (grammar: `expr*` or `expr+`)
```zig
// Grammar: toplevel ::= (operation | attribute-alias-def | type-alias-def)*
fn parseToplevel() !Module {
    var operations = std.ArrayList(Operation).init(self.allocator);
    while (!self.isAtEnd()) {
        try operations.append(try self.parseOperation());
    }
    return Module{ .operations = operations.items };
}
```

#### Alternation (grammar: `expr0 | expr1 | expr2`)
```zig
// Grammar: type ::= type-alias | dialect-type | builtin-type
fn parseType() !Type {
    return switch (self.peek()) {
        .exclamation => try self.parseDialectType(),
        .identifier => try self.parseBuiltinType(),
        else => error.ExpectedType,
    };
}
```

### Printer Patterns

#### Optional Elements (grammar: `expr?`)
```zig
// Grammar: trailing-location ::= `loc` `(` location `)`?
fn printTrailingLocation() !void {
    if (location) |loc| {
        try self.writer.writeAll(" loc(");
        try self.writer.writeAll(loc.source);
        try self.writer.writeByte(')');
    }
}
```

#### Repetition (grammar: `expr*` or `expr+`)
```zig
// Grammar: value-use-list ::= value-use (`,` value-use)*
fn printValueUseList() !void {
    for (operands, 0..) |operand, i| {
        if (i > 0) try self.writer.writeAll(", ");
        try self.printValueUse(operand);
    }
}
```

#### Alternation (grammar: `expr0 | expr1 | expr2`)
```zig
// Grammar: type ::= type-alias | dialect-type | builtin-type
fn printType() !void {
    switch (typ) {
        .type_alias => |alias| try self.printTypeAlias(alias),
        .dialect => |dialect| try self.printDialectType(dialect),
        .builtin => |builtin| try self.printBuiltinType(builtin),
    }
}
```

### Formatting Guidelines for Printer

1. **Whitespace**: Use single spaces around operators (`=`, `:`, `->`)
2. **Indentation**: Use 2 spaces per level for regions and blocks
3. **Newlines**:
   - Between top-level operations
   - For each operation in a region/block
   - After block labels
4. **Parentheses**: Match MLIR conventions:
   - Empty operand lists: `()`
   - Single types in function signatures: `i32 -> i32`
   - Multiple types: `(i32, i32) -> (i32, i32)`
5. **Quotes**: Always quote operation names in generic format: `"arith.constant"`

## Resources

- **MLIR Language Reference**: https://mlir.llvm.org/docs/LangRef/
- **MLIR Builtin Dialect**: https://mlir.llvm.org/docs/Dialects/Builtin/
- **Grammar File**: `grammar.ebnf` (in this repository)
- **LLVM MLIR Tests**: https://github.com/llvm/llvm-project/tree/main/mlir/test

## Getting Started

1. Read `grammar.ebnf` to understand the structure
2. Look at the existing code in `src/root.zig` for the API structure
3. Start with the lexer - tokenizing is the foundation
4. Create simple examples in `examples/` and validate with `mlir-opt`
5. Implement parser functions one grammar rule at a time
6. Implement corresponding printer functions
7. Create roundtrip tests to validate parser and printer work together
8. Always include grammar comments!

## Using the Printer

### Basic Usage

```zig
const std = @import("std");
const mlir = @import("mlir-parser");

// Parse MLIR source
var module = try mlir.parse(allocator, source);
defer module.deinit();

// Print to string
const output = try mlir.print(allocator, module);
defer allocator.free(output);

// Print to writer
const stdout = std.io.getStdOut().writer().any();
try mlir.format(module, stdout);
```

### Roundtrip Testing Workflow

When developing new parser features, always follow this workflow:

1. **Write the parser function** with grammar comment
   ```zig
   // Grammar: tensor-type ::= `tensor` `<` dimension-list type `>`
   fn parseTensorType() !TensorType { ... }
   ```

2. **Write the printer function** with matching grammar comment
   ```zig
   // Grammar: tensor-type ::= `tensor` `<` dimension-list type `>`
   fn printTensorType() !void { ... }
   ```

3. **Create a roundtrip test**
   ```zig
   test "roundtrip - tensor type" {
       const source = "%0 = \"test.op\"() : () -> tensor<4x8xf32>";
       try testRoundtrip(testing.allocator, source);
   }
   ```

4. **Validate with mlir-opt** (if applicable)
   ```bash
   echo '%0 = "test.op"() : () -> tensor<4x8xf32>' | mlir-opt --verify-diagnostics
   ```

5. **Debug roundtrip failures**
   - Print the first output and inspect for formatting issues
   - Check for missing grammar elements
   - Verify AST nodes are being populated correctly

### Debugging Tips

**If roundtrip test fails:**

1. Print both versions to see the difference:
   ```zig
   std.debug.print("First print:  {s}\n", .{printed1});
   std.debug.print("Second print: {s}\n", .{printed2});
   ```

2. Check that parser captures all information from source

3. Check that printer outputs all AST node information

4. Look for whitespace/formatting differences

5. Ensure grammar comments match between parser and printer

## Why This Approach?

**Clarity**: Anyone can look at a parser or printer function and immediately see what grammar rule it implements.

**Correctness**: By following the grammar exactly, we ensure the parser handles all valid MLIR. Roundtrip testing validates that no information is lost.

**Maintainability**: When the MLIR spec changes, we can easily identify which parser/printer functions need updates.

**Collaboration**: New contributors can understand the code by reading it alongside the grammar.

**Testing**: Simple → complex progression ensures we build on a solid foundation. Roundtrip tests catch bugs early.

**Reliability**: The printer enables validation that parsing is complete and correct by ensuring stable roundtrips.

---

Remember:
- **Every parser function needs a grammar comment. No exceptions.**
- **Every printer function needs a matching grammar comment.**
- **Every new feature needs a roundtrip test.**

This is what makes the codebase maintainable, understandable, and correct.
