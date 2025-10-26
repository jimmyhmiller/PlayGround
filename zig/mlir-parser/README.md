# MLIR Parser for Zig

A grammar-driven recursive descent parser for MLIR (Multi-Level Intermediate Representation) written in Zig.

## Project Status

**21/22 tests passing** ✅

This is a work-in-progress implementation that strictly follows the MLIR grammar specification. Each parser function is annotated with its corresponding grammar rule for clarity and maintainability.

## What Works

### ✅ Lexer (`src/lexer.zig`)
- **Complete tokenization** of MLIR source
- All token types from grammar lines 5-15, 23-33
- Handles:
  - Identifiers: `bare-id`, `value-id` (`%0`), `symbol-ref-id` (`@foo`), `caret-id` (`^bb0`)
  - Literals: integers, floats, strings, hexadecimal
  - Type/attribute aliases: `!alias`, `#alias`
  - Punctuation: `()`, `{}`, `[]`, `<>`, `,`, `:`, `=`, `->`, `::`
  - Keywords: `loc`, `func`, `return`, `module`
  - Comments (line comments with `//`)

**All lexer tests pass!**

### ✅ AST (`src/ast.zig`)
- Complete AST node definitions for all grammar productions
- Types: `Module`, `Operation`, `Block`, `Region`, `Type`, `Attribute`
- Proper memory management with `deinit()` methods
- Each AST node corresponds to a grammar rule

**All AST tests pass!**

### ✅ Type Parsing (`src/parser.zig`)
Fully implements type parsing according to grammar lines 71-105:

- **Integer types**: `i32`, `i64`, `si8`, `ui16` (signless, signed, unsigned)
- **Float types**: `f16`, `f32`, `f64`, `f80`, `f128`, `bf16`, `tf32`
- **Index type**: `index`
- **Function types**: `(i32, f64) -> i32`
- **Dialect types**: `!llvm.ptr<i32>`, `!custom<...>`
- **Type aliases**: `!myalias`

**All type parsing tests pass!**

### ⚠️ Operation Parsing (Partial)
Basic structure is complete, but custom operations need refinement:

- ✅ **Op result lists**: `%0 = ...`, `%0, %1 = ...`, `%0:2 = ...`
- ✅ **Generic operations**: String-based generic ops with full syntax
- ✅ **Value uses**: `%0`, `%1#2` (with result numbers)
- ✅ **Dictionary attributes**: `{attr1 = value, attr2 = value}`
- ✅ **Attribute values**: integers, floats, strings, booleans, arrays
- ❌ **Custom operations**: `arith.constant`, `func.return` (1 failing test)

**Issue**: `parseCustomOperation` currently consumes too many tokens. The parser needs to know when a custom operation ends (likely at newline or when seeing another `value-id` that starts a new operation).

### 🔨 Not Yet Implemented
- Block parsing (`block-label`, `block-arg-list`)
- Region parsing (`region`, `entry-block`)
- Tensor/memref/vector/complex/tuple types
- Successor lists
- Location information (partially implemented)
- Full custom operation format handling

## Project Structure

```
mlir-parser/
├── grammar.ebnf              # Official MLIR grammar (source of truth)
├── CLAUDE.md                 # Developer guide with rules and patterns
├── README.md                 # This file
├── src/
│   ├── lexer.zig            # Tokenization (Grammar lines 5-15, 23-33)
│   ├── ast.zig              # AST node definitions
│   ├── parser.zig           # Recursive descent parser (main logic)
│   ├── root.zig             # Public API exports
│   └── main.zig             # CLI tool
├── examples/
│   ├── 01_simple_constant.mlir  # Validated with mlir-opt ✓
│   ├── 02_simple_add.mlir       # Validated with mlir-opt ✓
│   └── 03_basic_block.mlir      # Validated with mlir-opt ✓
└── test/
    ├── basic_test.zig
    └── integration_test.zig
```

## Grammar-Driven Approach

**Every parser function MUST have a grammar comment.** This is the core principle of this project.

Example:
```zig
// Grammar: type ::= type-alias | dialect-type | builtin-type
pub fn parseType(self: *Parser) ParseError!ast.Type {
    // Implementation follows the grammar rule exactly
    ...
}
```

See `CLAUDE.md` for complete development guidelines.

## Testing Strategy

1. **Simple examples first**: Start with `%0 = arith.constant 42 : i32`
2. **Validate with mlir-opt**: Every test case must pass `mlir-opt --verify-diagnostics`
3. **Incremental complexity**: Add operations, then blocks, then regions
4. **Real-world examples**: Test with actual MLIR from LLVM test suite

## Building and Testing

```bash
# Run all tests
zig build test

# Run just the parser tests
zig test src/parser.zig

# Run just the lexer tests
zig test src/lexer.zig

# Build the executable
zig build

# Run the CLI
zig build run
```

## Validation with mlir-opt

All example files can be validated:

```bash
mlir-opt --verify-diagnostics examples/01_simple_constant.mlir
mlir-opt --verify-diagnostics examples/02_simple_add.mlir
mlir-opt --verify-diagnostics examples/03_basic_block.mlir
```

## Current Test Results

```
Lexer:          7/7 tests passing ✅
AST:            2/2 tests passing ✅
Parser:        10/11 tests passing ⚠️ (1 failing: custom operation parsing)
Basic tests:    2/2 tests passing ✅
Integration:    2/2 tests passing ✅
Total:         21/22 tests passing (95.5%)
```

## Next Steps (In Priority Order)

1. **Fix custom operation parsing** (failing test in `src/parser.zig:757`)
   - Determine proper termination conditions for custom ops
   - Handle dialect-specific syntax properly
   - Likely need to stop at newline or when seeing `%` at start of line

2. **Implement block parsing** (Grammar lines 54-62)
   - `block-label ::= block-id block-arg-list? ':'`
   - Handle block arguments with types

3. **Implement region parsing** (Grammar lines 66-67)
   - Nested `{ }` with operations and blocks

4. **Add complex types**
   - Tensor: `tensor<17x4x13xf32>`
   - MemRef: `memref<10x20xf32>`
   - Vector: `vector<4xf32>`

5. **Find and test real MLIR examples**
   - Search LLVM project test suite
   - Test with actual dialects (LLVM, Arith, Func, etc.)

## Contributing

When adding new features:

1. ✅ **Write the test first** (with real MLIR validated by mlir-opt)
2. ✅ **Add grammar comment** to every parser function
3. ✅ **Follow the grammar exactly** - don't improvise
4. ✅ **Keep failing tests** - they show what needs work!
5. ✅ **Test incrementally** - simple before complex

## References

- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin/)
- Grammar file: `grammar.ebnf` (in this repository)
- Development guide: `CLAUDE.md` (in this repository)

---

**Key Philosophy**: This parser is built to be maintainable and correct by strictly following the grammar. When in doubt, consult `grammar.ebnf` and the MLIR Language Reference.
