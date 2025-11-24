# Lispier Tools

This document describes the command-line tools available for inspecting Lispier compilation stages.

## Available Tools

### show-reader

Shows the tokenization and reader output for source code.

**Usage:**
```bash
zig build show-reader -- '<source-code>'
```

**Examples:**
```bash
# Simple expression
zig build show-reader -- '(+ 1 2 3)'

# Nested lists and vectors
zig build show-reader -- '(foo [1 2 3] {:key "value"})'

# Namespaced symbols
zig build show-reader -- '(arith.addi 1 2)'
```

**Output:**
- **TOKENS**: Shows all tokens with their type, line, column, and lexeme
- **READER OUTPUT**: Shows the s-expression values produced by the reader

### show-ast

Shows the Abstract Syntax Tree (AST) produced by parsing source code.

**Usage:**
```bash
zig build show-ast -- '<source-code>'
```

**Examples:**
```bash
# Simple operation
zig build show-ast -- '(arith.addi 1 2)'

# Variable binding
zig build show-ast -- '(def x (+ 1 2))'

# Let expression with multiple bindings
zig build show-ast -- '(let [a 10 b 20] (+ a b))'

# Type annotation
zig build show-ast -- '(: (+ 1 2) i32)'

# Function type
zig build show-ast -- '(-> [i32 i32] [i32])'

# Region with block
zig build show-ast -- '(region (block (+ 1 2)))'
```

**Output:**
Shows the full AST structure including:
- Node types (Operation, Def, Let, TypeAnnotation, etc.)
- Operands and arguments
- Nested structures (regions, blocks)
- Type information
- Literal values

## Example Workflows

### Understanding how code is parsed

```bash
# Step 1: See tokens and s-expressions
zig build show-reader -- '(def result (arith.addi 10 20))'

# Step 2: See the AST structure
zig build show-ast -- '(def result (arith.addi 10 20))'
```

### Debugging parsing issues

```bash
# Check if tokenization is correct
zig build show-reader -- '(your problematic code here)'

# Check if AST matches expectations
zig build show-ast -- '(your problematic code here)'
```

### Learning the syntax

```bash
# Explore how different constructs are represented
zig build show-ast -- '(let [x 1 y 2] (+ x y))'
zig build show-ast -- '(: myvalue i32)'
zig build show-ast -- '(region (block ^entry (arith.constant 42)))'
```

## Supported Syntax

### Basic Values
- **Numbers**: `1`, `3.14`, `-42`
- **Strings**: `"hello world"`
- **Keywords**: `:key`, `:name`
- **Booleans**: `true`, `false`
- **Nil**: `nil`
- **Symbols**: `foo`, `bar`, `arith.addi`, `my-func`

### Collections
- **Lists**: `(1 2 3)`
- **Vectors**: `[1 2 3]`
- **Maps**: `{:key1 value1 :key2 value2}`

### Special Forms
- **def**: `(def [names...] value)` - Define bindings
- **let**: `(let [name value ...] body...)` - Local bindings
- **Type annotation**: `(: value type)` - Annotate with type
- **Function type**: `(-> [arg-types...] [return-types...])` - Function signature
- **region**: `(region blocks...)` - MLIR region
- **block**: `(block ^label (arg:type ...) operations...)` - MLIR block
- **require-dialect**: `(require-dialect name)` - Load MLIR dialect

### Operations
Operations follow the pattern: `(namespace.operation operands... {attrs...} regions...)`

Example: `(arith.addi 1 2)` becomes an operation node with namespace "arith" and name "addi"

## Known Limitations

1. **SSA Values**: The tokenizer doesn't currently support `%` for SSA value names (e.g., `%0`, `%result`)
2. **Complex Types**: Some complex MLIR types may not display fully
3. **Attributes**: Operation attributes are shown but may need better formatting
4. **Error Messages**: Errors show stack traces but could be more user-friendly

## Building from Source

Both tools are built automatically with:
```bash
zig build
```

The executables are placed in `zig-out/bin/`:
- `zig-out/bin/show-reader`
- `zig-out/bin/show-ast`

You can run them directly:
```bash
./zig-out/bin/show-reader '(+ 1 2)'
./zig-out/bin/show-ast '(def x 42)'
```

## Tips

1. **Quote your code**: Always quote the source code argument to prevent shell interpretation
   ```bash
   zig build show-ast -- '(+ 1 2)'  # Good
   zig build show-ast -- (+ 1 2)    # Bad - shell will interpret parens
   ```

2. **Complex examples**: Use single quotes to avoid escaping issues
   ```bash
   zig build show-reader -- '(def x "hello world")'
   ```

3. **Multi-line code**: Works with multi-line input
   ```bash
   zig build show-ast -- '(def x 1)
   (def y 2)
   (+ x y)'
   ```

## Future Enhancements

Planned improvements:
- Support for SSA value syntax (`%name`)
- Pretty-printed output with colors
- JSON output mode for programmatic use
- Source location tracking in error messages
- Interactive mode for exploring large ASTs
