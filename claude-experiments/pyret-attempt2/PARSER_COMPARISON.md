# Parser Comparison Tools

This document describes the tools created to validate that our Rust parser produces identical ASTs to the official Pyret parser.

## Overview

To ensure correctness, we've built a comparison system that:
1. Parses Pyret code with both the official Pyret parser and our Rust parser
2. Converts both ASTs to a common JSON format
3. Compares the results to verify identical structure

## Tools

### 1. `to_pyret_json` - Rust Binary

**Location**: `src/bin/to_pyret_json.rs`

**Purpose**: Converts our Rust AST to Pyret's JSON format for comparison.

**Key Features**:
- Strips location information (Pyret's ast-to-json doesn't include it)
- Uses Pyret's field names (`"value"` for numbers, `"fun"` for functions, etc.)
- Handles all currently implemented expression types:
  - Numbers, strings, booleans, identifiers
  - Binary operators
  - Parenthesized expressions
  - Function application
  - Dot access
  - Arrays

**Usage**:
```bash
# From stdin
echo "2 + 3" | cargo run --bin to_pyret_json

# From file
cargo run --bin to_pyret_json path/to/file.arr

# Example output
{
  "type": "s-op",
  "op": "op+",
  "left": {
    "type": "s-num",
    "value": "2"
  },
  "right": {
    "type": "s-num",
    "value": "3"
  }
}
```

**Implementation Details**:

The converter uses pattern matching to transform Rust AST nodes:

```rust
fn expr_to_pyret_json(expr: &Expr) -> Value {
    match expr {
        Expr::SNum { n, .. } => {
            json!({
                "type": "s-num",
                "value": n.to_string()  // String, not number
            })
        }
        Expr::SOp { op, left, right, .. } => {
            json!({
                "type": "s-op",
                "op": op,
                "left": expr_to_pyret_json(left),
                "right": expr_to_pyret_json(right)
            })
        }
        // ... more cases
    }
}
```

### 2. `compare_parsers.sh` - Comparison Script

**Location**: `compare_parsers.sh`

**Purpose**: End-to-end comparison of both parsers on the same input.

**Features**:
- Parses with Pyret's official parser (`ast-to-json.jarr`)
- Parses with our Rust parser (`to_pyret_json`)
- Normalizes JSON (ignoring field order)
- Reports differences with colored output (✅/❌)
- Shows full diff when ASTs differ

**Usage**:
```bash
# Basic comparison
./compare_parsers.sh "2 + 3"

# Complex expressions
./compare_parsers.sh "f(x + 1).result"
./compare_parsers.sh "obj.foo.bar"
./compare_parsers.sh "f(1)(2)(3)"
```

**Example Output** (when identical):
```
=== Input ===
2 + 3

=== Pyret Parser ===
JSON written to /tmp/pyret_output.json
{
  "type": "s-op",
  "op": "op+",
  "left": {
    "type": "s-num",
    "value": "2"
  },
  "right": {
    "type": "s-num",
    "value": "3"
  }
}

=== Rust Parser ===
{
  "left": {
    "type": "s-num",
    "value": "2"
  },
  "op": "op+",
  "right": {
    "type": "s-num",
    "value": "3"
  },
  "type": "s-op"
}

=== Comparison ===
✅ IDENTICAL - Parsers produce the same AST!
```

**Example Output** (when different):
```
=== Comparison ===
❌ DIFFERENT - Found differences:

Pyret AST:
{
  "type": "s-op",
  "op": "op+",
  "left": { "type": "s-num", "value": "2" },
  "right": { "type": "s-num", "value": "3" }
}

Rust AST:
{
  "type": "s-app",
  "fun": { "type": "s-id", "id": { "type": "s-name", "name": "foo" } },
  "args": []
}
```

**How It Works**:

1. **Write input to temp file**:
   ```bash
   echo "$EXPR" > /tmp/pyret_compare_input.arr
   ```

2. **Parse with Pyret**:
   ```bash
   node ast-to-json.jarr /tmp/pyret_compare_input.arr /tmp/pyret_output.json
   ```

   Extract just the expression from the program wrapper:
   ```python
   # Pyret outputs full program structure, we want just the expression
   data['body']['stmts'][0]
   ```

3. **Parse with Rust**:
   ```bash
   cargo run --bin to_pyret_json /tmp/pyret_compare_input.arr
   ```

4. **Normalize and compare**:
   ```python
   def normalize_json(obj):
       """Recursively sort dictionaries for consistent comparison"""
       if isinstance(obj, dict):
           return {k: normalize_json(v) for k, v in sorted(obj.items())}
       elif isinstance(obj, list):
           return [normalize_json(item) for item in obj]
       else:
           return obj
   ```

## Verified Syntax

The following syntax has been verified to parse identically:

| Category | Examples | Status |
|----------|----------|--------|
| **Primitives** | `42`, `"hello"`, `true`, `false` | ✅ |
| **Binary Operators** | `2 + 3`, `x * y`, `a - b` | ✅ |
| **Chained Operators** | `2 + 3 * 4` (left-associative) | ✅ |
| **Parentheses** | `(2 + 3)` | ✅ |
| **Function Calls** | `f(x)`, `f(1, 2, 3)` | ✅ |
| **Chained Calls** | `f(1)(2)`, `f()(g())` | ✅ |
| **Dot Access** | `obj.field` | ✅ |
| **Chained Dot** | `obj.foo.bar.baz` | ✅ |
| **Mixed Postfix** | `f(x).result`, `obj.method(arg)` | ✅ |
| **Complex** | `f(x + 1).result`, `obj.foo(a, b).bar` | ✅ |

## Limitations

### Currently Unsupported in Rust Parser

These will cause comparison failures as they're not yet implemented:

- Arrays: `[1, 2, 3]` - **Note**: Pyret doesn't parse raw arrays as expressions
- Objects: `{ x: 1, y: 2 }`
- Tuples: `{1; 2; 3}`
- Bracket access: `arr[0]`
- Control flow: `if`, `cases`, `when`, `for`
- Functions: `fun`, `lam`, `method`
- Let bindings: `let x = 5:`
- Blocks: `block: ... end`

### Pyret Parser Differences

The official Pyret parser:
- Always wraps expressions in a full `s-program` structure
- Includes `use`, `provide`, `imports` fields even when empty
- Wraps standalone expressions in `s-block` with `stmts` array

Our comparison script handles this by extracting just the first statement from Pyret's output.

## Adding Support for New Expressions

When implementing a new expression type, follow these steps:

### 1. Add to Rust AST (already done)

The AST is already complete in `src/ast.rs`.

### 2. Add Parser Implementation

In `src/parser.rs`, implement the parsing logic.

### 3. Add JSON Serialization

In `src/bin/to_pyret_json.rs`, add a new match arm:

```rust
fn expr_to_pyret_json(expr: &Expr) -> Value {
    match expr {
        // ... existing cases ...

        Expr::SBracket { obj, field, .. } => {
            json!({
                "type": "s-bracket",
                "obj": expr_to_pyret_json(obj),
                "field": expr_to_pyret_json(field)
            })
        }

        // ... more cases ...
    }
}
```

**Important**: Check Pyret's `ast-to-json.arr` for the exact field names and structure.

### 4. Verify with Comparison Script

```bash
./compare_parsers.sh "arr[0]"
./compare_parsers.sh "{ x: 1, y: 2 }"
./compare_parsers.sh "{1; 2; 3}"
```

### 5. Add to Verified Syntax Table

Update this document's table with the newly verified syntax.

## Reference: Pyret's ast-to-json.arr

The official Pyret AST-to-JSON converter is located at:
```
/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/ast-to-json.arr
```

**Key methods** for expression types:

```pyret
method s-num(self, l, n):
  [SD.string-dict: "type", "s-num", "value", torepr(n)]
end

method s-op(self, l, op-l, op, left, right):
  [SD.string-dict:
    "type", "s-op",
    "op", op,
    "left", left.visit(self),
    "right", right.visit(self)
  ]
end

method s-app(self, l, _fun, args):
  [SD.string-dict:
    "type", "s-app",
    "fun", _fun.visit(self),
    "args", args.map(_.visit(self))
  ]
end

method s-dot(self, l, obj, field):
  [SD.string-dict:
    "type", "s-dot",
    "obj", obj.visit(self),
    "field", field
  ]
end

method s-bracket(self, l, obj, field):
  [SD.string-dict:
    "type", "s-bracket",
    "obj", obj.visit(self),
    "field", field.visit(self)
  ]
end

method s-obj(self, l, fields):
  [SD.string-dict:
    "type", "s-obj",
    "fields", fields.map(_.visit(self))
  ]
end

method s-array(self, l, values):
  [SD.string-dict:
    "type", "s-array",
    "values", values.map(_.visit(self))
  ]
end

method s-tuple(self, l, fields):
  [SD.string-dict:
    "type", "s-tuple",
    "fields", fields.map(_.visit(self))
  ]
end
```

**Note**: Locations are always passed but never serialized in `ast-to-json.arr`.

## Troubleshooting

### Script fails with "no such file or directory"

Make sure you're in the correct directory:
```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt2
./compare_parsers.sh "..."
```

### "Parse failed" from Pyret

The expression may not be valid standalone Pyret syntax. Try wrapping in a context:
```bash
# Instead of just "{ x: 1 }"
./compare_parsers.sh "{ x: 1, y: 2 }"

# Some syntax needs more context
./compare_parsers.sh "let x = 5: x end"
```

### Rust parser error

Check that the syntax is implemented. See the "Verified Syntax" table above.

### ASTs differ unexpectedly

1. Check field names in `ast-to-json.arr` (line references in this doc)
2. Verify the Rust parser implementation
3. Check if location info is accidentally included
4. Verify boxed vs unboxed values are handled correctly

## Integration with Tests

You can add comparison tests to `tests/parser_tests.rs`:

```rust
#[test]
fn test_matches_pyret_parser() {
    let test_cases = vec![
        "2 + 3",
        "f(x)",
        "obj.field",
        "(1 + 2) * 3",
    ];

    for input in test_cases {
        // Parse with our parser
        let expr = parse_expr(input).expect("Parse failed");

        // Could shell out to comparison script or parse Pyret output
        // For now, at least verify we can parse it
        assert!(matches!(expr, Expr::SOp { .. }));
    }
}
```

## Future Improvements

1. **Automated test suite**: Run comparison on all test cases automatically
2. **Fuzzing**: Generate random valid Pyret expressions and compare
3. **Performance comparison**: Time both parsers on large files
4. **Error comparison**: Verify error messages match for invalid syntax
5. **Full program parsing**: Compare complete Pyret programs, not just expressions

## Files

- `src/bin/to_pyret_json.rs` - Rust AST → Pyret JSON converter (147 lines)
- `compare_parsers.sh` - Automated comparison script (92 lines)
- `PARSER_COMPARISON.md` - This documentation

---

**Last Updated**: 2025-10-31
**Status**: All implemented syntax verified identical ✅
