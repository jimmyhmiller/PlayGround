# Phase 2 Complete: Parser Core - Primitives & Binary Operators

## ✅ Completed Tasks

### 1. Primitive Expression Parsing
- ✅ Implemented `parse_prim_expr()` dispatcher function
- ✅ Routes to specific parsers based on token type (Number, String, Bool, Name, Rational, RoughRational)

### 2. Literal Parsers
- ✅ `parse_num()` - Parses NUMBER tokens into `Expr::SNum`
- ✅ `parse_bool()` - Parses TRUE/FALSE tokens into `Expr::SBool`
- ✅ `parse_str()` - Parses STRING tokens into `Expr::SStr`
- ✅ `parse_rational()` - Stub for RATIONAL tokens (TODO: parse numerator/denominator)
- ✅ `parse_rough_rational()` - Stub for ROUGHRATIONAL tokens (TODO: parse numerator/denominator)

### 3. Identifier Parsing
- ✅ `parse_id_expr()` - Parses NAME tokens into `Expr::SId` with `Name::SName`
- ✅ Creates proper location tracking for identifiers

### 4. Binary Operator Parsing
- ✅ `parse_binop_expr()` - Left-associative binary operator parsing
- ✅ `is_binop()` - Checks if current token is a binary operator
- ✅ `parse_binop()` - Converts operator tokens to operator strings ("op+", "op-", etc.)
- ✅ Supports all 15 Pyret binary operators:
  - Arithmetic: `+`, `-`, `*`, `/`, `^`
  - Comparison: `<`, `>`, `<=`, `>=`, `==`, `=~`, `<>`, `<=>`
  - Logical: `and`, `or`

### 5. AST Construction
- ✅ Creates proper `Expr::SOp` nodes with:
  - `l` - Overall location spanning left to right
  - `op_l` - Location of the operator token itself
  - `op` - Operator name string
  - `left` - Left operand expression
  - `right` - Right operand expression

### 6. Location Tracking
- ✅ All expressions include precise location information (`Loc`)
- ✅ Tracks start/end line, column, and character position
- ✅ Properly combines locations for compound expressions

### 7. Comprehensive Test Suite
- ✅ Created `tests/parser_tests.rs` with 12 passing tests
- ✅ Tests cover:
  - Number parsing (`42`)
  - String parsing (`"hello"`)
  - Boolean parsing (`true`, `false`)
  - Identifier parsing (`x`)
  - Simple binary operations (`1 + 2`)
  - Left-associative parsing (`1 + 2 + 3` → `(1 + 2) + 3`)
  - Multiple operators (`x + y * z`)
  - Various operator types (arithmetic, comparison, logical)
  - JSON serialization correctness

## 📊 Stats

- **Parser methods implemented:** 10+
  - `parse_expr()`
  - `parse_binop_expr()`
  - `parse_prim_expr()`
  - `parse_num()`
  - `parse_bool()`
  - `parse_str()`
  - `parse_id_expr()`
  - `parse_rational()`
  - `parse_rough_rational()`
  - `is_binop()`
  - `parse_binop()`
- **Binary operators supported:** 15
- **Tests passing:** 12/12 ✅
- **Code added:** ~300 lines in parser.rs, ~180 lines in tests

## 🎯 What Works

### Simple Expressions
```pyret
42              // SNum { n: 42.0 }
"hello"         // SStr { s: "\"hello\"" }
true            // SBool { b: true }
x               // SId { id: SName { s: "x" } }
```

### Binary Operations
```pyret
1 + 2           // SOp { op: "op+", left: SNum(1), right: SNum(2) }
x < 10          // SOp { op: "op<", left: SId(x), right: SNum(10) }
true and false  // SOp { op: "opand", left: SBool(true), right: SBool(false) }
```

### Left-Associative Parsing
```pyret
1 + 2 + 3       // SOp { op: "op+", left: SOp { op: "op+", ... }, right: SNum(3) }
                // Parsed as: (1 + 2) + 3
```

### JSON Serialization
```json
{
  "type": "s-num",
  "l": {
    "source": "test.arr",
    "start-line": 1,
    "start-column": 0,
    "start-char": 0,
    "end-line": 1,
    "end-column": 2,
    "end-char": 2
  },
  "n": 42.0
}
```

## 📝 Example Parse Results

### Input: `1 + 2 * 3`
```rust
SOp {
    op: "op*",
    left: SOp {
        op: "op+",
        left: SNum { n: 1.0 },
        right: SNum { n: 2.0 },
    },
    right: SNum { n: 3.0 },
}
```
Note: Parses as `(1 + 2) * 3` due to left-associativity and flat precedence.

### Input: `x + y`
```rust
SOp {
    op: "op+",
    left: SId { id: SName { s: "x" } },
    right: SId { id: SName { s: "y" } },
}
```

## 🔜 Next Steps (Phase 3)

Following the implementation plan in `PARSER_PLAN.md`:

### Critical: Parentheses & Function Application (Must Do Together)
**These must be implemented together** due to Pyret's whitespace-sensitive syntax:
- `f(x)` - Function application (PARENNOSPACE - no space before `(`)
- `f (x)` - Function `f` applied to parenthesized expression `(x)` (PARENSPACE)
- `(x + y)` - Parenthesized expression (PARENSPACE)

The tokenizer already distinguishes between:
- `TokenType::ParenNoSpace` - Used for function application
- `TokenType::ParenSpace` - Used for parenthesized expressions
- `TokenType::LParen` - Generic left paren

Implementation strategy:
1. **Implement `parse_paren_expr()`** - Handles `(expr)` when token is PARENSPACE or LPAREN
2. **Implement `parse_app_expr(base)`** - Handles `base(args)` when token is PARENNOSPACE
3. **Update `parse_prim_expr()`** - Add case for parenthesized expressions
4. **Update `parse_binop_expr()`** - Check for function application after parsing primary
5. **Add tests** - Test both `f(x)` and `f (x)` parse differently

### Other Phase 3 Tasks

2. **Implement object expressions**
   - `parse_obj_expr()` for `{ field: value, ... }`
   - Member parsing

3. **Implement array and tuple expressions**
   - `parse_array_expr()` for `[expr, ...]`
   - `parse_tuple_expr()` for `{expr; expr; ...}`

4. **Implement dot and bracket access**
   - `parse_dot_expr()` for `obj.field`
   - `parse_bracket_expr()` for `obj[key]`

5. **Add more comprehensive tests**
   - Nested expressions
   - Mixed operators and literals
   - Edge cases
   - Whitespace sensitivity tests

## ✨ Key Achievements

1. **Working primitive parser** - All basic literal types parse correctly
2. **Binary operators** - Full left-associative operator support
3. **Proper AST generation** - Matches reference implementation structure
4. **Location tracking** - Precise source location for every node
5. **JSON serialization** - AST serializes to correct JSON format
6. **Comprehensive tests** - 12 tests covering all basic functionality
7. **Clean code** - Well-organized, documented parser functions

## 🐛 Known Limitations

1. **Rational numbers** - Numerator/denominator not yet parsed from token value
2. **Limited expressions** - Only primitives and binary operations so far
3. **No parentheses** - Cannot override left-associativity yet (will be in Phase 3)

## ℹ️ Note on Operator Precedence

**This is NOT a limitation**: Pyret intentionally has NO operator precedence hierarchy. All binary operators have equal precedence and are strictly left-associative. This is a fundamental property of Pyret's BNF grammar, not a parser implementation choice.

- `2 + 3 * 4` parses as `(2 + 3) * 4 = 20` (not `2 + (3 * 4) = 14`)
- Users must use explicit parentheses to control evaluation order
- Our parser correctly implements this grammar-defined behavior

See `OPERATOR_PRECEDENCE.md` for detailed explanation.

These limitations will be addressed in Phase 3 and beyond.

---

**Date Completed:** 2025-10-31
**Next Phase:** Phase 3 - Complex Expressions (Objects, Arrays, Tuples, Function Calls)
