# Phase 3 Partial Complete: Parenthesized Expressions & Function Application

## ✅ Completed Tasks (2025-10-31)

### Critical Whitespace-Sensitive Features Implemented

This phase tackled one of the most complex aspects of Pyret's syntax: **whitespace-sensitive parentheses** that distinguish between function application and parenthesized expressions.

## 🎯 What Was Implemented

### 1. Parenthesized Expressions (`parse_paren_expr()`)
**Location:** `src/parser.rs:462-479`

- Parses expressions wrapped in parentheses: `(expr)`
- Handles both `ParenSpace` and `LParen` tokens
- Returns `Expr::SParen { l, expr }` wrapping the inner expression
- Supports nested parentheses: `((x))`
- Allows complex expressions: `(1 + 2 * 3)`

**Example:**
```pyret
(42)        → SParen { expr: SNum { n: 42 } }
(1 + 2)     → SParen { expr: SOp { op: "op+", ... } }
```

### 2. Function Application (`parse_app_expr()`)
**Location:** `src/parser.rs:481-520`

- Parses function calls with explicit parentheses: `f(x, y, z)`
- Expects `ParenNoSpace` token (no whitespace before `(`)
- Handles comma-separated argument lists
- Supports zero arguments: `f()`
- Returns `Expr::SApp { l, _fun, args }`

**Example:**
```pyret
f(x)           → SApp { _fun: SId(f), args: [SId(x)] }
f(x, y, z)     → SApp { _fun: SId(f), args: [SId(x), SId(y), SId(z)] }
g(1 + 2, 3)    → SApp { _fun: SId(g), args: [SOp(...), SNum(3)] }
```

### 3. Chained Function Calls
**Location:** `src/parser.rs:208-210`

- Supports curried function calls: `f(x)(y)(z)`
- Left-associative application: `(f(x))(y)`
- Works with any expression as function: `(get_fn())(arg)`

**Example:**
```pyret
f(x)(y)     → SApp {
                _fun: SApp { _fun: SId(f), args: [SId(x)] },
                args: [SId(y)]
              }
```

### 4. Juxtaposition Application (Whitespace-Sensitive)
**Location:** `src/parser.rs:212-244, 259-287`

This is the most subtle feature: **adjacent expressions with whitespace form function application**.

- `f (x)` → Function `f` applied to parenthesized expression `(x)`
- Only applies when left expression is an identifier or function application
- Creates `SApp` with single `SParen` argument

**Example:**
```pyret
f (x)       → SApp { _fun: SId(f), args: [SParen(SId(x))] }
            // Note: argument is SParen wrapping x, not x directly
```

### 5. Updated `parse_prim_expr()`
**Location:** `src/parser.rs:346-357`

Added cases for parenthesized expressions:
```rust
TokenType::ParenSpace | TokenType::LParen => self.parse_paren_expr(),
```

### 6. Enhanced `parse_binop_expr()`
**Location:** `src/parser.rs:199-344`

Major enhancement to handle multiple postfix operators:
1. Parse primary expression
2. Check for `ParenNoSpace` → direct function call
3. Check for `ParenSpace` + identifier → juxtaposition application
4. Parse binary operators with same checks on right side

### 7. Critical Tokenizer Fix
**Location:** `src/tokenizer.rs:817-821`

**Bug Fixed:** Name tokens weren't setting `paren_is_for_exp = false`

**Problem:**
- After parsing identifier `f`, the flag stayed `true` (from initialization)
- Next `(` would incorrectly tokenize as `ParenSpace` instead of `ParenNoSpace`
- Result: `f(x)` parsed as `f (x)` (wrong!)

**Solution:**
```rust
_ => {
    // For Name tokens, set paren_is_for_exp to false so that f(x) gets ParenNoSpace
    self.paren_is_for_exp = false;
    TokenType::Name
}
```

## 📊 Whitespace Sensitivity Matrix

This is the core of Pyret's unique syntax:

| Syntax | Tokenization | AST | Semantics |
|--------|--------------|-----|-----------|
| `f(x)` | `Name ParenNoSpace Name RParen` | `SApp { _fun: f, args: [x] }` | Direct function call |
| `f (x)` | `Name ParenSpace Name RParen` | `SApp { _fun: f, args: [SParen(x)] }` | Function applied to paren expr |
| `(x)` | `ParenSpace Name RParen` | `SParen { expr: x }` | Parenthesized expression |
| `(x + y)` | `ParenSpace ... RParen` | `SParen { expr: SOp(...) }` | Grouped expression |

**Key Distinction:**
- **No space** before `(` → Function application with direct arguments
- **Space** before `(` → Parenthesized expression (may be used as argument via juxtaposition)

## 🧪 Comprehensive Test Suite

### Tests Added: 12 new tests
**Location:** `tests/parser_tests.rs:203-512`

#### Parenthesized Expression Tests (3 tests)
1. ✅ `test_parse_simple_paren_expr` - Basic `(42)`
2. ✅ `test_parse_paren_with_binop` - Complex `(1 + 2)`
3. ✅ `test_parse_nested_parens` - Nested `((5))`

#### Function Application Tests (6 tests)
4. ✅ `test_parse_simple_function_call` - Basic `f(x)`
5. ✅ `test_parse_function_call_multiple_args` - Multi-arg `f(x, y, z)`
6. ✅ `test_parse_function_call_no_args` - Zero args `f()`
7. ✅ `test_parse_chained_function_calls` - Currying `f(x)(y)`
8. ✅ `test_parse_function_call_with_expr_args` - Complex args `f(1 + 2, 3 * 4)`

#### Whitespace Distinction Tests (2 tests)
9. ✅ `test_whitespace_paren_space` - Verifies `f (x)` → `SApp { args: [SParen(x)] }`
10. ✅ `test_whitespace_no_space` - Verifies `f(x)` → `SApp { args: [x] }`

#### Mixed Expression Tests (2 tests)
11. ✅ `test_parse_function_in_binop` - Operations `f(x) + g(y)`
12. ✅ `test_parse_paren_changes_associativity` - Grouping `1 + (2 * 3)`

### Test Results
```
test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out

Total project tests: 35 passed
```

## 📈 Code Changes Summary

### Files Modified
1. **src/parser.rs** (+140 lines)
   - Added `parse_paren_expr()` method
   - Added `parse_app_expr()` method
   - Updated `parse_prim_expr()` to handle parentheses
   - Enhanced `parse_binop_expr()` for function application and juxtaposition
   - Added cases to location extraction matches

2. **src/tokenizer.rs** (+4 lines)
   - Fixed critical bug: Name tokens now set `paren_is_for_exp = false`

3. **tests/parser_tests.rs** (+320 lines)
   - Added 12 comprehensive tests
   - Added debug token printing helper

### Lines of Code
- Parser: 708 lines (was 568)
- Tests: 513 lines (was 202)
- Tokenizer fix: 4 lines changed

## 🔍 Technical Deep Dive

### The Juxtaposition Problem

Pyret allows **implicit function application** through juxtaposition:
```pyret
f (x)  // means: f applied to (x)
```

This is NOT just `f` followed by `(x)` as separate expressions. The parser must recognize this pattern and create a function application node.

**Implementation Strategy:**
1. After parsing primary expression, check for `ParenSpace`
2. Only apply juxtaposition if left side is identifier or function application
3. Parse the parenthesized expression as a single argument
4. Wrap in `SApp` node

**Why the restriction?**
Numbers, strings, etc. cannot be "functions" in Pyret:
```pyret
42 (x)   // Not valid - numbers aren't callable
"hi" (x) // Not valid - strings aren't callable
```

### Parser Flow Diagram

```
parse_expr()
  ↓
parse_binop_expr()
  ↓
parse_prim_expr() → returns base expression
  ↓
Check ParenNoSpace?
  ↓ YES → parse_app_expr(base) → chaining loop
  ↓ NO
Check ParenSpace AND (base is SId or SApp)?
  ↓ YES → create SApp with SParen argument
  ↓ NO
Check binop?
  ↓ YES → parse right side with same logic
  ↓ NO
return base
```

### Location Tracking

Every node includes precise location information:
```rust
Loc::new(
    file_name,
    start_line, start_col, start_char,
    end_line, end_col, end_char
)
```

Function applications span from the function name to the closing paren:
```pyret
f(x, y)
^     ^
start end
```

## 💡 Key Design Decisions

### 1. Why Two Loops in `parse_binop_expr()`?

We have separate loops for:
- **ParenNoSpace checking** (lines 208-210, 255-257)
- **ParenSpace + juxtaposition checking** (lines 212-244, 259-287)

**Reason:** These are different operations with different semantics:
- `ParenNoSpace` always means function application
- `ParenSpace` only means application when left side is callable

### 2. Why Box Arguments in `parse_comma_list`?

```rust
self.parse_comma_list(|p| p.parse_expr().map(Box::new))?
```

**Reason:** `SApp` stores `args: Vec<Box<Expr>>`, and we parse them directly as boxed values to avoid unnecessary allocations.

### 3. Why Check Token Type Before Parsing?

```rust
while self.matches(&TokenType::ParenNoSpace) {
    left = self.parse_app_expr(left)?;
}
```

**Reason:** We use `matches()` (peek without consuming) to avoid parsing when we shouldn't, then `expect()` inside `parse_app_expr()` to consume the token.

## 🐛 Bugs Fixed

### Tokenizer Bug: Incorrect ParenSpace After Names

**Symptom:** `f(x)` was being parsed as `f (x)` with wrong AST

**Root Cause:**
- Tokenizer initialized `paren_is_for_exp = true` (line 232)
- After tokenizing `Name`, flag wasn't reset to `false`
- Next `(` checked flag and used `ParenSpace` instead of `ParenNoSpace`

**Fix:** Set `paren_is_for_exp = false` when creating `Name` tokens

**Impact:** All function calls now parse correctly

## 📝 Examples

### Example 1: Simple Function Call
```pyret
Input:  f(x, y)
Tokens: Name, ParenNoSpace, Name, Comma, Name, RParen, Eof

AST:
SApp {
  l: Loc { ... },
  _fun: Box(SId { id: SName { s: "f" } }),
  args: [
    Box(SId { id: SName { s: "x" } }),
    Box(SId { id: SName { s: "y" } })
  ]
}
```

### Example 2: Juxtaposition Application
```pyret
Input:  f (x)
Tokens: Name, ParenSpace, Name, RParen, Eof

AST:
SApp {
  l: Loc { ... },
  _fun: Box(SId { id: SName { s: "f" } }),
  args: [
    Box(SParen {
      l: Loc { ... },
      expr: Box(SId { id: SName { s: "x" } })
    })
  ]
}
```

### Example 3: Chained Calls
```pyret
Input:  f(x)(y)
Tokens: Name, ParenNoSpace, Name, RParen, ParenNoSpace, Name, RParen, Eof

AST:
SApp {
  _fun: Box(SApp {
    _fun: Box(SId { s: "f" }),
    args: [Box(SId { s: "x" })]
  }),
  args: [Box(SId { s: "y" })]
}
```

### Example 4: Mixed with Operators
```pyret
Input:  f(x) + g(y)
Tokens: Name, ParenNoSpace, Name, RParen, Plus, Name, ParenNoSpace, Name, RParen, Eof

AST:
SOp {
  op: "op+",
  left: Box(SApp { _fun: f, args: [x] }),
  right: Box(SApp { _fun: g, args: [y] })
}
```

## 🔜 Next Steps (Continue Phase 3)

The following Phase 3 features are still TODO:

### Remaining Expression Types

1. **Object Expressions** (`parse_obj_expr()`)
   - Syntax: `{ field: value, method() -> expr: body end }`
   - AST: `Expr::SObj { l, fields }`
   - Priority: HIGH (very common in Pyret)

2. **Array Expressions** (`parse_array_expr()`)
   - Syntax: `[expr, expr, ...]`
   - AST: `Expr::SArray { l, values }`
   - Priority: HIGH

3. **Tuple Expressions** (`parse_tuple_expr()`)
   - Syntax: `{expr; expr; ...}`
   - AST: `Expr::STuple { l, fields }`
   - Priority: MEDIUM

4. **Dot Access** (`parse_dot_expr()`)
   - Syntax: `obj.field`
   - AST: `Expr::SDot { l, obj, field }`
   - Priority: HIGH

5. **Bracket Access** (`parse_bracket_expr()`)
   - Syntax: `obj[key]`
   - AST: `Expr::SBracket { l, obj, key }`
   - Priority: HIGH

6. **Extended Dot Access** (`parse_get_bang_expr()`)
   - Syntax: `obj!field`
   - AST: `Expr::SGetBang { l, obj, field }`
   - Priority: LOW

7. **Instantiation** (`parse_construct_expr()`)
   - Syntax: `constructor(args)`
   - Requires lookahead/context
   - Priority: MEDIUM

8. **Lambda Expressions** (start of Phase 5)
   - Syntax: `lam(args): body end`
   - AST: `Expr::SLam { ... }`
   - Priority: HIGH

### Testing Priorities

- Add tests for complex nested expressions
- Test edge cases with whitespace
- Test error handling for malformed expressions
- Add performance tests for deeply nested parens

## ✨ Key Achievements

1. ✅ **Whitespace-sensitive parsing** - Correctly distinguishes `f(x)` from `f (x)`
2. ✅ **Parenthesized expressions** - Full support including nested parens
3. ✅ **Function application** - Direct calls, chaining, zero args, multiple args
4. ✅ **Juxtaposition application** - Implicit function calls via adjacency
5. ✅ **Critical tokenizer bug fix** - Name tokens now produce correct paren types
6. ✅ **Comprehensive test coverage** - 12 new tests, 24 total passing
7. ✅ **Clean, documented code** - Well-structured with inline comments

## 📚 References

- **Pyret Grammar:** `/pyret-lang/src/js/base/pyret-grammar.bnf`
  - Lines for `paren-expr`: `LPAREN expr RPAREN | PARENSPACE expr RPAREN`
  - Lines for `app-expr`: `expr PARENNOSPACE (expr COMMA)* RPAREN`

- **Tokenizer:** `src/tokenizer.rs:1031-1044`
  - Paren tokenization logic with whitespace detection

- **Parser Plan:** `PARSER_PLAN.md`
  - Phase 3 section (lines 142-178 in PHASE2_COMPLETE.md)

---

**Date Completed:** 2025-10-31
**Status:** Phase 3 - Partially Complete (20% → 35%)
**Next Task:** Implement object expressions (`parse_obj_expr()`)
**Tests Passing:** 24/24 parser tests, 35/35 total tests
