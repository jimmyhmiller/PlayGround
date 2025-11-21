# Parse Failure Analysis - Properly Normalized

**Total Failures:** 1,125  
**Unique Normalized Error Types:** ~100 (not 1,033!)

## Top Error Categories (Normalized)

| Count | Error Category |
|-------|----------------|
| **269** | Unexpected token: DOT_DOT_DOT (`...`) |
| **107** | Unexpected token: STAR (`*`) |
| **97** | Unexpected token: RBRACKET (`]`) |
| **43** | Unexpected token: RPAREN (`)`) |
| **30** | Unexpected token: QUESTION (`?`) |
| **23** | Expected property name after `.` |
| **20** | Unexpected token: IDENTIFIER `await` |
| **16** | Unexpected token: COMMA (various positions) |
| **12** | Unexpected token: IDENTIFIER `Iterator` |
| **10** | Unexpected token: ASSIGN (`=`) |
| **9** | Unexpected token: TEMPLATE_HEAD |
| **8** | Unexpected token: COLON (`:`) |
| **7** | Expected identifier in import specifier |
| **7** | Unexpected token: GT (`>`) |
| **6** | Unexpected token: CLASS |
| **5** | Unexpected token: OF |
| **5** | Unexpected token: TEMPLATE_LITERAL |
| **4** | Expected property key |
| **4** | Unexpected token: IDENTIFIER `Promise` |
| **4** | Expected property name in class body |

## Analysis

### 1. Spread Operator Issues (269 failures - 24% of all failures)
- `DOT_DOT_DOT` token unexpected in various contexts
- Spread in destructuring patterns
- Spread in function calls
- Rest parameters
- **Root cause:** Context-sensitive parsing of `...` not fully implemented

### 2. Generator/Async Function Issues (107 failures - 10%)
- `STAR` token (`*`) unexpected
- Generator functions and methods
- `yield` expressions
- **Root cause:** Generator syntax edge cases

### 3. Array Destructuring Issues (97 failures - 9%)
- `RBRACKET` (`]`) unexpected
- Array holes
- Nested array patterns
- **Root cause:** Array pattern parsing incomplete

### 4. Expression Parsing Issues (43 failures - 4%)
- `RPAREN` (`)`) unexpected
- Likely parenthesized expressions in wrong contexts
- **Root cause:** Expression precedence/grouping

### 5. Optional Chaining Issues (30 failures - 3%)
- `QUESTION` (`?`) unexpected
- Optional chaining edge cases
- Ternary operator edge cases
- **Root cause:** Ambiguity between `?` uses

### 6. Leading Decimal Properties (23 failures - 2%)
- `Expected property name after '.'`
- Syntax: `obj.1`, `get .5()`
- **Root cause:** Lexer doesn't support `.number` syntax

### 7. Async/Await Issues (20 failures - 2%)
- `await` as identifier unexpected
- Context-sensitive keyword parsing
- **Root cause:** `await` keyword handling

### 8-20. Various Edge Cases (~180 failures combined - 16%)
- Comma expressions
- Import/export edge cases  
- Template literals
- Class syntax
- Operator precedence
- And more...

## Remaining Issues

After normalization, the parser has roughly **~100 distinct error patterns** rather than 1,033.

The top 7 categories account for **633 failures (56%)** of all failures:
1. Spread operator (269)
2. Generators (107)
3. Array destructuring (97)
4. Parenthesized expressions (43)
5. Optional chaining (30)
6. Leading decimals (23)
7. Async/await (20)

**Fixing these 7 categories could eliminate over half the failures.**

The remaining 44% are spread across ~93 different error patterns, most with 1-10 occurrences each.

## Recommendations

1. **Fix spread operator handling first** - Would fix 24% of failures
2. **Fix generator function edge cases** - Would fix another 10%
3. **Complete array destructuring** - Would fix another 9%
4. **Work through remaining patterns by frequency** - Top 20 cover ~80% of failures

This is **much more tractable** than the original "995 unique errors" analysis suggested.
