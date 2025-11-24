# Remaining Test262 Failures - Implementation Guide

**Status as of 2025-11-20**: 126 failures remaining (30,474/37,513 files passing = 81.23%)

This document details the remaining categories of failures and provides implementation guidance for fixing them.

---

## Overview

| Category | Failures | % of Total | Priority | Complexity |
|----------|----------|------------|----------|------------|
| Numeric Literal Lexer Issues | 15 | 11.90% | **HIGH** | Medium |
| Import/Export Edge Cases | 14 | 11.11% | **HIGH** | Low |
| Yield/Await as Identifiers | 11 | 8.73% | **HIGH** | Medium |
| HTML Comment Syntax | 7 | 5.56% | **MEDIUM** | Low |
| ASI Edge Cases | 12 | 9.52% | **MEDIUM** | High |
| For-in `let` binding | 6 | 4.76% | **MEDIUM** | Medium |
| Import.meta in Import Statements | 3 | 2.38% | **LOW** | Low |
| Generator Function Naming | 2 | 1.59% | **LOW** | Low |
| Miscellaneous | 56 | 44.44% | **VARIES** | Varies |

---

## Category 1: Numeric Literal Lexer Issues (15 failures, 11.90%)

### Problem
The lexer incorrectly tokenizes numeric literals ending with a decimal point (e.g., `1.` or `0.`).

**Error Pattern:**
```
ExpectedToken|SEMICOLON|Expected property name after '.'
```

### Example Test Cases
```javascript
// test/built-ins/Array/15.4.4.16-7-b-1.js
var x = 1.toString();  // 1. should be tokenized as complete number

// test/staging/sm/destructuring/obj-rest-non-string-computed-property-1e0.js
let value = 1e0.valueOf();  // 1e0. is valid numeric literal

// test/staging/sm/destructuring/obj-rest-non-string-computed-property-1dot.js
let x = 1..toString();  // First dot ends number, second is member access
```

### Root Cause
**File:** `Lexer.java` (number tokenization logic)

The lexer likely stops tokenizing a number when it encounters a `.` followed by a non-digit, but JavaScript allows numeric literals to end with `.` when followed by a method call or operator.

Valid patterns:
- `1.` - complete numeric literal (value: 1.0)
- `1.toString()` - `1.` followed by method call
- `1..toString()` - `1.` followed by `.toString()`
- `1e0.` - scientific notation followed by `.`

### Implementation Steps

1. **Locate number tokenization in Lexer.java:**
   ```java
   // Find the method that handles NUMBER tokens
   // Likely in scanToken() or a number-specific method
   ```

2. **Update decimal point handling:**
   ```java
   // When encountering '.' during number tokenization:
   if (peek() == '.') {
       advance(); // consume '.'

       // Look ahead to determine if this ends the number
       if (isDigit(peek())) {
           // Continue tokenizing fractional part
           while (isDigit(peek())) advance();
       }
       // else: '.' ends the number, it's part of member access
   }
   ```

3. **Handle edge case: `1..toString()`:**
   - First `.` ends the number literal `1.`
   - Second `.` starts member access
   - Requires lookahead to distinguish

4. **Test with example:**
   ```javascript
   1.toString()    // Should parse as: (1.).toString()
   1..toString()   // Should parse as: (1.).toString()
   1.5.toString()  // Should parse as: (1.5).toString()
   ```

**Estimated Time:** 2-3 hours
**Risk:** Medium (could break existing number parsing)
**Files to Modify:** `Lexer.java`

---

## Category 2: Import/Export Edge Cases (14 failures, 11.11%)

### Problem 2a: Empty Import/Export Specifiers (10 failures)

**Error Pattern:**
```
ExpectedToken|RBRACE|Expected identifier in import specifier
ExpectedToken|RBRACE|Expected identifier in export specifier
```

### Example Test Cases
```javascript
// Empty import with trailing comma
import { } from './module.js';
import { a, } from './module.js';  // trailing comma before }

// Empty export
export { };
export { a, } from './module.js';  // trailing comma before }
```

### Root Cause
**File:** `Parser.java` - `parseImportDeclaration()` and `parseExportDeclaration()`

The parser expects at least one identifier in import/export specifiers and doesn't handle:
1. Empty specifier lists `{ }`
2. Trailing commas before closing brace `{ a, }`

### Implementation Steps

1. **Fix import specifiers (around line 2000-2100 in Parser.java):**
   ```java
   // In parseImportDeclaration(), modify specifier parsing:
   if (match(TokenType.LBRACE)) {
       consume(TokenType.LBRACE, "Expected '{' for import specifiers");

       // Handle empty specifiers
       if (check(TokenType.RBRACE)) {
           consume(TokenType.RBRACE, "Expected '}'");
           // specifiers list remains empty
       } else {
           do {
               // Parse specifier
               ImportSpecifier spec = parseImportSpecifier();
               specifiers.add(spec);

               // Handle trailing comma
               if (match(TokenType.COMMA)) {
                   if (check(TokenType.RBRACE)) {
                       break;  // Trailing comma, stop here
                   }
               } else {
                   break;  // No comma, must be end
               }
           } while (!check(TokenType.RBRACE));

           consume(TokenType.RBRACE, "Expected '}'");
       }
   }
   ```

2. **Apply same fix to export specifiers**

3. **Test cases:**
   ```javascript
   import { } from './a.js';
   import { a, } from './b.js';
   import { a, b, } from './c.js';
   export { };
   export { a, };
   ```

**Estimated Time:** 1-2 hours
**Risk:** Low
**Files to Modify:** `Parser.java`

### Problem 2b: import.meta in Import Statements (4 failures)

**Error Pattern:**
```
ExpectedToken|DOT|Expected 'from' after import specifiers
```

### Example Test Cases
```javascript
// test/language/import/import-defer/goal-*.js
import defer * as ns from './module.js';
// import.meta is being confused with import statement
```

### Root Cause
Parser sees `import.meta` and tries to parse it as an import declaration instead of a meta property.

### Implementation Steps

1. **In `parseStatement()`, check for import.meta:**
   ```java
   case IMPORT:
       advance();

       // Check for import.meta
       if (match(TokenType.DOT)) {
           if (check(TokenType.IDENTIFIER) && peek().lexeme().equals("meta")) {
               // This is import.meta, backtrack and parse as expression
               // Return expression statement
           }
       }

       // Otherwise continue with import declaration
       return parseImportDeclaration();
   ```

**Estimated Time:** 1 hour
**Risk:** Low
**Files to Modify:** `Parser.java`

---

## Category 3: Yield/Await as Identifiers (11 failures, 8.73%)

### Problem
`yield` and `await` can be used as identifiers in certain contexts (labeled statements, object properties) but parser treats them as keywords.

**Error Patterns:**
```
ExpectedToken|COLON|Expected ';' after expression
UnexpectedToken|COLON
UnexpectedToken|ASSIGN
```

### Example Test Cases
```javascript
// test/language/statements/labeled/value-yield-*.js
yield: 42;  // labeled statement where label is "yield"

// test/language/statements/labeled/value-await-*.js
await: 42;  // labeled statement where label is "await"

// As object property
let obj = {
  yield: 1,
  await: 2
};

// As variable in nested non-generator function
function f() {
  yield = 10;  // 'yield' is allowed as identifier in non-generator
}
```

### Root Cause
**File:** `Parser.java` - Multiple locations

The lexer likely emits `YIELD` and `AWAIT` as special token types, but they should be treated as `IDENTIFIER` in certain contexts:

1. **Non-generator functions:** `yield` is a regular identifier
2. **Non-async functions:** `await` is a regular identifier
3. **Labeled statements:** Both can be labels
4. **Object properties:** Both can be property names

### Implementation Steps

1. **Create contextual keyword helper:**
   ```java
   private boolean isContextualKeyword(String lexeme) {
       return lexeme.equals("yield") || lexeme.equals("await");
   }

   private boolean canUseYieldAsIdentifier() {
       // Check parser state
       return !isInGeneratorFunction;
   }

   private boolean canUseAwaitAsIdentifier() {
       // Check parser state
       return !isInAsyncFunction && !isInModuleContext;
   }
   ```

2. **Fix labeled statement parsing (around line 100-200):**
   ```java
   // In parseStatement(), check for labeled statements:
   if (check(TokenType.IDENTIFIER) ||
       (check(TokenType.YIELD) && canUseYieldAsIdentifier()) ||
       (check(TokenType.AWAIT) && canUseAwaitAsIdentifier())) {

       Token potentialLabel = peek();
       advance();

       if (match(TokenType.COLON)) {
           // It's a labeled statement
           Statement stmt = parseStatement();
           return new LabeledStatement(..., potentialLabel.lexeme(), stmt);
       } else {
           // Backtrack, parse as expression
           // ...
       }
   }
   ```

3. **Fix object property names:**
   ```java
   // In parseObjectProperty():
   if (check(TokenType.IDENTIFIER) ||
       check(TokenType.YIELD) ||
       check(TokenType.AWAIT)) {
       // All allowed as property names
       Token key = advance();
       // ...
   }
   ```

4. **Fix variable declarations in non-generator functions:**
   ```java
   // Track function context:
   private int generatorDepth = 0;
   private int asyncDepth = 0;

   // When entering generator: generatorDepth++
   // When exiting generator: generatorDepth--
   ```

**Estimated Time:** 3-4 hours
**Risk:** Medium (affects many parsing contexts)
**Files to Modify:** `Parser.java`, possibly `Lexer.java`

---

## Category 4: HTML Comment Syntax (7 failures, 5.56%)

### Problem
JavaScript allows HTML-style comments in script mode: `<!--` for single-line comments and `-->` at start of line.

**Error Pattern:**
```
UnexpectedToken|GT
```

### Example Test Cases
```javascript
// test/language/comments/hashbang/single-line-html-close.js
<!-- This is a comment
var x = 1;

// test/language/comments/hashbang/multi-line-html-close.js
/*
--> This should be treated as comment at line start
*/
```

### Root Cause
**File:** `Lexer.java`

The lexer doesn't recognize HTML comment syntax:
- `<!--` starts a single-line comment (like `//`)
- `-->` at the beginning of a line is a single-line comment

### Implementation Steps

1. **In Lexer.java, add HTML comment detection:**
   ```java
   // In scanToken():
   case '<':
       if (match('!')) {
           if (match('-')) {
               if (match('-')) {
                   // <!-- found, treat as single-line comment
                   skipLineComment();
                   return scanToken();  // Skip this token
               }
           }
       }
       // Otherwise handle as LT operator
       break;

   case '-':
       if (match('-')) {
           if (match('>')) {
               // --> found
               // Only valid at line start
               if (isAtLineStart) {
                   skipLineComment();
                   return scanToken();
               }
           }
       }
       // Otherwise handle as MINUS or DECREMENT
       break;
   ```

2. **Track line start position:**
   ```java
   private boolean isAtLineStart = true;

   // Reset when encountering newline
   // Set to false when encountering non-whitespace
   ```

**Estimated Time:** 1-2 hours
**Risk:** Low (only affects script mode)
**Files to Modify:** `Lexer.java`

---

## Category 5: ASI Edge Cases (12 failures, 9.52%)

### Problem
Automatic Semicolon Insertion (ASI) has complex rules that aren't fully implemented.

**Error Patterns:**
```
Expected ';' after expression
Expected ';' after do-while statement
```

### Example Test Cases
```javascript
// test/language/asi/S7.9_A5.2_T1.js
x = y
y = z  // ASI should insert ; after first line

// test/language/asi/S7.9_A9_T1.js
do {
  //
} while (false)  // ASI should insert ; after while

// test/language/module-code/top-level-await/multi-line-asi-*.js
await foo
'string'  // ASI with different line terminators (CR, LS, PS)
```

### Root Cause
**File:** `Parser.java` - Multiple locations

ASI rules (ECMAScript spec 11.9):
1. Insert `;` when next token is `}`, `)`, or end of input
2. Insert `;` when next token is on new line and would cause parse error
3. Insert `;` after `do-while` statements
4. Special handling for `return`, `throw`, `break`, `continue`, `yield`, `module export`

### Implementation Steps

1. **Add ASI helper method:**
   ```java
   private boolean shouldInsertSemicolon() {
       if (isAtEnd()) return true;
       if (check(TokenType.RBRACE)) return true;
       if (check(TokenType.SEMICOLON)) return false;

       // Check if current token is on new line
       Token current = peek();
       Token previous = previous();
       if (current.line() > previous.line()) {
           return true;  // Newline allows ASI
       }

       return false;
   }

   private void consumeSemicolonWithASI(String errorMessage) {
       if (match(TokenType.SEMICOLON)) {
           return;  // Explicit semicolon
       }

       if (shouldInsertSemicolon()) {
           return;  // ASI applies
       }

       throw new ParseException("ExpectedToken", peek(), TokenType.SEMICOLON,
                                "after statement", errorMessage);
   }
   ```

2. **Apply to expression statements:**
   ```java
   // In parseExpressionStatement():
   Expression expr = parseExpression();
   consumeSemicolonWithASI("Expected ';' after expression");
   ```

3. **Fix do-while ASI:**
   ```java
   // In parseDoWhileStatement():
   Statement body = parseStatement();
   consume(TokenType.WHILE, "Expected 'while'");
   consume(TokenType.LPAREN, "Expected '('");
   Expression condition = parseExpression();
   consume(TokenType.RPAREN, "Expected ')'");

   // ASI applies after do-while
   consumeSemicolonWithASI("Expected ';' after do-while statement");
   ```

4. **Handle different line terminators:**
   - Ensure lexer recognizes `\r`, `\n`, `\r\n`, `\u2028` (LS), `\u2029` (PS)
   - Track line numbers correctly for all terminators

**Estimated Time:** 4-5 hours
**Risk:** High (affects many statements, could break working code)
**Files to Modify:** `Parser.java`, possibly `Lexer.java`

---

## Category 6: For-in `let` binding (6 failures, 4.76%)

### Problem
Parser doesn't allow `let` as a binding identifier in for-in loops in non-strict mode.

**Error Pattern:**
```
UnexpectedToken|LET
```

### Example Test Cases
```javascript
// test/staging/sm/Function/rest-parameter-names.js
for (let let in {}) {}  // 'let' can be identifier in non-strict mode

// test/language/statements/for-in/head-var-bound-names-let.js
for (var let in {}) {}  // 'let' as variable name
```

### Root Cause
**File:** `Parser.java` - `parseForStatement()`

In non-strict mode, `let` can be used as an identifier. The parser is rejecting it unconditionally.

### Implementation Steps

1. **Track strict mode context:**
   ```java
   private boolean isStrictMode = false;

   // Set based on:
   // - "use strict" directive
   // - module context (always strict)
   // - class bodies (always strict)
   ```

2. **Fix for-in/for-of variable parsing:**
   ```java
   // In parseForStatement():
   if (match(TokenType.VAR) || match(TokenType.LET) || match(TokenType.CONST)) {
       TokenType declType = previous().type();

       // Allow 'let' as identifier in non-strict mode
       if (check(TokenType.LET) && !isStrictMode && declType == TokenType.VAR) {
           Token identifier = advance();  // consume 'let' as identifier
           // Continue parsing with 'let' as the variable name
       } else {
           Token identifier = consume(TokenType.IDENTIFIER, "Expected identifier");
       }
   }
   ```

**Estimated Time:** 1-2 hours
**Risk:** Low
**Files to Modify:** `Parser.java`

---

## Category 7: Miscellaneous Issues (56 failures, 44.44%)

### Problem 7a: Generator Function Naming (2 failures)

**Error Pattern:**
```
ExpectedToken|YIELD|Expected function name
```

**Example:**
```javascript
function* yield() {}  // 'yield' can be generator name in non-generator context
```

**Fix:** Allow `yield` as function name in function declarations when not inside a generator.

---

### Problem 7b: Optional Chaining with Numeric Literals (2 failures)

**Error Pattern:**
```
ExpectedToken|NUMBER|Expected property name after '?.'
```

**Example:**
```javascript
obj?.4  // Invalid, but needs better error
```

**Fix:** This is actually invalid syntax. May be test expectation issue.

---

### Problem 7c: Boolean/null as Property Keys (2 failures)

**Error Pattern:**
```
ExpectedToken|TRUE|Expected property key
```

**Example:**
```javascript
{ true: 1, false: 2, null: 3 }
```

**Fix:** Allow literal keywords as object property keys.

---

### Problem 7d: Arrow Function in For Loop Head (3 failures)

**Error Pattern:**
```
UnexpectedToken|ARROW
```

**Example:**
```javascript
for (x => x; ;) {}  // Invalid but needs proper error
```

**Fix:** Detect and report better error for invalid syntax.

---

### Problem 7e: Variable Declaration Edge Cases (3 failures)

**Error Pattern:**
```
ExpectedToken|ASSIGN|Expected identifier in variable declaration
```

**Example:**
```javascript
let
= 42;  // 'let' split across lines
```

**Fix:** Improve ASI handling for `let` declarations.

---

### Problem 7f: Async Iterator Protocols (5+ failures)

**Various patterns** related to `async` keyword in iterator contexts.

**Fix:** Review async/await support in iterator methods.

---

## Implementation Priority

### Phase 1: Quick Wins (2-3 hours total)
1. ✅ Empty import/export specifiers (10 failures) - **HIGHEST ROI**
2. ✅ HTML comments (7 failures)
3. ✅ import.meta handling (3 failures)
4. ✅ For-in `let` binding (6 failures)

**Expected reduction: 26 failures → 100 remaining**

### Phase 2: Medium Complexity (5-7 hours total)
1. ✅ Numeric literal lexer fix (15 failures)
2. ✅ Yield/await as identifiers (11 failures)
3. ✅ Generator function naming (2 failures)

**Expected reduction: 28 failures → 72 remaining**

### Phase 3: Complex ASI (4-5 hours)
1. ✅ ASI edge cases (12 failures)

**Expected reduction: 12 failures → 60 remaining**

### Phase 4: Miscellaneous (8-10 hours)
1. ✅ Review and fix remaining 60+ individual edge cases
2. ✅ Some may be test infrastructure issues
3. ✅ Some may require spec clarification

---

## Testing Strategy

After each fix:
1. Run full Test262 suite: `mvn test`
2. Check categorization: `node scripts/normalize_json_errors.js`
3. Verify no regressions in previously passing tests
4. Update this document with progress

---

## Progress Tracking

| Date | Failures | Change | Fix Applied |
|------|----------|--------|-------------|
| 2025-11-20 | 126 | baseline | Starting point |
| TBD | TBD | TBD | TBD |

---

## Notes

- Total Test262 suite: 37,513 files
- Current passing: 30,474 (81.23%)
- Target: 95%+ (35,637+ files)
- Remaining to fix: ~5,163 files across all categories

The fixes outlined above should address the systematic errors. The long tail of miscellaneous failures will require individual analysis.
