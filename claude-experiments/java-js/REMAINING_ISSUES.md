# Test262 Issues - ALL FIXED! 🎉

**Current Status: 100.00% accuracy (43,209 / 43,209 files passing)**

All Test262 oracle mismatches have been fixed!

---

## Summary of Fixes

1. ~~**BigInt Literal Property Names** (1 file) - ✅ FIXED~~
2. ~~**Position Tracking** (6 files) - ✅ FIXED~~
3. ~~**Contextual Keyword Parsing** (5 files) - ✅ FIXED~~
4. ~~**Async Arrow Function Parsing** (2 files) - ✅ FIXED~~
5. ~~**Operator Precedence** (1 file) - ✅ FIXED~~

**Total: 15 test files fixed**

---

## Issue 1: BigInt Literal Property Names (1 file) ✅ FIXED

### Status
**FIXED** - Parser.java:2844-2887 and Parser.java:782-824

### Solution
Added BigInt literal detection when parsing property keys in both object literals and class bodies. When a number token ends with 'n', we now:
1. Set `value` to `null` (since BigInt can't be represented in JSON)
2. Store the numeric part in the `bigint` field
3. Keep the full `raw` string including the 'n' suffix
4. Convert hex/octal/binary BigInts to decimal for the `bigint` field

### Files Modified
- `Parser.java` - Object literal property key parsing (line ~2844)
- `Parser.java` - Class property key parsing (line ~782)

---

## Issue 2: Position Tracking (6 files) ✅ FIXED

### Status
**FIXED** - Lexer.java:158

### Solution
Fixed column tracking when backtracking position during optional chaining lookahead. When the lexer detects `?.` followed by a decimal digit (like `?.30`), it needs to backtrack and treat it as two separate tokens (`?` and `.30`). The bug was that we decremented `position` but forgot to decrement `column`, causing all subsequent tokens on that line to have off-by-one column positions.

### Files Fixed
- `language/expressions/optional-chaining/punctuator-decimal-lookahead.js` ✅
- `language/expressions/new.target/asi.js` ✅
- `language/comments/multi-line-asi-carriage-return.js` ✅
- `language/comments/multi-line-asi-line-feed.js` ✅
- `language/comments/multi-line-asi-paragraph-separator.js` ✅
- `language/comments/multi-line-asi-line-separator.js` ✅

### Fix Applied
Added `column--` when backtracking position in the optional chaining lookahead code

---

## Issue 3: Contextual Keywords (5 files) ✅ FIXED

### Status
**FIXED** - Parser.java:600-605 and Parser.java:3155-3158

### Solution
Fixed generator context management in function parameter parsing. The issue was that we were setting the generator context AFTER parsing parameters, but parameter default values (like `x = yield`) need the correct context.

### Files Fixed
- `language/statements/function/param-dflt-yield-non-strict.js` ✅
- `language/expressions/function/param-dflt-yield-non-strict.js` ✅
- `language/statements/class/elements/syntax/valid/grammar-field-named-set-followed-by-generator-asi.js` ✅
- `language/statements/class/elements/syntax/valid/grammar-field-named-get-followed-by-generator-asi.js` ✅
- `language/statements/with/12.10.1-13-s.js` ✅

### Problem 3a: `yield` as Identifier in Non-Strict Mode - FIXED

**Files:**
- `language/statements/function/param-dflt-yield-non-strict.js`
- `language/expressions/function/param-dflt-yield-non-strict.js`

#### Example
```javascript
var yield = 23;  // In non-strict mode, 'yield' is just an identifier

function *g() {
  function f(x = yield) {  // 'yield' here should be Identifier, not YieldExpression
    paramValue = x;
  }
  f();
}
```

#### Expected vs Actual
```
.body[2].body.body[0].params[0].right.type: expected="Identifier", actual="YieldExpression"
```

#### Root Cause
In non-strict mode, `yield` is a regular identifier (not a keyword). Our parser is treating it as a `YieldExpression` even in non-strict mode.

The parser needs to check:
1. Are we in strict mode?
2. Are we inside a generator function?
3. Context matters: `function *g() { function f(x = yield) {} }` - the inner function `f` is NOT a generator, so `yield` should be an identifier.

### Problem 3b: `get`/`set` Field ASI

**Files:**
- `language/statements/class/elements/syntax/valid/grammar-field-named-get-followed-by-generator-asi.js`
- `language/statements/class/elements/syntax/valid/grammar-field-named-set-followed-by-generator-asi.js`

#### Example
```javascript
class A {
  get      // This is a field named "get"
  *a() {}  // This is a generator method named "a"
}
```

#### Expected vs Actual
```
.body[0].body.body[0].type: expected="PropertyDefinition", actual="MethodDefinition"
```

#### Root Cause
When `get` or `set` is followed by a line terminator, it should be treated as a field name (with ASI), not as the start of a getter/setter. Our parser is treating it as a getter/setter method.

The fix requires checking for line terminators after `get`/`set` keywords in class bodies.

### Fix Location
- `Parser.java` - `yield` identifier vs expression logic
- `Parser.java` - Class member parsing for `get`/`set` ASI
- Need to track strict mode context
- Need to track generator function context

### Complexity
**Medium-High** - Requires context-aware parsing and strict mode tracking

---

## Issue 4: Async Arrow Functions (2 files) ✅ FIXED

### Status
**FIXED** - Parser.java:1639-1643

### Solution
Added line terminator check between `async` and the next token when detecting async arrow functions. According to the grammar `async [no LineTerminator here] AsyncArrowBindingIdentifier`, a line terminator after `async` means it's not an async arrow function, so `async` should be parsed as a regular identifier.

### Fix Applied
Check `asyncToken.line() != nextToken.line()` before treating the construct as an async arrow function. If there's a line terminator, fall through to parse `async` as a regular identifier.

---

## Issue 5: Operator Precedence (1 file) ✅ FIXED

### Status
**FIXED** - Parser.java:3090-3097

### Solution
Added tagged template literal handling in the `new` expression parsing. After parsing the callee and handling member access (`.` and `[`), we now also check for template literals and parse them as tagged templates. This ensures that `new tag`template`` correctly parses as `new (tag`template`)` rather than `(new tag)`template``.

### Fix Applied
Added template literal check and TaggedTemplateExpression creation in the NEW case, treating tagged templates as part of the MemberExpression that `new` applies to.

---

## Issue 6: Multi-line Comment ASI (3 files)

### Files
- `language/comments/multi-line-asi-carriage-return.js`
- `language/comments/multi-line-asi-line-feed.js`
- `language/comments/multi-line-asi-paragraph-separator.js`
- `language/comments/multi-line-asi-line-separator.js`

### Note
The test `language/statements/with/12.10.1-13-s.js` has been removed from the mismatch list as it's now passing.

---

## Fixes Completed (in order)

1. ~~**BigInt Literal Value**~~ - ✅ FIXED (1 file)
2. ~~**Position Tracking**~~ - ✅ FIXED (6 files)
3. ~~**Operator Precedence (Tagged Templates)**~~ - ✅ FIXED (1 file)
4. ~~**Async Arrow Line Terminator**~~ - ✅ FIXED (2 files)
5. ~~**Contextual Keywords**~~ - ✅ FIXED (5 files)

**All issues resolved! 🎉**

---

## Implementation Notes

### Strict Mode Tracking
Several fixes require tracking whether we're in strict mode:
- `yield` identifier vs expression
- Future compatibility for other contextual keywords

Suggestion: Add a `strictMode` boolean to the parser state that gets set when:
- A "use strict" directive is found
- We're in a module (modules are always strict)
- We're in a class body (class bodies are always strict)

### Generator Context Tracking
The `yield` fix requires knowing if we're inside a generator function:
- Track generator context depth
- Decrement when exiting generator function scope
- `yield` is only a keyword inside generators

### Line Terminator Checks
Multiple fixes need to check for line terminators:
- Async arrow functions
- Class field `get`/`set` ASI
- Already implemented for `break`/`continue`

Pattern:
```java
if (previousToken.line() != peek().line()) {
    // Line terminator detected
}
```
