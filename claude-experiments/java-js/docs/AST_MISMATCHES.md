# AST Mismatch Investigation

## Summary

Out of 43,774 cached test262 files, there are **9 AST mismatches (0.02%)**. These represent edge cases where the Java parser's AST differs from Acorn's AST.

## Test Results

- **Total files scanned:** 51,680
- **Files with cache:** 43,774
- **Exact matches:** 43,765 (99.98%)
- **AST mismatches:** 9 (0.02%)
- **Parse failures:** 0 (0.00%)

## Mismatched Files

### 1. Special Line Terminators (4 files)

Files that test handling of non-standard line terminators:

1. `language/comments/multi-line-asi-carriage-return.js`
   - Contains: `\r` (Carriage Return)

2. `language/comments/multi-line-asi-line-separator.js`
   - Contains: `\u2028` (LINE SEPARATOR)

3. `language/comments/multi-line-asi-line-feed.js`
   - Contains: `\n` (Line Feed - standard)

4. `language/comments/multi-line-asi-paragraph-separator.js`
   - Contains: `\u2029` (PARAGRAPH SEPARATOR)

**Issue:** These test automatic semicolon insertion (ASI) with different line terminator types. The mismatch likely stems from how the Java parser handles these special Unicode line terminators when calculating source positions or ASI behavior.

**ECMAScript Spec:** Section 11.3 defines line terminators as:
- `\u000A` (LINE FEED - LF)
- `\u000D` (CARRIAGE RETURN - CR)
- `\u2028` (LINE SEPARATOR - LS)
- `\u2029` (PARAGRAPH SEPARATOR - PS)

### 2. Class Field ASI (2 files)

5. `language/statements/class/elements/syntax/valid/grammar-field-named-set-followed-by-generator-asi.js`
6. `language/statements/class/elements/syntax/valid/grammar-field-named-get-followed-by-generator-asi.js`

**Issue:** These test automatic semicolon insertion in class fields when followed by generator methods. The parser may be handling ASI differently for class field declarations.

### 3. BigInt Property Names (1 file)

7. `language/expressions/object/literal-property-name-bigint.js`

**Issue:** This file uses BigInt literals as property names (e.g., `{ 1n: value }`). The mismatch may be in how BigInt values are represented in the AST. Acorn's JSON serialization converts `typeof value === "bigint"` to `null`, but the Java parser might not handle BigInt the same way.

**Example code from file:**
```javascript
let o = { 999999999999999999n: true };
o = { 1n() { return "bar"; } };
let { 1n: a } = { "1": "foo" };
```

### 4. new.target ASI (1 file)

8. `language/expressions/new.target/asi.js`

**Issue:** Tests automatic semicolon insertion with `new.target` meta-property. The parser may be handling ASI differently around this ES6 feature.

### 5. with Statement Strict Mode (1 file)

9. `language/statements/with/12.10.1-13-s.js`

**Issue:** Tests `with` statement in strict mode (which should be an error). This might be a difference in how the error is represented in the AST or how strict mode is detected.

## Root Causes

Based on the analysis, the mismatches fall into these categories:

### 1. Line Terminator Handling (4 files)
The Java lexer may not properly recognize all four ECMAScript line terminators when:
- Calculating line/column positions
- Determining automatic semicolon insertion points
- Normalizing line endings

**Fix Location:** `src/main/java/com/jsparser/Lexer.java`

### 2. Automatic Semicolon Insertion (4 files)
ASI edge cases in:
- Class field declarations
- After `new.target` expressions
- With special line terminators

**Fix Location:** `src/main/java/com/jsparser/Parser.java` - ASI logic

### 3. BigInt Handling (1 file)
The Java parser may not:
- Parse BigInt literals in all contexts (property names)
- Serialize BigInt values the same way as Acorn

**Fix Location:**
- `src/main/java/com/jsparser/Lexer.java` - BigInt lexing
- AST node classes - BigInt value representation

### 4. Strict Mode Detection (1 file)
Possible difference in how strict mode violations are detected/represented.

**Fix Location:** `src/main/java/com/jsparser/Parser.java` - Strict mode handling

## Recommendations

1. **Priority: Medium-Low** - 99.98% match rate is excellent. These are edge cases.

2. **Line Terminator Fix:**
   - Update lexer to properly handle all four ECMAScript line terminators
   - Ensure consistent line/column counting across all terminator types
   - Test: `isLineTerminator()` should return true for `\n`, `\r`, `\u2028`, `\u2029`

3. **BigInt Fix:**
   - Ensure BigInt literals are parsed in all property name contexts
   - Match Acorn's JSON serialization of BigInt (convert to `null`)

4. **ASI Fix:**
   - Review ASI logic for class fields
   - Review ASI logic for `new.target` expressions
   - Ensure ASI works correctly with all line terminators

5. **Testing:**
   - Add unit tests for each mismatch category
   - Run Test262Runner after each fix to verify improvement

## Files for Reference

- Mismatched file list: See Test262Runner output
- Debug script: `scripts/debug-mismatch.js`
- Cache generator: `scripts/generate-test262-cache.js`
