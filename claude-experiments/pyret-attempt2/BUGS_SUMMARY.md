# Bug Summary - Comprehensive Testing Results

**Date**: 2025-10-31
**Discovered By**: Comprehensive test suite
**Total Bugs Filed**: 6

---

## Overview

During the development of a comprehensive test suite for the Pyret parser, systematic testing revealed 6 issues with the parser's validation and error handling. All bugs have been filed in `BUGS.md` with unique IDs for tracking.

---

## High Severity Bugs (2)

### 1. Parser accepts trailing tokens after valid expression
**ID**: `ideal-thorny-kingfisher`
**Severity**: High
**File**: `tests/error_tests.rs:316`

**Issue**: Parser stops after the first valid expression and silently ignores trailing tokens.

**Example**:
```rust
parse_expr("42 unexpected")  // Returns Ok(SNum{42}), ignoring "unexpected"
```

**Impact**: Users don't get syntax errors for malformed input, making debugging harder.

---

### 2. Parser accepts multiple expressions without separator
**ID**: `oily-awkward-hedgehog`
**Severity**: High
**File**: `tests/error_tests.rs:322`

**Issue**: Parser accepts input with multiple expressions lacking proper delimiters.

**Example**:
```rust
parse_expr("1 + 2 3 + 4")  // Should error but succeeds
```

**Impact**: Invalid syntax silently accepted, hiding errors from users.

---

## Medium Severity Bugs (3)

### 3. Parser accepts unmatched closing parenthesis
**ID**: `comfortable-pink-hedgehog`
**Severity**: Medium
**File**: `tests/error_tests.rs:27`

**Issue**: Extra closing parentheses are accepted without error.

**Example**:
```rust
parse_expr("42)")  // Should error but succeeds
```

**Impact**: Reduces error detection capability.

---

### 4. Parser accepts unmatched closing bracket
**ID**: `palatable-edible-crayfish`
**Severity**: Medium
**File**: `tests/error_tests.rs:34`

**Issue**: Extra closing brackets are accepted without error.

**Example**:
```rust
parse_expr("1]")  // Should error but succeeds
```

**Impact**: Similar to parenthesis issue, reduces error detection.

---

### 5. Parser accepts invalid characters after valid expression
**ID**: `reflecting-enchanting-caribou`
**Severity**: Medium
**File**: `tests/error_tests.rs:316`

**Issue**: Invalid tokens after expressions are silently ignored.

**Example**:
```rust
parse_expr("f(x) @")  // Should error on "@" but succeeds
```

**Impact**: Prevents early error detection.

---

## Low Severity / Design Decisions (1)

### 6. Number representation differs from official Pyret
**ID**: `stunning-foolhardy-snake`
**Severity**: Low
**File**: `src/parser.rs` (parse_number)

**Issue**: We use floats (`3.14`), Pyret uses rationals (`157/50`).

**Example**:
```rust
// Rust:  Expr::SNum { n: 3.14 }
// Pyret: { type: "s-num", value: "157/50" }
```

**Impact**: AST comparisons fail for decimal numbers. Integration tests skip decimals.

**Note**: This is a design decision that may require discussion. Rationals are more precise but floats are simpler.

---

## Common Root Cause

Most bugs (5/6) stem from **lenient parsing** - the parser successfully returns after parsing one valid expression without checking for:
- Trailing tokens
- EOF (end of file)
- Invalid subsequent tokens

**Fix Strategy**: Add validation after expression parsing to ensure:
1. All input has been consumed
2. No trailing invalid tokens
3. Proper error messages for malformed input

---

## Test Coverage

These bugs were discovered by:

- **Error Tests** (`tests/error_tests.rs`): 41 tests specifically checking error conditions
- **Integration Tests** (`tests/comparison_tests.rs`): 53 tests comparing with official Pyret parser
- **Systematic Tests** (`test_all_syntax.sh`): 150+ test cases

---

## Verification Commands

To verify each bug:

```bash
# High severity
cargo test test_expression_with_trailing_tokens
cargo test test_multiple_expressions

# Medium severity
cargo test test_error_unmatched_paren_right
cargo test test_error_unmatched_bracket_right
cargo test test_valid_prefix_invalid_suffix

# Low severity (design decision)
./compare_parsers.sh "3.14"
```

---

## Next Steps

### Immediate
1. Review bugs with team/maintainer
2. Decide on acceptable leniency level
3. Prioritize fixes (high severity first)

### Implementation
1. Add EOF checking after expression parsing
2. Add trailing token validation
3. Improve delimiter matching
4. Consider rational number support

### Testing
1. Update error tests after fixes
2. Verify all tests pass
3. Add regression tests

---

## Documentation Updates

All bugs are documented in:
- ✅ `BUGS.md` - Full bug tracker with IDs
- ✅ `tests/error_tests.rs` - Test code with bug IDs
- ✅ `TEST_SUITE_SUMMARY.md` - Testing summary
- ✅ `BUGS_SUMMARY.md` - This file

---

## Bug Tracker Usage

View all bugs:
```bash
bug-tracker list
```

View specific bug:
```bash
bug-tracker view comfortable-pink-hedgehog
```

Close a bug (after fixing):
```bash
bug-tracker close comfortable-pink-hedgehog
```

---

**Summary**: Comprehensive testing successfully identified 6 issues requiring attention. High-quality bug reports with reproduction steps, code snippets, and test cases are available in `BUGS.md`. The test suite provides a solid foundation for verifying fixes and preventing regressions.
