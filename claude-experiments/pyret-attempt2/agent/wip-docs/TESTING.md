# Comprehensive Testing Guide

This document describes the complete testing infrastructure for the Pyret parser project.

## Overview

The testing strategy consists of five complementary approaches:

1. **Unit Tests** - Fast, focused tests for specific features
2. **Integration Tests** - Compare against official Pyret parser
3. **Systematic Tests** - Comprehensive coverage of all syntax combinations
4. **Property-Based Tests** - Fuzz testing with random expressions
5. **Error Tests** - Validate error handling and edge cases

## Quick Start

```bash
# Run all Rust tests (unit + integration + error tests)
cargo test

# Run just parser unit tests
cargo test --test parser_tests

# Run integration tests (compares with official Pyret)
cargo test --test comparison_tests

# Run error handling tests
cargo test --test error_tests

# Run comprehensive syntax test suite
./test_all_syntax.sh

# Run fuzzer (generates 100 random expressions)
./test_fuzzer.py

# Run fuzzer with custom settings
./test_fuzzer.py --count 500 --max-depth 7 --seed 42
```

## Test Suite Components

### 1. Unit Tests (`tests/parser_tests.rs`)

**Purpose**: Fast, focused tests for individual parsing features.

**Coverage**:
- Primitives (numbers, strings, booleans, identifiers)
- Binary operators (all 15 operators)
- Left-associativity
- Parenthesized expressions
- Function application
- Whitespace sensitivity
- Dot access
- Array expressions
- Mixed expressions

**Usage**:
```bash
cargo test --test parser_tests

# Run specific test
cargo test test_parse_simple_addition

# Debug tokens
DEBUG_TOKENS=1 cargo test test_name
```

**Example**:
```rust
#[test]
fn test_parse_simple_addition() {
    let expr = parse_expr("1 + 2").expect("Failed to parse");
    match expr {
        Expr::SOp { op, .. } => assert_eq!(op, "op+"),
        _ => panic!("Expected SOp"),
    }
}
```

**Stats**: 35+ tests, all passing ✅

---

### 2. Integration Tests (`tests/comparison_tests.rs`)

**Purpose**: Validate that our parser produces identical ASTs to the official Pyret parser.

**How it works**:
1. Runs `compare_parsers.sh` script
2. Parses expression with both Pyret and Rust parsers
3. Compares normalized JSON output
4. Fails if ASTs differ

**Coverage**:
- All primitives
- All binary operators
- Chained operators
- Parentheses
- Function calls (simple, chained, with expressions)
- Whitespace sensitivity
- Dot access (simple, chained)
- Mixed postfix operators
- Arrays
- Edge cases (deeply nested, long expressions)

**Usage**:
```bash
# Run all comparison tests
cargo test --test comparison_tests

# Run specific category
cargo test test_pyret_match_arrays

# Verbose mode (see comparison output)
cargo test -- --nocapture
```

**Important**: Requires official Pyret parser at:
```
/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang
```

**Stats**: 80+ integration tests

---

### 3. Systematic Test Suite (`test_all_syntax.sh`)

**Purpose**: Systematically test all verified syntax combinations.

**Categories** (13 total):
1. Primitives (numbers, strings, booleans, identifiers)
2. Binary Operators (all 15 operators)
3. Left-Associativity (chained operators)
4. Parentheses (simple, nested, grouping)
5. Function Calls (args, chaining)
6. Whitespace Sensitivity
7. Dot Access (simple, chained)
8. Mixed Postfix Operators
9. Arrays (empty, nested, with expressions)
10. Complex Combinations
11. Deep Nesting (edge cases)
12. Long Expressions (edge cases)
13. Real-World Patterns

**Usage**:
```bash
# Run all systematic tests
./test_all_syntax.sh

# Verbose mode (see each test)
./test_all_syntax.sh --verbose
```

**Output**:
```
========================================
Comprehensive Pyret Parser Test Suite
========================================

Category 1: Primitives
..........
Category 2: Binary Operators
..................
...

========================================
Test Summary
========================================
Total tests: 150
Passed: 150
Failed: 0

✅ All tests passed!
```

**Stats**: 150+ systematic tests

---

### 4. Property-Based Testing (`test_fuzzer.py`)

**Purpose**: Generate random valid Pyret expressions and test them.

**Strategy**:
- Randomly generates expressions up to specified depth
- Ensures expressions are syntactically valid
- Tests against official Pyret parser
- Finds edge cases not covered by manual tests

**Expression Types Generated**:
- Primitives
- Binary operations
- Parenthesized expressions
- Function calls (with/without whitespace)
- Dot access
- Arrays
- Chained calls
- Chained dot access
- Mixed postfix operators

**Usage**:
```bash
# Generate and test 100 expressions
./test_fuzzer.py

# Custom settings
./test_fuzzer.py --count 500 --max-depth 7

# Reproducible tests (with seed)
./test_fuzzer.py --seed 42

# Verbose mode
./test_fuzzer.py --verbose

# Save failures to file
./test_fuzzer.py --save-failures failures.txt
```

**Example Output**:
```
============================================================
Pyret Parser Fuzzer
============================================================
Generating 100 random expressions (max depth: 5)

....................................F...................F...
........................

============================================================
Results
============================================================
Total: 100
Passed: 98 (98%)
Failed: 2 (2%)

Failed expressions:
  ❌ f(x + 1).foo.bar[0]
  ❌ [[obj.field, g(y)], x is y]
```

**Stats**: Configurable (default 100 tests per run)

---

### 5. Error Tests (`tests/error_tests.rs`)

**Purpose**: Ensure parser handles errors gracefully.

**Coverage**:
- Unmatched delimiters (`(42`, `[1, 2`)
- Mismatched delimiters (`(1]`, `[2)`)
- Invalid operators (`1 +`, `+ 1`, `1 + + 2`)
- Invalid function calls (`f(x y)`, `f(x,)`)
- Invalid arrays (`[1 2]`, `[1,,2]`)
- Invalid dot access (`obj.`, `.field`)
- Empty input (``, `   `)
- Edge cases (deeply nested, very long)
- Invalid characters
- Parser state errors

**Usage**:
```bash
# Run all error tests
cargo test --test error_tests

# Run specific category
cargo test test_error_unmatched
```

**Example**:
```rust
#[test]
fn test_error_unmatched_paren_left() {
    let result = parse_expr("(42");
    assert!(result.is_err(), "Should error on unmatched paren");
}
```

**Stats**: 40+ error tests

---

## Test Statistics Summary

| Test Suite | Count | Purpose | Speed |
|------------|-------|---------|-------|
| Unit Tests | 35+ | Feature-specific | Fast (ms) |
| Integration Tests | 80+ | Validate against Pyret | Medium (sec) |
| Systematic Tests | 150+ | Comprehensive coverage | Slow (min) |
| Fuzzer | Configurable | Find edge cases | Slow (min) |
| Error Tests | 40+ | Error handling | Fast (ms) |
| **Total** | **300+** | | |

---

## Comparison Tools

### `compare_parsers.sh`

**Purpose**: Compare single expression between Pyret and Rust parsers.

**Usage**:
```bash
./compare_parsers.sh "2 + 3"
./compare_parsers.sh "f(x).result"
```

**Output** (when identical):
```
=== Input ===
2 + 3

=== Pyret Parser ===
{
  "type": "s-op",
  "op": "op+",
  "left": { "type": "s-num", "value": "2" },
  "right": { "type": "s-num", "value": "3" }
}

=== Rust Parser ===
{
  "type": "s-op",
  "op": "op+",
  "left": { "type": "s-num", "value": "2" },
  "right": { "type": "s-num", "value": "3" }
}

=== Comparison ===
✅ IDENTICAL - Parsers produce the same AST!
```

### `to_pyret_json` Binary

**Purpose**: Convert Rust AST to Pyret's JSON format.

**Usage**:
```bash
# From stdin
echo "2 + 3" | cargo run --bin to_pyret_json

# From file
cargo run --bin to_pyret_json path/to/file.arr
```

---

## Running Tests in CI/CD

Recommended test sequence for continuous integration:

```bash
#!/bin/bash
set -e

echo "1. Running unit tests..."
cargo test --test parser_tests

echo "2. Running error tests..."
cargo test --test error_tests

echo "3. Running integration tests..."
cargo test --test comparison_tests

echo "4. Running systematic tests..."
./test_all_syntax.sh

echo "5. Running fuzzer..."
./test_fuzzer.py --count 200 --seed 12345

echo "All tests passed! ✅"
```

---

## Test Coverage

### Verified Syntax ✅

All syntax in this table has been verified identical to official Pyret parser:

| Category | Examples | Tests |
|----------|----------|-------|
| **Primitives** | `42`, `"hello"`, `true`, `x` | ✅ |
| **Binary Operators** | `2 + 3`, `x * y`, `a and b` | ✅ |
| **Chained Operators** | `2 + 3 * 4` (left-assoc) | ✅ |
| **Parentheses** | `(2 + 3)`, `((x))` | ✅ |
| **Function Calls** | `f(x)`, `f(1, 2)`, `f()` | ✅ |
| **Chained Calls** | `f(x)(y)`, `f()(g())` | ✅ |
| **Dot Access** | `obj.field`, `obj.a.b.c` | ✅ |
| **Mixed Postfix** | `f(x).result`, `obj.foo()` | ✅ |
| **Arrays** | `[1, 2, 3]`, `[[1], [2]]` | ✅ |
| **Complex** | `f(x + 1).foo.bar` | ✅ |

### Not Yet Implemented ❌

These features will cause test failures:

- Objects: `{ x: 1, y: 2 }`
- Tuples: `{1; 2; 3}`
- Bracket access: `arr[0]`
- Control flow: `if`, `cases`, `when`, `for`
- Functions: `fun`, `lam`, `method`
- Let bindings: `let x = 5:`
- Blocks: `block: ... end`

---

## Debugging Failed Tests

### Unit Test Failed

```bash
# See detailed error
cargo test test_name -- --nocapture

# Debug tokens
DEBUG_TOKENS=1 cargo test test_name

# Check AST structure
cargo test test_name -- --nocapture | grep -A 20 "got"
```

### Integration Test Failed

```bash
# Run comparison script directly
./compare_parsers.sh "expression that failed"

# Check Rust output
echo "expr" | cargo run --bin to_pyret_json

# Check Pyret output
cd /Users/jimmyhmiller/Documents/Code/open-source/pyret-lang
node ast-to-json.jarr test.arr output.json
```

### Systematic Test Failed

```bash
# Run in verbose mode to see which test
./test_all_syntax.sh --verbose

# Test specific expression
./compare_parsers.sh "the failing expression"
```

### Fuzzer Found Issue

```bash
# Re-run with same seed to reproduce
./test_fuzzer.py --seed <seed-from-output> --verbose

# Save failures for analysis
./test_fuzzer.py --save-failures failures.txt

# Test specific failure
./compare_parsers.sh "expression from failures.txt"
```

---

## Best Practices

1. **Run unit tests frequently** during development (they're fast)
2. **Run integration tests** before committing
3. **Run systematic tests** before pushing to main
4. **Run fuzzer periodically** to find edge cases
5. **Add regression tests** when fixing bugs
6. **Keep error tests updated** as parser evolves

---

## Adding New Tests

### Adding Unit Test

```rust
// In tests/parser_tests.rs
#[test]
fn test_my_new_feature() {
    let expr = parse_expr("my syntax").expect("Failed to parse");
    match expr {
        Expr::MyNewType { .. } => {},
        _ => panic!("Expected MyNewType"),
    }
}
```

### Adding Integration Test

```rust
// In tests/comparison_tests.rs
#[test]
fn test_pyret_match_my_feature() {
    assert_matches_pyret("my syntax");
}
```

### Adding to Systematic Suite

```bash
# In test_all_syntax.sh
test_expr "my syntax" "description"
```

### Adding to Fuzzer

```python
# In test_fuzzer.py, add to ExprGenerator class
def _my_new_expr(self, depth: int) -> str:
    """Generate my new expression type"""
    return "generated syntax"
```

---

## Performance

Typical runtimes on development machine:

- Unit tests: ~50ms
- Integration tests: ~5s (includes script overhead)
- Systematic tests: ~2min (150+ comparisons)
- Fuzzer (100 exprs): ~1min
- Error tests: ~100ms

**Total comprehensive test run**: ~4 minutes

---

## Troubleshooting

### "compare_parsers.sh: command not found"

```bash
chmod +x compare_parsers.sh test_all_syntax.sh test_fuzzer.py
```

### "node: command not found"

Install Node.js or ensure Pyret's dependencies are available.

### "Python script fails"

Ensure Python 3 is installed:
```bash
python3 --version
```

### Tests timeout

Increase timeout in scripts or reduce test count:
```bash
./test_fuzzer.py --count 50
```

---

## References

- [PARSER_COMPARISON.md](PARSER_COMPARISON.md) - Comparison tools documentation
- [README.md](README.md) - Project overview
- [NEXT_STEPS.md](NEXT_STEPS.md) - Implementation guide

---

**Last Updated**: 2025-10-31
**Test Coverage**: 300+ tests across 5 test suites
**Status**: All implemented features verified ✅
