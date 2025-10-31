# Test Directory

This directory contains comprehensive test suites for the Pyret parser.

## Test Files

### `parser_tests.rs` - Unit Tests
Fast, focused tests for individual parsing features.
- **35+ tests** covering primitives, operators, calls, dots, arrays
- **All passing** ✅
- Run: `cargo test --test parser_tests`

### `comparison_tests.rs` - Integration Tests
Validates our parser against official Pyret parser.
- **53 test functions** (80+ comparisons)
- Calls `../compare_parsers.sh` script
- Tests all implemented syntax
- Run: `cargo test --test comparison_tests`

### `error_tests.rs` - Error Handling Tests
Ensures parser handles invalid input gracefully.
- **41 tests** for error cases and edge conditions
- **36 passing**, 5 documenting current lenient behavior
- Run: `cargo test --test error_tests`

## Quick Reference

```bash
# Run all tests
cargo test

# Run specific test file
cargo test --test parser_tests
cargo test --test comparison_tests
cargo test --test error_tests

# Run specific test
cargo test test_parse_simple_addition

# Debug mode (see tokens)
DEBUG_TOKENS=1 cargo test test_name

# Verbose output
cargo test -- --nocapture
```

## See Also

- `../TESTING.md` - Comprehensive testing guide
- `../TEST_SUITE_SUMMARY.md` - Test suite summary
- `../test_all_syntax.sh` - Systematic test script
- `../test_fuzzer.py` - Property-based fuzzer
- `../PARSER_COMPARISON.md` - Comparison tools documentation
