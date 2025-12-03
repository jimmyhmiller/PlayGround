# Clojure Compatibility Test Suite

This directory contains automated tests that verify our JIT compiler's output matches Clojure's behavior.

## Running Tests

### Run all Clojure compatibility tests (recommended):
```bash
cargo test --test clojure_compatibility
```

### Run a specific test:
```bash
cargo test --test clojure_compatibility test_addition
```

### Run all tests including unit tests:
```bash
cargo test
```

### Legacy shell script (deprecated):
```bash
./tests/compare_with_clojure.sh tests/basic_compatibility.clj
```

## Test Implementation

Tests are written as Rust integration tests in `tests/clojure_compatibility.rs`. Each test:

1. Runs an expression through our JIT compiler (via `cargo run`)
2. Runs the same expression through Clojure (via `clj`)
3. Compares the outputs and asserts they match

The tests use `tempfile` crate to create unique temporary files for each test, allowing safe parallel execution.

## Output Matching

Our implementation now matches Clojure's output format exactly:

| Value | Output |
|-------|--------|
| `nil` | (nothing) |
| `true` | `true` |
| `false` | `false` |
| Integers | Untagged number (e.g., `42`) |

### Example:
```bash
$ echo "nil" | cargo run --release --quiet
$ # (no output - matches Clojure)

$ echo "true" | cargo run --release --quiet
true

$ echo "(+ 1 2)" | cargo run --release --quiet
3
```

## Current Test Coverage (26 tests)

- **Literals**: nil, true, false, integers (including negative)
- **Equality**: Proper distinction between nil/false/0
- **Comparisons**: `<`, `>` (both true and false cases)
- **Arithmetic**: `+`, `-`, `*` (various combinations)
- **Let expressions**: Empty bodies, bindings, nested lets, shadowing

## Adding New Tests

To add new compatibility tests, edit `tests/clojure_compatibility.rs`:

```rust
#[test]
fn test_my_feature() {
    assert_matches_clojure("(my-expr)");
}
```

The `assert_matches_clojure` helper will:
- Run the expression in both implementations
- Compare outputs
- Show a clear diff on failure

## Integration with CI

The tests integrate seamlessly with Cargo's test framework:

```bash
cargo test --test clojure_compatibility
```

Exit code 0 = all tests pass
Exit code 101 = one or more tests failed

## Test Results

✅ **All 26 tests pass!**

The tests run in parallel safely using proper temporary file handling, and consistently verify that our implementation matches Clojure's behavior.
