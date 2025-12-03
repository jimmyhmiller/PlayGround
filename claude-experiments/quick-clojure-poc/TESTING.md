# Testing Infrastructure

This document describes the testing setup for the Clojure JIT compiler.

## Overview

We have two types of tests:

1. **Unit tests** (`cargo test --lib`) - 29 tests for internal components
2. **Clojure compatibility tests** (`cargo test --test clojure_compatibility`) - 26 integration tests

## Clojure Compatibility Tests

The integration tests verify that our JIT compiler produces **exactly** the same output as Clojure 1.11.1.

### How It Works

Each test:
1. Writes a Clojure expression to a temporary file using `tempfile` crate
2. Executes it through our JIT compiler: `cargo run --release --quiet <file>`
3. Executes the same expression through Clojure: `clj -e "<expr>"`
4. Compares the outputs character-by-character
5. Fails with a clear diff if they don't match

### Running the Tests

```bash
# Run all compatibility tests (recommended)
cargo test --test clojure_compatibility

# Run a specific test
cargo test --test clojure_compatibility test_addition

# Run with verbose output
cargo test --test clojure_compatibility -- --nocapture

# Run all tests (unit + integration)
cargo test
```

### Test Coverage

**26 tests covering:**

- Literals (nil, true, false, integers, negative numbers)
- Equality (nil ≠ false ≠ 0)
- Comparisons (<, >)
- Arithmetic (+, -, *)
- Let expressions (empty bodies, bindings, nesting, shadowing)

**Not yet covered (commented out):**

- Boolean logic (and, or)
- If expressions
- More complex features

### Output Format Compatibility

Our implementation matches Clojure's output exactly:

| Value | Output |
|-------|--------|
| `nil` | (nothing - empty) |
| `true` | `true` |
| `false` | `false` |
| Integers | Untagged value (e.g., `42`) |

This is implemented in `src/main.rs::print_tagged_value()`.

## Adding New Tests

To add a new compatibility test:

1. Edit `tests/clojure_compatibility.rs`
2. Add a new test function:

```rust
#[test]
fn test_my_feature() {
    assert_matches_clojure("(my-clojure-expression)");
}
```

3. Run the test:

```bash
cargo test --test clojure_compatibility test_my_feature
```

The helper function `assert_matches_clojure()` handles all the complexity of running both implementations and comparing outputs.

## Thread Safety

Tests run in parallel by default. Each test creates its own temporary file using Rust's `tempfile` crate, which guarantees:

- Unique filenames (no collisions)
- Automatic cleanup on test completion
- Works across different platforms

This eliminates race conditions and makes tests reliable.

## CI Integration

The tests integrate seamlessly with CI systems:

```yaml
# Example GitHub Actions
- name: Run tests
  run: cargo test --test clojure_compatibility
```

Exit codes:
- `0` = All tests passed
- `101` = One or more tests failed

## Test Results

✅ All 26 Clojure compatibility tests pass
✅ All 29 unit tests pass
✅ Tests run reliably in parallel
✅ Output matches Clojure exactly

## Dependencies

The compatibility tests require:

1. **Rust/Cargo** - For building and running our implementation
2. **Clojure CLI** (`clj` command) - For comparison testing
3. **tempfile** crate - For safe temporary file handling (dev dependency)

To install Clojure CLI:
```bash
# macOS
brew install clojure/tools/clojure

# Linux
curl -O https://download.clojure.org/install/linux-install-1.11.1.1403.sh
chmod +x linux-install-1.11.1.1403.sh
sudo ./linux-install-1.11.1.1403.sh
```
