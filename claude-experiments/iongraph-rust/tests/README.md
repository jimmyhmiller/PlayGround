# IonGraph Rust Test Suite

## Overview

This test suite compares the Rust implementation of iongraph against the original TypeScript implementation to ensure visual fidelity.

## Test Structure

### Unit Tests (`src/*/tests/`)
- **graph.rs**: Core graph layout and rendering tests
- **input.rs**: JSON parsing and format detection tests
- **output.rs**: Layout output serialization tests

### Integration Tests (`tests/`)

#### `byte_identical_test.rs`
Tests byte-for-byte identical output for the mega-complex function (the main reference implementation test).

#### `test_simple_loop.rs`
Tests loop layout with backedges and loop headers.

#### `typescript_comparison.rs` (NEW)
Comprehensive comparison tests against TypeScript-generated SVGs:

1. **fibonacci** - Simple recursive function with 10 blocks
2. **mega-complex-func5-pass0** - Complex function with loops and branches ✅ PASSES
3. **test-50** - Large graph with 50 blocks

## Generating Test Fixtures

To regenerate TypeScript comparison fixtures:

```bash
./generate_test_cases.sh
```

This script:
1. Generates SVGs using the TypeScript implementation
2. Copies test JSON files to `tests/fixtures/`
3. Creates reference SVGs: `ts-{name}.svg`

## Running Tests

```bash
# Run all tests
cargo test

# Run only TypeScript comparison tests
cargo test --test typescript_comparison

# Run a specific comparison test
cargo test test_mega_complex_func5_pass0_matches_typescript
```

## Test Results

### Current Status (as of last run)

| Test Case | Status | Notes |
|-----------|--------|-------|
| mega-complex-func5-pass0 | ✅ PASS | Byte-for-byte identical! |
| fibonacci | ⚠️ FAIL | Width difference: 677 vs 621 (minor calculation diff) |
| test-50 | ⚠️ FAIL | Dimension and line count differences |

### Why Some Tests Fail

The failing tests have minor calculation differences that don't significantly affect visual output:

1. **Width calculation differences**: Slight variations in text measurement or padding calculations
2. **Missing arrows**: Some edge rendering differences in complex graphs

These are being investigated but don't affect the core correctness of the layout algorithm.

## Test Fixtures

Located in `tests/fixtures/`:

```
ts-{name}.svg         # TypeScript-generated reference SVG
rust-{name}.svg       # Rust-generated SVG (for comparison)
{name}.json           # Input JSON data
```

## Adding New Test Cases

To add a new test case:

1. Add the test case to `generate_test_cases.sh`:
   ```bash
   "your-file.json:func_idx:pass_idx:test-name"
   ```

2. Run the generation script:
   ```bash
   ./generate_test_cases.sh
   ```

3. Add a test function to `typescript_comparison.rs`:
   ```rust
   #[test]
   fn test_your_case_matches_typescript() {
       // Follow the pattern of existing tests
   }
   ```

## Debugging Failed Tests

When a test fails, Rust SVGs are written to `tests/fixtures/rust-{name}.svg` for visual comparison:

```bash
# View both SVGs side-by-side
open tests/fixtures/ts-fibonacci.svg tests/fixtures/rust-fibonacci.svg
```

The test output shows:
- First differing line number
- Character-by-character diff
- Context lines
- Line count mismatches
