# Ion Examples Testing Infrastructure

## Overview

This directory contains a comprehensive testing infrastructure that compares the Rust implementation against the TypeScript implementation for all ion-examples.

## Files

- **`ion-examples/`** - 37 JSON files with MIR data (2863 function/pass combinations)
- **`generate-all-fixtures.mjs`** - Generates TypeScript SVG fixtures for all examples
- **`generate-test-suite.mjs`** - Generates Rust tests from the manifest
- **`regenerate-all-tests.sh`** - Master script to regenerate everything
- **`tests/fixtures/ion-examples/`** - Generated TypeScript SVG fixtures (2742 files)
- **`tests/fixtures/ion-examples/manifest.json`** - Metadata about all fixtures
- **`tests/ion_examples_comprehensive.rs`** - Comprehensive Rust test suite (2742 tests)

## Quick Start

### Regenerate All Fixtures and Tests

```bash
./regenerate-all-tests.sh
```

This will:
1. Generate TypeScript SVG fixtures for all ion-examples (using ~/Documents/Code/open-source/iongraph2)
2. Create a manifest with metadata about each fixture
3. Generate comprehensive Rust tests that compare Rust output to TypeScript fixtures

### Run All Tests

```bash
# Run all comprehensive tests
cargo test --test ion_examples_comprehensive

# Run with output (to see which tests pass/fail)
cargo test --test ion_examples_comprehensive -- --nocapture

# Run tests for a specific file
cargo test --test ion_examples_comprehensive test_mega_complex

# Run a specific test
cargo test --test ion_examples_comprehensive test_simple_add_func0_pass0
```

### Run Original mega-complex Tests

```bash
cargo test --test mega_complex_comprehensive
```

## Test Coverage

### Statistics

- **Total JSON files**: 37
- **Total function/pass combinations**: 2863
- **Successful fixtures**: 2742 (95.8%)
- **Failed fixtures**: 121 (4.2% - due to TypeScript bugs)
- **Total test cases**: 2742

### Files with Most Tests

1. **mega-complex.json**: 496 tests (15 functions)
2. **ultra-complex.json**: 432 tests (13 functions)
3. **for-of.json**: 133 tests (4 functions)
4. **array-foreach.json**: 127 tests (4 functions)
5. **array-reduce.json**: 112 tests (2 functions)

### Files with Fewest Tests

1. **destructuring.json**: 7 tests (2 functions)
2. **branching.json**: 11 tests (2 functions)
3. **closure.json**: 11 tests (2 functions)
4. **switch.json**: 11 tests (2 functions)

## How It Works

### 1. Fixture Generation (`generate-all-fixtures.mjs`)

For each JSON file in `ion-examples/`:
- Parses the JSON to count functions and passes
- For each function/pass combination:
  - Runs the TypeScript implementation: `node generate-svg-function.mjs <json> <func> <pass> <output>`
  - Saves the SVG to `tests/fixtures/ion-examples/ts-<name>-func<N>-pass<M>.svg`
  - Records metadata in the manifest

### 2. Test Generation (`generate-test-suite.mjs`)

Reads the manifest and generates:
- One Rust test function for each fixture
- Tests are organized by file with comments
- Each test:
  - Loads the TypeScript SVG fixture
  - Parses the JSON and renders with Rust
  - Compares byte-for-byte
  - Reports detailed diffs on failure

### 3. Test Execution

When you run `cargo test --test ion_examples_comprehensive`:
- Each test independently loads its fixture and JSON
- Generates Rust SVG with identical parameters
- Compares outputs character-by-character
- Reports first difference with context on failure

## Interpreting Results

### Passing Tests

```
test test_simple_add_func0_pass0 ... ok
```

This means the Rust implementation generates **byte-for-byte identical** SVG to TypeScript for this function/pass.

### Failing Tests

```
test test_fibonacci_func0_pass5 ... FAILED

=== DIFFERENCE FOUND in fibonacci_func0_pass5 ===
TypeScript lines: 25
Rust lines: 25

First difference at line 2:
TS:   <svg width="379.8" height="198.8" xmlns="http://www.w3.org/2000/svg">
Rust: <svg width="428" height="232" xmlns="http://www.w3.org/2000/svg">
  First char diff at column 12: '3' vs '4'
```

This shows:
- Which test failed
- Total line count (structure might match)
- First differing line
- Character-level difference
- Context lines

## Common Failure Patterns

### 1. Dimension Differences

```
TS:   <svg width="379.8" height="198.8" ...>
Rust: <svg width="428" height="232" ...>
```

**Cause**: Layout algorithm calculates different block sizes/positions
**Fix**: Debug Rust layout calculations to match TypeScript exactly

### 2. Coordinate Differences

```
TS:   <rect x="10.5" y="20.3" ...>
Rust: <rect x="12" y="24" ...>
```

**Cause**: Block positioning or spacing calculations differ
**Fix**: Compare block layout logic between implementations

### 3. Structure Differences

```
TypeScript lines: 25
Rust lines: 18
```

**Cause**: Missing elements or different SVG structure
**Fix**: Check SVG rendering logic for missing/extra elements

## Debugging Failed Tests

### 1. Run Single Test

```bash
cargo test --test ion_examples_comprehensive test_fibonacci_func0_pass5 -- --nocapture
```

### 2. Generate Both Outputs Manually

```bash
# TypeScript (already exists as fixture)
cat tests/fixtures/ion-examples/ts-fibonacci-func0-pass5.svg

# Rust (generate manually)
./target/release/iongraph-rust ion-examples/fibonacci.json 0 5 /tmp/rust-output.svg
cat /tmp/rust-output.svg

# Compare
diff tests/fixtures/ion-examples/ts-fibonacci-func0-pass5.svg /tmp/rust-output.svg
```

### 3. Visual Comparison

Open both SVGs in a browser to see visual differences:

```bash
open tests/fixtures/ion-examples/ts-fibonacci-func0-pass5.svg
open /tmp/rust-output.svg
```

## Known Issues

### TypeScript Implementation Bugs

121 fixtures failed to generate due to this error:
```
TypeError: insEl.classList.contains is not a function
```

These appear in later optimization passes for certain functions. The TypeScript implementation has a bug in the `updateHotness` function.

**Impact**: We can't test those 121 function/pass combinations until the TypeScript bug is fixed.

**Affected files**: Mostly late passes (33-35) of various functions

## Maintenance

### Adding New Examples

1. Add new JSON file to `ion-examples/`
2. Run `./regenerate-all-tests.sh`
3. New tests will automatically be generated

### Updating TypeScript Implementation

If you update the TypeScript implementation:
```bash
cd ~/Documents/Code/open-source/iongraph2
npm install
npm run build
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/iongraph-rust
./regenerate-all-tests.sh
```

### Updating Test Generation Logic

If you modify `generate-test-suite.mjs`:
```bash
node generate-test-suite.mjs  # Regenerate tests only
cargo test --test ion_examples_comprehensive
```

## File Naming Convention

### Fixtures
```
ts-<json-basename>-func<N>-pass<M>.svg
```

Examples:
- `ts-simple-add-func0-pass0.svg` - simple-add.json, function 0, pass 0
- `ts-mega-complex-func5-pass12.svg` - mega-complex.json, function 5, pass 12

### Tests
```
test_<json_basename>_func<N>_pass<M>()
```

Examples:
- `test_simple_add_func0_pass0()` - Tests simple-add.json, function 0, pass 0
- `test_mega_complex_func5_pass12()` - Tests mega-complex.json, function 5, pass 12

## Performance Notes

### Fixture Generation
- **Time**: ~2-5 minutes for all 2742 fixtures
- **Disk space**: ~50-100 MB for all SVG fixtures
- **Bottleneck**: Spawning Node.js processes for each fixture

### Test Execution
- **Time**: Varies by CPU (parallel by default)
- **Memory**: Each test is independent, no shared state
- **Tip**: Use `cargo test -- --test-threads=8` to control parallelism

## Next Steps

1. **Fix Layout Issues**: Debug why Rust produces different dimensions
2. **Analyze Failures**: Run tests and categorize failure types
3. **Fix Implementation**: Update Rust code to match TypeScript exactly
4. **Achieve 100%**: Goal is all 2742 tests passing
