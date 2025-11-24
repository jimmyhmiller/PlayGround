# Testing Guide

This document explains how to run the Test262 test suite and analyze the results.

## Prerequisites

- Java 21 or higher
- Maven (or use the included Maven wrapper `./mvnw`)
- Test262 test files in `test-oracles/test262/`

## Running Tests

### Run All Tests

To run the entire test suite including unit tests and Test262 compliance tests:

```bash
./mvnw test
```

### Run Only Test262 Tests

To run just the Test262 compliance test suite:

```bash
./mvnw test -Dtest=Test262Runner
```

This will:
- Scan all JavaScript files in `test-oracles/test262/test/`
- Parse each file using the parser
- Compare the generated AST with the reference AST from ESTree
- Report passing and failing tests

### Run a Single Debug Test

To run a specific test case for debugging:

```bash
./mvnw test -Dtest=Debug262Test
```

This runs the test file specified in `Debug262Test.java`.

### Clean Build and Test

To ensure a fresh build before testing:

```bash
./mvnw clean test
```

## Test Results

### Understanding the Output

The Test262Runner will output:
- Total files scanned (e.g., "Total files scanned: 51350")
- First 20 mismatched files with error messages
- Summary of failure types
- Locations of detailed failure reports

Example output:
```
First 20 mismatched files:
  test-oracles/test262/test/language/asi/S7.9_A5.2_T1.js
  test-oracles/test262/test/language/expressions/yield/rhs-template-middle.js
  ...

✓ Wrote all 59 failures to: /tmp/all_test262_failures.txt
✓ Wrote 59 JSON failures to: /tmp/all_test262_failures.json
```

### Analyzing Failures

After running tests, detailed failure information is written to:

1. **Text report**: `/tmp/all_test262_failures.txt`
   - Line-by-line list of failing test files with error messages
   - Easy to grep and analyze

2. **JSON report**: `/tmp/all_test262_failures.json`
   - Structured failure data for programmatic analysis
   - Contains file paths, error types, and error messages

### Viewing Failure Details

To see all failures:
```bash
cat /tmp/all_test262_failures.txt
```

To count failures:
```bash
cat /tmp/all_test262_failures.txt | wc -l
```

To search for specific error types:
```bash
grep "Expected.*semicolon" /tmp/all_test262_failures.txt
```

To group failures by error type:
```bash
cat /tmp/all_test262_failures.txt | cut -d: -f2- | sort | uniq -c | sort -rn
```

## Calculating Pass Rate

To calculate the current pass rate:

```bash
# Get total files
./mvnw test -Dtest=Test262Runner 2>&1 | grep "Total files scanned"

# Get failure count
cat /tmp/all_test262_failures.txt | wc -l

# Calculate pass rate
# Pass rate = ((total - failures) / total) * 100
```

For example, with 51,350 total files and 59 failures:
- Passing: 51,291
- Pass rate: 99.89%

## Current Test Results

As of the latest run:
- **Total Test Files**: 51,350
- **Passing**: 51,291
- **Failing**: 59
- **Pass Rate**: 99.89%

## Running Specific Test Categories

The Test262Runner automatically scans all test files in the test-oracles directory. To focus on specific categories, you can:

1. Temporarily move test files to a different directory
2. Modify the Test262Runner to filter by path
3. Use the failure reports to identify patterns

## Debugging Failed Tests

To debug a specific failing test:

1. Find the test file path from the failure report
2. Copy the test to `Debug262Test.java`
3. Run: `./mvnw test -Dtest=Debug262Test`
4. The test will output:
   - Source code
   - Generated AST (our parser)
   - Expected AST (oracle/reference)
   - Match result

Example:
```bash
# Find a failing test
grep "asi" /tmp/all_test262_failures.txt | head -1

# Edit Debug262Test.java to point to that test file
# Run the debug test
./mvnw test -Dtest=Debug262Test
```

## Test262 Test Structure

The test files are organized by feature:
- `test-oracles/test262/test/language/` - Core language features
- `test-oracles/test262/test/built-ins/` - Built-in objects and functions
- `test-oracles/test262/test/annexB/` - Annex B (legacy) features
- `test-oracles/test262/test/staging/` - Experimental features

Each test file contains:
- JavaScript source code to parse
- Metadata comments describing the test
- Expected behavior

## Continuous Integration

To integrate with CI/CD:

```bash
# Run tests and fail if there are any failures
./mvnw test
if [ $? -ne 0 ]; then
    echo "Tests failed!"
    exit 1
fi
```

## Performance

Test execution time varies by machine:
- Full test suite: ~10-15 seconds
- Single test: ~100-200ms

The Test262Runner uses caching for better performance on subsequent runs.

## Troubleshooting

### Tests Not Found
If tests fail to find test files:
```bash
# Verify test-oracles directory exists
ls -la test-oracles/test262/test/

# Regenerate test cache if needed
# (Check scripts directory for cache generation scripts)
```

### Out of Memory Errors
If you encounter OOM errors:
```bash
# Increase Maven memory
export MAVEN_OPTS="-Xmx2g"
./mvnw test
```

### Compilation Errors
If the parser fails to compile:
```bash
# Clean and rebuild
./mvnw clean compile
```

## Related Scripts

Additional analysis scripts are available in the `scripts/` directory:
- `analyze-failures.js` - Categorize and analyze failure patterns
- `analyze_failures.sh` - Shell script for failure analysis
- `generate_complete_errors.js` - Generate comprehensive error reports
- `normalize_json_errors.js` - Normalize error messages for comparison

Example usage:
```bash
cd scripts
node analyze-failures.js
```
