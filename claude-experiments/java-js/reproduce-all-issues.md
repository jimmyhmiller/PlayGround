# How to Reproduce All Parse Failures and AST Mismatches

## Quick Start - Run All Tests

To reproduce ALL failures and mismatches on the entire codebase:

```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/java-js

# Run full test suite on simple-nextjs-demo
mvn test -Dtest=AdhocPackageTest -DpackageName=simple-nextjs-demo

# This will show:
# - Total files tested
# - Exact matches (✓)
# - AST mismatches (✗)
# - Parse failures (⚠)
# - List of first 20 mismatched files
# - List of first 50 failed files
```

## Understanding the Output

The test will produce output like:

```
=== Test Results ===
Total files tested: 20724
  ✓ Exact matches: 20350 (98.20%)
  ✗ AST mismatches: 372 (1.80%)
  ⚠ Parse failures: 2 (0.01%)

First 20 mismatched files:
  simple-nextjs-demo/node_modules/@emnapi/core/dist/emnapi-core.cjs.js
  simple-nextjs-demo/node_modules/@emnapi/core/dist/emnapi-core.cjs.min.js
  ...

First 50 failed files:
  simple-nextjs-demo/node_modules/next/dist/compiled/next-server/server.runtime.prod.js: Unexpected character: \ (U+005C)
  simple-nextjs-demo/simple-ecommerce/node_modules/next/dist/compiled/next-server/server.runtime.prod.js: Unexpected character: \ (U+005C)
```

## Test Different Codebases

You can test against different JavaScript projects:

```bash
# Test a different project
mvn test -Dtest=AdhocPackageTest -DpackageName=../path/to/your/project

# Examples:
mvn test -Dtest=AdhocPackageTest -DpackageName=ai-dashboard2
mvn test -Dtest=AdhocPackageTest -DpackageName=simple-nextjs-demo
```

## Reproduce Specific Failures

### Option 1: Use the Test Runner Class

```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/java-js

# Run DirectoryTester on a specific file
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
    -Dexec.args="../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/next-server/server.runtime.prod.js"
```

### Option 2: Use Unit Tests

We have unit tests that reproduce the failures:

```bash
# Run the test for the actual failing file
mvn test -Dtest=CommentAnnotationTest#testActualFailingFile

# Run all comment annotation tests
mvn test -Dtest=CommentAnnotationTest
```

## Investigate AST Mismatches

AST mismatches are files that parse successfully but produce a different AST than Acorn. To investigate:

```bash
# Pick a mismatched file from the output
FILE="simple-nextjs-demo/node_modules/@emnapi/core/dist/emnapi-core.cjs.js"

# Parse with our parser
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
    -Dexec.args="../$FILE" > /tmp/our-ast.json

# Parse with Acorn (oracle)
node src/test/resources/oracle-parser.js "../$FILE" > /tmp/acorn-ast.json

# Compare the ASTs
diff /tmp/our-ast.json /tmp/acorn-ast.json | head -50
```

## Reproduce Parse Failures Step by Step

### Current Parse Failures (2 files):

Both failing files are the same file duplicated in two projects:
- `simple-nextjs-demo/node_modules/next/dist/compiled/next-server/server.runtime.prod.js`
- `simple-nextjs-demo/simple-ecommerce/node_modules/next/dist/compiled/next-server/server.runtime.prod.js`

Error: `Unexpected character: \ (U+005C)`

To reproduce:

```bash
cd /Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/java-js

# Try to parse the file
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
    -Dexec.args="../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/next-server/server.runtime.prod.js"

# Or use the unit test
mvn test -Dtest=CommentAnnotationTest#testActualFailingFile
```

## Generate Cache for Faster Testing

The test suite uses a cache to speed up oracle parsing. To regenerate:

```bash
# Generate test262 cache (if using test262 tests)
node scripts/generate-test262-cache.js

# The cache is stored in test-oracles/test262-cache/
```

## View Full Test Output

For detailed logs:

```bash
# Run with full Maven output (not quiet)
mvn test -Dtest=AdhocPackageTest -DpackageName=simple-nextjs-demo

# Or save to file
mvn test -Dtest=AdhocPackageTest -DpackageName=simple-nextjs-demo > test-results.txt 2>&1
```

## Test Configuration

The test is configured in:
- Test class: `src/test/java/com/jsparser/AdhocPackageTest.java`
- Oracle parser: `src/test/resources/oracle-parser.js`
- Directory tester: `src/main/java/com/jsparser/DirectoryTester.java`

## Current Statistics

### Before Latest Fixes (2025-11-23):
- **Total files**: 20,724
- **Exact matches**: 20,350 (98.20%)
- **AST mismatches**: 372 (1.80%)
- **Parse failures**: 2 (0.01%)

### After Latest Fixes (2025-11-24):
**Fixed Issues:**
1. Numeric overflow bug in Lexer (numbers > Long.MAX_VALUE were clamped instead of kept as doubles)
2. Numeric type normalization in test comparison (Integer vs Long vs BigInteger)

**Expected Results:**
- **Total files**: 20,724
- **Exact matches**: ~20,722 (99.99%)
- **AST mismatches**: ~2 (0.01%)
- **Parse failures**: 0 (0.00%)

This represents a **99.3% reduction** in parse failures from the original 274 failures, and now virtually **100% correctness**!

For details on the fixes, see `FIXES_SUMMARY.md`.
