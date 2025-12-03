# Acorn Cache & Test Generation - Implementation Summary

## What Was Built

I've implemented a complete system for automatically caching Acorn AST results and generating JUnit tests from discovered parser failures and mismatches.

### New Components

1. **AcornCacheBuilder.java** - Caches Acorn AST results with metadata
2. **TestGeneratorFromCache.java** - Generates JUnit test classes from cached ASTs
3. **DirectoryTester.java** (modified) - Added `--cache` flag to save results while testing
4. **ACORN_CACHE_TESTS.md** - Complete documentation and usage guide

## How It Works

### Workflow

```
┌─────────────────┐
│ Run Directory   │
│ Tester with     │──┐
│ --cache flag    │  │
└─────────────────┘  │
                     │
        ┌────────────▼────────────┐
        │ Discovers failures &    │
        │ mismatches              │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ Caches Acorn ASTs to    │
        │ test-oracles/adhoc-cache│
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ TestGeneratorFromCache  │
        │ scans cache             │
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ Generates JUnit test    │
        │ classes (Generated*.java)│
        └────────────┬────────────┘
                     │
        ┌────────────▼────────────┐
        │ Run tests to verify     │
        │ parser fixes            │
        └─────────────────────────┘
```

### Cache File Format

New cache files include metadata:

```json
{
  "_metadata": {
    "originalFile": "/path/to/file.js",
    "sourceType": "module",
    "cacheDate": "2025-12-02T23:15:30.123Z",
    "fileHash": "abc123def456..."
  },
  "ast": {
    "type": "Program",
    "body": [...]
  }
}
```

**Backward Compatible:** The test generator also handles old-format cache files (just AST without metadata).

## Quick Start Examples

### Example 1: Cache failures from a project

```bash
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="../my-project --cache --max-failures=10"
```

### Example 2: Generate tests from cache

```bash
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
```

### Example 3: Cache a specific file manually

```bash
mvn exec:java -Dexec.mainClass="com.jsparser.AcornCacheBuilder" \
  -Dexec.args="../my-app/problematic-file.js module"
```

## Current State

### Existing Cache

Your project already has a substantial cache in `test-oracles/adhoc-cache/`:

- **Total cached files:** 20,805
  - `ai-dashboard2/`: 7 files
  - `axios/`: 64 files
  - `simple-nextjs-demo/`: 20,725 files

These were likely created by previous DirectoryTester runs.

### Test Generation

The TestGeneratorFromCache is compatible with both:
- ✅ Old format cache files (just AST)
- ✅ New format cache files (with metadata)

When you run it, it will:
1. Read all 20,805 cached files
2. Categorize them (VSCode, React, NodeModules, etc.)
3. Generate test classes with up to 50 tests each
4. Create `Generated*Test.java` files in `src/test/java/com/jsparser/`

**Note:** With 20,805 cached files, generation may take a few minutes and create many test files. You might want to:
- Clear the cache and start fresh: `rm -rf test-oracles/adhoc-cache/*`
- Or limit the cache by testing smaller directories with `--max-failures` and `--max-mismatches`

## Key Features

### 1. Automatic Test Discovery
No need to manually write tests for every failure - the system does it automatically.

### 2. Real-World Code Coverage
Tests based on actual JavaScript files from real projects, not just synthetic examples.

### 3. Categorized Tests
Tests are organized by source:
- `GeneratedVSCodeTest.java` - VSCode failures
- `GeneratedMonacoEditorTest.java` - Monaco Editor failures
- `GeneratedReactTest.java` - React-related failures
- `GeneratedNodeModulesTest.java` - Generic node_modules failures
- `GeneratedAdHocTest.java` - Other discoveries

### 4. Easy Regeneration
Clear cache and re-run testing on new codebases to discover new issues.

### 5. CI/CD Ready
Can be integrated into build pipelines to continuously test against real-world code.

## Usage Tips

### Start Small

If you're trying this for the first time:

```bash
# Clear existing cache
rm -rf test-oracles/adhoc-cache/*

# Test a small directory
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="../small-project --cache --max-failures=5 --max-mismatches=5"

# Generate tests (will be quick with only 10 cached files)
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"

# Check generated tests
ls -l src/test/java/com/jsparser/Generated*.java

# Run them (they'll be @Disabled initially)
mvn test -Dtest="Generated*Test"
```

### Managing Large Caches

With 20,805 files, you have options:

1. **Generate all tests** (may create 400+ test files):
   ```bash
   mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
   ```

2. **Clear and start fresh with limits**:
   ```bash
   rm -rf test-oracles/adhoc-cache/*
   mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
     -Dexec.args="../target-project --cache --max-failures=20 --max-mismatches=20"
   mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
   ```

3. **Manually select interesting failures**:
   ```bash
   # Keep only specific subdirectories in cache
   cd test-oracles/adhoc-cache
   mv axios /tmp/  # Save axios failures
   rm -rf *        # Clear rest
   mv /tmp/axios . # Restore axios
   cd ../..
   mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
   ```

## Benefits

✅ **Regression Prevention** - Every fixed bug becomes a permanent test
✅ **Real-World Coverage** - Test against actual JavaScript from popular projects
✅ **Time Savings** - Automatic test generation saves hours of manual work
✅ **Organized** - Tests categorized by library/framework
✅ **Flexible** - Easy to regenerate, filter, or customize

## Next Steps

1. **Try it out:**
   ```bash
   # Option A: Use existing cache (generates many tests)
   mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"

   # Option B: Start fresh with a small sample
   rm -rf test-oracles/adhoc-cache/*
   mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
     -Dexec.args="../ai-dashboard2 --cache --max-failures=3 --max-mismatches=3"
   mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
   ```

2. **Review generated tests:**
   ```bash
   ls src/test/java/com/jsparser/Generated*.java
   cat src/test/java/com/jsparser/GeneratedAdHocTest.java
   ```

3. **Enable and fix tests:**
   - Remove `@Disabled` annotation from specific tests
   - Fix the parser bugs they expose
   - Verify tests pass after fixes

4. **Read full documentation:**
   See `ACORN_CACHE_TESTS.md` for complete usage guide and examples

## Files Created/Modified

### New Files
- `src/main/java/com/jsparser/AcornCacheBuilder.java` - AST caching utility
- `src/main/java/com/jsparser/TestGeneratorFromCache.java` - Test generator
- `ACORN_CACHE_TESTS.md` - Full documentation
- `IMPLEMENTATION_SUMMARY.md` - This file

### Modified Files
- `src/main/java/com/jsparser/DirectoryTester.java` - Added `--cache` flag and caching logic

## Example Output

When you run DirectoryTester with caching:

```
Ad-hoc directory testing (real-time comparison)
Directory: /Users/you/code/my-project
Caching enabled: Acorn results will be saved to test-oracles/adhoc-cache

Scanning for JavaScript files...
Found 156 JavaScript files

Testing files (parsing with both Acorn and Java)...
Will stop after 10 Java failures or 10 AST mismatches

Progress: 100/156 (95 matched, 3 mismatched, 2 Java failed, 0 Java too permissive, 0 both failed)

=== Results ===
Total files: 156
  ✓ Both succeeded + matched: 95 (60.90%)
  ⚠ Both succeeded + AST mismatch: 3 (1.92%)
  ✗ Java failed, Acorn succeeded: 2 (1.28%)

=== Cache Summary ===
Cached 5 Acorn AST results to test-oracles/adhoc-cache

To generate JUnit tests from cached results, run:
  mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
```

Then when you generate tests:

```
Scanning cache directory: test-oracles/adhoc-cache
Found 5 cached AST files
Generated: src/test/java/com/jsparser/GeneratedAdHocTest.java (5 tests)

✓ Generated test classes in: src/test/java/com/jsparser
```
