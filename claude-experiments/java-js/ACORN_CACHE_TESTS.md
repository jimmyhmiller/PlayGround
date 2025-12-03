# Acorn Cache-Based Test Generation

This document explains how to automatically generate JUnit tests from discovered parser failures and AST mismatches.

## Overview

When testing the Java parser against large codebases, you'll discover files that:
1. **Fail to parse** - Acorn succeeds but Java parser fails (parser bugs)
2. **Have AST mismatches** - Both parse successfully but produce different ASTs

Instead of manually creating tests for each failure, this system automatically:
- Caches Acorn's AST results for problematic files
- Generates JUnit test classes from the cache
- Creates ready-to-run regression tests

## Quick Start

### 1. Run DirectoryTester with caching enabled

Test a directory and cache failures/mismatches:

```bash
# Test a project and cache all failures + mismatches
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="../my-project --cache"

# Limit failures and mismatches for faster testing
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="../my-project --cache --max-failures=20 --max-mismatches=20"
```

This will:
- Parse all JavaScript files with both Acorn and the Java parser
- Cache Acorn AST results for any failures or mismatches to `test-oracles/adhoc-cache/`
- Print a summary showing how many results were cached

### 2. Generate JUnit tests from cache

```bash
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
```

This will:
- Scan the cache directory (`test-oracles/adhoc-cache/`)
- Group cached files by category (VSCode, React, MonacoEditor, etc.)
- Generate test classes in `src/test/java/com/jsparser/`
- Create `Generated*Test.java` files with up to 50 tests each

### 3. Run the generated tests

```bash
# Run all tests including generated ones
mvn test

# Run only generated tests
mvn test -Dtest="Generated*Test"
```

**Note:** Generated tests are initially `@Disabled` so they won't fail your build. Enable them when you're ready to fix the bugs.

## Usage Examples

### Example 1: Testing VSCode installation

```bash
# Test VSCode and cache first 10 failures
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="$HOME/.vscode/extensions --cache --max-failures=10"

# Generate tests
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"

# Review generated test file
cat src/test/java/com/jsparser/GeneratedVSCodeTest.java
```

### Example 2: Testing a React project

```bash
# Test React project and cache all mismatches
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="../my-react-app --cache --max-mismatches=50"

# Generate tests
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
```

### Example 3: Testing node_modules

```bash
# Test specific package in node_modules
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="../my-project/node_modules/monaco-editor --cache --max-failures=5"

# Generate tests
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
```

## Generated Test Structure

Generated test classes follow this pattern:

```java
@Disabled("Auto-generated tests - enable when ready to fix")
public class GeneratedMonacoEditorTest {

    @Test
    @DisplayName("node_modules/monaco-editor/esm/vs/editor/browser/viewParts/lineNumbers/lineNumbers.js")
    void test_monaco_editor_lineNumbers_0() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/..._monaco-editor_..._lineNumbers.js.json",
            "node_modules/monaco-editor/esm/vs/editor/browser/viewParts/lineNumbers/lineNumbers.js",
            true
        );
    }

    // Helper methods for AST comparison...
}
```

## Cache File Format

Cached AST files are stored as JSON with metadata:

```json
{
  "_metadata": {
    "originalFile": "/path/to/original/file.js",
    "sourceType": "module",
    "cacheDate": "2025-12-02T23:15:30.123Z",
    "fileHash": "abc123..."
  },
  "ast": {
    "type": "Program",
    "body": [...],
    ...
  }
}
```

## Workflow

### Development Workflow

1. **Discover issues:** Run DirectoryTester with `--cache` on a large codebase
2. **Generate tests:** Convert cached results into JUnit tests
3. **Fix bugs:** Enable specific tests and fix parser bugs
4. **Verify:** Re-run tests to confirm fixes
5. **Cleanup:** Delete fixed tests or move them to permanent test suites

### Continuous Testing Workflow

```bash
#!/bin/bash
# Script to continuously test against real-world code

# Test multiple projects
for project in ../project1 ../project2 ../project3; do
  echo "Testing $project..."
  mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
    -Dexec.args="$project --cache --max-failures=10"
done

# Generate tests from all discoveries
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"

# Run tests to see current state
mvn test -Dtest="Generated*Test"
```

## Tools Reference

### DirectoryTester

**Purpose:** Test directory of JS files against Acorn (oracle parser)

**Arguments:**
- `<directory>` - Directory to scan for JavaScript files
- `--cache` - Enable caching of Acorn results for failures/mismatches
- `--max-failures=N` - Stop after N Java parser failures
- `--max-mismatches=N` - Stop after N AST mismatches
- `--max-too-permissive=N` - Stop after N cases where Java is too permissive

**Example:**
```bash
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="../myapp --cache --max-failures=20 --max-mismatches=20"
```

### TestGeneratorFromCache

**Purpose:** Generate JUnit tests from cached Acorn results

**Arguments:**
- Optional: `<cache-dir>` - Cache directory (default: test-oracles/adhoc-cache)
- Optional: `<output-dir>` - Test output directory (default: src/test/java/com/jsparser)

**Example:**
```bash
# Use defaults
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"

# Custom directories
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache" \
  -Dexec.args="custom-cache custom-tests"
```

### AcornCacheBuilder

**Purpose:** Manually cache a single file's Acorn AST

**Arguments:**
- `<file-path>` - JavaScript file to cache
- Optional: `[sourceType]` - 'script' or 'module' (auto-detected if omitted)

**Example:**
```bash
# Auto-detect source type
mvn exec:java -Dexec.mainClass="com.jsparser.AcornCacheBuilder" \
  -Dexec.args="../myapp/src/index.js"

# Force module mode
mvn exec:java -Dexec.mainClass="com.jsparser.AcornCacheBuilder" \
  -Dexec.args="../myapp/src/index.js module"
```

## Tips

### Managing Generated Tests

1. **Review before committing:** Check generated tests make sense
2. **Enable selectively:** Remove `@Disabled` from specific tests to work on them
3. **Categorize:** Tests are auto-categorized by source (VSCode, React, etc.)
4. **Clean up:** Delete or archive test classes for fixed issues

### Cache Management

```bash
# View cache contents
ls -lah test-oracles/adhoc-cache/

# Clear cache
rm -rf test-oracles/adhoc-cache/*

# Count cached files
find test-oracles/adhoc-cache -name "*.json" | wc -l
```

### Finding Specific Issues

```bash
# Find all failures related to class syntax
mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
  -Dexec.args="../projects-with-classes --cache --max-failures=100" \
  | grep "class body"

# Test only .mjs files (modules)
# (Requires modifying DirectoryTester to filter by extension)
```

## Troubleshooting

### "No cache files found"

Run DirectoryTester with `--cache` first to populate the cache.

### "File not found" in generated tests

The original source file was moved or deleted. Either:
- Re-run DirectoryTester to refresh cache
- Or skip/delete that test

### Tests fail even though ASTs match

Check normalization logic in test helpers - may need to normalize additional fields.

### Too many tests generated

Use `--max-failures` and `--max-mismatches` to limit discoveries, or manually delete unwanted cache files before generating tests.

## Advanced Usage

### Custom Test Organization

Edit `TestGeneratorFromCache.determineCategory()` to customize how tests are grouped:

```java
private String determineCategory(CacheEntry entry) {
    String path = entry.originalFile.toLowerCase();
    if (path.contains("my-special-lib")) {
        return "MySpecialLib";
    }
    // ... existing logic
}
```

### Batch Processing Multiple Projects

```bash
#!/bin/bash
# test-all-projects.sh

PROJECTS=(
  "../project-a"
  "../project-b"
  "../project-c"
)

for proj in "${PROJECTS[@]}"; do
  echo "=== Testing $proj ==="
  mvn exec:java -Dexec.mainClass="com.jsparser.DirectoryTester" \
    -Dexec.args="$proj --cache --max-failures=10 --max-mismatches=10"
done

echo "=== Generating tests ==="
mvn exec:java -Dexec.mainClass="com.jsparser.TestGeneratorFromCache"
```

### Comparing Cache Over Time

```bash
# Snapshot current cache
cp -r test-oracles/adhoc-cache test-oracles/adhoc-cache-$(date +%Y%m%d)

# After fixes, compare
diff -r test-oracles/adhoc-cache-20250101 test-oracles/adhoc-cache
```

## Benefits

✅ **Automatic test discovery** - No manual test writing for discovered issues
✅ **Real-world code coverage** - Tests based on actual JavaScript in the wild
✅ **Regression prevention** - Cached tests ensure fixes stay fixed
✅ **Organized by category** - Tests grouped by framework/library
✅ **Easy to update** - Re-cache and regenerate as needed
✅ **CI/CD ready** - Can integrate into build pipelines

## See Also

- `Test262Runner.java` - For spec-based testing
- `AwaitIdentifierMismatchTest.java` - Example of hand-written cache-based test
- `DirectoryTester.java` - Main ad-hoc testing tool
