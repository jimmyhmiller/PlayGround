# Scripts

This directory contains helper scripts for testing and comparing the Pyret parser.

## Prerequisites

1. **Pyret-lang repository**: You need a local copy of the official Pyret repository.
2. **Node.js**: Required to run the official Pyret parser.
3. **Python 3**: Required for JSON comparison.

## Configuration

The scripts require the `PYRET_REPO` environment variable to be set, pointing to your local copy of the official Pyret repository.

### Setting PYRET_REPO (Required)

Add this to your `~/.bashrc`, `~/.zshrc`, or similar:

```bash
export PYRET_REPO="/path/to/your/pyret-lang"
```

Or set it temporarily for a single command:

```bash
PYRET_REPO="/path/to/your/pyret-lang" ./scripts/compare_parsers.sh "2 + 3"
```

Or for a session:

```bash
export PYRET_REPO="/path/to/your/pyret-lang"
./scripts/compare_parsers.sh "2 + 3"
```

## Scripts

### compare_parsers.sh

Compares a single Pyret expression or file against the official parser.

**Usage:**

```bash
# Compare an expression
./scripts/compare_parsers.sh "2 + 3"
./scripts/compare_parsers.sh "fun f(x): x + 1 end"

# Compare a file
./scripts/compare_parsers.sh path/to/file.arr
```

**Output:**
- ✅ IDENTICAL - if both parsers produce the same AST
- ❌ DIFFERENT - if there are differences (shows a diff)
- ❌ RUST PARSER ERROR - if the Rust parser fails

### compare_all_arr_files.sh

Tests the parser against ALL `.arr` files in the Pyret repository.

**Usage:**

```bash
./scripts/compare_all_arr_files.sh
```

**Features:**
- Caches official Pyret parser output for faster re-runs
- Shows progress every 10 files
- Generates detailed reports in `bulk_test_results/`
- Color-coded output (green=pass, red=fail, yellow=skip)

**Output files:**
- `bulk_test_results/passing_files.txt` - List of files that parse correctly
- `bulk_test_results/failing_files.txt` - List of files that fail
- `bulk_test_results/failure_analysis.txt` - Detailed error analysis

**Cache:**
- Cached ASTs are stored in `$PYRET_REPO/cache/ast-json/`
- Delete the cache directory to force re-parsing with official parser

## Examples

Assuming you have `PYRET_REPO` set:

```bash
# Quick test of a simple expression
./scripts/compare_parsers.sh "1 + 2 + 3"

# Test a full function
./scripts/compare_parsers.sh "fun factorial(n): if n == 0: 1 else: n * factorial(n - 1) end end"

# Test a file from the Pyret test suite
./scripts/compare_parsers.sh "$PYRET_REPO/tests/pyret/tests/test-strings.arr"

# Run full repository comparison (takes a while!)
./scripts/compare_all_arr_files.sh
```

Or without setting it permanently:

```bash
# Set for single command
PYRET_REPO=/path/to/pyret-lang ./scripts/compare_parsers.sh "1 + 2 + 3"

# Set for session
export PYRET_REPO=/path/to/pyret-lang
./scripts/compare_parsers.sh "1 + 2 + 3"
./scripts/compare_all_arr_files.sh
```

## Troubleshooting

### "PYRET_REPO environment variable is not set"

Set the `PYRET_REPO` environment variable:

```bash
export PYRET_REPO="/path/to/your/pyret-lang"
```

Or provide it inline:

```bash
PYRET_REPO=/path/to/pyret-lang ./scripts/compare_parsers.sh "2 + 3"
```

### "does not contain ast-to-json.jarr"

Make sure you're pointing to a valid pyret-lang repository with the AST conversion script.

### Scripts are slow

For `compare_all_arr_files.sh`:
- The first run will be slow as it caches all official parser outputs
- Subsequent runs will be much faster
- You can delete the cache to start fresh: `rm -rf $PYRET_REPO/cache/ast-json/`
