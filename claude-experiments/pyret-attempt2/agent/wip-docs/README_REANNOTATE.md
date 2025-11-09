# How to Re-Annotate After Making Parser Changes

## Quick Start

After making changes to the parser, run:

```bash
./reannotate.sh
```

This takes about 2-5 minutes and will update `bulk_test_results/failing_files.txt` with fresh annotations.

## What It Does

The script performs three steps:

### Step 1: Re-test All Files
- Tests all 299 files with your current parser
- Captures parse errors or success status
- Writes results to `bulk_test_results/failure_analysis.txt`

### Step 2: Categorize Parse Errors  
- Analyzes error messages to identify missing features
- Categories include:
  - `import from` - `import x, y from file(...)`
  - `provide block` - `provide { x: x } end`
  - `type alias` - `type Name = Type`
  - `cases block` - `cases(T) x block: ... end`
  - etc.

### Step 3: Check AST Differences
- For files that parse successfully, compares AST with official parser
- Categories include:
  - `✅ MATCHES official parser` - Perfect!
  - `🔸 underscore` - `_` handling issue
  - `🔸 dot/bang operator` - `.!` operator issue
  - `❌ other AST difference` - Needs investigation

## After Re-Annotation

View the summary:
```bash
python3 print_summary.py
```

Check specific categories:
```bash
# See all files with underscore issues
grep "underscore" bulk_test_results/failing_files.txt

# See all files with type alias issues
grep "type alias" bulk_test_results/failing_files.txt

# Count how many files now match
grep "MATCHES" bulk_test_results/failing_files.txt | wc -l
```

## Manual Scripts

If you need more control, you can run individual steps:

### Just re-test files:
```bash
# Run analyzer manually
./analyze_failures.sh
```

### Just categorize existing results:
```bash
# Re-categorize without re-testing
python3 final_accurate_categorize.py
python3 categorize_mismatches.py
```

### Test a single file:
```bash
# See if it parses
cargo run --bin to_pyret_json /path/to/file.arr

# Compare with official parser
./compare_parsers.sh /path/to/file.arr
```

## Workflow Example

```bash
# 1. Make parser changes
vim src/parser.rs

# 2. Re-annotate
./reannotate.sh

# 3. Check what improved
python3 print_summary.py

# 4. See which files now match
grep "MATCHES" bulk_test_results/failing_files.txt
```

## Files Involved

- `reannotate.sh` - Main re-annotation script
- `final_accurate_categorize.py` - Categorizes parse errors and tests files
- `categorize_mismatches.py` - Categorizes AST differences
- `print_summary.py` - Displays summary statistics
- `bulk_test_results/failing_files.txt` - Annotated file list (updated by scripts)
- `bulk_test_results/failure_analysis.txt` - Raw test results (regenerated each run)

## Troubleshooting

**Script is slow:**
- It tests 299 files, so 2-5 minutes is normal
- Most time is spent running the parser and official parser

**"File not found" errors:**
- Make sure `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang` exists
- Check that the pyret-lang repository is fully cloned

**Python errors:**
- Scripts require Python 3.6+
- No external dependencies needed

**Want to test a subset:**
```bash
# Create a test file with just a few paths
head -20 bulk_test_results/failing_files.txt > test_subset.txt

# Modify reannotate.sh to read from test_subset.txt instead
```
