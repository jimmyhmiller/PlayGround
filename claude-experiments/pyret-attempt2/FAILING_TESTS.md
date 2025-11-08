# Failing Tests Analysis

**Date:** 2025-11-08
**Status:** 273/296 tests passing (92.2%)
**Failing:** 23 tests (7.8%)

## Summary

After fixing large rational number support, 23 tests remain failing. The issues fall into several categories:

### Categories of Failures

#### 1. Decimal to Fraction Simplification (Most Common)
**Issue:** Official Pyret simplifies decimal fractions, but our parser doesn't.

**Example:**
- Official: `2.034` → `"1017/500"` (simplified fraction)
- Ours: `2.034` → `"20339999999/10000000000"` (full precision)

**Affected Tests:**
- `test_full_file_test_numbers`
- `test_full_file_test_roughnum`
- `test_full_file_test_rounding`
- `test_full_file_test_statistics`
- `test_full_file_test_within`
- `test_full_file_test_bar_chart`
- Possibly others

**Fix Required:** Implement GCD-based fraction simplification in `float_to_fraction_string()` function in `src/bin/to_pyret_json.rs`

#### 2. Scientific Notation Conversion
**Issue:** Our heuristic for when to use scientific notation doesn't match Pyret's.

**Example:**
- Official: `~0.00001` → `"~0.00001"` (preserved)
- Ours: `~0.00001` → `"~1e-5"` (converted to scientific)

**Affected Tests:**
- `test_full_file_test_adaptive_simpson`
- Possibly others

**Fix Required:** Adjust the logic in JSON serialization to better match when Pyret uses scientific notation. The current `> 50 characters` threshold is too simple.

#### 3. Missing AST Fields
**Issue:** Some AST nodes are missing fields that the official parser includes.

**Example:**
- `s-provide-all` is missing `hidden: []` field

**Affected Tests:**
- `test_full_file_test_import_data_from_data_star`

**Fix Required:** Add missing `hidden` field to `SProvideAll` AST node and update serialization.

#### 4. Compiler/Type-Checker Files (Unknown Issues)
**Issue:** Not yet analyzed - may have multiple different issues.

**Affected Tests:**
- `test_full_file_benchmark_adding_ones_2000`
- `test_full_file_benchmark_anf_loop_compiler`
- `test_full_file_compiler_anf_loop_compiler`
- `test_full_file_compiler_ast_util`
- `test_full_file_compiler_compile_lib`
- `test_full_file_compiler_js_dag_utils`
- `test_full_file_compiler_pyret`
- `test_full_file_compiler_resolve_scope`
- `test_full_file_compiler_type_defaults`
- `test_full_file_test_parse_errors`
- `test_full_file_trove_charts`
- `test_full_file_trove_error`
- `test_full_file_trove_matrices`
- `test_full_file_type_check_tests`
- `test_full_file_type_check_tests_because`

**Fix Required:** Individual analysis needed for each file.

## Complete List of Failing Tests

1. `test_full_file_benchmark_adding_ones_2000` - Not analyzed
2. `test_full_file_benchmark_anf_loop_compiler` - Not analyzed
3. `test_full_file_compiler_anf_loop_compiler` - Not analyzed
4. `test_full_file_compiler_ast_util` - Not analyzed
5. `test_full_file_compiler_compile_lib` - Not analyzed
6. `test_full_file_compiler_js_dag_utils` - Not analyzed
7. `test_full_file_compiler_pyret` - Not analyzed
8. `test_full_file_compiler_resolve_scope` - Not analyzed
9. `test_full_file_compiler_type_defaults` - Not analyzed
10. `test_full_file_test_adaptive_simpson` - Scientific notation conversion issue
11. `test_full_file_test_bar_chart` - Likely decimal simplification
12. `test_full_file_test_import_data_from_data_star` - Missing `hidden` field in `s-provide-all`
13. `test_full_file_test_numbers` - Decimal to fraction simplification
14. `test_full_file_test_parse_errors` - Not analyzed
15. `test_full_file_test_roughnum` - Decimal to fraction simplification
16. `test_full_file_test_rounding` - Decimal to fraction simplification
17. `test_full_file_test_statistics` - Decimal to fraction simplification
18. `test_full_file_test_within` - Decimal to fraction simplification
19. `test_full_file_trove_charts` - Not analyzed
20. `test_full_file_trove_error` - Not analyzed
21. `test_full_file_trove_matrices` - Not analyzed
22. `test_full_file_type_check_tests` - Not analyzed
23. `test_full_file_type_check_tests_because` - Not analyzed

## How to Investigate a Failing Test

To see the specific difference for a test:

```bash
# Run the comparison script on the specific file
./compare_parsers.sh tests/pyret-files/full-files/tests/test-numbers.arr

# Or for compiler files:
./compare_parsers.sh tests/pyret-files/full-files/compiler/compile-lib.arr
```

The script will show a diff highlighting exactly what's different between the official parser and our implementation.

## Priority Fixes

1. **High Priority:** Fix decimal to fraction simplification - would fix ~6-7 tests at once
2. **Medium Priority:** Fix scientific notation heuristic - would fix 1-2 tests
3. **Medium Priority:** Add missing `hidden` field to `SProvideAll` - would fix 1 test
4. **Low Priority:** Individually analyze compiler/type-checker files - requires more investigation

## Recent Progress

**2025-11-08:** Fixed large rational number support (i64 → String for arbitrary precision)
- Changed `SFrac` and `SRfrac` AST nodes to use String instead of i64
- Fixed parsing to strip leading `+` signs from numerators
- Added scientific notation conversion for very long decimals (>50 chars)
- Result: 272 → 273 passing tests (+1)
