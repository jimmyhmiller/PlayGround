# Bulk Test Analysis - Complete Feature Breakdown

**Date:** 2025-11-04  
**Total Files Analyzed:** 299

## Overview

All 299 failing files have been analyzed and annotated with the specific missing feature or issue causing the failure.

## Status Breakdown

### ✅ Files That Parse (43 files, 14%)

**However:** None produce identical ASTs to the official parser yet. Issues:

- **20 files** - Underscore handling (`_` parsed as `s-name` instead of `s-underscore`)
- **4 files** - Dot/bang operator handling 
- **19 files** - Other AST differences (need investigation)

### 💥 Crashes (1 file)

- `tools/benchmark/auto-report-programs/adding-ones-2000.arr` - Stack overflow from 2000 deeply-nested binary operations

### ❌ Parser Failures (255 files, 85%)

## Missing Features (Priority Order)

### 1. 🔧 Import/Export Features (171 files, 57%)

**Highest Impact** - Blocks majority of real programs

#### Specific Features Needed:

- **74 files** - `advanced import/provide/type` (mixed/complex cases)
- **11 files** - `type in provide` - `provide: type T, data D end`
- **9 files** - `data in provide` - `provide: data D end`
- **8 files** - `provide block` - `provide { x: x, f: f } end`
- **7 files** - `import from` - `import x, y from file("path")`
- **5 files** - `asterisk in provide` - `provide: *, type T end`
- **5 files** - `provide as` - `provide x as y`
- **2 files** - `import without as` - `import file("x")`
- **1 file** - `module in provide` - `provide: module M end`
- **1 file** - `provide from` - `provide from M: x end`

### 2. 🏁 Prelude Features (48 files, 16%)

**Critical** - Needed for many test and library files

- `#lang pyret` directives
- `provide-types` statements
- Other prelude syntax

### 3. 📝 Type System (32 files, 11%)

**Foundational** - Type aliases used throughout codebase

- **32 files** - Type aliases: `type Name = Type`
- **0 files** - Newtype declarations (not encountered yet)

### 4. 🎯 Runtime Features (7 files, 2%)

**Small Set** - Advanced expression forms

- **3 files** - Lambda block: `lam(...) block: ... end`
- **2 files** - Cases block: `cases(T) x block: ... end`
- **1 file** - Table literal: `table: col row: val end`
- **1 file** - Ask expression: `ask: | cond then: body end`

### 5. ❓ Unknown Features (45 files, 15%)

**Need Investigation** - Requires manual analysis

- **44 files** - Unknown runtime features
- **1 file** - Unknown error type

## AST Mismatch Issues (in files that parse)

### Underscore Handling (20 files)

**Issue:** Parser treats `_` as `s-name` with name="_" instead of `s-underscore`

**Example:**
```pyret
cases(List) x:
  | link(_, _) => ...  # Should be s-underscore, not s-name
end
```

**Files affected:**
- tests/lib-test/lib-test-main.arr
- tests/all.arr
- tests/pyret/tests/test-file.arr
- src/arr/trove/valueskeleton.arr
- src/arr/trove/either.arr
- (+ 15 more)

### Dot/Bang Operator Handling (4 files)

**Issue:** Incorrect AST structure for `.!` operators

**Files affected:**
- tests/pyret/tests/test-each-loop.arr
- ast-to-json-v2.arr
- src/arr/compiler/locators/npm.arr
- src/arr/compiler/locators/jsfile.arr

### Other AST Differences (19 files)

**Issue:** Various structural differences - needs deeper investigation

**Files affected:**
- tests/pyret/tests/test-contracts.arr
- tests/pyret/tests/test-roughnum.arr
- tests/pyret/tests/test-flatness.arr
- (+ 16 more)

## Implementation Priority

Based on impact and difficulty:

1. **Fix underscore handling** (20 files) - Quick fix, medium impact
2. **Implement import/export features** (171 files) - High effort, highest impact
3. **Implement prelude features** (48 files) - Medium effort, high impact  
4. **Implement type aliases** (32 files) - Medium effort, medium impact
5. **Fix dot/bang operators** (4 files) - Low effort, low impact
6. **Investigate "other differences"** (19 files) - Variable effort
7. **Implement runtime features** (7 files) - Low effort, low impact
8. **Investigate unknowns** (45 files) - Variable effort

## Notes

- All annotations are in `bulk_test_results/failing_files.txt`
- Each line format: `filename  # category/issue`
- Categories are specific and actionable
- Stack overflow issue indicates parser recursion needs optimization for deeply nested expressions
