# Focus Areas - Remaining JSON Serialization Issues

**Status:** Parser is 100% complete! All failures are in JSON serialization.

**Location:** All issues are in `src/bin/to_pyret_json.rs`

## Quick Reference

| Minimal Case | Issue | Impact | Status |
|--------------|-------|--------|--------|
| ~~`6/3`~~ | ~~Fraction simplification~~ | ~~6 failures~~ | ✅ **FIXED** |
| `1.5e-300` | Scientific notation negative exponents | ~10 failures | ⏳ TODO |
| `~-6.928203230` | Trailing zero stripping | ~1-2 failures | ⏳ TODO |

## ✅ Issue #1: Fraction Simplification - **FIXED!**

### Minimal Case
```pyret
6/3
```

### Solution Implemented
Removed GCD simplification from `parse_rational()` in `src/parser.rs`. Explicit fractions are now kept unsimplified (e.g., `6/3` stays as `6/3`, not `2/1`).

### Fix Details
- **Location:** `src/parser.rs:2529-2537`
- **Change:** Removed GCD calculation and simplification logic
- **Result:** Fractions are stored as-is, matching Pyret behavior
- **Note:** Decimals converted to fractions (e.g., `2.5` → `5/2`) are still simplified correctly in JSON serialization

### Tests Fixed
- ✅ `test_fraction_simplification_issue` - Now passing!
- ✅ 6 additional comparison tests fixed
- **Progress:** 279/297 passing (93.9%), up from 273/296 (92.2%)

---

## Issue #2: Scientific Notation with Negative Exponents

### Minimal Case
```pyret
1.5e-300
```

### Current Behavior
- **Our output:** `{"type": "s-num", "value": "0"}`
- **Pyret output:** `{"type": "s-num", "value": "3/2000...000"}` (exact fraction with 300+ zeros)

### Problem
Our `expand_scientific_notation()` only handles positive exponents. For negative exponents, we fall back to f64 parsing which underflows to 0 for very small numbers.

### Location
`src/bin/to_pyret_json.rs:68-124` (expand_scientific_notation function)

### Fix Strategy
Extend `expand_scientific_notation()` to handle negative exponents by creating fractions:
- `1.5e-300` → numerator: `15`, denominator: `1000...000` (301 zeros)
- `0.001e-300` → numerator: `1`, denominator: `1000...000` (303 zeros)

**Algorithm:**
```rust
// For negative exponent like 1.5e-300:
// 1. Parse mantissa: 1.5 → numerator = 15, implicit_denominator = 10
// 2. Apply exponent: denominator = 10 * 10^300 = 10^301
// 3. Result: "15" / "10000...000" (301 zeros)
```

### Tests Affected
- `test_full_file_test_within.arr` (~10 occurrences)

---

## Issue #3: Trailing Zero Stripping in Rough Numbers

### Minimal Case
```pyret
~-6.928203230
```

### Current Behavior
- **Our output:** `{"type": "s-num", "value": "~-6.928203230"}`
- **Pyret output:** `{"type": "s-num", "value": "~-6.92820323"}`

### Problem
We preserve the original rough number string exactly, but Pyret strips trailing zeros from decimal parts.

### Location
`src/bin/to_pyret_json.rs:240-295` (rough number normalization)

### Fix Strategy
Add trailing zero stripping for rough number decimals:
```rust
// After stripping leading + and handling .0:
if normalized.contains('.') && !normalized.contains('e') {
    // Strip trailing zeros: "~-6.928203230" → "~-6.92820323"
    normalized = normalized.trim_end_matches('0');
    // Don't strip the decimal point itself
    if normalized.ends_with('.') {
        normalized.push('0'); // Keep at least one zero
    }
}
```

### Tests Affected
- `test_full_file_test_statistics.arr` (~1-2 occurrences)

---

## Testing

Run specific minimal test cases:
```bash
cargo test --test comparison_tests _issue -- --ignored --test-threads=1
```

Run affected full-file tests:
```bash
cargo test --test comparison_tests test_full_file_test_roughnum
cargo test --test comparison_tests test_full_file_test_within
cargo test --test comparison_tests test_full_file_test_statistics
```

Compare minimal cases manually:
```bash
./compare_parsers.sh <(echo '6/3')
./compare_parsers.sh <(echo '1.5e-300')
./compare_parsers.sh <(echo '~-6.928203230')
```

---

## Implementation Order

**Recommended order (easiest to hardest):**

1. ✅ **Issue #1** - Fraction simplification - **COMPLETED!**
   - Removed GCD simplification from `parse_rational()`
   - Test: `test_fraction_simplification_issue` ✅ PASSING

2. **Issue #3** - Trailing zero stripping (15 minutes) - **NEXT**
   - Simple string manipulation
   - Test: `test_full_file_test_statistics`

3. **Issue #2** - Negative exponent expansion (1-2 hours)
   - Most complex: requires arbitrary precision fraction creation
   - Test: `test_full_file_test_within`

---

## Success Criteria

Progress toward completion:
- ✅ `test_fraction_simplification_issue` passes - **DONE!**
- ⏳ `test_scientific_notation_negative_exponent_issue` passes (remove `#[ignore]`)
- ⏳ `test_rough_number_trailing_zeros_issue` passes (remove `#[ignore]`)
- ⏳ Remaining full-file tests pass: test-roughnum, test-within, test-statistics

**Current Progress:**
- ✅ Issue #1 fixed: +6 tests
- Expected remaining: ~12-15 more tests → **Target: 291-294/297 (98%+)**

---

**Last Updated:** 2025-11-08 (evening)
**Current Status:** 279/297 passing (93.9%), 2 ignored, 18 failing
**Previous Status:** 273/296 passing (92.2%), 3 ignored, 23 failing
