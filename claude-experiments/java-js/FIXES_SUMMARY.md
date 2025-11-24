# Summary of Fixes

## Issues Fixed

### 1. Numeric Overflow/Precision Issue in Lexer

**Problem**: The lexer was incorrectly handling decimal numbers larger than `Long.MAX_VALUE`. When parsing a number like `9223372036854776000` (which is larger than `Long.MAX_VALUE` = `9223372036854775807`), the parser would:

1. Parse the string as a double: `9.223372036854776E18`
2. Cast it to long: `(long) 9.223372036854776E18` → `9223372036854775807`
3. Cast it back to double: `(double) 9223372036854775807` → `9.223372036854776E18`
4. Compare the values and incorrectly conclude no precision was lost
5. Return the clamped long value instead of the double

This was caused by IEEE-754 double precision limitations. The double `9.223372036854776E18` and `(double)Long.MAX_VALUE` are **exactly the same value** in IEEE-754 representation (same bit pattern: `43e0000000000000`), so the round-trip conversion check failed to detect the overflow.

**Solution**: Changed `Lexer.java` line 767 to parse the original string directly as a long using `Long.parseLong()` instead of casting a double to long. This avoids all floating point precision issues:

```java
// Before (buggy):
if (doubleValue == Math.floor(doubleValue) && !Double.isInfinite(doubleValue)) {
    long longValue = (long) doubleValue;  // Clamps to Long.MAX_VALUE!
    if ((double) longValue == doubleValue) {  // False positive due to FP precision
        ...
    }
}

// After (fixed):
if (doubleValue == Math.floor(doubleValue) && !Double.isInfinite(doubleValue)) {
    try {
        long longValue = Long.parseLong(numberStr);  // Direct parse - throws on overflow
        ...
    } catch (NumberFormatException e) {
        // Number is outside long range, keep as double
        literal = doubleValue;
    }
}
```

**Files Changed**:
- `src/main/java/com/jsparser/Lexer.java` (lines 761-781)

**Tests Added**:
- `src/test/java/com/jsparser/NumericOverflowTest.java` - Comprehensive test for numeric overflow handling

### 2. Numeric Type Normalization in Test Comparison

**Problem**: When comparing ASTs, Jackson deserializes JSON numbers into different Java types depending on their size:
- Small numbers → `Integer`
- Medium numbers → `Long`
- Large numbers → `BigInteger`

This caused AST comparisons to fail even when the actual numeric values were identical, because Java's `Objects.deepEquals()` considers `Long(123)` and `BigInteger(123)` to be different objects.

**Solution**: Added a `normalizeNumericValues()` method in `AdhocPackageTest.java` that normalizes all numbers in the actual AST to match the expected types before comparison. This allows semantically equivalent ASTs with different numeric type representations to be considered equal.

**Files Changed**:
- `src/test/java/com/jsparser/AdhocPackageTest.java` (lines 143, 274-317)
- `src/test/java/com/jsparser/AstDiffTest.java` (lines 73, 198-241)

## Impact

Before fixes:
- **Parse failures**: 2 files (0.01%)
- **AST mismatches**: 372 files (1.80%)
- **Total issues**: 374 files (1.81%)

Expected after fixes:
- **Parse failures**: 0 files (0%)
- **AST mismatches**: Significantly reduced (most were due to numeric type mismatches)
- **Exact matches**: ~100%

## Related Files

### Modified:
1. `src/main/java/com/jsparser/Lexer.java` - Fixed numeric overflow handling
2. `src/test/java/com/jsparser/AdhocPackageTest.java` - Added numeric normalization for comparisons
3. `src/test/java/com/jsparser/AstDiffTest.java` - Added numeric normalization for debug tool

### Added:
1. `src/test/java/com/jsparser/NumericOverflowTest.java` - Tests for numeric overflow edge cases
2. `scripts/compare-ast.js` - Script to compare AST differences (created for debugging)

## Testing

Run the full test suite to verify all fixes:

```bash
mvn test -Dtest=AdhocPackageTest -DpackageName=simple-nextjs-demo
```

Run the specific numeric overflow tests:

```bash
mvn test -Dtest=NumericOverflowTest
```

## Technical Notes

### IEEE-754 Double Precision

JavaScript numbers are IEEE-754 doubles, which can only represent integers exactly up to 2^53 (9,007,199,254,740,992). Beyond this, there are gaps between representable values.

Key insights:
- `Long.MAX_VALUE` = 9,223,372,036,854,775,807 (2^63 - 1)
- The nearest IEEE-754 double to this is 9,223,372,036,854,776,000
- This same double value is the nearest representation for several integer values near Long.MAX_VALUE
- Casting a double to long in Java clamps values outside the range rather than throwing an exception
- Direct string parsing with `Long.parseLong()` is the only reliable way to detect overflow

### Comparison Strategy

The numeric normalization approach allows the test framework to:
1. Accept ASTs with different Java numeric types (Integer vs Long vs BigInteger)
2. Compare them semantically rather than structurally
3. Maintain correctness by preserving actual numeric values
4. Match Acorn's behavior which uses JavaScript Numbers (all are doubles internally)
