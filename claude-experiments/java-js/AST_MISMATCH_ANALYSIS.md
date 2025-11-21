# AST Mismatch Analysis

## Summary
Out of 37,512 Test262 files with cached oracles:
- ✓ **30,458 exact matches (81.20%)**
- ✗ **7,054 AST mismatches (18.80%)**
- ⚠ **0 parse failures (0.00%)**

## Main Categories of Mismatches

### 1. BigInt Literal Support (HIGH PRIORITY)
**Issue**: Literal nodes for BigInt values are missing the `bigint` field and have `value: null` instead of the bigint value.

**Examples**:
- `1n` - BigInt literal
- `1234567890123456789n` - Large BigInt

**Current AST**:
```json
{
  "type": "Literal",
  "value": null,
  "raw": "1n"
}
```

**Expected AST**:
```json
{
  "type": "Literal",
  "value": null,
  "raw": "1n",
  "bigint": "1"
}
```

**Affected Files**: ~50+ files based on first 10 mismatches
- harness/compare-array-message.js
- harness/deepEqual-primitives-bigint.js
- staging/Temporal/v8/plain-date-time-constructor.js
- staging/Temporal/v8/plain-date-from.js
- staging/Temporal/v8/instant-constructor.js
- And many more Temporal tests

**Fix Required**:
- Add `bigint` field to Literal class
- Populate it when parsing BigInt literals
- Store the string representation (without 'n' suffix)

---

### 2. Array Holes/Elisions (HIGH PRIORITY)
**Issue**: Sparse array literals have incorrect element counts. Array holes are being represented differently than expected.

**Examples**:
- `[,]` - Single hole (should have 1 element: null)
- `[,,]` - Two holes (should have 2 elements: null, null)
- `[1,,3]` - Hole in middle

**Current behavior**: Appears to be creating 2 elements for `[,]` instead of 1

**Expected AST** for `[,]`:
```json
{
  "type": "ArrayExpression",
  "elements": [null]
}
```

**Current AST** for `[,]` (inferred):
```json
{
  "type": "ArrayExpression",
  "elements": [null, null]
}
```

**Affected Files**: ~10+ files
- harness/compare-array-sparse.js
- harness/testTypedArray-conversions.js
- harness/nativeFunctionMatcher.js
- staging/Temporal/Regex/old/instant.js

**Fix Required**:
- Review array literal parsing logic for trailing commas
- Ensure holes are counted correctly (comma = hole, trailing comma doesn't add extra hole)

---

## Priority Ranking

1. **BigInt Literal Support** - Affects ~50+ files
   - Add `bigint` field to Literal AST node
   - Parse and store bigint string value
   - Test with various bigint literals

2. **Array Holes/Elisions** - Affects ~10+ files
   - Fix array literal parsing for holes
   - Handle trailing commas correctly
   - Test sparse array patterns

3. **Other Edge Cases** - Remaining ~7,000 files
   - Need deeper investigation
   - May include other missing features or subtle differences

## Next Steps

1. Fix BigInt support first (largest impact)
2. Fix array holes second (clear issue)
3. Run tests again to measure improvement
4. Investigate remaining mismatches in detail

## Test Commands

To reproduce specific issues:
```bash
# Check a specific bigint file
node -e "console.log(require('acorn').parse('1n', {ecmaVersion: 2020}))"

# Check sparse array
node -e "console.log(JSON.stringify(require('acorn').parse('[,]', {ecmaVersion: 2020}), null, 2))"
```

To re-run analysis:
```bash
./mvnw test -Dtest=FindActualMismatchTest
```
