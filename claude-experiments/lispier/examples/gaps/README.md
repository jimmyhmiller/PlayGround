# MLIR Support Gaps in Lispier

This directory contains test files that demonstrate gaps in lispier's MLIR support.
Each file documents specific features that either don't work or have limitations.

## Summary of Gaps Found

### 1. Type Parsing Gaps

**Files:** `tensor_types.lisp`, `memref_advanced.lisp`

The tokenizer truncates types containing `?` inside angle brackets.

- `tensor<4x4xf32>` → works (but see attribute issue below)
- `tensor<?xf32>` → **FAILS** - truncated to `tensor<`
- `memref<?xf32>` → **FAILS** - truncated to `memref<`

**Root Cause:** In `tokenizer.rs:119-138`, when parsing inside angle brackets,
the `?` character is not in the allowed character list.

**Workaround:** Use string syntax: `"memref<?xf32>"` instead of bare symbol.

### 2. Dense Attributes Not Properly Parsed

**File:** `dense_attributes.lisp`

Dense tensor attributes like `dense<[1,2,3,4]>` are stored as strings, not proper
MLIR attributes. The generated IR shows:
```mlir
%0 = "arith.constant"() <{value = "dense<[1, 2, 3, 4]> : tensor<4xi32>"}> : () -> tensor<4xi32>
```

This won't verify correctly because `value` should be a DenseElementsAttr, not StringAttr.

### 3. Operations Requiring Explicit Result Types

**Files:** `affine_dialect.lisp`, `linalg_dialect.lisp`, `scf_while.lisp`

**STATUS: FIXED**

Previously, operations would fail with "type inference was requested but operation does not support it".
This was because we maintained a hardcoded `is_void_operation()` list.

**Solution:** We added `operation_supports_type_inference()` to our Melior fork, which queries
MLIR's `InferTypeOpInterface` at runtime. Now lispier correctly detects which operations
support type inference without needing a hardcoded list.

### 4. scf.if / scf.while Support

**Files:** `scf_if.lisp`, `scf_while.lisp`

- `scf.if` with result works
- `scf.while` fails due to `scf.condition` not being recognized as void
- `scf.if` with only "then" region (no else) untested

### 5. scf.parallel Not Working

**File:** `scf_parallel.lisp`

Parallel loops with reductions fail with "invalid operand type".

### 6. Affine Maps and Complex Layouts

**File:** `memref_advanced.lisp`

Types with commas cause tokenizer errors:
- `memref<5x5xf32, strided<[10, 1], offset: 0>>` → **FAILS**
- `affine_map<(d0, d1) -> (d0 + d1)>` → **FAILS**

## Working Features

These files demonstrate features that DO work:

| File | Feature |
|------|---------|
| `arith_division.lisp` | Division, remainder, negation, type casts |
| `math_dialect.lisp` | sqrt, exp, log, sin, cos, tan, pow, abs, floor, ceil |
| `complex_types.lisp` | Complex number types and operations |
| `vector_types.lisp` | Vector types and operations |
| `index_dialect.lisp` | Index type operations |
| `tuple_types.lisp` | Tuple types with builtin.unrealized_conversion_cast |
| `scf_if.lisp` | Basic scf.if with two regions |

## Running Tests

```bash
# Test a specific file
./target/release/lispier show-ir examples/gaps/math_dialect.lisp

# Test all files in the gaps directory
for f in examples/gaps/*.lisp; do
  echo "=== $f ==="
  ./target/release/lispier show-ir "$f" 2>&1 | head -10
done
```

## Fixes Needed

1. **Tokenizer:** ~~Add `?` to allowed characters inside angle brackets~~ **FIXED**
2. **Tokenizer:** Decide how to handle commas in types (currently not allowed)
3. ~~**IR Generator:** Add more operations to `is_void_operation()` list~~ **FIXED** - Now uses `operation_supports_type_inference()`
4. **Attribute Parsing:** Implement proper dense attribute parsing
