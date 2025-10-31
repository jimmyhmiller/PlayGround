# Test Status - Unsupported Features

This document tracks which MLIR features are currently unsupported by our parser. Tests have been added to demonstrate these failures.

## Test Results Summary

**Total tests:** 38
**Passed:** 27
**Failed:** 10
**Skipped:** 1

## ✅ Currently Supported Features

- Basic integer types (i32, i64, si32, ui64, etc.)
- Float types (f16, f32, f64, f80, f128, bf16, tf32)
- Index type
- Generic operation syntax (from `mlir-opt -mlir-print-op-generic`)
- Operation results and operands
- Dictionary properties (`<{...}>`)
- Dictionary attributes (`{...}`)
- Function types (`() -> i32`, `(i32, i32) -> i32`)
- Value uses
- Multiple operation results with count notation (`%0:2`)
- Value use with result number (`%0#1`)
- Basic location tracking

## ❌ Currently Failing Tests (Expected)

### 1. **Operation with Successors** (test/unsupported_features_test.zig:11)
- **Grammar:** `successor-list ::= '[' successor (',' successor)* ']'`
- **Example file:** `examples/unsupported_01_successors.mlir`
- **Error:** `Expected colon, but got lbracket at line 4, column 17`
- **What's needed:** Parse successor lists for control flow operations (branches)
```mlir
"cf.cond_br"(%2)[^bb1, ^bb2] : (i1) -> ()
```

### 2. **Operation with Regions** (test/unsupported_features_test.zig:44)
- **Grammar:** `region-list ::= '(' region (',' region)* ')'`
- **Grammar:** `region ::= '{' entry-block? block* '}'`
- **Example file:** `examples/unsupported_02_regions.mlir`
- **Error:** `Expected colon, but got lparen at line 2, column 14`
- **What's needed:** Parse regions for operations like `scf.if`, `scf.for`
```mlir
"scf.if"(%0) ({
  %1 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  "scf.yield"(%1) : (i32) -> ()
}, {
  %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "scf.yield"(%2) : (i32) -> ()
}) : (i1) -> i32
```

### 3. **Basic Blocks with Labels** (test/unsupported_features_test.zig:71)
- **Grammar:** `block-label ::= block-id block-arg-list? ':'`
- **Grammar:** `block-id ::= caret-id`
- **Example file:** `examples/unsupported_01_successors.mlir`
- **Status:** Skipped (no parseBlock function exists)
- **What's needed:** Implement block parsing with labels and block arguments
```mlir
^bb0(%arg0: i32, %arg1: i32):
  %0 = "arith.addi"(%arg0, %arg1) : (i32, i32) -> i32
```

### 4. **Tensor Types** (test/unsupported_features_test.zig:77)
- **Grammar:** Part of `builtin-type`
- **Example file:** `examples/unsupported_03_tensor_types.mlir`
- **Error:** `Unknown type`
- **What's needed:** Parse tensor types with shapes
```mlir
tensor<4x8xf32>
tensor<?x10xf64>
tensor<*xf32>  // unranked
```

### 5. **Memref Types** (test/unsupported_features_test.zig:91)
- **Grammar:** Part of `builtin-type`
- **Example file:** `examples/unsupported_04_memref_types.mlir`
- **Error:** `Unknown type`
- **What's needed:** Parse memref types with layouts and memory spaces
```mlir
memref<16x16xf64>
memref<?x?xf32>  // dynamic dimensions
memref<256xf32, 1>  // with memory space
```

### 6. **Vector Types** (test/unsupported_features_test.zig:105)
- **Grammar:** Part of `builtin-type`
- **Example file:** `examples/unsupported_05_vector_types.mlir`
- **Error:** `Unknown type`
- **What's needed:** Parse vector types (SIMD)
```mlir
vector<4xf32>
vector<4x8xf64>
vector<[4]xi32>  // scalable
```

### 7. **Complex Types** (test/unsupported_features_test.zig:119)
- **Grammar:** Part of `builtin-type`
- **Example file:** `examples/unsupported_06_complex_tuple_types.mlir`
- **Error:** `Unknown type`
- **What's needed:** Parse complex number types
```mlir
complex<f32>
complex<f64>
```

### 8. **Tuple Types** (test/unsupported_features_test.zig:133)
- **Grammar:** Part of `builtin-type`
- **Example file:** `examples/unsupported_06_complex_tuple_types.mlir`
- **Error:** `Unknown type`
- **What's needed:** Parse tuple types
```mlir
tuple<i32, f64, index>
tuple<tensor<2x2xf32>, memref<10xi32>, i1>
```

### 9. **Pretty Dialect Types** (test/unsupported_features_test.zig:231)
- **Grammar:** `pretty-dialect-type ::= dialect-namespace '.' pretty-dialect-type-lead-ident dialect-type-body?`
- **Example file:** `examples/unsupported_10_dialect_types_attrs.mlir`
- **Error:** Assertion failure
- **What's needed:** Parse pretty-printed dialect types (not just opaque)
```mlir
!llvm.ptr
!llvm.ptr<i32>
!llvm.array<10 x i32>
```

### 10. **Type Alias Usage** (test/unsupported_features_test.zig:251)
- **Grammar:** `type-alias-def ::= '!' alias-name '=' type`
- **Example file:** `examples/unsupported_09_aliases.mlir`
- **Error:** `Expected generic operation (string literal)`
- **What's needed:** Improve parseModule to handle type alias definitions at module level
```mlir
!my_int = i32
%0 = "arith.constant"() <{value = 42 : !my_int}> : () -> !my_int
```

### 11. **Attribute Alias Usage** (test/unsupported_features_test.zig:270)
- **Grammar:** `attribute-alias-def ::= '#' alias-name '=' attribute-value`
- **Example file:** `examples/unsupported_09_aliases.mlir`
- **Error:** `Expected generic operation (string literal)`
- **What's needed:** Improve parseModule to handle attribute alias definitions at module level
```mlir
#zero_i32 = 0 : i32
%0 = "test.op"() {value = #zero_i32} : () -> i32
```

## 📝 Implementation Priority

Based on MLIR usage patterns, suggested implementation order:

1. **High Priority** (commonly used in all dialects):
   - Regions (required for control flow)
   - Successors (required for control flow)
   - Basic blocks with labels
   - Tensor types (core to ML dialects)
   - Vector types (core to codegen)

2. **Medium Priority** (frequently used):
   - Memref types
   - Type and attribute aliases
   - Pretty dialect types

3. **Lower Priority** (specialized):
   - Complex types
   - Tuple types

## 🔍 How to Use These Tests

These tests are **intentionally failing**. They serve as a roadmap for implementation:

1. Pick a feature to implement
2. Look at the corresponding test in `test/unsupported_features_test.zig`
3. Look at the example file in `examples/unsupported_*.mlir`
4. Refer to the grammar rules in the test comments
5. Implement the parser functions with proper grammar comments
6. Watch the test pass!

## Running Tests

```bash
# Run all tests (will show failures)
zig build test

# Run only unsupported features tests
zig test test/unsupported_features_test.zig --dep mlir_parser
```

## Validating Examples

All example files can be validated with `mlir-opt`:

```bash
mlir-opt --verify-diagnostics examples/unsupported_01_successors.mlir
mlir-opt --verify-diagnostics examples/unsupported_02_regions.mlir
# etc.
```

---

**Last Updated:** 2025-10-25
**Parser Version:** 0.1.0
**MLIR Grammar Reference:** grammar.ebnf (from https://mlir.llvm.org/docs/LangRef/)
