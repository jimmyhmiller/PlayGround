# C-Compatible Structs for MLIR Collections

This document describes the C-compatible struct layer that enables direct field access to collection types from MLIR code.

## Overview

Instead of calling C export functions to access collection fields, you can now work with C-compatible struct layouts that can be directly manipulated from MLIR using `llvm.getelementptr` and `llvm.load/store` operations.

## Architecture

### 1. C-Compatible Struct Layouts (`src/collections/c_structs.zig`)

**CVectorLayout**: `!llvm.struct<(ptr, i64, i64, i64)>`
```zig
extern struct CVectorLayout {
    data: ?[*]u8,      // Field 0: Pointer to element array
    len: usize,        // Field 1: Number of elements
    capacity: usize,   // Field 2: Allocated capacity
    elem_size: usize,  // Field 3: Size of each element in bytes
}
```

**CMapLayout**: `!llvm.struct<(ptr, i64, i64, i64)>`
```zig
extern struct CMapLayout {
    entries: ?[*]u8,   // Field 0: Pointer to entries array
    len: usize,        // Field 1: Number of key-value pairs
    capacity: usize,   // Field 2: Allocated capacity
    entry_size: usize, // Field 3: Size of each entry in bytes
}
```

### 2. C API Extensions (`src/collections/c_api.zig`)

New export functions for struct conversion:

- `vector_value_to_layout()` - Convert PersistentVector → CVectorLayout
- `vector_layout_to_value()` - Convert CVectorLayout → PersistentVector
- `vector_layout_destroy()` - Free a CVectorLayout
- `vector_layout_get_len()` - Get length field (example accessor)
- `vector_layout_create_empty()` - Create empty layout
- Similar functions for CMapLayout

### 3. MLIR Type Helpers (`src/mlir/collection_types.zig`)

Helper functions for working with struct types in MLIR:

**Type Creation**:
- `createVectorLayoutType()` - Returns `!llvm.struct<(ptr, i64, i64, i64)>`
- `createMapLayoutType()` - Returns map struct type

**Field Indices**:
```zig
pub const VectorLayoutField = enum(i32) {
    data = 0,
    len = 1,
    capacity = 2,
    elem_size = 3,
};
```

**Code Generation**:
- `generateLoadVectorLen()` - Generate GEP + load for length field
- `generateLoadVectorData()` - Generate GEP + load for data pointer
- `generateStoreVectorLen()` - Generate GEP + store for length
- `generateVectorElementAccess()` - Generate element access code

## Usage Examples

### Example 1: Reading Vector Length

```lisp
;; Function that extracts the length from a CVectorLayout pointer
(defn get_vector_length [(: %vec_layout_ptr !llvm.ptr)] i64
  ;; GEP to access the 'len' field (offset 1 in the struct)
  (op %len_field_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec_layout_ptr (constant (: 1 i32)) : !llvm.ptr]))

  ;; Load the i64 length value
  (op %len (: i64) (llvm.load [%len_field_ptr]))

  ;; Return it
  (return %len))
```

### Example 2: Accessing Vector Data

```lisp
(defn get_vector_data [(: %vec_ptr !llvm.ptr)] !llvm.ptr
  ;; GEP to access the 'data' field (offset 0)
  (op %data_field_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec_ptr (constant (: 0 i32)) : !llvm.ptr]))

  ;; Load the pointer value
  (op %data_ptr (: !llvm.ptr) (llvm.load [%data_field_ptr]))

  (return %data_ptr))
```

### Example 3: Working with Elements

```lisp
(defn sum_vector_lengths [(: %vec1_ptr !llvm.ptr) (: %vec2_ptr !llvm.ptr)] i64
  ;; Access len field of first vector
  (op %len1_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec1_ptr (constant (: 1 i32)) : !llvm.ptr]))
  (op %len1 (: i64) (llvm.load [%len1_ptr]))

  ;; Access len field of second vector
  (op %len2_ptr (: !llvm.ptr)
      (llvm.getelementptr [%vec2_ptr (constant (: 1 i32)) : !llvm.ptr]))
  (op %len2 (: i64) (llvm.load [%len2_ptr]))

  ;; Sum the lengths
  (op %sum (: i64) (llvm.add [%len1 %len2]))

  (return %sum))
```

See `examples/vector_struct_access.lisp` for more examples.

## Memory Layout

The structs use the C ABI with explicit field offsets:

```
CVectorLayout (32 bytes on 64-bit):
+--------+--------+--------+--------+
|  data  |  len   |  cap   |  size  |
| ptr(8) | i64(8) | i64(8) | i64(8) |
+--------+--------+--------+--------+
Offset:    0        8        16       24
Field:     0        1        2        3
```

These offsets match MLIR's expectations for `llvm.getelementptr` with constant integer indices.

## Benefits

### Performance
- **Direct Memory Access**: No function call overhead
- **Inline-able**: GEP operations can be optimized away by LLVM
- **Cache-Friendly**: Struct fields are contiguous in memory

### Type Safety
- **MLIR Type System**: Understands struct layout
- **Compile-Time Checks**: Wrong field access caught at compile time
- **Clear API**: Field indices are explicit enums

### Simplicity
- **Fewer Export Functions**: Don't need a C function for every accessor
- **Self-Documenting**: Struct layout is visible in MLIR IR
- **Standard Patterns**: Use familiar GEP + load/store operations

## Tradeoffs

### ABI Stability
- **Pro**: Clear, documented layout
- **Con**: Changing struct layout breaks compatibility
- **Mitigation**: Version the structs if needed

### Encapsulation
- **Pro**: Direct access is faster
- **Con**: Internal structure exposed
- **Mitigation**: Document which fields are stable API

### Immutability
- **Pro**: Direct mutation possible if needed
- **Con**: Bypasses persistent data structure guarantees
- **Mitigation**: Use conversion functions to maintain immutability

## Testing

Run the struct access tests:
```bash
zig build test
```

Tests verify:
- C struct field offsets match MLIR expectations
- Conversion between PersistentVector and CVectorLayout
- MLIR type creation for struct types
- Code generation for field access
- Field enum indices match GEP requirements

## Files

- `src/collections/c_structs.zig` - C-compatible struct definitions
- `src/collections/c_api.zig` - C API export functions (extended)
- `src/mlir/collection_types.zig` - MLIR type helpers
- `examples/vector_struct_access.lisp` - Usage examples
- `test/struct_access_test.zig` - Comprehensive tests

## Future Work

While not implemented in this iteration, potential extensions include:

1. **Builder Helpers** (`src/builder.zig`):
   - `buildVectorLen(vec_ptr)` → emit GEP + load
   - `buildVectorData(vec_ptr)` → emit GEP for data
   - `buildVectorElementAccess(vec_ptr, index)` → emit element GEP

2. **Macro Support**:
   - `(vec-len %ptr)` macro → expands to GEP + load
   - `(vec-data %ptr)` macro → expands to data access
   - `(vec-at %ptr %idx)` macro → expands to element access

3. **Generic Collections**:
   - Type-parameterized struct layouts
   - Runtime type information for safety

## Conclusion

This C-compatible struct layer provides a clean, efficient way to manipulate collection data structures from MLIR code while maintaining compatibility with existing opaque pointer APIs. The approach balances performance, type safety, and API simplicity.
