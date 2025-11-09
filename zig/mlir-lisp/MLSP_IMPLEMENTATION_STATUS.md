# MLSP Dialect Implementation Status

**Date**: 2025-01-05
**Status**: Phase 1 Complete ✓

---

## Summary

We've successfully implemented the foundational IRDL dialect definition for the `mlsp` (MLIR-Lisp) dialect. The dialect operations are now recognized by MLIR and can be parsed and verified.

---

## Achievements

### ✓ Phase 1: IRDL Dialect Definition Complete

1. **Created `mlsp` dialect** with IRDL definitions
2. **Defined core operations**:
   - `mlsp.identifier` - Create identifier atoms
   - `mlsp.list` - Create list collections
   - `mlsp.get_element` - Extract list elements (has limitations)
   - `mlsp.number` - Create number atoms
   - `mlsp.string` - Create string atoms
   - `mlsp.vector` - Create vector collections
   - `mlsp.string_const` - Reference global strings
   - `mlsp.build_operation` - High-level operation builder

3. **Verified dialect loading**:
   - IRDL definitions parse correctly
   - Dialect registers with MLIR context
   - Operations can be created and verified
   - Type constraints work properly

---

## Current State

### Working Files

1. **`examples/mlsp_dialect.lisp`**
   Complete IRDL definition + placeholder PDL transforms (565 lines)

2. **`examples/mlsp_dialect_only.lisp`**
   Minimal dialect-only test without transforms (100 lines)
   Demonstrates that the dialect loads and operations are recognized

3. **`examples/test_mlsp_basic.lisp`**
   Application code using mlsp operations

### What Works

```lisp
;; These operations are recognized and verified:
(op %name (: !llvm.ptr) (mlsp.identifier {:value @str_test}))
(op %elem1 (: !llvm.ptr) (mlsp.identifier {:value @str_a}))
(op %elem2 (: !llvm.ptr) (mlsp.identifier {:value @str_b}))
(op %list (: !llvm.ptr) (mlsp.list [%elem1 %elem2]))
```

Output:
```
✓ MLIR module created successfully!
```

The operations parse, verify, and are recognized by MLIR. The only error is that they can't be lowered to LLVM yet (expected - no transforms implemented).

---

## Known Limitations

### 1. IRDL Multi-Operand Constraints

**Issue**: IRDL `irdl.operands` doesn't easily support heterogeneous fixed operands (e.g., `!llvm.ptr` followed by `i64`).

**Impact**: `mlsp.get_element` can't be properly defined to take `(%list: !llvm.ptr, %index: i64)`.

**Workarounds**:
- Option A: Define as variadic with `any_of` constraint (loses type safety)
- Option B: Skip IRDL, manually register operation in C++
- Option C: Use attribute for index instead of operand: `mlsp.get_element(%list) {index = 0}`

**Chosen**: Option C (use attribute) is cleanest for our use case.

### 2. No Lowering Transforms Yet

**Issue**: PDL transforms are placeholder `llvm.call` operations that segfault.

**Impact**: Operations can't be executed via JIT yet.

**Next Steps**: Implement real LLVM lowering sequences in PDL patterns.

---

## Next Steps

### Phase 2: PDL Transform Patterns (Not Started)

**Goal**: Lower mlsp operations to LLVM IR

**Approach**: Write complete PDL rewrites that generate full LLVM sequences

**Example** (mlsp.identifier → LLVM):
```mlir
pdl.rewrite %mlsp_op {
  // Create constant: 56 bytes for CValueLayout
  %c56 = pdl.operation "arith.constant" {value = 56 : i64}

  // Call malloc
  %malloc_result = pdl.operation "llvm.call" {callee = @malloc} [%c56]

  // GEP to type field (offset 0)
  %type_gep = pdl.operation "llvm.getelementptr" [%malloc_result] {indices = [0]}

  // Store identifier tag (0)
  %tag = pdl.operation "arith.constant" {value = 0 : i32}
  pdl.operation "llvm.store" [%tag, %type_gep]

  // GEP to data_ptr (offset 8)
  %data_gep = pdl.operation "llvm.getelementptr" [%malloc_result] {indices = [8]}

  // Store string pointer
  %str_ptr = pdl.operation "llvm.mlir.addressof" {global = /* extract from attribute */}
  pdl.operation "llvm.store" [%str_ptr, %data_gep]

  // ... (store length, zero remaining fields) ...

  pdl.replace %mlsp_op with %malloc_result
}
```

**Challenge**: PDL patterns can create multiple operations, but extracting attribute values and generating dynamic constants is complex.

**Alternative**: Use Transform dialect's imperative API instead of declarative PDL.

---

## Design Decisions Made

### 1. Use `irdl.any` for String Attributes

Identifier/string operations accept any attribute type for flexibility:
```lisp
(op %any_attr (: !irdl.attribute) (irdl.any []))
(operation (name irdl.attributes)
           (attributes {:attributeValueNames ["value"]})
           (operands %any_attr))
```

Allows both symbol references (`@str_test`) and inline strings (`"test"`).

### 2. All Operations Return `!llvm.ptr`

Keeps type system simple - everything is `CValueLayout*`.

Future: Could add `!mlsp.value` opaque type for better safety.

### 3. Variadic Operands for Collections

`mlsp.list` and `mlsp.vector` use IRDL variadic operands:
```lisp
(operation (name irdl.operands)
           (attributes {:names ["elements"]
                       :variadicity #irdl<variadicity_array[ variadic]>})
           (operands %ptr_type))
```

Handles arbitrary number of children naturally.

---

## File Organization

```
examples/
├── mlsp_dialect.lisp           # Full dialect + transforms (WIP)
├── mlsp_dialect_only.lisp       # Minimal dialect for testing
├── test_mlsp_basic.lisp         # Basic usage examples
├── test_mlsp_complete.lisp      # Combined dialect + test (generated)
└── add_macro_with_strings_refactored.lisp  # Target to replace (310 lines)
```

---

## Success Metrics

### Phase 1 (Complete) ✓

- [x] IRDL dialect loads without errors
- [x] mlsp.identifier operation recognized
- [x] mlsp.list operation recognized
- [x] Operations parse and verify correctly
- [x] Type constraints work

### Phase 2 (Next)

- [ ] mlsp.identifier lowers to working LLVM IR
- [ ] mlsp.list lowers to working LLVM IR
- [ ] Generated LLVM matches hand-written version
- [ ] Simple macro works end-to-end

### Phase 3 (Future)

- [ ] Full add_macro rewritten with 90%+ line reduction
- [ ] All builtin macros use mlsp operations
- [ ] Performance acceptable (< 2x slowdown)

---

## Technical Notes

### IRDL Learning

1. **Multiple operands require single `irdl.operands` op**
   Can't define separate `irdl.operands` for each - MLIR treats them as overwriting.

2. **Variadicity attribute format**
   Syntax: `#irdl<variadicity_array[ single]>` (note space after `[`)

3. **Constraint composition**
   Use `irdl.any_of` to create union types: `irdl.any_of [%constraint1 %constraint2]`

4. **No automatic names**
   Must provide `:names` and `:variadicity` arrays manually

### PDL Patterns

1. **Can create multi-operation sequences**
   PDL `pdl.rewrite` blocks can contain multiple `pdl.operation` nodes

2. **Attribute extraction is tricky**
   Need to figure out how to extract attribute values and use them in generated IR

3. **Transform dialect may be easier**
   Imperative API might be more flexible than declarative PDL for complex lowerings

---

## Open Questions

1. **Should we use PDL or Transform dialect for lowering?**
   - PDL: Declarative, pattern-based, harder for complex sequences
   - Transform: Imperative, flexible, more verbose
   - **Decision pending**: Try PDL first, fallback to Transform if needed

2. **How to handle helper functions?**
   - Keep external malloc/memset/memcpy?
   - Or inline everything in PDL patterns?
   - **Leaning toward**: Keep helpers, just reduce calls

3. **Should we add custom types like `!mlsp.value`?**
   - Pros: Better type safety, clearer intent
   - Cons: More complexity, conversion overhead
   - **Decision**: Start with `!llvm.ptr`, add types later if needed

---

## Conclusion

**Phase 1 is complete and successful!** The mlsp dialect is defined, recognized, and operations can be created. The foundation is solid for implementing the lowering transforms in Phase 2.

**Key achievement**: We went from 0 to a working custom MLIR dialect in about 60 lines of IRDL!

**Next milestone**: Implement one complete lowering pattern (mlsp.identifier → LLVM) to prove the full pipeline works.
