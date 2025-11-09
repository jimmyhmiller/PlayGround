# MLSP Dialect Design: High-Level Macro Building

## Vision

Create an IRDL dialect called `mlsp` (MLIR-Lisp) that provides high-level operations for building CValueLayout structures. Use PDL/Transform to lower these to LLVM operations, dramatically simplifying macro implementations.

### Before vs After

**Current state**: `add_macro_with_strings_refactored.lisp` - **~310 lines** of malloc/GEP/store boilerplate

**Target state**: **~15 lines** of high-level mlsp operations

```mlir
mlsp.macro @addMacro(%args: !mlsp.value) -> !mlsp.value {
  // Extract: type, arg1, arg2 from (+ (: type) arg1 arg2)
  %type_expr = mlsp.get_element %args[0]
  %arg1 = mlsp.get_element %args[1]
  %arg2 = mlsp.get_element %args[2]
  %type = mlsp.get_element %type_expr[1]

  // Build operation with named fields
  %result = mlsp.build_operation {
    name = "arith.addi",
    result_types = [%type],
    operands = [%arg1, %arg2]
  }

  mlsp.return %result
}
```

---

## Core Dialect Operations

### Value Construction

#### `mlsp.identifier`
Creates an identifier/atom CValueLayout from a string constant.

**Syntax**: `%val = mlsp.identifier "name"`

**Lowers to**: malloc(56) + store type tag + store string ptr + store length

**Replaces**: ~7 lines of LLVM operations per identifier

---

#### `mlsp.list`
Creates a list CValueLayout from variadic children.

**Syntax**: `%list = mlsp.list(%val1, %val2, %val3)`

**Lowers to**:
- Allocate array for pointers
- Malloc CValueLayout
- Store list tag + array ptr + length

**Replaces**: ~22 lines of allocate-value-array + create-list boilerplate

---

#### `mlsp.get_element`
Extracts element from list by index.

**Syntax**: `%elem = mlsp.get_element %list[0]`

**Lowers to**:
- Load data_ptr from offset 8
- GEP into array at index
- Load pointer

**Replaces**: ~3-4 lines of GEP/load operations

---

#### `mlsp.string_const`
References a global string constant.

**Syntax**: `%str = mlsp.string_const @my_string`

**Lowers to**: `llvm.mlir.addressof @my_string`

**Replaces**: Direct 1:1 replacement, but cleaner abstraction

---

### Sugar Operations

#### `mlsp.build_operation`
High-level operation builder with named fields.

**Syntax**:
```mlir
%op = mlsp.build_operation {
  name = %name_str,           // mlsp.value (identifier)
  result_types = %types_list, // mlsp.value (list)
  operands = %ops_list        // mlsp.value (list)
}
```

**Lowers to**:
- Create "operation" identifier
- Create "name" identifier
- Create name section list
- Create "result-types" identifier
- ... (full operation structure)
- Final mlsp.list wrapping everything

**Replaces**: ~150+ lines of manual CValueLayout construction

---

#### `mlsp.extract_args` (Future)
Pattern-based argument extraction from s-expressions.

**Syntax**: `%type, %arg1, %arg2 = mlsp.extract_args %input : (+ (: $type) $arg1 $arg2)`

**Lowers to**: Multiple mlsp.get_element calls

**Replaces**: Manual indexing and validation

---

## Implementation Phases

### Phase 1: IRDL Dialect Definition ✅ (Prerequisites Complete)

**Status**: We now have working IRDL + PDL transform pipeline from `demo_dialect_complete.lisp`

**File**: `mlsp_dialect.lisp`

```lisp
(operation
  (name irdl.dialect)
  (attributes {:sym_name @mlsp})
  (regions
    (region
      (block
        ;; Define mlsp.identifier operation
        (operation
          (name irdl.operation)
          (attributes {:sym_name @identifier})
          (regions
            (region
              (block
                ;; Takes string attribute "value"
                (op %str_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.attributes)
                  (attributes {:attributeValueNames ["value"]})
                  (operands %str_type))

                ;; Returns !llvm.ptr (CValueLayout*)
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operands %ptr_type))))))

        ;; Define mlsp.list operation
        (operation
          (name irdl.operation)
          (attributes {:sym_name @list})
          (regions
            (region
              (block
                ;; Takes variadic !llvm.ptr operands
                (op %ptr_type (: !irdl.attribute) (irdl.is {:expected !llvm.ptr} []))
                (operation
                  (name irdl.operands)
                  (attributes {:names ["elements"] :variadicity #irdl<variadicity_array[ variadic]>})
                  (operands %ptr_type))

                ;; Returns !llvm.ptr
                (operation
                  (name irdl.results)
                  (attributes {:names ["result"] :variadicity #irdl<variadicity_array[ single]>})
                  (operands %ptr_type))))))

        ;; Define mlsp.get_element operation
        ;; TODO: Additional operations...
        ))))
```

**Tasks**:
- [x] Understand IRDL syntax (completed via demo_dialect_complete.lisp)
- [x] Learn PDL transform patterns (completed)
- [ ] Define complete mlsp dialect operations
- [ ] Test dialect loads successfully

---

### Phase 2: PDL Transform Patterns

**File**: `mlsp_transforms.lisp` (embedded in same file as dialect)

**Example**: Lower `mlsp.identifier` to LLVM

```lisp
(operation
  (name builtin.module)
  (attributes {})
  (regions
    (region
      (block
        (operation
          (name transform.with_pdl_patterns)
          (regions
            (region
              (block [^bb0]
                (arguments [(: %root !transform.any_op)])

                ;; PDL Pattern: mlsp.identifier -> LLVM malloc+store
                (operation
                  (name pdl.pattern)
                  (attributes {:benefit (: 1 i16) :sym_name @mlsp_identifier_to_llvm})
                  (regions
                    (region
                      (block
                        ;; Match mlsp.identifier
                        (operation
                          (name pdl.attribute)
                          (result-bindings [%str_val])
                          (result-types !pdl.attribute))
                        (operation
                          (name pdl.type)
                          (result-bindings [%ptr_type])
                          (result-types !pdl.type))
                        (operation
                          (name pdl.operation)
                          (attributes {:opName "mlsp.identifier"
                                     :attributeValueNames ["value"]
                                     :operandSegmentSizes array<i32: 0, 1, 1>})
                          (operands %str_val %ptr_type)
                          (result-bindings [%mlsp_op])
                          (result-types !pdl.operation))

                        ;; Rewrite to LLVM sequence
                        (operation
                          (name pdl.rewrite)
                          (attributes {:operandSegmentSizes array<i32: 1, 0>})
                          (operands %mlsp_op)
                          (regions
                            (region
                              (block
                                ;; TODO: Generate:
                                ;; 1. malloc(56)
                                ;; 2. store identifier_tag
                                ;; 3. store string ptr
                                ;; 4. store length
                                ;; 5. zero remaining fields

                                ;; For now: simple operation replacement
                                (operation
                                  (name pdl.operation)
                                  (attributes {:opName "llvm.call"
                                             :attributeValueNames ["callee"]
                                             :operandSegmentSizes array<i32: 0, 1, 1>})
                                  (operands %str_val %ptr_type)
                                  (result-bindings [%llvm_op])
                                  (result-types !pdl.operation))

                                (operation
                                  (name pdl.replace)
                                  (attributes {:operandSegmentSizes array<i32: 1, 1, 0>})
                                  (operands %mlsp_op %llvm_op))))))))))

                ;; Transform sequence
                (operation
                  (name transform.sequence)
                  (attributes {:failure_propagation_mode (: 1 i32)
                             :operandSegmentSizes array<i32: 1, 0>})
                  (operands %root)
                  (result-types !transform.any_op)
                  (regions
                    (region
                      (block [^bb1]
                        (arguments [(: %arg1 !transform.any_op)])
                        (operation
                          (name transform.pdl_match)
                          (attributes {:pattern_name @mlsp_identifier_to_llvm})
                          (operands %arg1)
                          (result-bindings [%matched])
                          (result-types !transform.any_op))
                        (operation
                          (name transform.yield)))))))))))))))
```

**Challenge**: PDL can only create single operations, but we need sequences.

**Solutions**:
1. **Use PDL multi-operation patterns** - Research if PDL supports creating multiple ops
2. **Use Transform dialect directly** - Skip PDL, use transform.structured.* ops
3. **Hybrid approach** - PDL for matching, custom Transform ops for lowering
4. **Helper functions** - Keep some LLVM functions (malloc wrapper, etc.), just reduce calls

**Approach**: Each PDL pattern's rewrite region will contain the full LLVM sequence.

**Example structure** (simplified):
```lisp
(operation (name pdl.rewrite) (operands %mlsp_op)
  (regions (region (block
    ;; Create constant for malloc size
    (op %c56 ... (pdl.operation "arith.constant" {value = 56}))
    ;; Call malloc
    (op %ptr ... (pdl.operation "llvm.call" {callee = @malloc} [%c56]))
    ;; GEP to type field
    (op %type_ptr ... (pdl.operation "llvm.getelementptr" [%ptr] {...}))
    ;; Store identifier tag
    (op %tag ... (pdl.operation "arith.constant" {value = 0}))
    (op ... (pdl.operation "llvm.store" [%tag %type_ptr]))
    ;; ... repeat for all fields ...
    ;; Replace original mlsp.identifier with final pointer
    (operation (name pdl.replace) (operands %mlsp_op %ptr))))))
```

**Tasks**:
- [ ] Write complete PDL pattern for `mlsp.identifier` → LLVM sequence
- [ ] Write complete PDL pattern for `mlsp.list` → LLVM sequence
- [ ] Handle complex case: `mlsp.build_operation` (may need 50+ pdl.operation nodes!)
- [ ] Test transforms apply correctly
- [ ] Verify generated LLVM matches hand-written version

---

### Phase 3: Integration Testing

**File**: `test_mlsp_macro.lisp`

```lisp
;; IRDL + Transforms (auto-detected and loaded)
;; ... mlsp dialect definition ...
;; ... PDL transform patterns ...

;; Application: Simple macro using mlsp operations
(defn addMacro [(: %args !llvm.ptr)] !llvm.ptr
  ;; High-level mlsp operations
  (op %type_expr (: !llvm.ptr) (mlsp.get_element [%args] {:index (: 0 i64)}))
  (op %arg1 (: !llvm.ptr) (mlsp.get_element [%args] {:index (: 1 i64)}))
  (op %arg2 (: !llvm.ptr) (mlsp.get_element [%args] {:index (: 2 i64)}))
  (op %type (: !llvm.ptr) (mlsp.get_element [%type_expr] {:index (: 1 i64)}))

  (op %name (: !llvm.ptr) (mlsp.identifier {:value "arith.addi"}))
  (op %types_list (: !llvm.ptr) (mlsp.list [%type]))
  (op %ops_list (: !llvm.ptr) (mlsp.list [%arg1 %arg2]))

  (op %result (: !llvm.ptr) (mlsp.build_operation [%name %types_list %ops_list]))

  (return %result))

;; Test harness
(defn main [] i64
  ;; TODO: Create test input, call addMacro, verify output
  (constant %c (: 42 i64))
  (return %c))
```

**Verification**:
1. Dialect loads successfully
2. Transforms apply without errors
3. Generated LLVM matches manual version
4. Macro produces correct CValueLayout structures
5. Reduced line count vs original

**Tasks**:
- [ ] Create minimal test macro
- [ ] Verify transforms execute
- [ ] Compare generated LLVM with hand-written version
- [ ] Measure line count reduction

---

### Phase 4: Full Macro Rewrite

**Goal**: Rewrite `add_macro_with_strings_refactored.lisp` using mlsp dialect

**Metrics**:
- **Before**: 310 lines
- **Target**: < 20 lines
- **Reduction**: ~93%

**Tasks**:
- [ ] Port addMacro to mlsp operations
- [ ] Verify functionality is identical
- [ ] Document any limitations
- [ ] Performance comparison

---

## Open Questions & Challenges

### 1. PDL Multi-Operation Creation ✅ RESOLVED

**Answer**: YES! PDL `pdl.rewrite` regions can contain arbitrary operation sequences.

**How it works**:
- Within `pdl.rewrite` block, we can create multiple `pdl.operation` nodes
- Each represents a new payload operation to create
- Chain them together using SSA values
- Final `pdl.replace` swaps the matched op with the last created op

**Example**: `mlsp.identifier` can lower to full malloc+GEP+store sequence in a single PDL pattern

**No fallback needed** - PDL is sufficient for our needs!

---

### 2. Attribute vs Operand Handling

**Question**: Should string values be attributes or operands in mlsp operations?

**Current approach**: Attributes for constants (e.g., `{:value "name"}`)

**Trade-offs**:
- **Attributes**: Simpler syntax, but harder to compose
- **Operands**: More flexible, but requires constant materialization

**Decision**: Start with attributes, migrate to operands if needed

---

### 3. Type System

**Question**: Should mlsp have its own type system or reuse `!llvm.ptr`?

**Options**:
1. **All `!llvm.ptr`**: Simple, compatible, but loses type safety
2. **Custom types**: `!mlsp.value`, `!mlsp.list`, `!mlsp.identifier` - Better safety, more complex
3. **Hybrid**: Start with `!llvm.ptr`, add typed wrappers later

**Decision**: Start with `!llvm.ptr`, evaluate type safety needs later

---

### 4. Error Handling

**Question**: How do mlsp operations handle errors (e.g., invalid list index)?

**Options**:
1. **Runtime checks**: Add bounds checking in lowered LLVM code
2. **Static verification**: Use MLIR verifiers to catch errors at compile time
3. **Unchecked**: Assume correct usage (fastest, least safe)

**Decision**: Start unchecked, add verification if bugs emerge

---

### 5. Performance

**Question**: Will PDL transform overhead affect JIT compilation time?

**Considerations**:
- Current manual LLVM is verbose but straightforward
- Transform patterns add complexity
- Need to benchmark compilation time

**Mitigation**: If transforms are slow, add caching or pre-lowering pass

---

## Success Criteria

### Minimum Viable Product (MVP)

- [x] IRDL dialect loads successfully (proven with demo_dialect_complete.lisp)
- [ ] `mlsp.identifier` operation works end-to-end
- [ ] `mlsp.list` operation works end-to-end
- [ ] At least 50% line reduction on simple macro

### Full Success

- [ ] All core operations implemented
- [ ] `mlsp.build_operation` works correctly
- [ ] add_macro rewritten with 90%+ reduction
- [ ] Generated code is functionally equivalent
- [ ] Compilation time is acceptable (< 2x slowdown)

### Stretch Goals

- [ ] `mlsp.extract_args` pattern matching
- [ ] Type-safe `!mlsp.value` types
- [ ] Documentation and tutorial
- [ ] Port all builtin macros to mlsp dialect

---

## Timeline Estimate

| Phase | Estimated Time | Dependencies |
|-------|---------------|--------------|
| Phase 1: IRDL Definition | ✅ Complete | demo_dialect_complete.lisp |
| Phase 2: PDL Transforms | 2-3 days | Phase 1, PDL research |
| Phase 3: Integration Testing | 1 day | Phase 2 |
| Phase 4: Full Rewrite | 1-2 days | Phase 3 |

**Total**: ~5-7 days of focused work

---

## Next Steps

1. **Define complete mlsp dialect** - Add all operations to IRDL
2. **Research PDL multi-op creation** - Determine lowering strategy
3. **Implement mlsp.identifier lowering** - Prove the concept works
4. **Test end-to-end** - Verify transforms apply correctly
5. **Iterate and expand** - Add remaining operations

---

## References

- ✅ `examples/demo_dialect_complete.lisp` - Working IRDL + PDL example
- ✅ `examples/add_macro_with_strings_refactored.lisp` - Current macro implementation (310 lines)
- 🔗 [MLIR IRDL Documentation](https://mlir.llvm.org/docs/Dialects/IRDL/)
- 🔗 [MLIR PDL Documentation](https://mlir.llvm.org/docs/Dialects/PDLOps/)
- 🔗 [MLIR Transform Dialect](https://mlir.llvm.org/docs/Dialects/Transform/)
