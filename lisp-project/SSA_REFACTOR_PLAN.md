# SSA Name Refactoring Plan

**Bug ID:** somber-stupid-caterpillar  
**Status:** Implementation needed  
**Priority:** High

## Problem Summary

MLIR builder expects numeric SSA indices (0, 1, 2...) but we've switched to symbolic names (%0, %1, %arg0) to match actual MLIR syntax. This causes crashes when compiling.

## Implementation Steps

### 1. Update ValueTracker (Lines 182-215)

**Current:** Array-based tracker using numeric indices
```lisp
(def ValueTracker (: Type)
  (Struct
    [values (Array MlirValue 256)]
    [count I32]))

(def value-tracker-lookup (: (-> [(Pointer ValueTracker) I32] MlirValue))
  (fn [tracker idx] ...))
```

**Needed:** HashMap-based tracker using string keys
```lisp
;; Simple linked-list based map for SSA name -> MlirValue
(def ValueMapEntry (: Type)
  (Struct
    [name (Pointer U8)]
    [value MlirValue]
    [next (Pointer ValueMapEntry)]))

(def ValueTracker (: Type)
  (Struct
    [head (Pointer ValueMapEntry)]))

(def value-tracker-register (: (-> [(Pointer ValueTracker) (Pointer U8) MlirValue] I32))
  (fn [tracker name value]
    ;; Create new entry at head of linked list
    ))

(def value-tracker-lookup (: (-> [(Pointer ValueTracker) (Pointer U8)] MlirValue))
  (fn [tracker name]
    ;; Walk linked list comparing names with strcmp
    ))
```

### 2. Update add-result-types-to-state (Lines 397-425)

**Current:** Expects vector of type strings: `[i32]`
```lisp
result-types-val → [i32]
```

**Needed:** Expects vector of `[name type]` pairs: `[%0 i32]`
```lisp
result-types-val → [[%0 i32]]

;; For each result-type pair:
(let [pair (vector-element result-types-val idx)  ; [%0 i32]
      name-elem (vector-element pair 0)           ; %0
      type-elem (vector-element pair 1)           ; i32
      name-str (pointer-field-read name-elem str_val)
      type-str (pointer-field-read type-elem str_val)]
  ;; Parse type, add to state, return name for registration
  )
```

### 3. Update add-operands-to-state (Lines 427-457)

**Current:** Uses atoi() to convert strings to indices
```lisp
operand-str (: (Pointer U8)) (pointer-field-read elem str_val)
operand-idx (: I32) (atoi operand-str)  ; ← FAILS on "%0"
operand-val (: MlirValue) (value-tracker-lookup tracker operand-idx)
```

**Needed:** Look up by name directly
```lisp
operand-name (: (Pointer U8)) (pointer-field-read elem str_val)  ; "%0"
operand-val (: MlirValue) (value-tracker-lookup tracker operand-name)
```

### 4. Update build-mlir-block (Lines 364-395)

**Current:** Creates blocks with 0 arguments, never registers them
```lisp
block (: MlirBlock) (mlirBlockCreate 0 ...)  ; ← Always 0 args
```

**Needed:** Parse block args and register them in tracker
```lisp
;; Get block args from BlockNode
(let [args-val (: (Pointer types/Value)) (pointer-field-read block-node args)]
  ;; args-val is [[%arg0 i32]] - vector of [name type] pairs
  (if (= args-tag types/ValueTag/Vector)
    (let [vec-ptr ...
          count ... ; Number of arguments
          arg-types (: (Array MlirType 8)) (array MlirType 8)]
      ;; Parse each argument: [%arg0 i32]
      (while (< idx count)
        (let [arg-pair (vector-element args-val idx)  ; [%arg0 i32]
              arg-name-elem (vector-element arg-pair 0)  ; %arg0
              arg-type-elem (vector-element arg-pair 1)  ; i32
              arg-name (pointer-field-read arg-name-elem str_val)
              arg-type-str (pointer-field-read arg-type-elem str_val)
              mlir-type (parse-type-string builder arg-type-str)]
          (array-set! arg-types idx mlir-type)
          (set! idx (+ idx 1))))
      
      ;; Create block with arguments
      (let [block (: MlirBlock) (mlirBlockCreate count (array-ptr arg-types 0) ...)]
        ;; Register each block argument in tracker
        (let [idx (: I32) 0]
          (while (< idx count)
            (let [arg-pair (vector-element args-val idx)
                  arg-name-elem (vector-element arg-pair 0)
                  arg-name (pointer-field-read arg-name-elem str_val)
                  arg-value (: MlirValue) (mlirBlockGetArgument block (cast I64 idx))]
              (value-tracker-register tracker arg-name arg-value)
              (set! idx (+ idx 1)))))
        block))))
```

### 5. Update build-mlir-operation (Lines 526-570)

**Needed:** Register operation results by name from result-types
```lisp
;; After creating operation:
(let [op (: MlirOperation) (mlirOperationCreate state-ptr)]
  (mlirBlockAppendOwnedOperation parent-block op)
  
  ;; Register each result by name from result-types vector
  (let [result-types-val (pointer-field-read op-node result-types)
        result-tag (pointer-field-read result-types-val tag)]
    (if (= result-tag types/ValueTag/Vector)
      (let [idx (: I32) 0
            count ...]
        (while (< idx count)
          (let [pair (vector-element result-types-val idx)  ; [%0 i32]
                name-elem (vector-element pair 0)
                name-str (pointer-field-read name-elem str_val)
                result-val (: MlirValue) (mlirOperationGetResult op (cast I64 idx))]
            (value-tracker-register tracker name-str result-val)
            (set! idx (+ idx 1)))))
      op)))
```

## Helper Functions Needed

```lisp
;; Extract element from vector by index
(def vector-element (: (-> [(Pointer types/Value) I32] (Pointer types/Value)))
  (fn [vec-val idx]
    (let [vec-ptr (pointer-field-read vec-val vec_val)
          vector-struct (cast (Pointer types/Vector) vec-ptr)
          data (pointer-field-read vector-struct data)
          elem-offset (* (cast I64 idx) 8)
          elem-ptr-loc (cast (Pointer U8) (+ (cast I64 data) elem-offset))
          elem-ptr-ptr (cast (Pointer (Pointer types/Value)) elem-ptr-loc)]
      (dereference elem-ptr-ptr))))
```

## Testing Strategy

1. Start with simple.lisp (no block args, simple SSA values)
2. Then add.lisp (multiple SSA values, arithmetic)
3. Then test_arg.lisp (block arguments)
4. Finally fib.lisp (complex control flow, nested blocks)

## Files to Modify

- `src/mlir_builder.lisp` - Main implementation
- `tests/*.lisp` - Already updated with new syntax ✓

## Dependencies

- Need strcmp from string.h (already declared)
- Need malloc for linked list entries (already declared)
- MLIR C API functions (already declared)
