# Operation Flattening: WAST-Style Nested Operations

## Motivation

MLIR uses strict SSA (Static Single Assignment) form, which requires:
- Every value must be defined before use
- Operations cannot be nested - each operation is a separate statement
- All operations must bind their results to SSA values (like `%0`, `%result`)

However, WebAssembly's text format (WAST/WAT) supports **nested expressions** for convenience:
```wasm
(func $add (result i32)
  (i32.add
    (i32.const 10)
    (i32.const 32)))
```

This is much more readable than the flattened SSA form. We want to bring this convenience to our MLIR-Lisp while still generating valid MLIR IR.

## Pipeline Integration

The operation flattener fits into the compilation pipeline as follows:

```
Reader → Macro Expander → Operation Flattener → Parser → Builder
```

**Why after macro expansion?**
- Macros like `(call @func i64)` expand into full `(operation ...)` forms
- The flattener needs to see actual operations with result-bindings
- Macro-generated operations can themselves be nested

**Why before parsing?**
- Works on generic Value AST structures (easier to manipulate)
- Parser continues to expect flat, sequential operations
- Builder continues to use simple value_map lookups

## Core Algorithm

### High-Level Overview

1. **Traverse** all blocks in the module
2. **For each block's operations list**, process sequentially (left-to-right)
3. **For each operation**, check its `(operands ...)` clause
4. **If an operand is** `(operation ...)`:
   - Recursively flatten the nested operation (depth-first)
   - Extract or generate result-bindings
   - Hoist the flattened operation(s) before the parent
   - Replace the nested operation with its result value ID
5. **Return** the flattened operations list

### Gensym for Auto-Generated Bindings

When a nested operation lacks explicit `(result-bindings ...)`, we auto-generate:
- Format: `%result_G0`, `%result_G1`, `%result_G2`, etc.
- Counter increments for each auto-generated binding
- User-provided bindings are always preserved (never overwritten)

### Evaluation Order Guarantee

Operations are flattened **depth-first, left-to-right**:
- Ensures dependencies are satisfied (nested operands flattened first)
- Preserves evaluation order for operations with side effects
- Left operands are evaluated before right operands

---

## Detailed Examples

### Example 1: Single Nested Operand

**Input (after macro expansion):**
```lisp
(block
  (arguments [])
  (operation
    (name func.return)
    (operands (operation
               (name arith.constant)
               (result-types i64)
               (attributes { :value (: 99 i64) })))))
```

**Output (after flattening):**
```lisp
(block
  (arguments [])
  (operation
    (name arith.constant)
    (result-bindings [%result_G0])
    (result-types i64)
    (attributes { :value (: 99 i64) }))
  (operation
    (name func.return)
    (operands %result_G0)))
```

**What happened:**
1. Detected nested `arith.constant` in operands of `func.return`
2. No result-bindings present → generated `%result_G0`
3. Hoisted `arith.constant` operation before `func.return`
4. Replaced nested operation with `%result_G0`

---

### Example 2: Multiple Nested Operands

**Input:**
```lisp
(block
  (arguments [])
  (operation
    (name arith.addi)
    (result-bindings [%sum])
    (result-types i64)
    (operands
      (operation
        (name arith.constant)
        (result-types i64)
        (attributes { :value (: 10 i64) }))
      (operation
        (name arith.constant)
        (result-types i64)
        (attributes { :value (: 32 i64) })))))
```

**Output:**
```lisp
(block
  (arguments [])
  (operation
    (name arith.constant)
    (result-bindings [%result_G0])
    (result-types i64)
    (attributes { :value (: 10 i64) }))
  (operation
    (name arith.constant)
    (result-bindings [%result_G1])
    (result-types i64)
    (attributes { :value (: 32 i64) }))
  (operation
    (name arith.addi)
    (result-bindings [%sum])
    (result-types i64)
    (operands %result_G0 %result_G1)))
```

**What happened:**
1. Processed operands left-to-right
2. First operand: nested `arith.constant` → generated `%result_G0`, hoisted
3. Second operand: nested `arith.constant` → generated `%result_G1`, hoisted
4. Both nested operations replaced with their generated bindings
5. Original `arith.addi` now has flat operands: `%result_G0 %result_G1`

**Evaluation order:** First constant (10), then second constant (32), then addition

---

### Example 3: Mixed Nested and Value IDs

**Input:**
```lisp
(block
  (arguments [%x : i64])
  (operation
    (name arith.addi)
    (result-bindings [%result])
    (result-types i64)
    (operands
      (operation
        (name arith.constant)
        (result-types i64)
        (attributes { :value (: 42 i64) }))
      %x)))
```

**Output:**
```lisp
(block
  (arguments [%x : i64])
  (operation
    (name arith.constant)
    (result-bindings [%result_G0])
    (result-types i64)
    (attributes { :value (: 42 i64) }))
  (operation
    (name arith.addi)
    (result-bindings [%result])
    (result-types i64)
    (operands %result_G0 %x)))
```

**What happened:**
1. First operand: nested operation → generated `%result_G0`, hoisted
2. Second operand: value ID `%x` → passed through unchanged
3. Final operands: `%result_G0 %x`

**Key insight:** Existing value IDs are never modified, only nested operations are transformed

---

### Example 4: Deeply Nested Operations (3 Levels)

**Input:**
```lisp
(block
  (arguments [])
  (operation
    (name arith.muli)
    (result-bindings [%product])
    (result-types i64)
    (operands
      (operation
        (name arith.addi)
        (result-types i64)
        (operands
          (operation
            (name arith.constant)
            (result-types i64)
            (attributes { :value (: 5 i64) }))
          (operation
            (name arith.constant)
            (result-types i64)
            (attributes { :value (: 3 i64) }))))
      (operation
        (name arith.constant)
        (result-types i64)
        (attributes { :value (: 2 i64) })))))
```

**Output:**
```lisp
(block
  (arguments [])
  (operation
    (name arith.constant)
    (result-bindings [%result_G0])
    (result-types i64)
    (attributes { :value (: 5 i64) }))
  (operation
    (name arith.constant)
    (result-bindings [%result_G1])
    (result-types i64)
    (attributes { :value (: 3 i64) }))
  (operation
    (name arith.addi)
    (result-bindings [%result_G2])
    (result-types i64)
    (operands %result_G0 %result_G1))
  (operation
    (name arith.constant)
    (result-bindings [%result_G3])
    (result-types i64)
    (attributes { :value (: 2 i64) }))
  (operation
    (name arith.muli)
    (result-bindings [%product])
    (result-types i64)
    (operands %result_G2 %result_G3)))
```

**What happened:**
1. First operand of `muli` is nested `addi`
2. Recursively process `addi`:
   - Its first operand: constant 5 → `%result_G0`
   - Its second operand: constant 3 → `%result_G1`
   - `addi` itself → `%result_G2` (auto-generated since no bindings)
3. Second operand of `muli`: constant 2 → `%result_G3`
4. Final `muli` operands: `%result_G2 %result_G3`

**Evaluation order:** 5, 3, (5+3), 2, ((5+3)*2)

**Depth-first guarantee:** Inner nested operations are fully flattened before outer ones

---

### Example 5: User-Provided Bindings are Preserved

**Input:**
```lisp
(block
  (arguments [])
  (operation
    (name func.return)
    (operands
      (operation
        (name arith.constant)
        (result-bindings [%my_constant])
        (result-types i64)
        (attributes { :value (: 99 i64) })))))
```

**Output:**
```lisp
(block
  (arguments [])
  (operation
    (name arith.constant)
    (result-bindings [%my_constant])
    (result-types i64)
    (attributes { :value (: 99 i64) }))
  (operation
    (name func.return)
    (operands %my_constant)))
```

**What happened:**
1. Nested operation already has `(result-bindings [%my_constant])`
2. Used the existing binding (no auto-generation)
3. Hoisted the operation as-is
4. Replaced with `%my_constant`

**Key rule:** User-provided bindings are **always preserved**, never overwritten by gensym

---

### Example 6: Macro-Generated Operations with Bindings

**Input (using `call` macro):**
```lisp
(block
  (arguments [])
  (operation
    (name arith.addi)
    (result-types i64)
    (operands
      (call @get_number i64)
      (call @get_number i64))))
```

**After macro expansion (before flattening):**
```lisp
(block
  (arguments [])
  (operation
    (name arith.addi)
    (result-types i64)
    (operands
      (operation
        (name func.call)
        (result-bindings [%result0])
        (result-types i64)
        (attributes { :callee @get_number }))
      (operation
        (name func.call)
        (result-bindings [%result0])
        (result-types i64)
        (attributes { :callee @get_number })))))
```

**After flattening:**
```lisp
(block
  (arguments [])
  (operation
    (name func.call)
    (result-bindings [%result0])
    (result-types i64)
    (attributes { :callee @get_number }))
  (operation
    (name func.call)
    (result-bindings [%result0])
    (result-types i64)
    (attributes { :callee @get_number }))
  (operation
    (name arith.addi)
    (result-types i64)
    (operands %result0 %result0)))
```

**What happened:**
1. Both nested `func.call` operations already have `(result-bindings [%result0])` from macro
2. Preserved both bindings as `%result0`
3. Both calls hoisted before `arith.addi`
4. Final operands: `%result0 %result0`

**Note:** Both calls use the same binding name. This is valid MLIR - the second `%result0` shadows the first.

---

## Edge Cases and Multiple Results

### Multiple Result Bindings

MLIR operations can produce multiple results:
```lisp
(operation
  (name some.multi_result_op)
  (result-bindings [%r1 %r2 %r3])
  (result-types i32 i64 f32))
```

**When nested as an operand, which result should we use?**

**Rule:** Use the **first result** (index 0)

**Example:**
```lisp
(operation
  (name some.consumer)
  (operands
    (operation
      (name some.multi_result_op)
      (result-bindings [%r1 %r2 %r3])
      (result-types i32 i64 f32))))
```

**Flattens to:**
```lisp
(operation
  (name some.multi_result_op)
  (result-bindings [%r1 %r2 %r3])
  (result-types i32 i64 f32))
(operation
  (name some.consumer)
  (operands %r1))
```

**Future extension:** If needed, we could add explicit result selection:
```lisp
(operands (operation-result 1 (operation ...)))
```
This would use `%r2` (index 1) instead of the default `%r1` (index 0).

---

## Technical Specification

### Data Structures

```zig
pub const OperationFlattener = struct {
    allocator: std.mem.Allocator,
    gensym_counter: usize,

    pub fn init(allocator: std.mem.Allocator) OperationFlattener {
        return .{
            .allocator = allocator,
            .gensym_counter = 0,
        };
    }

    pub fn gensym(self: *OperationFlattener) ![]const u8 {
        const counter = self.gensym_counter;
        self.gensym_counter += 1;
        return try std.fmt.allocPrint(
            self.allocator,
            "%result_G{d}",
            .{counter}
        );
    }

    pub fn flattenModule(self: *OperationFlattener, value: *Value) !*Value
    fn flattenBlock(self: *OperationFlattener, block: *Value) !*Value
    fn flattenOperations(self: *OperationFlattener, ops: []const *Value) ![]const *Value
    fn flattenOperation(self: *OperationFlattener, op: *Value) !FlattenResult
};

const FlattenResult = struct {
    operations: []const *Value,  // Hoisted operations (0 or more)
    value_id: *Value,             // The result value ID to use
};
```

### Value AST Transformation

Operations are represented as Value structures:
```zig
// Nested operation (before flattening)
Value{
    .type = .list,
    .data = .{ .list = [...] }  // (operation ...)
}

// After flattening, operand becomes:
Value{
    .type = .value_id,
    .data = .{ .atom = "%result_G0" }
}
```

### Gensym Format

- Pattern: `%result_G{counter}`
- Examples: `%result_G0`, `%result_G1`, `%result_G2`, ...
- Counter is global within a flattening pass
- Unique across all auto-generated bindings

---

## Limitations and Error Cases

### 1. Operations Without Results

If a nested operation has no `result-types` and no `result-bindings`, it's an error:

```lisp
(operation
  (name func.return)
  (operands
    (operation
      (name func.return)  ; func.return has no results!
      (operands %x))))
```

**Error:** "Nested operation 'func.return' in operand position must produce a result"

### 2. Side Effects and Evaluation Order

The flattener guarantees left-to-right evaluation, but it cannot reason about side effects:

```lisp
(operation
  (name arith.addi)
  (operands
    (operation (name llvm.call) ...)  ; May have side effects
    (operation (name llvm.call) ...))) ; May have side effects
```

Both calls will be hoisted and executed left-to-right, but the flattener doesn't know or enforce this at a semantic level.

### 3. Type Checking

The flattener does **not** perform type checking. It trusts that:
- Nested operations have valid result-types
- The types match what the parent operation expects

Type errors will be caught later by MLIR validation.

---

## Testing Strategy

### Unit Tests (`test/operation_flattener_test.zig`)

1. **Single nested operand**
   - One nested operation
   - Auto-generated binding

2. **Multiple nested operands**
   - Two nested operations in one operation's operands
   - Both get auto-generated bindings
   - Correct left-to-right order

3. **Mixed nested and value IDs**
   - Some operands nested, some are value IDs
   - Value IDs pass through unchanged

4. **Deeply nested (3 levels)**
   - Operation containing nested operation containing nested operation
   - All flattened correctly
   - Correct dependency order

5. **User-provided bindings**
   - Nested operation with explicit result-bindings
   - Binding is preserved (not overwritten)

6. **Macro-generated operations**
   - Nested `(call ...)` after macro expansion
   - Uses macro-provided bindings

7. **Error case: No results**
   - Nested operation without result-types or result-bindings
   - Should error appropriately

### Integration Tests

1. **Full compilation pipeline**
   - Source with nested operations → compile → execute
   - Verify correct MLIR IR generated
   - Verify correct execution results

2. **Complex real-world examples**
   - Nested arithmetic expressions
   - Function calls with nested arguments
   - Control flow with nested conditions

---

## Future Extensions

### 1. Explicit Result Selection

For operations with multiple results:
```lisp
(operands (result 1 (operation ...)))
```
Would select the second result instead of the first.

### 2. Nested Operations in Other Positions

Currently only supports nesting in `(operands ...)`. Could extend to:
- Attributes (if they can contain runtime values)
- Successors (for branch destinations)

### 3. Type Inference

Could infer result-types from context:
```lisp
; Function returns i64, infer constant type
(operation (name func.return)
  (operands (operation (name arith.constant)
                       (attributes { :value 99 }))))
```

Would auto-detect that the constant should be `i64` based on function signature.

---

## Implementation Checklist

- [x] Create `src/operation_flattener.zig`
- [x] Implement `OperationFlattener` struct with gensym
- [x] Implement `flattenModule()` - entry point
- [x] Implement `flattenBlock()` - processes block operations
- [x] Implement `flattenOperations()` - main algorithm
- [x] Implement `flattenOperation()` - handles single operation
- [x] Add helper functions for Value AST manipulation
- [x] Integrate into `main.zig` pipeline (after macro expansion)
- [x] Create `test/operation_flattener_test.zig`
- [x] Write all unit tests from testing strategy
- [ ] Create integration test example files
- [ ] Verify all examples in this doc work correctly
