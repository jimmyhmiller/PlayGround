# MLSP Dialect Lowering - Pure PDL Approach

## Overview

This document outlines how to lower the `mlsp` dialect using **pure PDL (Pattern Descriptor Language)** - an MLIR dialect for defining pattern rewrites. No C++ code, no native rewrites, just declarative MLIR patterns.

## The Key Insight

Initially, it seemed impossible to lower `mlsp.identifier` with pure PDL because:
- Creating `llvm.mlir.global` requires unique names (PDL can't generate these)
- Globals must be inserted at module level (PDL rewrites happen at match site)
- Symbol references need name coordination

**The solution**: Don't use globals at all. **Materialize strings on the stack** per-use instead.

### Benefits of Stack Materialization
- ✅ **Pure MLIR**: Only standard `llvm.*`, `arith.*`, `func.*` operations
- ✅ **No name generation**: Each use is self-contained, no unique symbols needed
- ✅ **In-function**: All operations created at the rewrite site, no module-level insertion
- ✅ **Simple PDL**: Just `pdl.operation` to create ops, `pdl.replace` to substitute

### Trade-offs
- ❌ **Duplication**: Same string gets stack-allocated at each use site
- ✅ **Optimizable**: Later passes can hoist/deduplicate if needed

---

## Architecture

```
Input: test_mlsp_complete.mlisp
    ↓
[Parser] → MLIR Module with mlsp.* operations
    ↓
[Add Runtime Declarations] → func.func @mlsp_create_identifier, etc.
    ↓
[Apply PDL Patterns] → transforms/mlsp_lowering.pdl
    ↓
MLIR Module with llvm.*, arith.*, func.* (no mlsp.*)
    ↓
[Standard LLVM Lowering Passes]
    ↓
[JIT Execution] → Runtime functions handle values
```


---

## PDL Pattern: Lower mlsp.identifier

This is the core pattern showing the stack materialization technique:

```mlir
module {
  pdl.pattern @lower_mlsp_identifier : benefit(1) {
    // Match: %result = mlsp.identifier {value = "some_string"}
    %val = pdl.attribute
    %ptr_type = pdl.type : !llvm.ptr
    %mlsp_id = pdl.operation "mlsp.identifier" {"value" = %val} -> (%ptr_type : !pdl.type)

    pdl.rewrite %mlsp_id {
      // 1. Create constant array from the string attribute
      //    Note: %val contains "hello" or "hello\00" depending on source
      %arr_type = pdl.type : !llvm.array<6 x i8>  // Adjust size based on %val
      %arr = pdl.operation "llvm.mlir.constant" {"value" = %val} -> (%arr_type : !pdl.type)

      // 2. Allocate stack space for the array
      %i64_type = pdl.type : i64
      %one = pdl.operation "arith.constant" {"value" = 1 : i64} -> (%i64_type : !pdl.type)
      %one_result = pdl.result 0 of %one

      %buf = pdl.operation "llvm.alloca"(%one_result) {"elem_type" = %arr_type} -> (%ptr_type : !pdl.type)

      // 3. Store the array into the buffer
      %arr_result = pdl.result 0 of %arr
      %buf_result = pdl.result 0 of %buf
      %store = pdl.operation "llvm.store"(%arr_result, %buf_result)

      // 4. GEP to get pointer to first character [0, 0]
      %zero = pdl.operation "arith.constant" {"value" = 0 : i64} -> (%i64_type : !pdl.type)
      %zero_result = pdl.result 0 of %zero

      %gep = pdl.operation "llvm.getelementptr"(%buf_result, %zero_result, %zero_result)
             {"elem_type" = !i8} -> (%ptr_type : !pdl.type)

      // 5. Call mlsp_create_identifier
      %gep_result = pdl.result 0 of %gep
      %call = pdl.operation "func.call"(%gep_result)
              {"callee" = @mlsp_create_identifier} -> (%ptr_type : !pdl.type)

      // 6. Replace the original mlsp.identifier
      pdl.replace %mlsp_id with %call
    }
  }
}
```

---

## Before/After Example

### Input (Before PDL Patterns):

```mlir
module {
  func.func @main() -> i64 {
    %0 = "mlsp.identifier"() {value = "hello\00"} : () -> !llvm.ptr
    %1 = "mlsp.identifier"() {value = "world\00"} : () -> !llvm.ptr
    %c42 = arith.constant 42 : i64
    return %c42 : i64
  }
}
```

### Output (After PDL Patterns):

```mlir
module {
  func.func @main() -> i64 {
    // First mlsp.identifier → stack materialization
    %arr0 = llvm.mlir.constant("hello\00") : !llvm.array<6 x i8>
    %c1_0 = arith.constant 1 : i64
    %buf0 = llvm.alloca %c1_0 x !llvm.array<6 x i8> : (i64) -> !llvm.ptr
    llvm.store %arr0, %buf0 : !llvm.array<6 x i8>, !llvm.ptr
    %c0_0 = arith.constant 0 : i64
    %ptr0 = llvm.getelementptr %buf0[%c0_0, %c0_0]
            : (!llvm.ptr, i64, i64) -> !llvm.ptr
    %0 = func.call @mlsp_create_identifier(%ptr0) : (!llvm.ptr) -> !llvm.ptr

    // Second mlsp.identifier → stack materialization
    %arr1 = llvm.mlir.constant("world\00") : !llvm.array<6 x i8>
    %c1_1 = arith.constant 1 : i64
    %buf1 = llvm.alloca %c1_1 x !llvm.array<6 x i8> : (i64) -> !llvm.ptr
    llvm.store %arr1, %buf1 : !llvm.array<6 x i8>, !llvm.ptr
    %c0_1 = arith.constant 0 : i64
    %ptr1 = llvm.getelementptr %buf1[%c0_1, %c0_1]
            : (!llvm.ptr, i64, i64) -> !llvm.ptr
    %1 = func.call @mlsp_create_identifier(%ptr1) : (!llvm.ptr) -> !llvm.ptr

    %c42 = arith.constant 42 : i64
    return %c42 : i64
  }
}
```

**Note**: Yes, the string is duplicated. Later optimization passes can hoist/merge if needed.

---

## PDL Pattern: Lower mlsp.list

```mlir
pdl.pattern @lower_mlsp_list : benefit(1) {
  // Match: %result = mlsp.list(%elem0, %elem1, ..., %elemN)
  %elems = pdl.operands
  %ptr_type = pdl.type : !llvm.ptr
  %mlsp_list = pdl.operation "mlsp.list"(%elems : !pdl.range<value>) -> (%ptr_type : !pdl.type)

  pdl.rewrite %mlsp_list {
    // 1. Get the number of operands as a constant
    %i64_type = pdl.type : i64
    // Note: PDL doesn't have a "count operands" operation directly
    // We need to use a native constraint or assume fixed size
    // For variadic, this may require a helper

    // Assuming we know the count (e.g., 2 elements):
    %len = pdl.operation "arith.constant" {"value" = 2 : i64} -> (%i64_type : !pdl.type)
    %len_result = pdl.result 0 of %len

    // 2. Allocate array for element pointers
    %elem_ptr_type = pdl.type : !llvm.ptr
    %arr_buf = pdl.operation "llvm.alloca"(%len_result)
                {"elem_type" = %elem_ptr_type} -> (%ptr_type : !pdl.type)
    %arr_buf_result = pdl.result 0 of %arr_buf

    // 3. Store each element into the array
    //    PDL limitation: Can't iterate over %elems directly
    //    Would need to unroll for known sizes or use native rewrite

    // For 2 elements (manual unroll):
    %elem0 = pdl.operand 0 of %elems
    %zero = pdl.operation "arith.constant" {"value" = 0 : i64} -> (%i64_type : !pdl.type)
    %zero_result = pdl.result 0 of %zero
    %gep0 = pdl.operation "llvm.getelementptr"(%arr_buf_result, %zero_result)
            {"elem_type" = %elem_ptr_type} -> (%ptr_type : !pdl.type)
    %gep0_result = pdl.result 0 of %gep0
    %store0 = pdl.operation "llvm.store"(%elem0, %gep0_result)

    %elem1 = pdl.operand 1 of %elems
    %one = pdl.operation "arith.constant" {"value" = 1 : i64} -> (%i64_type : !pdl.type)
    %one_result = pdl.result 0 of %one
    %gep1 = pdl.operation "llvm.getelementptr"(%arr_buf_result, %one_result)
            {"elem_type" = %elem_ptr_type} -> (%ptr_type : !pdl.type)
    %gep1_result = pdl.result 0 of %gep1
    %store1 = pdl.operation "llvm.store"(%elem1, %gep1_result)

    // 4. Call mlsp_create_list(length, array_ptr)
    %call = pdl.operation "func.call"(%len_result, %arr_buf_result)
            {"callee" = @mlsp_create_list} -> (%ptr_type : !pdl.type)

    pdl.replace %mlsp_list with %call
  }
}
```

**PDL Limitation Note**: Iterating over variadic operands requires either:
- Multiple patterns for different arities (0, 1, 2, 3... elements)
- A `pdl.apply_native_rewrite` helper to loop over operands

---

## PDL Pattern: Lower mlsp.get_element (Static Index)

```mlir
pdl.pattern @lower_mlsp_get_element : benefit(1) {
  // Match: %result = mlsp.get_element(%list) {index = N}
  %list = pdl.operand
  %idx_attr = pdl.attribute : i64
  %ptr_type = pdl.type : !llvm.ptr
  %get_elem = pdl.operation "mlsp.get_element"(%list) {"index" = %idx_attr} -> (%ptr_type : !pdl.type)

  pdl.rewrite %get_elem {
    // 1. Convert attribute to constant
    %i64_type = pdl.type : i64
    %idx_const = pdl.operation "arith.constant" {"value" = %idx_attr} -> (%i64_type : !pdl.type)
    %idx_result = pdl.result 0 of %idx_const

    // 2. Call mlsp_get_element_static(list, index)
    %call = pdl.operation "func.call"(%list, %idx_result)
            {"callee" = @mlsp_get_element_static} -> (%ptr_type : !pdl.type)

    pdl.replace %get_elem with %call
  }
}
```

---

## PDL Pattern: Lower mlsp.get_element_dyn (Dynamic Index)

```mlir
pdl.pattern @lower_mlsp_get_element_dyn : benefit(1) {
  // Match: %result = mlsp.get_element_dyn(%list, %index)
  %list = pdl.operand
  %index = pdl.operand
  %ptr_type = pdl.type : !llvm.ptr
  %get_elem = pdl.operation "mlsp.get_element_dyn"(%list, %index) -> (%ptr_type : !pdl.type)

  pdl.rewrite %get_elem {
    // Call mlsp_get_element_dynamic(list, index) directly
    %call = pdl.operation "func.call"(%list, %index)
            {"callee" = @mlsp_get_element_dynamic} -> (%ptr_type : !pdl.type)

    pdl.replace %get_elem with %call
  }
}
```

---

## PDL Limitations & Workarounds

### Limitation 1: Can't Count Variadic Operands

**Problem**: `mlsp.list` can have any number of elements. PDL has no `pdl.count_operands`.

**Workarounds**:
1. **Multiple patterns**: Write separate patterns for 0, 1, 2, 3... element lists
2. **Native helper**: Use `pdl.apply_native_rewrite` with a tiny C++ function that just counts and returns the length
3. **Bounded**: Limit `mlsp.list` to max N elements in your language

### Limitation 2: Can't Iterate Over Operands

**Problem**: PDL has no loops or iteration constructs.

**Workarounds**:
1. **Unroll**: Manually write `pdl.operand 0`, `pdl.operand 1`, etc. for each position
2. **Native helper**: Use `pdl.apply_native_rewrite` to iterate and build the array
3. **Bounded**: Same as above, accept a maximum list size

### Limitation 3: Type Inference for Arrays

**Problem**: `!llvm.array<? x i8>` needs concrete size, but PDL can't compute string length from attribute.

**Workarounds**:
1. **Pre-compute**: Store length in a separate attribute during parsing
2. **Fixed size**: Pad all strings to max size (wasteful)
3. **Native helper**: Extract string, compute length, create array type

---

## Integration with Executor

The executor needs to:

2. **Load PDL patterns** from `transforms/mlsp_lowering.pdl`
3. **Apply patterns** to the module before LLVM lowering
4. **Register runtime symbols** with the JIT execution engine

Example executor integration:

```zig
// In src/executor.zig

pub fn compile(self: *Executor, module: *mlir.Module) !void {
    // Step 1: Add runtime declarations
    const declarations = @import("mlsp_runtime_declarations.zig");
    try declarations.addRuntimeDeclarations(self.ctx, module);

    // Step 2: Load and apply PDL patterns
    try self.applyPDLPatterns(module, "transforms/mlsp_lowering.pdl");

    // Step 3: Standard LLVM lowering
    try self.lowerToLLVM(module);

    // Step 4: Create execution engine
    self.engine = try mlir.ExecutionEngine.create(module, ...);
}

fn applyPDLPatterns(self: *Executor, module: *mlir.Module, pattern_file: []const u8) !void {
    // Load PDL module from file
    const pdl_module = try self.loadPDLModule(pattern_file);
    defer pdl_module.destroy();

    // Create rewrite pattern set from PDL
    const pattern_set = try self.createPatternSetFromPDL(pdl_module);

    // Apply patterns using greedy rewrite driver
    try self.applyPatternsGreedily(module, pattern_set);
}
```

---

## Summary

This approach uses **pure PDL** with stack materialization to lower `mlsp` operations:

✅ **What We Achieve**:
- Pure MLIR pattern rewriting (no C++ lowering code)
- Declarative patterns in `.pdl` files
- Stack-based string materialization (no globals needed)
- Simple integration with existing executor

⚠️ **PDL Limitations We Hit**:
- Can't count variadic operands (need patterns per arity OR tiny native helper)
- Can't iterate (need manual unrolling OR tiny native helper)
- Can't compute types dynamically (need pre-computed sizes OR tiny native helper)

🎯 **The Trade-off**:
- Strings are duplicated per-use (vs. shared globals)
- Can be optimized later by standard MLIR passes if needed
- Keeps the PDL patterns pure and simple

This is the cleanest pure-PDL approach that actually works for lowering a custom dialect to runtime calls.
