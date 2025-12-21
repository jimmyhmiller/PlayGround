# MLIR Workarounds in Lispier

This document explains workarounds in `ir_gen.rs` and their status.

## The Problem: Melior Doesn't Expose MLIR Metadata

MLIR operations are defined in **ODS (Operation Definition Specification)**, a tablegen language that generates C++ code. Each operation definition includes:

- Whether it produces results (and how many)
- Whether operands are variadic (variable count) or optional
- What interfaces it implements (like `InferTypeOpInterface`)
- Constraints and verifiers

**The gap:** Melior wraps MLIR's C API, which is low-level and doesn't expose this metadata. So we can't ask "does this operation produce results?" or "does this operation have variadic operands?" - we just have to know.

---

## FIXED: Type Inference for Void Operations

### The Problem

Previously we had a hardcoded `is_void_operation()` function:

```rust
fn is_void_operation(name: &str) -> bool {
    matches!(name, "memref.store" | "memref.dealloc" | "func.func" | ...)
}
```

This was needed because `OperationBuilder::enable_result_type_inference()` throws an error when called on operations that don't support it.

### The Solution

We added `operation_supports_type_inference()` to our Melior fork, which wraps MLIR's `mlirOperationImplementsInterfaceStatic()` with `mlirInferTypeOpInterfaceTypeID()`:

```rust
// In melior/src/context.rs
impl Context {
    /// Returns `true` if an operation supports type inference.
    pub fn operation_supports_type_inference(&self, operation_name: &str) -> bool {
        unsafe {
            mlirOperationImplementsInterfaceStatic(
                name.to_raw(),
                self.raw,
                mlirInferTypeOpInterfaceTypeID(),
            )
        }
    }
}
```

Now lispier uses this API instead of maintaining a hardcoded list:

```rust
// Check if operation supports type inference using MLIR's InferTypeOpInterface
let supports_inference = context.operation_supports_type_inference(name);

// Enable type inference only if operation supports it
if supports_inference {
    op_builder = op_builder.enable_result_type_inference();
}
```

---

## Remaining Workaround: `operandSegmentSizes`

### What It Does

```rust
if name == "gpu.launch" {
    let segment_sizes = Attribute::parse(context,
        "array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>")?;
    op_builder = op_builder.add_attributes(&[(
        Identifier::new(context, "operandSegmentSizes"),
        segment_sizes,
    )]);
}
```

Hardcodes how operands are grouped for specific operations.

### Why It Exists

Some MLIR operations have **variadic operands** - they can accept a variable number of values for certain parameters. For example, `gpu.launch` is defined in ODS as:

```tablegen
let arguments = (ins
  Variadic<Index>:$asyncDependencies,     // 0 or more
  Index:$gridSizeX,                        // exactly 1
  Index:$gridSizeY,                        // exactly 1
  Index:$gridSizeZ,                        // exactly 1
  Index:$blockSizeX,                       // exactly 1
  Index:$blockSizeY,                       // exactly 1
  Index:$blockSizeZ,                       // exactly 1
  Optional<Index>:$clusterSizeX,           // 0 or 1
  Optional<Index>:$clusterSizeY,           // 0 or 1
  Optional<Index>:$clusterSizeZ,           // 0 or 1
  Optional<Index>:$dynamicSharedMemorySize // 0 or 1
);
```

When you call `add_operands([val1, val2, val3, ...])`, MLIR receives a flat list. It has no idea which values go with which parameter.

**The `operandSegmentSizes` attribute** tells MLIR: "the first 0 operands are asyncDependencies, the next 1 is gridSizeX, the next 1 is gridSizeY..." etc.

For our `gpu.launch` usage:
```
array<i32: 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0>
         │  │  │  │  │  │  │  │  │  │  └─ dynamicSharedMemorySize (0)
         │  │  │  │  │  │  │  │  │  └─ clusterSizeZ (0)
         │  │  │  │  │  │  │  │  └─ clusterSizeY (0)
         │  │  │  │  │  │  │  └─ clusterSizeX (0)
         │  │  │  │  │  │  └─ blockSizeZ (1)
         │  │  │  │  │  └─ blockSizeY (1)
         │  │  │  │  └─ blockSizeX (1)
         │  │  │  └─ gridSizeZ (1)
         │  │  └─ gridSizeY (1)
         │  └─ gridSizeX (1)
         └─ asyncDependencies (0)
```

### Why It's Bad

- Magic numbers that must match MLIR's ODS definition exactly
- If MLIR changes the operation definition, our code silently breaks
- Each variadic operation needs its own hardcoded handler
- No way to automatically derive this from Melior

### The Right Solution

Melior should provide:
```rust
op_builder.add_variadic_operands(&[
    ("asyncDependencies", &[]),
    ("gridSizeX", &[grid_x]),
    ("gridSizeY", &[grid_y]),
    // ...
])
```

Or expose ODS metadata so we can generate this automatically.

---

## Operations Still Needing `operandSegmentSizes`

- `gpu.launch` - 11 segments
- `cf.cond_br` - 3 segments (condition, true_args, false_args)
- `scf.while` - likely needs it
- `scf.parallel` - likely needs it
- Any operation with `Variadic` or `Optional` operands in ODS

---

## Summary

| Workaround | Status | Notes |
|------------|--------|-------|
| `is_void_operation()` | **FIXED** | Now uses `context.operation_supports_type_inference()` |
| `operandSegmentSizes` | Remaining | Still needs hardcoded handling per operation |

The type inference issue was solved by adding `operation_supports_type_inference()` to our Melior fork, which wraps MLIR's C API for querying operation interfaces. The `operandSegmentSizes` issue still requires a solution - either exposing ODS operand segment metadata through Melior, or requiring users to explicitly specify named operand groups.
