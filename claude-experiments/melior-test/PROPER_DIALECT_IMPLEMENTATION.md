# Proper MLIR Dialect Implementation Requirements

This document outlines what would be needed to implement tensor_ops as a proper MLIR dialect, based on the test failures and current limitations.

## Current State

The tensor_ops "dialect" is currently just a namespace for operations. It lacks:
- Proper registration with MLIR
- Operation verification
- Type system integration
- In-place lowering
- Round-trip capability

## Test Results Summary

Running `cargo test --test tensor_dialect_tests`:

### ✅ Passing Tests
1. `test_dialect_registration` - Passes but only because we allow unregistered dialects
2. `test_lowering_preserves_semantics` - Passes but should fail (lowering creates new module)
3. `test_conversion_patterns` - Passes (placeholder test)
4. `test_dialect_types` - Passes (placeholder test)

### ❌ Failing Tests
1. `test_operation_verification` - FAILED: Operations accept invalid inputs (no verification)
2. `test_dialect_with_standard_passes` - FAILED: Module verification fails
3. `test_round_trip` - SEGFAULT: Printing module with tensor_ops crashes

## Requirements for Proper Implementation

### 1. C++ Integration (Not possible with pure Rust/melior)
```cpp
// TensorOpsDialect.h
class TensorOpsDialect : public mlir::Dialect {
public:
  explicit TensorOpsDialect(mlir::MLIRContext *context);
  static llvm::StringRef getDialectNamespace() { return "tensor_ops"; }
  
  void initialize();
};
```

### 2. TableGen Definitions (.td files)
```tablegen
// TensorOpsDialect.td
def TensorOps_Dialect : Dialect {
  let name = "tensor_ops";
  let cppNamespace = "::mlir::tensor_ops";
  let description = "High-level tensor operations";
}

// Operation definitions
def TensorOps_AddOp : TensorOps_Op<"add"> {
  let summary = "Element-wise tensor addition";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  
  let verifier = [{
    // Verify that input types match and are compatible
  }];
}
```

### 3. Operation Classes with Verification
```cpp
class AddOp : public Op<AddOp, OpTrait::NOperands<2>::Impl,
                       OpTrait::OneResult> {
public:
  LogicalResult verify() {
    // Check operand types match
    // Check result type is correct
    return success();
  }
};
```

### 4. Proper Registration
```cpp
void registerTensorOpsDialect(DialectRegistry &registry) {
  registry.insert<TensorOpsDialect>();
}

// In the dialect's initialize method:
void TensorOpsDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "TensorOps.cpp.inc"
  >();
}
```

### 5. Conversion Patterns for Lowering
```cpp
class TensorAddToArithAddPattern : public OpRewritePattern<tensor_ops::AddOp> {
  LogicalResult matchAndRewrite(tensor_ops::AddOp op,
                                PatternRewriter &rewriter) const override {
    // Replace tensor_ops.add with arith.addf/addi
    rewriter.replaceOpWithNewOp<arith::AddFOp>(
      op, op.getType(), op.getLhs(), op.getRhs());
    return success();
  }
};
```

### 6. Conversion Pass
```cpp
class TensorOpsToStandardPass
    : public PassWrapper<TensorOpsToStandardPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<TensorOpsDialect>();
    
    RewritePatternSet patterns(&getContext());
    patterns.add<TensorAddToArithAddPattern>(&getContext());
    
    if (failed(applyPartialConversion(getOperation(), target, patterns)))
      signalPassFailure();
  }
};
```

## Limitations of melior

The melior Rust bindings cannot create a proper dialect because:

1. **No TableGen support** - Cannot generate operation definitions
2. **No C++ interop** - Cannot inherit from mlir::Dialect
3. **No registration hooks** - Cannot properly register with MLIR's type system
4. **No pattern rewriting** - Cannot create proper lowering patterns
5. **Limited verification** - Cannot add custom verification logic

## Workarounds in Current Implementation

1. Using `mlirContextSetAllowUnregisteredDialects(true)`
2. Creating a new module during "lowering" instead of transforming
3. Manual operation building without verification
4. No type inference or checking

## Conclusion

A proper tensor_ops dialect requires C++ integration with MLIR's infrastructure. The current Rust/melior implementation can only create a superficial dialect that:
- Uses the namespace syntax but isn't truly registered
- Cannot verify operations
- Cannot be lowered using MLIR's pattern rewriting
- Causes crashes when interacting with MLIR's core systems

For production use, either:
1. Use existing MLIR dialects (linalg, tensor, etc.)
2. Implement the dialect in C++ with proper TableGen
3. Accept the limitations of unregistered operations