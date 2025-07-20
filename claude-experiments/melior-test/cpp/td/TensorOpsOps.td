//===- TensorOpsOps.td - TensorOps operation definitions -----------------===//
//
// This file defines the operations in the TensorOps dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOROPS_OPS
#define TENSOROPS_OPS

include "TensorOpsDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TensorOps Operations
//===----------------------------------------------------------------------===//

def TensorOps_ConstantOp : TensorOps_Op<"constant", [
    Pure,
    ConstantLike,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Tensor constant operation";
  let description = [{
    Creates a constant tensor with the given value.
    
    Example:
    ```mlir
    %0 = tensor_ops.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
    ```
  }];
  
  let arguments = (ins ElementsAttr:$value);
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = "attr-dict $value `:` type($result)";
  
  let hasFolder = 1;
  let hasVerifier = 1;
}

def TensorOps_AddOp : TensorOps_Op<"add", [
    Pure,
    SameOperandsAndResultType,
    Commutative,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise tensor addition";
  let description = [{
    Performs element-wise addition of two tensors of the same shape.
    
    Example:
    ```mlir
    %result = tensor_ops.add %lhs, %rhs : tensor<4x4xf32>
    ```
  }];
  
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = "operands attr-dict `:` type($result)";
  
  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TensorOps_MulOp : TensorOps_Op<"mul", [
    Pure,
    SameOperandsAndResultType,
    Commutative,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Element-wise tensor multiplication";
  let description = [{
    Performs element-wise multiplication of two tensors of the same shape.
    
    Example:
    ```mlir
    %result = tensor_ops.mul %lhs, %rhs : tensor<4x4xf32>
    ```
  }];
  
  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = "operands attr-dict `:` type($result)";
  
  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def TensorOps_ReshapeOp : TensorOps_Op<"reshape", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Tensor reshape operation";
  let description = [{
    Reshapes a tensor to a new shape while preserving the total number of elements.
    
    Example:
    ```mlir
    %result = tensor_ops.reshape %input [2, 8] : tensor<4x4xf32> -> tensor<2x8xf32>
    ```
  }];
  
  let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$shape);
  let results = (outs AnyRankedTensor:$result);
  
  let assemblyFormat = "$input $shape attr-dict `:` type($input) `->` type($result)";
  
  let hasVerifier = 1;
}

#endif // TENSOROPS_OPS