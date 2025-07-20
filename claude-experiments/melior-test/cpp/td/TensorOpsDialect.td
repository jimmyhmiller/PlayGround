//===- TensorOpsDialect.td - TensorOps dialect definition ----------------===//
//
// This file defines the TensorOps dialect for high-level tensor operations.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOROPS_DIALECT
#define TENSOROPS_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TensorOps Dialect Definition
//===----------------------------------------------------------------------===//

def TensorOps_Dialect : Dialect {
  let name = "tensor_ops";
  let summary = "High-level tensor operations dialect";
  let description = [{
    The TensorOps dialect provides high-level operations for tensor manipulation
    including element-wise arithmetic, reshaping, and other common tensor
    operations. These operations are designed to be lowered to standard MLIR
    dialects like arith, tensor, and linalg.
  }];
  
  let cppNamespace = "::mlir::tensor_ops";
  
  // Use default type and attribute parsers/printers
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  
  // Enable constant materialization for folding
  let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base TensorOps Operation Definition
//===----------------------------------------------------------------------===//

class TensorOps_Op<string mnemonic, list<Trait> traits = []> :
    Op<TensorOps_Dialect, mnemonic, traits>;

#endif // TENSOROPS_DIALECT