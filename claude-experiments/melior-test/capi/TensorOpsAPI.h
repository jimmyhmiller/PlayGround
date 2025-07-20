//===- TensorOpsAPI.h - C API for TensorOps dialect ---------------------===//
//
// This file defines the C API for the TensorOps dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOROPS_CAPI_TENSOROPSAPI_H
#define TENSOROPS_CAPI_TENSOROPSAPI_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(TensorOps, tensor_ops);

//===----------------------------------------------------------------------===//
// Operation constructors
//===----------------------------------------------------------------------===//

/// Creates a tensor_ops.constant operation.
MLIR_CAPI_EXPORTED MlirOperation mlirTensorOpsCreateConstantOp(
    MlirContext ctx,
    MlirAttribute value,
    MlirType resultType,
    MlirLocation loc);

/// Creates a tensor_ops.add operation.
MLIR_CAPI_EXPORTED MlirOperation mlirTensorOpsCreateAddOp(
    MlirContext ctx,
    MlirValue lhs,
    MlirValue rhs,
    MlirType resultType,
    MlirLocation loc);

/// Creates a tensor_ops.mul operation.
MLIR_CAPI_EXPORTED MlirOperation mlirTensorOpsCreateMulOp(
    MlirContext ctx,
    MlirValue lhs,
    MlirValue rhs,
    MlirType resultType,
    MlirLocation loc);

/// Creates a tensor_ops.reshape operation.
MLIR_CAPI_EXPORTED MlirOperation mlirTensorOpsCreateReshapeOp(
    MlirContext ctx,
    MlirValue input,
    MlirAttribute shape,
    MlirType resultType,
    MlirLocation loc);

#ifdef __cplusplus
}
#endif

#endif // TENSOROPS_CAPI_TENSOROPSAPI_H