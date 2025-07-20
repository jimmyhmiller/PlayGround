//===- TensorOpsAPI.cpp - C API for TensorOps dialect -------------------===//
//
// This file implements the C API for the TensorOps dialect.
//
//===----------------------------------------------------------------------===//

#include "TensorOpsAPI.h"
#include "TensorOps/TensorOpsDialect.h"
#include "TensorOps/TensorOpsOps.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"

using namespace mlir;
using namespace mlir::tensor_ops;

//===----------------------------------------------------------------------===//
// Dialect registration
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TensorOps, tensor_ops, TensorOpsDialect)

//===----------------------------------------------------------------------===//
// Operation constructors
//===----------------------------------------------------------------------===//

MlirOperation mlirTensorOpsCreateConstantOp(MlirContext ctx,
                                             MlirAttribute value,
                                             MlirType resultType,
                                             MlirLocation loc) {
  MLIRContext *context = unwrap(ctx);
  OpBuilder builder(context);
  
  auto elementsAttr = unwrap(value).dyn_cast<ElementsAttr>();
  if (!elementsAttr) {
    return wrap(static_cast<Operation *>(nullptr));
  }
  
  auto op = builder.create<ConstantOp>(
      unwrap(loc),
      unwrap(resultType),
      elementsAttr);
  
  return wrap(op.getOperation());
}

MlirOperation mlirTensorOpsCreateAddOp(MlirContext ctx,
                                       MlirValue lhs,
                                       MlirValue rhs,
                                       MlirType resultType,
                                       MlirLocation loc) {
  MLIRContext *context = unwrap(ctx);
  OpBuilder builder(context);
  
  auto op = builder.create<AddOp>(
      unwrap(loc),
      unwrap(resultType),
      unwrap(lhs),
      unwrap(rhs));
  
  return wrap(op.getOperation());
}

MlirOperation mlirTensorOpsCreateMulOp(MlirContext ctx,
                                       MlirValue lhs,
                                       MlirValue rhs,
                                       MlirType resultType,
                                       MlirLocation loc) {
  MLIRContext *context = unwrap(ctx);
  OpBuilder builder(context);
  
  auto op = builder.create<MulOp>(
      unwrap(loc),
      unwrap(resultType),
      unwrap(lhs),
      unwrap(rhs));
  
  return wrap(op.getOperation());
}

MlirOperation mlirTensorOpsCreateReshapeOp(MlirContext ctx,
                                           MlirValue input,
                                           MlirAttribute shape,
                                           MlirType resultType,
                                           MlirLocation loc) {
  MLIRContext *context = unwrap(ctx);
  OpBuilder builder(context);
  
  auto shapeAttr = unwrap(shape).dyn_cast<ArrayAttr>();
  if (!shapeAttr) {
    return wrap(static_cast<Operation *>(nullptr));
  }
  
  auto op = builder.create<ReshapeOp>(
      unwrap(loc),
      unwrap(resultType),
      unwrap(input),
      shapeAttr);
  
  return wrap(op.getOperation());
}