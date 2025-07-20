//===- TensorOpsDialect.cpp - TensorOps dialect implementation ----------===//
//
// This file implements the TensorOps dialect.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsDialect.h"
#include "TensorOps/TensorOpsOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::tensor_ops;

#include "TensorOps/TensorOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TensorOps dialect
//===----------------------------------------------------------------------===//

void TensorOpsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TensorOps/TensorOpsOps.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// Constant materialization
//===----------------------------------------------------------------------===//

Operation *TensorOpsDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value,
                                                  Type type,
                                                  Location loc) {
  if (auto elementsAttr = value.dyn_cast<ElementsAttr>()) {
    if (elementsAttr.getType() == type) {
      return builder.create<ConstantOp>(loc, type, elementsAttr);
    }
  }
  return nullptr;
}