//===- TensorOpsOps.h - TensorOps operation declarations ----------------===//
//
// This file declares the operations in the TensorOps dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOROPS_TENSOROPSOPS_H
#define TENSOROPS_TENSOROPSOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "TensorOps/TensorOpsOps.h.inc"

#endif // TENSOROPS_TENSOROPSOPS_H