//===- TensorOpsOps.cpp - TensorOps operation implementations -----------===//
//
// This file implements the operations in the TensorOps dialect.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsOps.h"
#include "TensorOps/TensorOpsDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::tensor_ops;

#define GET_OP_CLASSES
#include "TensorOps/TensorOpsOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  auto valueAttr = attributes.get("value").dyn_cast_or_null<ElementsAttr>();
  if (!valueAttr)
    return failure();
    
  inferredReturnTypes.push_back(valueAttr.getType());
  return success();
}

LogicalResult ConstantOp::verify() {
  auto type = getType();
  auto attr = getValue();
  
  if (type != attr.getType()) {
    return emitOpError("type mismatch between result type ")
           << type << " and attribute type " << attr.getType();
  }
  
  return success();
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue();
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  if (operands.size() != 2)
    return failure();
    
  auto lhsType = operands[0].getType();
  auto rhsType = operands[1].getType();
  
  if (lhsType != rhsType)
    return failure();
    
  inferredReturnTypes.push_back(lhsType);
  return success();
}

LogicalResult AddOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto resultType = getResult().getType();
  
  if (lhsType != rhsType) {
    return emitOpError("operand types must match, got ")
           << lhsType << " and " << rhsType;
  }
  
  if (lhsType != resultType) {
    return emitOpError("result type must match operand types, got ")
           << resultType << " but operands are " << lhsType;
  }
  
  return success();
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  // Fold if both operands are constants
  if (adaptor.getLhs() && adaptor.getRhs()) {
    auto lhsAttr = adaptor.getLhs().dyn_cast<ElementsAttr>();
    auto rhsAttr = adaptor.getRhs().dyn_cast<ElementsAttr>();
    
    if (lhsAttr && rhsAttr) {
      // TODO: Implement constant folding for element-wise addition
      // This would require iterating through elements and performing addition
    }
  }
  
  // Identity: x + 0 = x
  if (adaptor.getRhs()) {
    if (auto rhsAttr = adaptor.getRhs().dyn_cast<ElementsAttr>()) {
      if (rhsAttr.isSplat() && rhsAttr.getSplatValue<APFloat>().isZero()) {
        return getLhs();
      }
    }
  }
  
  return {};
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  if (operands.size() != 2)
    return failure();
    
  auto lhsType = operands[0].getType();
  auto rhsType = operands[1].getType();
  
  if (lhsType != rhsType)
    return failure();
    
  inferredReturnTypes.push_back(lhsType);
  return success();
}

LogicalResult MulOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();
  auto resultType = getResult().getType();
  
  if (lhsType != rhsType) {
    return emitOpError("operand types must match, got ")
           << lhsType << " and " << rhsType;
  }
  
  if (lhsType != resultType) {
    return emitOpError("result type must match operand types, got ")
           << resultType << " but operands are " << lhsType;
  }
  
  return success();
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  // Identity: x * 1 = x
  if (adaptor.getRhs()) {
    if (auto rhsAttr = adaptor.getRhs().dyn_cast<ElementsAttr>()) {
      if (rhsAttr.isSplat() && rhsAttr.getSplatValue<APFloat>().isExactlyValue(1.0)) {
        return getLhs();
      }
    }
  }
  
  // Zero: x * 0 = 0
  if (adaptor.getRhs()) {
    if (auto rhsAttr = adaptor.getRhs().dyn_cast<ElementsAttr>()) {
      if (rhsAttr.isSplat() && rhsAttr.getSplatValue<APFloat>().isZero()) {
        return adaptor.getRhs();
      }
    }
  }
  
  return {};
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

LogicalResult ReshapeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  if (operands.size() != 1)
    return failure();
    
  auto inputType = operands[0].getType().dyn_cast<RankedTensorType>();
  if (!inputType)
    return failure();
    
  auto shapeAttr = attributes.get("shape").dyn_cast_or_null<ArrayAttr>();
  if (!shapeAttr)
    return failure();
    
  SmallVector<int64_t> newShape;
  for (auto attr : shapeAttr) {
    auto intAttr = attr.dyn_cast<IntegerAttr>();
    if (!intAttr)
      return failure();
    newShape.push_back(intAttr.getInt());
  }
  
  auto resultType = RankedTensorType::get(newShape, inputType.getElementType());
  inferredReturnTypes.push_back(resultType);
  return success();
}

LogicalResult ReshapeOp::verify() {
  auto inputType = getInput().getType().dyn_cast<RankedTensorType>();
  auto resultType = getResult().getType().dyn_cast<RankedTensorType>();
  
  if (!inputType || !resultType) {
    return emitOpError("requires ranked tensor types");
  }
  
  if (inputType.getElementType() != resultType.getElementType()) {
    return emitOpError("element types must match");
  }
  
  // Check that total number of elements is preserved
  int64_t inputSize = 1;
  for (auto dim : inputType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      // Cannot verify for dynamic shapes
      return success();
    }
    inputSize *= dim;
  }
  
  int64_t resultSize = 1;
  for (auto dim : resultType.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      // Cannot verify for dynamic shapes
      return success();
    }
    resultSize *= dim;
  }
  
  if (inputSize != resultSize) {
    return emitOpError("total number of elements must be preserved, got ")
           << inputSize << " input elements and " << resultSize << " result elements";
  }
  
  return success();
}