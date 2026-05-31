// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MALDOCA_JS_IR_CAST_H_
#define MALDOCA_JS_IR_CAST_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace maldoca {

template <typename OpT>
absl::StatusOr<OpT> Cast(mlir::Operation *operation);

template <typename OpT>
absl::StatusOr<OpT> Cast(mlir::Value value);

template <typename OpT>
absl::StatusOr<OpT> Cast(mlir::Attribute attr);

template <typename OpT>
absl::StatusOr<OpT> Cast(mlir::Operation *operation) {
  auto op = llvm::dyn_cast<OpT>(operation);
  if (op == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", typeid(OpT).name(), ", got ",
                     operation->getName().getStringRef().str(), "."));
  }
  return op;
}

template <typename OpT>
absl::StatusOr<OpT> Cast(mlir::Value value) {
  if (value.getDefiningOp() == nullptr) {
    return absl::InvalidArgumentError("Value has no defining op.");
  }
  return Cast<OpT>(value.getDefiningOp());
}

template <typename OpT>
absl::StatusOr<OpT> Cast(mlir::Attribute attr) {
  auto element_attr = llvm::dyn_cast<OpT>(attr);
  if (element_attr == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid attribute type. Expected ", typeid(OpT).name(), "."));
  }
  return element_attr;
}

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_CAST_H_
