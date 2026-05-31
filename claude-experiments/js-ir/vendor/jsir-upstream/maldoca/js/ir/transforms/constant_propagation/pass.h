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

#ifndef MALDOCA_JS_IR_TRANSFORMS_CONSTANT_PROPAGATION_PASS_H_
#define MALDOCA_JS_IR_TRANSFORMS_CONSTANT_PROPAGATION_PASS_H_

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "maldoca/js/ir/analyses/constant_propagation/analysis.h"

namespace maldoca {

mlir::LogicalResult PerformConstantPropagation(mlir::Operation *op,
                                               const BabelScopes &scopes);

mlir::LogicalResult PerformConstantPropagation(
    mlir::Operation *op, JsirConstantPropagationAnalysis &analysis);

struct JsirConstantPropagationPass
    : public mlir::PassWrapper<JsirConstantPropagationPass,
                               mlir::OperationPass<>> {
  using Base =
      mlir::PassWrapper<JsirConstantPropagationPass, mlir::OperationPass<>>;

  explicit JsirConstantPropagationPass(const BabelScopes *scopes)
      : Base(), scopes_(*scopes) {}

  void runOnOperation() override {
    if (mlir::failed(PerformConstantPropagation(getOperation(), scopes_))) {
      // Failure means that some invariants in the IR have been broken, and the
      // IR might be in an invalid state.
      signalPassFailure();
    }
  }

  const BabelScopes &scopes_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_TRANSFORMS_CONSTANT_PROPAGATION_PASS_H_
