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

#include "maldoca/js/ir/transforms/peel_parentheses/pass.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

void PeelParentheses(mlir::Operation *root) {
  mlir::MLIRContext *context = root->getContext();
  mlir::OpBuilder builder{context};

  root->walk([&](JsirParenthesizedExpressionOp op) {
    op.replaceAllUsesWith(op.getExpression());
    op.erase();
  });

  root->walk([&](JsirParenthesizedExpressionRefOp op) {
    op.replaceAllUsesWith(op.getExpression());
    op.erase();
  });
}

}  // namespace maldoca
