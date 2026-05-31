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

#include "maldoca/js/ir/transforms/move_named_functions/pass.h"

#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "maldoca/js/ir/ir.h"

namespace maldoca {

void MoveNamedFunctions(mlir::Operation *root) {
  for (mlir::Region &region : root->getRegions()) {
    // If there is no block in the region, do nothing.
    if (region.empty()) {
      continue;
    }

    // Stores all top-level named function ops in the blocks of the region.
    // Note that ops are owned by blocks, and this vector only keeps references
    // to the ops and doesn't hold ownership.
    std::vector<JsirFunctionDeclarationOp> named_function_ops;
    for (mlir::Block &block : region) {
      for (mlir::Operation &op : block) {
        // If this op has regions, recursively move named functions to the top
        // of the enclosing regions. This recursive call only modifies the
        // internals of the op and doesn't move the op itself.
        MoveNamedFunctions(&op);

        if (auto function_declaration_op =
                llvm::dyn_cast<JsirFunctionDeclarationOp>(&op)) {
          named_function_ops.push_back(function_declaration_op);
        }
      }
    }

    // Now we move the ops to the front of the region. This is internally
    // moving linked list nodes.
    mlir::Block *entry_block = &region.front();
    for (auto op : llvm::reverse(named_function_ops)) {
      op->moveBefore(entry_block, entry_block->begin());
    }
  }
}

}  // namespace maldoca
