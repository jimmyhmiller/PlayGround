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

// +-----------------------------------+-----------------------------------+
// | Before                            | After                             |
// +-----------------------------------+-----------------------------------+
// | obj = {                           | obj = {                           |
// |   short_hand,                     |   "short_hand": short_hand,       |*
// |   property_identifier: 1,         |   "property_identifier": 1,       |*
// |   "property_string_literal": 2,   |   "property_string_literal": 2,   |
// |   1.0: 3,                         |   1.0: 3,                         |
// |   ["property_computed"]: 4,       |   ["property_computed"]: 4,       |
// |   method_identifier() {},         |   method_identifier() {},         |
// |   "property_string_literal"() {}, |   "property_string_literal"() {}, |
// |   1.0() {},                       |   1.0() {},                       |
// |   ["property_computed"]() {},     |   ["property_computed"]() {},     |
// |   ...spread_element               |   ...spread_element               |
// | };                                | };                                |
// +-----------------------------------+-----------------------------------+

// +-------------------------------+-----------------------------------------+
// | Before                        | After                                   |
// +-------------------------------+-----------------------------------------+
// | {                             | {                                       |
// |   lvalue_shorthand,           |   "lvalue_shorthand": lvalue_shorthand, |*
// |   identifier: lvalue_1,       |   "identifier": lvalue_1,               |*
// |   'string_literal': lvalue_2, |   'string_literal': lvalue_2,           |
// |   1.0: lvalue_3,              |   1.0: lvalue_3,                        |
// |   ['computed']: lvalue_4,     |   ['computed']: lvalue_4                |
// |   ...lvalue_rest              |   ...lvalue_rest                        |
// | } = obj;                      | } = obj;                                |
// +-------------------------------+-----------------------------------------+

#ifndef MALDOCA_JS_IR_TRANSFORMS_NORMALIZE_OBJECT_PROPERTIES_PASS_H_
#define MALDOCA_JS_IR_TRANSFORMS_NORMALIZE_OBJECT_PROPERTIES_PASS_H_

#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace maldoca {

void NormalizeObjectProperties(mlir::Operation *root);

struct NormalizeObjectPropertiesPass
    : public mlir::PassWrapper<NormalizeObjectPropertiesPass,
                               mlir::OperationPass<>> {
  void runOnOperation() override {
    mlir::Operation *root = getOperation();
    NormalizeObjectProperties(root);
  }
};

}  // namespace maldoca

#endif  // MALDOCA_JS_IR_TRANSFORMS_NORMALIZE_OBJECT_PROPERTIES_PASS_H_
