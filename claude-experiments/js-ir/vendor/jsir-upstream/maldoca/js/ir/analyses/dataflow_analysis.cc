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

#include "maldoca/js/ir/analyses/dataflow_analysis.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataFlowFramework.h"

namespace maldoca {

mlir::ChangeResult JsirExecutable::Join(const JsirExecutable &other) {
  bool prev_executable = executable_;
  executable_ |= other.executable_;
  return executable_ == prev_executable ? mlir::ChangeResult::NoChange
                                        : mlir::ChangeResult::Change;
}

void JsirExecutable::print(llvm::raw_ostream &os) const {
  if (executable_) {
    os << "<executable>";
  } else {
    os << "<not executable>";
  }
}

}  // namespace maldoca
