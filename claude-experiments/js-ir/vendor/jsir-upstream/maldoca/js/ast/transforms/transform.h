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

#ifndef MALDOCA_JS_AST_TRANSFORMS_TRANSFORM_H_
#define MALDOCA_JS_AST_TRANSFORMS_TRANSFORM_H_

#include <optional>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/driver/driver.pb.h"

namespace maldoca {

// Transforms the given AST, and optionally returns an analysis result.
//
// Parameters:
// - original_source:
//   The original source code that source ranges in the AST refer to.
// - scopes:
//   The Babel scopes for the AST.
// - config:
//   The transform config.
// - ast:
//   The AST to transform. Note that AST does not need to match original_source.
// - optional_analysis_result:
//   If the transform performs some analysis, then this will be set to the
//   analysis result. Otherwise, it will be unchanged.
absl::Status TransformJsAst(
    std::optional<absl::string_view> original_source, const BabelScopes &scopes,
    const JsAstTransformConfig &config, JsFile &ast,
    std::optional<JsAstAnalysisResult> &optional_analysis_result);

}  // namespace maldoca

#endif  // MALDOCA_JS_AST_TRANSFORMS_TRANSFORM_H_
