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

#ifndef MALDOCA_JS_QUICKJS_BABEL_QUICKJS_BABEL_H_
#define MALDOCA_JS_QUICKJS_BABEL_QUICKJS_BABEL_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"
#include "maldoca/js/quickjs/quickjs.h"
#include "quickjs/quickjs.h"

namespace maldoca {

class QuickJsBabel : public Babel {
 public:
  QuickJsBabel();
  ~QuickJsBabel() override;

  absl::StatusOr<BabelParseResult> Parse(absl::string_view source_code,
                                         const BabelParseRequest& request,
                                         absl::Duration timeout) override;

  absl::StatusOr<BabelGenerateResult> Generate(const BabelAstString& ast_string,
                                               const BabelGenerateOptions& opts,
                                               absl::Duration timeout) final;

 private:
  std::unique_ptr<JSRuntime, QjsRuntimeDeleter> qjs_runtime_;
  std::unique_ptr<JSContext, QjsContextDeleter> qjs_context_;
  QjsValue parse_;
  QjsValue generate_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_QUICKJS_BABEL_QUICKJS_BABEL_H_
