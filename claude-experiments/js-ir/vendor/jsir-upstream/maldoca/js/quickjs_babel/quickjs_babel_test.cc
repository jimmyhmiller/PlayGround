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

#include "maldoca/js/quickjs_babel/quickjs_babel.h"

#include <memory>

#include "gtest/gtest.h"
#include "maldoca/js/babel/babel_test.h"

namespace maldoca {
namespace {

INSTANTIATE_TEST_SUITE_P(
    QuickJsBabelTest, BabelTest,
    testing::Values(BabelTestParams{
        .babel_factory = [] { return std::make_unique<QuickJsBabel>(); },
        .stack_overflow_error_pbtxt = R"pb(
          errors { name: "{error}" message: "" }
        )pb",
    }));

}  // namespace
}  // namespace maldoca
