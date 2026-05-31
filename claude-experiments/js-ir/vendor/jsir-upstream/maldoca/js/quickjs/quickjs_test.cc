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

#include "maldoca/js/quickjs/quickjs.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/cleanup/cleanup.h"
#include "absl/strings/string_view.h"
#include "quickjs/quickjs-libc.h"
#include "quickjs/quickjs.h"

namespace maldoca {
namespace {

using ::testing::StrEq;

TEST(QuickJsTest, QuickJsTest) {
  std::unique_ptr<JSRuntime, QjsRuntimeDeleter> runtime{JS_NewRuntime()};

  std::unique_ptr<JSContext, QjsContextDeleter> context{
      JS_NewContext(runtime.get())};

  js_std_init_handlers(runtime.get());
  absl::Cleanup free_handlers = [&] { js_std_free_handlers(runtime.get()); };

  QjsValue global{context.get(), JS_GetGlobalObject(context.get())};
  ASSERT_TRUE(JS_IsObject(global.get()));

  // This ensures that console.log(...) gets printed to stdout.
  {
    int argc = 0;
    char *argv[] = {nullptr};
    js_std_add_helpers(context.get(), argc, argv);
  }

  static constexpr absl::string_view kSource = R"js(
    function add(a, b) {
      // Without js_std_add_helpers(), this would throw a JS exception.
      console.log("add");

      return a + b;
    }

    // The result of the last expression is returned.
    add
  )js";

  QjsValue add{
      context.get(),
      JS_Eval(context.get(), kSource.data(), kSource.length(), "filename.js",
              JS_EVAL_TYPE_GLOBAL),
  };

  ASSERT_TRUE(JS_IsFunction(context.get(), add.get()));

  // Verify that `add` is in the global object.
  QjsValue global_add{
      context.get(),
      JS_GetPropertyStr(context.get(), global.get(), "add"),
  };
  ASSERT_TRUE(JS_IsFunction(context.get(), global_add.get()));

  // Verify that we can programmatically call the `add` function.
  {
    std::string a = "hello";
    std::string b = " world";
    std::vector<QjsValue> args = {
        QjsValue{context.get(),
                 JS_NewStringLen(context.get(), a.data(), a.length())},
        QjsValue{context.get(),
                 JS_NewStringLen(context.get(), b.data(), b.length())},
    };

    std::vector<JSValue> arg_refs;
    for (const QjsValue &arg : args) {
      arg_refs.push_back(arg.get());
    }

    QjsValue result{
        context.get(),
        JS_Call(context.get(), /*func_obj=*/add.get(),
                /*this_obj=*/global.get(),
                /*argc=*/args.size(), /*argv=*/arg_refs.data()),
    };
    ASSERT_TRUE(JS_IsString(result.get()));

    const char *result_cstr = JS_ToCString(context.get(), result.get());
    absl::Cleanup free_result = [&] {
      JS_FreeCString(context.get(), result_cstr);
    };

    ASSERT_NE(result_cstr, nullptr);
    EXPECT_THAT(result_cstr, StrEq("hello world"));
  }
}

}  // namespace
}  // namespace maldoca
