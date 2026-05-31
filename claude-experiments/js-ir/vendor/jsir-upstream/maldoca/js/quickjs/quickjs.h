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

#ifndef MALDOCA_JS_QUICKJS_QUICKJS_H_
#define MALDOCA_JS_QUICKJS_QUICKJS_H_

#include <cstddef>
#include <optional>
#include <string>
#include "absl/cleanup/cleanup.h"
#include "quickjs/quickjs.h"

namespace maldoca {

// Usage:
//
// std::unique_ptr<JSRuntime, QjsRuntimeDeleter> qjs_runtime{JS_NewRuntime()};
struct QjsRuntimeDeleter {
  void operator()(JSRuntime *runtime) {
    if (runtime == nullptr) {
      return;
    }
    JS_FreeRuntime(runtime);
  }
};

// Usage:
//
// JSRuntime *qjs_runtime = ...;
// std::unique_ptr<JSContext, QjsContextDeleter> qjs_context{
//     JS_NewContext(qjs_runtime)};
struct QjsContextDeleter {
  void operator()(JSContext *context) {
    if (context == nullptr) {
      return;
    }
    JS_FreeContext(context);
  }
};

// `JSValue` is a ref-counted pointer, and the user is responsible for:
// - Calling `JS_FreeValue()` to decrease the ref-count;
// - Calling `JS_DupValue()` to increase the ref-count.
//
// This class wraps `JSValue` to automatically manage the ref-count.
//
// See unit test for usage examples.
class QjsValue {
 public:
  explicit QjsValue(JSContext *context, JSValue value)
      : context_(context), value_(value) {}

  ~QjsValue() {
    if (!JS_IsNull(value_)) {
      JS_FreeValue(context_, value_);
    }
  }

  // ---------------------------------------------------------------------------
  // Copy
  // ---------------------------------------------------------------------------

  QjsValue(const QjsValue &other)
      : context_(other.context_),
        value_(JS_DupValue(other.context_, other.value_)) {}

  QjsValue &operator=(const QjsValue &other) {
    reset(other.context_, JS_DupValue(other.context_, other.get()));
    return *this;
  }

  // ---------------------------------------------------------------------------
  // Move
  // ---------------------------------------------------------------------------

  QjsValue(QjsValue &&other)
      : context_(other.context_), value_(other.release()) {}

  QjsValue &operator=(QjsValue &&other) {
    reset(other.context_, other.release());
    return *this;
  }

  JSValue release() {
    JSValue value = value_;
    value_ = JS_NULL;
    return value;
  }

  JSValue get() const { return value_; }

  void reset(JSContext *context, JSValue value) {
    JSContext *current_context = context_;
    context_ = context;

    JSValue current_value = value_;
    value_ = value;

    if (!JS_IsNull(current_value)) {
      JS_FreeValue(current_context, current_value);
    }
  }

  // ---------------------------------------------------------------------------
  // ToString
  // ---------------------------------------------------------------------------

  std::optional<std::string> ToString() const {
    size_t len;
    const char *c_str = JS_ToCStringLen(context_, &len, value_);

    absl::Cleanup cleanup = [&] { JS_FreeCString(context_, c_str); };

    if (c_str == nullptr) {
      return std::nullopt;
    }

    return std::string{c_str, len};
  }

 private:
  JSContext *context_;
  JSValue value_;
};

}  // namespace maldoca

#endif  // MALDOCA_JS_QUICKJS_QUICKJS_H_
