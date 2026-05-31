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

#ifndef MALDOCA_JS_AST_UTIL_H_
#define MALDOCA_JS_AST_UTIL_H_

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/optional.h"
#include "nlohmann/json.hpp"
#include "maldoca/js/ast/ast.generated.h"
#include "maldoca/js/babel/babel.h"
#include "maldoca/js/babel/babel.pb.h"

namespace maldoca {

// Converts a source code string into an AST string.
// Parses the source using Babel and returns a string representing the AST.
// Returns an error if the parsing was not successful.
absl::StatusOr<BabelAstString> GetAstStringFromSource(
    Babel& babel, std::string_view source, const BabelParseRequest& request,
    absl::Duration timeout);

template <typename JsonT = nlohmann::json>
absl::StatusOr<JsonT> GetAstJsonFromAstString(
    const BabelAstString& babel_ast_string,
    std::optional<const int> recursion_depth_limit_op) {
  typename JsonT::parser_callback_t callback = nullptr;

  // To enforce a limit on the depth of recursion we provide a callback to the
  // json parsing function so we can see the recursion depth of elements.
  // Parsing continues to the end of the source, there is no way to interrupt
  // it. We keep track of the max depth so the returned error gives insight into
  // actual depths seen at runtime.
  int max_depth = 0;
  if (recursion_depth_limit_op.has_value()) {
    callback = [&max_depth](int depth, typename JsonT::parse_event_t event,
                            JsonT& parsed) {
      max_depth = std::max(max_depth, depth);
      // Always return true. Returning false will omit pieces of the json.
      // Since we're only doing this based on depth, it may result in invalid
      // code.
      return true;
    };
  }

  auto json_ast = JsonT::parse(babel_ast_string.value(),
                               /*callback=*/callback,
                               /*allow_exceptions=*/false,
                               /*ignore_comments=*/false);
  if (json_ast.is_discarded()) {
    return absl::InternalError("Failed to parse AST, invalid JSON.");
  }

  if (recursion_depth_limit_op.has_value() &&
      max_depth > *recursion_depth_limit_op) {
    return absl::InternalError(absl::StrFormat(
        "AST exceeded recursion depth. Max depth: %d, limit: %d", max_depth,
        *recursion_depth_limit_op));
  }

  return json_ast;
}

// Attempts to convert the AST string in JSON format into a JsFile AST.
// Allows for setting custom options (recursion depth limit). If a recursion
// depth limit is provided then we will return an error if the JSON
// representation of the source is nested more deeply than the limit. Returns an
// error if the AST is invalid.
absl::StatusOr<std::unique_ptr<JsFile>> GetFileAstFromAstString(
    const BabelAstString& babel_ast_string,
    std::optional<const int> recursion_depth_limit_op);

// Attempts to parse a source code string into a JsFile AST.
// Allows for setting custom options (timeout, request settings, recursion
// depth limit). If a recursion depth limit is provided then we will return an
// error if the JSON representation of the source is nested more deeply than the
// limit. Returns an error if the parsing was not successful.
absl::StatusOr<std::tuple<std::unique_ptr<JsFile>, BabelScopes>>
GetFileAstFromSource(Babel& babel, std::string_view source,
                     const BabelParseRequest& request, absl::Duration timeout,
                     absl::optional<const int> recursion_depth_limit_op);

// Attempts to parse a source code string into a JsFile AST using default
// babel options. This overload just lets callers not depend on the
// BabelParseRequest. Returns an error if parsing was not successful.
absl::StatusOr<std::tuple<std::unique_ptr<JsFile>, BabelScopes>>
GetFileAstFromSource(
    Babel& babel, std::string_view source, absl::Duration timeout,
    absl::optional<const int> recursion_depth_limit_op = absl::nullopt);

// Unescape a string literal.
// Example: "\x73\x63\x72\x69\x70\x74" => "script"
void CUnescapeStringLiteral(JsStringLiteral& string_literal);

// Unescape all string literals in an AST.
void CUnescapeStringLiteralsInFile(JsFile& file);

BabelAstString GetAstStringFromFileAst(const JsFile& file);

// Pretty-prints the provided AST string.
// Uses babel to re-generate new source from the AST. The regenerated source
// will be pretty-printed. Returns an error status if parsing or generating
// fails.
absl::StatusOr<std::string> PrettyPrintSourceFromAstString(
    Babel& babel, const BabelAstString& babel_ast_string,
    absl::Duration timeout);

// Similar to `PrettyPrintSourceFromAstString` but takes JsFile and dumps it
// into an AST string before pretty-printing.
absl::StatusOr<std::string> PrettyPrintSourceFromFileAst(
    Babel& babel, const JsFile& file, absl::Duration timeout);

// Similar to `PrettyPrintSourceFromAstString` but takes raw source code and
// parses it into an AST string before pretty-printing.
absl::StatusOr<std::string> PrettyPrintSourceFromSourceString(
    Babel& babel, std::string_view source_string,
    const BabelParseRequest& request, absl::Duration timeout);

absl::StatusOr<std::string> PrettyPrintSourceFromSourceString(
    Babel& babel, std::string_view source_string, absl::Duration timeout);

}  // namespace maldoca

#endif  // MALDOCA_JS_AST_UTIL_H_
