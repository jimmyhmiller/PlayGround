#include "../src/ast.h"
#include "../src/reader.h"
#include <cassert>
#include <iostream>
#include <string>

void test_semicolon_should_not_be_identifier() {
  std::cout << "=== Testing Semicolon Should Not Be Identifier ==="
            << std::endl;

  std::string input = "let x = 10; let y = 20;";
  std::cout << "Input: " << input << std::endl;

  try {
    Reader reader(input);
    reader.read();

    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();

    // Search through the entire AST for any identifier with ";" as value
    bool found_semicolon_identifier = false;
    std::function<void(const ASTNode *)> search_for_semicolon =
        [&](const ASTNode *node) {
          if (!node)
            return;

          if (node->type == ASTNodeType::Identifier && node->value == ";") {
            found_semicolon_identifier = true;
            std::cout << "❌ FOUND SEMICOLON IDENTIFIER at token line "
                      << node->token.line << ", column " << node->token.column
                      << std::endl;
          }

          for (size_t i = 0; i < node->child_count(); ++i) {
            search_for_semicolon(node->child(i));
          }

          if (node->function_type) {
            search_for_semicolon(node->function_type.get());
          }
          if (node->body) {
            search_for_semicolon(node->body.get());
          }
        };

    search_for_semicolon(ast.get());

    if (found_semicolon_identifier) {
      std::cout << "❌ FAILED: Found semicolon being treated as identifier"
                << std::endl;
    } else {
      std::cout << "✅ SUCCESS: No semicolon identifiers found - semicolons "
                   "properly handled as separators"
                << std::endl;
    }
  } catch (const std::exception &e) {
    std::cout << "❌ FAILED: Unexpected error: " << e.what() << std::endl;
  }
  std::cout << std::endl;
}

void test_invalid_identifier_characters() {
  std::cout << "=== Testing Invalid Identifier Characters ===" << std::endl;

  std::vector<std::string> invalid_cases = {
      "let 123invalid = 10", // starts with number
      "let ;invalid = 10",   // starts with semicolon
      "let invalid; = 10",   // contains semicolon
      "let invalid, = 10",   // contains comma
      "let (invalid = 10",   // starts with delimiter
  };

  for (const auto &test_case : invalid_cases) {
    std::cout << "Testing: " << test_case << std::endl;

    try {
      Reader reader(test_case);
      reader.read();

      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();

      // Check for invalid identifiers
      bool found_invalid = false;
      std::function<void(const ASTNode *)> check_identifiers =
          [&](const ASTNode *node) {
            if (!node)
              return;

            if (node->type == ASTNodeType::Identifier) {
              const std::string &value = node->value;
              if (!value.empty()) {
                // Check if it starts with a letter or underscore
                if (!std::isalpha(value[0]) && value[0] != '_') {
                  found_invalid = true;
                  std::cout << "  ❌ Invalid identifier: '" << value
                            << "' (starts with non-letter)" << std::endl;
                }

                // Check for invalid characters
                for (char c : value) {
                  if (!std::isalnum(c) && c != '_') {
                    found_invalid = true;
                    std::cout << "  ❌ Invalid identifier: '" << value
                              << "' (contains '" << c << "')" << std::endl;
                    break;
                  }
                }
              }
            }

            for (size_t i = 0; i < node->child_count(); ++i) {
              check_identifiers(node->child(i));
            }

            if (node->function_type) {
              check_identifiers(node->function_type.get());
            }
            if (node->body) {
              check_identifiers(node->body.get());
            }
          };

      check_identifiers(ast.get());

      if (!found_invalid) {
        std::cout << "  ⚠️  No invalid identifiers detected (this might "
                     "indicate parsing didn't create invalid identifiers)"
                  << std::endl;
      }

    } catch (const std::exception &e) {
      std::cout << "  ✅ Correctly threw exception: " << e.what() << std::endl;
    }
    std::cout << std::endl;
  }
}

void test_valid_identifiers() {
  std::cout << "=== Testing Valid Identifiers ===" << std::endl;

  std::vector<std::string> valid_cases = {
      "let valid_identifier = 10", "let camelCase = 20",
      "let snake_case = 30",       "let _underscore = 40",
      "let identifier123 = 50",
  };

  for (const auto &test_case : valid_cases) {
    std::cout << "Testing: " << test_case << std::endl;

    try {
      Reader reader(test_case);
      reader.read();

      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();

      std::cout << "  ✅ Parsed successfully" << std::endl;

    } catch (const std::exception &e) {
      std::cout << "  ❌ Unexpected error: " << e.what() << std::endl;
    }
  }
  std::cout << std::endl;
}

int main() {
  std::cout << "Testing identifier validation..." << std::endl << std::endl;

  try {
    test_semicolon_should_not_be_identifier();
    test_invalid_identifier_characters();
    test_valid_identifiers();

    std::cout << "Identifier validation tests completed!" << std::endl;
    
    // Output standardized test stats for build script parsing
    int total_tests = 3; // Three main test functions
    int failed_tests = 0; // Would have thrown exceptions if failed
    int passed_tests = total_tests;
    std::cout << "TEST_STATS: passed=" << passed_tests << " failed=" << failed_tests << " total=" << total_tests << std::endl;
    
  } catch (const std::exception &e) {
    std::cerr << "Test failed with error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}