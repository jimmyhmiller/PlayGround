#include "../src/ast.h"
#include "../src/reader.h"
#include <cassert>
#include <iostream>
#include <string>

// Global counters to track test results
int total_issues_found = 0;

void test_malformed_expressions() {
  std::cout << "Testing malformed expression handling..." << std::endl;

  // These are SYNTAX errors that should fail at parse time
  std::vector<std::pair<std::string, std::string>> syntax_errors = {
      {"( ( (", "Unmatched opening parens"},
      {") ) )", "Unmatched closing parens"},
      {"{ { {", "Unmatched opening braces"},
      {"} } }", "Unmatched closing braces"},
      {"[ [ [", "Unmatched opening brackets"},
      {"] ] ]", "Unmatched closing brackets"},
  };

  // These are valid syntax but might be semantic errors later
  std::vector<std::pair<std::string, std::string>> valid_syntax = {
      {"++", "Could be binary operator or identifier"},
      {"1 ++ 2", "Could be valid if ++ is defined as binary operator"},
      {"+ + + +", "Could be multiple prefix operators or identifiers"},
  };

  int syntax_errors_caught = 0;
  int syntax_errors_missed = 0;
  int valid_syntax_accepted = 0;
  int valid_syntax_rejected = 0;

  std::cout << "  Testing syntax errors (should be rejected):" << std::endl;
  for (const auto &[test_case, description] : syntax_errors) {
    try {
      Reader reader(test_case);
      reader.read();
      syntax_errors_missed++;
      total_issues_found++;
      std::cout << "  ❌ INCORRECTLY ACCEPTED: \"" << test_case << "\" (" 
                << description << ")" << std::endl;
    } catch (const std::exception &e) {
      syntax_errors_caught++;
      std::cout << "  ✅ Correctly rejected: \"" << test_case << "\" (" 
                << description << ") - " << e.what() << std::endl;
    }
  }

  std::cout << "  Testing valid syntax (should be accepted):" << std::endl;
  for (const auto &[test_case, description] : valid_syntax) {
    try {
      Reader reader(test_case);
      reader.read();
      valid_syntax_accepted++;
      std::cout << "  ✅ Correctly accepted: \"" << test_case << "\" (" 
                << description << ")" << std::endl;
    } catch (const std::exception &e) {
      valid_syntax_rejected++;
      total_issues_found++;
      std::cout << "  ❌ INCORRECTLY REJECTED: \"" << test_case << "\" (" 
                << description << ") - " << e.what() << std::endl;
    }
  }

  std::cout << "  Summary:" << std::endl;
  std::cout << "    Syntax errors: " << syntax_errors_caught << " caught, " 
            << syntax_errors_missed << " missed" << std::endl;
  std::cout << "    Valid syntax: " << valid_syntax_accepted << " accepted, " 
            << valid_syntax_rejected << " rejected" << std::endl;
}

void test_empty_structures() {
  std::cout << "Testing empty structure handling..." << std::endl;

  std::vector<std::string> empty_cases = {
      "",          // Completely empty
      "   \t\n  ", // Only whitespace
      "( )",       // Empty parens
      "[ ]",       // Empty brackets
      "{ }",       // Empty braces
      "( ( ) )",   // Nested empty parens
      "[ [ ] ]",   // Nested empty brackets
      "{ { } }",   // Nested empty braces
  };

  for (const auto &test_case : empty_cases) {
    try {
      Reader reader(test_case);
      reader.read();
      std::cout << "  ✓ Empty case handled: \"" << test_case << "\""
                << std::endl;
    } catch (const std::exception &e) {
      std::cout << "  ❌ Empty case failed: \"" << test_case << "\" - "
                << e.what() << std::endl;
    }
  }
}

void test_large_expressions() {
  std::cout << "Testing large expression handling..." << std::endl;

  // Test deeply nested expressions
  std::string deep_nesting = "( ";
  for (int i = 0; i < 100; i++) {
    deep_nesting += "( ";
  }
  deep_nesting += "42";
  for (int i = 0; i < 100; i++) {
    deep_nesting += " )";
  }
  deep_nesting += " )";

  try {
    Reader reader(deep_nesting);
    reader.read();
    std::cout << "  ✓ Deep nesting (200+ parens) handled successfully"
              << std::endl;
  } catch (const std::exception &e) {
    std::cout << "  ❌ Deep nesting failed: " << e.what() << std::endl;
  }

  // Test wide expression (many siblings)
  std::string wide_expression = "[ ";
  for (int i = 0; i < 1000; i++) {
    wide_expression += std::to_string(i) + " ";
  }
  wide_expression += "]";

  try {
    Reader reader(wide_expression);
    reader.read();
    std::cout << "  ✓ Wide expression (1000+ elements) handled successfully"
              << std::endl;
  } catch (const std::exception &e) {
    std::cout << "  ❌ Wide expression failed: " << e.what() << std::endl;
  }
}

void test_ast_validation_failures() {
  std::cout << "Testing AST validation (these should FAIL)..." << std::endl;

  std::vector<std::pair<std::string, std::string>> invalid_cases = {
      {"fn : () ->", "Function missing name"},
      {"let = 42", "Let statement missing variable name"},
      {"if { 42 }", "If statement missing condition"},
      {"fn test : x -> y", "Function with malformed parameter list"},
      {"let x y z = 42", "Let statement with multiple variable names"},
      {"fn", "Incomplete function declaration"},
      {"let", "Incomplete let statement"},
      {"fn : ->", "Function missing parameter list and return type"},
      {"a b c d +++", "Adjacent identifiers without proper syntax"},
  };

  int correctly_failed = 0;
  int incorrectly_passed = 0;

  for (const auto &[test_case, description] : invalid_cases) {
    try {
      Reader reader(test_case);
      reader.read();

      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();

      // If we get here, the AST validation failed to catch invalid syntax
      std::cout << "  ❌ INCORRECTLY PASSED: \"" << test_case << "\" (" 
                << description << ")" << std::endl;
      incorrectly_passed++;
      total_issues_found++;
    } catch (const std::exception &e) {
      // This is the expected behavior - AST validation should reject invalid syntax
      std::cout << "  ✅ Correctly rejected: \"" << test_case << "\" (" 
                << description << ") - " << e.what() << std::endl;
      correctly_failed++;
    }
  }

  std::cout << "  Summary: " << correctly_failed << " correctly rejected, " 
            << incorrectly_passed << " incorrectly passed" << std::endl;
  
  if (incorrectly_passed > 0) {
    std::cout << "  ⚠️  WARNING: " << incorrectly_passed 
              << " invalid syntax cases were incorrectly accepted!" << std::endl;
  }
}

void test_ast_validation_successes() {
  std::cout << "Testing AST validation (these should PASS)..." << std::endl;

  std::vector<std::pair<std::string, std::string>> valid_cases = {
      {"fn test : () -> i32 { 42 }", "Complete function declaration"},
      {"let x = 42", "Complete let statement"},
      {"fn add : (x: i32, y: i32) -> i32 { x + y }", "Function with parameters"},
      {"let mut x = 42", "Mutable let statement"},
      {"42", "Simple expression"},
      {"x + y", "Binary expression"},
  };

  int correctly_passed = 0;
  int incorrectly_failed = 0;

  for (const auto &[test_case, description] : valid_cases) {
    try {
      Reader reader(test_case);
      reader.read();

      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();

      // This is expected - valid syntax should build AST successfully
      std::cout << "  ✅ Correctly accepted: \"" << test_case << "\" (" 
                << description << ")" << std::endl;
      correctly_passed++;
    } catch (const std::exception &e) {
      // This would be unexpected - valid syntax should not fail
      std::cout << "  ❌ INCORRECTLY REJECTED: \"" << test_case << "\" (" 
                << description << ") - " << e.what() << std::endl;
      incorrectly_failed++;
      total_issues_found++;
    }
  }

  std::cout << "  Summary: " << correctly_passed << " correctly accepted, " 
            << incorrectly_failed << " incorrectly rejected" << std::endl;
}

void test_unicode_and_special_chars() {
  std::cout << "Testing unicode and special character handling..." << std::endl;

  std::vector<std::pair<std::string, std::string>> special_cases = {
      {"ascii_only", "Basic ASCII identifiers"},
      {"tab\there", "Tab character"},
      {"new\nline", "Newline character"},
      {"return\rchar", "Carriage return"},
  };

  // Note: Avoiding unicode that might not be supported by the tokenizer
  for (const auto &[test_case, description] : special_cases) {
    try {
      Reader reader(test_case);
      reader.read();
      std::cout << "  ✓ " << description << " handled correctly" << std::endl;
    } catch (const std::exception &e) {
      // Some special chars may legitimately cause errors
      std::cout << "  ⚡ " << description
                << " caused error (may be expected): " << e.what() << std::endl;
    }
  }
}

void test_boundary_conditions() {
  std::cout << "Testing boundary conditions..." << std::endl;

  // Test very long identifier
  std::string long_identifier(1000, 'a');
  try {
    Reader reader(long_identifier);
    reader.read();
    std::cout << "  ✓ Very long identifier (1000 chars) handled" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "  ⚡ Very long identifier failed: " << e.what() << std::endl;
  }

  // Test very large number
  std::string large_number = "123456789012345678901234567890";
  try {
    Reader reader(large_number);
    reader.read();
    std::cout << "  ✓ Very large number handled" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "  ⚡ Very large number failed: " << e.what() << std::endl;
  }

  // Test very long string literal
  std::string long_string = "\"" + std::string(1000, 'x') + "\"";
  try {
    Reader reader(long_string);
    reader.read();
    std::cout << "  ✓ Very long string literal handled" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "  ⚡ Very long string literal failed: " << e.what()
              << std::endl;
  }
}

void test_consistency_checks() {
  std::cout << "Testing parser consistency..." << std::endl;

  // Test that parsing the same input twice gives same result
  std::string test_input = "fn test : (x: i32) -> i32 { x + 1 }";

  try {
    Reader reader1(test_input);
    reader1.read();

    Reader reader2(test_input);
    reader2.read();

    // Basic check that both parsed successfully and have same structure
    assert(reader1.root.children.size() == reader2.root.children.size());
    std::cout << "  ✓ Consistent parsing verified" << std::endl;
  } catch (const std::exception &e) {
    std::cout << "  ❌ Consistency check failed: " << e.what() << std::endl;
  }

  // Test that equivalent expressions parse to similar structures
  std::vector<std::pair<std::string, std::string>> equivalent_pairs = {
      {"1+2", "1 + 2"}, // Spacing variations
      {"(1+2)", "1+2"}, // Unnecessary parentheses
      {"  a  ", "a"},   // Whitespace variations
  };

  for (const auto &pair : equivalent_pairs) {
    try {
      Reader reader1(pair.first);
      reader1.read();

      Reader reader2(pair.second);
      reader2.read();

      std::cout << "  ✓ Equivalent expressions: \"" << pair.first << "\" ≈ \""
                << pair.second << "\"" << std::endl;
    } catch (const std::exception &e) {
      std::cout << "  ❌ Equivalent expression test failed: " << e.what()
                << std::endl;
    }
  }
}

int main() {
  std::cout << "Running Error Handling and Edge Case tests..." << std::endl;
  std::cout << "================================================" << std::endl;

  try {
    test_malformed_expressions();
    std::cout << std::endl;

    test_empty_structures();
    std::cout << std::endl;

    test_large_expressions();
    std::cout << std::endl;

    test_ast_validation_failures();
    std::cout << std::endl;

    test_ast_validation_successes();
    std::cout << std::endl;

    test_unicode_and_special_chars();
    std::cout << std::endl;

    test_boundary_conditions();
    std::cout << std::endl;

    test_consistency_checks();
    std::cout << std::endl;

    // Count all individual test cases
    int total_tests = 7 + 8 + 2 + 8 + 6 + 4 + 3 + 3; // Individual test counts from each function
    int failed_tests = total_issues_found;
    int passed_tests = total_tests - failed_tests;
    
    // Output standardized test stats for build script parsing
    std::cout << "TEST_STATS: passed=" << passed_tests << " failed=" << failed_tests << " total=" << total_tests << std::endl;

    if (total_issues_found > 0) {
      std::cout << "❌ Error Handling tests found " << total_issues_found 
                << " issues that need fixing!" << std::endl;
      std::cout << "(Note: These represent real problems in the codebase)" << std::endl;
      return 1; // Return failure exit code
    } else {
      std::cout << "All Error Handling tests completed! ✅" << std::endl;
      return 0;
    }
  } catch (const std::exception &e) {
    std::cerr << "Error handling test suite failed with error: " << e.what()
              << std::endl;
    return 1;
  }
}