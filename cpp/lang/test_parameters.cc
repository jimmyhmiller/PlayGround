#include "src/ast.h"
#include "src/ast_code.h"
#include "src/reader.h"
#include <iostream>
#include <string>

bool test_simple_parameters() {
  std::cout << "=== Testing simple parameters ===" << std::endl;

  std::string input = "fn add : (x: i32, y: i32) -> i32 { x + y }";
  std::cout << "Input:  " << input << std::endl;

  Reader reader(input);
  reader.read();

  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();

  std::string output = ast_to_code(ast.get());
  std::cout << "Output: " << output << std::endl;

  // Check if we have duplicate parameters
  if (output.find("/* unknown expression */") != std::string::npos) {
    std::cout << "❌ PROBLEM: Contains unknown expression - AST building failed!" << std::endl;
    return false;
  } else if (output.find("y, y") != std::string::npos) {
    std::cout << "❌ PROBLEM: Duplicate parameters detected!" << std::endl;
    return false;
  } else if (output.find("x: i32, y: i32") != std::string::npos) {
    std::cout << "✅ SUCCESS: Parameters look correct!" << std::endl;
    return true;
  } else {
    std::cout << "❓ UNKNOWN: Parameters format unexpected" << std::endl;
    return false;
  }
}

bool test_single_parameter() {
  std::cout << "=== Testing single parameter ===" << std::endl;

  std::string input = "fn double : (x: i32) -> i32 { x * 2 }";
  std::cout << "Input:  " << input << std::endl;

  Reader reader(input);
  reader.read();

  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();

  std::string output = ast_to_code(ast.get());
  std::cout << "Output: " << output << std::endl;

  if (output.find("/* unknown expression */") != std::string::npos) {
    std::cout << "❌ PROBLEM: Contains unknown expression - AST building failed!" << std::endl;
    return false;
  } else if (output.find("(x: i32)") != std::string::npos) {
    std::cout << "✅ SUCCESS: Single parameter looks correct!" << std::endl;
    return true;
  } else {
    std::cout << "❌ PROBLEM: Single parameter format incorrect" << std::endl;
    return false;
  }
}

bool test_no_parameters() {
  std::cout << "=== Testing no parameters ===" << std::endl;

  std::string input = "fn test : () -> i32 { 42 }";
  std::cout << "Input:  " << input << std::endl;

  Reader reader(input);
  reader.read();

  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();

  std::string output = ast_to_code(ast.get());
  std::cout << "Output: " << output << std::endl;

  if (output.find("/* unknown expression */") != std::string::npos) {
    std::cout << "❌ PROBLEM: Contains unknown expression - AST building failed!" << std::endl;
    return false; // Return failure indicator
  } else if (output.find("()") != std::string::npos) {
    std::cout << "✅ SUCCESS: Empty parameters look correct!" << std::endl;
    return true;
  } else {
    std::cout << "❌ PROBLEM: Empty parameters format incorrect" << std::endl;
    return false;
  }
}

int main() {
  std::cout << "Testing parameter parsing issues..." << std::endl << std::endl;

  int failed_tests = 0;
  int total_tests = 3;
  
  if (!test_no_parameters()) failed_tests++;
  std::cout << std::endl;
  
  if (!test_single_parameter()) failed_tests++;
  std::cout << std::endl;
  
  if (!test_simple_parameters()) failed_tests++;
  std::cout << std::endl;

  int passed_tests = total_tests - failed_tests;
  
  // Output standardized test stats for build script parsing
  std::cout << "TEST_STATS: passed=" << passed_tests << " failed=" << failed_tests << " total=" << total_tests << std::endl;

  if (failed_tests > 0) {
    std::cout << "❌ " << failed_tests << " parameter tests failed!" << std::endl;
    return 1; // Return failure exit code
  } else {
    std::cout << "✅ All parameter tests passed!" << std::endl;
    return 0;
  }
}