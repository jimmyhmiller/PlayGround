#include "src/ast.h"
#include "src/ast_code.h"
#include "src/reader.h"
#include <iostream>
#include <string>

void test_simple_parameters() {
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
  if (output.find("y, y") != std::string::npos) {
    std::cout << "❌ PROBLEM: Duplicate parameters detected!" << std::endl;
  } else if (output.find("x: i32, y: i32") != std::string::npos) {
    std::cout << "✅ SUCCESS: Parameters look correct!" << std::endl;
  } else {
    std::cout << "❓ UNKNOWN: Parameters format unexpected" << std::endl;
  }
  std::cout << std::endl;
}

void test_single_parameter() {
  std::cout << "=== Testing single parameter ===" << std::endl;
  
  std::string input = "fn double : (x: i32) -> i32 { x * 2 }";
  std::cout << "Input:  " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  std::string output = ast_to_code(ast.get());
  std::cout << "Output: " << output << std::endl;
  
  if (output.find("(x: i32)") != std::string::npos) {
    std::cout << "✅ SUCCESS: Single parameter looks correct!" << std::endl;
  } else {
    std::cout << "❌ PROBLEM: Single parameter format incorrect" << std::endl;
  }
  std::cout << std::endl;
}

void test_no_parameters() {
  std::cout << "=== Testing no parameters ===" << std::endl;
  
  std::string input = "fn test : () -> i32 { 42 }";
  std::cout << "Input:  " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  std::string output = ast_to_code(ast.get());
  std::cout << "Output: " << output << std::endl;
  
  if (output.find("()") != std::string::npos) {
    std::cout << "✅ SUCCESS: Empty parameters look correct!" << std::endl;
  } else {
    std::cout << "❌ PROBLEM: Empty parameters format incorrect" << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  std::cout << "Testing parameter parsing issues..." << std::endl << std::endl;
  
  test_no_parameters();
  test_single_parameter();
  test_simple_parameters();
  
  return 0;
}