#include "../src/ast.h"
#include "../src/reader.h"
#include <cassert>
#include <iostream>
#include <string>

void test_assignment_vs_expression() {
  std::cout << "=== Testing Assignment vs Expression ===" << std::endl;
  
  std::string input = "x = y + 1";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  // Should be an assignment statement, not a binary expression
  auto first = ast->child(0);
  std::cout << "First child type: " << static_cast<int>(first->type) << std::endl;
  
  if (first->type == ASTNodeType::AssignmentStatement) {
    std::cout << "✅ SUCCESS: Assignment parsed correctly" << std::endl;
  } else {
    std::cout << "❌ FAILED: Assignment parsed as: " << static_cast<int>(first->type) << std::endl;
  }
  std::cout << std::endl;
}

void test_empty_list_literal() {
  std::cout << "=== Testing Empty List Literal ===" << std::endl;
  
  std::string input = "[]";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  auto first = ast->child(0);
  auto expr = first->child(0);
  std::cout << "Expression type: " << static_cast<int>(expr->type) << std::endl;
  
  if (expr->type == ASTNodeType::ListLiteral) {
    std::cout << "✅ SUCCESS: Empty list parsed correctly" << std::endl;
  } else {
    std::cout << "❌ FAILED: Empty list parsed as: " << static_cast<int>(expr->type) << std::endl;
  }
  std::cout << std::endl;
}

void test_list_literal_with_elements() {
  std::cout << "=== Testing List Literal with Elements ===" << std::endl;
  
  std::string input = "[1, 2, 3]";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  auto first = ast->child(0);
  auto expr = first->child(0);
  std::cout << "Expression type: " << static_cast<int>(expr->type) << std::endl;
  
  if (expr->type == ASTNodeType::ListLiteral) {
    std::cout << "✅ SUCCESS: List literal parsed correctly" << std::endl;
  } else {
    std::cout << "❌ FAILED: List literal parsed as: " << static_cast<int>(expr->type) << std::endl;
  }
  std::cout << std::endl;
}

void test_tuple_literal() {
  std::cout << "=== Testing Tuple Literal ===" << std::endl;
  
  std::string input = "(a, b)";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  auto first = ast->child(0);
  auto expr = first->child(0);
  std::cout << "Expression type: " << static_cast<int>(expr->type) << std::endl;
  
  if (expr->type == ASTNodeType::TupleLiteral) {
    std::cout << "✅ SUCCESS: Tuple literal parsed correctly" << std::endl;
  } else {
    std::cout << "❌ FAILED: Tuple literal parsed as: " << static_cast<int>(expr->type) << std::endl;
  }
  std::cout << std::endl;
}

void test_lambda_expression() {
  std::cout << "=== Testing Lambda Expression ===" << std::endl;
  
  std::string input = "x => f(x)";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  std::cout << "AST has " << ast->child_count() << " children" << std::endl;
  
  auto first = ast->child(0);
  std::cout << "First child type: " << static_cast<int>(first->type) << std::endl;
  
  if (first->type == ASTNodeType::ExpressionStatement) {
    auto expr = first->child(0);
    if (expr->type == ASTNodeType::LambdaExpression) {
      std::cout << "✅ SUCCESS: Lambda expression parsed correctly" << std::endl;
    } else {
      std::cout << "❌ FAILED: Lambda expression parsed as: " << static_cast<int>(expr->type) << std::endl;
    }
  } else {
    std::cout << "❌ FAILED: Expected ExpressionStatement, got: " << static_cast<int>(first->type) << std::endl;
  }
  std::cout << std::endl;
}

void test_function_call() {
  std::cout << "=== Testing Function Call ===" << std::endl;
  
  std::string input = "head(xs)";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  auto first = ast->child(0);
  auto expr = first->child(0);
  std::cout << "Expression type: " << static_cast<int>(expr->type) << std::endl;
  
  if (expr->type == ASTNodeType::FunctionCall) {
    std::cout << "✅ SUCCESS: Function call parsed correctly" << std::endl;
  } else {
    std::cout << "❌ FAILED: Function call parsed as: " << static_cast<int>(expr->type) << std::endl;
  }
  std::cout << std::endl;
}

void test_logical_or() {
  std::cout << "=== Testing Logical OR ===" << std::endl;
  
  std::string input = "a == [] || b == []";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  auto first = ast->child(0);
  auto expr = first->child(0);
  std::cout << "Expression type: " << static_cast<int>(expr->type) << std::endl;
  
  if (expr->type == ASTNodeType::BinaryExpression) {
    std::cout << "✅ SUCCESS: Logical OR parsed as binary expression" << std::endl;
  } else {
    std::cout << "❌ FAILED: Logical OR parsed as: " << static_cast<int>(expr->type) << std::endl;
  }
  std::cout << std::endl;
}

void test_generic_types() {
  std::cout << "=== Testing Generic Types ===" << std::endl;
  
  std::string input = "fn id : (x: t) -> t { x }";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  auto first = ast->child(0);
  std::cout << "Function type: " << static_cast<int>(first->type) << std::endl;
  
  if (first->type == ASTNodeType::FunctionDeclaration) {
    std::cout << "✅ SUCCESS: Generic function parsed correctly" << std::endl;
  } else {
    std::cout << "❌ FAILED: Generic function parsed as: " << static_cast<int>(first->type) << std::endl;
  }
  std::cout << std::endl;
}

void test_complex_types() {
  std::cout << "=== Testing Complex Types ===" << std::endl;
  
  std::string input = "fn len : (xs: List t) -> u32 { 0 }";
  std::cout << "Input: " << input << std::endl;
  
  Reader reader(input);
  reader.read();
  
  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();
  
  auto first = ast->child(0);
  std::cout << "Function type: " << static_cast<int>(first->type) << std::endl;
  
  if (first->type == ASTNodeType::FunctionDeclaration) {
    std::cout << "✅ SUCCESS: Complex type function parsed correctly" << std::endl;
  } else {
    std::cout << "❌ FAILED: Complex type function parsed as: " << static_cast<int>(first->type) << std::endl;
  }
  std::cout << std::endl;
}

int main() {
  std::cout << "Testing example.txt syntax support..." << std::endl << std::endl;
  
  try {
    test_assignment_vs_expression();
    test_empty_list_literal();
    test_list_literal_with_elements();
    test_tuple_literal();
    test_lambda_expression();
    test_function_call();
    test_logical_or();
    test_generic_types();
    test_complex_types();
    
    std::cout << "Example syntax tests completed!" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Test failed with error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}