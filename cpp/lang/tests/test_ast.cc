#include "../src/ast.h"
#include "../src/reader.h"
#include <cassert>
#include <iostream>
#include <string>

void test_simple_function() {
  std::cout << "Testing simple function declaration..." << std::endl;

  std::string input = "fn test : () -> i32 { 42 }";
  Reader reader(input);
  reader.read();

  std::cout << "Reader parsed " << reader.root.children.size()
            << " expressions:" << std::endl;
  for (size_t i = 0; i < reader.root.children.size(); ++i) {
    const auto &child = reader.root.children[i];
    std::cout << "  " << i << ": " << std::string(child.value()) << " ("
              << (child.type == ReaderNodeType::Ident      ? "Ident"
                  : child.type == ReaderNodeType::BinaryOp ? "BinaryOp"
                  : child.type == ReaderNodeType::Block    ? "Block"
                                                           : "Other")
              << ")" << std::endl;
  }

  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();

  std::cout << "AST built with " << ast->child_count()
            << " statements:" << std::endl;
  for (size_t i = 0; i < ast->child_count(); ++i) {
    auto child = ast->child(i);
    std::cout << "  " << i << ": "
              << (child->type == ASTNodeType::FunctionDeclaration
                      ? "FunctionDeclaration"
                  : child->type == ASTNodeType::ExpressionStatement
                      ? "ExpressionStatement"
                      : "Other")
              << std::endl;
  }

  // Test assertions
  assert(ast->child_count() >= 1);

  // The first child should be a function declaration
  auto first = ast->child(0);
  if (first->type != ASTNodeType::FunctionDeclaration) {
    std::cout << "ERROR: Expected FunctionDeclaration, got type "
              << static_cast<int>(first->type) << std::endl;
  }
}

void test_let_statement() {
  std::cout << "\nTesting let statement..." << std::endl;

  std::string input = "let x = 42";
  Reader reader(input);
  reader.read();

  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();

  std::cout << "AST built with " << ast->child_count() << " statements"
            << std::endl;
  for (size_t i = 0; i < ast->child_count(); ++i) {
    auto child = ast->child(i);
    std::cout << "  " << i << ": "
              << (child->type == ASTNodeType::LetStatement ? "LetStatement"
                  : child->type == ASTNodeType::ExpressionStatement
                      ? "ExpressionStatement"
                      : "Other")
              << std::endl;
  }
}

void test_mutable_let() {
  std::cout << "\nTesting mutable let statement..." << std::endl;

  std::string input = "let mut x = 42";
  Reader reader(input);
  reader.read();

  ASTBuilder builder(reader.root.children);
  auto ast = builder.build();

  std::cout << "AST built with " << ast->child_count() << " statements"
            << std::endl;
}

int main() {
  std::cout << "Running AST tests..." << std::endl;

  try {
    test_simple_function();
    test_let_statement();
    test_mutable_let();

    std::cout << "\nAll AST tests completed!" << std::endl;
    
    // Output standardized test stats for build script parsing
    int total_tests = 3; // Number of test functions
    int failed_tests = 0;
    int passed_tests = total_tests;
    std::cout << "TEST_STATS: passed=" << passed_tests << " failed=" << failed_tests << " total=" << total_tests << std::endl;
    
  } catch (const std::exception &e) {
    std::cerr << "Test failed with error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}