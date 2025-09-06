#include "src/ast.h"
#include "src/reader.h"
#include <iostream>
#include <fstream>

void print_ast_node(const ASTNode* node, int depth = 0) {
  if (!node) return;
  
  std::string indent(depth * 2, ' ');
  std::cout << indent << "Type: " << static_cast<int>(node->type) 
            << " (" << node->name << ")" << std::endl;
  
  if (node->type == ASTNodeType::FunctionDeclaration) {
    std::cout << indent << "  Function: " << node->name << std::endl;
    if (node->function_type) {
      std::cout << indent << "  Function Type Structure:" << std::endl;
      print_ast_node(node->function_type.get(), depth + 2);
    }
    if (node->body) {
      std::cout << indent << "  Function Body Structure:" << std::endl;
      print_ast_node(node->body.get(), depth + 2);
    } else {
      std::cout << indent << "  Function Body: NULL" << std::endl;
    }
  }
  
  for (size_t i = 0; i < node->child_count(); ++i) {
    print_ast_node(node->child(i), depth + 1);
  }
}

int main() {
  std::ifstream file("example.txt");
  std::string content((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
  
  try {
    Reader reader(content);
    reader.read();
    
    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();
    
    print_ast_node(ast.get());
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}