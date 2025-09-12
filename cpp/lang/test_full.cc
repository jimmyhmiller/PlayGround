#include "src/reader.h"
#include "src/ast.h"
#include "src/ast_code.h"
#include <iostream>

void debug_ast_node(const ASTNode* node, int depth = 0) {
    if (!node) {
        std::string indent(depth * 2, ' ');
        std::cout << indent << "null" << std::endl;
        return;
    }
    
    std::string indent(depth * 2, ' ');
    std::cout << indent << "Type: " << (int)node->type << ", Name: '" << node->name << "', Value: '" << node->value << "'" << std::endl;
    
    if (node->function_type) {
        std::cout << indent << "Function Type:" << std::endl;
        debug_ast_node(node->function_type.get(), depth + 1);
    }
    
    if (node->body) {
        std::cout << indent << "Body:" << std::endl;
        debug_ast_node(node->body.get(), depth + 1);
    }
    
    for (size_t i = 0; i < node->child_count(); ++i) {
        std::cout << indent << "Child " << i << ":" << std::endl;
        debug_ast_node(node->child(i), depth + 1);
    }
}

int main() {
    std::string input = "fn double : (x: i32) -> i32 { x * 2 }";
    std::cout << "Testing: " << input << std::endl;
    
    Reader reader(input);
    reader.read();
    
    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();
    
    std::cout << "AST structure:" << std::endl;
    debug_ast_node(ast.get());
    
    std::cout << "\nGenerated code:" << std::endl;
    std::string output = ast_to_code(ast.get());
    std::cout << output << std::endl;
    
    return 0;
}