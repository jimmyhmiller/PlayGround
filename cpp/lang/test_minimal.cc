#include "src/reader.h"
#include "src/ast.h"
#include "src/ast_code.h"
#include <iostream>

int main() {
    std::string input = "fn test : () -> i32 { 42 }";
    
    Reader reader(input);
    reader.read();
    
    // Create a builder with just the empty parameter list part
    const ReaderNode& colon_node = reader.root.children[1]; // the ":" binary op
    const ReaderNode& arrow_node = colon_node.children[1]; // the "->" binary op  
    const ReaderNode& param_node = arrow_node.children[0]; // the "()" list
    
    std::cout << "Parameter node type: " << (int)param_node.type << std::endl;
    std::cout << "Parameter node value: '" << param_node.value() << "'" << std::endl;
    std::cout << "Parameter node children count: " << param_node.children.size() << std::endl;
    
    // Test parse_parameter_list directly
    ASTBuilder param_builder({param_node});
    auto param_list = param_builder.parse_parameter_list();
    
    if (param_list) {
        std::cout << "Parameter list parsed successfully!" << std::endl;
        std::cout << "Parameter list type: " << (int)param_list->type << std::endl;
        std::cout << "Parameter list children: " << param_list->child_count() << std::endl;
    } else {
        std::cout << "Parameter list parsing returned nullptr!" << std::endl;
    }
    
    return 0;
}