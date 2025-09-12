#include "src/reader.h"
#include "src/ast.h"
#include <iostream>

void print_reader_node(const ReaderNode& node, int depth = 0) {
    std::string indent(depth * 2, ' ');
    std::cout << indent << node.value() << " (" << (int)node.type << ")";
    if (!node.children.empty()) {
        std::cout << " {" << std::endl;
        for (const auto& child : node.children) {
            print_reader_node(child, depth + 1);
        }
        std::cout << indent << "}";
    }
    std::cout << std::endl;
}

void test_invalid_case(const std::string& input) {
    std::cout << "=== Testing: " << input << " ===" << std::endl;
    
    try {
        Reader reader(input);
        reader.read();
        
        std::cout << "Reader structure:" << std::endl;
        for (size_t i = 0; i < reader.root.children.size(); ++i) {
            std::cout << "Expression " << i << ":" << std::endl;
            print_reader_node(reader.root.children[i], 1);
        }
        
        ASTBuilder builder(reader.root.children);
        auto ast = builder.build();
        
        std::cout << "AST built successfully - THIS IS THE PROBLEM!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Correctly caught error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    test_invalid_case("fn");
    test_invalid_case("let");
    test_invalid_case("fn : () ->");
    test_invalid_case("let = 42");
    test_invalid_case("if { 42 }");
    test_invalid_case("fn test : x -> y");
    test_invalid_case("fn : ->");
    test_invalid_case("let x y z = 42");
    
    return 0;
}