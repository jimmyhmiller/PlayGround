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

void debug_case(const std::string& input) {
    std::cout << "=== Debugging: " << input << " ===" << std::endl;
    
    Reader reader(input);
    reader.read();
    
    std::cout << "Reader structure:" << std::endl;
    for (size_t i = 0; i < reader.root.children.size(); ++i) {
        std::cout << "Expression " << i << ":" << std::endl;
        print_reader_node(reader.root.children[i], 1);
    }
    
    std::cout << std::endl;
}

int main() {
    debug_case("fn test : () -> i32 { 42 }");
    debug_case("fn double : (x: i32) -> i32 { x * 2 }");
    debug_case("fn add : (x: i32, y: i32) -> i32 { x + y }");
    
    return 0;
}