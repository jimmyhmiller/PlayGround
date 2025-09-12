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

void test_case(const std::string& input) {
    std::cout << "=== Testing: \"" << input << "\" ===" << std::endl;
    
    try {
        Reader reader(input);
        reader.read();
        
        std::cout << "Reader parsed " << reader.root.children.size() << " expressions:" << std::endl;
        for (size_t i = 0; i < reader.root.children.size(); ++i) {
            std::cout << "Expression " << i << ":" << std::endl;
            print_reader_node(reader.root.children[i], 1);
        }
        
        ASTBuilder builder(reader.root.children);
        auto ast = builder.build();
        
        std::cout << "AST built successfully - SHOULD THIS BE ALLOWED?" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Correctly caught error: " << e.what() << std::endl;
    }
    
    std::cout << std::endl;
}

int main() {
    test_case("a b c d +++");
    test_case("a b c");
    
    return 0;
}