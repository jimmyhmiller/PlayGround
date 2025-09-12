#include "src/reader.h"
#include "src/ast.h"
#include <iostream>

int main() {
    std::string test_case = "a b c d +++";
    std::cout << "Testing: \"" << test_case << "\" (Trailing operators with no operand)" << std::endl;
    
    try {
        Reader reader(test_case);
        reader.read();
        
        ASTBuilder builder(reader.root.children);
        auto ast = builder.build();
        
        std::cout << "❌ INCORRECTLY PASSED" << std::endl;
        return 1;
    } catch (const std::exception &e) {
        std::cout << "✅ Correctly rejected: " << e.what() << std::endl;
        return 0;
    }
}