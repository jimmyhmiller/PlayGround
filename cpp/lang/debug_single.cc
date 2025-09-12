#include "src/reader.h"
#include "src/ast.h"
#include <iostream>

int main() {
    std::string test_case = "fn";
    std::cout << "Testing: " << test_case << std::endl;
    
    try {
        Reader reader(test_case);
        reader.read();
        
        ASTBuilder builder(reader.root.children);
        auto ast = builder.build();
        
        std::cout << "❌ INCORRECTLY PASSED" << std::endl;
    } catch (const std::exception &e) {
        std::cout << "✅ Correctly rejected: " << e.what() << std::endl;
    }
    
    return 0;
}