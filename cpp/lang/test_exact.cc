#include "src/reader.h"
#include "src/ast.h"
#include <iostream>
#include <vector>

int main() {
    std::vector<std::pair<std::string, std::string>> invalid_cases = {
        {"fn", "Incomplete function declaration"},
        {"let", "Incomplete let statement"},
        {"fn : () ->", "Function missing name"},
        {"let = 42", "Let statement missing variable name"},
    };
    
    int correctly_failed = 0;
    int incorrectly_passed = 0;
    
    for (const auto &[test_case, description] : invalid_cases) {
        std::cout << "Testing: \"" << test_case << "\"" << std::endl;
        try {
            Reader reader(test_case);
            reader.read();
            
            ASTBuilder builder(reader.root.children);
            auto ast = builder.build();
            
            std::cout << "  ❌ INCORRECTLY PASSED: \"" << test_case << "\" (" 
                      << description << ")" << std::endl;
            incorrectly_passed++;
        } catch (const std::exception &e) {
            std::cout << "  ✅ Correctly rejected: \"" << test_case << "\" (" 
                      << description << ") - " << e.what() << std::endl;
            correctly_failed++;
        }
    }
    
    std::cout << "Summary: " << correctly_failed << " correctly rejected, " 
              << incorrectly_passed << " incorrectly passed" << std::endl;
    
    return 0;
}