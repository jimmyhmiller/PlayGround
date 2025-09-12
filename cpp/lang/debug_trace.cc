#include "src/reader.h"
#include "src/ast.h"
#include <iostream>

int main() {
    std::string test_case = "a b c d +++";
    
    try {
        std::cout << "1. Creating reader..." << std::endl;
        Reader reader(test_case);
        
        std::cout << "2. Reading..." << std::endl;
        reader.read();
        
        std::cout << "3. Reader parsed " << reader.root.children.size() << " expressions" << std::endl;
        
        std::cout << "4. Creating AST builder..." << std::endl;
        ASTBuilder builder(reader.root.children);
        
        std::cout << "5. Calling build()..." << std::endl;
        auto ast = builder.build();
        
        std::cout << "6. AST built successfully - THIS IS THE PROBLEM!" << std::endl;
        return 1;
        
    } catch (const std::exception &e) {
        std::cout << "✅ Exception caught: " << e.what() << std::endl;
        return 0;
    }
}