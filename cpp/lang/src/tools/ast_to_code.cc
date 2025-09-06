#include "../ast.h"
#include "../ast_code.h"
#include "../reader.h"
#include <iostream>
#include <sstream>
#include <string>

int main() {
  std::ostringstream buffer;
  buffer << std::cin.rdbuf();
  std::string input = buffer.str();

  if (input.empty()) {
    std::cerr << "Error: No input provided" << std::endl;
    return 1;
  }

  try {
    Reader reader(input);
    reader.read();

    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();

    std::cout << ast_to_code(ast.get()) << std::endl;

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}