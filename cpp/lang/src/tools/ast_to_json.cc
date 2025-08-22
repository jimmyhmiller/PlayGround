#include "../ast.h"
#include "../ast_json.h"
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

    // TODO: Implement AST building and conversion
    std::cout << "AST building not yet implemented" << std::endl;

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}