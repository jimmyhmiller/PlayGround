#include "ast.h"
#include "ast_json.h"
#include "reader.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

std::string read_file(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return buffer.str();
}

void show_help(const std::string &program_name) {
  std::cout << "Usage: " << program_name << " [args...]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --help         Show this help message\n\n";
  std::cout << "Default behavior: reads and parses example.txt with reader and "
               "builds AST.\n";
}

int main(int argc, char *argv[]) {
  if (argc > 1) {
    std::string arg = argv[1];

    if (arg == "--help" || arg == "-h") {
      show_help(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      std::cerr << "Run '" << argv[0] << " --help' for help.\n";
      return 1;
    }
  }

  try {
    std::string content = read_file("example.txt");
    std::cout << "Reading example.txt (" << content.length()
              << " characters)...\n";

    Reader reader(content);
    reader.read();

    std::cout << "Parsed " << reader.root.children.size()
              << " reader expressions\n";

    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();

    std::cout << "Built AST with " << ast->child_count()
              << " top-level statements\n";

    size_t function_count = 0;
    size_t let_count = 0;
    size_t other_count = 0;

    for (size_t i = 0; i < ast->child_count(); ++i) {
      auto child = ast->child(i);
      if (child->type == ASTNodeType::FunctionDeclaration) {
        function_count++;
      } else if (child->type == ASTNodeType::LetStatement) {
        let_count++;
      } else {
        other_count++;
      }
    }

    std::cout << "Found: " << function_count << " functions, " << let_count
              << " let statements, " << other_count << " other statements\n";

    std::cout << "\nAST JSON representation:\n";
    std::cout << ast_to_json(ast.get()) << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}