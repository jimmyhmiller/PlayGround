#include "../../llvm/llvm_codegen.h"
#include "../ast.h"
#include "../reader.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

void show_help() {
  std::cout << "Usage: ast_to_llvm [options]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  --full-ir    Show full LLVM IR including wrapper function"
            << std::endl;
  std::cout
      << "  --no-exec    Don't execute the code, just generate and show IR"
      << std::endl;
  std::cout << "  --help       Show this help message" << std::endl;
  std::cout << std::endl;
  std::cout << "Reads code from stdin and compiles to LLVM IR." << std::endl;
}

int main(int argc, char *argv[]) {
  bool show_full_ir = false;
  bool no_exec = false;

  // Parse command line arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--full-ir") == 0) {
      show_full_ir = true;
    } else if (strcmp(argv[i], "--no-exec") == 0) {
      no_exec = true;
    } else if (strcmp(argv[i], "--help") == 0) {
      show_help();
      return 0;
    } else {
      std::cerr << "Unknown option: " << argv[i] << std::endl;
      show_help();
      return 1;
    }
  }

  std::ostringstream buffer;
  buffer << std::cin.rdbuf();
  std::string input = buffer.str();

  if (input.empty()) {
    std::cerr << "Error: No input provided" << std::endl;
    return 1;
  }

  try {
    // Parse the input
    Reader reader(input);
    reader.read();

    // Build AST
    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();

    // Generate LLVM IR
    LLVMCodeGenerator codegen;
    codegen.generateCode(ast.get());

    if (show_full_ir) {
      std::cout << "Generated LLVM IR (before execution wrapper):" << std::endl;
      std::cout << "=============================================="
                << std::endl;
      codegen.printIR();
    } else {
      std::cout << "Generated LLVM IR:" << std::endl;
      std::cout << "==================" << std::endl;
      codegen.printIR();
    }

    if (!no_exec) {
      std::cout << std::endl << "Executing expression..." << std::endl;
      double result = codegen.executeExpression();
      std::cout << "Result: " << result << std::endl;

      if (show_full_ir) {
        std::cout << std::endl
                  << "Full LLVM IR (with execution wrapper):" << std::endl;
        std::cout << "=======================================" << std::endl;
        codegen.printFullIR();
      }
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}