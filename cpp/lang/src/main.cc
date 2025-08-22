#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "reader.h"

std::string read_file(const std::string& filename) {
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
  std::cout << "Default behavior: reads and parses example.txt with reader.\n";
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
    std::cout << "Reading example.txt (" << content.length() << " characters)...\n";
    
    Reader reader(content);
    reader.read();
    
    std::cout << "Parsed " << reader.root.children.size() << " top-level expressions\n";
    
    for (size_t i = 0; i < reader.root.children.size(); ++i) {
      std::cout << "Expression " << (i + 1) << ": " 
                << (reader.root.children[i].type == ReaderNodeType::Ident ? "Identifier" :
                    reader.root.children[i].type == ReaderNodeType::Literal ? "Literal" :
                    reader.root.children[i].type == ReaderNodeType::BinaryOp ? "BinaryOp" :
                    reader.root.children[i].type == ReaderNodeType::Block ? "Block" :
                    reader.root.children[i].type == ReaderNodeType::List ? "List" : "Other")
                << " '" << reader.root.children[i].value() << "'\n";
    }
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}