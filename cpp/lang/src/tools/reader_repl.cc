#include "../reader.h"
#include <cstdio>  // for fileno
#include <cstring> // for strcmp
#include <iostream>
#include <string>
#include <unistd.h> // for isatty

std::string indent(int level) { return std::string(level * 2, ' '); }

std::string node_type_to_string(ReaderNodeType type) {
  switch (type) {
  case ReaderNodeType::Ident:
    return "ident";
  case ReaderNodeType::Literal:
    return "literal";
  case ReaderNodeType::List:
    return "list";
  case ReaderNodeType::Block:
    return "block";
  case ReaderNodeType::BinaryOp:
    return "binary-op";
  case ReaderNodeType::PrefixOp:
    return "prefix-op";
  case ReaderNodeType::PostfixOp:
    return "postfix-op";
  case ReaderNodeType::Call:
    return "call";
  default:
    return "unknown";
  }
}

void print_node(const ReaderNode &node, int level = 0) {
  std::cout << indent(level) << "(" << node_type_to_string(node.type);

  if (node.type != ReaderNodeType::List && !node.value().empty()) {
    std::cout << " \"" << node.value() << "\"";
    std::cout << " [" << node.token.line << ":" << node.token.column << "]";
  }

  if (!node.children.empty()) {
    std::cout << "\n";
    for (const auto &child : node.children) {
      print_node(child, level + 1);
    }
    std::cout << indent(level) << ")";
  } else {
    std::cout << ")";
  }

  if (level > 0) {
    std::cout << "\n";
  }
}

void print_sexpr(const ReaderNode &node) {
  if (node.type == ReaderNodeType::BinaryOp) {
    std::cout << "(" << node.value();
    for (const auto &child : node.children) {
      std::cout << " ";
      print_sexpr(child);
    }
    std::cout << ")";
  } else if (node.type == ReaderNodeType::PrefixOp) {
    std::cout << "(" << node.value();
    for (const auto &child : node.children) {
      std::cout << " ";
      print_sexpr(child);
    }
    std::cout << ")";
  } else if (node.type == ReaderNodeType::PostfixOp) {
    std::cout << "(postfix-" << node.value();
    for (const auto &child : node.children) {
      std::cout << " ";
      print_sexpr(child);
    }
    std::cout << ")";
  } else if (node.type == ReaderNodeType::Call) {
    std::cout << "(call";
    for (const auto &child : node.children) {
      std::cout << " ";
      print_sexpr(child);
    }
    std::cout << ")";
  } else if (node.type == ReaderNodeType::List) {
    std::cout << "(";
    bool first = true;
    for (const auto &child : node.children) {
      if (!first)
        std::cout << " ";
      first = false;
      print_sexpr(child);
    }
    std::cout << ")";
  } else if (node.type == ReaderNodeType::Block) {
    std::cout << "(block";
    for (const auto &child : node.children) {
      std::cout << " ";
      print_sexpr(child);
    }
    std::cout << ")";
  } else if (node.type == ReaderNodeType::Ident) {
    if (node.children.empty()) {
      std::cout << node.value();
    } else {
      std::cout << "(" << node.value();
      for (const auto &child : node.children) {
        std::cout << " ";
        print_sexpr(child);
      }
      std::cout << ")";
    }
  } else if (node.type == ReaderNodeType::Literal) {
    if (node.children.empty()) {
      std::cout << node.value();
    } else {
      std::cout << "(" << node.value();
      for (const auto &child : node.children) {
        std::cout << " ";
        print_sexpr(child);
      }
      std::cout << ")";
    }
  } else {
    // Generic case - any node type with potential children
    if (node.children.empty()) {
      std::cout << node.value();
    } else {
      std::cout << "(" << node.value();
      for (const auto &child : node.children) {
        std::cout << " ";
        print_sexpr(child);
      }
      std::cout << ")";
    }
  }
}

void print_parsed_result(const Reader &reader, bool sexpr_mode = false) {
  if (sexpr_mode) {
    if (reader.root.children.empty()) {
      std::cout << "()\n";
    } else if (reader.root.children.size() == 1) {
      print_sexpr(reader.root.children[0]);
      std::cout << "\n";
    } else {
      std::cout << "(";
      bool first = true;
      for (const auto &child : reader.root.children) {
        if (!first)
          std::cout << " ";
        first = false;
        print_sexpr(child);
      }
      std::cout << ")\n";
    }
  } else {
    if (reader.root.children.empty()) {
      std::cout << "(empty)\n";
    } else if (reader.root.children.size() == 1) {
      print_node(reader.root.children[0]);
      std::cout << "\n";
    } else {
      print_node(reader.root);
      std::cout << "\n";
    }
  }
}

void run_interactive_repl(bool sexpr_mode) {
  std::cout << "Reader REPL - Enter expressions, press Enter twice to execute "
               "(Ctrl+D to exit):\n";
  std::cout << ">>> ";

  std::string line;
  std::string input;

  while (std::getline(std::cin, line)) {
    if (line.empty()) {
      if (input.empty()) {
        std::cout << ">>> ";
        continue;
      } else {
        try {
          Reader reader(input);
          reader.read();
          print_parsed_result(reader, sexpr_mode);
        } catch (const std::exception &e) {
          std::cout << "Error: " << e.what() << "\n";
        }

        input.clear();
        std::cout << ">>> ";
      }
    } else {
      if (!input.empty()) {
        input += "\n";
      }
      input += line;
      std::cout << "... ";
    }
  }

  std::cout << "\nGoodbye!\n";
}

void run_batch_mode(bool sexpr_mode) {
  std::string line;
  std::string input;

  while (std::getline(std::cin, line)) {
    if (!input.empty()) {
      input += "\n";
    }
    input += line;
  }

  if (!input.empty()) {
    try {
      Reader reader(input);
      reader.read();
      print_parsed_result(reader, sexpr_mode);
    } catch (const std::exception &e) {
      std::cout << "Error: " << e.what() << "\n";
    }
  }
}

void run_reader_repl(bool sexpr_mode) {
  bool is_interactive = isatty(fileno(stdin));

  if (is_interactive) {
    run_interactive_repl(sexpr_mode);
  } else {
    run_batch_mode(sexpr_mode);
  }
}

int main(int argc, char *argv[]) {
  bool sexpr_mode = false;

  // Check for --sexpr flag
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--sexpr") == 0) {
      sexpr_mode = true;
      break;
    }
  }

  run_reader_repl(sexpr_mode);
  return 0;
}