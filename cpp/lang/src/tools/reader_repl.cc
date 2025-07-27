#include "../lang.h"
#include <iostream>
#include <string>
std::string indent(int level) {
    return std::string(level * 2, ' ');
}

std::string node_type_to_string(ReaderNodeType type) {
    switch (type) {
        case ReaderNodeType::Ident: return "ident";
        case ReaderNodeType::Literal: return "literal";
        case ReaderNodeType::List: return "list";
        case ReaderNodeType::Block: return "block";
        case ReaderNodeType::BinaryOp: return "binary-op";
        case ReaderNodeType::PrefixOp: return "prefix-op";
        case ReaderNodeType::PostfixOp: return "postfix-op";
        case ReaderNodeType::Call: return "call";
        default: return "unknown";
    }
}

void print_node(const ReaderNode& node, int level = 0) {
    std::cout << indent(level) << "(" << node_type_to_string(node.type);
    
    if (node.type != ReaderNodeType::List && !node.value().empty()) {
        std::cout << " \"" << node.value() << "\"";
        std::cout << " [" << node.token.line << ":" << node.token.column << "]";
    }
    
    if (!node.children.empty()) {
        std::cout << "\n";
        for (const auto& child : node.children) {
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

void run_reader_repl() {
    std::cout << "Reader REPL - Enter expressions, press Enter twice to execute (Ctrl+D to exit):\n";
    std::cout << ">>> ";
    
    std::string line;
    std::string input;
    
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            if (input.empty()) {
                // First empty line, just show prompt again
                std::cout << ">>> ";
                continue;
            } else {
                // Second empty line, execute the accumulated input
                try {
                    Reader reader(input);
                    reader.read();
                    
                    if (reader.root.children.empty()) {
                        std::cout << "(empty)\n";
                    } else if (reader.root.children.size() == 1) {
                        // For single expressions, print without the outer list wrapper
                        print_node(reader.root.children[0]);
                        std::cout << "\n";
                    } else {
                        // For multiple expressions, show the list
                        print_node(reader.root);
                        std::cout << "\n";
                    }
                } catch (const std::exception& e) {
                    std::cout << "Error: " << e.what() << "\n";
                }
                
                // Reset for next input
                input.clear();
                std::cout << ">>> ";
            }
        } else {
            // Non-empty line, add to input buffer
            if (!input.empty()) {
                input += "\n";
            }
            input += line;
            std::cout << "... ";
        }
    }
    
    std::cout << "\nGoodbye!\n";
}