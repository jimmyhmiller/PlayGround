#include <cassert>
#include <iostream>
#include <string_view>
#include "lang.h"


// Method implementations for Tokenizer
Token Tokenizer::handle_whitespace(const std::string_view input) {
    int start = pos;
    int start_line = line;
    int start_column = column;
    while (!at_end(input) && is_whitespace(input[pos])) {
      consume(input);
    }
    return Token{TokenType::Whitespace, input.substr(start, pos - start), start_line, start_column};
  }

Token Tokenizer::next_token(const std::string_view input) {
    while (!at_end(input)) {
      char current_char = input[pos];
      if (is_whitespace(current_char)) {
        return handle_whitespace(input);
      } else if (std::isalpha(current_char) || current_char == '_') {
        int start = pos;
        int start_line = line;
        int start_column = column;
        while (!at_end(input) && (std::isalnum(input[pos]) || input[pos] == '_' || input[pos] == '!')) {
          consume(input);
        }
        return Token{TokenType::Identifier, input.substr(start, pos - start), start_line, start_column};
      } else if (std::isdigit(current_char)) {
        int start = pos;
        int start_line = line;
        int start_column = column;
        while (!at_end(input) && std::isdigit(input[pos])) {
          consume(input);
        }
        // Check for decimal point followed by digits
        if (!at_end(input) && input[pos] == '.' && pos + 1 < input.size() && std::isdigit(input[pos + 1])) {
          consume(input); // consume the '.'
          while (!at_end(input) && std::isdigit(input[pos])) {
            consume(input);
          }
        }
        return Token{TokenType::NUMBER, input.substr(start, pos - start), start_line, start_column};
      } else if (current_char == '"') {
        int start = pos;
        int start_line = line;
        int start_column = column;
        consume(input); // Skip opening quote
        std::string str_value;
        while (!at_end(input) && input[pos] != '"') {
          if (input[pos] == '\\' && pos + 1 < input.size()) {
            // Handle escape sequences
            consume(input); // consume the backslash
            if (!at_end(input)) {
              char escape_char = input[pos];
              switch (escape_char) {
                case 'n': str_value += '\n'; break;
                case 't': str_value += '\t'; break;
                case 'r': str_value += '\r'; break;
                case '\\': str_value += '\\'; break;
                case '"': str_value += '"'; break;
                default: 
                  // For unknown escape sequences, include both backslash and character
                  str_value += '\\';
                  str_value += escape_char;
                  break;
              }
              consume(input);
            }
          } else {
            str_value += input[pos];
            consume(input);
          }
        }
        if (!at_end(input)) consume(input); // Skip the closing quote
        
        // For strings without escape sequences, we can return the original substring
        bool has_escapes = false;
        for (char c : str_value) {
            if (c == '\n' || c == '\t' || c == '\r' || c == '\\' || c == '"') {
                has_escapes = true;
                break;
            }
        }
        
        if (!has_escapes && str_value == input.substr(start + 1, pos - start - 2)) {
            // No escape sequences, return original substring
            return Token{TokenType::String, input.substr(start + 1, pos - start - 2), start_line, start_column};
        } else {
            // Has escape sequences, need to store the processed string
            // Note: This creates a potential memory management issue that should be addressed
            // in a production implementation. For now, we'll use a static string pool.
            static std::vector<std::string> string_pool;
            string_pool.reserve(1000); // Reserve space to prevent reallocation
            string_pool.push_back(std::move(str_value));
            return Token{TokenType::String, std::string_view(string_pool.back()), start_line, start_column};
        }
      } else if (is_delimiter(current_char)) {
        int start_line = line;
        int start_column = column;
        Token token{TokenType::Delimiter, std::string_view(&input[pos], 1), start_line, start_column};
        consume(input);
        return token;
      } else if (is_separator(current_char)) {
        int start_line = line;
        int start_column = column;
        Token token{TokenType::Separator, std::string_view(&input[pos], 1), start_line, start_column};
        consume(input);
        return token;
      } else if (is_operator(current_char)) {
        int start = pos;
        int start_line = line;
        int start_column = column;
        
        // Handle multi-character operators
        if (current_char == '=' && pos + 1 < input.size() && input[pos + 1] == '=') {
          consume(input, 2); // consume "=="
          return Token{TokenType::Operator, input.substr(start, 2), start_line, start_column};
        } else if (current_char == '!' && pos + 1 < input.size() && input[pos + 1] == '=') {
          consume(input, 2); // consume "!="
          return Token{TokenType::Operator, input.substr(start, 2), start_line, start_column};
        } else if (current_char == '<' && pos + 1 < input.size() && input[pos + 1] == '=') {
          consume(input, 2); // consume "<="
          return Token{TokenType::Operator, input.substr(start, 2), start_line, start_column};
        } else if (current_char == '>' && pos + 1 < input.size() && input[pos + 1] == '=') {
          consume(input, 2); // consume ">="
          return Token{TokenType::Operator, input.substr(start, 2), start_line, start_column};
        } else {
          // Single character operator
          Token token{TokenType::Operator, std::string_view(&input[pos], 1), start_line, start_column};
          consume(input);
          return token;
        }
      } else {
        assert(false && "Unexpected character in input");
      }
    }
    return Token{TokenType::End, "", line, column};
  }




// Method implementations for Reader
Token Reader::advance() {
        do {
            current_token = tokenizer.next_token(input);
        } while (current_token.type == TokenType::Whitespace && current_token.type != TokenType::End);
        
        return current_token;
    }

Token Reader::peek() const {
        return current_token;
    }

void Reader::read() {
        // Initialize by getting the first token
        advance();
        
        while (current_token.type != TokenType::End) {
            root.add_child(parse_expression());
        }
    }

int Reader::get_binding_power(const Token& token, bool isPostfix) {
        if (token.type == TokenType::Operator) {
            if (isPostfix && token.value == "!") return 40;
            if (token.value == "==" || token.value == "!=" || token.value == "<" || token.value == ">" || token.value == "<=" || token.value == ">=") return 5;
            if (token.value == "+" || token.value == "-") return 10;
            if (token.value == "*" || token.value == "/") return 20;
            if (token.value == "^") return 30;
        }
        return 0;
    }

ReaderNode Reader::parse_expression(int rightBindingPower) {
        Token token = peek();
        advance(); // Move past the current token
        ReaderNode left = parse_prefix(token);

        while (rightBindingPower < get_binding_power(peek())) {
            token = current_token;
            advance(); // Move past the operator
            left = parse_infix(std::move(left), token);
        }

        while (rightBindingPower < get_binding_power(peek(), true)) {
            token = current_token;
            advance(); // Move past the postfix operator
            left = parse_postfix(std::move(left), token);
        }

        return left;
    }

ReaderNode Reader::parse_prefix(const Token& token) {
        if (token.type == TokenType::End) {
            throw std::runtime_error("Unexpected end of input in parsePrefix");
        }
        if (token.type == TokenType::NUMBER) {
            return ReaderNode(ReaderNodeType::Literal, token);
        }
        if (token.type == TokenType::String) {
            return ReaderNode(ReaderNodeType::Literal, token);
        }
        if (token.type == TokenType::Identifier) {
            return ReaderNode(ReaderNodeType::Ident, token);
        }
        if (token.type == TokenType::Delimiter && token.value == "{") {
            std::vector<ReaderNode> statements;
            
            // Parse statements until we hit "}"
            while (current_token.type != TokenType::End && current_token.value != "}") {
                statements.push_back(parse_expression());
            }
            
            // Check for proper closing
            if (current_token.type == TokenType::End) {
                throw std::runtime_error("Unclosed block '{' starting at line " + std::to_string(token.line) + ", column " + std::to_string(token.column));
            }
            
            // Consume the closing "}"
            if (current_token.value == "}") {
                advance();
            }
            
            return ReaderNode(ReaderNodeType::Block, token, std::move(statements));
        }
        if (token.type == TokenType::Delimiter && token.value == "(") {
            std::vector<ReaderNode> elements;
            
            // Parse elements until we hit ")"
            while (current_token.type != TokenType::End && current_token.value != ")") {
                elements.push_back(parse_expression());
            }
            
            // Check for proper closing
            if (current_token.type == TokenType::End) {
                throw std::runtime_error("Unclosed parenthesis '(' starting at line " + std::to_string(token.line) + ", column " + std::to_string(token.column));
            }
            
            // Consume the closing ")"
            if (current_token.value == ")") {
                advance();
            }
            
            return ReaderNode(ReaderNodeType::List, token, std::move(elements));
        }
        if (token.type == TokenType::Delimiter && token.value == "[") {
            std::vector<ReaderNode> elements;
            
            // Parse elements until we hit "]"
            while (current_token.type != TokenType::End && current_token.value != "]") {
                elements.push_back(parse_expression());
            }
            
            // Check for proper closing
            if (current_token.type == TokenType::End) {
                throw std::runtime_error("Unclosed bracket '[' starting at line " + std::to_string(token.line) + ", column " + std::to_string(token.column));
            }
            
            // Consume the closing "]"
            if (current_token.value == "]") {
                advance();
            }
            
            return ReaderNode(ReaderNodeType::List, token, std::move(elements));
        }
        if (token.type == TokenType::Operator) {
            if (token.value == "-") { // unary minus
                return ReaderNode(ReaderNodeType::PrefixOp, token, {parse_expression(100)});
            } else {
                // Treat other operators as identifiers when in prefix position
                return ReaderNode(ReaderNodeType::Ident, token);
            }
        }
        throw std::runtime_error("Unexpected token in parsePrefix: " + std::string(token.value) + " (type: " + std::to_string(static_cast<int>(token.type)) + ")");
    }

ReaderNode Reader::parse_infix(ReaderNode left, const Token& token) {
        int bp = get_binding_power(token);
        bool rightAssoc = (token.value == "^");
        int nextBp = rightAssoc ? bp - 1 : bp;
        ReaderNode right = parse_expression(nextBp);
        return ReaderNode(ReaderNodeType::BinaryOp, token, {std::move(left), std::move(right)});
    }

ReaderNode Reader::parse_postfix(ReaderNode left, const Token& token) {
        return ReaderNode(ReaderNodeType::PostfixOp, token, {std::move(left)});
    }

ReaderNode Reader::parse_block() {
        Token block_token = current_token; // Save the "{" token info
        std::vector<ReaderNode> statements;
        
        // We're called when current_token is "{", so advance past it
        advance();
        
        // Parse statements until we hit "}"
        while (current_token.type != TokenType::End && current_token.value != "}") {
            statements.push_back(parse_expression());
        }
        
        // Consume the closing "}"
        if (current_token.value == "}") {
            advance();
        }
        
        return ReaderNode(ReaderNodeType::Block, block_token, std::move(statements));
    }

// Include tools
#include "tools/reader_repl.cc"

void show_help(const std::string& program_name) {
    std::cout << "Usage: " << program_name << " [tool] [args...]\n\n";
    std::cout << "Available tools:\n";
    std::cout << "  reader-repl    Interactive REPL for the reader/parser\n";
    std::cout << "  --help         Show this help message\n\n";
    std::cout << "If no tool is specified, runs default behavior.\n";
}

#ifndef CABIN_TEST
int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string tool = argv[1];
        
        if (tool == "--help" || tool == "-h") {
            show_help(argv[0]);
            return 0;
        } else if (tool == "reader-repl") {
            run_reader_repl();
            return 0;
        } else {
            std::cerr << "Unknown tool: " << tool << "\n";
            std::cerr << "Run '" << argv[0] << " --help' for available tools.\n";
            return 1;
        }
    }
    
    // Default behavior
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
#endif

#ifdef CABIN_TEST
#include <cassert>
#include <vector>

// Include the reader tests
#include "../tests/test_reader.cc"

// Helper function to tokenize entire input
std::vector<Token> tokenize_all(const std::string_view input) {
  Tokenizer tokenizer;
  std::vector<Token> tokens;
  
  while (!tokenizer.at_end(input)) {
    tokens.push_back(tokenizer.next_token(input));
  }
  
  return tokens;
}

// Test helper to check token properties
void assert_token(const Token& token, TokenType expected_type, 
                  const std::string_view expected_value,
                  int expected_line, int expected_column) {
  assert(token.type == expected_type);
  assert(token.value == expected_value);
  assert(token.line == expected_line);
  assert(token.column == expected_column);
}

int main() {
  // Test 1: Basic whitespace tokenization
  {
    std::string input = "   \t\n  ";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 1);
    assert_token(tokens[0], TokenType::Whitespace, "   \t\n  ", 1, 0);
  }

  // Test 2: Identifier tokenization
  {
    std::string input = "hello world_123 _test";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 5);
    assert_token(tokens[0], TokenType::Identifier, "hello", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 5);
    assert_token(tokens[2], TokenType::Identifier, "world_123", 1, 6);
    assert_token(tokens[3], TokenType::Whitespace, " ", 1, 15);
    assert_token(tokens[4], TokenType::Identifier, "_test", 1, 16);
  }

  // Test 3: Number tokenization
  {
    std::string input = "123 456 789";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 5);
    assert_token(tokens[0], TokenType::NUMBER, "123", 1, 0);
    assert_token(tokens[2], TokenType::NUMBER, "456", 1, 4);
    assert_token(tokens[4], TokenType::NUMBER, "789", 1, 8);
  }

  // Test 4: String tokenization
  {
    std::string input = "\"hello\" \"world with spaces\"";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 3);
    assert_token(tokens[0], TokenType::String, "hello", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 7);
    assert_token(tokens[2], TokenType::String, "world with spaces", 1, 8);
  }

  // Test 5: Delimiter and Separator tokenization
  {
    std::string input = "();,{}[]";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 8);
    // Check delimiters
    assert_token(tokens[0], TokenType::Delimiter, "(", 1, 0);
    assert_token(tokens[1], TokenType::Delimiter, ")", 1, 1);
    assert_token(tokens[2], TokenType::Separator, ";", 1, 2);
    assert_token(tokens[3], TokenType::Separator, ",", 1, 3);
    assert_token(tokens[4], TokenType::Delimiter, "{", 1, 4);
    assert_token(tokens[5], TokenType::Delimiter, "}", 1, 5);
    assert_token(tokens[6], TokenType::Delimiter, "[", 1, 6);
    assert_token(tokens[7], TokenType::Delimiter, "]", 1, 7);
  }

  // Test 6: Line and column tracking
  {
    std::string input = "hello\nworld\n123";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 5);
    assert_token(tokens[0], TokenType::Identifier, "hello", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, "\n", 1, 5);
    assert_token(tokens[2], TokenType::Identifier, "world", 2, 0);
    assert_token(tokens[3], TokenType::Whitespace, "\n", 2, 5);
    assert_token(tokens[4], TokenType::NUMBER, "123", 3, 0);
  }

  // Test 7: Complex multiline example
  {
    std::string input = "func add(x, y) {\n  return x + y\n}";
    auto tokens = tokenize_all(input);
    
    // Check first line tokens
    assert_token(tokens[0], TokenType::Identifier, "func", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 4);
    assert_token(tokens[2], TokenType::Identifier, "add", 1, 5);
    assert_token(tokens[3], TokenType::Delimiter, "(", 1, 8);
    
    // Find the return token and check its position
    for (size_t i = 0; i < tokens.size(); i++) {
      if (tokens[i].value == "return") {
        assert_token(tokens[i], TokenType::Identifier, "return", 2, 2);
        break;
      }
    }
  }

  // Test 8: Empty string handling
  {
    std::string input = "\"\"";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 1);
    assert_token(tokens[0], TokenType::String, "", 1, 0);
  }

  // Test 9: Mixed content
  {
    std::string input = "x = 42 + \"test\"";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 9);
    assert_token(tokens[0], TokenType::Identifier, "x", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 1);
    assert_token(tokens[2], TokenType::Operator, "=", 1, 2);
    assert_token(tokens[3], TokenType::Whitespace, " ", 1, 3);
    assert_token(tokens[4], TokenType::NUMBER, "42", 1, 4);
    assert_token(tokens[5], TokenType::Whitespace, " ", 1, 6);
    assert_token(tokens[6], TokenType::Operator, "+", 1, 7);
    assert_token(tokens[7], TokenType::Whitespace, " ", 1, 8);
    assert_token(tokens[8], TokenType::String, "test", 1, 9);
  }

  // Test 10: Float parsing
  {
    std::string input = "3.14 42.0 0.5";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 5);
    assert_token(tokens[0], TokenType::NUMBER, "3.14", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 4);
    assert_token(tokens[2], TokenType::NUMBER, "42.0", 1, 5);
    assert_token(tokens[3], TokenType::Whitespace, " ", 1, 9);
    assert_token(tokens[4], TokenType::NUMBER, "0.5", 1, 10);
  }

  // Test 11: Float in expression
  {
    std::string input = "pi = 3.14159";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 5);
    assert_token(tokens[0], TokenType::Identifier, "pi", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 2);
    assert_token(tokens[2], TokenType::Operator, "=", 1, 3);
    assert_token(tokens[3], TokenType::Whitespace, " ", 1, 4);
    assert_token(tokens[4], TokenType::NUMBER, "3.14159", 1, 5);
  }

  // Test 12: Dot operator vs float
  {
    std::string input = "obj.prop 3.14 x.";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 8);
    assert_token(tokens[0], TokenType::Identifier, "obj", 1, 0);
    assert_token(tokens[1], TokenType::Operator, ".", 1, 3);
    assert_token(tokens[2], TokenType::Identifier, "prop", 1, 4);
    assert_token(tokens[3], TokenType::Whitespace, " ", 1, 8);
    assert_token(tokens[4], TokenType::NUMBER, "3.14", 1, 9);
    assert_token(tokens[5], TokenType::Whitespace, " ", 1, 13);
    assert_token(tokens[6], TokenType::Identifier, "x", 1, 14);
    assert_token(tokens[7], TokenType::Operator, ".", 1, 15);
  }

  // Test 13: Escape sequences in strings
  {
    std::string input = "\"hello\\nworld\"";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 1);
    assert(tokens[0].type == TokenType::String);
    assert(tokens[0].value == "hello\nworld");
    assert(tokens[0].line == 1);
    assert(tokens[0].column == 0);
  }

  // Test 14: Various escape sequences
  {
    std::string input = "\"tab:\\tthere\" \"quote:\\\"here\\\"\" \"backslash:\\\\\"";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 5);
    assert_token(tokens[0], TokenType::String, "tab:\tthere", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 13);
    assert_token(tokens[2], TokenType::String, "quote:\"here\"", 1, 14);
    assert_token(tokens[3], TokenType::Whitespace, " ", 1, 30);
    assert_token(tokens[4], TokenType::String, "backslash:\\", 1, 31);
  }

  // Test 15: Unknown escape sequences
  {
    std::string input = "\"unknown:\\x escape\"";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 1);
    assert(tokens[0].type == TokenType::String);
    assert(tokens[0].value == "unknown:\\x escape");
  }

  // Test 16: String with carriage return
  {
    std::string input = "\"line1\\rline2\"";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 1);
    assert(tokens[0].type == TokenType::String);
    assert(tokens[0].value == "line1\rline2");
  }

  // Test 17: Identifiers with ! suffix
  {
    std::string input = "foo! bar! test_case!";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 5);
    assert_token(tokens[0], TokenType::Identifier, "foo!", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 4);
    assert_token(tokens[2], TokenType::Identifier, "bar!", 1, 5);
    assert_token(tokens[3], TokenType::Whitespace, " ", 1, 9);
    assert_token(tokens[4], TokenType::Identifier, "test_case!", 1, 10);
  }

  // Test 18: ! as prefix operator vs identifier suffix
  {
    std::string input = "!x x! !y!";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 7);
    assert_token(tokens[0], TokenType::Operator, "!", 1, 0);
    assert_token(tokens[1], TokenType::Identifier, "x", 1, 1);
    assert_token(tokens[2], TokenType::Whitespace, " ", 1, 2);
    assert_token(tokens[3], TokenType::Identifier, "x!", 1, 3);
    assert_token(tokens[4], TokenType::Whitespace, " ", 1, 5);
    assert_token(tokens[5], TokenType::Operator, "!", 1, 6);
    assert_token(tokens[6], TokenType::Identifier, "y!", 1, 7);
  }

  // Test 19: Multiple ! characters in identifier
  {
    std::string input = "test!! foo!!!";
    auto tokens = tokenize_all(input);
    assert(tokens.size() == 3);
    assert_token(tokens[0], TokenType::Identifier, "test!!", 1, 0);
    assert_token(tokens[1], TokenType::Whitespace, " ", 1, 6);
    assert_token(tokens[2], TokenType::Identifier, "foo!!!", 1, 7);
  }

  std::cout << "All tokenizer tests passed!" << std::endl;
  
  // Run reader tests
  std::cout << "Running test_reader_simple_numbers..." << std::endl;
  test_reader_simple_numbers();
  std::cout << "Running test_reader_binary_operations..." << std::endl;
  test_reader_binary_operations();
  std::cout << "Running test_reader_operator_precedence..." << std::endl;
  test_reader_operator_precedence();
  std::cout << "Running test_reader_right_associative..." << std::endl;
  test_reader_right_associative();
  std::cout << "Running test_reader_unary_minus..." << std::endl;
  test_reader_unary_minus();
  std::cout << "Running test_reader_postfix_operator..." << std::endl;
  test_reader_postfix_operator();
  std::cout << "Running test_reader_complex_expression..." << std::endl;
  test_reader_complex_expression();
  std::cout << "Running test_reader_multiple_expressions..." << std::endl;
  test_reader_multiple_expressions();
  std::cout << "Running test_reader_node_equality..." << std::endl;
  test_reader_node_equality();
  std::cout << "Running test_reader_empty_input..." << std::endl;
  test_reader_empty_input();
  std::cout << "Running test_reader_whitespace_handling..." << std::endl;
  test_reader_whitespace_handling();
  
  std::cout << "All Reader tests passed!" << std::endl;
  
  return 0;
}
#endif
