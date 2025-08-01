#include "lang.h"
#include <cassert>
#include <string_view>

// Method implementations for Tokenizer
Token Tokenizer::handle_whitespace(const std::string_view input) {
  int start = pos;
  int start_line = line;
  int start_column = column;
  while (!at_end(input) && is_whitespace(input[pos])) {
    consume(input);
  }
  return Token{TokenType::Whitespace, input.substr(start, pos - start),
               start_line, start_column};
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
      while (!at_end(input) && (std::isalnum(input[pos]) || input[pos] == '_' ||
                                input[pos] == '!')) {
        consume(input);
      }
      return Token{TokenType::Identifier, input.substr(start, pos - start),
                   start_line, start_column};
    } else if (std::isdigit(current_char)) {
      int start = pos;
      int start_line = line;
      int start_column = column;
      while (!at_end(input) && std::isdigit(input[pos])) {
        consume(input);
      }
      // Check for decimal point followed by digits
      if (!at_end(input) && input[pos] == '.' &&
          static_cast<size_t>(pos + 1) < input.size() &&
          std::isdigit(input[pos + 1])) {
        consume(input); // consume the '.'
        while (!at_end(input) && std::isdigit(input[pos])) {
          consume(input);
        }
      }
      return Token{TokenType::NUMBER, input.substr(start, pos - start),
                   start_line, start_column};
    } else if (current_char == '"') {
      int start = pos;
      int start_line = line;
      int start_column = column;
      consume(input); // Skip opening quote
      std::string str_value;
      while (!at_end(input) && input[pos] != '"') {
        if (input[pos] == '\\' && static_cast<size_t>(pos + 1) < input.size()) {
          // Handle escape sequences
          consume(input); // consume the backslash
          if (!at_end(input)) {
            char escape_char = input[pos];
            switch (escape_char) {
            case 'n':
              str_value += '\n';
              break;
            case 't':
              str_value += '\t';
              break;
            case 'r':
              str_value += '\r';
              break;
            case '\\':
              str_value += '\\';
              break;
            case '"':
              str_value += '"';
              break;
            default:
              // For unknown escape sequences, include both backslash and
              // character
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
      if (!at_end(input))
        consume(input); // Skip the closing quote

      // For strings without escape sequences, we can return the original
      // substring
      bool has_escapes = false;
      for (char c : str_value) {
        if (c == '\n' || c == '\t' || c == '\r' || c == '\\' || c == '"') {
          has_escapes = true;
          break;
        }
      }

      if (!has_escapes &&
          str_value == input.substr(start + 1, pos - start - 2)) {
        // No escape sequences, return original substring
        return Token{TokenType::String,
                     input.substr(start + 1, pos - start - 2), start_line,
                     start_column};
      } else {
        // Has escape sequences, need to store the processed string
        // Note: This creates a potential memory management issue that should be
        // addressed in a production implementation. For now, we'll use a static
        // string pool.
        static std::vector<std::string> string_pool;
        string_pool.reserve(1000); // Reserve space to prevent reallocation
        string_pool.push_back(std::move(str_value));
        return Token{TokenType::String, std::string_view(string_pool.back()),
                     start_line, start_column};
      }
    } else if (is_delimiter(current_char)) {
      int start_line = line;
      int start_column = column;
      Token token{TokenType::Delimiter, std::string_view(&input[pos], 1),
                  start_line, start_column};
      consume(input);
      return token;
    } else if (is_separator(current_char)) {
      int start_line = line;
      int start_column = column;
      Token token{TokenType::Separator, std::string_view(&input[pos], 1),
                  start_line, start_column};
      consume(input);
      return token;
    } else if (is_operator(current_char)) {
      int start = pos;
      int start_line = line;
      int start_column = column;

      // Handle multi-character operators
      if (current_char == '=' && static_cast<size_t>(pos + 1) < input.size() &&
          input[pos + 1] == '=') {
        consume(input, 2); // consume "=="
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '!' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '=') {
        consume(input, 2); // consume "!="
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '<' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '=') {
        consume(input, 2); // consume "<="
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else if (current_char == '>' &&
                 static_cast<size_t>(pos + 1) < input.size() &&
                 input[pos + 1] == '=') {
        consume(input, 2); // consume ">="
        return Token{TokenType::Operator, input.substr(start, 2), start_line,
                     start_column};
      } else {
        // Single character operator
        Token token{TokenType::Operator, std::string_view(&input[pos], 1),
                    start_line, start_column};
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
  } while (current_token.type == TokenType::Whitespace &&
           current_token.type != TokenType::End);

  return current_token;
}

Token Reader::peek() const { return current_token; }

void Reader::read() {
  // Initialize by getting the first token
  advance();

  while (current_token.type != TokenType::End) {
    root.add_child(parse_expression());
  }
}

int Reader::get_binding_power(const Token &token, bool isPostfix) {
  if (token.type == TokenType::Operator) {
    if (isPostfix && token.value == "!")
      return 40;
    // For binary operations, ! should have no precedence (treated as identifier)
    if (!isPostfix && token.value == "!")
      return 0;
    if (token.value == "==" || token.value == "!=" || token.value == "<" ||
        token.value == ">" || token.value == "<=" || token.value == ">=")
      return 5;
    if (token.value == "+" || token.value == "-")
      return 10;
    if (token.value == "*" || token.value == "/")
      return 20;
    if (token.value == "^")
      return 30;
    // Other operators have lower precedence
    return 2;
  }
  if (token.type == TokenType::Separator) {
    // Separators have very low precedence (lower than comparison operators)
    return 1;
  }
  return 0;
}

ReaderNode Reader::parse_expression(int rightBindingPower) {
  Token token = peek();
  advance(); // Move past the current token
  ReaderNode left = parse_prefix(token);

  while (rightBindingPower < get_binding_power(peek())) {
    token = current_token;
    
    // Before consuming the operator, check if there's a valid right operand
    // Look ahead to see what comes after the operator
    Tokenizer saved_tokenizer = tokenizer;
    advance(); // Move past the operator temporarily
    Token next_token = peek();
    
    // Check if the next token can be a valid right operand
    bool has_valid_right_operand = (next_token.type != TokenType::End) &&
        !(next_token.type == TokenType::Delimiter && 
          (next_token.value == "}" || next_token.value == ")" || next_token.value == "]")) &&
        !(next_token.type == TokenType::Separator);
    
    if (!has_valid_right_operand) {
      // Restore the tokenizer state and break out of infix parsing
      tokenizer = saved_tokenizer;
      current_token = token;
      break;
    }
    
    // Valid right operand available, proceed with infix parsing
    left = parse_infix(std::move(left), token);
  }

  while (rightBindingPower < get_binding_power(peek(), true)) {
    token = current_token;
    advance(); // Move past the postfix operator
    left = parse_postfix(std::move(left), token);
  }

  return left;
}

ReaderNode Reader::parse_prefix(const Token &token) {
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
      throw std::runtime_error("Unclosed block '{' starting at line " +
                               std::to_string(token.line) + ", column " +
                               std::to_string(token.column));
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
      throw std::runtime_error("Unclosed parenthesis '(' starting at line " +
                               std::to_string(token.line) + ", column " +
                               std::to_string(token.column));
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
      throw std::runtime_error("Unclosed bracket '[' starting at line " +
                               std::to_string(token.line) + ", column " +
                               std::to_string(token.column));
    }

    // Consume the closing "]"
    if (current_token.value == "]") {
      advance();
    }

    return ReaderNode(ReaderNodeType::List, token, std::move(elements));
  }
  if (token.type == TokenType::Operator) {
    if (token.value == "-") { // unary minus
      // Check if there's a valid operand following
      Token next = peek();
      if (next.type == TokenType::End || 
          (next.type == TokenType::Delimiter && (next.value == "}" || next.value == ")" || next.value == "]"))) {
        // No operand available, treat as identifier
        return ReaderNode(ReaderNodeType::Ident, token);
      } else {
        // Valid operand available, treat as prefix operator
        return ReaderNode(ReaderNodeType::PrefixOp, token,
                          {parse_expression(100)});
      }
    } else {
      // Treat other operators as identifiers when in prefix position
      return ReaderNode(ReaderNodeType::Ident, token);
    }
  }
  if (token.type == TokenType::Separator) {
    // Treat separators as identifiers when in prefix position
    return ReaderNode(ReaderNodeType::Ident, token);
  }
  throw std::runtime_error(
      "Unexpected token in parsePrefix: " + std::string(token.value) +
      " (type: " + std::to_string(static_cast<int>(token.type)) + ")");
}

ReaderNode Reader::parse_infix(ReaderNode left, const Token &token) {
  int bp = get_binding_power(token);
  bool rightAssoc = (token.value == "^");
  int nextBp = rightAssoc ? bp - 1 : bp;
  ReaderNode right = parse_expression(nextBp);
  return ReaderNode(ReaderNodeType::BinaryOp, token,
                    {std::move(left), std::move(right)});
}

ReaderNode Reader::parse_postfix(ReaderNode left, const Token &token) {
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