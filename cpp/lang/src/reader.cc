#include "reader.h"
#include <stdexcept>

Token Reader::advance() {
  do {
    current_token = tokenizer.next_token(input);
  } while (current_token.type == TokenType::Whitespace &&
           current_token.type != TokenType::End);

  return current_token;
}

Token Reader::peek() const { return current_token; }

void Reader::read() {
  advance();

  while (current_token.type != TokenType::End) {
    root.children.push_back(parse_expression());
  }
}

Reader::OperatorInfo Reader::get_operator_info(const Token &token, bool isPostfix) {
  if (token.type == TokenType::Operator) {
    if (isPostfix && token.value == "!")
      return {40, LEFT};
    if (!isPostfix && token.value == "!")
      return {0, RIGHT};
    if (token.value == "||")
      return {2, LEFT};
    if (token.value == "==" || token.value == "!=" || token.value == "<" ||
        token.value == ">" || token.value == "<=" || token.value == ">=")
      return {5, LEFT};
    if (token.value == "+" || token.value == "-")
      return {10, LEFT};
    if (token.value == "->")
      return {8, RIGHT};  // Higher precedence than : and right-associative
    if (token.value == "=>")
      return {1, RIGHT};  // Low precedence, right-associative
    if (token.value == "*" || token.value == "/")
      return {20, LEFT};
    if (token.value == "^")
      return {30, RIGHT};  // Exponentiation is typically right-associative
    if (token.value == ".")
      return {35, LEFT};
    return {2, LEFT};
  }
  if (token.type == TokenType::Separator) {
    if (token.value == ":")
      return {3, RIGHT};  // Lower precedence than ->, right-associative
    // Comma is not a binary operator - it's only a list separator
    return {0, LEFT};  // No precedence for other separators
  }
  if (isPostfix && token.type == TokenType::Delimiter && token.value == "(") {
    return {50, LEFT};  // Function calls have very high precedence
  }
  return {0, LEFT};
}

int Reader::get_binding_power(const Token &token, bool isPostfix) {
  return get_operator_info(token, isPostfix).precedence;
}

ReaderNode Reader::parse_expression(int rightBindingPower) {
  Token token = peek();
  advance(); // Move past the current token
  ReaderNode left = parse_prefix(token);

  while (rightBindingPower < get_binding_power(peek())) {
    token = current_token;

    Tokenizer saved_tokenizer = tokenizer;
    advance(); // Move past the operator temporarily
    Token next_token = peek();

    bool has_valid_right_operand =
        (next_token.type != TokenType::End) &&
        !(next_token.type == TokenType::Delimiter &&
          (next_token.value == "}" || next_token.value == ")" ||
           next_token.value == "]")) &&
        !(next_token.type == TokenType::Separator);

    if (!has_valid_right_operand) {
      tokenizer = saved_tokenizer;
      current_token = token;
      break;
    }

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

    // Parse statements with semicolon or comma separators
    while (current_token.type != TokenType::End && current_token.value != "}") {
      statements.push_back(parse_expression());
      
      // If we hit a semicolon or comma, consume it and continue
      if (current_token.value == ";" || current_token.value == ",") {
        advance(); // consume the separator
      }
      // If we don't hit a separator, we should still continue parsing
      // The last expression in a block doesn't need a separator
    }

    if (current_token.type == TokenType::End) {
      throw std::runtime_error("Unclosed block '{' starting at line " +
                               std::to_string(token.line) + ", column " +
                               std::to_string(token.column));
    }

    if (current_token.value == "}") {
      advance();
    }

    return ReaderNode(ReaderNodeType::Block, token, std::move(statements));
  }
  if (token.type == TokenType::Delimiter && token.value == "(") {
    std::vector<ReaderNode> elements;

    // Parse comma-separated expressions
    while (current_token.type != TokenType::End && current_token.value != ")") {
      elements.push_back(parse_expression());
      
      // If we hit a comma, consume it and continue
      if (current_token.value == ",") {
        advance(); // consume the comma
      } else if (current_token.value != ")") {
        // If it's not a comma and not a closing paren, we have an error
        break;
      }
    }

    if (current_token.type == TokenType::End) {
      throw std::runtime_error("Unclosed parenthesis '(' starting at line " +
                               std::to_string(token.line) + ", column " +
                               std::to_string(token.column));
    }

    if (current_token.value == ")") {
      advance();
    }

    return ReaderNode(ReaderNodeType::List, token, std::move(elements));
  }
  if (token.type == TokenType::Delimiter && token.value == "[") {
    std::vector<ReaderNode> elements;

    while (current_token.type != TokenType::End && current_token.value != "]") {
      elements.push_back(parse_expression());
    }

    if (current_token.type == TokenType::End) {
      throw std::runtime_error("Unclosed bracket '[' starting at line " +
                               std::to_string(token.line) + ", column " +
                               std::to_string(token.column));
    }

    if (current_token.value == "]") {
      advance();
    }

    return ReaderNode(ReaderNodeType::List, token, std::move(elements));
  }
  if (token.type == TokenType::Operator) {
    if (token.value == "-") { // unary minus
      Token next = peek();
      if (next.type == TokenType::End ||
          (next.type == TokenType::Delimiter &&
           (next.value == "}" || next.value == ")" || next.value == "]"))) {
        return ReaderNode(ReaderNodeType::Ident, token);
      } else {
        return ReaderNode(ReaderNodeType::PrefixOp, token,
                          {parse_expression(100)});
      }
    } else {
      return ReaderNode(ReaderNodeType::Ident, token);
    }
  }
  if (token.type == TokenType::Separator) {
    return ReaderNode(ReaderNodeType::Ident, token);
  }
  throw std::runtime_error(
      "Unexpected token in parsePrefix: " + std::string(token.value) +
      " (type: " + std::to_string(static_cast<int>(token.type)) + ")");
}

ReaderNode Reader::parse_infix(ReaderNode left, const Token &token) {
  OperatorInfo info = get_operator_info(token);
  int nextBp = (info.associativity == RIGHT) ? info.precedence : info.precedence + 1;
  ReaderNode right = parse_expression(nextBp);
  return ReaderNode(ReaderNodeType::BinaryOp, token,
                    {std::move(left), std::move(right)});
}

ReaderNode Reader::parse_postfix(ReaderNode left, const Token &token) {
  if (token.type == TokenType::Delimiter && token.value == "(") {
    // This is a function call
    std::vector<ReaderNode> arguments;
    
    // Parse comma-separated arguments
    while (current_token.type != TokenType::End && current_token.value != ")") {
      arguments.push_back(parse_expression());
      
      // If we hit a comma, consume it and continue
      if (current_token.value == ",") {
        advance(); // consume the comma
      } else if (current_token.value != ")") {
        // If it's not a comma and not a closing paren, we have an error
        break;
      }
    }
    
    if (current_token.value == ")") {
      advance(); // consume the closing parenthesis
    }
    
    // Create a Call node with the function as first child, then arguments
    ReaderNode call_node(ReaderNodeType::Call, token);
    call_node.children.push_back(std::move(left));
    for (auto& arg : arguments) {
      call_node.children.push_back(std::move(arg));
    }
    
    return call_node;
  }
  
  return ReaderNode(ReaderNodeType::PostfixOp, token, {std::move(left)});
}

ReaderNode Reader::parse_block() {
  Token block_token = current_token; // Save the "{" token info
  std::vector<ReaderNode> statements;

  advance();

  while (current_token.type != TokenType::End && current_token.value != "}") {
    statements.push_back(parse_expression());
  }

  if (current_token.value == "}") {
    advance();
  }

  return ReaderNode(ReaderNodeType::Block, block_token, std::move(statements));
}