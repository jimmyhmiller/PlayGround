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

int Reader::get_binding_power(const Token &token, bool isPostfix) {
  if (token.type == TokenType::Operator) {
    if (isPostfix && token.value == "!")
      return 40;
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
    if (token.value == ".")
      return 35;
    return 2;
  }
  if (token.type == TokenType::Separator) {
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

    while (current_token.type != TokenType::End && current_token.value != "}") {
      statements.push_back(parse_expression());
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

    while (current_token.type != TokenType::End && current_token.value != ")") {
      elements.push_back(parse_expression());
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

  advance();

  while (current_token.type != TokenType::End && current_token.value != "}") {
    statements.push_back(parse_expression());
  }

  if (current_token.value == "}") {
    advance();
  }

  return ReaderNode(ReaderNodeType::Block, block_token, std::move(statements));
}