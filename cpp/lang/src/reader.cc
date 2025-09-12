#include "reader.h"
#include <cctype>
#include <stdexcept>

Token Reader::advance() {
  do {
    current_token = tokenizer.next_token(input);
  } while ((current_token.type == TokenType::Whitespace ||
            current_token.type == TokenType::Comment) &&
           current_token.type != TokenType::End);

  return current_token;
}

Token Reader::peek() const { return current_token; }

void Reader::read() {
  advance();

  std::vector<ReaderNode> statements;
  bool found_semicolon = false;

  while (current_token.type != TokenType::End) {
    statements.push_back(parse_expression());

    // If we encounter a semicolon, consume it and continue parsing statements
    if (current_token.type == TokenType::Separator &&
        current_token.value == ";") {
      found_semicolon = true;
      advance(); // consume the semicolon
    }
  }

  // Only wrap in 'do' block if we found semicolons
  if (found_semicolon && statements.size() > 1) {
    Token do_token{TokenType::Identifier, "do", 1, 0}; // synthetic token
    ReaderNode do_node(ReaderNodeType::Ident, do_token);
    for (auto &stmt : statements) {
      do_node.children.push_back(std::move(stmt));
    }
    root.children.push_back(std::move(do_node));
  } else {
    // For non-semicolon separated statements, add them directly (old behavior)
    for (auto &stmt : statements) {
      root.children.push_back(std::move(stmt));
    }
  }
}

Reader::OperatorInfo Reader::get_operator_info(const Token &token,
                                               bool isPostfix) {
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
      return {8, RIGHT}; // Higher precedence than : and right-associative
    if (token.value == "=>")
      return {1, RIGHT}; // Low precedence, right-associative
    if (token.value == "*" || token.value == "/")
      return {20, LEFT};
    if (token.value == "^")
      return {30, RIGHT}; // Exponentiation is typically right-associative
    if (token.value == ".")
      return {35, LEFT};
    return {2, LEFT};
  }
  if (token.type == TokenType::Separator) {
    if (token.value == ":")
      return {3, RIGHT}; // Lower precedence than ->, right-associative
    // Comma is not a binary operator - it's only a list separator
    return {0, LEFT}; // No precedence for other separators
  }
  if (isPostfix && token.type == TokenType::Delimiter && token.value == "(") {
    return {50, LEFT}; // Function calls have very high precedence
  }
  if (isPostfix && token.type == TokenType::Delimiter && token.value == "{") {
    // Give very low binding power for struct literals so they only attach in
    // limited contexts
    return {0, LEFT}; // Lowest precedence
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
  if (token.type == TokenType::INTEGER || token.type == TokenType::FLOAT) {
    return ReaderNode(ReaderNodeType::Literal, token);
  }
  if (token.type == TokenType::String) {
    return ReaderNode(ReaderNodeType::Literal, token);
  }
  if (token.type == TokenType::Boolean) {
    return ReaderNode(ReaderNodeType::Literal, token);
  }
  if (token.type == TokenType::Identifier) {
    // Special handling for 'if' as prefix operator
    if (token.value == "if") {
      // Parse if condition then-block [else else-block]
      std::vector<ReaderNode> if_parts;

      // Parse condition with binding power higher than blocks but lower than
      // comparisons
      ReaderNode condition = parse_expression(
          2); // Higher than block postfix (1) but lower than == (5)
      if_parts.push_back(std::move(condition));

      // Parse then-block (expecting a block)
      if (current_token.type == TokenType::Delimiter &&
          current_token.value == "{") {
        ReaderNode then_block = parse_expression();
        if_parts.push_back(std::move(then_block));
      }

      // Check for optional else clause
      if (current_token.type == TokenType::Identifier &&
          current_token.value == "else") {
        advance(); // consume 'else'

        // Parse else-block
        if (current_token.type == TokenType::Delimiter &&
            current_token.value == "{") {
          ReaderNode else_block = parse_expression();
          if_parts.push_back(std::move(else_block));
        }
      }

      ReaderNode if_node(ReaderNodeType::PrefixOp, token);
      for (auto &part : if_parts) {
        if_node.children.push_back(std::move(part));
      }

      return if_node;
    }

    return ReaderNode(ReaderNodeType::Ident, token);
  }
  if (token.type == TokenType::Separator) {
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

    // Parse space or comma-separated expressions
    while (current_token.type != TokenType::End && current_token.value != ")") {
      elements.push_back(parse_expression());

      // If we hit a comma, consume it and continue
      if (current_token.value == ",") {
        advance(); // consume the comma
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
  throw std::runtime_error(
      "Unexpected token in parsePrefix: " + std::string(token.value) +
      " (type: " + std::to_string(static_cast<int>(token.type)) + ")");
}

ReaderNode Reader::parse_infix(ReaderNode left, const Token &token) {
  OperatorInfo info = get_operator_info(token);
  int nextBp =
      (info.associativity == RIGHT) ? info.precedence - 1 : info.precedence + 1;
  ReaderNode right = parse_expression(nextBp);
  return ReaderNode(ReaderNodeType::BinaryOp, token,
                    {std::move(left), std::move(right)});
}

ReaderNode Reader::parse_postfix(ReaderNode left, const Token &token) {
  if (token.type == TokenType::Delimiter && token.value == "(") {
    // This is a function call
    std::vector<ReaderNode> arguments;

    // Parse space or comma-separated arguments
    while (current_token.type != TokenType::End && current_token.value != ")") {
      arguments.push_back(parse_expression());

      // If we hit a comma, consume it and continue
      if (current_token.value == ",") {
        advance(); // consume the comma
      }
    }

    if (current_token.value == ")") {
      advance(); // consume the closing parenthesis
    }

    // Create a Call node with the function as first child, then arguments
    ReaderNode call_node(ReaderNodeType::Call, token);
    call_node.children.push_back(std::move(left));
    for (auto &arg : arguments) {
      call_node.children.push_back(std::move(arg));
    }

    return call_node;
  }

  if (token.type == TokenType::Delimiter && token.value == "{") {
    // This is a potential struct literal - only attach blocks to uppercase
    // identifiers Check if left operand is an uppercase identifier (struct
    // name)
    if (left.type != ReaderNodeType::Ident || left.value().empty() ||
        !std::isupper(left.value()[0])) {
      // Not an uppercase identifier, don't treat as struct literal
      // But we've already consumed the { token, so we need to parse the block
      // and return it as a postfix operation (which will likely cause issues
      // upstream)
      std::vector<ReaderNode> block_contents;

      while (current_token.type != TokenType::End &&
             current_token.value != "}") {
        block_contents.push_back(parse_expression());
      }

      if (current_token.value == "}") {
        advance(); // consume the closing brace
      }

      ReaderNode block_node(ReaderNodeType::Block, token);
      for (auto &content : block_contents) {
        block_node.children.push_back(std::move(content));
      }

      return ReaderNode(ReaderNodeType::PostfixOp, token, {std::move(left)});
    }

    // Parse the block contents
    std::vector<ReaderNode> block_contents;

    while (current_token.type != TokenType::End && current_token.value != "}") {
      block_contents.push_back(parse_expression());
    }

    if (current_token.value == "}") {
      advance(); // consume the closing brace
    }

    // Create a block node
    ReaderNode block_node(ReaderNodeType::Block, token);
    for (auto &content : block_contents) {
      block_node.children.push_back(std::move(content));
    }

    // Attach the block as a child of the identifier (left operand)
    left.children.push_back(std::move(block_node));

    return left;
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