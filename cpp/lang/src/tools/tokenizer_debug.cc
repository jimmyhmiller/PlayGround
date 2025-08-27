#include "../tokenizer.h"
#include <iostream>
#include <string>

std::string token_type_to_string(TokenType type) {
  switch (type) {
  case TokenType::Identifier:
    return "Identifier";
  case TokenType::NUMBER:
    return "NUMBER";
  case TokenType::String:
    return "String";
  case TokenType::Operator:
    return "Operator";
  case TokenType::Delimiter:
    return "Delimiter";
  case TokenType::Separator:
    return "Separator";
  case TokenType::Comment:
    return "Comment";
  case TokenType::Whitespace:
    return "Whitespace";
  case TokenType::End:
    return "End";
  default:
    return "Unknown";
  }
}

int main() {
  std::string line;
  std::string input;

  while (std::getline(std::cin, line)) {
    if (!input.empty()) {
      input += "\n";
    }
    input += line;
  }

  if (input.empty()) {
    std::cout << "No input provided\n";
    return 0;
  }

  Tokenizer tokenizer;
  Token token;

  std::cout << "Tokens:\n";
  std::cout << "-------\n";

  do {
    token = tokenizer.next_token(input);
    std::cout << token_type_to_string(token.type) << ": \"" << token.value
              << "\" [" << token.line << ":" << token.column << "]\n";
  } while (token.type != TokenType::End);

  return 0;
}