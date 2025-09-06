#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <string_view>

enum class TokenType {
  Identifier,
  NUMBER,
  String,
  Boolean,
  Operator,
  Delimiter,
  Separator,
  Comment,
  Whitespace,
  End
};

struct Token {
  TokenType type;
  std::string_view value;
  int line;
  int column;
};

class Tokenizer {
public:
  int pos = 0;
  int line = 1;
  int column = 0;

  bool at_end(const std::string_view input) const {
    return static_cast<size_t>(pos) >= input.size();
  }

  void consume(const std::string_view input, int count = 1) {
    for (int i = 0; i < count && !at_end(input); ++i) {
      if (input[pos] == '\n') {
        line++;
        column = 0;
      } else {
        column++;
      }
      pos++;
    }
  }

  bool is_whitespace(char c) const {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }
  bool is_delimiter(char c) const {
    return c == '(' || c == ')' || c == '{' || c == '}' || c == '[' || c == ']';
  }
  bool is_separator(char c) const { return c == ',' || c == ';' || c == ':'; }
  bool is_operator(char c) const {
    return c == '+' || c == '-' || c == '*' || c == '/' || c == '=' ||
           c == '<' || c == '>' || c == '!' || c == '&' || c == '|' ||
           c == '^' || c == '~' || c == '%' || c == '?' || c == '.';
  }

  Token handle_whitespace(const std::string_view input);
  Token next_token(const std::string_view input);
};

#endif