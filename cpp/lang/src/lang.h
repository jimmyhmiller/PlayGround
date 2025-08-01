#ifndef LANG_H
#define LANG_H

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

enum class TokenType {
  Identifier,
  NUMBER,
  String,
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

enum class ReaderNodeType {
  Ident,
  Literal,
  List,
  Block,
  BinaryOp,
  PrefixOp,
  PostfixOp,
  Call,
};

struct ReaderNode {
  ReaderNodeType type;
  Token token;
  std::vector<ReaderNode> children;

  ReaderNode(ReaderNodeType type, Token token) : type(type), token(token) {}

  ReaderNode(ReaderNodeType type, Token token, std::vector<ReaderNode> children)
      : type(type), token(token), children(std::move(children)) {}

  void add_child(ReaderNode child) { children.push_back(std::move(child)); }

  // Helper to get the value for backwards compatibility
  std::string_view value() const { return token.value; }

  bool operator==(const ReaderNode &other) const {
    return type == other.type && token.value == other.token.value &&
           children == other.children;
  }
};

// The goal is for tokenizer to be incremental
// so it will not tokenize the whole input,
// but will just return the next token
// when requested, and will keep track of the position
// in the input string.
struct Tokenizer {
  int pos = 0;
  int line = 1;
  int column = 0;

  Tokenizer() {}

  bool at_end(const std::string_view input) const {
    return static_cast<size_t>(pos) >= input.size();
  }

  void consume(const std::string_view input, int count = 1) {
    for (int i = 0; i < count; i++) {
      if (static_cast<size_t>(pos) < input.size() && input[pos] == '\n') {
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
    return c == '(' || c == ')' || c == '[' || c == ']' || c == '{' || c == '}';
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

struct Reader {
  ReaderNode root =
      ReaderNode(ReaderNodeType::List, Token{TokenType::End, "", 0, 0});
  Tokenizer tokenizer;
  std::string_view input;
  Reader(const std::string_view input) : input(input) {}
  Token current_token;

  Token advance();
  Token peek() const;
  void read();
  int get_binding_power(const Token &token, bool isPostfix = false);
  ReaderNode parse_expression(int rightBindingPower = 0);
  ReaderNode parse_prefix(const Token &token);
  ReaderNode parse_infix(ReaderNode left, const Token &token);
  ReaderNode parse_postfix(ReaderNode left, const Token &token);
  ReaderNode parse_block();
};

#endif // LANG_H