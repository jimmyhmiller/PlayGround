#ifndef READER_H
#define READER_H

#include "tokenizer.h"
#include <string_view>
#include <vector>

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

  std::string_view value() const { return token.value; }

  void add_child(const ReaderNode &child) { children.push_back(child); }

  bool operator==(const ReaderNode &other) const {
    if (type != other.type)
      return false;
    if (token.value != other.token.value)
      return false;
    if (children.size() != other.children.size())
      return false;
    for (size_t i = 0; i < children.size(); ++i) {
      if (!(children[i] == other.children[i]))
        return false;
    }
    return true;
  }
};

struct Reader {
  ReaderNode root =
      ReaderNode(ReaderNodeType::List, Token{TokenType::End, "", 0, 0});
  Tokenizer tokenizer;
  std::string_view input;
  Token current_token;

  Reader(const std::string_view input) : input(input) {}

  enum Associativity {
    LEFT,
    RIGHT
  };
  
  struct OperatorInfo {
    int precedence;
    Associativity associativity;
  };
  
  Token advance();
  Token peek() const;
  void read();
  OperatorInfo get_operator_info(const Token &token, bool isPostfix = false);
  int get_binding_power(const Token &token, bool isPostfix = false);
  ReaderNode parse_expression(int rightBindingPower = 0);
  ReaderNode parse_prefix(const Token &token);
  ReaderNode parse_infix(ReaderNode left, const Token &token);
  ReaderNode parse_postfix(ReaderNode left, const Token &token);
  ReaderNode parse_block();
};

#endif