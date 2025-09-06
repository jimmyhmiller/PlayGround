#ifndef AST_H
#define AST_H

#include "reader.h"
#include "tokenizer.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

enum class ASTNodeType {
  // Literals and identifiers
  Identifier,
  NumberLiteral,
  StringLiteral,
  BoolLiteral,
  ListLiteral,
  TupleLiteral,

  // Expressions
  BinaryExpression,
  UnaryExpression,
  CallExpression,
  FunctionCall,
  LambdaExpression,

  // Statements
  FunctionDeclaration,
  LetStatement,
  MutableLetStatement,
  IfStatement,
  LoopStatement,
  BreakStatement,
  ReturnStatement,
  ExpressionStatement,
  AssignmentStatement,

  // Type expressions
  TypeIdentifier,
  FunctionType,
  GenericType,
  TupleType,

  // Program structure
  Program,
  Block,

  // Parameters and arguments
  Parameter,
  Argument,

  // Struct declarations
  StructDeclaration,
  FieldDeclaration,
};

struct ASTNode {
  ASTNodeType type;
  Token token;
  std::vector<std::unique_ptr<ASTNode>> children;
  std::string value;

  // Specific fields for different node types
  std::string name; // For identifiers, function names, etc.
  std::unique_ptr<ASTNode> function_type; // For function declarations
  std::unique_ptr<ASTNode> body; // For function declarations, blocks, etc.

  ASTNode(ASTNodeType type, Token token, std::string value = "")
      : type(type), token(token), value(value) {}

  void add_child(std::unique_ptr<ASTNode> child) {
    children.push_back(std::move(child));
  }

  ASTNode *child(size_t index) const {
    return index < children.size() ? children[index].get() : nullptr;
  }

  size_t child_count() const { return children.size(); }
};

struct ASTBuilder {
  std::vector<ReaderNode> reader_nodes;
  size_t current_index;

  ASTBuilder(std::vector<ReaderNode> nodes)
      : reader_nodes(std::move(nodes)), current_index(0) {}

  std::unique_ptr<ASTNode> build();
  void preprocess_function_calls();
  ReaderNode preprocess_node_recursively(const ReaderNode &node);
  std::unique_ptr<ASTNode> parse_expression(int rightBindingPower = 0);
  std::unique_ptr<ASTNode> parse_prefix();
  std::unique_ptr<ASTNode> parse_infix(std::unique_ptr<ASTNode> left);
  std::unique_ptr<ASTNode> parse_statement();
  std::unique_ptr<ASTNode> parse_function_declaration();
  std::unique_ptr<ASTNode> parse_struct_declaration();
  std::unique_ptr<ASTNode> parse_let_statement();
  std::unique_ptr<ASTNode> parse_if_statement();
  std::unique_ptr<ASTNode> parse_loop_statement();
  std::unique_ptr<ASTNode> parse_block();
  std::unique_ptr<ASTNode> parse_type();
  std::unique_ptr<ASTNode> parse_parameter_list();
  void parse_parameters_recursive(const ReaderNode &node, ASTNode *param_list);
  std::unique_ptr<ASTNode>
  parse_call_expression(std::unique_ptr<ASTNode> callee);
  std::unique_ptr<ASTNode> parse_lambda_expression();

  int get_binding_power(const ReaderNode &node);
  bool is_at_end() const;
  const ReaderNode &current() const;
  const ReaderNode &advance();
  const ReaderNode &peek(size_t offset = 1) const;
  bool match_token(const std::string &value) const;
  bool match_type(ReaderNodeType type) const;
};

#endif