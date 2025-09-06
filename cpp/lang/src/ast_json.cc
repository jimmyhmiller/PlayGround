#include "ast_json.h"
#include <iomanip>
#include <sstream>

std::string ast_node_type_to_string(ASTNodeType type) {
  switch (type) {
  case ASTNodeType::Identifier:
    return "Identifier";
  case ASTNodeType::NumberLiteral:
    return "NumberLiteral";
  case ASTNodeType::StringLiteral:
    return "StringLiteral";
  case ASTNodeType::BoolLiteral:
    return "BoolLiteral";
  case ASTNodeType::ListLiteral:
    return "ListLiteral";
  case ASTNodeType::TupleLiteral:
    return "TupleLiteral";
  case ASTNodeType::BinaryExpression:
    return "BinaryExpression";
  case ASTNodeType::UnaryExpression:
    return "UnaryExpression";
  case ASTNodeType::CallExpression:
    return "CallExpression";
  case ASTNodeType::FunctionCall:
    return "FunctionCall";
  case ASTNodeType::LambdaExpression:
    return "LambdaExpression";
  case ASTNodeType::FunctionDeclaration:
    return "FunctionDeclaration";
  case ASTNodeType::LetStatement:
    return "LetStatement";
  case ASTNodeType::MutableLetStatement:
    return "MutableLetStatement";
  case ASTNodeType::IfStatement:
    return "IfStatement";
  case ASTNodeType::LoopStatement:
    return "LoopStatement";
  case ASTNodeType::BreakStatement:
    return "BreakStatement";
  case ASTNodeType::ReturnStatement:
    return "ReturnStatement";
  case ASTNodeType::ExpressionStatement:
    return "ExpressionStatement";
  case ASTNodeType::AssignmentStatement:
    return "AssignmentStatement";
  case ASTNodeType::TypeIdentifier:
    return "TypeIdentifier";
  case ASTNodeType::FunctionType:
    return "FunctionType";
  case ASTNodeType::GenericType:
    return "GenericType";
  case ASTNodeType::TupleType:
    return "TupleType";
  case ASTNodeType::Program:
    return "Program";
  case ASTNodeType::Block:
    return "Block";
  case ASTNodeType::Parameter:
    return "Parameter";
  case ASTNodeType::Argument:
    return "Argument";
  case ASTNodeType::StructDeclaration:
    return "StructDeclaration";
  case ASTNodeType::FieldDeclaration:
    return "FieldDeclaration";
  default:
    return "Unknown";
  }
}

std::string escape_json_string(const std::string &str) {
  std::ostringstream escaped;
  for (char c : str) {
    switch (c) {
    case '"':
      escaped << "\\\"";
      break;
    case '\\':
      escaped << "\\\\";
      break;
    case '\b':
      escaped << "\\b";
      break;
    case '\f':
      escaped << "\\f";
      break;
    case '\n':
      escaped << "\\n";
      break;
    case '\r':
      escaped << "\\r";
      break;
    case '\t':
      escaped << "\\t";
      break;
    default:
      if (c >= 0 && c < 32) {
        escaped << "\\u" << std::hex << std::setfill('0') << std::setw(4)
                << static_cast<int>(c);
      } else {
        escaped << c;
      }
      break;
    }
  }
  return escaped.str();
}

std::string ast_to_json(const ASTNode *node, int indent) {
  if (!node) {
    return "null";
  }

  std::string indentStr(indent * 2, ' ');
  std::string nextIndentStr((indent + 1) * 2, ' ');

  std::ostringstream json;
  json << "{\n";
  json << nextIndentStr << "\"type\": \"" << ast_node_type_to_string(node->type)
       << "\",\n";

  if (!node->name.empty()) {
    json << nextIndentStr << "\"name\": \"" << escape_json_string(node->name)
         << "\",\n";
  }

  if (!node->value.empty()) {
    json << nextIndentStr << "\"value\": \"" << escape_json_string(node->value)
         << "\",\n";
  }

  json << nextIndentStr << "\"token\": {\n";
  json << nextIndentStr << "  \"type\": " << static_cast<int>(node->token.type)
       << ",\n";
  json << nextIndentStr << "  \"value\": \""
       << escape_json_string(std::string(node->token.value)) << "\",\n";
  json << nextIndentStr << "  \"line\": " << node->token.line << ",\n";
  json << nextIndentStr << "  \"column\": " << node->token.column << "\n";
  json << nextIndentStr << "}";

  // Add function type if present
  if (node->function_type) {
    json << ",\n"
         << nextIndentStr << "\"function_type\": "
         << ast_to_json(node->function_type.get(), indent + 1);
  }

  // Add body if present
  if (node->body) {
    json << ",\n"
         << nextIndentStr
         << "\"body\": " << ast_to_json(node->body.get(), indent + 1);
  }

  // Add children if present
  if (node->child_count() > 0) {
    json << ",\n" << nextIndentStr << "\"children\": [\n";

    for (size_t i = 0; i < node->child_count(); ++i) {
      if (i > 0) {
        json << ",\n";
      }
      json << nextIndentStr << "  " << ast_to_json(node->child(i), indent + 2);
    }

    json << "\n" << nextIndentStr << "]";
  }

  json << "\n" << indentStr << "}";
  return json.str();
}