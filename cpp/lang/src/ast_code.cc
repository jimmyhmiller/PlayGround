#include "ast_code.h"
#include <sstream>

std::string ast_to_code(const ASTNode* node, int indent) {
  if (!node) {
    return "";
  }
  
  switch (node->type) {
    case ASTNodeType::Program:
      {
        std::ostringstream result;
        for (size_t i = 0; i < node->child_count(); ++i) {
          if (i > 0) result << "\n\n";
          result << ast_to_code(node->child(i), indent);
        }
        return result.str();
      }
      
    case ASTNodeType::FunctionDeclaration:
      return generate_function_declaration(node, indent);
      
    case ASTNodeType::LetStatement:
      return generate_let_statement(node, indent);
      
    case ASTNodeType::MutableLetStatement:
      return generate_mutable_let_statement(node, indent);
      
    case ASTNodeType::AssignmentStatement:
      return generate_assignment_statement(node, indent);
      
    case ASTNodeType::IfStatement:
      return generate_if_statement(node, indent);
      
    case ASTNodeType::LoopStatement:
      return generate_loop_statement(node, indent);
      
    case ASTNodeType::BreakStatement:
      return generate_break_statement(node, indent);
      
    case ASTNodeType::ExpressionStatement:
      if (node->child_count() > 0) {
        return generate_expression(node->child(0), indent);
      }
      return "";
      
    case ASTNodeType::Block:
      return generate_block(node, indent);
      
    case ASTNodeType::BinaryExpression:
    case ASTNodeType::UnaryExpression:
    case ASTNodeType::CallExpression:
    case ASTNodeType::FunctionCall:
    case ASTNodeType::LambdaExpression:
    case ASTNodeType::ListLiteral:
    case ASTNodeType::TupleLiteral:
    case ASTNodeType::Identifier:
    case ASTNodeType::NumberLiteral:
    case ASTNodeType::StringLiteral:
      return generate_expression(node, indent);
      
    default:
      return "/* unsupported node type */";
  }
}

std::string generate_function_declaration(const ASTNode* node, int indent) {
  std::string indentStr(indent * 2, ' ');
  std::ostringstream result;
  
  result << indentStr << "fn " << node->name;
  
  if (node->function_type) {
    result << " : ";
    
    // Generate parameter list
    if (node->function_type->child_count() > 0) {
      result << generate_parameter_list(node->function_type->child(0));
    } else {
      result << "()";
    }
    
    result << " -> ";
    
    // Generate return type
    if (node->function_type->child_count() > 1) {
      result << generate_type(node->function_type->child(1));
    } else {
      result << "void";
    }
  }
  
  if (node->body) {
    result << " " << generate_block(node->body.get(), indent);
  }
  
  return result.str();
}

std::string generate_let_statement(const ASTNode* node, int indent) {
  std::string indentStr(indent * 2, ' ');
  std::ostringstream result;
  
  result << indentStr << "let ";
  
  // Get variable name from first child
  if (node->child_count() > 0) {
    auto var_node = node->child(0);
    if (var_node->type == ASTNodeType::Identifier) {
      result << var_node->value;
    }
  }
  
  // Get initialization expression from second child
  if (node->child_count() > 1) {
    result << " = " << generate_expression(node->child(1), 0);
  }
  
  return result.str();
}

std::string generate_mutable_let_statement(const ASTNode* node, int indent) {
  std::string indentStr(indent * 2, ' ');
  std::ostringstream result;
  
  result << indentStr << "let mut ";
  
  // Get variable name from first child
  if (node->child_count() > 0) {
    auto var_node = node->child(0);
    if (var_node->type == ASTNodeType::Identifier) {
      result << var_node->value;
    }
  }
  
  // Get initialization expression from second child
  if (node->child_count() > 1) {
    result << " = " << generate_expression(node->child(1), 0);
  }
  
  return result.str();
}

std::string generate_assignment_statement(const ASTNode* node, int indent) {
  std::string indentStr(indent * 2, ' ');
  std::ostringstream result;
  
  // Get variable name from first child
  if (node->child_count() > 0) {
    auto var_node = node->child(0);
    if (var_node->type == ASTNodeType::Identifier) {
      result << indentStr << var_node->value;
    }
  }
  
  // Get assignment expression from second child
  if (node->child_count() > 1) {
    result << " = " << generate_expression(node->child(1), 0);
  }
  
  return result.str();
}

std::string generate_if_statement(const ASTNode* node, int indent) {
  std::string indentStr(indent * 2, ' ');
  std::ostringstream result;
  
  result << indentStr << "if ";
  
  // Get condition from first child
  if (node->child_count() > 0) {
    result << generate_expression(node->child(0), 0);
  }
  
  // Get then block from second child
  if (node->child_count() > 1) {
    result << " " << generate_block(node->child(1), indent);
  }
  
  // Get else block from third child (if present)
  if (node->child_count() > 2) {
    result << " else " << generate_block(node->child(2), indent);
  }
  
  return result.str();
}

std::string generate_loop_statement(const ASTNode* node, int indent) {
  std::string indentStr(indent * 2, ' ');
  std::ostringstream result;
  
  result << indentStr << "loop";
  
  // Get loop body from first child
  if (node->child_count() > 0) {
    result << " " << generate_block(node->child(0), indent);
  }
  
  return result.str();
}

std::string generate_break_statement(const ASTNode* /* node */, int indent) {
  std::string indentStr(indent * 2, ' ');
  return indentStr + "break";
}

std::string generate_expression(const ASTNode* node, int /* indent */) {
  if (!node) return "";
  
  switch (node->type) {
    case ASTNodeType::Identifier:
      return node->value.empty() ? node->name : node->value;
      
    case ASTNodeType::NumberLiteral:
    case ASTNodeType::StringLiteral:
      return node->value;
      
    case ASTNodeType::BinaryExpression:
      if (node->child_count() >= 2) {
        return generate_expression(node->child(0), 0) + " " + 
               node->value + " " + 
               generate_expression(node->child(1), 0);
      }
      return "";
      
    case ASTNodeType::UnaryExpression:
      if (node->child_count() >= 1) {
        return node->value + generate_expression(node->child(0), 0);
      }
      return "";
      
    case ASTNodeType::CallExpression:
    case ASTNodeType::FunctionCall:
      {
        std::ostringstream result;
        if (node->child_count() > 0) {
          result << generate_expression(node->child(0), 0) << "(";
          for (size_t i = 1; i < node->child_count(); ++i) {
            if (i > 1) result << ", ";
            result << generate_expression(node->child(i), 0);
          }
          result << ")";
        }
        return result.str();
      }
      
    case ASTNodeType::ListLiteral:
      {
        std::ostringstream result;
        result << "[";
        for (size_t i = 0; i < node->child_count(); ++i) {
          if (i > 0) result << ", ";
          result << generate_expression(node->child(i), 0);
        }
        result << "]";
        return result.str();
      }
      
    case ASTNodeType::TupleLiteral:
      {
        std::ostringstream result;
        result << "(";
        for (size_t i = 0; i < node->child_count(); ++i) {
          if (i > 0) result << ", ";
          result << generate_expression(node->child(i), 0);
        }
        result << ")";
        return result.str();
      }
      
    case ASTNodeType::LambdaExpression:
      {
        std::ostringstream result;
        if (node->child_count() >= 2) {
          result << generate_expression(node->child(0), 0) << " => ";
          result << generate_expression(node->child(1), 0);
        }
        return result.str();
      }
      
    default:
      return "/* unknown expression */";
  }
}

std::string generate_block(const ASTNode* node, int indent) {
  std::string indentStr(indent * 2, ' ');
  std::string nextIndentStr((indent + 1) * 2, ' ');
  std::ostringstream result;
  
  result << "{\n";
  
  for (size_t i = 0; i < node->child_count(); ++i) {
    auto stmt = ast_to_code(node->child(i), indent + 1);
    if (!stmt.empty()) {
      // Check if the statement already has proper indentation
      // If it starts with spaces, it's already indented, otherwise add indentation
      if (stmt[0] != ' ') {
        result << nextIndentStr << stmt;
      } else {
        result << stmt;
      }
      
      // Add semicolon logic:
      // - All statements that don't return values should have semicolons
      // - The last expression in a block should NOT have a semicolon (it's the return value)
      bool is_last_statement = (i == node->child_count() - 1);
      bool always_needs_semicolon = (node->child(i)->type == ASTNodeType::LetStatement ||
                                     node->child(i)->type == ASTNodeType::MutableLetStatement ||
                                     node->child(i)->type == ASTNodeType::AssignmentStatement ||
                                     node->child(i)->type == ASTNodeType::BreakStatement);
      bool is_expression_statement = (node->child(i)->type == ASTNodeType::ExpressionStatement);
      
      if (always_needs_semicolon || (is_expression_statement && !is_last_statement)) {
        result << ";";
      }
      
      result << "\n";
    }
  }
  
  result << indentStr << "}";
  return result.str();
}

std::string generate_parameter_list(const ASTNode* params) {
  if (!params || params->child_count() == 0) {
    return "()";
  }
  
  std::ostringstream result;
  result << "(";
  
  bool first = true;
  for (size_t i = 0; i < params->child_count(); ++i) {
    auto param = params->child(i);
    if (param->type == ASTNodeType::Parameter && !param->name.empty()) {
      if (!first) result << ", ";
      first = false;
      
      result << param->name;
      
      // Add type if present
      if (param->child_count() > 0) {
        result << ": " << generate_type(param->child(0));
      }
    }
  }
  
  result << ")";
  return result.str();
}

std::string generate_type(const ASTNode* type_node) {
  if (!type_node) return "";
  
  switch (type_node->type) {
    case ASTNodeType::TypeIdentifier:
      return type_node->value;
      
    case ASTNodeType::GenericType:
      {
        std::ostringstream result;
        if (type_node->child_count() > 0) {
          // First child is the base type name
          result << generate_type(type_node->child(0));
          
          // Rest are type parameters
          if (type_node->child_count() > 1) {
            result << "(";
            for (size_t i = 1; i < type_node->child_count(); ++i) {
              if (i > 1) result << ", ";
              result << generate_type(type_node->child(i));
            }
            result << ")";
          }
        }
        return result.str();
      }
      
    case ASTNodeType::FunctionType:
      {
        std::ostringstream result;
        
        // Check if this function type should be wrapped in parentheses
        // This happens when it was originally in parentheses like ((a) -> c)
        bool needs_parens = !type_node->name.empty() && type_node->name == "parenthesized";
        
        if (needs_parens) result << "(";
        
        if (type_node->child_count() > 0) {
          result << generate_parameter_list(type_node->child(0));
        } else {
          result << "()";
        }
        result << " -> ";
        if (type_node->child_count() > 1) {
          result << generate_type(type_node->child(1));
        } else {
          result << "void";
        }
        
        if (needs_parens) result << ")";
        
        return result.str();
      }
      
    case ASTNodeType::TupleType:
      {
        std::ostringstream result;
        result << "(";
        for (size_t i = 0; i < type_node->child_count(); ++i) {
          if (i > 0) result << ", ";
          result << generate_type(type_node->child(i));
        }
        result << ")";
        return result.str();
      }
      
    default:
      return "unknown_type";
  }
}