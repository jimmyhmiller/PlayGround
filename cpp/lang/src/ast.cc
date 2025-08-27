#include "ast.h"
#include <stdexcept>
#include <functional>

std::unique_ptr<ASTNode> ASTBuilder::build() {
  auto program = std::make_unique<ASTNode>(ASTNodeType::Program,
                                           Token{TokenType::End, "", 0, 0});

  // Use proper precedence climbing

  while (!is_at_end()) {
    auto stmt = parse_statement();
    if (stmt) {
      program->add_child(std::move(stmt));
    }
  }

  return program;
}

void ASTBuilder::preprocess_function_calls() {
  std::vector<ReaderNode> processed_nodes;
  
  for (size_t i = 0; i < reader_nodes.size(); ++i) {
    // Check for function call pattern: identifier followed by list
    if (i + 1 < reader_nodes.size() &&
        reader_nodes[i].type == ReaderNodeType::Ident &&
        reader_nodes[i + 1].type == ReaderNodeType::List &&
        reader_nodes[i + 1].token.value == "(") {
      
      // Create a synthetic Call node
      ReaderNode call_node(ReaderNodeType::Call, reader_nodes[i].token);
      call_node.children.push_back(reader_nodes[i]);
      
      // Add arguments from the list
      for (const auto& arg : reader_nodes[i + 1].children) {
        call_node.children.push_back(arg);
      }
      
      processed_nodes.push_back(std::move(call_node));
      ++i; // Skip the next node since we consumed it
    } else {
      // Also preprocess nested structures (like binary operations)
      ReaderNode processed = preprocess_node_recursively(reader_nodes[i]);
      processed_nodes.push_back(std::move(processed));
    }
  }
  
  reader_nodes = std::move(processed_nodes);
}

ReaderNode ASTBuilder::preprocess_node_recursively(const ReaderNode& node) {
  ReaderNode processed = node;
  processed.children.clear();
  
  // Recursively preprocess children, looking for function call patterns
  for (size_t i = 0; i < node.children.size(); ++i) {
    if (i + 1 < node.children.size() &&
        node.children[i].type == ReaderNodeType::Ident &&
        node.children[i + 1].type == ReaderNodeType::List &&
        node.children[i + 1].token.value == "(") {
      
      // Create a synthetic Call node
      ReaderNode call_node(ReaderNodeType::Call, node.children[i].token);
      call_node.children.push_back(node.children[i]);
      
      // Add arguments from the list
      for (const auto& arg : node.children[i + 1].children) {
        call_node.children.push_back(arg);
      }
      
      processed.children.push_back(std::move(call_node));
      ++i; // Skip the next node since we consumed it
    } else {
      // Recursively process this child
      ReaderNode child_processed = preprocess_node_recursively(node.children[i]);
      processed.children.push_back(std::move(child_processed));
    }
  }
  
  return processed;
}

bool ASTBuilder::is_at_end() const {
  return current_index >= reader_nodes.size();
}

const ReaderNode &ASTBuilder::current() const {
  if (is_at_end()) {
    static ReaderNode end_node(ReaderNodeType::Ident,
                               Token{TokenType::End, "", 0, 0});
    return end_node;
  }
  return reader_nodes[current_index];
}

const ReaderNode &ASTBuilder::advance() {
  if (!is_at_end()) {
    current_index++;
  }
  return current();
}

const ReaderNode &ASTBuilder::peek(size_t offset) const {
  size_t peek_index = current_index + offset;
  if (peek_index >= reader_nodes.size()) {
    static ReaderNode end_node(ReaderNodeType::Ident,
                               Token{TokenType::End, "", 0, 0});
    return end_node;
  }
  return reader_nodes[peek_index];
}

bool ASTBuilder::match_token(const std::string &value) const {
  return !is_at_end() && current().value() == value;
}

bool ASTBuilder::match_type(ReaderNodeType type) const {
  return !is_at_end() && current().type == type;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_statement() {
  if (match_token("fn")) {
    return parse_function_declaration();
  }
  if (match_token("let")) {
    return parse_let_statement();
  }
  if (match_token("if")) {
    return parse_if_statement();
  }
  if (match_token("loop")) {
    return parse_loop_statement();
  }
  if (match_token("break")) {
    auto break_stmt =
        std::make_unique<ASTNode>(ASTNodeType::BreakStatement, current().token);
    advance();
    return break_stmt;
  }

  if (current().type == ReaderNodeType::BinaryOp && current().value() == "=" &&
      current().children.size() == 2 &&
      current().children[0].type == ReaderNodeType::Ident) {

    auto assignment = std::make_unique<ASTNode>(ASTNodeType::AssignmentStatement,
                                                current().token);
    auto var_name = std::make_unique<ASTNode>(
        ASTNodeType::Identifier, current().children[0].token,
        std::string(current().children[0].value()));
    assignment->add_child(std::move(var_name));

    ASTBuilder rhs_builder({current().children[1]});
    auto rhs = rhs_builder.parse_expression();
    if (rhs) {
      assignment->add_child(std::move(rhs));
    }

    advance();
    return assignment;
  }

  auto expr = parse_expression();
  if (expr) {
    auto expr_stmt = std::make_unique<ASTNode>(ASTNodeType::ExpressionStatement,
                                               current().token);
    expr_stmt->add_child(std::move(expr));
    return expr_stmt;
  }

  advance();
  return nullptr;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_function_declaration() {
  if (!match_token("fn")) {
    return nullptr;
  }

  auto fn_node = std::make_unique<ASTNode>(ASTNodeType::FunctionDeclaration,
                                           current().token);
  advance();

  // With the new reader structure, we expect: (: name (-> params return_type))
  if (!is_at_end() && current().type == ReaderNodeType::BinaryOp &&
      current().value() == ":") {
    const ReaderNode &colon_op = current();
    advance();

    // Extract function name from left side of :
    if (colon_op.children.size() >= 1 &&
        colon_op.children[0].type == ReaderNodeType::Ident) {
      fn_node->name = std::string(colon_op.children[0].value());
    }

    // The right side of : should be the function type (-> params return_type)
    if (colon_op.children.size() >= 2) {
      const ReaderNode &func_type_node = colon_op.children[1];
      
      if (func_type_node.type == ReaderNodeType::BinaryOp && func_type_node.value() == "->") {
        // Create function type
        auto func_type = std::make_unique<ASTNode>(ASTNodeType::FunctionType, func_type_node.token);
        
        // Parse parameters from left side of ->
        if (func_type_node.children.size() >= 1) {
          ASTBuilder param_builder({func_type_node.children[0]});
          auto params = param_builder.parse_parameter_list();
          if (params) {
            func_type->add_child(std::move(params));
          }
        }
        
        // Parse return type from right side of ->
        if (func_type_node.children.size() >= 2) {
          ASTBuilder return_type_builder({func_type_node.children[1]});
          auto return_type = return_type_builder.parse_type();
          if (return_type) {
            func_type->add_child(std::move(return_type));
          }
        }
        
        fn_node->function_type = std::move(func_type);
      }
    }
  }

  // Look for the function body (block)
  if (!is_at_end() && match_type(ReaderNodeType::Block)) {
    auto body = parse_block();
    if (body) {
      fn_node->body = std::move(body);
    }
  }

  return fn_node;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_let_statement() {
  if (!match_token("let")) {
    return nullptr;
  }

  Token let_token = current().token;
  advance();

  // Check for 'mut' keyword
  bool is_mutable = false;
  if (!is_at_end() && match_token("mut")) {
    is_mutable = true;
    advance();
  }

  // Create appropriate node type
  std::unique_ptr<ASTNode> let_node;
  if (is_mutable) {
    let_node = std::make_unique<ASTNode>(ASTNodeType::MutableLetStatement, let_token);
  } else {
    let_node = std::make_unique<ASTNode>(ASTNodeType::LetStatement, let_token);
  }

  // Check if the next expression is an assignment
  if (!is_at_end() && current().type == ReaderNodeType::BinaryOp &&
      current().value() == "=" && current().children.size() == 2) {

    const ReaderNode &assignment = current();

    // Extract variable name from left side of assignment
    if (assignment.children[0].type == ReaderNodeType::Ident) {
      auto var_name = std::make_unique<ASTNode>(
          ASTNodeType::Identifier, assignment.children[0].token,
          std::string(assignment.children[0].value()));
      let_node->add_child(std::move(var_name));

      // Extract initialization expression from right side
      ASTBuilder rhs_builder({assignment.children[1]});
      auto init_expr = rhs_builder.parse_expression();
      if (init_expr) {
        let_node->add_child(std::move(init_expr));
      }
    }

    advance(); // consume the assignment expression
  }

  return let_node;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_if_statement() {
  if (!match_token("if")) {
    return nullptr;
  }

  auto if_node =
      std::make_unique<ASTNode>(ASTNodeType::IfStatement, current().token);
  advance();

  auto condition = parse_expression();
  if (condition) {
    if_node->add_child(std::move(condition));
  }

  if (match_type(ReaderNodeType::Block)) {
    auto then_block = parse_block();
    if (then_block) {
      if_node->add_child(std::move(then_block));
    }
  }

  if (match_token("else")) {
    advance();
    if (match_type(ReaderNodeType::Block)) {
      auto else_block = parse_block();
      if (else_block) {
        if_node->add_child(std::move(else_block));
      }
    } else if (match_token("if")) {
      auto else_if = parse_if_statement();
      if (else_if) {
        if_node->add_child(std::move(else_if));
      }
    }
  }

  return if_node;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_loop_statement() {
  if (!match_token("loop")) {
    return nullptr;
  }

  auto loop_node =
      std::make_unique<ASTNode>(ASTNodeType::LoopStatement, current().token);
  advance();

  if (match_type(ReaderNodeType::Block)) {
    auto body = parse_block();
    if (body) {
      loop_node->add_child(std::move(body));
    }
  }

  return loop_node;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_block() {
  if (!match_type(ReaderNodeType::Block)) {
    return nullptr;
  }

  auto block_node =
      std::make_unique<ASTNode>(ASTNodeType::Block, current().token);
  const ReaderNode &block_reader = current();
  advance();

  // Parse statements directly (no need to flatten since semicolons are now handled by reader)
  ASTBuilder block_builder(std::vector<ReaderNode>(block_reader.children));
  while (!block_builder.is_at_end()) {
    auto stmt = block_builder.parse_statement();
    if (stmt) {
      block_node->add_child(std::move(stmt));
    }
  }

  return block_node;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_parameter_list() {
  if (!match_type(ReaderNodeType::List)) {
    return nullptr;
  }

  auto param_list = std::make_unique<ASTNode>(ASTNodeType::Parameter, current().token);
  const ReaderNode &list_reader = current();
  advance();
  
  // Parse all parameters in the list
  for (const auto &child : list_reader.children) {
    parse_parameters_recursive(child, param_list.get());
  }

  return param_list;
}

void ASTBuilder::parse_parameters_recursive(const ReaderNode &node, ASTNode *param_list) {
  if (node.type == ReaderNodeType::List) {
    // Handle list of parameters
    for (const auto &child : node.children) {
      parse_parameters_recursive(child, param_list);
    }
  } else if (node.type == ReaderNodeType::BinaryOp && node.value() == ":") {
    // This is a typed parameter: name : type
    if (node.children.size() == 2) {
      auto param = std::make_unique<ASTNode>(ASTNodeType::Parameter, node.token);
      
      if (node.children[0].type == ReaderNodeType::Ident) {
        param->name = std::string(node.children[0].value());
        
        // Parse the type
        ASTBuilder type_builder({node.children[1]});
        auto param_type = type_builder.parse_type();
        if (param_type) {
          param->add_child(std::move(param_type));
        }
        
        param_list->add_child(std::move(param));
      }
    }
  } else if (node.type == ReaderNodeType::Ident) {
    // Simple untyped parameter
    auto param = std::make_unique<ASTNode>(ASTNodeType::Parameter, node.token);
    param->name = std::string(node.value());
    param_list->add_child(std::move(param));
  }
}


std::unique_ptr<ASTNode> ASTBuilder::parse_type() {
  if (match_type(ReaderNodeType::List)) {
    // Handle parenthesized types like ((a) -> c) and tuple types like (a, b)
    const ReaderNode &list_reader = current();
    advance();
    
    if (list_reader.children.size() == 1) {
      // Single element in parentheses - parse it as a type and mark as parenthesized
      ASTBuilder type_builder({list_reader.children[0]});
      auto type_node = type_builder.parse_type();
      if (type_node && type_node->type == ASTNodeType::FunctionType) {
        // Mark function types as parenthesized to preserve the original parentheses
        type_node->name = "parenthesized";
      }
      return type_node;
    } else if (list_reader.children.size() > 1) {
      // Multiple elements - this is a tuple type
      auto tuple_type = std::make_unique<ASTNode>(ASTNodeType::TupleType, list_reader.token);
      for (const auto &child : list_reader.children) {
        ASTBuilder child_builder({child});
        auto child_type = child_builder.parse_type();
        if (child_type) {
          tuple_type->add_child(std::move(child_type));
        }
      }
      return tuple_type;
    }
    // Empty list
    return nullptr;
  } else if (match_type(ReaderNodeType::Ident)) {
    auto type_node =
        std::make_unique<ASTNode>(ASTNodeType::TypeIdentifier, current().token,
                                  std::string(current().value()));
    advance();
    return type_node;
  } else if (current().type == ReaderNodeType::BinaryOp && current().value() == "->") {
    // Function type: param_types -> return_type
    auto func_type = std::make_unique<ASTNode>(ASTNodeType::FunctionType, current().token);
    const ReaderNode &arrow_node = current();
    advance();
    
    if (arrow_node.children.size() >= 1) {
      // Parse parameter types (left side of ->)
      ASTBuilder param_builder({arrow_node.children[0]});
      auto param_types = param_builder.parse_parameter_list();
      if (param_types) {
        func_type->add_child(std::move(param_types));
      }
    }
    
    if (arrow_node.children.size() >= 2) {
      // Parse return type (right side of ->)
      ASTBuilder return_builder({arrow_node.children[1]});
      auto return_type = return_builder.parse_type();
      if (return_type) {
        func_type->add_child(std::move(return_type));
      }
    }
    
    return func_type;
  } else if (match_type(ReaderNodeType::Call)) {
    // Parameterized type using function call syntax: List(t)
    const ReaderNode &call_reader = current();
    auto generic_type = std::make_unique<ASTNode>(ASTNodeType::GenericType, call_reader.token);
    advance();
    
    // Parse all children (type name + type parameters)
    for (const auto &child : call_reader.children) {
      ASTBuilder child_builder({child});
      auto parsed_child = child_builder.parse_type();
      if (parsed_child) {
        generic_type->add_child(std::move(parsed_child));
      }
    }
    
    return generic_type;
  }


  return nullptr;
}

int ASTBuilder::get_binding_power(const ReaderNode &node) {
  if (node.type == ReaderNodeType::BinaryOp) {
    if (node.value() == "||") {
      return 2;
    }
    if (node.value() == "==" || node.value() == "!=" || node.value() == "<" ||
        node.value() == ">" || node.value() == "<=" || node.value() == ">=") {
      return 5;
    }
    if (node.value() == "+" || node.value() == "-") {
      return 10;
    }
    if (node.value() == "*" || node.value() == "/") {
      return 20;
    }
    if (node.value() == "^") {
      return 30;
    }
    if (node.value() == ".") {
      return 35;
    }
    if (node.value() == "=") {
      return 2;
    }
    if (node.value() == "=>") {
      return 1;
    }
    return 1;
  }
  return 0;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_expression(int rightBindingPower) {
  if (is_at_end()) {
    return nullptr;
  }

  // Parse prefix (left-hand side)
  std::unique_ptr<ASTNode> left = parse_prefix();
  
  // Precedence climbing for infix operations
  while (!is_at_end() && rightBindingPower < get_binding_power(current())) {
    left = parse_infix(std::move(left));
  }
  
  return left;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_prefix() {
  if (is_at_end()) {
    return nullptr;
  }

  const ReaderNode &current_reader = current();

  if (current_reader.type == ReaderNodeType::Ident) {
    auto node = std::make_unique<ASTNode>(ASTNodeType::Identifier, current_reader.token,
                                         std::string(current_reader.value()));
    advance();
    return node;
  } else if (current_reader.type == ReaderNodeType::Call) {
    // Function call from reader
    auto call = std::make_unique<ASTNode>(ASTNodeType::FunctionCall, current_reader.token);
    advance();
    
    // Parse all children (function name + arguments)
    for (const auto &child : current_reader.children) {
      ASTBuilder child_builder({child});
      auto parsed_child = child_builder.parse_expression();
      if (parsed_child) {
        call->add_child(std::move(parsed_child));
      }
    }
    
    return call;
  } else if (current_reader.type == ReaderNodeType::Literal) {
    std::unique_ptr<ASTNode> node;
    if (current_reader.token.type == TokenType::NUMBER) {
      node = std::make_unique<ASTNode>(ASTNodeType::NumberLiteral,
                                       current_reader.token,
                                       std::string(current_reader.value()));
    } else if (current_reader.token.type == TokenType::String) {
      node = std::make_unique<ASTNode>(ASTNodeType::StringLiteral,
                                       current_reader.token,
                                       std::string(current_reader.value()));
    }
    advance();
    return node;
  } else if (current_reader.type == ReaderNodeType::List) {
    if (current_reader.token.value == "[") {
      // List literal
      auto list = std::make_unique<ASTNode>(ASTNodeType::ListLiteral, current_reader.token);
      advance();
      
      ASTBuilder list_builder(std::vector<ReaderNode>(current_reader.children));
      while (!list_builder.is_at_end()) {
        auto element = list_builder.parse_expression();
        if (element) {
          list->add_child(std::move(element));
        }
      }
      
      return list;
    } else if (current_reader.token.value == "(") {
      // Parentheses - could be grouping or tuple
      advance();
      ASTBuilder list_builder(std::vector<ReaderNode>(current_reader.children));
      
      if (list_builder.reader_nodes.size() == 0) {
        // Empty parentheses
        return std::make_unique<ASTNode>(ASTNodeType::TupleLiteral, current_reader.token);
      } else if (list_builder.reader_nodes.size() == 1) {
        // Check if it's a comma operation (tuple)
        const ReaderNode &single_child = list_builder.reader_nodes[0];
        if (single_child.type == ReaderNodeType::BinaryOp && single_child.value() == ",") {
          // Tuple literal
          auto tuple = std::make_unique<ASTNode>(ASTNodeType::TupleLiteral, current_reader.token);
          
          std::function<void(const ReaderNode&)> parse_comma_elements = [&](const ReaderNode& node) {
            if (node.type == ReaderNodeType::BinaryOp && node.value() == ",") {
              if (node.children.size() == 2) {
                parse_comma_elements(node.children[0]);
                parse_comma_elements(node.children[1]);
              }
            } else {
              ASTBuilder elem_builder({node});
              auto elem = elem_builder.parse_expression();
              if (elem) {
                tuple->add_child(std::move(elem));
              }
            }
          };
          
          parse_comma_elements(single_child);
          return tuple;
        } else {
          // Single element - just grouping
          return list_builder.parse_expression();
        }
      } else {
        // Multiple elements - tuple
        auto tuple = std::make_unique<ASTNode>(ASTNodeType::TupleLiteral, current_reader.token);
        while (!list_builder.is_at_end()) {
          auto expr = list_builder.parse_expression();
          if (expr) {
            tuple->add_child(std::move(expr));
          }
        }
        return tuple;
      }
    }
  } else if (current_reader.type == ReaderNodeType::BinaryOp) {
    // Handle binary operations from reader
    if (current_reader.value() == "=>") {
      // Lambda expression
      auto lambda = std::make_unique<ASTNode>(ASTNodeType::LambdaExpression, current_reader.token);
      advance();
      
      if (current_reader.children.size() >= 2) {
        // Left side is the parameter(s)
        ASTBuilder param_builder({current_reader.children[0]});
        auto param = param_builder.parse_expression();
        if (param) {
          lambda->add_child(std::move(param));
        }
        
        // Right side - now function calls are handled by reader
        ASTBuilder body_builder({current_reader.children[1]});
        auto body = body_builder.parse_expression();
        if (body) {
          lambda->add_child(std::move(body));
        }
      }
      
      return lambda;
    } else {
      // Regular binary expression
      auto binary = std::make_unique<ASTNode>(ASTNodeType::BinaryExpression,
                                              current_reader.token,
                                              std::string(current_reader.value()));
      advance();
      
      if (current_reader.children.size() >= 2) {
        ASTBuilder left_builder({current_reader.children[0]});
        auto left_operand = left_builder.parse_expression();
        if (left_operand) {
          binary->add_child(std::move(left_operand));
        }
        
        ASTBuilder right_builder({current_reader.children[1]});
        auto right_operand = right_builder.parse_expression();
        if (right_operand) {
          binary->add_child(std::move(right_operand));
        }
      }
      
      return binary;
    }
  }
  
  advance();
  return nullptr;
}

std::unique_ptr<ASTNode> ASTBuilder::parse_infix(std::unique_ptr<ASTNode> left) {
  if (is_at_end()) {
    return left;
  }

  const ReaderNode &op_node = current();
  
  if (op_node.type == ReaderNodeType::BinaryOp) {
    auto binary = std::make_unique<ASTNode>(ASTNodeType::BinaryExpression,
                                            op_node.token,
                                            std::string(op_node.value()));
    advance();
    
    int bp = get_binding_power(op_node);
    bool rightAssoc = (op_node.value() == "^" || op_node.value() == "=>");
    int nextBp = rightAssoc ? bp - 1 : bp;
    
    auto right = parse_expression(nextBp);
    if (right) {
      binary->add_child(std::move(left));
      binary->add_child(std::move(right));
      return binary;
    }
  }
  
  return left;
}
