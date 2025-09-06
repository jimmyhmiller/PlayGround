#ifndef AST_CODE_H
#define AST_CODE_H

#include "ast.h"
#include <string>

std::string ast_to_code(const ASTNode *node, int indent = 0);
std::string generate_function_declaration(const ASTNode *node, int indent);
std::string generate_struct_declaration(const ASTNode *node, int indent);
std::string generate_let_statement(const ASTNode *node, int indent);
std::string generate_mutable_let_statement(const ASTNode *node, int indent);
std::string generate_assignment_statement(const ASTNode *node, int indent);
std::string generate_if_statement(const ASTNode *node, int indent);
std::string generate_loop_statement(const ASTNode *node, int indent);
std::string generate_break_statement(const ASTNode *node, int indent);
std::string generate_expression(const ASTNode *node, int indent);
std::string generate_block(const ASTNode *node, int indent);
std::string generate_parameter_list(const ASTNode *params);
std::string generate_type(const ASTNode *type_node);

#endif