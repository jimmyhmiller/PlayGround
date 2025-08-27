#ifndef AST_JSON_H
#define AST_JSON_H

#include "ast.h"
#include <string>

std::string ast_node_type_to_string(ASTNodeType type);
std::string ast_to_json(const ASTNode *node, int indent = 0);
std::string escape_json_string(const std::string &str);

#endif