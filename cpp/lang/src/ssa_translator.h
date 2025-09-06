#ifndef SSA_TRANSLATOR_H
#define SSA_TRANSLATOR_H

#include "ast.h"
#include "ssa_instruction.h"
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

struct BlockIdHash {
  std::size_t operator()(const BlockId& id) const {
    return std::hash<size_t>()(id.id);
  }
};

struct PhiIdHash {
  std::size_t operator()(const PhiId& id) const {
    return std::hash<size_t>()(id.id);
  }
};

struct FunctionSSA {
  std::string function_name;
  std::vector<SSABlock> blocks;
  std::unordered_map<PhiId, SSAPhi, PhiIdHash> phis;
  std::unordered_set<BlockId, BlockIdHash> sealed_blocks;
  
  FunctionSSA(const std::string& name) : function_name(name) {}
};

class SSATranslator {
private:
  std::unordered_map<std::string, std::unordered_map<BlockId, SSAValue, BlockIdHash>> definitions;
  std::unordered_set<BlockId, BlockIdHash> sealed_blocks;
  std::unordered_map<BlockId, std::unordered_map<std::string, PhiId>, BlockIdHash> incomplete_phis;
  std::unordered_map<PhiId, SSAPhi, PhiIdHash> phis;
  std::vector<SSABlock> blocks;
  size_t next_variable_id;
  size_t next_phi_id;
  BlockId current_block;
  bool current_block_ended;   // Track if current block has ended (e.g., after break)
  BlockId current_loop_exit;  // Track current loop exit for break statements
  
  // Multi-function support
  std::vector<FunctionSSA> functions;
  std::string current_function;
  
  // Track phi operand resolution to prevent infinite recursion
  std::unordered_set<PhiId, PhiIdHash> resolving_phi_operands;

public:
  SSATranslator();
  ~SSATranslator();
  
  SSAValue translate(const ASTNode* ast);
  
  // Accessors for visualization
  const std::vector<SSABlock>& get_blocks() const { return blocks; }
  const std::unordered_map<PhiId, SSAPhi, PhiIdHash>& get_phis() const { return phis; }
  const std::unordered_map<BlockId, std::unordered_map<std::string, PhiId>, BlockIdHash>& get_incomplete_phis() const { return incomplete_phis; }
  const std::unordered_set<BlockId, BlockIdHash>& get_sealed_blocks() const { return sealed_blocks; }
  
  // Multi-function accessors
  const std::vector<FunctionSSA>& get_functions() const { return functions; }

private:
  void write_variable(const std::string& variable, BlockId block_id, const SSAValue& value);
  SSAValue read_variable(const std::string& variable, BlockId block_id);
  SSAValue read_variable_recursively(const std::string& variable, BlockId block_id);
  SSAValue add_phi_operands(const std::string& variable, PhiId phi_id);
  SSAValue try_remove_trivial_phi(PhiId phi_id);
  void seal_block(BlockId block_id);
  Variable get_temp_variable(const std::string& prefix);
  PhiId create_phi(BlockId block_id);
  void add_phi_use(PhiId phi_id, BlockId block_id, size_t instruction_offset);
  void add_phi_phi_use(PhiId phi_id, PhiId user_phi_id);
  void replace_phi_uses(PhiId phi_id, const SSAValue& replacement);
  void replace_value_at_location(BlockId block_id, size_t instruction_offset,
                                PhiId old_phi_id, const SSAValue& new_value);
  static void replace_value_in_instruction(SSAInstruction& instruction,
                                          PhiId old_phi_id, const SSAValue& new_value);
  void remove_phi(PhiId phi_id);
  BlockId create_block();
  
  // AST node translation methods
  SSAValue translate_literal(const ASTNode* node);
  SSAValue translate_identifier(const ASTNode* node);
  SSAValue translate_binary_expression(const ASTNode* node);
  SSAValue translate_unary_expression(const ASTNode* node);
  SSAValue translate_assignment(const ASTNode* node);
  SSAValue translate_if_statement(const ASTNode* node);
  SSAValue translate_loop_statement(const ASTNode* node);
  SSAValue translate_block(const ASTNode* node);
  SSAValue translate_expression_statement(const ASTNode* node);
  SSAValue translate_break_statement(const ASTNode* node);
  SSAValue translate_function_call(const ASTNode* node);
  SSAValue translate_list_literal(const ASTNode* node);
  SSAValue translate_tuple_literal(const ASTNode* node);
  SSAValue translate_function_declaration(const ASTNode* node);
  
  // Helper methods to convert AST operators to SSA operators
  BinaryOperator convert_binary_operator(const std::string& op);
  UnaryOperator convert_unary_operator(const std::string& op);
};

#endif