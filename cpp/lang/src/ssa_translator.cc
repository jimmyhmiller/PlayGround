#include "ssa_translator.h"
#include <stdexcept>
#include <iostream>

SSATranslator::SSATranslator() 
    : next_variable_id(0), next_phi_id(0), current_block(0), current_block_ended(false), current_loop_exit(-1) {
  std::cerr << "SSATranslator constructor called" << std::endl;
  BlockId initial_block = create_block();
  current_block = initial_block;
}

SSATranslator::~SSATranslator() {
  std::cerr << "SSATranslator destructor called - blocks: " << blocks.size() 
            << " phis: " << phis.size() << std::endl;
}

SSAValue SSATranslator::translate(const ASTNode* ast) {
  if (!ast) {
    return SSAValue::undefined();
  }
  
  // Debug: warn if trying to translate when current_block is invalid
  if (current_block.id == static_cast<size_t>(-1)) {
    std::cerr << "WARNING: translate called with invalid current_block (-1) for AST type: " << static_cast<int>(ast->type) << std::endl;
  }
  
  switch (ast->type) {
    case ASTNodeType::NumberLiteral:
      return translate_literal(ast);
    case ASTNodeType::BoolLiteral:
      return translate_literal(ast);
    case ASTNodeType::Identifier:
      return translate_identifier(ast);
    case ASTNodeType::BinaryExpression:
      return translate_binary_expression(ast);
    case ASTNodeType::UnaryExpression:
      return translate_unary_expression(ast);
    case ASTNodeType::LetStatement:
    case ASTNodeType::MutableLetStatement:
    case ASTNodeType::AssignmentStatement:
      return translate_assignment(ast);
    case ASTNodeType::IfStatement:
      return translate_if_statement(ast);
    case ASTNodeType::LoopStatement:
      return translate_loop_statement(ast);
    case ASTNodeType::Block:
      return translate_block(ast);
    case ASTNodeType::ExpressionStatement:
      return translate_expression_statement(ast);
    case ASTNodeType::Program:
      return translate_block(ast);
    case ASTNodeType::BreakStatement:
      return translate_break_statement(ast);
    case ASTNodeType::FunctionCall:
      return translate_function_call(ast);
    case ASTNodeType::ListLiteral:
      return translate_list_literal(ast);
    case ASTNodeType::TupleLiteral:
      return translate_tuple_literal(ast);
    case ASTNodeType::FunctionDeclaration:
      return translate_function_declaration(ast);
    case ASTNodeType::Parameter:
      // Parameters are handled in function declaration, return undefined
      return SSAValue::undefined();
    default:
      std::cerr << "Unsupported AST node type for SSA translation: " << static_cast<int>(ast->type) << "\n";
      return SSAValue::undefined();
  }
}

void SSATranslator::write_variable(const std::string& variable, BlockId block_id, const SSAValue& value) {
  definitions[variable][block_id] = value;
}

SSAValue SSATranslator::read_variable(const std::string& variable, BlockId block_id) {
  auto var_it = definitions.find(variable);
  if (var_it != definitions.end()) {
    auto block_it = var_it->second.find(block_id);
    if (block_it != var_it->second.end()) {
      return block_it->second;
    }
    
    // TODO: Add function parameter lookup without causing infinite recursion
  }
  return read_variable_recursively(variable, block_id);
}

SSAValue SSATranslator::read_variable_recursively(const std::string& variable, BlockId block_id) {
  // Safety check: don't process invalid block IDs
  if (block_id.id == static_cast<size_t>(-1) || block_id.id >= blocks.size()) {
    std::cerr << "ERROR: read_variable_recursively called with invalid block_id: " << block_id.id << std::endl;
    return SSAValue::undefined();
  }
  
  SSAValue value;
  
  if (sealed_blocks.find(block_id) == sealed_blocks.end()) {
    // Block is not sealed yet - create an incomplete phi
    PhiId phi_id = create_phi(block_id);
    Variable phi_var = get_temp_variable("phi");
    value = SSAValue::var(phi_var);
    incomplete_phis[block_id][variable] = phi_id;
    
    // Add phi instruction to the block
    blocks[block_id.id].instructions.insert(
      blocks[block_id.id].instructions.begin(),
      SSAInstruction::phi(phi_var, phi_id)
    );
  } else {
    const SSABlock& block = blocks[block_id.id];
    if (block.predecessors.size() == 1) {
      value = read_variable(variable, block.predecessors[0]);
    } else {
      PhiId phi_id = create_phi(block_id);
      Variable phi_var = get_temp_variable("phi");
      SSAValue val = SSAValue::var(phi_var);
      write_variable(variable, block_id, val);
      
      // Add phi instruction to the block
      blocks[block_id.id].instructions.insert(
        blocks[block_id.id].instructions.begin(),
        SSAInstruction::phi(phi_var, phi_id)
      );
      
      // Only add phi operands immediately if block is sealed
      add_phi_operands(variable, phi_id);
      value = val;
    }
  }
  
  write_variable(variable, block_id, value);
  return value;
}

SSAValue SSATranslator::add_phi_operands(const std::string& variable, PhiId phi_id) {
  BlockId block_id = phis[phi_id].block_id;
  
  // Key fix: Check if this phi already has operands to prevent infinite recursion
  if (!phis[phi_id].operands.empty()) {
    return SSAValue::undefined();
  }
  
  // Prevent infinite recursion in phi operand resolution
  if (resolving_phi_operands.count(phi_id)) {
    // We're already resolving this phi - return undefined to break the cycle
    return SSAValue::undefined();
  }
  
  resolving_phi_operands.insert(phi_id);
  
  const SSABlock& block = blocks[block_id.id];
  
  for (BlockId predecessor : block.predecessors) {
    SSAValue value = read_variable(variable, predecessor);
    phis[phi_id].operands.push_back(value);
    
    if (value.is_phi()) {
      add_phi_phi_use(value.phi_id, phi_id);
    }
  }
  
  resolving_phi_operands.erase(phi_id);
  
  return try_remove_trivial_phi(phi_id);
}

SSAValue SSATranslator::try_remove_trivial_phi(PhiId phi_id) {
  SSAValue same = SSAValue::undefined();
  bool same_set = false;
  const SSAPhi& phi = phis[phi_id];
  
  for (const SSAValue& op : phi.operands) {
    if (op.is_same_phi(phi_id)) {
      continue;
    }
    
    if (!same_set) {
      same = op;
      same_set = true;
    } else if (!(same.type == op.type && 
                ((same.type == SSAValueType::Literal && same.literal_value == op.literal_value) ||
                 (same.type == SSAValueType::Var && same.variable.name == op.variable.name) ||
                 (same.type == SSAValueType::Phi && same.phi_id.id == op.phi_id.id)))) {
      return SSAValue::phi(phi_id);
    }
  }
  
  SSAValue replacement = same_set ? same : SSAValue::undefined();
  
  replace_phi_uses(phi_id, replacement);
  
  for (const PhiReference& user : phi.uses) {
    if (user.type == PhiReferenceType::Phi) {
      try_remove_trivial_phi(user.phi_id);
    }
  }
  
  return replacement;
}

void SSATranslator::seal_block(BlockId block_id) {
  sealed_blocks.insert(block_id);
  blocks[block_id.id].seal();
  
  auto incomplete_it = incomplete_phis.find(block_id);
  if (incomplete_it != incomplete_phis.end()) {
    for (const auto& [variable, phi_id] : incomplete_it->second) {
      add_phi_operands(variable, phi_id);
    }
  }
}

Variable SSATranslator::get_temp_variable(const std::string& prefix) {
  return Variable::temp(next_variable_id++);
}

PhiId SSATranslator::create_phi(BlockId block_id) {
  // Safety check: don't create phis for invalid blocks
  if (block_id.id == static_cast<size_t>(-1) || block_id.id >= blocks.size()) {
    std::cerr << "ERROR: create_phi called with invalid block_id: " << block_id.id << std::endl;
    // Return a dummy phi ID to avoid crashes
    return PhiId(0);
  }
  
  PhiId phi_id(next_phi_id++);
  std::cerr << "Creating phi " << phi_id.id << " in block " << block_id.id << std::endl;
  phis[phi_id] = SSAPhi(phi_id, block_id);
  
  size_t instruction_offset = blocks[block_id.id].instructions.size();
  add_phi_use(phi_id, block_id, instruction_offset);
  
  return phi_id;
}

void SSATranslator::add_phi_use(PhiId phi_id, BlockId block_id, size_t instruction_offset) {
  phis[phi_id].uses.push_back(PhiReference::instruction(block_id, instruction_offset));
}

void SSATranslator::add_phi_phi_use(PhiId phi_id, PhiId user_phi_id) {
  phis[phi_id].uses.push_back(PhiReference::phi(user_phi_id));
}

void SSATranslator::replace_phi_uses(PhiId phi_id, const SSAValue& replacement) {
  auto phi_it = phis.find(phi_id);
  if (phi_it == phis.end()) return;
  
  const SSAPhi& phi = phi_it->second;
  for (const PhiReference& phi_ref : phi.uses) {
    switch (phi_ref.type) {
      case PhiReferenceType::Instruction:
        replace_value_at_location(phi_ref.block_id, phi_ref.instruction_offset, phi_id, replacement);
        break;
      case PhiReferenceType::Phi: {
        auto other_phi_it = phis.find(phi_ref.phi_id);
        if (other_phi_it != phis.end()) {
          for (SSAValue& operand : other_phi_it->second.operands) {
            if (operand.is_same_phi(phi_id)) {
              operand = replacement;
            }
          }
        }
        break;
      }
    }
  }
}

void SSATranslator::replace_value_at_location(BlockId block_id, size_t instruction_offset,
                                             PhiId old_phi_id, const SSAValue& new_value) {
  if (block_id.id < blocks.size() && instruction_offset < blocks[block_id.id].instructions.size()) {
    replace_value_in_instruction(blocks[block_id.id].instructions[instruction_offset], old_phi_id, new_value);
  }
  
  for (auto& [phi_id, phi] : phis) {
    for (SSAValue& operand : phi.operands) {
      if (operand.is_same_phi(old_phi_id)) {
        operand = new_value;
      }
    }
  }
}

void SSATranslator::replace_value_in_instruction(SSAInstruction& instruction,
                                                PhiId old_phi_id, const SSAValue& new_value) {
  switch (instruction.type) {
    case SSAInstructionType::Assign:
      if (instruction.value.is_same_phi(old_phi_id)) {
        instruction.value = new_value;
      }
      break;
    case SSAInstructionType::BinaryOp:
      if (instruction.left.is_same_phi(old_phi_id)) {
        instruction.left = new_value;
      }
      if (instruction.right.is_same_phi(old_phi_id)) {
        instruction.right = new_value;
      }
      break;
    case SSAInstructionType::UnaryOp:
      if (instruction.operand.is_same_phi(old_phi_id)) {
        instruction.operand = new_value;
      }
      break;
    case SSAInstructionType::ConditionalJump:
      if (instruction.condition.is_same_phi(old_phi_id)) {
        instruction.condition = new_value;
      }
      break;
    case SSAInstructionType::Print:
      if (instruction.value.is_same_phi(old_phi_id)) {
        instruction.value = new_value;
      }
      break;
    default:
      break;
  }
}

void SSATranslator::remove_phi(PhiId phi_id) {
  std::cerr << "Removing phi " << phi_id.id << std::endl;
  phis.erase(phi_id);
  
  for (auto& [block_id, block_phis] : incomplete_phis) {
    for (auto it = block_phis.begin(); it != block_phis.end();) {
      if (it->second.id == phi_id.id) {
        it = block_phis.erase(it);
      } else {
        ++it;
      }
    }
  }
}

BlockId SSATranslator::create_block() {
  BlockId block_id(blocks.size());
  std::cerr << "Creating block " << block_id.id << std::endl;
  blocks.emplace_back(block_id);
  return block_id;
}

SSAValue SSATranslator::translate_literal(const ASTNode* node) {
  int value;
  if (node->type == ASTNodeType::NumberLiteral) {
    value = std::stoi(node->value);
  } else if (node->type == ASTNodeType::BoolLiteral) {
    value = (node->value == "true") ? 1 : 0;
  } else {
    throw std::runtime_error("Unsupported literal type");
  }
  
  Variable temp_var = get_temp_variable("lit");
  blocks[current_block.id].add_instruction(
    SSAInstruction::assign(temp_var, SSAValue::literal(value))
  );
  return SSAValue::var(temp_var);
}

SSAValue SSATranslator::translate_identifier(const ASTNode* node) {
  return read_variable(node->value, current_block);
}

SSAValue SSATranslator::translate_binary_expression(const ASTNode* node) {
  if (node->child_count() < 2) {
    return SSAValue::undefined();
  }
  
  SSAValue left_value = translate(node->child(0));
  SSAValue right_value = translate(node->child(1));
  
  Variable dest_var = get_temp_variable("binop");
  BinaryOperator op = convert_binary_operator(node->value);
  
  blocks[current_block.id].add_instruction(
    SSAInstruction::binary_op(dest_var, left_value, op, right_value)
  );
  
  return SSAValue::var(dest_var);
}

SSAValue SSATranslator::translate_unary_expression(const ASTNode* node) {
  if (node->child_count() < 1) {
    return SSAValue::undefined();
  }
  
  SSAValue operand_value = translate(node->child(0));
  Variable dest_var = get_temp_variable("unaryop");
  UnaryOperator op = convert_unary_operator(node->value);
  
  blocks[current_block.id].add_instruction(
    SSAInstruction::unary_op(dest_var, op, operand_value)
  );
  
  return SSAValue::var(dest_var);
}

SSAValue SSATranslator::translate_assignment(const ASTNode* node) {
  if (node->child_count() < 2) {
    return SSAValue::undefined();
  }
  
  std::string variable_name;
  if (node->type == ASTNodeType::AssignmentStatement) {
    variable_name = node->child(0)->value;
  } else {
    variable_name = node->child(0)->value;
  }
  
  SSAValue rhs = translate(node->child(1));
  write_variable(variable_name, current_block, rhs);
  
  return rhs;
}

SSAValue SSATranslator::translate_if_statement(const ASTNode* node) {
  if (node->child_count() < 2) {
    return SSAValue::undefined();
  }
  
  SSAValue condition_value = translate(node->child(0));
  
  BlockId then_block = create_block();
  BlockId else_block = create_block();
  
  blocks[then_block.id].add_predecessor(current_block);
  blocks[else_block.id].add_predecessor(current_block);
  
  blocks[current_block.id].add_instruction(
    SSAInstruction::conditional_jump(condition_value, then_block, else_block)
  );
  
  seal_block(current_block);
  
  BlockId merge_block = create_block();
  
  current_block = then_block;
  current_block_ended = false;
  translate(node->child(1));
  blocks[merge_block.id].add_predecessor(current_block);
  blocks[current_block.id].add_instruction(SSAInstruction::jump(merge_block));
  seal_block(then_block);
  
  current_block = else_block;
  current_block_ended = false;
  if (node->child_count() > 2) {
    translate(node->child(2));
  }
  blocks[merge_block.id].add_predecessor(current_block);
  blocks[current_block.id].add_instruction(SSAInstruction::jump(merge_block));
  seal_block(else_block);
  
  seal_block(merge_block);
  current_block = merge_block;
  current_block_ended = false;
  
  return SSAValue::undefined();
}

SSAValue SSATranslator::translate_loop_statement(const ASTNode* node) {
  if (node->child_count() < 1) {
    return SSAValue::undefined();
  }
  
  BlockId loop_start = create_block();
  BlockId loop_body = create_block();
  BlockId loop_end = create_block();
  
  // Add predecessor from entry to loop start
  blocks[loop_start.id].add_predecessor(current_block);
  
  // Jump to loop start
  blocks[current_block.id].add_instruction(SSAInstruction::jump(loop_start));
  
  // Seal the current block since all its successors are known
  seal_block(current_block);
  
  // Switch to loop start block
  current_block = loop_start;
  
  // Loop start jumps directly to loop body (infinite loop)
  blocks[current_block.id].add_instruction(SSAInstruction::jump(loop_body));
  
  // Add predecessors for loop body
  blocks[loop_body.id].add_predecessor(loop_start);
  
  // Don't seal loop_end yet - break statements will add it as predecessor
  
  // Process loop body
  current_block = loop_body;
  current_block_ended = false;
  
  // Set loop exit for break statements
  BlockId saved_loop_exit = current_loop_exit;
  current_loop_exit = loop_end;
  
  translate(node->child(0));
  
  // Restore previous loop exit
  current_loop_exit = saved_loop_exit;
  
  // If the loop body didn't end with a break, jump back to loop start
  if (!current_block_ended) {
    blocks[current_block.id].add_instruction(SSAInstruction::jump(loop_start));
    // Add the back-edge predecessor
    blocks[loop_start.id].add_predecessor(current_block);
  }
  
  // Critical: seal blocks in the correct order following Rust implementation
  // 1. Seal loop_body after processing it  
  seal_block(loop_body);
  // 2. Seal loop_start LAST, after the back-edge is added
  seal_block(loop_start);
  // 3. Seal loop_end after all break statements have been processed
  seal_block(loop_end);
  
  current_block = loop_end;
  current_block_ended = false;
  return SSAValue::undefined();
}

SSAValue SSATranslator::translate_block(const ASTNode* node) {
  SSAValue value = SSAValue::undefined();
  for (size_t i = 0; i < node->child_count(); ++i) {
    value = translate(node->child(i));
  }
  return value;
}

SSAValue SSATranslator::translate_expression_statement(const ASTNode* node) {
  if (node->child_count() > 0) {
    return translate(node->child(0));
  }
  return SSAValue::undefined();
}

BinaryOperator SSATranslator::convert_binary_operator(const std::string& op) {
  if (op == "+") return BinaryOperator::Add;
  if (op == "-") return BinaryOperator::Subtract;
  if (op == "*") return BinaryOperator::Multiply;
  if (op == "/") return BinaryOperator::Divide;
  if (op == "==") return BinaryOperator::Equal;
  if (op == "!=") return BinaryOperator::NotEqual;
  if (op == "<") return BinaryOperator::LessThan;
  if (op == "<=") return BinaryOperator::LessThanOrEqual;
  if (op == ">") return BinaryOperator::GreaterThan;
  if (op == ">=") return BinaryOperator::GreaterThanOrEqual;
  if (op == "||") return BinaryOperator::LogicalOr;
  if (op == "&&") return BinaryOperator::LogicalAnd;
  
  throw std::runtime_error("Unknown binary operator: " + op);
}

UnaryOperator SSATranslator::convert_unary_operator(const std::string& op) {
  if (op == "-") return UnaryOperator::Negate;
  if (op == "!") return UnaryOperator::Not;
  
  throw std::runtime_error("Unknown unary operator: " + op);
}

SSAValue SSATranslator::translate_break_statement(const ASTNode* node) {
  std::cerr << "Processing break statement, current_block: " << current_block.id << std::endl;
  if (current_loop_exit.id != -1) {
    blocks[current_block.id].add_instruction(SSAInstruction::jump(current_loop_exit));
    blocks[current_loop_exit.id].add_predecessor(current_block);
    // Mark current block as ended (no more instructions should be added)
    current_block_ended = true;
    std::cerr << "Break statement processed, current_block_ended = true" << std::endl;
  }
  return SSAValue::undefined();
}

SSAValue SSATranslator::translate_function_call(const ASTNode* node) {
  // For now, treat function calls as returning a placeholder value
  // In a full implementation, this would need proper function call handling
  Variable temp_var = get_temp_variable("call");
  blocks[current_block.id].add_instruction(
    SSAInstruction::assign(temp_var, SSAValue::literal(0))
  );
  return SSAValue::var(temp_var);
}

SSAValue SSATranslator::translate_list_literal(const ASTNode* node) {
  // Translate all elements first
  for (size_t i = 0; i < node->child_count(); ++i) {
    translate(node->child(i));
  }
  
  // For now, treat list literals as a placeholder value
  // In a full implementation, this would need proper list construction
  Variable temp_var = get_temp_variable("list");
  blocks[current_block.id].add_instruction(
    SSAInstruction::assign(temp_var, SSAValue::literal(0))
  );
  return SSAValue::var(temp_var);
}

SSAValue SSATranslator::translate_tuple_literal(const ASTNode* node) {
  // Translate all elements first
  for (size_t i = 0; i < node->child_count(); ++i) {
    translate(node->child(i));
  }
  
  // For now, treat tuple literals as a placeholder value
  // In a full implementation, this would need proper tuple construction
  Variable temp_var = get_temp_variable("tuple");
  blocks[current_block.id].add_instruction(
    SSAInstruction::assign(temp_var, SSAValue::literal(0))
  );
  return SSAValue::var(temp_var);
}

SSAValue SSATranslator::translate_function_declaration(const ASTNode* node) {
  // Save the current state
  auto saved_definitions = definitions;
  auto saved_sealed_blocks = sealed_blocks;
  auto saved_incomplete_phis = incomplete_phis;
  auto saved_phis = phis;
  auto saved_blocks = blocks;
  auto saved_next_variable_id = next_variable_id;
  auto saved_next_phi_id = next_phi_id;
  auto saved_current_block = current_block;
  
  // Reset state for new function
  definitions.clear();
  sealed_blocks.clear();
  incomplete_phis.clear();
  phis.clear();
  blocks.clear();
  next_variable_id = 0;
  next_phi_id = 0;
  
  // Create initial block for the function
  BlockId initial_block = create_block();
  current_block = initial_block;
  
  // Get function name
  std::string function_name = node->name;
  current_function = function_name;
  
  // Handle function parameters (they're nested under function_type)
  if (node->function_type && node->function_type->child_count() > 0) {
    // First child of function_type is the parameter list
    const ASTNode* param_list = node->function_type->child(0);
    if (param_list && param_list->type == ASTNodeType::Parameter) {
      // This is a parameter list, check its children for individual parameters
      for (size_t j = 0; j < param_list->child_count(); ++j) {
        const ASTNode* param = param_list->child(j);
        if (param && param->type == ASTNodeType::Parameter && !param->name.empty()) {
          // Use the actual parameter name as the SSA variable
          Variable param_var(param->name);
          SSAValue param_value = SSAValue::var(param_var);
          
          // Write parameter to current block (will be available during function translation)
          write_variable(param->name, current_block, param_value);
          
          // Don't add assignment instructions - parameters are predefined SSA values
          // They exist from function entry without explicit assignment
        }
      }
    }
  }
  
  // Translate function body
  if (node->body) {
    SSAValue body_result = translate(node->body.get());
    
    // If the function body produces a value, generate a return instruction
    if (body_result.type != SSAValueType::Undefined) {
      blocks[current_block.id].add_instruction(
        SSAInstruction::ret(body_result)
      );
    }
  }
  
  // Store the function SSA
  FunctionSSA func_ssa(function_name);
  func_ssa.blocks = blocks;
  func_ssa.phis = phis;
  func_ssa.sealed_blocks = sealed_blocks;
  functions.push_back(std::move(func_ssa));
  
  // Restore the previous state
  definitions = saved_definitions;
  sealed_blocks = saved_sealed_blocks;
  incomplete_phis = saved_incomplete_phis;
  phis = saved_phis;
  blocks = saved_blocks;
  next_variable_id = saved_next_variable_id;
  next_phi_id = saved_next_phi_id;
  current_block = saved_current_block;
  
  return SSAValue::undefined();
}