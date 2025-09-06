#include "../src/ssa_translator.h"
#include "../src/ast.h"
#include "../src/reader.h"
#include <cassert>
#include <iostream>
#include <string>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <queue>

struct DominatorAnalysis {
  std::unordered_map<size_t, std::unordered_set<size_t>> dominators;
  std::unordered_map<size_t, std::unordered_set<size_t>> predecessors;
  std::unordered_map<size_t, std::unordered_set<size_t>> successors;
  
  void compute_cfg(const std::vector<SSABlock>& blocks);
  void compute_dominators(const std::vector<SSABlock>& blocks);
  bool dominates(size_t block_a, size_t block_b) const;
};

struct SSAValidationResult {
  bool is_valid;
  std::vector<std::string> errors;
  
  void add_error(const std::string& error) {
    is_valid = false;
    errors.push_back(error);
  }
};

class SSAFormalPropertiesValidator {
public:
  SSAValidationResult validate_all_properties(const SSATranslator& translator);
  
private:
  SSAValidationResult validate_single_assignment_property(const std::vector<SSABlock>& blocks);
  SSAValidationResult validate_dominance_property(const std::vector<SSABlock>& blocks);
  SSAValidationResult validate_phi_placement(const std::vector<SSABlock>& blocks, 
                                            const std::unordered_map<PhiId, SSAPhi, PhiIdHash>& phis);
  SSAValidationResult validate_cfg_integrity(const std::vector<SSABlock>& blocks);
  SSAValidationResult validate_variable_liveness(const std::vector<SSABlock>& blocks);
  
  std::unordered_set<std::string> collect_all_variables(const std::vector<SSABlock>& blocks);
  std::unordered_map<std::string, std::unordered_set<size_t>> get_variable_definitions(const std::vector<SSABlock>& blocks);
  std::unordered_map<std::string, std::unordered_set<size_t>> get_variable_uses(const std::vector<SSABlock>& blocks);
  void collect_instruction_variables(const SSAInstruction& instr, 
                                   std::unordered_set<std::string>& defined,
                                   std::unordered_set<std::string>& used);
};

void DominatorAnalysis::compute_cfg(const std::vector<SSABlock>& blocks) {
  for (const auto& block : blocks) {
    size_t block_id = block.id.id;
    
    for (const auto& pred : block.predecessors) {
      predecessors[block_id].insert(pred.id);
      successors[pred.id].insert(block_id);
    }
    
    for (const auto& instr : block.instructions) {
      if (instr.type == SSAInstructionType::Jump) {
        successors[block_id].insert(instr.target.id);
        predecessors[instr.target.id].insert(block_id);
      } else if (instr.type == SSAInstructionType::ConditionalJump) {
        successors[block_id].insert(instr.true_target.id);
        successors[block_id].insert(instr.false_target.id);
        predecessors[instr.true_target.id].insert(block_id);
        predecessors[instr.false_target.id].insert(block_id);
      }
    }
  }
}

void DominatorAnalysis::compute_dominators(const std::vector<SSABlock>& blocks) {
  if (blocks.empty()) return;
  
  compute_cfg(blocks);
  
  size_t entry_block = 0;
  std::unordered_set<size_t> all_blocks;
  for (const auto& block : blocks) {
    all_blocks.insert(block.id.id);
  }
  
  dominators[entry_block].insert(entry_block);
  
  for (const auto& block : blocks) {
    size_t block_id = block.id.id;
    if (block_id != entry_block) {
      dominators[block_id] = all_blocks;
    }
  }
  
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& block : blocks) {
      size_t block_id = block.id.id;
      if (block_id == entry_block) continue;
      
      std::unordered_set<size_t> new_dominators = all_blocks;
      
      for (size_t pred : predecessors[block_id]) {
        std::unordered_set<size_t> intersection;
        for (size_t dom : dominators[pred]) {
          if (new_dominators.count(dom)) {
            intersection.insert(dom);
          }
        }
        new_dominators = intersection;
      }
      new_dominators.insert(block_id);
      
      if (new_dominators != dominators[block_id]) {
        dominators[block_id] = new_dominators;
        changed = true;
      }
    }
  }
}

bool DominatorAnalysis::dominates(size_t block_a, size_t block_b) const {
  auto it = dominators.find(block_b);
  return it != dominators.end() && it->second.count(block_a) > 0;
}

SSAValidationResult SSAFormalPropertiesValidator::validate_all_properties(const SSATranslator& translator) {
  SSAValidationResult result;
  result.is_valid = true;
  
  const auto& blocks = translator.get_blocks();
  const auto& phis = translator.get_phis();
  const auto& functions = translator.get_functions();
  
  auto single_assignment_result = validate_single_assignment_property(blocks);
  if (!single_assignment_result.is_valid) {
    result.add_error("Single Assignment Property violated in main blocks");
    for (const auto& error : single_assignment_result.errors) {
      result.errors.push_back("  " + error);
    }
  }
  
  auto dominance_result = validate_dominance_property(blocks);
  if (!dominance_result.is_valid) {
    result.add_error("Dominance Property violated in main blocks");
    for (const auto& error : dominance_result.errors) {
      result.errors.push_back("  " + error);
    }
  }
  
  auto phi_result = validate_phi_placement(blocks, phis);
  if (!phi_result.is_valid) {
    result.add_error("Phi Placement Property violated in main blocks");
    for (const auto& error : phi_result.errors) {
      result.errors.push_back("  " + error);
    }
  }
  
  auto cfg_result = validate_cfg_integrity(blocks);
  if (!cfg_result.is_valid) {
    result.add_error("Control Flow Graph integrity violated in main blocks");
    for (const auto& error : cfg_result.errors) {
      result.errors.push_back("  " + error);
    }
  }
  
  auto liveness_result = validate_variable_liveness(blocks);
  if (!liveness_result.is_valid) {
    result.add_error("Variable Liveness property violated in main blocks");
    for (const auto& error : liveness_result.errors) {
      result.errors.push_back("  " + error);
    }
  }
  
  for (const auto& func : functions) {
    auto func_single_assignment = validate_single_assignment_property(func.blocks);
    if (!func_single_assignment.is_valid) {
      result.add_error("Single Assignment Property violated in function: " + func.function_name);
    }
    
    auto func_dominance = validate_dominance_property(func.blocks);
    if (!func_dominance.is_valid) {
      result.add_error("Dominance Property violated in function: " + func.function_name);
    }
    
    auto func_phi = validate_phi_placement(func.blocks, func.phis);
    if (!func_phi.is_valid) {
      result.add_error("Phi Placement violated in function: " + func.function_name);
    }
  }
  
  return result;
}

SSAValidationResult SSAFormalPropertiesValidator::validate_single_assignment_property(const std::vector<SSABlock>& blocks) {
  SSAValidationResult result;
  result.is_valid = true;
  
  std::unordered_set<std::string> assigned_variables;
  
  for (const auto& block : blocks) {
    for (const auto& instr : block.instructions) {
      if (instr.type == SSAInstructionType::Assign ||
          instr.type == SSAInstructionType::BinaryOp ||
          instr.type == SSAInstructionType::UnaryOp ||
          instr.type == SSAInstructionType::Phi) {
        
        std::string var_name = instr.dest.name;
        if (assigned_variables.count(var_name)) {
          result.add_error("Variable '" + var_name + "' assigned more than once (violates single assignment)");
        } else {
          assigned_variables.insert(var_name);
        }
      }
    }
  }
  
  return result;
}

SSAValidationResult SSAFormalPropertiesValidator::validate_dominance_property(const std::vector<SSABlock>& blocks) {
  SSAValidationResult result;
  result.is_valid = true;
  
  if (blocks.empty()) return result;
  
  DominatorAnalysis dom_analysis;
  dom_analysis.compute_dominators(blocks);
  
  auto definitions = get_variable_definitions(blocks);
  auto uses = get_variable_uses(blocks);
  
  for (const auto& var_uses : uses) {
    const std::string& variable = var_uses.first;
    const auto& use_blocks = var_uses.second;
    
    auto def_it = definitions.find(variable);
    if (def_it == definitions.end()) {
      result.add_error("Variable '" + variable + "' used but never defined");
      continue;
    }
    
    const auto& def_blocks = def_it->second;
    
    for (size_t use_block : use_blocks) {
      bool dominated_by_some_def = false;
      for (size_t def_block : def_blocks) {
        if (dom_analysis.dominates(def_block, use_block) || def_block == use_block) {
          dominated_by_some_def = true;
          break;
        }
      }
      
      if (!dominated_by_some_def) {
        result.add_error("Variable '" + variable + "' used in block " + std::to_string(use_block) + 
                        " but not dominated by any definition");
      }
    }
  }
  
  return result;
}

SSAValidationResult SSAFormalPropertiesValidator::validate_phi_placement(const std::vector<SSABlock>& blocks,
                                                                        const std::unordered_map<PhiId, SSAPhi, PhiIdHash>& phis) {
  SSAValidationResult result;
  result.is_valid = true;
  
  DominatorAnalysis dom_analysis;
  dom_analysis.compute_dominators(blocks);
  
  for (const auto& phi_pair : phis) {
    const SSAPhi& phi = phi_pair.second;
    size_t phi_block = phi.block_id.id;
    
    if (dom_analysis.predecessors[phi_block].size() < 2) {
      result.add_error("Phi function in block " + std::to_string(phi_block) + 
                      " has fewer than 2 predecessors (unnecessary phi)");
    }
    
    if (phi.operands.size() != dom_analysis.predecessors[phi_block].size()) {
      result.add_error("Phi function in block " + std::to_string(phi_block) + 
                      " operand count doesn't match predecessor count");
    }
    
    for (const auto& operand : phi.operands) {
      if (operand.type == SSAValueType::Phi) {
        result.add_error("Phi function contains another phi as operand (potential cycle)");
      }
    }
  }
  
  return result;
}

SSAValidationResult SSAFormalPropertiesValidator::validate_cfg_integrity(const std::vector<SSABlock>& blocks) {
  SSAValidationResult result;
  result.is_valid = true;
  
  std::unordered_set<size_t> block_ids;
  for (const auto& block : blocks) {
    block_ids.insert(block.id.id);
  }
  
  for (const auto& block : blocks) {
    for (const auto& pred : block.predecessors) {
      if (block_ids.find(pred.id) == block_ids.end()) {
        result.add_error("Block " + std::to_string(block.id.id) + 
                        " has predecessor " + std::to_string(pred.id) + " that doesn't exist");
      }
    }
    
    for (const auto& instr : block.instructions) {
      if (instr.type == SSAInstructionType::Jump) {
        if (block_ids.find(instr.target.id) == block_ids.end()) {
          result.add_error("Jump instruction targets non-existent block " + std::to_string(instr.target.id));
        }
      } else if (instr.type == SSAInstructionType::ConditionalJump) {
        if (block_ids.find(instr.true_target.id) == block_ids.end()) {
          result.add_error("Conditional jump true target " + std::to_string(instr.true_target.id) + " doesn't exist");
        }
        if (block_ids.find(instr.false_target.id) == block_ids.end()) {
          result.add_error("Conditional jump false target " + std::to_string(instr.false_target.id) + " doesn't exist");
        }
      }
    }
  }
  
  return result;
}

SSAValidationResult SSAFormalPropertiesValidator::validate_variable_liveness(const std::vector<SSABlock>& blocks) {
  SSAValidationResult result;
  result.is_valid = true;
  
  auto definitions = get_variable_definitions(blocks);
  auto uses = get_variable_uses(blocks);
  
  for (const auto& var_defs : definitions) {
    const std::string& variable = var_defs.first;
    if (uses.find(variable) == uses.end()) {
      result.add_error("Variable '" + variable + "' is defined but never used (dead code)");
    }
  }
  
  return result;
}

std::unordered_map<std::string, std::unordered_set<size_t>> SSAFormalPropertiesValidator::get_variable_definitions(const std::vector<SSABlock>& blocks) {
  std::unordered_map<std::string, std::unordered_set<size_t>> definitions;
  
  for (const auto& block : blocks) {
    for (const auto& instr : block.instructions) {
      if (instr.type == SSAInstructionType::Assign ||
          instr.type == SSAInstructionType::BinaryOp ||
          instr.type == SSAInstructionType::UnaryOp ||
          instr.type == SSAInstructionType::Phi) {
        definitions[instr.dest.name].insert(block.id.id);
      }
    }
  }
  
  return definitions;
}

std::unordered_map<std::string, std::unordered_set<size_t>> SSAFormalPropertiesValidator::get_variable_uses(const std::vector<SSABlock>& blocks) {
  std::unordered_map<std::string, std::unordered_set<size_t>> uses;
  
  for (const auto& block : blocks) {
    for (const auto& instr : block.instructions) {
      std::unordered_set<std::string> used_vars;
      std::unordered_set<std::string> defined_vars;
      collect_instruction_variables(instr, defined_vars, used_vars);
      
      for (const std::string& var : used_vars) {
        uses[var].insert(block.id.id);
      }
    }
  }
  
  return uses;
}

void SSAFormalPropertiesValidator::collect_instruction_variables(const SSAInstruction& instr,
                                                               std::unordered_set<std::string>& defined,
                                                               std::unordered_set<std::string>& used) {
  switch (instr.type) {
    case SSAInstructionType::Assign:
      defined.insert(instr.dest.name);
      if (instr.value.type == SSAValueType::Var) {
        used.insert(instr.value.variable.name);
      }
      break;
      
    case SSAInstructionType::BinaryOp:
      defined.insert(instr.dest.name);
      if (instr.left.type == SSAValueType::Var) {
        used.insert(instr.left.variable.name);
      }
      if (instr.right.type == SSAValueType::Var) {
        used.insert(instr.right.variable.name);
      }
      break;
      
    case SSAInstructionType::UnaryOp:
      defined.insert(instr.dest.name);
      if (instr.operand.type == SSAValueType::Var) {
        used.insert(instr.operand.variable.name);
      }
      break;
      
    case SSAInstructionType::ConditionalJump:
      if (instr.condition.type == SSAValueType::Var) {
        used.insert(instr.condition.variable.name);
      }
      break;
      
    case SSAInstructionType::Print:
      if (instr.value.type == SSAValueType::Var) {
        used.insert(instr.value.variable.name);
      }
      break;
      
    case SSAInstructionType::Return:
      if (instr.value.type == SSAValueType::Var) {
        used.insert(instr.value.variable.name);
      }
      break;
      
    case SSAInstructionType::Phi:
      defined.insert(instr.dest.name);
      break;
      
    case SSAInstructionType::Jump:
      break;
  }
}

void test_single_assignment_property() {
  std::cout << "=== Testing Single Assignment Property ===\n";
  
  std::vector<std::pair<std::string, bool>> test_cases = {
    {"let x = 10; let y = x;", true},
    {"let x = 10; if x > 5 { let x = 20; } let y = x;", true},  // Different x in SSA
    {"let a = 1; let b = 2; let c = a + b;", true},
    {"fn test : (x: t) -> t { let y = x; y }", true}
  };
  
  SSAFormalPropertiesValidator validator;
  
  for (const auto& test_case : test_cases) {
    std::cout << "Testing: " << test_case.first << "\n";
    
    try {
      Reader reader(test_case.first);
      reader.read();
      
      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();
      
      SSATranslator translator;
      translator.translate(ast.get());
      
      auto result = validator.validate_all_properties(translator);
      
      if (result.is_valid == test_case.second) {
        std::cout << "  ✅ " << (result.is_valid ? "Valid" : "Invalid") << " SSA (as expected)\n";
      } else {
        std::cout << "  ❌ Expected " << (test_case.second ? "valid" : "invalid") << " but got " << (result.is_valid ? "valid" : "invalid") << "\n";
        for (const auto& error : result.errors) {
          std::cout << "    " << error << "\n";
        }
      }
      
    } catch (const std::exception& e) {
      std::cout << "  ❌ Error: " << e.what() << "\n";
    }
  }
  std::cout << "\n";
}

void test_dominance_property() {
  std::cout << "=== Testing Dominance Property ===\n";
  
  std::vector<std::string> test_cases = {
    "let x = 10; let y = x;",
    "let x = 10; if x > 5 { let y = x; } let z = x;",
    "let i = 0; loop { if i > 10 { break; } let j = i; i = i + 1; }"
  };
  
  SSAFormalPropertiesValidator validator;
  
  for (const auto& test_case : test_cases) {
    std::cout << "Testing: " << test_case << "\n";
    
    try {
      Reader reader(test_case);
      reader.read();
      
      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();
      
      SSATranslator translator;
      translator.translate(ast.get());
      
      auto result = validator.validate_all_properties(translator);
      
      if (result.is_valid) {
        std::cout << "  ✅ Dominance property satisfied\n";
      } else {
        std::cout << "  ❌ Dominance property violated:\n";
        for (const auto& error : result.errors) {
          std::cout << "    " << error << "\n";
        }
      }
      
    } catch (const std::exception& e) {
      std::cout << "  ❌ Error: " << e.what() << "\n";
    }
  }
  std::cout << "\n";
}

void test_phi_function_properties() {
  std::cout << "=== Testing Phi Function Properties ===\n";
  
  std::vector<std::string> test_cases = {
    "let x = 10; if x > 5 { x = 20; } let y = x;",
    "let a = 1; let b = 2; if a > 0 { a = b + 1; b = a * 2; } let c = a + b;",
    "let i = 0; loop { if i > 10 { break; } i = i + 1; }"
  };
  
  SSAFormalPropertiesValidator validator;
  
  for (const auto& test_case : test_cases) {
    std::cout << "Testing: " << test_case << "\n";
    
    try {
      Reader reader(test_case);
      reader.read();
      
      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();
      
      SSATranslator translator;
      translator.translate(ast.get());
      
      auto result = validator.validate_all_properties(translator);
      
      if (result.is_valid) {
        std::cout << "  ✅ Phi functions correctly placed\n";
      } else {
        std::cout << "  ❌ Phi function issues:\n";
        for (const auto& error : result.errors) {
          std::cout << "    " << error << "\n";
        }
      }
      
    } catch (const std::exception& e) {
      std::cout << "  ❌ Error: " << e.what() << "\n";
    }
  }
  std::cout << "\n";
}

int main() {
  std::cout << "Running SSA Formal Properties Tests...\n\n";
  
  try {
    test_single_assignment_property();
    test_dominance_property();
    test_phi_function_properties();
    
    std::cout << "SSA formal properties tests completed!\n";
  } catch (const std::exception &e) {
    std::cerr << "Test failed with error: " << e.what() << "\n";
    return 1;
  }
  
  return 0;
}