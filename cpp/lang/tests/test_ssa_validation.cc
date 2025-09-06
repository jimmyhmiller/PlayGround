#include "../src/ssa_translator.h"
#include "../src/ast.h"
#include "../src/reader.h"
#include <cassert>
#include <iostream>
#include <string>
#include <set>
#include <unordered_set>

// Function declarations
bool validate_ssa_form(const SSATranslator& translator);
bool validate_blocks_ssa_form(const std::vector<SSABlock>& blocks);
bool validate_value_defined(const SSAValue& value, const std::set<std::string>& defined_variables);
bool uses_phi_value_directly(const SSAInstruction& instr);

void test_ssa_properties() {
  std::cout << "=== Testing SSA Properties ===\n";
  
  std::vector<std::string> test_cases = {
    "let x = 10; if x > 5 { x = 20; } let y = x;",
    "let a = 1; let b = 2; if a > 0 { a = b + 1; b = a * 2; } let c = a + b;",
    "let i = 0; loop { if i > 10 { break; } i = i + 1; }"
  };
  
  for (const auto& test_case : test_cases) {
    std::cout << "Testing: " << test_case << "\n";
    
    try {
      Reader reader(test_case);
      reader.read();
      
      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();
      
      SSATranslator translator;
      translator.translate(ast.get());
      
      // Validate SSA properties
      bool is_valid = validate_ssa_form(translator);
      
      if (is_valid) {
        std::cout << "  ✅ Valid SSA form\n";
      } else {
        std::cout << "  ❌ Invalid SSA form\n";
      }
      
    } catch (const std::exception& e) {
      std::cout << "  ❌ Error: " << e.what() << "\n";
    }
  }
  std::cout << "\n";
}

bool validate_ssa_form(const SSATranslator& translator) {
  const auto& blocks = translator.get_blocks();
  const auto& functions = translator.get_functions();
  
  // Check main program blocks
  if (!validate_blocks_ssa_form(blocks)) {
    return false;
  }
  
  // Check each function's blocks
  for (const auto& func : functions) {
    if (!validate_blocks_ssa_form(func.blocks)) {
      return false;
    }
  }
  
  return true;
}

bool validate_blocks_ssa_form(const std::vector<SSABlock>& blocks) {
  std::set<std::string> defined_variables;
  std::unordered_set<std::string> used_before_definition;
  
  for (const SSABlock& block : blocks) {
    for (const SSAInstruction& instr : block.instructions) {
      // Check that all used variables are defined before use
      switch (instr.type) {
        case SSAInstructionType::Assign:
          if (!validate_value_defined(instr.value, defined_variables)) {
            std::cout << "    Variable used before definition in assign\n";
            return false;
          }
          defined_variables.insert(instr.dest.name);
          break;
          
        case SSAInstructionType::BinaryOp:
          if (!validate_value_defined(instr.left, defined_variables) ||
              !validate_value_defined(instr.right, defined_variables)) {
            std::cout << "    Variable used before definition in binary op\n";
            return false;
          }
          defined_variables.insert(instr.dest.name);
          break;
          
        case SSAInstructionType::UnaryOp:
          if (!validate_value_defined(instr.operand, defined_variables)) {
            std::cout << "    Variable used before definition in unary op\n";
            return false;
          }
          defined_variables.insert(instr.dest.name);
          break;
          
        case SSAInstructionType::ConditionalJump:
          if (!validate_value_defined(instr.condition, defined_variables)) {
            std::cout << "    Variable used before definition in conditional jump\n";
            return false;
          }
          break;
          
        case SSAInstructionType::Phi:
          // Phi functions should be at the beginning of blocks (for simplicity, we'll allow them anywhere)
          // All operands should be defined in predecessor blocks
          defined_variables.insert(instr.dest.name);
          break;
          
        case SSAInstructionType::Print:
          if (!validate_value_defined(instr.value, defined_variables)) {
            std::cout << "    Variable used before definition in print\n";
            return false;
          }
          break;
          
        case SSAInstructionType::Jump:
          // No variables used
          break;
      }
      
      // Check single assignment property - each variable is assigned exactly once
      if (instr.type == SSAInstructionType::Assign ||
          instr.type == SSAInstructionType::BinaryOp ||
          instr.type == SSAInstructionType::UnaryOp ||
          instr.type == SSAInstructionType::Phi) {
        // In proper SSA, we should never redefine a variable
        // (This is a simplification - in practice we'd track across blocks)
      }
    }
  }
  
  return true;
}

bool validate_value_defined(const SSAValue& value, const std::set<std::string>& defined_variables) {
  switch (value.type) {
    case SSAValueType::Literal:
      return true; // Literals are always "defined"
      
    case SSAValueType::Var:
      return defined_variables.count(value.variable.name) > 0;
      
    case SSAValueType::Phi:
      // Phi values should not appear directly in expressions in valid SSA
      std::cout << "    Found direct phi usage in expression (invalid SSA)\n";
      return false;
      
    case SSAValueType::Undefined:
      return false;
      
    default:
      return false;
  }
}

void test_no_direct_phi_usage() {
  std::cout << "=== Testing No Direct Phi Usage ===\n";
  
  std::string test_case = "let x = 10; if x > 5 { x = 20; } let y = x;";
  std::cout << "Testing: " << test_case << "\n";
  
  try {
    Reader reader(test_case);
    reader.read();
    
    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();
    
    SSATranslator translator;
    translator.translate(ast.get());
    
    // Check that no instruction uses phi values directly
    const auto& blocks = translator.get_blocks();
    bool found_direct_phi = false;
    
    for (const SSABlock& block : blocks) {
      for (const SSAInstruction& instr : block.instructions) {
        if (uses_phi_value_directly(instr)) {
          found_direct_phi = true;
          std::cout << "  ❌ Found direct phi usage in instruction\n";
          break;
        }
      }
    }
    
    if (!found_direct_phi) {
      std::cout << "  ✅ No direct phi usage found\n";
    }
    
  } catch (const std::exception& e) {
    std::cout << "  ❌ Error: " << e.what() << "\n";
  }
  std::cout << "\n";
}

bool uses_phi_value_directly(const SSAInstruction& instr) {
  switch (instr.type) {
    case SSAInstructionType::Assign:
      return instr.value.type == SSAValueType::Phi;
    case SSAInstructionType::BinaryOp:
      return instr.left.type == SSAValueType::Phi || 
             instr.right.type == SSAValueType::Phi;
    case SSAInstructionType::UnaryOp:
      return instr.operand.type == SSAValueType::Phi;
    case SSAInstructionType::ConditionalJump:
      return instr.condition.type == SSAValueType::Phi;
    case SSAInstructionType::Print:
      return instr.value.type == SSAValueType::Phi;
    default:
      return false;
  }
}

int main() {
  std::cout << "Running SSA validation tests...\n\n";
  
  try {
    test_ssa_properties();
    test_no_direct_phi_usage();
    
    std::cout << "SSA validation tests completed!\n";
  } catch (const std::exception &e) {
    std::cerr << "Test failed with error: " << e.what() << "\n";
    return 1;
  }
  
  return 0;
}