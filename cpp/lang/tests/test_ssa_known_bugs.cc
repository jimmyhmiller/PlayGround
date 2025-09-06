#include "../src/ssa_translator.h"
#include "../src/ast.h"
#include "../src/reader.h"
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

void test_segfault_protection() {
  std::cout << "=== Testing Segfault Protection ===\n";
  
  std::vector<std::string> problematic_cases = {
    "fn id : (x: t) -> t { x }",
    "let x = 10; loop { if x > 5 { break; } x = x + 1; }",
    "fn test : () -> i32 { loop { break; } }",
    "let a = 1; if a > 0 { a = 2; } a"
  };
  
  for (const auto& test_case : problematic_cases) {
    std::cout << "Testing: " << test_case << "\n";
    
    try {
      Reader reader(test_case);
      reader.read();
      
      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();
      
      SSATranslator translator;
      auto result = translator.translate(ast.get());
      
      std::cout << "  ✅ No segfault (translates successfully)\n";
      
    } catch (const std::runtime_error& e) {
      std::cout << "  ❌ Runtime error (but no segfault): " << e.what() << "\n";
    } catch (const std::exception& e) {
      std::cout << "  ❌ Exception (but no segfault): " << e.what() << "\n";
    } catch (...) {
      std::cout << "  ❌ Unknown exception (but no segfault)\n";
    }
  }
  std::cout << "\n";
}

void test_phi_function_parameter_resolution() {
  std::cout << "=== Testing Phi Function Parameter Resolution ===\n";
  
  std::vector<std::string> phi_cases = {
    "fn test : (x: t) -> t { if true { x = 10; } x }",
    "fn merge : (a: i32, b: i32) -> i32 { if a > b { a = a + 1; } else { b = b + 1; } a + b }",
    "let x = 5; if x > 3 { x = 10; } else { x = 20; } x"
  };
  
  for (const auto& test_case : phi_cases) {
    std::cout << "Testing: " << test_case << "\n";
    
    try {
      Reader reader(test_case);
      reader.read();
      
      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();
      
      SSATranslator translator;
      translator.translate(ast.get());
      
      const auto& phis = translator.get_phis();
      bool has_undefined_phi = false;
      
      for (const auto& phi_pair : phis) {
        const SSAPhi& phi = phi_pair.second;
        for (const auto& operand : phi.operands) {
          if (operand.type == SSAValueType::Undefined) {
            std::cout << "  ❌ Found phi with undefined operand (φ(?))\n";
            has_undefined_phi = true;
          }
        }
      }
      
      if (!has_undefined_phi && !phis.empty()) {
        std::cout << "  ✅ All phi operands are properly resolved\n";
      } else if (phis.empty()) {
        std::cout << "  ⚠️ No phi functions generated\n";
      }
      
    } catch (const std::exception& e) {
      std::cout << "  ❌ Error: " << e.what() << "\n";
    }
  }
  std::cout << "\n";
}

void test_single_assignment_violations() {
  std::cout << "=== Testing Single Assignment Property Violations ===\n";
  
  std::vector<std::pair<std::string, bool>> cases = {
    {"let x = 10; x = 20; x", false},  // Should be invalid - reassignment
    {"let x = 10; let x = 20; x", true},  // Should be valid - different variables in SSA
    {"let x = 10; { let x = 20; } x", true},  // Should be valid - different scopes
    {"fn test : (x: t) -> t { x = 10; x }", false}  // Should be invalid - parameter reassignment
  };
  
  for (const auto& test_case : cases) {
    std::cout << "Testing: " << test_case.first << " (expect " << (test_case.second ? "valid" : "invalid") << ")\n";
    
    try {
      Reader reader(test_case.first);
      reader.read();
      
      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();
      
      SSATranslator translator;
      translator.translate(ast.get());
      
      const auto& blocks = translator.get_blocks();
      const auto& functions = translator.get_functions();
      
      // Check for variable reassignment violations
      std::set<std::string> assigned_vars;
      bool has_violation = false;
      
      // Check main program blocks
      for (const auto& block : blocks) {
        for (const auto& instr : block.instructions) {
          if (instr.type == SSAInstructionType::Assign ||
              instr.type == SSAInstructionType::BinaryOp ||
              instr.type == SSAInstructionType::UnaryOp) {
            
            if (assigned_vars.count(instr.dest.name)) {
              has_violation = true;
              break;
            }
            assigned_vars.insert(instr.dest.name);
          }
        }
      }
      
      // Check function blocks  
      for (const auto& func : functions) {
        assigned_vars.clear();  // Reset for each function
        for (const auto& block : func.blocks) {
          for (const auto& instr : block.instructions) {
            if (instr.type == SSAInstructionType::Assign ||
                instr.type == SSAInstructionType::BinaryOp ||
                instr.type == SSAInstructionType::UnaryOp) {
              
              if (assigned_vars.count(instr.dest.name)) {
                has_violation = true;
                break;
              }
              assigned_vars.insert(instr.dest.name);
            }
          }
        }
      }
      
      bool is_valid = !has_violation;
      if (is_valid == test_case.second) {
        std::cout << "  ✅ " << (is_valid ? "Valid" : "Invalid") << " as expected\n";
      } else {
        std::cout << "  ❌ Expected " << (test_case.second ? "valid" : "invalid") 
                 << " but got " << (is_valid ? "valid" : "invalid") << "\n";
      }
      
    } catch (const std::exception& e) {
      std::cout << "  ❌ Error: " << e.what() << "\n";
    }
  }
  std::cout << "\n";
}

void test_loop_structure_correctness() {
  std::cout << "=== Testing Loop Structure Correctness ===\n";
  
  std::vector<std::string> loop_cases = {
    "loop { break; }",
    "let i = 0; loop { if i > 10 { break; } i = i + 1; }",
    "loop { loop { break; } break; }"  // Nested loops
  };
  
  for (const auto& test_case : loop_cases) {
    std::cout << "Testing: " << test_case << "\n";
    
    try {
      Reader reader(test_case);
      reader.read();
      
      ASTBuilder builder(reader.root.children);
      auto ast = builder.build();
      
      SSATranslator translator;
      translator.translate(ast.get());
      
      const auto& blocks = translator.get_blocks();
      
      // Check for proper loop structure (no fake conditions)
      bool has_fake_condition = false;
      for (const auto& block : blocks) {
        for (const auto& instr : block.instructions) {
          // Look for assignments like "v1 := 1" followed by conditional jump
          if (instr.type == SSAInstructionType::Assign &&
              instr.value.type == SSAValueType::Literal &&
              instr.value.literal_value == 1) {
            
            // Check if this is followed by a conditional jump using the same variable
            for (size_t i = 0; i < block.instructions.size() - 1; i++) {
              if (&block.instructions[i] == &instr) {
                const auto& next_instr = block.instructions[i + 1];
                if (next_instr.type == SSAInstructionType::ConditionalJump &&
                    next_instr.condition.type == SSAValueType::Var &&
                    next_instr.condition.variable.name == instr.dest.name) {
                  has_fake_condition = true;
                }
                break;
              }
            }
          }
        }
      }
      
      if (!has_fake_condition) {
        std::cout << "  ✅ No fake loop conditions found\n";
      } else {
        std::cout << "  ❌ Found fake loop condition (v1 := 1; if v1 then...)\n";
      }
      
    } catch (const std::exception& e) {
      std::cout << "  ❌ Error: " << e.what() << "\n";
    }
  }
  std::cout << "\n";
}

void test_function_isolation() {
  std::cout << "=== Testing Multi-Function Variable Isolation ===\n";
  
  std::string multi_function_case = R"(
    fn first : (x: i32) -> i32 { let y = x + 1; y }
    fn second : (x: i32) -> i32 { let y = x * 2; y }
  )";
  
  std::cout << "Testing multi-function variable isolation\n";
  
  try {
    Reader reader(multi_function_case);
    reader.read();
    
    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();
    
    SSATranslator translator;
    translator.translate(ast.get());
    
    const auto& functions = translator.get_functions();
    
    if (functions.size() >= 2) {
      // Check that variables don't leak between functions
      std::set<std::string> first_func_vars;
      std::set<std::string> second_func_vars;
      
      // Collect variables from first function
      for (const auto& block : functions[0].blocks) {
        for (const auto& instr : block.instructions) {
          if (instr.type == SSAInstructionType::Assign ||
              instr.type == SSAInstructionType::BinaryOp ||
              instr.type == SSAInstructionType::UnaryOp) {
            first_func_vars.insert(instr.dest.name);
          }
        }
      }
      
      // Collect variables from second function
      for (const auto& block : functions[1].blocks) {
        for (const auto& instr : block.instructions) {
          if (instr.type == SSAInstructionType::Assign ||
              instr.type == SSAInstructionType::BinaryOp ||
              instr.type == SSAInstructionType::UnaryOp) {
            second_func_vars.insert(instr.dest.name);
          }
        }
      }
      
      // Check for overlapping temp variables (indicating leakage)
      bool has_leakage = false;
      for (const auto& var : first_func_vars) {
        if (var.starts_with("v") && second_func_vars.count(var)) {
          has_leakage = true;
          std::cout << "  ❌ Variable " << var << " appears in both functions\n";
        }
      }
      
      if (!has_leakage) {
        std::cout << "  ✅ Functions properly isolated\n";
      }
      
    } else {
      std::cout << "  ⚠️ Expected 2 functions but got " << functions.size() << "\n";
    }
    
  } catch (const std::exception& e) {
    std::cout << "  ❌ Error: " << e.what() << "\n";
  }
  
  std::cout << "\n";
}

int main() {
  std::cout << "Running SSA Known Bugs Tests...\n\n";
  
  try {
    test_segfault_protection();
    test_phi_function_parameter_resolution();
    test_single_assignment_violations();
    test_loop_structure_correctness();
    test_function_isolation();
    
    std::cout << "SSA known bugs tests completed!\n";
  } catch (const std::exception &e) {
    std::cerr << "Test failed with error: " << e.what() << "\n";
    return 1;
  }
  
  return 0;
}