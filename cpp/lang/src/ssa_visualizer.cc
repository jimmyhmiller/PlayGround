#include "ssa_visualizer.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

SSAVisualizer::SSAVisualizer(const SSATranslator* translator) : translator(translator) {}

std::string SSAVisualizer::generate_dot() const {
  std::ostringstream dot;
  
  dot << "digraph SSA {\n";
  dot << "    rankdir=TB;\n";
  dot << "    node [shape=box, style=rounded];\n";
  dot << "    \n";
  
  const auto& functions = translator->get_functions();
  const auto& main_blocks = translator->get_blocks();
  
  // Handle main program blocks (if any)
  if (!main_blocks.empty()) {
    dot << "    subgraph cluster_main {\n";
    dot << "        label=\"Main Program\";\n";
    dot << "        style=filled;\n";
    dot << "        fillcolor=lightgray;\n";
    dot << "        \n";
    
    for (const SSABlock& block : main_blocks) {
      dot << "    " << generate_block_node(block);
    }
    
    for (const SSABlock& block : main_blocks) {
      for (BlockId pred : block.predecessors) {
        dot << "        block_" << pred.id << " -> block_" << block.id.id << ";\n";
      }
    }
    
    dot << "    }\n";
  }
  
  // Handle function blocks
  for (const auto& func : functions) {
    dot << "    subgraph cluster_" << func.function_name << " {\n";
    dot << "        label=\"Function: " << func.function_name << "\";\n";
    dot << "        style=filled;\n";
    dot << "        fillcolor=lightblue;\n";
    dot << "        \n";
    
    for (const SSABlock& block : func.blocks) {
      dot << "    " << generate_block_node_for_function(block, func.function_name);
    }
    
    for (const SSABlock& block : func.blocks) {
      for (BlockId pred : block.predecessors) {
        dot << "        " << func.function_name << "_block_" << pred.id 
            << " -> " << func.function_name << "_block_" << block.id.id << ";\n";
      }
    }
    
    dot << "    }\n";
  }
  
  if (has_phi_nodes()) {
    dot << "\n    // Legend\n";
    dot << "    subgraph cluster_legend {\n";
    dot << "        label=\"Legend\";\n";
    dot << "        style=dotted;\n";
    dot << "        \"Φ = Phi function\" [shape=plaintext];\n";
    dot << "    }\n";
  }
  
  dot << "}\n";
  return dot.str();
}

std::string SSAVisualizer::generate_block_node(const SSABlock& block) const {
  std::ostringstream node;
  
  node << "    block_" << block.id.id << " [label=\"";
  node << "Block " << block.id.id << "\\n";
  node << "─────────────\\n";
  
  const auto& incomplete_phis = translator->get_incomplete_phis();
  auto incomplete_it = incomplete_phis.find(block.id);
  if (incomplete_it != incomplete_phis.end()) {
    const auto& phis_map = translator->get_phis();
    for (const auto& [var, phi_id] : incomplete_it->second) {
      auto phi_it = phis_map.find(phi_id);
      if (phi_it != phis_map.end()) {
        const SSAPhi& phi = phi_it->second;
        node << "Φ(" << escape_dot_string(var) << ") = ";
        
        bool first = true;
        for (const SSAValue& operand : phi.operands) {
          if (!first) node << ", ";
          node << format_value(operand);
          first = false;
        }
        node << "\\n";
      }
    }
    if (!incomplete_it->second.empty()) {
      node << "─────────────\\n";
    }
  }
  
  for (const SSAInstruction& instruction : block.instructions) {
    node << format_instruction(instruction) << "\\n";
  }
  
  const auto& sealed_blocks = translator->get_sealed_blocks();
  if (sealed_blocks.find(block.id) != sealed_blocks.end()) {
    node << "─────────────\\n";
    node << "[SEALED]\\n";
  }
  
  node << "\"];\n";
  return node.str();
}

std::string SSAVisualizer::generate_block_node_for_function(const SSABlock& block, const std::string& function_name) const {
  std::ostringstream node;
  
  node << "    " << function_name << "_block_" << block.id.id << " [label=\"";
  node << "Block " << block.id.id << "\\n";
  node << "─────────────\\n";
  
  for (const SSAInstruction& instruction : block.instructions) {
    node << format_instruction(instruction) << "\\n";
  }
  
  // For functions, we can't easily access the per-function sealed blocks
  // So we'll just mark all blocks as sealed for now
  node << "─────────────\\n";
  node << "[SEALED]\\n";
  
  node << "\"];\n";
  return node.str();
}

std::string SSAVisualizer::format_instruction(const SSAInstruction& instruction) const {
  switch (instruction.type) {
    case SSAInstructionType::Assign:
      return format_variable(instruction.dest) + " := " + format_value(instruction.value);
      
    case SSAInstructionType::BinaryOp:
      return format_variable(instruction.dest) + " := " + 
             format_value(instruction.left) + " " + 
             format_binary_operator(instruction.binary_operator) + " " + 
             format_value(instruction.right);
             
    case SSAInstructionType::UnaryOp:
      return format_variable(instruction.dest) + " := " + 
             format_unary_operator(instruction.unary_operator) + " " +
             format_value(instruction.operand);
             
    case SSAInstructionType::Jump:
      return "jump block_" + std::to_string(instruction.target.id);
      
    case SSAInstructionType::ConditionalJump:
      return "if " + format_value(instruction.condition) + 
             " then block_" + std::to_string(instruction.true_target.id) + 
             " else block_" + std::to_string(instruction.false_target.id);
             
    case SSAInstructionType::Print:
      return "print " + format_value(instruction.value);
      
    case SSAInstructionType::Phi: {
      // Format phi instruction: dest = φ(operand1, operand2, ...)
      const auto& phis_map = translator->get_phis();
      auto phi_it = phis_map.find(instruction.phi_id);
      if (phi_it != phis_map.end()) {
        const SSAPhi& phi = phi_it->second;
        std::ostringstream result;
        result << format_variable(instruction.dest) << " := φ(";
        bool first = true;
        for (const SSAValue& operand : phi.operands) {
          if (!first) result << ", ";
          result << format_value(operand);
          first = false;
        }
        result << ")";
        return result.str();
      } else {
        return format_variable(instruction.dest) + " := φ(?)";
      }
    }
    
    case SSAInstructionType::Return:
      return "return " + format_value(instruction.value);
      
    default:
      return "unknown instruction";
  }
}

std::string SSAVisualizer::format_value(const SSAValue& value) const {
  switch (value.type) {
    case SSAValueType::Literal:
      return std::to_string(value.literal_value);
      
    case SSAValueType::Var:
      return format_variable(value.variable);
      
    case SSAValueType::Phi: {
      const auto& phis_map = translator->get_phis();
      auto phi_it = phis_map.find(value.phi_id);
      if (phi_it != phis_map.end()) {
        const SSAPhi& phi = phi_it->second;
        if (phi.operands.empty()) {
          return "Φ" + std::to_string(value.phi_id.id);
        } else {
          std::ostringstream result;
          result << "Φ" << value.phi_id.id << "(";
          bool first = true;
          for (const SSAValue& operand : phi.operands) {
            if (!first) result << ",";
            result << format_value(operand);
            first = false;
          }
          result << ")";
          return result.str();
        }
      } else {
        return "Φ" + std::to_string(value.phi_id.id);
      }
    }
    
    case SSAValueType::Undefined:
      return "⊥";
      
    default:
      return "unknown";
  }
}

std::string SSAVisualizer::format_variable(const Variable& variable) const {
  return escape_dot_string(variable.name);
}

std::string SSAVisualizer::format_binary_operator(BinaryOperator op) const {
  switch (op) {
    case BinaryOperator::Add: return "+";
    case BinaryOperator::Subtract: return "-";
    case BinaryOperator::Multiply: return "*";
    case BinaryOperator::Divide: return "/";
    case BinaryOperator::Equal: return "==";
    case BinaryOperator::NotEqual: return "!=";
    case BinaryOperator::LessThan: return "<";
    case BinaryOperator::LessThanOrEqual: return "<=";
    case BinaryOperator::GreaterThan: return ">";
    case BinaryOperator::GreaterThanOrEqual: return ">=";
    case BinaryOperator::LogicalOr: return "||";
    case BinaryOperator::LogicalAnd: return "&&";
    default: return "?";
  }
}

std::string SSAVisualizer::format_unary_operator(UnaryOperator op) const {
  switch (op) {
    case UnaryOperator::Negate: return "-";
    case UnaryOperator::Not: return "!";
    default: return "?";
  }
}

bool SSAVisualizer::has_phi_nodes() const {
  return !translator->get_incomplete_phis().empty();
}

std::string SSAVisualizer::escape_dot_string(const std::string& str) const {
  std::ostringstream escaped;
  for (char c : str) {
    switch (c) {
      case '"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\n': escaped << "\\n"; break;
      case '\t': escaped << "\\t"; break;
      default: escaped << c; break;
    }
  }
  return escaped.str();
}

bool SSAVisualizer::render_to_file(const std::string& filepath) const {
  std::ofstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << std::endl;
    return false;
  }
  
  file << generate_dot();
  file.close();
  
  std::cout << "SSA dot graph written to: " << filepath << std::endl;
  return true;
}

bool SSAVisualizer::render_to_png(const std::string& png_path) const {
  std::string dot_path = png_path;
  size_t last_dot = dot_path.find_last_of('.');
  if (last_dot != std::string::npos) {
    dot_path = dot_path.substr(0, last_dot) + ".dot";
  } else {
    dot_path += ".dot";
  }
  
  if (!render_to_file(dot_path)) {
    return false;
  }
  
  std::string command = "dot -Tpng -o \"" + png_path + "\" \"" + dot_path + "\"";
  int result = std::system(command.c_str());
  
  if (result != 0) {
    std::cerr << "Failed to run graphviz dot command" << std::endl;
    return false;
  }
  
  std::cout << "SSA graph rendered to: " << png_path << std::endl;
  return true;
}

bool SSAVisualizer::render_and_open(const std::string& png_path) const {
  if (!render_to_png(png_path)) {
    return false;
  }
  
#ifdef __APPLE__
  std::string command = "open \"" + png_path + "\"";
#elif __linux__
  std::string command = "xdg-open \"" + png_path + "\"";
#else
  std::string command = "start \"\" \"" + png_path + "\"";
#endif
  
  int result = std::system(command.c_str());
  if (result != 0) {
    std::cerr << "Failed to open image viewer" << std::endl;
    return false;
  }
  
  return true;
}