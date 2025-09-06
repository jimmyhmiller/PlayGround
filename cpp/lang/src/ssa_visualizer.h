#ifndef SSA_VISUALIZER_H
#define SSA_VISUALIZER_H

#include "ssa_translator.h"
#include <string>

class SSAVisualizer {
private:
  const SSATranslator* translator;

public:
  SSAVisualizer(const SSATranslator* translator);
  
  std::string generate_dot() const;
  bool render_to_file(const std::string& filepath) const;
  bool render_to_png(const std::string& png_path) const;
  bool render_and_open(const std::string& png_path) const;

private:
  std::string generate_block_node(const SSABlock& block) const;
  std::string generate_block_node_for_function(const SSABlock& block, const std::string& function_name) const;
  std::string format_instruction(const SSAInstruction& instruction) const;
  std::string format_value(const SSAValue& value) const;
  std::string format_variable(const Variable& variable) const;
  std::string format_binary_operator(BinaryOperator op) const;
  std::string format_unary_operator(UnaryOperator op) const;
  bool has_phi_nodes() const;
  std::string escape_dot_string(const std::string& str) const;
};

#endif