#include "../ast.h"
#include "../reader.h"
#include "../ssa_translator.h"
#include "../ssa_visualizer.h"
#include <iostream>
#include <sstream>
#include <string>

int main() {
  std::ostringstream buffer;
  buffer << std::cin.rdbuf();
  std::string input = buffer.str();

  if (input.empty()) {
    std::cerr << "Error: No input provided" << std::endl;
    return 1;
  }

  try {
    Reader reader(input);
    reader.read();

    ASTBuilder builder(reader.root.children);
    auto ast = builder.build();

    SSATranslator translator;
    translator.translate(ast.get());

    std::cout << "SSA conversion completed successfully!" << std::endl;
    std::cout << "Generated " << translator.get_blocks().size() << " basic blocks" << std::endl;
    std::cout << "Generated " << translator.get_phis().size() << " phi functions" << std::endl;

    SSAVisualizer visualizer(&translator);
    
    if (visualizer.render_to_file("ssa_graph.dot")) {
      std::cout << "SSA graph saved to ssa_graph.dot" << std::endl;
    }
    
    if (visualizer.render_to_png("ssa_graph.png")) {
      std::cout << "SSA graph visualization saved to ssa_graph.png" << std::endl;
    }

    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}