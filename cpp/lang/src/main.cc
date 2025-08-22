#include <iostream>
#include <string>

void show_help(const std::string &program_name) {
  std::cout << "Usage: " << program_name << " [args...]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --help         Show this help message\n\n";
  std::cout << "Default behavior: prints hello world.\n";
}

int main(int argc, char *argv[]) {
  if (argc > 1) {
    std::string arg = argv[1];

    if (arg == "--help" || arg == "-h") {
      show_help(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      std::cerr << "Run '" << argv[0] << " --help' for help.\n";
      return 1;
    }
  }

  std::cout << "Hello, world!" << std::endl;
  return 0;
}