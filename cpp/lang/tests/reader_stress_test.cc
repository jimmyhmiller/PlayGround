#include "../src/reader.h"
#include <cassert>
#include <iostream>
#include <random>
#include <string>
#include <vector>

class RandomTextGenerator {
private:
  std::mt19937 rng;
  std::uniform_int_distribution<int> choice_dist;
  std::uniform_int_distribution<int> length_dist;
  std::uniform_int_distribution<int> depth_dist;
  std::uniform_int_distribution<int> char_dist;

  // Token type pools
  std::vector<std::string> identifiers = {
      "foo",   "bar", "baz", "hello", "world", "test", "var", "func",
      "class", "obj", "x",   "y",     "z",     "a",    "b",   "c"};

  std::vector<std::string> numbers = {
      "0",    "1",   "42",  "123",   "999",  "3.14",  "0.5",  "2.71",
      "10.0", "0.0", "100", "0.001", "42.5", "99.99", "1.414"};

  std::vector<std::string> strings = {
      "\"hello\"",     "\"world\"",      "\"test\"",      "\"foo bar\"",
      "\"\"",          "\"123\"",        "\"a\"",         "\"hello world\"",
      "\"test\\nme\"", "\"tab\\there\"", "\"quote\\\"\"", "\"back\\\\slash\""};

  std::vector<std::string> operators = {
      "+", "-", "*", "/", "=", "==", "!=", "<", ">", "<=", ">=", "^", "!", "."};

  std::vector<std::string> separators = {",", ";", ":"};

  std::vector<std::pair<std::string, std::string>> delimiters = {
      {"(", ")"}, {"[", "]"}, {"{", "}"}};

public:
  RandomTextGenerator(unsigned seed = std::random_device{}())
      : rng(seed), choice_dist(0, 100), length_dist(1, 8), depth_dist(0, 4),
        char_dist(0, 25) {}

  std::string random_identifier() {
    if (choice_dist(rng) < 80) {
      // Use predefined identifier
      return identifiers[choice_dist(rng) % identifiers.size()];
    } else {
      // Generate random identifier
      std::string id;
      int len = length_dist(rng);
      for (int i = 0; i < len; i++) {
        if (i == 0) {
          id += ('a' + char_dist(rng));
        } else {
          if (choice_dist(rng) < 70) {
            id += ('a' + char_dist(rng));
          } else if (choice_dist(rng) < 85) {
            id += ('0' + (char_dist(rng) % 10));
          } else {
            id += '_';
          }
        }
      }
      // Randomly add ! suffix
      if (choice_dist(rng) < 20) {
        id += '!';
      }
      return id;
    }
  }

  std::string random_number() {
    return numbers[choice_dist(rng) % numbers.size()];
  }

  std::string random_string() {
    return strings[choice_dist(rng) % strings.size()];
  }

  std::string random_operator() {
    return operators[choice_dist(rng) % operators.size()];
  }

  std::string random_separator() {
    return separators[choice_dist(rng) % separators.size()];
  }

  std::string random_atom() {
    int choice = choice_dist(rng) % 100;
    if (choice < 40) {
      return random_identifier();
    } else if (choice < 70) {
      return random_number();
    } else if (choice < 85) {
      return random_string();
    } else if (choice < 95) {
      return random_operator();
    } else {
      return random_separator();
    }
  }

  std::string generate_expression(int max_depth = 3) {
    if (max_depth <= 0 || choice_dist(rng) < 30) {
      // Base case: return an atom
      return random_atom();
    }

    int structure_choice = choice_dist(rng) % 100;
    if (structure_choice < 30) {
      // Binary expression: atom op atom
      return random_atom() + " " + random_operator() + " " +
             generate_expression(max_depth - 1);
    } else if (structure_choice < 50) {
      // Prefix expression: op atom
      return random_operator() + " " + generate_expression(max_depth - 1);
    } else if (structure_choice < 60) {
      // Postfix expression: atom op
      return generate_expression(max_depth - 1) + " " + random_operator();
    } else if (structure_choice < 75) {
      // Parenthesized expression
      return "( " + generate_expression(max_depth - 1) + " )";
    } else if (structure_choice < 85) {
      // Block expression
      std::string block = "{ ";
      int num_statements = 1 + (choice_dist(rng) % 3);
      for (int i = 0; i < num_statements; i++) {
        if (i > 0)
          block += " ";
        block += generate_expression(max_depth - 1);
      }
      block += " }";
      return block;
    } else if (structure_choice < 95) {
      // List/array expression
      std::string list = "[ ";
      int num_elements = 1 + (choice_dist(rng) % 4);
      for (int i = 0; i < num_elements; i++) {
        if (i > 0)
          list += " ";
        list += generate_expression(max_depth - 1);
      }
      list += " ]";
      return list;
    } else {
      // Multiple expressions separated by space
      std::string multi = generate_expression(max_depth - 1);
      int num_extra = 1 + (choice_dist(rng) % 2);
      for (int i = 0; i < num_extra; i++) {
        multi += " " + generate_expression(max_depth - 1);
      }
      return multi;
    }
  }

  std::string generate_complex_text(int complexity = 5) {
    std::string result;
    int num_expressions = 1 + (choice_dist(rng) % complexity);

    for (int i = 0; i < num_expressions; i++) {
      if (i > 0) {
        result += " ";
      }
      result += generate_expression(depth_dist(rng));
    }

    return result;
  }
};

class StressTest {
private:
  RandomTextGenerator generator;
  int tests_run = 0;
  int tests_passed = 0;

public:
  StressTest(unsigned seed = std::random_device{}()) : generator(seed) {}

  bool test_parse(const std::string &input) {
    tests_run++;
    try {
      Reader reader(input);
      reader.read();

      // Basic sanity checks
      // 1. Root should exist
      assert(reader.root.type == ReaderNodeType::List);

      // 2. Any successful parse should have consistent structure
      validate_tree(reader.root);

      tests_passed++;
      return true;
    } catch (const std::exception &e) {
      std::cerr << "FAILED to parse: \"" << input << "\"\n";
      std::cerr << "Error: " << e.what() << "\n";
      return false;
    }
  }

  void validate_tree(const ReaderNode &node) {
    // Validate that the tree structure is consistent
    for (const auto &child : node.children) {
      // Each child should have valid type
      assert(static_cast<int>(child.type) >= 0 &&
             static_cast<int>(child.type) <=
                 static_cast<int>(ReaderNodeType::Call));

      // Recursively validate children
      validate_tree(child);

      // Type-specific validations
      switch (child.type) {
      case ReaderNodeType::BinaryOp:
        assert(child.children.size() == 2);
        break;
      case ReaderNodeType::PrefixOp:
      case ReaderNodeType::PostfixOp:
        assert(child.children.size() == 1);
        break;
      case ReaderNodeType::Literal:
      case ReaderNodeType::Ident:
        // Literals and identifiers should have no children (terminal nodes)
        assert(child.children.empty());
        break;
      case ReaderNodeType::List:
      case ReaderNodeType::Block:
        // Lists and blocks can have any number of children
        break;
      case ReaderNodeType::Call:
        // Calls should have at least one child (the function)
        assert(!child.children.empty());
        break;
      }
    }
  }

  void run_stress_tests(int num_tests = 1000) {
    std::cout << "Running " << num_tests << " stress tests...\n";

    for (int i = 0; i < num_tests; i++) {
      if (i % 100 == 0) {
        std::cout << "Progress: " << i << "/" << num_tests << " ("
                  << (100 * i / num_tests) << "%)\n";
      }

      // Generate random text with varying complexity
      int complexity = 1 + (i % 10); // Increase complexity as we go
      std::string test_input = generator.generate_complex_text(complexity);

      test_parse(test_input);
    }

    std::cout << "\nStress test results:\n";
    std::cout << "Tests run: " << tests_run << "\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Success rate: " << (100.0 * tests_passed / tests_run)
              << "%\n";

    if (tests_passed == tests_run) {
      std::cout << "✓ All stress tests passed!\n";
    } else {
      std::cout << "✗ Some tests failed!\n";
      exit(1);
    }
  }

  void run_specific_structure_tests() {
    std::cout << "Running specific structure tests...\n";

    // Test deeply nested structures
    std::vector<std::string> specific_tests = {
        "( ( ( 1 + 2 ) * 3 ) - 4 )",
        "{ a + b { c * d } e - f }",
        "[ 1 2 [ 3 4 [ 5 ] ] 6 ]",
        "foo ! bar ! baz !",
        "- - - 42",
        "a + b * c ^ d ^ e",
        "{ [ ( x + y ) ] }",
        "func ( arg1 arg2 ) { return arg1 + arg2 }",
        "obj . method ( \"arg\" )",
        "x == y != z < w > v <= u >= t",
        "\"string with spaces\" + 42.5 * ( a ! )",
        "[ ] { } ( )",
        "a ; b , c : d",
        "; , :", // Test standalone separators
    };

    for (const auto &test : specific_tests) {
      std::cout << "Testing: " << test << "\n";
      if (!test_parse(test)) {
        std::cout << "✗ Specific test failed!\n";
        exit(1);
      }
    }

    std::cout << "✓ All specific structure tests passed!\n";
  }
};

int main() {
  std::cout << "Reader Stress Test\n";
  std::cout << "==================\n\n";

  // Use a fixed seed for reproducible tests
  unsigned seed = 12345;
  std::cout << "Using seed: " << seed << "\n\n";

  StressTest tester(seed);

  // Run specific structure tests first
  tester.run_specific_structure_tests();
  std::cout << "\n";

  // Run random stress tests
  tester.run_stress_tests(1000);

  std::cout << "\n✓ All stress tests completed successfully!\n";
  std::cout << "The reader successfully parsed all generated text into valid "
               "trees.\n";

  return 0;
}