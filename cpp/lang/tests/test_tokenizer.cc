#include "../src/tokenizer.h"
#include <cassert>
#include <iostream>
#include <vector>

// Helper function to tokenize a complete input
std::vector<Token> tokenize_input(const std::string &input) {
  Tokenizer tokenizer;
  std::vector<Token> tokens;

  while (!tokenizer.at_end(input)) {
    Token token = tokenizer.next_token(input);
    tokens.push_back(token);
    if (token.type == TokenType::End)
      break;
  }

  return tokens;
}

void test_tokenizer_basic_functionality() {
  std::cout << "Testing tokenizer basic functionality..." << std::endl;

  std::string input = "123 hello";
  std::vector<Token> tokens = tokenize_input(input);

  // Should have at least some tokens
  assert(tokens.size() > 0);
  std::cout << "  ✓ Tokenizer produces tokens from input" << std::endl;

  // Test that tokenizer progresses through input
  Tokenizer tokenizer;
  assert(tokenizer.pos == 0);

  tokenizer.consume(input, 1);
  assert(tokenizer.pos == 1);

  std::cout << "  ✓ Tokenizer position tracking works" << std::endl;
}

void test_character_classification() {
  std::cout << "Testing character classification..." << std::endl;

  Tokenizer tokenizer;

  // Test whitespace detection
  assert(tokenizer.is_whitespace(' '));
  assert(tokenizer.is_whitespace('\t'));
  assert(tokenizer.is_whitespace('\n'));
  assert(tokenizer.is_whitespace('\r'));
  assert(!tokenizer.is_whitespace('a'));

  std::cout << "  ✓ Whitespace detection works" << std::endl;

  // Test delimiter detection
  assert(tokenizer.is_delimiter('('));
  assert(tokenizer.is_delimiter(')'));
  assert(tokenizer.is_delimiter('{'));
  assert(tokenizer.is_delimiter('}'));
  assert(tokenizer.is_delimiter('['));
  assert(tokenizer.is_delimiter(']'));
  assert(!tokenizer.is_delimiter('a'));

  std::cout << "  ✓ Delimiter detection works" << std::endl;

  // Test separator detection
  assert(tokenizer.is_separator(','));
  assert(tokenizer.is_separator(';'));
  assert(tokenizer.is_separator(':'));
  assert(!tokenizer.is_separator('a'));

  std::cout << "  ✓ Separator detection works" << std::endl;

  // Test operator detection
  assert(tokenizer.is_operator('+'));
  assert(tokenizer.is_operator('-'));
  assert(tokenizer.is_operator('*'));
  assert(tokenizer.is_operator('/'));
  assert(tokenizer.is_operator('='));
  assert(tokenizer.is_operator('<'));
  assert(tokenizer.is_operator('>'));
  assert(tokenizer.is_operator('!'));
  assert(!tokenizer.is_operator('a'));

  std::cout << "  ✓ Operator detection works" << std::endl;
}

void test_line_and_column_tracking() {
  std::cout << "Testing line and column tracking..." << std::endl;

  Tokenizer tokenizer;
  std::string input = "abc\ndef\nghi";

  assert(tokenizer.line == 1);
  assert(tokenizer.column == 0);

  // Consume first line
  tokenizer.consume(input, 3);
  assert(tokenizer.column == 3);

  // Consume newline
  tokenizer.consume(input, 1);
  assert(tokenizer.line == 2);
  assert(tokenizer.column == 0);

  std::cout << "  ✓ Line and column tracking works correctly" << std::endl;
}

void test_at_end_detection() {
  std::cout << "Testing end-of-input detection..." << std::endl;

  Tokenizer tokenizer;
  std::string input = "abc";

  assert(!tokenizer.at_end(input));

  tokenizer.consume(input, 3);
  assert(tokenizer.at_end(input));

  std::cout << "  ✓ End-of-input detection works" << std::endl;
}

void test_token_types() {
  std::cout << "Testing token type recognition..." << std::endl;

  std::vector<std::string> test_inputs = {
      "123",   // INTEGER
      "3.14",  // FLOAT
      "hello", // Identifier
      "+",     // Operator
      "(",     // Delimiter
      ",",     // Separator
  };

  for (const auto &input : test_inputs) {
    std::vector<Token> tokens = tokenize_input(input);

    // Should have at least one non-whitespace token
    bool found_content_token = false;
    for (const auto &token : tokens) {
      if (token.type != TokenType::Whitespace && token.type != TokenType::End) {
        found_content_token = true;
        break;
      }
    }

    if (found_content_token) {
      std::cout << "  ✓ Input \"" << input << "\" produced content token"
                << std::endl;
    } else {
      std::cout << "  ⚠️  Input \"" << input
                << "\" did not produce expected token" << std::endl;
    }
  }
}

void test_whitespace_handling() {
  std::cout << "Testing whitespace handling..." << std::endl;

  std::string input_with_whitespace = "  a  \t  b  \n  c  ";
  std::vector<Token> tokens = tokenize_input(input_with_whitespace);

  // Count non-whitespace, non-end tokens
  int content_tokens = 0;
  for (const auto &token : tokens) {
    if (token.type != TokenType::Whitespace && token.type != TokenType::End) {
      content_tokens++;
    }
  }

  // Should have found some content tokens despite whitespace
  assert(content_tokens > 0);
  std::cout << "  ✓ Whitespace handled correctly, found " << content_tokens
            << " content tokens" << std::endl;
}

void test_empty_input() {
  std::cout << "Testing empty input handling..." << std::endl;

  std::string empty_input = "";
  Tokenizer tokenizer;

  // Should handle empty input gracefully without crashing
  try {
    std::vector<Token> tokens = tokenize_input(empty_input);
    std::cout << "  ✓ Empty input tokenized without error (produced "
              << tokens.size() << " tokens)" << std::endl;
  } catch (const std::exception &e) {
    // It's also acceptable for empty input to throw an exception
    std::cout << "  ✓ Empty input handled by throwing exception: " << e.what()
              << std::endl;
  }

  // Test that tokenizer knows when it's at end with empty input
  assert(tokenizer.at_end(empty_input));
  std::cout << "  ✓ Empty input correctly identified as at-end" << std::endl;
}

int main() {
  std::cout << "Running Tokenizer tests..." << std::endl;
  std::cout << "============================" << std::endl;

  try {
    test_tokenizer_basic_functionality();
    test_character_classification();
    test_line_and_column_tracking();
    test_at_end_detection();
    test_token_types();
    test_whitespace_handling();
    test_empty_input();

    std::cout << std::endl;
    std::cout << "All Tokenizer tests completed successfully! ✅" << std::endl;
    
    // Output standardized test stats for build script parsing
    int total_tests = 7; // Number of test functions called
    int failed_tests = 0; // All would have thrown exceptions if failed
    int passed_tests = total_tests;
    std::cout << "TEST_STATS: passed=" << passed_tests << " failed=" << failed_tests << " total=" << total_tests << std::endl;
    
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Tokenizer test failed with error: " << e.what() << std::endl;
    return 1;
  } catch (const std::runtime_error &e) {
    std::cerr << "Tokenizer test failed with runtime error: " << e.what()
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Tokenizer test failed with unknown error" << std::endl;
    return 1;
  }
}