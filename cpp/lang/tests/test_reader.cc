#include "../src/reader.h"
// #include "../src/ast.h" // TODO: Re-enable when AST conversion is implemented
#include <cassert>
#include <iostream>

void test_reader_simple_numbers() {
  {
    std::cout << "  Testing '42'..." << std::endl;
    Reader reader("42");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Literal);
    assert(reader.root.children[0].value() == "42");
  }

  {
    Reader reader("123 456");
    reader.read();
    assert(reader.root.children.size() == 2);
    assert(reader.root.children[0].type == ReaderNodeType::Literal);
    assert(reader.root.children[0].value() == "123");
    assert(reader.root.children[1].type == ReaderNodeType::Literal);
    assert(reader.root.children[1].value() == "456");
  }
}

void test_reader_binary_operations() {
  {
    Reader reader("1 + 2");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "+");
    assert(expr.children.size() == 2);
    assert(expr.children[0].type == ReaderNodeType::Literal);
    assert(expr.children[0].value() == "1");
    assert(expr.children[1].type == ReaderNodeType::Literal);
    assert(expr.children[1].value() == "2");
  }

  {
    Reader reader("10 - 5");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "-");
    assert(expr.children[0].value() == "10");
    assert(expr.children[1].value() == "5");
  }
}

void test_reader_operator_precedence() {
  {
    Reader reader("2 + 3 * 4");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "+");
    assert(expr.children[0].value() == "2");

    auto &right = expr.children[1];
    assert(right.type == ReaderNodeType::BinaryOp);
    assert(right.value() == "*");
    assert(right.children[0].value() == "3");
    assert(right.children[1].value() == "4");
  }

  {
    Reader reader("2 * 3 + 4");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "+");

    auto &left = expr.children[0];
    assert(left.type == ReaderNodeType::BinaryOp);
    assert(left.value() == "*");
    assert(left.children[0].value() == "2");
    assert(left.children[1].value() == "3");

    assert(expr.children[1].value() == "4");
  }
}

void test_reader_right_associative() {
  Reader reader("2 ^ 3 ^ 4");
  reader.read();
  assert(reader.root.children.size() == 1);
  auto &expr = reader.root.children[0];
  assert(expr.type == ReaderNodeType::BinaryOp);
  assert(expr.value() == "^");
  assert(expr.children[0].value() == "2");

  auto &right = expr.children[1];
  assert(right.type == ReaderNodeType::BinaryOp);
  assert(right.value() == "^");
  assert(right.children[0].value() == "3");
  assert(right.children[1].value() == "4");
}

void test_reader_unary_minus() {
  {
    Reader reader("-42");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::PrefixOp);
    assert(expr.value() == "-");
    assert(expr.children.size() == 1);
    assert(expr.children[0].type == ReaderNodeType::Literal);
    assert(expr.children[0].value() == "42");
  }

  {
    Reader reader("-2 + 3");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "+");

    auto &left = expr.children[0];
    assert(left.type == ReaderNodeType::PrefixOp);
    assert(left.value() == "-");
    assert(left.children[0].value() == "2");

    assert(expr.children[1].value() == "3");
  }
}

void test_reader_postfix_operator() {
  Reader reader("5 !");
  reader.read();
  assert(reader.root.children.size() == 1);
  auto &expr = reader.root.children[0];
  assert(expr.type == ReaderNodeType::PostfixOp);
  assert(expr.value() == "!");
  assert(expr.children.size() == 1);
  assert(expr.children[0].type == ReaderNodeType::Literal);
  assert(expr.children[0].value() == "5");
}

void test_reader_complex_expression() {
  Reader reader("-2 * 3 + 4 ^ 2 ^ 3");
  reader.read();
  assert(reader.root.children.size() == 1);
  auto &root = reader.root.children[0];

  assert(root.type == ReaderNodeType::BinaryOp);
  assert(root.value() == "+");

  auto &left = root.children[0];
  assert(left.type == ReaderNodeType::BinaryOp);
  assert(left.value() == "*");

  auto &leftLeft = left.children[0];
  assert(leftLeft.type == ReaderNodeType::PrefixOp);
  assert(leftLeft.value() == "-");
  assert(leftLeft.children[0].value() == "2");

  assert(left.children[1].value() == "3");

  auto &right = root.children[1];
  assert(right.type == ReaderNodeType::BinaryOp);
  assert(right.value() == "^");
  assert(right.children[0].value() == "4");

  auto &rightRight = right.children[1];
  assert(rightRight.type == ReaderNodeType::BinaryOp);
  assert(rightRight.value() == "^");
  assert(rightRight.children[0].value() == "2");
  assert(rightRight.children[1].value() == "3");
}

void test_reader_multiple_expressions() {
  Reader reader("1 + 2 3 * 4");
  reader.read();
  assert(reader.root.children.size() == 2);

  auto &expr1 = reader.root.children[0];
  assert(expr1.type == ReaderNodeType::BinaryOp);
  assert(expr1.value() == "+");
  assert(expr1.children[0].value() == "1");
  assert(expr1.children[1].value() == "2");

  auto &expr2 = reader.root.children[1];
  assert(expr2.type == ReaderNodeType::BinaryOp);
  assert(expr2.value() == "*");
  assert(expr2.children[0].value() == "3");
  assert(expr2.children[1].value() == "4");
}

void test_reader_node_equality() {
  Token token1{TokenType::INTEGER, "42", 1, 0};
  Token token2{TokenType::INTEGER, "42", 1, 0};
  Token token3{TokenType::INTEGER, "43", 1, 0};

  ReaderNode node1(ReaderNodeType::Literal, token1);
  ReaderNode node2(ReaderNodeType::Literal, token2);
  ReaderNode node3(ReaderNodeType::Literal, token3);

  assert(node1 == node2);
  assert(!(node1 == node3));

  Token plusToken{TokenType::Operator, "+", 1, 0};
  Token oneToken{TokenType::INTEGER, "1", 1, 0};
  Token twoToken{TokenType::INTEGER, "2", 1, 0};

  ReaderNode parent1(ReaderNodeType::BinaryOp, plusToken);
  parent1.add_child(ReaderNode(ReaderNodeType::Literal, oneToken));
  parent1.add_child(ReaderNode(ReaderNodeType::Literal, twoToken));

  ReaderNode parent2(ReaderNodeType::BinaryOp, plusToken);
  parent2.add_child(ReaderNode(ReaderNodeType::Literal, oneToken));
  parent2.add_child(ReaderNode(ReaderNodeType::Literal, twoToken));

  assert(parent1 == parent2);
}

void test_reader_empty_input() {
  Reader reader("");
  reader.read();
  assert(reader.root.children.size() == 0);
}

void test_reader_whitespace_handling() {
  Reader reader("  42   +   3  ");
  reader.read();
  assert(reader.root.children.size() == 1);
  auto &expr = reader.root.children[0];
  assert(expr.type == ReaderNodeType::BinaryOp);
  assert(expr.value() == "+");
  assert(expr.children[0].value() == "42");
  assert(expr.children[1].value() == "3");
}

void test_reader_identifiers() {
  {
    Reader reader("foo");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Ident);
    assert(reader.root.children[0].value() == "foo");
  }

  {
    Reader reader("hello world");
    reader.read();
    assert(reader.root.children.size() == 2);
    assert(reader.root.children[0].type == ReaderNodeType::Ident);
    assert(reader.root.children[0].value() == "hello");
    assert(reader.root.children[1].type == ReaderNodeType::Ident);
    assert(reader.root.children[1].value() == "world");
  }

  {
    Reader reader("var_name test123 func!");
    reader.read();
    assert(reader.root.children.size() == 3);
    assert(reader.root.children[0].value() == "var_name");
    assert(reader.root.children[1].value() == "test123");
    assert(reader.root.children[2].value() == "func!");
  }
}

void test_reader_strings() {
  {
    Reader reader("\"hello\"");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Literal);
    assert(reader.root.children[0].value() == "\"hello\"");
  }

  {
    Reader reader("\"hello world\" \"test\"");
    reader.read();
    assert(reader.root.children.size() == 2);
    assert(reader.root.children[0].value() == "\"hello world\"");
    assert(reader.root.children[1].value() == "\"test\"");
  }

  {
    Reader reader("\"\"");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].value() == "\"\"");
  }

  {
    Reader reader("\"escape\\ntest\"");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].value() == "\"escape\\ntest\"");
  }
}

void test_reader_decimal_numbers() {
  {
    Reader reader("3.14");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Literal);
    assert(reader.root.children[0].value() == "3.14");
  }

  {
    Reader reader("0.5 1.0 99.99");
    reader.read();
    assert(reader.root.children.size() == 3);
    assert(reader.root.children[0].value() == "0.5");
    assert(reader.root.children[1].value() == "1.0");
    assert(reader.root.children[2].value() == "99.99");
  }
}

void test_reader_numbers_with_underscores() {
  {
    Reader reader("1_000");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Literal);
    assert(reader.root.children[0].value() == "1_000");
  }

  {
    Reader reader("1_000_000");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Literal);
    assert(reader.root.children[0].value() == "1_000_000");
  }

  {
    Reader reader("3.14_159");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Literal);
    assert(reader.root.children[0].value() == "3.14_159");
  }

  {
    Reader reader("1_000 + 2_000");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::BinaryOp);
    assert(reader.root.children[0].children[0].value() == "1_000");
    assert(reader.root.children[0].children[1].value() == "2_000");
  }

  {
    Reader reader("[a b c]");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &list = reader.root.children[0];
    assert(list.type == ReaderNodeType::List);
    assert(list.children.size() == 3);
    assert(list.children[0].value() == "a");
    assert(list.children[1].value() == "b");
    assert(list.children[2].value() == "c");
  }

  {
    Reader reader("(1 + 2)");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &list = reader.root.children[0];
    assert(list.type == ReaderNodeType::List);
    assert(list.children.size() == 1);
    assert(list.children[0].type == ReaderNodeType::BinaryOp);
    assert(list.children[0].value() == "+");
  }
}

void test_reader_blocks() {
  {
    Reader reader("{}");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Block);
    assert(reader.root.children[0].children.size() == 0);
  }

  {
    Reader reader("{ a b c }");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &block = reader.root.children[0];
    assert(block.type == ReaderNodeType::Block);
    assert(block.children.size() == 3);
    assert(block.children[0].value() == "a");
    assert(block.children[1].value() == "b");
    assert(block.children[2].value() == "c");
  }

  {
    Reader reader("{ 1 + 2 3 * 4 }");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &block = reader.root.children[0];
    assert(block.type == ReaderNodeType::Block);
    assert(block.children.size() == 2);
    assert(block.children[0].type == ReaderNodeType::BinaryOp);
    assert(block.children[0].value() == "+");
    assert(block.children[1].type == ReaderNodeType::BinaryOp);
    assert(block.children[1].value() == "*");
  }
}

void test_reader_nested_structures() {
  {
    Reader reader("{ [ 1 2 ] ( 3 4 ) }");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &block = reader.root.children[0];
    assert(block.type == ReaderNodeType::Block);
    assert(block.children.size() == 1);

    auto &call = block.children[0];
    assert(call.type == ReaderNodeType::Call);
    assert(call.children.size() == 3);

    auto &list1 = call.children[0];
    assert(list1.type == ReaderNodeType::List);
    assert(list1.children.size() == 2);
    assert(list1.children[0].value() == "1");
    assert(list1.children[1].value() == "2");

    assert(call.children[1].value() == "3");
    assert(call.children[2].value() == "4");
  }

  {
    Reader reader("[ { a + b } { c * d } ]");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &list = reader.root.children[0];
    assert(list.type == ReaderNodeType::List);
    assert(list.children.size() == 2);

    auto &block1 = list.children[0];
    assert(block1.type == ReaderNodeType::Block);
    assert(block1.children.size() == 1);
    assert(block1.children[0].type == ReaderNodeType::BinaryOp);
    assert(block1.children[0].value() == "+");

    auto &block2 = list.children[1];
    assert(block2.type == ReaderNodeType::Block);
    assert(block2.children.size() == 1);
    assert(block2.children[0].type == ReaderNodeType::BinaryOp);
    assert(block2.children[0].value() == "*");
  }
}

void test_reader_dot_operator() {
  {
    Reader reader("obj . method");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == ".");
    assert(expr.children[0].value() == "obj");
    assert(expr.children[1].value() == "method");
  }

  {
    // Test precedence: dot should bind tighter than +
    Reader reader("a + b . c");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "+");
    assert(expr.children[0].value() == "a");

    auto &right = expr.children[1];
    assert(right.type == ReaderNodeType::BinaryOp);
    assert(right.value() == ".");
    assert(right.children[0].value() == "b");
    assert(right.children[1].value() == "c");
  }
}

void test_reader_comparison_operators() {
  {
    Reader reader("a == b");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "==");
    assert(expr.children[0].value() == "a");
    assert(expr.children[1].value() == "b");
  }

  {
    Reader reader("x != y");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "!=");
    assert(expr.children[0].value() == "x");
    assert(expr.children[1].value() == "y");
  }

  {
    Reader reader("a < b <= c > d >= e");
    reader.read();
    assert(reader.root.children.size() == 1);
    // This should parse as ((((a < b) <= c) > d) >= e) due to left
    // associativity
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == ">=");
  }
}

void test_reader_separators() {
  {
    Reader reader("a ; b , c : d");
    reader.read();
    // With semicolon parsing fix, this should now create a 'do' block
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::Ident);
    assert(reader.root.children[0].value() == "do");

    // The do block should contain the statements
    const auto &do_block = reader.root.children[0];
    assert(do_block.children.size() == 4);
    assert(do_block.children[0].value() == "a");
    assert(do_block.children[1].value() == "b");
    assert(do_block.children[2].value() == ",");
    assert(do_block.children[3].type == ReaderNodeType::BinaryOp);
    assert(do_block.children[3].value() == ":");
  }

  {
    // Test without semicolons - colon is postfix operator on comma
    Reader reader(", :");
    reader.read();
    assert(reader.root.children.size() == 1);
    assert(reader.root.children[0].type == ReaderNodeType::PostfixOp);
    assert(reader.root.children[0].value() == ":");
    assert(reader.root.children[0].children[0].value() == ",");
  }
}

// Tests for the recent fixes - operators as identifiers
void test_reader_operators_as_identifiers() {
  {
    // Test standalone operators at end of input - this gets parsed as postfix
    Reader reader("world /");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::PostfixOp);
    assert(expr.value() == "/");
    assert(expr.children[0].type == ReaderNodeType::Ident);
    assert(expr.children[0].value() == "world");
  }

  {
    // Test operators in blocks without operands
    Reader reader("{ - }");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &block = reader.root.children[0];
    assert(block.type == ReaderNodeType::Block);
    assert(block.children.size() == 1);
    assert(block.children[0].type == ReaderNodeType::Ident);
    assert(block.children[0].value() == "-");
  }

  {
    // Test operators in parentheses
    Reader reader("( >= != )");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &list = reader.root.children[0];
    assert(list.type == ReaderNodeType::List);
    assert(list.children.size() == 1);
    auto &postfix = list.children[0];
    assert(postfix.type == ReaderNodeType::PostfixOp);
    assert(postfix.value() == "!=");
    assert(postfix.children[0].type == ReaderNodeType::Ident);
    assert(postfix.children[0].value() == ">=");
  }

  {
    // Test operators in brackets
    Reader reader("[ * ]");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &list = reader.root.children[0];
    assert(list.type == ReaderNodeType::List);
    assert(list.children.size() == 1);
    assert(list.children[0].type == ReaderNodeType::Ident);
    assert(list.children[0].value() == "*");
  }

  {
    // Test mixed valid operations and standalone operators
    Reader reader("a + b /");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "+");
    assert(expr.children[0].value() == "a");

    // The right side should be "b /" as a postfix operation
    auto &right = expr.children[1];
    assert(right.type == ReaderNodeType::PostfixOp);
    assert(right.value() == "/");
    assert(right.children[0].value() == "b");
  }
}

void test_reader_prefix_operators_edge_cases() {
  {
    // Test unary minus with valid operand
    Reader reader("- 42");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::PrefixOp);
    assert(expr.value() == "-");
    assert(expr.children[0].value() == "42");
  }

  {
    // Test unary minus without operand (should be identifier)
    Reader reader("{ - }");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &block = reader.root.children[0];
    assert(block.children.size() == 1);
    assert(block.children[0].type == ReaderNodeType::Ident);
    assert(block.children[0].value() == "-");
  }

  {
    // Test multiple unary minuses
    Reader reader("- - - 42");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::PrefixOp);
    assert(expr.value() == "-");
    // Should be nested prefix operations
    assert(expr.children[0].type == ReaderNodeType::PrefixOp);
    assert(expr.children[0].value() == "-");
  }
}

void test_reader_complex_mixed_expressions() {
  {
    Reader reader("obj . method ( \"arg\" )");
    reader.read();
    assert(reader.root.children.size() == 1);
    // This parses as a single expression with function call postfix
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == ".");
    assert(expr.children[0].value() == "obj");

    auto &call = expr.children[1];
    assert(call.type == ReaderNodeType::Call);
    assert(call.children.size() == 2);
    assert(call.children[0].value() == "method");
    assert(call.children[1].value() == "\"arg\"");
  }

  {
    Reader reader("foo ! bar ! baz !");
    reader.read();
    assert(reader.root.children.size() == 3);
    // This parses as three separate postfix expressions
    for (int i = 0; i < 3; i++) {
      auto &expr = reader.root.children[i];
      assert(expr.type == ReaderNodeType::PostfixOp);
      assert(expr.value() == "!");
    }
  }

  {
    Reader reader("\"string with spaces\" + 42.5 * ( a ! )");
    reader.read();
    assert(reader.root.children.size() == 1);
    auto &expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "+");
  }
}

void test_reader_error_resilience() {
  // These should all parse without throwing exceptions
  {
    Reader reader("== ;");
    reader.read(); // Should not throw
    assert(reader.root.children.size() >= 1);
  }

  {
    Reader reader("baz ^");
    reader.read(); // Should not throw
    assert(reader.root.children.size() >= 1);
  }

  {
    Reader reader(". >= s 42.5 =");
    reader.read(); // Should not throw
    assert(reader.root.children.size() >= 1);
  }

  {
    Reader reader("{ baz } / = -");
    reader.read(); // Should not throw
    assert(reader.root.children.size() >= 1);
  }
}

int main() {
  std::cout << "Running Reader tests..." << std::endl;

  std::cout << "Testing simple numbers..." << std::endl;
  test_reader_simple_numbers();

  std::cout << "Testing binary operations..." << std::endl;
  test_reader_binary_operations();

  std::cout << "Testing operator precedence..." << std::endl;
  test_reader_operator_precedence();

  std::cout << "Testing right associative operators..." << std::endl;
  test_reader_right_associative();

  std::cout << "Testing unary minus..." << std::endl;
  test_reader_unary_minus();

  std::cout << "Testing postfix operators..." << std::endl;
  test_reader_postfix_operator();

  std::cout << "Testing complex expressions..." << std::endl;
  test_reader_complex_expression();

  std::cout << "Testing multiple expressions..." << std::endl;
  test_reader_multiple_expressions();

  std::cout << "Testing node equality..." << std::endl;
  test_reader_node_equality();

  std::cout << "Testing empty input..." << std::endl;
  test_reader_empty_input();

  std::cout << "Testing whitespace handling..." << std::endl;
  test_reader_whitespace_handling();

  std::cout << "Testing identifiers..." << std::endl;
  test_reader_identifiers();

  std::cout << "Testing strings..." << std::endl;
  test_reader_strings();

  std::cout << "Testing decimal numbers..." << std::endl;
  test_reader_decimal_numbers();

  std::cout << "Testing numbers with underscores..." << std::endl;
  test_reader_numbers_with_underscores();

  std::cout << "Testing blocks..." << std::endl;
  test_reader_blocks();

  std::cout << "Testing nested structures..." << std::endl;
  test_reader_nested_structures();

  std::cout << "Testing dot operator..." << std::endl;
  test_reader_dot_operator();

  std::cout << "Testing comparison operators..." << std::endl;
  test_reader_comparison_operators();

  std::cout << "Testing separators..." << std::endl;
  test_reader_separators();

  std::cout << "Testing operators as identifiers..." << std::endl;
  test_reader_operators_as_identifiers();

  std::cout << "Testing prefix operators edge cases..." << std::endl;
  test_reader_prefix_operators_edge_cases();

  std::cout << "Testing complex mixed expressions..." << std::endl;
  test_reader_complex_mixed_expressions();

  std::cout << "Testing error resilience..." << std::endl;
  test_reader_error_resilience();

  std::cout << "All Reader tests passed!" << std::endl;
  
  // Output standardized test stats for build script parsing
  int total_tests = 24; // Number of test functions called
  int failed_tests = 0; // All assertions would have failed if there were issues
  int passed_tests = total_tests;
  std::cout << "TEST_STATS: passed=" << passed_tests << " failed=" << failed_tests << " total=" << total_tests << std::endl;
  
  return 0;
}