#ifdef CABIN_TEST

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
        auto& expr = reader.root.children[0];
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
        auto& expr = reader.root.children[0];
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
        auto& expr = reader.root.children[0];
        assert(expr.type == ReaderNodeType::BinaryOp);
        assert(expr.value() == "+");
        assert(expr.children[0].value() == "2");
        
        auto& right = expr.children[1];
        assert(right.type == ReaderNodeType::BinaryOp);
        assert(right.value() == "*");
        assert(right.children[0].value() == "3");
        assert(right.children[1].value() == "4");
    }
    
    {
        Reader reader("2 * 3 + 4");
        reader.read();
        assert(reader.root.children.size() == 1);
        auto& expr = reader.root.children[0];
        assert(expr.type == ReaderNodeType::BinaryOp);
        assert(expr.value() == "+");
        
        auto& left = expr.children[0];
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
    auto& expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "^");
    assert(expr.children[0].value() == "2");
    
    auto& right = expr.children[1];
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
        auto& expr = reader.root.children[0];
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
        auto& expr = reader.root.children[0];
        assert(expr.type == ReaderNodeType::BinaryOp);
        assert(expr.value() == "+");
        
        auto& left = expr.children[0];
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
    auto& expr = reader.root.children[0];
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
    auto& root = reader.root.children[0];
    
    assert(root.type == ReaderNodeType::BinaryOp);
    assert(root.value() == "+");
    
    auto& left = root.children[0];
    assert(left.type == ReaderNodeType::BinaryOp);
    assert(left.value() == "*");
    
    auto& leftLeft = left.children[0];
    assert(leftLeft.type == ReaderNodeType::PrefixOp);
    assert(leftLeft.value() == "-");
    assert(leftLeft.children[0].value() == "2");
    
    assert(left.children[1].value() == "3");
    
    auto& right = root.children[1];
    assert(right.type == ReaderNodeType::BinaryOp);
    assert(right.value() == "^");
    assert(right.children[0].value() == "4");
    
    auto& rightRight = right.children[1];
    assert(rightRight.type == ReaderNodeType::BinaryOp);
    assert(rightRight.value() == "^");
    assert(rightRight.children[0].value() == "2");
    assert(rightRight.children[1].value() == "3");
}

void test_reader_multiple_expressions() {
    Reader reader("1 + 2 3 * 4");
    reader.read();
    assert(reader.root.children.size() == 2);
    
    auto& expr1 = reader.root.children[0];
    assert(expr1.type == ReaderNodeType::BinaryOp);
    assert(expr1.value() == "+");
    assert(expr1.children[0].value() == "1");
    assert(expr1.children[1].value() == "2");
    
    auto& expr2 = reader.root.children[1];
    assert(expr2.type == ReaderNodeType::BinaryOp);
    assert(expr2.value() == "*");
    assert(expr2.children[0].value() == "3");
    assert(expr2.children[1].value() == "4");
}

void test_reader_node_equality() {
    Token token1{TokenType::NUMBER, "42", 1, 0};
    Token token2{TokenType::NUMBER, "42", 1, 0};
    Token token3{TokenType::NUMBER, "43", 1, 0};
    
    ReaderNode node1(ReaderNodeType::Literal, token1);
    ReaderNode node2(ReaderNodeType::Literal, token2);
    ReaderNode node3(ReaderNodeType::Literal, token3);
    
    assert(node1 == node2);
    assert(!(node1 == node3));
    
    Token plusToken{TokenType::Operator, "+", 1, 0};
    Token oneToken{TokenType::NUMBER, "1", 1, 0};
    Token twoToken{TokenType::NUMBER, "2", 1, 0};
    
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
    auto& expr = reader.root.children[0];
    assert(expr.type == ReaderNodeType::BinaryOp);
    assert(expr.value() == "+");
    assert(expr.children[0].value() == "42");
    assert(expr.children[1].value() == "3");
}


#endif