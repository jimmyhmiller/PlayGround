package com.jsparser;

import org.junit.jupiter.api.Test;

public class ClassASITest {
    @Test
    public void testSetFollowedByGenerator() {
        String source = "class A {\n  set\n  *a(x) {}\n}";
        Lexer lexer = new Lexer(source);
        var tokens = lexer.tokenize();

        System.out.println("Tokens:");
        for (int i = 0; i < tokens.size(); i++) {
            Token token = tokens.get(i);
            System.out.println(i + ": " + token.type() + " -> '" + token.lexeme() + "' (line " + token.line() + ", col " + token.column() + ")");
        }
    }
}
