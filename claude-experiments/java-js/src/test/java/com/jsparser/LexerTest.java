package com.jsparser;

import org.junit.jupiter.api.Test;

import java.util.List;

public class LexerTest {
    @Test
    void testNumberLiteral() {
        String code = "42;";
        Lexer lexer = new Lexer(code);
        List<Token> tokens = lexer.tokenize();
        char[] src = code.toCharArray();

        for (Token token : tokens) {
            System.out.println("Type: " + token.type());
            System.out.println("Lexeme: " + token.lexeme(src));
            System.out.println("Literal: " + token.literal());
            if (token.literal() != null) {
                System.out.println("Literal class: " + token.literal().getClass().getName());
                System.out.println("Literal value: " + token.literal());
            }
            System.out.println();
        }
    }
}
