package com.jsparser;

import org.junit.jupiter.api.Test;

import java.util.List;

public class LexerTest {
    @Test
    void testNumberLiteral() {
        Lexer lexer = new Lexer("42;");
        List<Token> tokens = lexer.tokenize();

        for (Token token : tokens) {
            System.out.println("Type: " + token.type());
            System.out.println("Lexeme: " + token.lexeme());
            System.out.println("Literal: " + token.literal());
            if (token.literal() != null) {
                System.out.println("Literal class: " + token.literal().getClass().getName());
                System.out.println("Literal value: " + token.literal());
            }
            System.out.println();
        }
    }
}
