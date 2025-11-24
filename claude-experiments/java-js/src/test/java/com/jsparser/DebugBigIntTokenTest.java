package com.jsparser;

import org.junit.jupiter.api.Test;

public class DebugBigIntTokenTest {
    @Test
    public void testBigIntTokens() {
        String source = "let { 1n: a } = { \"1\": \"foo\" };";
        Lexer lexer = new Lexer(source);
        var tokens = lexer.tokenize();

        System.out.println("Tokens:");
        for (Token token : tokens) {
            System.out.println(token.type() + " -> " + token.lexeme() + " (literal: " + token.literal() + ")");
        }
    }
}
