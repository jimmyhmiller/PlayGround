package com.jsparser;

import org.junit.jupiter.api.Test;

public class DebugRegexScanTest {

    @Test
    void debugRegexScanning() {
        String source = "const isValidHeaderName = (str) => /^[-_a-zA-Z0-9^`|~,!#$%&'*+.]+$/.test(str.trim());";

        System.out.println("Source: " + source);
        System.out.println("Source length: " + source.length());

        Lexer lexer = new Lexer(source);

        System.out.println("\nTokenizing...");
        try {
            var tokens = lexer.tokenize();

            System.out.println("\nTokens found: " + tokens.size());
            for (int i = 0; i < tokens.size(); i++) {
                Token token = tokens.get(i);
                System.out.println("Token " + i + ": " + token.type() + " | lexeme: " + token.lexeme() +
                    " | line: " + token.line() + " | col: " + token.column() +
                    " | pos: " + token.position() + "-" + token.endPosition());
            }
        } catch (Exception e) {
            System.err.println("\nError during tokenization:");
            System.err.println("Message: " + e.getMessage());
            e.printStackTrace();
            throw e;
        }
    }
}
