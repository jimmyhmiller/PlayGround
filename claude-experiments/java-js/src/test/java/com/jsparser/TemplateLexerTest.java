package com.jsparser;

import org.junit.jupiter.api.Test;

import java.util.List;

public class TemplateLexerTest {
    @Test
    void debugTemplateTokens() {
        String source = "`/[\\\\w-\\\\uFFFF]/u`";
        Lexer lexer = new Lexer(source);
        List<Token> tokens = lexer.tokenize();

        System.out.println("Source: " + source);
        System.out.println("Source positions:");
        for (int i = 0; i < source.length(); i++) {
            System.out.println(i + ": '" + source.charAt(i) + "'");
        }
        System.out.println();

        for (Token token : tokens) {
            System.out.println(token.type() + " at " + token.position() +
                "-" + token.endPosition() +
                " (line=" + token.line() + ", col=" + token.column() + "): " +
                "lexeme=" + token.lexeme() +
                ", literal=" + token.literal());
        }
    }
}
