package com.jsparser;

import org.junit.jupiter.api.Test;

import java.util.List;

public class TestTemplateTokens {

    @Test
    public void testTemplateWithInterpolationTokens() {
        String source = "const x = 1;\nconst y = `\\\n${x}`;";

        System.out.println("=== Source ===");
        for (int i = 0; i < source.length(); i++) {
            char c = source.charAt(i);
            if (c == '\n') {
                System.out.println(" [\\n at position " + i + "]");
            } else {
                System.out.print(c);
            }
        }
        System.out.println("\n");

        Lexer lexer = new Lexer(source);
        List<Token> tokens = lexer.tokenize();

        System.out.println("=== Tokens ===");
        for (int i = 0; i < tokens.size(); i++) {
            Token token = tokens.get(i);
            System.out.printf("%2d. %-20s line=%d col=%d pos=%d lexeme='%s'%n",
                    i, token.type(), token.line(), token.column(), token.position(),
                    token.lexeme().replace("\n", "\\n"));
        }
    }
}
