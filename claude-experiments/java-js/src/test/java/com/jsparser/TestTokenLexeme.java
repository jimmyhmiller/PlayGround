package com.jsparser;

import org.junit.jupiter.api.Test;
import java.util.List;

public class TestTokenLexeme {

    @Test
    void testUnicodeEscapeInIdentifier() throws Exception {
        String code = "var \\u0063onst = 42;";
        Lexer lexer = new Lexer(code);
        List<Token> tokens = lexer.tokenize();
        
        System.out.println("Code: " + code);
        System.out.println("\nTokens:");
        for (Token token : tokens) {
            System.out.println("  " + token.type() + ": '" + token.lexeme() + "'");
        }
    }
}
