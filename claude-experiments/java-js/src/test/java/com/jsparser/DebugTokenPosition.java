package com.jsparser;

import org.junit.jupiter.api.Test;
import java.util.List;

public class DebugTokenPosition {
    @Test
    public void testDotNumber() {
        String code = "const value = true ?.30 : false;";
        Lexer lexer = new Lexer(code);
        List<Token> tokens = lexer.tokenize();
        
        for (Token token : tokens) {
            if (token.lexeme().equals(".30")) {
                System.out.println("Found .30 token:");
                System.out.println("  Type: " + token.type());
                System.out.println("  Lexeme: '" + token.lexeme() + "'");
                System.out.println("  Line: " + token.line());
                System.out.println("  Column: " + token.column());
                System.out.println("  Position: " + token.position());
                System.out.println("  Expected column: 20, Actual column: " + token.column());
            }
        }
    }
}
