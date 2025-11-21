package com.jsparser;

import org.junit.jupiter.api.Test;
import java.util.List;

public class TestDotScan {
    @Test
    public void testTrace() {
        String code = "true ?.30 : false";
        Lexer lexer = new Lexer(code);
        List<Token> tokens = lexer.tokenize();
        
        for (Token token : tokens) {
            System.out.println(token.type() + ": '" + token.lexeme() + "' at pos=" + token.position() + " col=" + token.column());
        }
    }
}
