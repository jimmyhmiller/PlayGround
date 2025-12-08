package com.jsparser;

import org.junit.jupiter.api.Test;
import java.util.List;

public class TestDotScan {
    @Test
    public void testTrace() {
        String code = "true ?.30 : false";
        Lexer lexer = new Lexer(code);
        List<Token> tokens = lexer.tokenize();
        char[] src = code.toCharArray();

        for (Token token : tokens) {
            System.out.println(token.type() + ": '" + token.lexeme(src) + "' at pos=" + token.position() + " col=" + token.column());
        }
    }
}
