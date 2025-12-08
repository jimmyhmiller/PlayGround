package com.jsparser;

import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class TemplatePositionTest {
    @Test
    void testTemplatePositions() {
        String source = "var x = `hello ${y} world`;";
        System.out.println("Source: " + source);
        System.out.println("Source length: " + source.length());

        Lexer lexer = new Lexer(source);
        var tokens = lexer.tokenize();

        System.out.println("\nTokens:");
        for (Token token : tokens) {
            System.out.printf("%s: pos=%d, len=%d, endPos=%d, lexeme='%s'%n",
                token.type(), token.position(), token.lexeme().length(),
                token.position() + token.lexeme().length(), token.lexeme());
        }

        Program program = Parser.parse(source);
        System.out.println("\nProgram end: " + program.end());
        System.out.println("Expected end (source length): " + source.length());
    }
}
