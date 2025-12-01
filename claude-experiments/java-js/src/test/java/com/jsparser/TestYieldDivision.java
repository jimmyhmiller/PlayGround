package com.jsparser;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

public class TestYieldDivision {
    @Test
    public void testYieldFollowedBySlash() {
        String input = "var yield = 12, a = 3;\nyield /a;";
        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();
        
        // Find the slash token after yield
        int yieldIndex = -1;
        for (int i = 0; i < tokens.size(); i++) {
            if (tokens.get(i).type() == TokenType.IDENTIFIER && 
                tokens.get(i).lexeme().equals("yield") && i > 5) {
                yieldIndex = i;
                break;
            }
        }
        
        assertTrue(yieldIndex > 0, "Should find second 'yield' token");
        
        // The next non-whitespace token should be SLASH (division), not REGEX
        Token nextToken = tokens.get(yieldIndex + 1);
        System.out.println("Token after yield: " + nextToken);
        
        assertEquals(TokenType.SLASH, nextToken.type(), 
            "After identifier 'yield', / should be division, not regex");
    }
    
    @Test
    public void testYieldNonRegexpParsing() throws IOException {
        String input = Files.readString(Path.of("test-oracles/test262/test/staging/sm/generators/yield-non-regexp.js"));

        try {
            Parser parser = new Parser(input);
            var ast = parser.parse();
            System.out.println("Parser succeeded!");
            // This file should parse successfully - yield is just a variable name
        } catch (Exception e) {
            System.out.println("Parser failed: " + e.getMessage());
            e.printStackTrace();
            fail("Parser should succeed on this file, but got error: " + e.getMessage());
        }
    }
}
