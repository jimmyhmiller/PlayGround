package com.jsparser;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestBackslashQuote {

    @Test
    public void testDoubleBackslashInString() {
        // Test the pattern found in the failing file: "\\\"
        String input = "var x = \"\\\\\";";  // This is: var x = "\\";

        System.out.println("Input: " + input);

        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();

        System.out.println("Tokenized " + tokens.size() + " tokens");
        for (var token : tokens) {
            System.out.println(token);
        }

        // Find the STRING token
        Token stringToken = tokens.stream()
            .filter(t -> t.type() == TokenType.STRING)
            .findFirst()
            .orElse(null);

        assertNotNull(stringToken, "Should find a STRING token");
        System.out.println("String token literal: " + stringToken.literal());
        // Source "\\\" has TWO backslashes, which JavaScript interprets as ONE backslash
        // The token literal should contain the interpreted value (ONE backslash)
        assertEquals("\\", stringToken.literal(), "String value should be one backslash (interpreted from \\\\)");
    }

    @Test
    public void testComplexPattern() {
        // Test the exact pattern from the file
        String input = "e[t-1]===\"/\"||e[t-1]===\"\\\\\"";

        System.out.println("Input: " + input);

        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();

        System.out.println("Tokenized " + tokens.size() + " tokens");
        for (var token : tokens) {
            System.out.println(token);
        }

        assertTrue(tokens.size() > 0, "Should tokenize successfully");
    }

    @Test
    public void testFragmentAroundPosition49272() throws Exception {
        // Test the actual fragment from the file
        String input = java.nio.file.Files.readString(java.nio.file.Path.of("/tmp/backslash-quote-fragment.js"));

        System.out.println("Fragment length: " + input.length());

        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();

        System.out.println("SUCCESS: Tokenized " + tokens.size() + " tokens");
    }
}
