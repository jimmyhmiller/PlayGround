package com.jsparser;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestTemplateLiteralEscape {

    @Test
    public void testTemplateWithEscapedSingleQuote() {
        // This is the exact pattern from the failing file
        // Note: \\' is an escaped single quote in the template literal
        String input = "console.error(`Error reading \"sharp\" dependencies from \"${o}/package.json\"\\'`);";

        System.out.println("Input: " + input);
        System.out.println("Length: " + input.length());

        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();

        System.out.println("\nTokenized " + tokens.size() + " tokens:");
        for (int i = 0; i < tokens.size(); i++) {
            System.out.println(i + ": " + tokens.get(i));
        }

        // Should successfully tokenize
        assertTrue(tokens.size() > 0, "Should tokenize successfully");

        // Find the template literal token(s)
        long templateTokens = tokens.stream()
            .filter(t -> t.type().name().contains("TEMPLATE"))
            .count();

        System.out.println("\nTemplate tokens found: " + templateTokens);
        assertTrue(templateTokens > 0, "Should have template literal tokens");
    }

    @Test
    public void testSimpleTemplateWithEscapedQuote() {
        String input = "`hello\\'world`";

        System.out.println("Input: " + input);

        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();

        System.out.println("Tokenized " + tokens.size() + " tokens:");
        for (var token : tokens) {
            System.out.println(token);
        }

        assertTrue(tokens.size() > 0, "Should tokenize successfully");
    }
}
