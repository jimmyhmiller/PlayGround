package com.jsparser;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestNestedTemplates {

    @Test
    public void testNestedTemplateWithRegex() throws Exception {
        // The exact pattern from the workbench file - read it to avoid escaping issues
        String input = java.nio.file.Files.readString(java.nio.file.Path.of("/tmp/workbench-template-test.js"));

        System.out.println("Input: " + input);
        System.out.println("Length: " + input.length());
        System.out.println("Chars 35-45: " + input.substring(35, 45));

        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();

        System.out.println("SUCCESS: Tokenized " + tokens.size() + " tokens");
        for (var token : tokens) {
            System.out.println(token);
        }

        assertTrue(tokens.size() > 0, "Should tokenize successfully");
    }

    @Test
    public void testSimpleNestedTemplate() {
        // Simpler nested template
        String input = "`outer${`inner`}outer`";

        System.out.println("Input: " + input);

        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();

        System.out.println("SUCCESS: Tokenized " + tokens.size() + " tokens");
        for (var token : tokens) {
            System.out.println(token);
        }

        assertTrue(tokens.size() > 0, "Should tokenize successfully");
    }

    @Test
    public void testNestedTemplateWithInterpolation() {
        // Nested template with interpolation
        String input = "`outer${`inner${x}`}outer`";

        System.out.println("Input: " + input);

        Lexer lexer = new Lexer(input);
        var tokens = lexer.tokenize();

        System.out.println("SUCCESS: Tokenized " + tokens.size() + " tokens");
        for (var token : tokens) {
            System.out.println(token);
        }

        assertTrue(tokens.size() > 0, "Should tokenize successfully");
    }
}
