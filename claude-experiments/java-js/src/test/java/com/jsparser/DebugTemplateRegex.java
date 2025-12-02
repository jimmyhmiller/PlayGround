package com.jsparser;

import org.junit.jupiter.api.Test;

public class DebugTemplateRegex {

    @Test
    public void testStepByStep() {
        // Test 1: Simple regex after template head
        String test1 = "`${/a/}`";
        System.out.println("\n=== Test 1: " + test1 + " ===");
        try {
            Lexer lexer = new Lexer(test1);
            var tokens = lexer.tokenize();
            System.out.println("SUCCESS: " + tokens.size() + " tokens");
            for (var t : tokens) {
                System.out.println("  " + t);
            }
        } catch (Exception e) {
            System.out.println("FAILED: " + e.getMessage());
        }

        // Test 2: Two interpolations with regex
        String test2 = "`${a}${/b/}`";
        System.out.println("\n=== Test 2: " + test2 + " ===");
        try {
            Lexer lexer = new Lexer(test2);
            var tokens = lexer.tokenize();
            System.out.println("SUCCESS: " + tokens.size() + " tokens");
            for (var t : tokens) {
                System.out.println("  " + t);
            }
        } catch (Exception e) {
            System.out.println("FAILED: " + e.getMessage());
        }

        // Test 3: Regex with escape in second interpolation
        String test3 = "`${a}${/\\\\[/}`";
        System.out.println("\n=== Test 3: " + test3 + " ===");
        try {
            Lexer lexer = new Lexer(test3);
            var tokens = lexer.tokenize();
            System.out.println("SUCCESS: " + tokens.size() + " tokens");
            for (var t : tokens) {
                System.out.println("  " + t);
            }
        } catch (Exception e) {
            System.out.println("FAILED: " + e.getMessage());
        }
    }
}
