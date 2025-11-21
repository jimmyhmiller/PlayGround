package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestObjectSyntax {
    @Test
    void testComputedProperty() throws Exception {
        try {
            String source = "var obj = {[x]: 1};";
            Parser.parse(source);
            System.out.println("✓ Computed property works");
        } catch (Exception e) {
            System.out.println("✗ Computed property failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testMethodShorthand() throws Exception {
        try {
            String source = "var obj = {foo() { return 1; }};";
            Parser.parse(source);
            System.out.println("✓ Method shorthand works");
        } catch (Exception e) {
            System.out.println("✗ Method shorthand failed: " + e.getMessage());
            throw e;
        }
    }
}
