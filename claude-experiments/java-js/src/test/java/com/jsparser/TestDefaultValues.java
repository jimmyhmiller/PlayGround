package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestDefaultValues {
    @Test
    void testArrayDefault() throws Exception {
        try {
            String source = "var [a = 1] = [];";
            Parser.parse(source);
            System.out.println("✓ Array default values work");
        } catch (Exception e) {
            System.out.println("✗ Array default values failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testObjectDefault() throws Exception {
        try {
            String source = "var {x = 1} = {};";
            Parser.parse(source);
            System.out.println("✓ Object default values work");
        } catch (Exception e) {
            System.out.println("✗ Object default values failed: " + e.getMessage());
            throw e;
        }
    }
}
