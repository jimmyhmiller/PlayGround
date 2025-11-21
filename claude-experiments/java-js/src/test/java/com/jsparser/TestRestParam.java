package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestRestParam {
    @Test
    void testRestParameter() throws Exception {
        try {
            String source = "function f(...args) { return args; }";
            Parser.parse(source);
            System.out.println("✓ Rest parameter works");
        } catch (Exception e) {
            System.out.println("✗ Rest parameter failed: " + e.getMessage());
            throw e;
        }
    }
}
