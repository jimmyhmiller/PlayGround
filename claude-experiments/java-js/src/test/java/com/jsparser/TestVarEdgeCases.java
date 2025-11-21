package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestVarEdgeCases {
    @Test
    void testKeywordInDestructuring() throws Exception {
        try {
            String source = "var {if: x} = obj;";
            Parser.parse(source);
            System.out.println("✓ Keyword in destructuring works");
        } catch (Exception e) {
            System.out.println("✗ Keyword in destructuring failed: " + e.getMessage());
            throw e;
        }
    }
}
