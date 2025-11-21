package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestPropertyKey {
    @Test
    void testKeywordAsKey() throws Exception {
        try {
            String source = "var obj = {if: 1};";
            Parser.parse(source);
            System.out.println("✓ Keyword as property key works");
        } catch (Exception e) {
            System.out.println("✗ Keyword as key failed: " + e.getMessage());
            throw e;
        }
    }
}
