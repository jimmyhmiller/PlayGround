package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestEdgeDestruct {
    @Test
    void testObjectSpreadInVariable() throws Exception {
        try {
            String source = "var {...x} = obj;";
            Parser.parse(source);
            System.out.println("✓ Object rest in variable works");
        } catch (Exception e) {
            System.out.println("✗ Object rest failed: " + e.getMessage());
            throw e;
        }
    }
}
