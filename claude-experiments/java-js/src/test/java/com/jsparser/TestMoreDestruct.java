package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestMoreDestruct {
    @Test
    void testNestedArray() throws Exception {
        String source = "var [[a]] = [[1]];";
        Parser.parse(source);
        System.out.println("✓ Nested array destructuring works");
    }

    @Test
    void testNestedObject() throws Exception {
        String source = "var {a: {b}} = {a: {b: 1}};";
        Parser.parse(source);
        System.out.println("✓ Nested object destructuring works");
    }
}
