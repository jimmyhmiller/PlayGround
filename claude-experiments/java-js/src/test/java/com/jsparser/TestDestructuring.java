package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestDestructuring {
    @Test
    void testSimpleArray() throws Exception {
        String source = "var [a, b] = [1, 2];";
        Parser.parse(source);
        System.out.println("✓ Simple array destructuring works");
    }

    @Test
    void testSimpleObject() throws Exception {
        String source = "var {x, y} = {x: 1, y: 2};";
        Parser.parse(source);
        System.out.println("✓ Simple object destructuring works");
    }

    @Test
    void testFunctionParam() throws Exception {
        String source = "function f([a, b]) { return a + b; }";
        Parser.parse(source);
        System.out.println("✓ Function parameter destructuring works");
    }
}
