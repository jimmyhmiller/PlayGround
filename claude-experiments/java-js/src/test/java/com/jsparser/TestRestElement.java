package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestRestElement {
    @Test
    void testRestElement() throws Exception {
        String source = "var [a, ...rest] = [1, 2, 3];";
        Parser.parse(source);
        System.out.println("✓ Rest element works");
    }

    @Test
    void testHoles() throws Exception {
        String source = "var [a, , c] = [1, 2, 3];";
        Parser.parse(source);
        System.out.println("✓ Holes work");
    }
}
