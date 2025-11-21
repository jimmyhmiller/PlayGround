package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestForLoopDestructuring {
    @Test
    void testForOfArray() throws Exception {
        try {
            String source = "for (let [a, b] of arr) { }";
            Parser.parse(source);
            System.out.println("✓ for-of with array destructuring works");
        } catch (Exception e) {
            System.out.println("✗ for-of with array destructuring failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testForOfObject() throws Exception {
        try {
            String source = "for (const {x, y} of arr) { }";
            Parser.parse(source);
            System.out.println("✓ for-of with object destructuring works");
        } catch (Exception e) {
            System.out.println("✗ for-of with object destructuring failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testForInObject() throws Exception {
        try {
            String source = "for (let {x} in obj) { }";
            Parser.parse(source);
            System.out.println("✓ for-in with object destructuring works");
        } catch (Exception e) {
            System.out.println("✗ for-in with object destructuring failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testForLoopDestructuring() throws Exception {
        try {
            String source = "for (let [a, b] = [1, 2]; a < 10; a++) { }";
            Parser.parse(source);
            System.out.println("✓ for loop with destructuring init works");
        } catch (Exception e) {
            System.out.println("✗ for loop with destructuring init failed: " + e.getMessage());
            throw e;
        }
    }
}
