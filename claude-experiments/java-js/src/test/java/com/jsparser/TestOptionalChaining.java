package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestOptionalChaining {
    @Test
    void testOptionalMember() throws Exception {
        try {
            String source = "var x = a?.b;";
            Parser.parse(source);
            System.out.println("✓ Optional member works");
        } catch (Exception e) {
            System.out.println("✗ Optional member failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testOptionalCall() throws Exception {
        try {
            String source = "var x = a?.();";
            Parser.parse(source);
            System.out.println("✓ Optional call works");
        } catch (Exception e) {
            System.out.println("✗ Optional call failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testOptionalComputed() throws Exception {
        try {
            String source = "var x = a?.[b];";
            Parser.parse(source);
            System.out.println("✓ Optional computed member works");
        } catch (Exception e) {
            System.out.println("✗ Optional computed member failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testOptionalChain() throws Exception {
        try {
            String source = "var x = a?.b?.c;";
            Parser.parse(source);
            System.out.println("✓ Optional chain works");
        } catch (Exception e) {
            System.out.println("✗ Optional chain failed: " + e.getMessage());
            throw e;
        }
    }
}
