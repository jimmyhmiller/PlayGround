package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestSuper {
    @Test
    void testSuperMember() throws Exception {
        try {
            String source = "class Foo extends Bar { method() { return super.x; } }";
            Parser.parse(source);
            System.out.println("✓ super.x works");
        } catch (Exception e) {
            System.out.println("✗ super.x failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testSuperCall() throws Exception {
        try {
            String source = "class Foo extends Bar { constructor() { super(); } }";
            Parser.parse(source);
            System.out.println("✓ super() works");
        } catch (Exception e) {
            System.out.println("✗ super() failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testSuperComputed() throws Exception {
        try {
            String source = "class Foo extends Bar { method() { return super[x]; } }";
            Parser.parse(source);
            System.out.println("✓ super[x] works");
        } catch (Exception e) {
            System.out.println("✗ super[x] failed: " + e.getMessage());
            throw e;
        }
    }
}
