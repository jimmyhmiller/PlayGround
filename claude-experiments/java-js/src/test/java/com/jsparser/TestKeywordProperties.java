package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestKeywordProperties {
    @Test
    void testClassProperty() throws Exception {
        try {
            String source = "var x = obj.class;";
            Parser.parse(source);
            System.out.println("✓ obj.class works");
        } catch (Exception e) {
            System.out.println("✗ obj.class failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testIfProperty() throws Exception {
        try {
            String source = "var x = obj.if;";
            Parser.parse(source);
            System.out.println("✓ obj.if works");
        } catch (Exception e) {
            System.out.println("✗ obj.if failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testReturnProperty() throws Exception {
        try {
            String source = "var x = obj.return;";
            Parser.parse(source);
            System.out.println("✓ obj.return works");
        } catch (Exception e) {
            System.out.println("✗ obj.return failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testNewKeywordProperty() throws Exception {
        try {
            String source = "var x = new Foo.class();";
            Parser.parse(source);
            System.out.println("✓ new Foo.class() works");
        } catch (Exception e) {
            System.out.println("✗ new Foo.class() failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testOptionalKeywordProperty() throws Exception {
        try {
            String source = "var x = obj?.return;";
            Parser.parse(source);
            System.out.println("✓ obj?.return works");
        } catch (Exception e) {
            System.out.println("✗ obj?.return failed: " + e.getMessage());
            throw e;
        }
    }
}
