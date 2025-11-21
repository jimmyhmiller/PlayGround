package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestMetaProperty {
    @Test
    void testNewTarget() throws Exception {
        try {
            String source = "function Foo() { var x = new.target; }";
            Parser.parse(source);
            System.out.println("✓ new.target works");
        } catch (Exception e) {
            System.out.println("✗ new.target failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testImportMeta() throws Exception {
        try {
            String source = "var x = import.meta;";
            Parser.parse(source);
            System.out.println("✓ import.meta works");
        } catch (Exception e) {
            System.out.println("✗ import.meta failed: " + e.getMessage());
            throw e;
        }
    }
}
