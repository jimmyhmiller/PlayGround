package com.jsparser;

import org.junit.jupiter.api.Test;

public class TestClass {
    @Test
    void testClassDeclaration() throws Exception {
        try {
            String source = "class Foo { constructor() { } method() { } }";
            Parser.parse(source);
            System.out.println("✓ Class declaration works");
        } catch (Exception e) {
            System.out.println("✗ Class declaration failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testClassExpression() throws Exception {
        try {
            String source = "var Foo = class { constructor() { } };";
            Parser.parse(source);
            System.out.println("✓ Class expression works");
        } catch (Exception e) {
            System.out.println("✗ Class expression failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testClassWithPrivateField() throws Exception {
        try {
            String source = "class Foo { #x = 1; }";
            Parser.parse(source);
            System.out.println("✓ Class with private field works");
        } catch (Exception e) {
            System.out.println("✗ Class with private field failed: " + e.getMessage());
            throw e;
        }
    }

    @Test
    void testClassExtends() throws Exception {
        try {
            String source = "class Foo extends Bar { }";
            Parser.parse(source);
            System.out.println("✓ Class extends works");
        } catch (Exception e) {
            System.out.println("✗ Class extends failed: " + e.getMessage());
            throw e;
        }
    }
}
