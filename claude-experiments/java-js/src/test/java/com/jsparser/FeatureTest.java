package com.jsparser;

import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.*;

public class FeatureTest {

    @Test
    @DisplayName("Features that WORK")
    void testWorkingFeatures() {
        // All of these should parse successfully
        assertDoesNotThrow(() -> Parser.parse("42;"));
        assertDoesNotThrow(() -> Parser.parse("\"hello\";"));
        assertDoesNotThrow(() -> Parser.parse("true;"));
        assertDoesNotThrow(() -> Parser.parse("x;"));
        assertDoesNotThrow(() -> Parser.parse("1 + 2;"));
        assertDoesNotThrow(() -> Parser.parse("x = 5;"));
        assertDoesNotThrow(() -> Parser.parse("obj.prop;"));
        assertDoesNotThrow(() -> Parser.parse("func(1, 2);"));
        assertDoesNotThrow(() -> Parser.parse("[1, 2, 3];"));
        assertDoesNotThrow(() -> Parser.parse("({x: 1, y: 2});"));
        assertDoesNotThrow(() -> Parser.parse("new Error(\"msg\");"));
        assertDoesNotThrow(() -> Parser.parse("a.b.c[d](e, f);"));
        assertDoesNotThrow(() -> Parser.parse("var x = 1;"));
        assertDoesNotThrow(() -> Parser.parse("let y = 2;"));
        assertDoesNotThrow(() -> Parser.parse("const PI = 3.14;"));
        assertDoesNotThrow(() -> Parser.parse("var x = 1, y = 2, z = 3;"));
        assertDoesNotThrow(() -> Parser.parse("{ }"));
        assertDoesNotThrow(() -> Parser.parse("{ var x = 1; }"));
        assertDoesNotThrow(() -> Parser.parse("!x;"));
        assertDoesNotThrow(() -> Parser.parse("-5;"));
        assertDoesNotThrow(() -> Parser.parse("typeof x;"));
        assertDoesNotThrow(() -> Parser.parse("x && y;"));
        assertDoesNotThrow(() -> Parser.parse("x || y;"));
        assertDoesNotThrow(() -> Parser.parse("x++;"));
        assertDoesNotThrow(() -> Parser.parse("++x;"));
        assertDoesNotThrow(() -> Parser.parse("if (x) y;"));
        assertDoesNotThrow(() -> Parser.parse("if (x) y; else z;"));
        assertDoesNotThrow(() -> Parser.parse("while (x) y;"));
        assertDoesNotThrow(() -> Parser.parse("while (x > 0) x--;"));
        assertDoesNotThrow(() -> Parser.parse("do x; while (y);"));
        assertDoesNotThrow(() -> Parser.parse("do { x++; } while (x < 10);"));
        assertDoesNotThrow(() -> Parser.parse("for (;;) x;"));
        assertDoesNotThrow(() -> Parser.parse("for (var i = 0; i < 10; i++) x;"));
        assertDoesNotThrow(() -> Parser.parse("for (let i = 0; i < 10; i++) x++;"));
        assertDoesNotThrow(() -> Parser.parse("while (true) break;"));
        assertDoesNotThrow(() -> Parser.parse("for (;;) { if (x) continue; }"));
        assertDoesNotThrow(() -> Parser.parse("x ? y : z;"));
        assertDoesNotThrow(() -> Parser.parse("var a = x > 0 ? x : -x;"));
        assertDoesNotThrow(() -> Parser.parse("function foo() {}"));
        assertDoesNotThrow(() -> Parser.parse("function add(a, b) { return a + b; }"));
        assertDoesNotThrow(() -> Parser.parse("var f = function() {};"));
        assertDoesNotThrow(() -> Parser.parse("var add = function(a, b) { return a + b; };"));
        assertDoesNotThrow(() -> Parser.parse("this;"));
        assertDoesNotThrow(() -> Parser.parse("this.x;"));
        assertDoesNotThrow(() -> Parser.parse("var f = () => 42;"));
        assertDoesNotThrow(() -> Parser.parse("var f = x => x + 1;"));
        assertDoesNotThrow(() -> Parser.parse("var f = (x, y) => x + y;"));
        assertDoesNotThrow(() -> Parser.parse("var s = `hello`;"));
        assertDoesNotThrow(() -> Parser.parse("var s = `hello ${world}`;"));
        assertDoesNotThrow(() -> Parser.parse("var x = a & b;"));
        assertDoesNotThrow(() -> Parser.parse("var x = a | b;"));
        assertDoesNotThrow(() -> Parser.parse("var x = a ^ b;"));
        assertDoesNotThrow(() -> Parser.parse("var x = a << 2;"));
        assertDoesNotThrow(() -> Parser.parse("var x = a >> 2;"));
        assertDoesNotThrow(() -> Parser.parse("var x = a >>> 2;"));

        System.out.println("✓ All supported features work correctly!");
    }

    @Test
    @DisplayName("Features that DON'T WORK (expected failures)")
    void testMissingFeatures() {
        // Test throw statement
        System.out.println("\nTesting missing features:");
        testFeature("throw new Error('test');", "throw statement");
        testFeature("try { x(); } catch(e) { }", "try-catch");
        testFeature("try { x(); } finally { y(); }", "try-finally");
        testFeature("try { x(); } catch(e) { } finally { y(); }", "try-catch-finally");
        testFeature("[...arr]", "spread in array");
        testFeature("var x = { ...y };", "spread in object");
        testFeature("function f(...args) { }", "rest parameters");
        testFeature("var [a, b] = arr;", "array destructuring");
        testFeature("var {a, b} = obj;", "object destructuring");
        testFeature("for (var x of arr) { }", "for-of loop");
        testFeature("for (var x in obj) { }", "for-in loop");
        testFeature("debugger;", "debugger statement");
        testFeature("with (obj) { x; }", "with statement");
        testFeature("switch (x) { case 1: break; }", "switch statement");
        testFeature("var f = function*() { yield 1; }", "generator function");
        testFeature("async function f() { }", "async function");
        testFeature("await x;", "await expression");
    }

    private void testFeature(String code, String featureName) {
        try {
            Parser.parse(code);
            System.out.println("  ✓ " + featureName + " - WORKS!");
        } catch (Exception e) {
            System.out.println("  ✗ " + featureName + " - " + e.getMessage().substring(0, Math.min(80, e.getMessage().length())));
        }
    }
}
