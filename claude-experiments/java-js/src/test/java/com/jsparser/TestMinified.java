package com.jsparser;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestMinified {

    @Test
    public void testMinifiedCodeWithReservedWords() {
        // Minified code often uses reserved words as property names without quotes
        // This is the pattern that might be failing: {false:something}
        String code = "var x={false:1,true:2,null:3,if:4,while:5};";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should parse minified code with reserved word properties");
    }

    @Test
    public void testDestructuringWithReservedWords() {
        // Destructuring with reserved words
        String code = "var {false:f,true:t}={false:1,true:2};";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should parse destructuring with reserved word properties");
    }

    @Test
    public void testNumberExceedingLongRange() {
        // Test the specific number from the error: 10000000000000000
        String code = "var big=10000000000000000;var obj={value:10000000000000000};";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should handle numbers exceeding long range");
    }

    @Test
    public void testComplexMinifiedPattern() {
        // Complex pattern that might appear in minified code
        String code = "({false:function(){return 10000000000000000}});";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should handle complex minified patterns");
    }
}
