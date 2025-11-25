package com.jsparser;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestLargeNumbers {

    @Test
    public void testVeryLargeNumber() {
        // 10^16 - this is larger than Long but valid in JavaScript
        String code = "var x = 10000000000000000;";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should handle large numbers like 10000000000000000");
    }

    @Test
    public void testNumbersNearMaxSafeInteger() {
        // Numbers near JavaScript's MAX_SAFE_INTEGER (2^53 - 1)
        String code = """
            var a = 9007199254740991;
            var b = 9007199254740992;
            var c = 10000000000000000;
            """;

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should handle numbers near MAX_SAFE_INTEGER");
    }

    @Test
    public void testVeryLargeHexNumber() {
        // Large hex numbers
        String code = "var x = 0x10000000000000;";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should handle large hex numbers");
    }

    @Test
    public void testScientificNotationLargeNumbers() {
        // Scientific notation with large exponents
        String code = "var x = 1e16; var y = 1e20; var z = 1.23e15;";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should handle scientific notation with large exponents");
    }
}
