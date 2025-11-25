package com.jsparser;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

public class TestReservedWordsAsProperties {

    @Test
    public void testFalseAsPropertyName() {
        // Reserved words can be used as property names in ES5+
        String code = "var obj = { false: 1, true: 2, null: 3 };";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should allow 'false' as property name");
    }

    @Test
    public void testKeywordsAsPropertyNames() {
        // All keywords should be allowed as property names
        String code = """
            var obj = {
                if: 1,
                while: 2,
                for: 3,
                function: 4,
                class: 5,
                return: 6,
                var: 7,
                let: 8,
                const: 9
            };
            """;

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should allow keywords as property names");
    }

    @Test
    public void testReservedWordsInMemberExpression() {
        // Reserved words can be used after dot in member expressions
        String code = "obj.false; obj.true; obj.null; obj.if; obj.while;";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should allow reserved words in member expressions");
    }

    @Test
    public void testReservedWordsInComputedProperties() {
        // Reserved words in computed property names
        String code = "var obj = { ['false']: 1 }; obj['false'];";

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should allow reserved words in computed properties");
    }

    @Test
    public void testMethodsWithReservedWordNames() {
        // Methods can have reserved word names
        String code = """
            var obj = {
                false() { return 1; },
                if() { return 2; }
            };
            """;

        assertDoesNotThrow(() -> {
            Parser.parse(code, false);
        }, "Should allow reserved words as method names");
    }
}
