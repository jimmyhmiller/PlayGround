package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for numeric literal overflow handling
 */
public class NumericOverflowTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void testLargeNumberBeyondLongMax() throws Exception {
        // 9223372036854776000 is larger than Long.MAX_VALUE (9223372036854775807)
        // Should be parsed as a double, not clamped to Long.MAX_VALUE
        String source = "const x = 9223372036854776000;";

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source, false);

        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        if (!expectedJson.equals(actualJson)) {
            System.out.println("EXPECTED:");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected));
            System.out.println("\nACTUAL:");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual));
        }

        assertEquals(expectedJson, actualJson, "AST should match oracle for large number beyond Long.MAX_VALUE");
    }

    @Test
    void testLongMaxValue() throws Exception {
        // Long.MAX_VALUE should still be parsed correctly
        String source = "const x = 9223372036854775807;";

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source, false);

        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        assertEquals(expectedJson, actualJson, "AST should match oracle for Long.MAX_VALUE");
    }

    @Test
    void testRegularIntegers() throws Exception {
        String source = "const a = 42; const b = 123456789;";

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source, false);

        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        assertEquals(expectedJson, actualJson, "AST should match oracle for regular integers");
    }

    @Test
    void testLargeIntegerRequiringLong() throws Exception {
        String source = "const x = 9007199254740992;"; // 2^53

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source, false);

        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        assertEquals(expectedJson, actualJson, "AST should match oracle for large integer");
    }

    @Test
    void testDoubleWithDecimalPoint() throws Exception {
        String source = "const x = 123.456;";

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source, false);

        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        assertEquals(expectedJson, actualJson, "AST should match oracle for decimal number");
    }
}
