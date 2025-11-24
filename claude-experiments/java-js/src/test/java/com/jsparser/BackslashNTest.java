package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test for backslash-n escape sequence parsing
 */
public class BackslashNTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void testBackslashN() throws Exception {
        String source = "var x = \"\\n\";";

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

        assertEquals(expectedJson, actualJson, "Should parse backslash-n correctly");
    }

    @Test
    void testComplexString() throws Exception {
        String source = "var x = \"completed()+\\\"\\n\\\"\";";

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

        assertEquals(expectedJson, actualJson, "Should parse complex string with escapes");
    }
}
