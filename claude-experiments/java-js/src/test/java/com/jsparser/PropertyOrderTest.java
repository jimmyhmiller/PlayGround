package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.*;
import org.junit.jupiter.api.Test;

public class PropertyOrderTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void testPropertyFieldOrder() throws Exception {
        // Create a simple object with a property
        String source = "var obj = {foo: function() {}};";

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source);

        String expectedJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected);
        String actualJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual);

        System.out.println("=== EXPECTED ===");
        System.out.println(expectedJson);
        System.out.println("\n=== ACTUAL ===");
        System.out.println(actualJson);

        // Find first difference
        String[] expectedLines = expectedJson.split("\n");
        String[] actualLines = actualJson.split("\n");

        for (int i = 0; i < Math.min(expectedLines.length, actualLines.length); i++) {
            if (!expectedLines[i].equals(actualLines[i])) {
                System.out.println("\nFirst diff at line " + (i+1) + ":");
                System.out.println("Expected: " + expectedLines[i]);
                System.out.println("Actual:   " + actualLines[i]);
                if (i + 1 < expectedLines.length) System.out.println("Expected: " + expectedLines[i+1]);
                if (i + 1 < actualLines.length) System.out.println("Actual:   " + actualLines[i+1]);
                break;
            }
        }
    }
}
