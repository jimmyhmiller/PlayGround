package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class MethodPropertyTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void testMethodProperty() throws Exception {
        String source = "var obj = {foo: function() {}};";

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source);

        String expectedJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected);
        String actualJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual);

        Object expectedObj = mapper.readValue(expectedJson, Object.class);
        Object actualObj = mapper.readValue(actualJson, Object.class);

        boolean match = java.util.Objects.deepEquals(expectedObj, actualObj);
        System.out.println("Structural equals: " + match);

        if (!match) {
            String[] expectedLines = expectedJson.split("\n");
            String[] actualLines = actualJson.split("\n");

            for (int i = 0; i < Math.min(expectedLines.length, actualLines.length); i++) {
                if (!expectedLines[i].equals(actualLines[i])) {
                    System.out.println("\nFirst diff at line " + (i+1) + ":");
                    System.out.println("Expected: " + expectedLines[i]);
                    System.out.println("Actual:   " + actualLines[i]);
                    for (int j = 1; j <= 3 && i+j < Math.min(expectedLines.length, actualLines.length); j++) {
                        System.out.println("Expected: " + expectedLines[i+j]);
                        System.out.println("Actual:   " + actualLines[i+j]);
                    }
                    break;
                }
            }
        }
    }
}
