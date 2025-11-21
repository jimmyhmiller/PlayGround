package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

public class SimplePropertyTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void testSimpleProperty() throws Exception {
        String source = "var obj = {foo: 1};";

        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source);

        String expectedJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected);
        String actualJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual);

        System.out.println("=== EXPECTED ===");
        System.out.println(expectedJson);
        System.out.println("\n=== ACTUAL ===");
        System.out.println(actualJson);

        Object expectedObj = mapper.readValue(expectedJson, Object.class);
        Object actualObj = mapper.readValue(actualJson, Object.class);

        System.out.println("\nStructural equals: " + java.util.Objects.deepEquals(expectedObj, actualObj));
    }
}
