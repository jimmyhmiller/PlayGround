package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

public class ObjectTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        // Note: {}, {x:1} etc in statement position are BlockStatements, not ObjectExpressions
        // We need to wrap in parens to make them expressions
        "({});",
        "({x: 1});",
        "({x: 1, y: 2});",
        "({\"key\": \"value\"});",
        "new Error();",
        "new Error(\"message\");",
        "new Error(\"message\", 123);",
    })
    void testObjectsAgainstOracle(String source) throws Exception {
        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source);

        assertNotNull(expected);
        assertNotNull(actual);

        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        System.out.println("Testing: " + source);
        if (!expectedJson.equals(actualJson)) {
            System.out.println("EXPECTED:");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected));
            System.out.println("\nACTUAL:");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual));
        }

        assertEquals(expectedJson, actualJson, "AST mismatch for: " + source);
    }
}
