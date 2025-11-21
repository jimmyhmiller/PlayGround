package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Paths;

public class TestMethodValue {

    @Test
    void testAsyncMethodValue() throws Exception {
        String code = Files.readString(Paths.get("test-oracles/test262/test/built-ins/Function/prototype/toString/async-method-class-statement.js"));
        Program ast = Parser.parse(code);

        ObjectMapper mapper = new ObjectMapper();
        JsonNode tree = mapper.readTree(mapper.writeValueAsString(ast));
        JsonNode methodValue = tree.get("body").get(1).get("body").get("body").get(0).get("value");

        // Load expected
        String expectedJson = Files.readString(Paths.get("test-oracles/test262-cache/built-ins/Function/prototype/toString/async-method-class-statement.js.json"));
        JsonNode expected = mapper.readTree(expectedJson);
        JsonNode expectedValue = expected.get("body").get(1).get("body").get("body").get(0).get("value");

        System.out.println("Expected value:");
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expectedValue));
        System.out.println("\nActual value:");
        System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(methodValue));

        // Compare
        if (!methodValue.equals(expectedValue)) {
            System.out.println("\nMISMATCH!");
            // Find the specific difference
            System.out.println("\nExpected async: " + expectedValue.get("async"));
            System.out.println("Actual async: " + methodValue.get("async"));
            System.out.println("\nExpected expression: " + expectedValue.get("expression"));
            System.out.println("Actual expression: " + methodValue.get("expression"));
            System.out.println("\nExpected generator: " + expectedValue.get("generator"));
            System.out.println("Actual generator: " + methodValue.get("generator"));
        } else {
            System.out.println("\nMATCH!");
        }
    }
}
