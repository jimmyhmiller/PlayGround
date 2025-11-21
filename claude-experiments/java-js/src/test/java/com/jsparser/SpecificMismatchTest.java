package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class SpecificMismatchTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void debugSpecificFile() throws Exception {
        Path testFile = Paths.get("test-oracles/test262/test/harness/verifyProperty-restore-accessor.js");
        Path cacheFile = Paths.get("test-oracles/test262-cache/harness/verifyProperty-restore-accessor.js.json");

        String source = Files.readString(testFile);
        String expectedJson = Files.readString(cacheFile);

        Program actualProgram = Parser.parse(source);
        String actualJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actualProgram);

        // Structural comparison
        Object expectedObj = mapper.readValue(expectedJson, Object.class);
        Object actualObj = mapper.readValue(actualJson, Object.class);

        System.out.println("Structural equals: " + java.util.Objects.deepEquals(expectedObj, actualObj));
        System.out.println("String equals: " + expectedJson.equals(actualJson));

        if (java.util.Objects.deepEquals(expectedObj, actualObj)) {
            System.out.println("MATCH!");
        } else {
            System.out.println("MISMATCH!");

            // Pretty print both for debugging
            String expectedPretty = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expectedObj);
            String actualPretty = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actualObj);

            // Find first difference
            String[] expectedLines = expectedPretty.split("\n");
            String[] actualLines = actualPretty.split("\n");

            System.out.println("\nComparing " + expectedLines.length + " expected lines vs " + actualLines.length + " actual lines");

            for (int i = 0; i < Math.min(expectedLines.length, actualLines.length); i++) {
                if (!expectedLines[i].equals(actualLines[i])) {
                    System.out.println("\nFirst diff at line " + (i+1) + ":");
                    System.out.println("Expected: " + expectedLines[i]);
                    System.out.println("Actual:   " + actualLines[i]);
                    if (i + 1 < Math.min(expectedLines.length, actualLines.length)) {
                        System.out.println("Expected: " + expectedLines[i+1]);
                        System.out.println("Actual:   " + actualLines[i+1]);
                    }
                    if (i + 2 < Math.min(expectedLines.length, actualLines.length)) {
                        System.out.println("Expected: " + expectedLines[i+2]);
                        System.out.println("Actual:   " + actualLines[i+2]);
                    }
                    break;
                }
            }
        }
    }
}
