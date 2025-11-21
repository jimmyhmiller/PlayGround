package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class PositionMismatchTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void debugPositionMismatch() throws Exception {
        // Test one of the files with position mismatch
        Path testFile = Paths.get("test-oracles/test262/test/harness/verifyProperty-undefined-desc.js");
        Path cacheFile = Paths.get("test-oracles/test262-cache/harness/verifyProperty-undefined-desc.js.json");

        String source = Files.readString(testFile);
        String expectedJson = Files.readString(cacheFile);

        Program actualProgram = Parser.parse(source);
        String actualJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actualProgram);

        String expectedPretty = mapper.writerWithDefaultPrettyPrinter()
            .writeValueAsString(mapper.readValue(expectedJson, Object.class));

        // Find first difference
        String[] expectedLines = expectedPretty.split("\n");
        String[] actualLines = actualJson.split("\n");

        System.out.println("Source length: " + source.length());
        System.out.println("Expected lines: " + expectedLines.length);
        System.out.println("Actual lines: " + actualLines.length);

        for (int i = 0; i < Math.min(expectedLines.length, actualLines.length); i++) {
            if (!expectedLines[i].equals(actualLines[i])) {
                System.out.println("\nFirst diff at line " + (i+1) + ":");
                System.out.println("Expected: " + expectedLines[i]);
                System.out.println("Actual:   " + actualLines[i]);
                for (int j = 1; j <= 5 && i+j < Math.min(expectedLines.length, actualLines.length); j++) {
                    System.out.println("Expected: " + expectedLines[i+j]);
                    System.out.println("Actual:   " + actualLines[i+j]);
                }

                // Show source around position 252-264
                if (source.length() >= 264) {
                    System.out.println("\nSource [240-270]: '" +
                        source.substring(240, Math.min(270, source.length())) + "'");
                    System.out.println("Source [252-256]: '" +
                        source.substring(252, 256) + "'");
                }
                break;
            }
        }
    }
}
