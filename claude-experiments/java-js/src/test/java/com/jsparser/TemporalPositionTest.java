package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class TemporalPositionTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void debugTemporalPosition() throws Exception {
        Path testFile = Paths.get("test-oracles/test262/test/staging/Temporal/Duration/old/balances-up-to-next-unit-after-rounding.js");
        Path cacheFile = Paths.get("test-oracles/test262-cache/staging/Temporal/Duration/old/balances-up-to-next-unit-after-rounding.js.json");

        String source = Files.readString(testFile);
        String expectedJson = Files.readString(cacheFile);

        Program actualProgram = Parser.parse(source);
        String actualJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actualProgram);

        String expectedPretty = mapper.writerWithDefaultPrettyPrinter()
            .writeValueAsString(mapper.readValue(expectedJson, Object.class));

        // Find first difference
        String[] expectedLines = expectedPretty.split("\n");
        String[] actualLines = actualJson.split("\n");

        for (int i = 0; i < Math.min(expectedLines.length, actualLines.length); i++) {
            if (!expectedLines[i].equals(actualLines[i])) {
                System.out.println("First diff at line " + (i+1) + ":");
                System.out.println("Expected: " + expectedLines[i]);
                System.out.println("Actual:   " + actualLines[i]);
                for (int j = 1; j <= 5 && i+j < Math.min(expectedLines.length, actualLines.length); j++) {
                    System.out.println("Expected: " + expectedLines[i+j]);
                    System.out.println("Actual:   " + actualLines[i+j]);
                }

                // Show source around the positions
                System.out.println("\nSource [280-310]: '" +
                    source.substring(280, Math.min(310, source.length())) + "'");
                System.out.println("Source [284-288]: '" + source.substring(284, 288) + "'");
                System.out.println("Source [288-292]: '" + source.substring(288, 292) + "'");

                // Count characters from start
                System.out.println("\nLine 10 content:");
                String[] lines = source.split("\n");
                if (lines.length >= 10) {
                    System.out.println("Line 10: '" + lines[9] + "'");
                    int posUpToLine10 = 0;
                    for (int k = 0; k < 9; k++) {
                        posUpToLine10 += lines[k].length() + 1; // +1 for newline
                    }
                    System.out.println("Position at start of line 10: " + posUpToLine10);
                }
                break;
            }
        }
    }
}
