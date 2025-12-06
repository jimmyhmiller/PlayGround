package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Objects;

public class DebugSingleMismatch {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void debugFirstFile() throws Exception {
        Path cacheFile = Paths.get("test-oracles/test262-cache/built-ins/decodeURIComponent/S15.1.3.2_A1.1_T1.js.json");
        Path sourceFile = Paths.get("test-oracles/test262/test/built-ins/decodeURIComponent/S15.1.3.2_A1.1_T1.js");

        // Read expected
        String expectedJson = Files.readString(cacheFile);
        Object expectedObj = mapper.readValue(expectedJson, Object.class);

        // Remove metadata
        if (expectedObj instanceof Map) {
            ((Map<?, ?>)expectedObj).remove("_metadata");
        }

        // Read source and parse
        String source = Files.readString(sourceFile);
        Program actualProgram = Parser.parse(source, false);
        String actualJson = mapper.writeValueAsString(actualProgram);
        Object actualObj = mapper.readValue(actualJson, Object.class);

        // Compare
        if (Objects.deepEquals(expectedObj, actualObj)) {
            System.out.println("✓ Match!");
        } else {
            System.out.println("✗ Mismatch!");
            System.out.println("\n=== Expected (first 500 chars) ===");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expectedObj).substring(0, Math.min(500, expectedJson.length())));
            System.out.println("\n=== Actual (first 500 chars) ===");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actualObj).substring(0, Math.min(500, actualJson.length())));
        }
    }
}
