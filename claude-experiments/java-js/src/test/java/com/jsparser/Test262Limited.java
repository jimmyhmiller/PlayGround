package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;

public class Test262Limited {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void checkFirst10Files() throws Exception {
        Path test262Dir = Paths.get("test-oracles/test262/test");
        Path cacheDir = Paths.get("test-oracles/test262-cache");

        // Start from built-ins directory to skip annexB
        Path startDir = Paths.get("test-oracles/test262/test/built-ins");

        int[] count = {0};
        int[] matched = {0};
        int[] mismatched = {0};

        try (Stream<Path> paths = Files.walk(startDir)) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".js"))
                 .limit(10)
                 .forEach(path -> {
                     count[0]++;

                     try {
                         String source = Files.readString(path);

                         // Get corresponding cache file
                         Path relativePath = test262Dir.relativize(path);
                         Path cacheFile = cacheDir.resolve(relativePath.toString() + ".json");

                         // If no cache, skip
                         if (!Files.exists(cacheFile)) {
                             System.out.println("No cache for: " + path);
                             return;
                         }

                         // Load expected AST from cache
                         String expectedJson = Files.readString(cacheFile);

                         // Check if file has module flag
                         boolean isModule = Parser.hasModuleFlag(source) ||
                             path.toString().endsWith("_FIXTURE.js");

                         // Parse with our parser
                         Program actualProgram = Parser.parse(source, isModule);
                         String actualJson = mapper.writeValueAsString(actualProgram);

                         // Parse both JSONs for structural comparison
                         Object expectedObj = mapper.readValue(expectedJson, Object.class);
                         Object actualObj = mapper.readValue(actualJson, Object.class);

                         // Remove _metadata field from expected
                         if (expectedObj instanceof Map) {
                             ((Map<?, ?>)expectedObj).remove("_metadata");
                         }

                         // Compare
                         if (Objects.deepEquals(expectedObj, actualObj)) {
                             matched[0]++;
                             System.out.println("✓ " + relativePath);
                         } else {
                             mismatched[0]++;
                             System.out.println("✗ " + relativePath);

                             // Show first difference
                             String expStr = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expectedObj);
                             String actStr = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actualObj);

                             if (!expStr.equals(actStr)) {
                                 System.out.println("  First 300 chars of expected:");
                                 System.out.println("  " + expStr.substring(0, Math.min(300, expStr.length())).replace("\n", "\n  "));
                                 System.out.println("  First 300 chars of actual:");
                                 System.out.println("  " + actStr.substring(0, Math.min(300, actStr.length())).replace("\n", "\n  "));
                             }
                         }
                     } catch (Exception e) {
                         System.out.println("✗ Error: " + path + ": " + e.getMessage());
                     }
                 });
        }

        System.out.println("\nSummary: " + matched[0] + " matched, " + mismatched[0] + " mismatched out of " + count[0]);
    }
}
