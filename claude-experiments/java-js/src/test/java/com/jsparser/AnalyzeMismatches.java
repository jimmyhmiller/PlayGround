package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

public class AnalyzeMismatches {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void analyzeMismatchPatterns() throws IOException {
        Path test262Dir = Paths.get("test-oracles/test262/test");
        Path cacheDir = Paths.get("test-oracles/test262-cache");

        Map<String, Integer> mismatchTypes = new HashMap<>();
        Map<String, List<String>> typeToFiles = new HashMap<>();
        List<String> sampleMismatches = new ArrayList<>();

        System.out.println("Analyzing mismatch patterns...\n");

        try (Stream<Path> paths = Files.walk(test262Dir)) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".js"))
                 .forEach(path -> {
                     try {
                         String source = Files.readString(path);

                         // Get corresponding cache file
                         Path relativePath = test262Dir.relativize(path);
                         Path cacheFile = cacheDir.resolve(relativePath.toString() + ".json");

                         if (!Files.exists(cacheFile)) {
                             return;
                         }

                         // Load expected AST from cache
                         String expectedJson = Files.readString(cacheFile);

                         // Parse with our parser
                         Program actualProgram = Parser.parse(source);
                         String actualJson = mapper.writeValueAsString(actualProgram);

                         // Parse both JSONs
                         JsonNode expected = mapper.readTree(expectedJson);
                         JsonNode actual = mapper.readTree(actualJson);

                         // Check if they match
                         if (!expected.equals(actual)) {
                             String mismatchType = findMismatchType(expected, actual, "");

                             mismatchTypes.merge(mismatchType, 1, Integer::sum);
                             typeToFiles.computeIfAbsent(mismatchType, k -> new ArrayList<>()).add(path.toString());

                             if (sampleMismatches.size() < 50) {
                                 sampleMismatches.add(path.toString() + " -> " + mismatchType);
                             }
                         }
                     } catch (Exception e) {
                         // Skip parse errors
                     }
                 });
        }

        // Print results sorted by count
        System.out.println("=== Mismatch Type Summary ===\n");
        mismatchTypes.entrySet().stream()
            .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
            .forEach(entry -> {
                System.out.printf("%-60s: %4d files\n", entry.getKey(), entry.getValue());

                // Print first 3 examples
                List<String> examples = typeToFiles.get(entry.getKey());
                if (examples != null && !examples.isEmpty()) {
                    for (int i = 0; i < Math.min(3, examples.size()); i++) {
                        String file = examples.get(i).replace("test-oracles/test262/test/", "");
                        System.out.printf("  → %s\n", file);
                    }
                }
                System.out.println();
            });

        System.out.println("\n=== First 50 Sample Mismatches ===\n");
        for (String sample : sampleMismatches) {
            System.out.println(sample);
        }
    }

    private String findMismatchType(JsonNode expected, JsonNode actual, String path) {
        if (expected == null && actual == null) {
            return "both-null";
        }
        if (expected == null) {
            return path + ": expected-null";
        }
        if (actual == null) {
            return path + ": actual-null";
        }

        if (expected.isObject() && actual.isObject()) {
            // Check for different fields
            Set<String> expectedFields = new HashSet<>();
            Set<String> actualFields = new HashSet<>();
            expected.fieldNames().forEachRemaining(expectedFields::add);
            actual.fieldNames().forEachRemaining(actualFields::add);

            if (!expectedFields.equals(actualFields)) {
                Set<String> onlyInExpected = new HashSet<>(expectedFields);
                onlyInExpected.removeAll(actualFields);
                Set<String> onlyInActual = new HashSet<>(actualFields);
                onlyInActual.removeAll(expectedFields);

                if (!onlyInExpected.isEmpty()) {
                    return "missing-field: " + path + "." + onlyInExpected.iterator().next();
                }
                if (!onlyInActual.isEmpty()) {
                    return "extra-field: " + path + "." + onlyInActual.iterator().next();
                }
            }

            // Check common fields
            for (String field : expectedFields) {
                if (actualFields.contains(field)) {
                    JsonNode expChild = expected.get(field);
                    JsonNode actChild = actual.get(field);
                    if (!expChild.equals(actChild)) {
                        // Special handling for common differences
                        if (field.equals("start") || field.equals("end")) {
                            return "position-offset: " + path + "." + field;
                        }
                        if (field.equals("type")) {
                            return "type-mismatch: " + path + " (expected=" + expChild.asText() + ", actual=" + actChild.asText() + ")";
                        }
                        if (field.equals("sourceType")) {
                            return "sourceType-mismatch: " + path;
                        }
                        if (field.equals("value") || field.equals("raw")) {
                            return "value-mismatch: " + path + "." + field;
                        }

                        return findMismatchType(expChild, actChild, path + "." + field);
                    }
                }
            }
        } else if (expected.isArray() && actual.isArray()) {
            if (expected.size() != actual.size()) {
                return "array-length: " + path + " (expected=" + expected.size() + ", actual=" + actual.size() + ")";
            }
            for (int i = 0; i < expected.size(); i++) {
                if (!expected.get(i).equals(actual.get(i))) {
                    return findMismatchType(expected.get(i), actual.get(i), path + "[" + i + "]");
                }
            }
        } else if (!expected.equals(actual)) {
            return "value-diff: " + path + " (expected=" + expected + ", actual=" + actual + ")";
        }

        return "unknown-mismatch";
    }
}
