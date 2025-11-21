package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.stream.Stream;

public class FindActualMismatchTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void findActualMismatches() throws Exception {
        Path test262Dir = Paths.get("test-oracles/test262/test");
        Path cacheDir = Paths.get("test-oracles/test262-cache");

        int checked = 0;
        int found = 0;

        try (Stream<Path> paths = Files.walk(test262Dir)) {
            for (Path path : paths.filter(Files::isRegularFile)
                    .filter(p -> p.toString().endsWith(".js"))
                    .limit(5000).toList()) {

                Path relativePath = test262Dir.relativize(path);
                Path cacheFile = cacheDir.resolve(relativePath.toString() + ".json");

                if (!Files.exists(cacheFile)) {
                    continue;
                }

                try {
                    String source = Files.readString(path);
                    String expectedJson = Files.readString(cacheFile);

                    Program actualProgram = Parser.parse(source);
                    String actualJson = mapper.writeValueAsString(actualProgram);

                    // Structural comparison like Test262Runner
                    Object expectedObj = mapper.readValue(expectedJson, Object.class);
                    Object actualObj = mapper.readValue(actualJson, Object.class);

                    checked++;

                    if (!java.util.Objects.deepEquals(expectedObj, actualObj)) {
                        found++;
                        if (found <= 10) {
                            System.out.println("\n=== MISMATCH #" + found + ": " + relativePath + " ===");
                            System.out.println("Source (first 100 chars): " + source.substring(0, Math.min(100, source.length())).replace("\n", " "));

                            // Find structural differences
                            findDifference("", expectedObj, actualObj);
                        }
                    }
                } catch (Exception e) {
                    // Skip failures
                }
            }
        }

        System.out.println("\n\nChecked: " + checked + " files, Found: " + found + " mismatches");
    }

    private void findDifference(String path, Object expected, Object actual) {
        if (java.util.Objects.deepEquals(expected, actual)) {
            return;
        }

        if (expected == null || actual == null) {
            System.out.println("  Null mismatch at " + path);
            System.out.println("    Expected: " + expected);
            System.out.println("    Actual: " + actual);
            return;
        }

        if (!expected.getClass().equals(actual.getClass())) {
            System.out.println("  Type mismatch at " + path);
            System.out.println("    Expected type: " + expected.getClass().getSimpleName());
            System.out.println("    Actual type: " + actual.getClass().getSimpleName());
            System.out.println("    Expected: " + expected);
            System.out.println("    Actual: " + actual);
            return;
        }

        if (expected instanceof java.util.Map) {
            java.util.Map<?, ?> expMap = (java.util.Map<?, ?>) expected;
            java.util.Map<?, ?> actMap = (java.util.Map<?, ?>) actual;

            // Check for missing keys
            for (Object key : expMap.keySet()) {
                if (!actMap.containsKey(key)) {
                    System.out.println("  Missing key at " + path + ": " + key);
                }
            }
            for (Object key : actMap.keySet()) {
                if (!expMap.containsKey(key)) {
                    System.out.println("  Extra key at " + path + ": " + key);
                }
            }

            // Compare values for common keys (limit output)
            int diffCount = 0;
            for (Object key : expMap.keySet()) {
                if (actMap.containsKey(key)) {
                    Object expVal = expMap.get(key);
                    Object actVal = actMap.get(key);
                    if (!java.util.Objects.deepEquals(expVal, actVal)) {
                        diffCount++;
                        if (diffCount <= 3) { // Only show first 3 differences
                            findDifference(path + "." + key, expVal, actVal);
                        }
                    }
                }
            }
            if (diffCount > 3) {
                System.out.println("  ... and " + (diffCount - 3) + " more differences in this object");
            }
        } else if (expected instanceof java.util.List) {
            java.util.List<?> expList = (java.util.List<?>) expected;
            java.util.List<?> actList = (java.util.List<?>) actual;

            if (expList.size() != actList.size()) {
                System.out.println("  List size mismatch at " + path);
                System.out.println("    Expected size: " + expList.size());
                System.out.println("    Actual size: " + actList.size());
                return;
            }

            int diffCount = 0;
            for (int i = 0; i < expList.size(); i++) {
                if (!java.util.Objects.deepEquals(expList.get(i), actList.get(i))) {
                    diffCount++;
                    if (diffCount <= 3) { // Only show first 3 differences
                        findDifference(path + "[" + i + "]", expList.get(i), actList.get(i));
                    }
                }
            }
            if (diffCount > 3) {
                System.out.println("  ... and " + (diffCount - 3) + " more differences in this array");
            }
        } else {
            System.out.println("  Value mismatch at " + path);
            System.out.println("    Expected: " + expected);
            System.out.println("    Actual: " + actual);
        }
    }
}
