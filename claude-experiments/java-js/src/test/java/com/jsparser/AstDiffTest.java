package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class AstDiffTest {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void compareSingleFile() throws IOException {
        // Get the first mismatched file
        String cacheFile = "test-oracles/adhoc-cache/simple-nextjs-demo/.._.._.._simple-nextjs-demo_simple-nextjs-demo_node_modules_@emnapi_core_dist_emnapi-core.cjs.js.json";

        if (!Files.exists(Paths.get(cacheFile))) {
            System.out.println("Cache file not found: " + cacheFile);
            return;
        }

        // Load expected AST from cache
        String expectedJson = Files.readString(Paths.get(cacheFile));

        @SuppressWarnings("unchecked")
        Map<String, Object> expectedMap = mapper.readValue(expectedJson, Map.class);

        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = (Map<String, Object>) expectedMap.get("_metadata");

        if (metadata == null) {
            System.out.println("No metadata found");
            return;
        }

        String sourceFile = (String) metadata.get("sourceFile");
        String sourceType = (String) metadata.get("sourceType");

        System.out.println("Analyzing: " + sourceFile);
        System.out.println("Source type: " + sourceType);
        System.out.println();

        // The metadata sourceFile is relative to where test-package.js was run
        // For now, hardcode the correct path
        Path actualSourcePath = Paths.get("../simple-nextjs-demo/simple-nextjs-demo/node_modules/@emnapi/core/dist/emnapi-core.cjs.js");
        System.out.println("Looking for file at: " + actualSourcePath.toAbsolutePath());

        if (!Files.exists(actualSourcePath)) {
            System.out.println("Source file not found: " + actualSourcePath);
            return;
        }

        // Read source
        String source = Files.readString(actualSourcePath);

        // Parse with our parser
        boolean isModule = "module".equals(sourceType);
        Program actualProgram = Parser.parse(source, isModule);
        String actualJson = mapper.writeValueAsString(actualProgram);

        // Parse both JSONs for structural comparison
        Object actualObj = mapper.readValue(actualJson, Object.class);

        // Remove _metadata field from expected
        expectedMap.remove("_metadata");

        // Normalize numeric values (same as AdhocPackageTest)
        normalizeNumericValues(expectedMap, actualObj);

        // Find differences
        List<String> differences = new ArrayList<>();
        findDifferences(expectedMap, actualObj, "", differences);

        if (differences.isEmpty()) {
            System.out.println("✓ ASTs match!");
        } else {
            System.out.println("✗ Found " + differences.size() + " differences:");
            System.out.println();

            // Group and count by pattern
            Map<String, Integer> patterns = new HashMap<>();
            for (String diff : differences) {
                String pattern = extractPattern(diff);
                patterns.merge(pattern, 1, Integer::sum);
            }

            // Show patterns
            patterns.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEach(e -> {
                    System.out.printf("[%d occurrences] %s%n", e.getValue(), e.getKey());
                });

            System.out.println();
            System.out.println("First 20 specific differences:");
            differences.stream().limit(20).forEach(d -> System.out.println("  " + d));
        }
    }

    private String extractPattern(String diff) {
        // Extract the type of difference (first part before the colon)
        int colonIndex = diff.indexOf(':');
        if (colonIndex > 0) {
            return diff.substring(0, colonIndex);
        }
        return diff;
    }

    @SuppressWarnings("unchecked")
    private void findDifferences(Object expected, Object actual, String path, List<String> differences) {
        if (expected == null && actual == null) {
            return;
        }

        if (expected == null || actual == null) {
            differences.add(path + ": null mismatch (expected=" + expected + ", actual=" + actual + ")");
            return;
        }

        if (expected.getClass() != actual.getClass()) {
            differences.add(path + ": type mismatch (expected=" + expected.getClass().getSimpleName() +
                ", actual=" + actual.getClass().getSimpleName() + ")");
            return;
        }

        if (expected instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            Set<String> allKeys = new HashSet<>();
            allKeys.addAll(expMap.keySet());
            allKeys.addAll(actMap.keySet());

            for (String key : allKeys) {
                String newPath = path.isEmpty() ? key : path + "." + key;

                if (!expMap.containsKey(key)) {
                    differences.add(newPath + ": extra key in actual");
                    continue;
                }

                if (!actMap.containsKey(key)) {
                    differences.add(newPath + ": missing key in actual");
                    continue;
                }

                findDifferences(expMap.get(key), actMap.get(key), newPath, differences);
            }
        } else if (expected instanceof List) {
            List<Object> expList = (List<Object>) expected;
            List<Object> actList = (List<Object>) actual;

            if (expList.size() != actList.size()) {
                differences.add(path + ": array length mismatch (expected=" + expList.size() +
                    ", actual=" + actList.size() + ")");
            }

            int minSize = Math.min(expList.size(), actList.size());
            for (int i = 0; i < minSize; i++) {
                findDifferences(expList.get(i), actList.get(i), path + "[" + i + "]", differences);
            }
        } else {
            // Check for numeric equivalence (e.g., BigInteger vs Long)
            if (expected instanceof Number && actual instanceof Number) {
                // Try to compare as numbers
                try {
                    java.math.BigDecimal expNum = new java.math.BigDecimal(expected.toString());
                    java.math.BigDecimal actNum = new java.math.BigDecimal(actual.toString());
                    if (expNum.compareTo(actNum) != 0) {
                        differences.add(path + ": numeric value mismatch (expected=" + expected +
                            ", actual=" + actual + ")");
                    }
                    // They're equal numerically, no difference
                    return;
                } catch (NumberFormatException e) {
                    // Fall through to regular comparison
                }
            }

            if (!Objects.deepEquals(expected, actual)) {
                String expStr = expected.toString();
                String actStr = actual.toString();
                String expType = expected.getClass().getSimpleName();
                String actType = actual.getClass().getSimpleName();
                if (expStr.length() > 50) expStr = expStr.substring(0, 50) + "...";
                if (actStr.length() > 50) actStr = actStr.substring(0, 50) + "...";
                differences.add(path + ": value mismatch [" + expType + " vs " + actType +
                    "] (expected=" + expStr + ", actual=" + actStr + ")");
            }
        }
    }

    @SuppressWarnings("unchecked")
    private void normalizeNumericValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            // Normalize all numeric values in both maps
            for (String key : new ArrayList<>(expMap.keySet())) {
                if (actMap.containsKey(key)) {
                    Object expVal = expMap.get(key);
                    Object actVal = actMap.get(key);

                    // If both are numbers but different types, normalize to expected type
                    if (expVal instanceof Number && actVal instanceof Number &&
                        expVal.getClass() != actVal.getClass()) {

                        // Try to normalize the actual value to match expected type
                        try {
                            if (expVal instanceof java.math.BigInteger) {
                                actMap.put(key, new java.math.BigInteger(actVal.toString()));
                            } else if (expVal instanceof Long) {
                                actMap.put(key, Long.valueOf(actVal.toString()));
                            } else if (expVal instanceof Integer) {
                                actMap.put(key, Integer.valueOf(actVal.toString()));
                            } else if (expVal instanceof Double) {
                                actMap.put(key, Double.valueOf(actVal.toString()));
                            }
                        } catch (NumberFormatException e) {
                            // If can't convert, leave as is
                        }
                    } else {
                        // Recursively normalize nested structures
                        normalizeNumericValues(expVal, actVal);
                    }
                }
            }
        } else if (expected instanceof List && actual instanceof List) {
            List<Object> expList = (List<Object>) expected;
            List<Object> actList = (List<Object>) actual;
            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {
                normalizeNumericValues(expList.get(i), actList.get(i));
            }
        }
    }
}
