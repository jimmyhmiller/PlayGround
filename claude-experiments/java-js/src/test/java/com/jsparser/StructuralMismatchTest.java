package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Objects;

public class StructuralMismatchTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    void debugStructuralMismatch() throws Exception {
        Path testFile = Paths.get("test-oracles/test262/test/staging/sm/TypedArray/sort_snans.js");
        Path cacheFile = Paths.get("test-oracles/test262-cache/staging/sm/TypedArray/sort_snans.js.json");

        String source = Files.readString(testFile);
        String expectedJson = Files.readString(cacheFile);

        Program actualProgram = Parser.parse(source);

        // Parse both to Map for structural comparison
        Map<?, ?> expectedObj = mapper.readValue(expectedJson, Map.class);
        Map<?, ?> actualObj = mapper.readValue(mapper.writeValueAsString(actualProgram), Map.class);

        System.out.println("Are they structurally equal? " + Objects.deepEquals(expectedObj, actualObj));

        if (!Objects.deepEquals(expectedObj, actualObj)) {
            // Find the actual difference
            findDifference("", expectedObj, actualObj);
        }
    }

    private void findDifference(String path, Object expected, Object actual) {
        if (Objects.deepEquals(expected, actual)) {
            return;
        }

        if (expected == null || actual == null) {
            System.out.println("Null mismatch at " + path);
            System.out.println("  Expected: " + expected);
            System.out.println("  Actual: " + actual);
            return;
        }

        if (!expected.getClass().equals(actual.getClass())) {
            System.out.println("Type mismatch at " + path);
            System.out.println("  Expected type: " + expected.getClass());
            System.out.println("  Actual type: " + actual.getClass());
            System.out.println("  Expected: " + expected);
            System.out.println("  Actual: " + actual);
            return;
        }

        if (expected instanceof Map) {
            Map<?, ?> expMap = (Map<?, ?>) expected;
            Map<?, ?> actMap = (Map<?, ?>) actual;

            // Check for missing keys
            for (Object key : expMap.keySet()) {
                if (!actMap.containsKey(key)) {
                    System.out.println("Missing key at " + path + ": " + key);
                }
            }
            for (Object key : actMap.keySet()) {
                if (!expMap.containsKey(key)) {
                    System.out.println("Extra key at " + path + ": " + key);
                }
            }

            // Compare values for common keys
            for (Object key : expMap.keySet()) {
                if (actMap.containsKey(key)) {
                    Object expVal = expMap.get(key);
                    Object actVal = actMap.get(key);
                    if (!Objects.deepEquals(expVal, actVal)) {
                        findDifference(path + "." + key, expVal, actVal);
                    }
                }
            }
        } else if (expected instanceof java.util.List) {
            java.util.List<?> expList = (java.util.List<?>) expected;
            java.util.List<?> actList = (java.util.List<?>) actual;

            if (expList.size() != actList.size()) {
                System.out.println("List size mismatch at " + path);
                System.out.println("  Expected size: " + expList.size());
                System.out.println("  Actual size: " + actList.size());
                return;
            }

            for (int i = 0; i < expList.size(); i++) {
                if (!Objects.deepEquals(expList.get(i), actList.get(i))) {
                    findDifference(path + "[" + i + "]", expList.get(i), actList.get(i));
                }
            }
        } else {
            System.out.println("Value mismatch at " + path);
            System.out.println("  Expected: " + expected);
            System.out.println("  Actual: " + actual);
        }
    }
}
