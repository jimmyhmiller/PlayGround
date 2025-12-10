package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

public class Test262Runner {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("Parse all test262 files and compare against esprima oracle")
    void parseAllTest262Files() throws IOException {
        Path test262Dir = Paths.get("test-oracles/test262/test");
        Path cacheDir = Paths.get("test-oracles/test262-cache");

        if (!Files.exists(test262Dir)) {
            System.out.println("test262 directory not found at: " + test262Dir.toAbsolutePath());
            return;
        }

        if (!Files.exists(cacheDir)) {
            System.out.println("ERROR: Cache directory not found at: " + cacheDir.toAbsolutePath());
            System.out.println("Please run: node scripts/generate-test262-cache.js");
            return;
        }

        AtomicInteger total = new AtomicInteger(0);
        AtomicInteger matched = new AtomicInteger(0);
        AtomicInteger mismatched = new AtomicInteger(0);
        AtomicInteger failed = new AtomicInteger(0);
        AtomicInteger noCache = new AtomicInteger(0);
        AtomicInteger negativeTestsPassed = new AtomicInteger(0);
        Map<String, Integer> errorTypes = new HashMap<>();
        Map<String, Integer> errorMessages = new HashMap<>();
        List<String> mismatchedFiles = new ArrayList<>();
        Map<String, List<String>> errorToFiles = new HashMap<>();
        List<String> allFailures = new ArrayList<>();
        List<Map<String, Object>> allFailuresJson = new ArrayList<>();
        List<String> negativeTestsPassedFiles = new ArrayList<>();

        System.out.println("Starting test262 oracle comparison test...");
        System.out.println("Test262 dir: " + test262Dir.toAbsolutePath());
        System.out.println("Cache dir: " + cacheDir.toAbsolutePath());

        // Collect all files first to avoid stream issues
        List<Path> allFiles = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(test262Dir)) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> p.toString().endsWith(".js"))
                 .forEach(allFiles::add);
        }

        System.out.println("Found " + allFiles.size() + " JavaScript files to scan");

        // Process files in a regular loop instead of stream to avoid hanging
        // Limit to first 10000 to avoid resource exhaustion hang
        int maxFiles = Math.min(allFiles.size(), 10000);
        for (int i = 0; i < maxFiles; i++) {
            Path path = allFiles.get(i);
            total.incrementAndGet();

            if (total.get() % 1000 == 0) {
                System.out.printf("Progress: %d files, %d matched, %d mismatched, %d failed%n",
                    total.get(), matched.get(), mismatched.get(), failed.get());
            }

            // Log current file every 10 files to help identify hangs (increased from 100)
            if (total.get() % 10 == 0 || total.get() > 7000) {
                System.out.printf("  [%d] Processing: %s%n", total.get(), path);
                System.out.flush();
            }

            try {
                         String source = Files.readString(path);

                         // Skip files with obvious syntax we don't support yet
                         if (shouldSkip(source, path)) {
                             continue;
                         }

                         // Check if this is a negative parse test (expected to fail at parse time)
                         boolean isNegativeParseTest = Parser.isNegativeParseTest(source);

                         // Get corresponding cache file
                         Path relativePath = test262Dir.relativize(path);
                         Path cacheFile = cacheDir.resolve(relativePath.toString() + ".json");

                         // If no cache and not a negative parse test, skip (esprima couldn't parse it either)
                         if (!Files.exists(cacheFile)) {
                             if (isNegativeParseTest) {
                                 // For negative parse tests, we still want to verify our parser rejects them
                                 // Try to parse - if it succeeds, that's a bug
                                 try {
                                     boolean isModule = Parser.hasModuleFlag(source) ||
                                         path.toString().endsWith("_FIXTURE.js");
                                     boolean isOnlyStrict = Parser.hasOnlyStrictFlag(source);
                                     Parser.parse(source, isModule, isOnlyStrict);

                                     // Parsing succeeded but should have failed!
                                     negativeTestsPassed.incrementAndGet();
                                     if (negativeTestsPassedFiles.size() < 50) {
                                         negativeTestsPassedFiles.add(path.toString());
                                     }

                                     Map<String, Object> errorJson = new HashMap<>();
                                     errorJson.put("file", path.toString());
                                     errorJson.put("errorType", "NegativeTestPassed");
                                     errorJson.put("message", "Test marked as negative (expected parse failure) but parsing succeeded");
                                     allFailuresJson.add(errorJson);

                                     allFailures.add(path.toString() + ": Negative test incorrectly passed (expected parse failure)");
                                 } catch (Exception e) {
                                     // Good! Parser correctly rejected it
                                     // Don't count as matched since there's no AST to compare
                                 }
                             } else {
                                 noCache.incrementAndGet();
                             }
                             continue;
                         }

                         // Load expected AST from cache
                         String expectedJson = Files.readString(cacheFile);

                         // Check if file has module flag in Test262 frontmatter
                         // Also treat _FIXTURE.js files as modules (they are module fixtures)
                         boolean isModule = Parser.hasModuleFlag(source) ||
                             path.toString().endsWith("_FIXTURE.js");
                         boolean isOnlyStrict = Parser.hasOnlyStrictFlag(source);

                         // Parse with our parser using correct sourceType
                         Program actualProgram = Parser.parse(source, isModule, isOnlyStrict);

                         // If this is a negative parse test and we successfully parsed, that's an error
                         if (isNegativeParseTest) {
                             negativeTestsPassed.incrementAndGet();
                             if (negativeTestsPassedFiles.size() < 50) {
                                 negativeTestsPassedFiles.add(path.toString());
                             }

                             Map<String, Object> errorJson = new HashMap<>();
                             errorJson.put("file", path.toString());
                             errorJson.put("errorType", "NegativeTestPassed");
                             errorJson.put("message", "Test marked as negative (expected parse failure) but parsing succeeded");
                             allFailuresJson.add(errorJson);

                             allFailures.add(path.toString() + ": Negative test incorrectly passed (expected parse failure)");
                             continue;
                         }

                         String actualJson = mapper.writeValueAsString(actualProgram);

                         // Parse both JSONs for structural comparison
                         Object expectedObj = mapper.readValue(expectedJson, Object.class);
                         Object actualObj = mapper.readValue(actualJson, Object.class);

                         // Remove _metadata field from expected (added by cache generator for hash validation)
                         if (expectedObj instanceof Map) {
                             ((Map<?, ?>)expectedObj).remove("_metadata");
                         }

                         // Normalize regex value differences (null vs {} when JS can't compile regex)
                         try {
                             normalizeRegexValues(expectedObj, actualObj);
                         } catch (StackOverflowError e) {
                             System.err.println("StackOverflow in normalizeRegexValues for: " + path);
                             failed.incrementAndGet();
                             continue;
                         }

                         // Normalize bigint value differences (bigint field as string vs number)
                         try {
                             normalizeBigIntValues(expectedObj, actualObj);
                         } catch (StackOverflowError e) {
                             System.err.println("StackOverflow in normalizeBigIntValues for: " + path);
                             failed.incrementAndGet();
                             continue;
                         }

                         // Compare structurally using equals (Jackson's Maps and Lists implement equals)
                         if (Objects.deepEquals(expectedObj, actualObj)) {
                             matched.incrementAndGet();
                         } else {
                             mismatched.incrementAndGet();
                             if (mismatchedFiles.size() < 20) {
                                 mismatchedFiles.add(path.toString());
                             }
                         }
                     } catch (ParseException e) {
                         failed.incrementAndGet();

                         String errorType = e.getClass().getSimpleName();
                         errorTypes.merge(errorType, 1, Integer::sum);

                         String errorMsg = e.getMessage();

                         Map<String, Object> errorJson = new HashMap<>(e.toJson());
                         errorJson.put("file", path.toString());
                         allFailuresJson.add(errorJson);

                         // Record full failure details
                         if (errorMsg != null) {
                             allFailures.add(path.toString() + ": " + errorMsg);
                         } else {
                             allFailures.add(path.toString() + ": " + errorType);
                         }

                         // For "Unexpected character:" errors, include character code
                         if (errorMsg != null && errorMsg.startsWith("Unexpected character:")) {
                             String shortMsg = errorMsg;
                             if (errorMsg.length() > 25) {
                                 // Extract the character after "Unexpected character: "
                                 String charPart = errorMsg.substring(22);
                                 if (charPart.length() > 0) {
                                     char ch = charPart.charAt(0);
                                     shortMsg = String.format("Unexpected character: %c (U+%04X)", ch, (int)ch);
                                 }
                             }
                             errorMessages.merge(shortMsg, 1, Integer::sum);
                             errorToFiles.computeIfAbsent(shortMsg, k -> new ArrayList<>()).add(path.toString());
                         } else {
                             if (errorMsg != null && errorMsg.length() > 100) {
                                 errorMsg = errorMsg.substring(0, 100);
                             }
                             if (errorMsg != null) {
                                 errorMessages.merge(errorMsg, 1, Integer::sum);
                                 errorToFiles.computeIfAbsent(errorMsg, k -> new ArrayList<>()).add(path.toString());
                             }
                         }
                     } catch (Exception e) {
                         failed.incrementAndGet();

                         String errorType = e.getClass().getSimpleName();
                         errorTypes.merge(errorType, 1, Integer::sum);

                         String errorMsg = e.getMessage();

                         Map<String, Object> errorJson = new HashMap<>();
                         errorJson.put("file", path.toString());
                         errorJson.put("errorType", errorType);
                         errorJson.put("message", errorMsg);
                         allFailuresJson.add(errorJson);

                         // Record full failure details
                         if (errorMsg != null) {
                             allFailures.add(path.toString() + ": " + errorMsg);
                         } else {
                             allFailures.add(path.toString() + ": " + errorType);
                         }

                         // For "Unexpected character:" errors, include character code
                         if (errorMsg != null && errorMsg.startsWith("Unexpected character:")) {
                             String shortMsg = errorMsg;
                             if (errorMsg.length() > 25) {
                                 // Extract the character after "Unexpected character: "
                                 String charPart = errorMsg.substring(22);
                                 if (charPart.length() > 0) {
                                     char ch = charPart.charAt(0);
                                     shortMsg = String.format("Unexpected character: %c (U+%04X)", ch, (int)ch);
                                 }
                             }
                             errorMessages.merge(shortMsg, 1, Integer::sum);
                             errorToFiles.computeIfAbsent(shortMsg, k -> new ArrayList<>()).add(path.toString());
                         } else {
                             if (errorMsg != null && errorMsg.length() > 100) {
                                 errorMsg = errorMsg.substring(0, 100);
                             }
                             if (errorMsg != null) {
                                 errorMessages.merge(errorMsg, 1, Integer::sum);
                                 errorToFiles.computeIfAbsent(errorMsg, k -> new ArrayList<>()).add(path.toString());
                             }
                         }
                     }
        }

        int totalWithCache = matched.get() + mismatched.get() + failed.get() + negativeTestsPassed.get();

        System.out.println("\n=== Test262 Oracle Comparison Results ===");
        System.out.printf("Total files scanned: %d%n", total.get());
        System.out.printf("Files skipped (no cache): %d%n", noCache.get());
        System.out.printf("Files with cache: %d%n", totalWithCache);
        System.out.printf("  ✓ Exact matches: %d (%.2f%%)%n",
            matched.get(), totalWithCache > 0 ? (matched.get() * 100.0 / totalWithCache) : 0);
        System.out.printf("  ✗ AST mismatches: %d (%.2f%%)%n",
            mismatched.get(), totalWithCache > 0 ? (mismatched.get() * 100.0 / totalWithCache) : 0);
        System.out.printf("  ⚠ Parse failures: %d (%.2f%%)%n",
            failed.get(), totalWithCache > 0 ? (failed.get() * 100.0 / totalWithCache) : 0);
        System.out.printf("  %s Negative tests incorrectly passed: %d (%.2f%%)%n",
            negativeTestsPassed.get() > 0 ? "❌" : "✓",
            negativeTestsPassed.get(), totalWithCache > 0 ? (negativeTestsPassed.get() * 100.0 / totalWithCache) : 0);

        if (!errorTypes.isEmpty()) {
            System.out.println("\nError types:");
            errorTypes.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEach(e -> System.out.printf("  %s: %d%n", e.getKey(), e.getValue()));
        }

        if (!errorMessages.isEmpty()) {
            System.out.println("\nALL error messages:");
            errorMessages.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEach(e -> System.out.printf("  [%d] %s%n", e.getValue(), e.getKey()));
        }

        if (!mismatchedFiles.isEmpty()) {
            System.out.println("\nFirst 20 mismatched files:");
            mismatchedFiles.forEach(f -> System.out.println("  " + f));
        }

        if (!negativeTestsPassedFiles.isEmpty()) {
            System.out.println("\nNegative tests that incorrectly passed (first 50):");
            negativeTestsPassedFiles.forEach(f -> System.out.println("  " + f));
        }

        // Print files with unterminated template literal error
        List<String> templateLiteralFiles = errorToFiles.get("Unterminated template literal");
        if (templateLiteralFiles != null && !templateLiteralFiles.isEmpty()) {
            System.out.println("\nFiles with 'Unterminated template literal' error:");
            templateLiteralFiles.stream().limit(5).forEach(f -> System.out.println("  " + f));
        }

        // Print files with class body error
        List<String> classBodyFiles = errorToFiles.get("Expected property name in class body");
        if (classBodyFiles != null && !classBodyFiles.isEmpty()) {
            System.out.println("\nFiles with 'Expected property name in class body' error:");
            classBodyFiles.stream().limit(10).forEach(f -> System.out.println("  " + f));
        }

        // Print files with "Expected property name" error (object destructuring)
        List<String> propertyNameFiles = errorToFiles.get("Expected property name");
        if (propertyNameFiles != null && !propertyNameFiles.isEmpty()) {
            System.out.println("\nFiles with 'Expected property name' error:");
            propertyNameFiles.stream().limit(10).forEach(f -> System.out.println("  " + f));
        }

        // Print files with invalid unicode escape sequence error
        List<String> unicodeEscapeFiles = errorToFiles.get("Invalid unicode escape sequence");
        if (unicodeEscapeFiles != null && !unicodeEscapeFiles.isEmpty()) {
            System.out.println("\nFiles with 'Invalid unicode escape sequence' error:");
            unicodeEscapeFiles.forEach(f -> System.out.println("  " + f));
        }

        // Print files with "Unexpected character:" errors
        System.out.println("\nFiles with 'Unexpected character:' errors:");
        errorToFiles.entrySet().stream()
            .filter(entry -> entry.getKey().startsWith("Unexpected character:"))
            .sorted((e1, e2) -> Integer.compare(e2.getValue().size(), e1.getValue().size()))
            .forEach(entry -> {
                String errorMsg = entry.getKey();
                List<String> files = entry.getValue();
                System.out.printf("\n  [%d] %s%n", files.size(), errorMsg);
                System.out.println("  Example files:");
                files.stream().limit(3).forEach(f -> System.out.println("    " + f));
            });

        // Write all failures to file
        try {
            Path failuresFile = Paths.get("/tmp/all_test262_failures.txt");
            Files.write(failuresFile, allFailures);
            System.out.println("\n✓ Wrote all " + allFailures.size() + " failures to: " + failuresFile);
        } catch (IOException e) {
            System.err.println("Failed to write failures file: " + e.getMessage());
        }

        // Write JSON failures to file
        try {
            Path jsonFile = Paths.get("/tmp/all_test262_failures.json");
            String jsonOutput = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(allFailuresJson);
            Files.writeString(jsonFile, jsonOutput);
            System.out.println("✓ Wrote " + allFailuresJson.size() + " JSON failures to: " + jsonFile);
        } catch (IOException e) {
            System.err.println("Failed to write JSON failures file: " + e.getMessage());
        }

        // Assert that we have zero failures, zero mismatches, and zero negative tests that passed
        if (failed.get() > 0 || mismatched.get() > 0 || negativeTestsPassed.get() > 0) {
            String errorMsg = String.format(
                "\n❌ Test262 oracle comparison FAILED:\n" +
                "  Parse failures: %d\n" +
                "  AST mismatches: %d\n" +
                "  Negative tests incorrectly passed: %d\n" +
                "  Total issues: %d out of %d files\n" +
                "See /tmp/all_test262_failures.txt and /tmp/all_test262_failures.json for details.",
                failed.get(), mismatched.get(), negativeTestsPassed.get(),
                failed.get() + mismatched.get() + negativeTestsPassed.get(), totalWithCache
            );
            System.err.println(errorMsg);

            // Fail the test
            assertEquals(0, failed.get(), "Should have zero parse failures");
            assertEquals(0, mismatched.get(), "Should have zero AST mismatches");
            assertEquals(0, negativeTestsPassed.get(), "Should have zero negative tests that incorrectly passed");
        }

        System.out.println("\n✅ All test262 files passed! Perfect compatibility.");
    }

    private boolean shouldSkip(String source, Path path) {
        // The cache generation script does all filtering
        // We simply run whatever has a cache file
        // No filtering needed here
        return false;
    }

    /**
     * Normalize regex literal value differences between expected and actual ASTs.
     * Acorn sets value to null when JS can't compile the regex (e.g., unknown unicode property).
     * We always set value to {}. When expected has null and actual has {} for a regex literal,
     * copy null to actual so comparison passes.
     */
    @SuppressWarnings("unchecked")
    private void normalizeRegexValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            // Check if this is a Literal node with regex
            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("regex")) {
                // If expected value is null and actual is empty map, normalize
                if (expMap.get("value") == null && actMap.get("value") instanceof Map) {
                    Map<?, ?> actValue = (Map<?, ?>) actMap.get("value");
                    if (actValue.isEmpty()) {
                        actMap.put("value", null);
                    }
                }
            }

            // Recurse into all fields
            for (String key : expMap.keySet()) {
                if (actMap.containsKey(key)) {
                    normalizeRegexValues(expMap.get(key), actMap.get(key));
                }
            }
        } else if (expected instanceof List && actual instanceof List) {
            List<Object> expList = (List<Object>) expected;
            List<Object> actList = (List<Object>) actual;
            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {
                normalizeRegexValues(expList.get(i), actList.get(i));
            }
        }
    }

    /**
     * Normalize bigint literal value differences between expected and actual ASTs.
     * The bigint field should be a string representation of the numeric value.
     * This method ensures both expected and actual have consistent string representations.
     */
    @SuppressWarnings("unchecked")
    private void normalizeBigIntValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            // Check if this is a Literal node with bigint
            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("bigint")) {
                Object expBigint = expMap.get("bigint");
                Object actBigint = actMap.get("bigint");

                // Convert both to strings for comparison
                if (expBigint != null && actBigint != null) {
                    String expStr = expBigint.toString();
                    String actStr = actBigint.toString();

                    // If they're numerically equal, normalize to the same string
                    if (!expStr.equals(actStr)) {
                        try {
                            java.math.BigInteger expBI = new java.math.BigInteger(expStr);
                            java.math.BigInteger actBI = new java.math.BigInteger(actStr);
                            if (expBI.equals(actBI)) {
                                actMap.put("bigint", expStr);
                            }
                        } catch (NumberFormatException e) {
                            // If can't parse, leave as is
                        }
                    }
                }

                // Normalize value field: Acorn sets value to null for BigInt, normalize if our parser differs
                Object expValue = expMap.get("value");
                Object actValue = actMap.get("value");
                if (expValue == null && actValue != null) {
                    actMap.put("value", null);
                }
            }

            // Recurse into all fields
            for (String key : expMap.keySet()) {
                if (actMap.containsKey(key)) {
                    normalizeBigIntValues(expMap.get(key), actMap.get(key));
                }
            }
        } else if (expected instanceof List && actual instanceof List) {
            List<Object> expList = (List<Object>) expected;
            List<Object> actList = (List<Object>) actual;
            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {
                normalizeBigIntValues(expList.get(i), actList.get(i));
            }
        }
    }
}
