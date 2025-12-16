package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

/**
 * Tests the parser against curated real-world JavaScript files.
 *
 * The curated files include:
 * - react/ - React development build (MIT)
 * - framer-motion/ - Animation library (MIT)
 * - motion-dom/ - Motion primitives (MIT)
 * - watchpack/ - File watcher (MIT)
 * - eslint-plugin-unicorn/ - ESLint rules (MIT)
 * - typescript-tests/ - TypeScript parser test baselines (Apache-2.0)
 * - oxc-fixtures/ - Parser edge case tests (MIT)
 *
 * See test-oracles/curated/README.md for license details.
 */
public class CuratedFilesTest {

    private static final ObjectMapper mapper = new ObjectMapper();
    private static final Path CURATED_DIR = Paths.get("test-oracles/curated");
    private static final Path CACHE_DIR = Paths.get("test-oracles/curated-cache");

    @Test
    @DisplayName("Parse all curated files and compare against acorn oracle")
    void parseAllCuratedFiles() throws IOException {
        if (!Files.exists(CURATED_DIR)) {
            System.out.println("Curated directory not found at: " + CURATED_DIR.toAbsolutePath());
            return;
        }

        if (!Files.exists(CACHE_DIR)) {
            System.out.println("ERROR: Cache directory not found at: " + CACHE_DIR.toAbsolutePath());
            System.out.println("Please run: node scripts/generate-curated-cache.js");
            return;
        }

        int total = 0;
        int matched = 0;
        int mismatched = 0;
        int failed = 0;
        List<String> mismatchedFiles = new ArrayList<>();
        List<String> failedFiles = new ArrayList<>();

        System.out.println("=== Curated Files Test ===");
        System.out.println("Curated dir: " + CURATED_DIR.toAbsolutePath());
        System.out.println("Cache dir: " + CACHE_DIR.toAbsolutePath());

        // Collect all JS/MJS files
        List<Path> allFiles = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(CURATED_DIR)) {
            paths.filter(Files::isRegularFile)
                 .filter(p -> {
                     String name = p.toString();
                     return name.endsWith(".js") || name.endsWith(".mjs");
                 })
                 .forEach(allFiles::add);
        }

        System.out.println("Found " + allFiles.size() + " JavaScript files\n");

        for (Path path : allFiles) {
            total++;
            String relativePath = CURATED_DIR.relativize(path).toString();

            try {
                // Get corresponding cache file
                Path cacheFile = CACHE_DIR.resolve(relativePath + ".json");

                if (!Files.exists(cacheFile)) {
                    System.out.println("SKIP (no cache): " + relativePath);
                    continue;
                }

                // Load expected AST from cache
                String expectedJson = Files.readString(cacheFile);
                Object expectedObj = mapper.readValue(expectedJson, Object.class);

                // Remove _metadata field
                if (expectedObj instanceof Map) {
                    ((Map<?, ?>) expectedObj).remove("_metadata");
                }

                // Read source file
                String source = Files.readString(path);

                // Determine if module (.mjs files are always modules)
                boolean isModule = path.toString().endsWith(".mjs");

                // Parse with our parser
                Program actualProgram = Parser.parse(source, isModule);
                String actualJson = mapper.writeValueAsString(actualProgram);
                Object actualObj = mapper.readValue(actualJson, Object.class);

                // Normalize differences
                normalizeRegexValues(expectedObj, actualObj);
                normalizeBigIntValues(expectedObj, actualObj);

                // Compare
                if (Objects.deepEquals(expectedObj, actualObj)) {
                    matched++;
                    System.out.println("✓ " + relativePath);
                } else {
                    mismatched++;
                    mismatchedFiles.add(relativePath);
                    System.out.println("✗ " + relativePath + " (AST mismatch)");
                }
            } catch (ParseException e) {
                failed++;
                failedFiles.add(relativePath + ": " + e.getMessage());
                System.out.println("⚠ " + relativePath + ": " + e.getMessage());
            } catch (Exception e) {
                failed++;
                failedFiles.add(relativePath + ": " + e.getMessage());
                System.out.println("⚠ " + relativePath + ": " + e.getMessage());
            }
        }

        // Summary
        System.out.println("\n=== Results ===");
        System.out.printf("Total files: %d%n", total);
        System.out.printf("  ✓ Exact matches: %d (%.1f%%)%n",
            matched, total > 0 ? (matched * 100.0 / total) : 0);
        System.out.printf("  ✗ AST mismatches: %d (%.1f%%)%n",
            mismatched, total > 0 ? (mismatched * 100.0 / total) : 0);
        System.out.printf("  ⚠ Parse failures: %d (%.1f%%)%n",
            failed, total > 0 ? (failed * 100.0 / total) : 0);

        if (!mismatchedFiles.isEmpty()) {
            System.out.println("\nMismatched files:");
            mismatchedFiles.forEach(f -> System.out.println("  " + f));
        }

        if (!failedFiles.isEmpty()) {
            System.out.println("\nFailed files:");
            failedFiles.forEach(f -> System.out.println("  " + f));
        }

        // Assert all tests pass
        assertEquals(0, failed, "Should have zero parse failures");
        assertEquals(0, mismatched, "Should have zero AST mismatches");

        System.out.println("\n✅ All curated files passed!");
    }

    @SuppressWarnings("unchecked")
    private void normalizeRegexValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("regex")) {
                if (expMap.get("value") == null && actMap.get("value") instanceof Map) {
                    Map<?, ?> actValue = (Map<?, ?>) actMap.get("value");
                    if (actValue.isEmpty()) {
                        actMap.put("value", null);
                    }
                }
            }

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

    @SuppressWarnings("unchecked")
    private void normalizeBigIntValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("bigint")) {
                Object expBigint = expMap.get("bigint");
                Object actBigint = actMap.get("bigint");

                if (expBigint != null && actBigint != null) {
                    String expStr = expBigint.toString();
                    String actStr = actBigint.toString();

                    if (!expStr.equals(actStr)) {
                        try {
                            java.math.BigInteger expBI = new java.math.BigInteger(expStr);
                            java.math.BigInteger actBI = new java.math.BigInteger(actStr);
                            if (expBI.equals(actBI)) {
                                actMap.put("bigint", expStr);
                            }
                        } catch (NumberFormatException e) {
                            // leave as is
                        }
                    }
                }

                Object expValue = expMap.get("value");
                Object actValue = actMap.get("value");
                if (expValue == null && actValue != null) {
                    actMap.put("value", null);
                }
            }

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
