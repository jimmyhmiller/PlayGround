package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

@EnabledIfSystemProperty(named = "packageName", matches = ".+")
public class AdhocPackageTest {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("Parse package files and compare against Acorn oracle")
    void parsePackageFiles() throws IOException {
        String packageName = System.getProperty("packageName");

        if (packageName == null || packageName.isEmpty()) {
            System.out.println("ERROR: packageName system property not set");
            System.out.println("Run with: mvn test -Dtest=AdhocPackageTest -DpackageName=<package-name>");
            return;
        }

        Path cacheDir = Paths.get("test-oracles/adhoc-cache",
            packageName.replaceAll("[^a-zA-Z0-9-]", "_"));

        if (!Files.exists(cacheDir)) {
            System.out.println("ERROR: Cache directory not found at: " + cacheDir.toAbsolutePath());
            System.out.println("Please run: node scripts/test-package.js " + packageName);
            return;
        }

        // Load summary
        Path summaryPath = cacheDir.resolve("_summary.json");
        if (!Files.exists(summaryPath)) {
            System.out.println("ERROR: Summary file not found at: " + summaryPath.toAbsolutePath());
            return;
        }

        String summaryJson = Files.readString(summaryPath);
        Map<String, Object> summary = mapper.readValue(summaryJson, Map.class);

        System.out.println("\n=== Ad-hoc Package Test ===");
        System.out.println("Package: " + summary.get("packageName"));
        System.out.println("Source directory: " + summary.get("sourceDir"));
        System.out.println("Cache directory: " + cacheDir.toAbsolutePath());

        @SuppressWarnings("unchecked")
        List<Map<String, Object>> successfulFiles =
            (List<Map<String, Object>>) summary.get("successfulFiles");

        if (successfulFiles == null || successfulFiles.isEmpty()) {
            System.out.println("No files to test (Acorn couldn't parse any files)");
            return;
        }

        AtomicInteger total = new AtomicInteger(0);
        AtomicInteger matched = new AtomicInteger(0);
        AtomicInteger mismatched = new AtomicInteger(0);
        AtomicInteger failed = new AtomicInteger(0);
        List<String> mismatchedFiles = new ArrayList<>();
        List<String> failedFilesList = new ArrayList<>();
        Map<String, Integer> errorMessages = new HashMap<>();

        System.out.println("\nTesting " + successfulFiles.size() + " files that Acorn successfully parsed...\n");

        for (Map<String, Object> fileInfo : successfulFiles) {
            total.incrementAndGet();

            String cacheFile = (String) fileInfo.get("cacheFile");
            String sourceFile = (String) fileInfo.get("file");
            String sourceType = (String) fileInfo.get("sourceType");

            if (total.get() % 10 == 0) {
                System.out.printf("Progress: %d/%d files, %d matched, %d mismatched, %d failed%n",
                    total.get(), successfulFiles.size(), matched.get(), mismatched.get(), failed.get());
            }

            try {
                // Load expected AST from cache
                String expectedJson = Files.readString(Paths.get(cacheFile));

                // Get source file path from cache metadata
                Object expectedObj = mapper.readValue(expectedJson, Object.class);
                if (!(expectedObj instanceof Map)) {
                    failed.incrementAndGet();
                    continue;
                }

                @SuppressWarnings("unchecked")
                Map<String, Object> expectedMap = (Map<String, Object>) expectedObj;

                @SuppressWarnings("unchecked")
                Map<String, Object> metadata = (Map<String, Object>) expectedMap.get("_metadata");

                if (metadata == null) {
                    failed.incrementAndGet();
                    failedFilesList.add(sourceFile + ": No metadata in cache");
                    continue;
                }

                // Get actual source file path
                String sourceDir = (String) summary.get("sourceDir");
                Path actualSourceFile = Paths.get(sourceDir, sourceFile);

                if (!Files.exists(actualSourceFile)) {
                    failed.incrementAndGet();
                    failedFilesList.add(sourceFile + ": Source file not found");
                    continue;
                }

                // Read source
                String source = Files.readString(actualSourceFile);

                // Determine if module based on metadata
                boolean isModule = "module".equals(sourceType);

                // Parse with our parser
                Program actualProgram = Parser.parse(source, isModule);
                String actualJson = mapper.writeValueAsString(actualProgram);

                // Parse both JSONs for structural comparison
                Object actualObj = mapper.readValue(actualJson, Object.class);

                // Remove _metadata field from expected
                expectedMap.remove("_metadata");

                // Normalize regex values
                normalizeRegexValues(expectedObj, actualObj);

                // Normalize bigint values
                normalizeBigIntValues(expectedObj, actualObj);

                // Normalize all numeric values (Jackson may use different types)
                normalizeNumericValues(expectedObj, actualObj);

                // Compare structurally
                if (Objects.deepEquals(expectedObj, actualObj)) {
                    matched.incrementAndGet();
                } else {
                    mismatched.incrementAndGet();
                    if (mismatchedFiles.size() < 20) {
                        mismatchedFiles.add(sourceFile);
                    }
                }
            } catch (ParseException e) {
                failed.incrementAndGet();
                String errorMsg = e.getMessage();
                if (errorMsg != null && errorMsg.length() > 100) {
                    errorMsg = errorMsg.substring(0, 100);
                }
                if (errorMsg != null) {
                    errorMessages.merge(errorMsg, 1, Integer::sum);
                }
                if (failedFilesList.size() < 50) {
                    failedFilesList.add(sourceFile + ": " + errorMsg);
                }
            } catch (Exception e) {
                failed.incrementAndGet();
                if (failedFilesList.size() < 50) {
                    failedFilesList.add(sourceFile + ": " + e.getMessage());
                }
            }
        }

        int totalTested = matched.get() + mismatched.get() + failed.get();

        System.out.println("\n=== Test Results ===");
        System.out.printf("Total files tested: %d%n", totalTested);
        System.out.printf("  ✓ Exact matches: %d (%.2f%%)%n",
            matched.get(), totalTested > 0 ? (matched.get() * 100.0 / totalTested) : 0);
        System.out.printf("  ✗ AST mismatches: %d (%.2f%%)%n",
            mismatched.get(), totalTested > 0 ? (mismatched.get() * 100.0 / totalTested) : 0);
        System.out.printf("  ⚠ Parse failures: %d (%.2f%%)%n",
            failed.get(), totalTested > 0 ? (failed.get() * 100.0 / totalTested) : 0);

        if (!mismatchedFiles.isEmpty()) {
            System.out.println("\nFirst 20 mismatched files:");
            mismatchedFiles.forEach(f -> System.out.println("  " + f));
        }

        if (!failedFilesList.isEmpty()) {
            System.out.println("\nFirst 50 failed files:");
            failedFilesList.forEach(f -> System.out.println("  " + f));
        }

        if (!errorMessages.isEmpty()) {
            System.out.println("\nError messages:");
            errorMessages.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .forEach(e -> System.out.printf("  [%d] %s%n", e.getValue(), e.getKey()));
        }
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
                            // If can't parse, leave as is
                        }
                    }
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
