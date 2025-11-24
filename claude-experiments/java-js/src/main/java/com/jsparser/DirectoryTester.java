package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;

/**
 * Test a directory of JavaScript files against Acorn in real-time.
 * No caching - parse both with Acorn and Java parser on-the-fly.
 */
public class DirectoryTester {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.err.println("Usage: java DirectoryTester <directory-path> [--max-failures=N] [--max-mismatches=N]");
            System.err.println("");
            System.err.println("Examples:");
            System.err.println("  java DirectoryTester ../my-project");
            System.err.println("  java DirectoryTester ~/code/app");
            System.err.println("  java DirectoryTester ../my-project --max-failures=10");
            System.err.println("  java DirectoryTester ../my-project --max-failures=10 --max-mismatches=10");
            System.err.println("");
            System.err.println("Options:");
            System.err.println("  --max-failures=N    Stop after N parse failures (default: unlimited)");
            System.err.println("  --max-mismatches=N  Stop after N AST mismatches (default: unlimited)");
            System.exit(1);
        }

        String dirPath = args[0];
        int maxFailures = Integer.MAX_VALUE;
        int maxMismatches = Integer.MAX_VALUE;

        // Parse arguments
        for (int i = 1; i < args.length; i++) {
            if (args[i].startsWith("--max-failures=")) {
                try {
                    maxFailures = Integer.parseInt(args[i].substring("--max-failures=".length()));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid value for --max-failures: " + args[i]);
                    System.exit(1);
                }
            } else if (args[i].startsWith("--max-mismatches=")) {
                try {
                    maxMismatches = Integer.parseInt(args[i].substring("--max-mismatches=".length()));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid value for --max-mismatches: " + args[i]);
                    System.exit(1);
                }
            }
        }

        Path targetDir = Paths.get(dirPath);

        if (!Files.exists(targetDir)) {
            System.err.println("Directory not found: " + targetDir.toAbsolutePath());
            System.exit(1);
        }

        System.out.println("Ad-hoc directory testing (real-time comparison)");
        System.out.println("Directory: " + targetDir.toAbsolutePath());
        System.out.println("");

        // Collect all files
        System.out.println("Scanning for JavaScript files...");
        List<Path> jsFiles = collectJsFiles(targetDir);
        System.out.println("Found " + jsFiles.size() + " JavaScript files\n");

        if (jsFiles.isEmpty()) {
            System.out.println("No JavaScript files found");
            return;
        }

        // Process files
        AtomicInteger processed = new AtomicInteger(0);
        AtomicInteger acornSuccess = new AtomicInteger(0);
        AtomicInteger acornFailed = new AtomicInteger(0);
        AtomicInteger javaSuccess = new AtomicInteger(0);
        AtomicInteger javaFailed = new AtomicInteger(0);
        AtomicInteger matched = new AtomicInteger(0);
        AtomicInteger mismatched = new AtomicInteger(0);
        AtomicInteger skipped = new AtomicInteger(0);

        List<String> failedFiles = new ArrayList<>();
        List<String> mismatchedFiles = new ArrayList<>();
        Map<String, Integer> errorMessages = new HashMap<>();

        System.out.println("Testing files (parsing with both Acorn and Java)...");
        if (maxFailures < Integer.MAX_VALUE || maxMismatches < Integer.MAX_VALUE) {
            if (maxFailures < Integer.MAX_VALUE && maxMismatches < Integer.MAX_VALUE) {
                System.out.println("Will stop after " + maxFailures + " parse failures or " + maxMismatches + " AST mismatches\n");
            } else if (maxFailures < Integer.MAX_VALUE) {
                System.out.println("Will stop after " + maxFailures + " parse failures\n");
            } else {
                System.out.println("Will stop after " + maxMismatches + " AST mismatches\n");
            }
        } else {
            System.out.println();
        }

        for (Path file : jsFiles) {
            // Check if we've hit max failures or mismatches
            if (javaFailed.get() >= maxFailures) {
                System.out.println("\n⚠ Reached maximum failure limit (" + maxFailures + "), stopping early");
                break;
            }
            if (mismatched.get() >= maxMismatches) {
                System.out.println("\n⚠ Reached maximum mismatch limit (" + maxMismatches + "), stopping early");
                break;
            }

            processed.incrementAndGet();

            if (processed.get() % 100 == 0) {
                System.out.printf("Progress: %d/%d (%d matched, %d mismatched, %d Java failed)%n",
                    processed.get(), jsFiles.size(), matched.get(), mismatched.get(), javaFailed.get());
            }

            String relativePath = targetDir.relativize(file).toString();

            try {
                String source = Files.readString(file);

                // Skip very large files (> 500KB)
                if (source.length() > 500000) {
                    skipped.incrementAndGet();
                    continue;
                }

                // Parse with Acorn
                AcornResult acornResult = parseWithAcorn(source, file);

                if (!acornResult.success) {
                    acornFailed.incrementAndGet();
                    skipped.incrementAndGet();
                    continue;
                }

                acornSuccess.incrementAndGet();

                // Parse with Java
                boolean isModule = acornResult.sourceType.equals("module");
                Program javaAst;
                try {
                    javaAst = Parser.parse(source, isModule);
                } catch (ParseException e) {
                    javaFailed.incrementAndGet();
                    String errorMsg = e.getMessage();
                    if (errorMsg != null && errorMsg.length() > 100) {
                        errorMsg = errorMsg.substring(0, 100);
                    }
                    if (errorMsg != null) {
                        errorMessages.merge(errorMsg, 1, Integer::sum);
                    }
                    if (failedFiles.size() < 50) {
                        failedFiles.add(relativePath + ": " + errorMsg);
                    }
                    continue;
                } catch (Exception e) {
                    javaFailed.incrementAndGet();
                    if (failedFiles.size() < 50) {
                        failedFiles.add(relativePath + ": " + e.getMessage());
                    }
                    continue;
                }

                javaSuccess.incrementAndGet();

                // Compare ASTs
                String acornJson = acornResult.astJson;
                String javaJson = mapper.writeValueAsString(javaAst);

                // Parse both for structural comparison
                Object acornObj = mapper.readValue(acornJson, Object.class);
                Object javaObj = mapper.readValue(javaJson, Object.class);

                // Normalize
                normalizeRegexValues(acornObj, javaObj);
                normalizeBigIntValues(acornObj, javaObj);

                // Sort keys for comparison (handles different serialization order)
                acornObj = sortMapKeys(acornObj);
                javaObj = sortMapKeys(javaObj);

                if (Objects.deepEquals(acornObj, javaObj)) {
                    matched.incrementAndGet();
                } else {
                    mismatched.incrementAndGet();
                    if (mismatchedFiles.size() < 20) {
                        mismatchedFiles.add(relativePath);
                    }
                }
            } catch (Exception e) {
                javaFailed.incrementAndGet();
                if (failedFiles.size() < 50) {
                    failedFiles.add(relativePath + ": " + e.getMessage());
                }
            }
        }

        // Print results
        System.out.println("\n=== Results ===");
        System.out.printf("Total files: %d%n", jsFiles.size());
        System.out.printf("Skipped (too large or Acorn failed): %d%n", skipped.get());
        System.out.println("\nAcorn results:");
        System.out.printf("  ✓ Successfully parsed: %d (%.2f%%)%n",
            acornSuccess.get(), (acornSuccess.get() * 100.0 / jsFiles.size()));
        System.out.printf("  ✗ Failed to parse: %d (%.2f%%)%n",
            acornFailed.get(), (acornFailed.get() * 100.0 / jsFiles.size()));

        System.out.println("\nJava parser results (on files Acorn succeeded):");
        int acornSuccessCount = acornSuccess.get();
        System.out.printf("  ✓ Successfully parsed: %d (%.2f%%)%n",
            javaSuccess.get(), acornSuccessCount > 0 ? (javaSuccess.get() * 100.0 / acornSuccessCount) : 0);
        System.out.printf("  ✗ Failed to parse: %d (%.2f%%)%n",
            javaFailed.get(), acornSuccessCount > 0 ? (javaFailed.get() * 100.0 / acornSuccessCount) : 0);

        System.out.println("\nComparison (on files both parsers succeeded):");
        int totalSuccess = javaSuccess.get();
        System.out.printf("  ✓ Exact matches: %d (%.2f%%)%n",
            matched.get(), totalSuccess > 0 ? (matched.get() * 100.0 / totalSuccess) : 0);
        System.out.printf("  ✗ AST mismatches: %d (%.2f%%)%n",
            mismatched.get(), totalSuccess > 0 ? (mismatched.get() * 100.0 / totalSuccess) : 0);

        if (!mismatchedFiles.isEmpty()) {
            System.out.println("\nFirst 20 mismatched files:");
            mismatchedFiles.forEach(f -> System.out.println("  " + f));
        }

        if (!failedFiles.isEmpty()) {
            System.out.println("\nFirst 50 failed files:");
            failedFiles.forEach(f -> System.out.println("  " + f));
        }

        if (!errorMessages.isEmpty()) {
            System.out.println("\nError messages:");
            errorMessages.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(20)
                .forEach(e -> System.out.printf("  [%d] %s%n", e.getValue(), e.getKey()));
        }
    }

    private static List<Path> collectJsFiles(Path dir) throws IOException {
        List<Path> files = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(dir)) {
            paths.filter(Files::isRegularFile)
                .filter(p -> {
                    String name = p.toString();
                    return (name.endsWith(".js") || name.endsWith(".mjs")) &&
                           !name.contains("/.git/");
                })
                .forEach(files::add);
        }
        return files;
    }

    private static AcornResult parseWithAcorn(String source, Path filePath) {
        try {
            // Determine source type
            String sourceType = "script";
            if (filePath.toString().endsWith(".mjs") ||
                source.contains("import ") ||
                source.contains("export ")) {
                sourceType = "module";
            }

            // Write AST to temp file to avoid pipe buffer limits (64KB)
            Path tempFile = Files.createTempFile("acorn-ast-", ".json");

            // Build Node.js command to parse with Acorn
            String[] cmd = {
                "node",
                "-e",
                "const acorn = require('acorn'); " +
                "const fs = require('fs'); " +
                "const source = fs.readFileSync(process.argv[1], 'utf-8'); " +
                "try { " +
                "  const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: '" + sourceType + "'}); " +
                "  fs.writeFileSync(process.argv[2], JSON.stringify(ast, (k,v) => typeof v === 'bigint' ? null : v)); " +
                "  process.exit(0); " +
                "} catch (e) { " +
                "  console.error('PARSE_ERROR: ' + e.message); " +
                "  process.exit(1); " +
                "}",
                filePath.toAbsolutePath().toString(),
                tempFile.toAbsolutePath().toString()
            };

            ProcessBuilder pb = new ProcessBuilder(cmd);
            Process process = pb.start();

            // Read stderr for error messages
            java.util.concurrent.CompletableFuture<String> errorFuture = java.util.concurrent.CompletableFuture.supplyAsync(() -> {
                try {
                    return new String(process.getErrorStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
                } catch (IOException e) {
                    return "";
                }
            });

            int exitCode = process.waitFor();
            String errorOutput = errorFuture.get();

            if (exitCode != 0) {
                Files.deleteIfExists(tempFile);
                return new AcornResult(false, null, null, errorOutput);
            }

            // Read AST from temp file
            String output = Files.readString(tempFile);
            Files.deleteIfExists(tempFile);

            return new AcornResult(true, output, sourceType, null);
        } catch (Exception e) {
            return new AcornResult(false, null, null, e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    private static void normalizeRegexValues(Object expected, Object actual) {
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
    private static Object sortMapKeys(Object obj) {
        if (obj instanceof Map) {
            Map<String, Object> map = (Map<String, Object>) obj;
            Map<String, Object> sortedMap = new java.util.TreeMap<>();
            for (Map.Entry<String, Object> entry : map.entrySet()) {
                sortedMap.put(entry.getKey(), sortMapKeys(entry.getValue()));
            }
            return sortedMap;
        } else if (obj instanceof List) {
            List<Object> list = (List<Object>) obj;
            List<Object> sortedList = new ArrayList<>();
            for (Object item : list) {
                sortedList.add(sortMapKeys(item));
            }
            return sortedList;
        }
        return obj;
    }

    @SuppressWarnings("unchecked")
    private static void normalizeBigIntValues(Object expected, Object actual) {
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

    private static class AcornResult {
        final boolean success;
        final String astJson;
        final String sourceType;
        final String error;

        AcornResult(boolean success, String astJson, String sourceType, String error) {
            this.success = success;
            this.astJson = astJson;
            this.sourceType = sourceType;
            this.error = error;
        }
    }
}
