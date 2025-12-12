package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
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
            System.err.println("Usage: java DirectoryTester <directory-path> [--max-failures=N] [--max-mismatches=N] [--max-too-permissive=N]");
            System.err.println("");
            System.err.println("Examples:");
            System.err.println("  java DirectoryTester ../my-project");
            System.err.println("  java DirectoryTester ~/code/app");
            System.err.println("  java DirectoryTester ../my-project --max-failures=10");
            System.err.println("  java DirectoryTester ../my-project --max-failures=10 --max-mismatches=10");
            System.err.println("  java DirectoryTester ../my-project --max-failures=5 --max-too-permissive=5");
            System.err.println("");
            System.err.println("Options:");
            System.err.println("  --max-failures=N        Stop after N Java parser failures (Acorn succeeded but Java failed) (default: unlimited)");
            System.err.println("  --max-mismatches=N      Stop after N AST mismatches (both succeeded but different ASTs) (default: unlimited)");
            System.err.println("  --max-too-permissive=N  Stop after N cases where Java is too permissive (Java succeeded but Acorn failed) (default: unlimited)");
            System.err.println("  --no-cache              Disable automatic caching of failures and mismatches");
            System.err.println("");
            System.err.println("Failures and mismatches are automatically cached to test-oracles/adhoc-cache");
            System.err.println("and JUnit tests are regenerated after testing completes.");
            System.err.println("");
            System.err.println("Categories:");
            System.err.println("  - Both succeeded + matched: Perfect agreement");
            System.err.println("  - Both succeeded + AST mismatch: Parsing succeeded but ASTs differ");
            System.err.println("  - Both failed: Both agree code is invalid");
            System.err.println("  - Java failed, Acorn succeeded: Java parser bug (real failure)");
            System.err.println("  - Java succeeded, Acorn failed: Java is too permissive (warning)");
            System.exit(1);
        }

        String dirPath = args[0];
        int maxFailures = Integer.MAX_VALUE;
        int maxMismatches = Integer.MAX_VALUE;
        int maxTooPermissive = Integer.MAX_VALUE;
        boolean enableCaching = true;  // Caching enabled by default

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
            } else if (args[i].startsWith("--max-too-permissive=")) {
                try {
                    maxTooPermissive = Integer.parseInt(args[i].substring("--max-too-permissive=".length()));
                } catch (NumberFormatException e) {
                    System.err.println("Invalid value for --max-too-permissive: " + args[i]);
                    System.exit(1);
                }
            } else if (args[i].equals("--no-cache")) {
                enableCaching = false;
            } else if (args[i].equals("--cache")) {
                enableCaching = true;  // Keep for backwards compatibility
            }
        }

        Path targetDir = Paths.get(dirPath);

        if (!Files.exists(targetDir)) {
            System.err.println("Directory not found: " + targetDir.toAbsolutePath());
            System.exit(1);
        }

        System.out.println("Ad-hoc directory testing (real-time comparison)");
        System.out.println("Directory: " + targetDir.toAbsolutePath());
        if (enableCaching) {
            System.out.println("Auto-caching enabled: failures/mismatches will be saved to test-oracles/adhoc-cache");
        }
        System.out.println("");

        // Collect all files
        System.out.println("Scanning for JavaScript files...");
        List<Path> jsFiles = collectJsFiles(targetDir);
        System.out.println("Found " + jsFiles.size() + " JavaScript files\n");

        if (jsFiles.isEmpty()) {
            System.out.println("No JavaScript files found");
            return;
        }

        // Setup cache builder if caching is enabled
        AcornCacheBuilder cacheBuilder = null;
        AtomicInteger cachedCount = new AtomicInteger(0);
        if (enableCaching) {
            Path cacheDir = Paths.get("test-oracles/adhoc-cache");
            cacheBuilder = new AcornCacheBuilder(cacheDir);
        }

        // Process files
        AtomicInteger processed = new AtomicInteger(0);
        AtomicInteger acornSuccess = new AtomicInteger(0);
        AtomicInteger acornFailed = new AtomicInteger(0);
        AtomicInteger javaSuccess = new AtomicInteger(0);
        AtomicInteger javaFailed = new AtomicInteger(0);
        AtomicInteger matched = new AtomicInteger(0);
        AtomicInteger mismatched = new AtomicInteger(0);
        AtomicInteger bothFailed = new AtomicInteger(0); // Both parsers agree code is invalid
        AtomicInteger javaFailedAcornSucceeded = new AtomicInteger(0); // Java failed, Acorn succeeded
        AtomicInteger javaSucceededAcornFailed = new AtomicInteger(0); // Java succeeded, Acorn failed (too permissive)

        List<String> javaFailedFiles = new ArrayList<>();
        List<String> javaSucceededAcornFailedFiles = new ArrayList<>();
        List<String> mismatchedFiles = new ArrayList<>();
        Map<String, Integer> errorMessages = new HashMap<>();

        System.out.println("Testing files (parsing with both Acorn and Java)...");
        if (maxFailures < Integer.MAX_VALUE || maxMismatches < Integer.MAX_VALUE || maxTooPermissive < Integer.MAX_VALUE) {
            List<String> limits = new ArrayList<>();
            if (maxFailures < Integer.MAX_VALUE) {
                limits.add(maxFailures + " Java failures");
            }
            if (maxMismatches < Integer.MAX_VALUE) {
                limits.add(maxMismatches + " AST mismatches");
            }
            if (maxTooPermissive < Integer.MAX_VALUE) {
                limits.add(maxTooPermissive + " too-permissive cases");
            }
            System.out.println("Will stop after " + String.join(" or ", limits) + "\n");
        } else {
            System.out.println();
        }

        // Log file for debugging hangs - write current file being processed
        Path progressLogPath = Paths.get("/tmp/directory-tester-progress.log");

        // Immediately write failures to files as they're found
        Path failuresPath = Paths.get("/tmp/java-parser-failures.txt");
        Path mismatchesPath = Paths.get("/tmp/java-parser-mismatches.txt");
        Path tooPermissivePath = Paths.get("/tmp/java-parser-too-permissive.txt");
        try {
            Files.deleteIfExists(failuresPath);
            Files.deleteIfExists(mismatchesPath);
            Files.deleteIfExists(tooPermissivePath);
        } catch (IOException e) {
            // Ignore
        }

        for (Path file : jsFiles) {
            // Check if we've hit max failures, mismatches, or too-permissive
            if (javaFailedAcornSucceeded.get() >= maxFailures) {
                System.out.println("\n⚠ Reached maximum Java failure limit (" + maxFailures + "), stopping early");
                break;
            }
            if (mismatched.get() >= maxMismatches) {
                System.out.println("\n⚠ Reached maximum mismatch limit (" + maxMismatches + "), stopping early");
                break;
            }
            if (javaSucceededAcornFailed.get() >= maxTooPermissive) {
                System.out.println("\n⚠ Reached maximum too-permissive limit (" + maxTooPermissive + "), stopping early");
                break;
            }

            processed.incrementAndGet();

            String relativePath = targetDir.relativize(file).toString();

            // Log current file to help debug hangs
            try (BufferedWriter logWriter = new BufferedWriter(new FileWriter(progressLogPath.toFile()))) {
                logWriter.write("Processing #" + processed.get() + ": " + file.toAbsolutePath());
                logWriter.newLine();
                logWriter.write("Relative: " + relativePath);
                logWriter.newLine();
                logWriter.flush();
            } catch (IOException logErr) {
                // Ignore logging errors
            }

            // Print progress every 100 files (or every 10 if less than 100 total)
            int progressInterval = jsFiles.size() < 100 ? 10 : 100;
            if (processed.get() % progressInterval == 0 || processed.get() == jsFiles.size()) {
                System.out.printf("\rProgress: %d/%d (%d matched, %d mismatched, %d Java failed, %d too permissive, %d both failed) - %s",
                    processed.get(), jsFiles.size(), matched.get(), mismatched.get(),
                    javaFailedAcornSucceeded.get(), javaSucceededAcornFailed.get(), bothFailed.get(),
                    relativePath.length() > 60 ? "..." + relativePath.substring(relativePath.length() - 57) : relativePath);
                System.out.flush();
            }

            try {
                String source = Files.readString(file);

                // Parse with Acorn
                AcornResult acornResult = parseWithAcorn(source, file);
                boolean acornSucceeded = acornResult.success;

                if (acornSucceeded) {
                    acornSuccess.incrementAndGet();
                } else {
                    acornFailed.incrementAndGet();
                }

                // Parse with Java - try both script and module mode
                // If Acorn succeeded, we know the correct mode; otherwise try both
                Program javaAst = null;
                boolean javaSucceeded = false;
                String javaError = null;
                boolean usedModuleMode = false;

                if (acornSucceeded) {
                    // We know the correct mode from Acorn
                    boolean isModule = acornResult.sourceType.equals("module");
                    try {
                        // Parse with timeout to prevent hanging on large files
                        javaAst = parseWithTimeout(source, isModule, 5);
                        javaSucceeded = true;
                        usedModuleMode = isModule;
                        javaSuccess.incrementAndGet();
                    } catch (java.util.concurrent.TimeoutException e) {
                        javaFailed.incrementAndGet();
                        javaError = "Timeout after 5 seconds";
                    } catch (ParseException e) {
                        javaFailed.incrementAndGet();
                        javaError = e.getMessage();

                        // Write first parse error details to file for debugging
                        if (javaFailed.get() == 1) {
                            try {
                                StringBuilder debug = new StringBuilder();
                                debug.append("File: ").append(relativePath).append("\n");
                                debug.append("Error: ").append(javaError).append("\n");

                                Token token = e.getToken();
                                if (token != null) {
                                    debug.append("Line: ").append(token.line()).append(" Column: ").append(token.column()).append("\n");
                                    debug.append("Token: ").append(token).append("\n");
                                }
                                debug.append("IsModule: ").append(isModule).append("\n\n");

                                // Try to get source context
                                if (token != null && token.line() > 0) {
                                    String[] lines = source.split("\n");
                                    int lineIdx = token.line() - 1;
                                    if (lineIdx >= 0 && lineIdx < lines.length) {
                                        String line = lines[lineIdx];
                                        int col = token.column();
                                        int start = Math.max(0, col - 100);
                                        int end = Math.min(line.length(), col + 100);
                                        debug.append("Source context:\n");
                                        debug.append(line.substring(start, end)).append("\n");
                                        if (col - start >= 0) {
                                            debug.append(" ".repeat(Math.min(col - start, 100))).append("^\n");
                                        }
                                    }
                                }

                                Files.writeString(Paths.get("/tmp/java-parser-error.txt"), debug.toString());
                                System.out.println("\n=== First Java parser error written to /tmp/java-parser-error.txt ===");
                            } catch (Exception debugErr) {
                                System.out.println("Debug error: " + debugErr.getMessage());
                            }
                        }

                        if (javaError != null && javaError.length() > 100) {
                            javaError = javaError.substring(0, 100);
                        }
                    } catch (Exception e) {
                        javaFailed.incrementAndGet();
                        javaError = e.getMessage();
                    }
                } else {
                    // Acorn failed - try both modes and see if either works
                    Program scriptAst = null;
                    Program moduleAst = null;
                    Exception scriptError = null;
                    Exception moduleError = null;

                    try {
                        scriptAst = parseWithTimeout(source, false, 5);
                    } catch (Exception e) {
                        scriptError = e;
                    }

                    try {
                        moduleAst = parseWithTimeout(source, true, 5);
                    } catch (Exception e) {
                        moduleError = e;
                    }

                    if (scriptAst != null) {
                        javaAst = scriptAst;
                        javaSucceeded = true;
                        usedModuleMode = false;
                        javaSuccess.incrementAndGet();
                    } else if (moduleAst != null) {
                        javaAst = moduleAst;
                        javaSucceeded = true;
                        usedModuleMode = true;
                        javaSuccess.incrementAndGet();
                    } else {
                        javaFailed.incrementAndGet();
                        javaError = scriptError != null ? scriptError.getMessage() :
                                   (moduleError != null ? moduleError.getMessage() : "Unknown error");
                        if (javaError != null && javaError.length() > 100) {
                            javaError = javaError.substring(0, 100);
                        }
                    }
                }

                // Analyze outcomes
                if (!acornSucceeded && !javaSucceeded) {
                    // Both parsers failed - they agree the code is invalid
                    bothFailed.incrementAndGet();
                } else if (acornSucceeded && !javaSucceeded) {
                    // Acorn succeeded but Java failed - Java parser bug
                    javaFailedAcornSucceeded.incrementAndGet();
                    if (javaError != null) {
                        errorMessages.merge(javaError, 1, Integer::sum);
                    }
                    if (javaFailedFiles.size() < 50) {
                        javaFailedFiles.add(relativePath + ": " + javaError);
                    }

                    // Immediately write failure to file (append)
                    try (BufferedWriter fw = new BufferedWriter(new FileWriter(failuresPath.toFile(), true))) {
                        fw.write(file.toAbsolutePath().toString() + ": " + javaError);
                        fw.newLine();
                        fw.flush();
                    } catch (IOException appendErr) {
                        // Ignore
                    }

                    // Cache Acorn result for test generation
                    if (cacheBuilder != null && acornResult.astJson != null) {
                        try {
                            AcornCacheBuilder.CacheResult cacheResult = cacheBuilder.cacheFile(file, acornResult.sourceType);
                            if (cacheResult.success) {
                                cachedCount.incrementAndGet();
                            }
                        } catch (Exception cacheErr) {
                            // Ignore caching errors
                        }
                    }
                } else if (!acornSucceeded && javaSucceeded) {
                    // Java succeeded but Acorn failed - Java is too permissive
                    javaSucceededAcornFailed.incrementAndGet();
                    if (javaSucceededAcornFailedFiles.size() < 50) {
                        javaSucceededAcornFailedFiles.add(relativePath);
                    }

                    // Immediately write too-permissive to file (append)
                    try (BufferedWriter fw = new BufferedWriter(new FileWriter(tooPermissivePath.toFile(), true))) {
                        fw.write(file.toAbsolutePath().toString());
                        fw.newLine();
                        fw.flush();
                    } catch (IOException appendErr) {
                        // Ignore
                    }
                } else {
                    // Both succeeded - compare ASTs
                    String acornJson = acornResult.astJson;
                    String javaJson = mapper.writeValueAsString(javaAst);

                    // Parse both for structural comparison
                    Object acornObj = mapper.readValue(acornJson, Object.class);
                    Object javaObj = mapper.readValue(javaJson, Object.class);

                    // Normalize
                    normalizeRegexValues(acornObj, javaObj);
                    normalizeBigIntValues(acornObj, javaObj);

                    if (Objects.deepEquals(acornObj, javaObj)) {
                        matched.incrementAndGet();
                    } else {
                        mismatched.incrementAndGet();
                        if (mismatchedFiles.size() < 20) {
                            mismatchedFiles.add(relativePath);
                        }

                        // Immediately write mismatch to file (append)
                        try (BufferedWriter fw = new BufferedWriter(new FileWriter(mismatchesPath.toFile(), true))) {
                            fw.write(file.toAbsolutePath().toString());
                            fw.newLine();
                            fw.flush();
                        } catch (IOException appendErr) {
                            // Ignore
                        }

                        // Write first mismatch ASTs to files for debugging
                        if (mismatched.get() == 1) {
                            try {
                                Files.writeString(Paths.get("/tmp/acorn-ast-mismatch.json"), acornJson);
                                Files.writeString(Paths.get("/tmp/java-ast-mismatch.json"), javaJson);
                                System.out.println("\n=== First mismatch found in: " + relativePath + " ===");
                                System.out.println("ASTs written to /tmp/acorn-ast-mismatch.json and /tmp/java-ast-mismatch.json");
                            } catch (IOException debugErr) {
                                // Ignore
                            }
                        }

                        // Cache Acorn result for test generation
                        if (cacheBuilder != null) {
                            try {
                                AcornCacheBuilder.CacheResult cacheResult = cacheBuilder.cacheFile(file, acornResult.sourceType);
                                if (cacheResult.success) {
                                    cachedCount.incrementAndGet();
                                }
                            } catch (Exception cacheErr) {
                                // Ignore caching errors
                            }
                        }
                    }
                }
            } catch (Exception e) {
                javaFailedAcornSucceeded.incrementAndGet();
                if (javaFailedFiles.size() < 50) {
                    javaFailedFiles.add(relativePath + ": [Comparison error] " + e.getMessage());
                }
            }
        }

        // Print results
        System.out.println();  // New line after progress
        System.out.println("\n=== Results ===");
        System.out.printf("Total files: %d%n", jsFiles.size());

        System.out.println("\nParser outcomes:");
        System.out.printf("  ✓ Both succeeded + matched: %d (%.2f%%)%n",
            matched.get(), (matched.get() * 100.0 / jsFiles.size()));
        System.out.printf("  ⚠ Both succeeded + AST mismatch: %d (%.2f%%)%n",
            mismatched.get(), (mismatched.get() * 100.0 / jsFiles.size()));
        System.out.printf("  ✓ Both failed (agree invalid): %d (%.2f%%)%n",
            bothFailed.get(), (bothFailed.get() * 100.0 / jsFiles.size()));
        System.out.printf("  ✗ Java failed, Acorn succeeded: %d (%.2f%%)%n",
            javaFailedAcornSucceeded.get(), (javaFailedAcornSucceeded.get() * 100.0 / jsFiles.size()));
        System.out.printf("  ⚠ Java succeeded, Acorn failed (too permissive): %d (%.2f%%)%n",
            javaSucceededAcornFailed.get(), (javaSucceededAcornFailed.get() * 100.0 / jsFiles.size()));

        System.out.println("\nAcorn results:");
        System.out.printf("  ✓ Successfully parsed: %d (%.2f%%)%n",
            acornSuccess.get(), (acornSuccess.get() * 100.0 / jsFiles.size()));
        System.out.printf("  ✗ Failed to parse: %d (%.2f%%)%n",
            acornFailed.get(), (acornFailed.get() * 100.0 / jsFiles.size()));

        System.out.println("\nJava parser results:");
        System.out.printf("  ✓ Successfully parsed: %d (%.2f%%)%n",
            javaSuccess.get(), (javaSuccess.get() * 100.0 / jsFiles.size()));
        System.out.printf("  ✗ Failed to parse: %d (%.2f%%)%n",
            javaFailed.get(), (javaFailed.get() * 100.0 / jsFiles.size()));

        if (!mismatchedFiles.isEmpty()) {
            System.out.println("\nFirst 20 AST mismatches (both parsers succeeded but different ASTs):");
            mismatchedFiles.forEach(f -> System.out.println("  " + f));
        }

        if (!javaFailedFiles.isEmpty()) {
            System.out.println("\nFirst 50 Java parser failures (Acorn succeeded):");
            javaFailedFiles.forEach(f -> System.out.println("  " + f));

            // Also write to a file for easy access
            try (BufferedWriter failureWriter = new BufferedWriter(new FileWriter("/tmp/java-parser-failures.txt"))) {
                for (String f : javaFailedFiles) {
                    failureWriter.write(f);
                    failureWriter.newLine();
                }
                System.out.println("  (Also written to /tmp/java-parser-failures.txt)");
            } catch (IOException e) {
                // Ignore
            }
        }

        if (!javaSucceededAcornFailedFiles.isEmpty()) {
            System.out.println("\nFirst 50 files where Java is too permissive (Acorn failed):");
            javaSucceededAcornFailedFiles.forEach(f -> System.out.println("  " + f));
        }

        if (!errorMessages.isEmpty()) {
            System.out.println("\nJava parser error messages:");
            errorMessages.entrySet().stream()
                .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                .limit(20)
                .forEach(e -> System.out.printf("  [%d] %s%n", e.getValue(), e.getKey()));
        }

        // Print file locations summary
        System.out.println("\n=== Output Files ===");
        if (javaFailedAcornSucceeded.get() > 0) {
            System.out.println("Java failures:    " + failuresPath.toAbsolutePath() + " (" + javaFailedAcornSucceeded.get() + " files)");
        }
        if (mismatched.get() > 0) {
            System.out.println("AST mismatches:   " + mismatchesPath.toAbsolutePath() + " (" + mismatched.get() + " files)");
            System.out.println("First mismatch:   /tmp/acorn-ast-mismatch.json, /tmp/java-ast-mismatch.json");
        }
        if (javaSucceededAcornFailed.get() > 0) {
            System.out.println("Too permissive:   " + tooPermissivePath.toAbsolutePath() + " (" + javaSucceededAcornFailed.get() + " files)");
        }
        if (javaFailedAcornSucceeded.get() == 0 && mismatched.get() == 0 && javaSucceededAcornFailed.get() == 0) {
            System.out.println("No failures or mismatches to record.");
        }

        if (enableCaching && cachedCount.get() > 0) {
            System.out.println("\n=== Cache Summary ===");
            System.out.printf("Cached %d files to test-oracles/adhoc-cache%n", cachedCount.get());

            // Automatically regenerate JUnit tests from cache
            System.out.println("\nRegenerating JUnit tests from cache...");
            try {
                Path cacheDir = Paths.get("test-oracles/adhoc-cache");
                Path testOutputDir = Paths.get("src/test/java/com/jsparser");
                TestGeneratorFromCache generator = new TestGeneratorFromCache(cacheDir, testOutputDir);
                generator.generateTests();
                System.out.println("\n✓ JUnit tests regenerated successfully!");
                System.out.println("  Run 'mvn test' to execute the new tests.");
            } catch (Exception e) {
                System.err.println("\n✗ Failed to regenerate tests: " + e.getMessage());
                System.err.println("  You can manually run: mvn exec:java -Dexec.mainClass=\"com.jsparser.TestGeneratorFromCache\"");
            }
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
        // Try both script and module mode, prefer the one that succeeds
        // If both succeed, prefer module if file is .mjs or has import/export
        AcornResult scriptResult = tryParseWithAcorn(source, filePath, "script");

        // If script mode timed out, don't bother trying module mode
        if (scriptResult.error != null && scriptResult.error.contains("Timeout")) {
            return scriptResult;
        }

        AcornResult moduleResult = tryParseWithAcorn(source, filePath, "module");

        // If both succeed, prefer module for .mjs files or files with import/export
        if (scriptResult.success && moduleResult.success) {
            boolean preferModule = filePath.toString().endsWith(".mjs") ||
                                  source.contains("import ") ||
                                  source.contains("export ") ||
                                  source.contains("flags:\n  - module") ||
                                  source.contains("flags: [module]");
            return preferModule ? moduleResult : scriptResult;
        }

        // Return whichever succeeded
        if (scriptResult.success) {
            return scriptResult;
        }
        if (moduleResult.success) {
            return moduleResult;
        }

        // Both failed, return script error
        return scriptResult;
    }

    private static AcornResult tryParseWithAcorn(String source, Path filePath, String sourceType) {
        try {
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

            // Add timeout to prevent hanging on problematic files
            boolean finished = process.waitFor(2, java.util.concurrent.TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                tempFile.toFile().delete();
                return new AcornResult(false, null, "script", "Timeout after 2 seconds");
            }

            int exitCode = process.exitValue();
            String errorOutput = "";
            try {
                errorOutput = errorFuture.get(1, java.util.concurrent.TimeUnit.SECONDS);
            } catch (java.util.concurrent.TimeoutException e) {
                errorOutput = "Timeout reading error output";
            }

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
                // Normalize the empty value map to null
                if (expMap.get("value") == null && actMap.get("value") instanceof Map) {
                    Map<?, ?> actValue = (Map<?, ?>) actMap.get("value");
                    if (actValue.isEmpty()) {
                        actMap.put("value", null);
                    }
                }

                // Normalize regex field: both Acorn and our parser have it, ensure they match
                Object expRegex = expMap.get("regex");
                Object actRegex = actMap.get("regex");

                if (expRegex instanceof Map && actRegex instanceof Map) {
                    Map<String, Object> expRegexMap = (Map<String, Object>) expRegex;
                    Map<String, Object> actRegexMap = (Map<String, Object>) actRegex;

                    // Both should have pattern and flags
                    // If they match, we're good. The comparison will handle this.
                    // No normalization needed if structures are identical.
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

    /**
     * Parse with timeout to prevent hanging on problematic files
     */
    private static Program parseWithTimeout(String source, boolean isModule, int timeoutSeconds)
            throws Exception {
        java.util.concurrent.ExecutorService executor = java.util.concurrent.Executors.newSingleThreadExecutor();
        try {
            java.util.concurrent.Future<Program> future = executor.submit(() -> Parser.parse(source, isModule));
            return future.get(timeoutSeconds, java.util.concurrent.TimeUnit.SECONDS);
        } catch (java.util.concurrent.TimeoutException e) {
            executor.shutdownNow();
            throw e;
        } catch (java.util.concurrent.ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof Exception) {
                throw (Exception) cause;
            }
            throw new RuntimeException(cause);
        } finally {
            executor.shutdown();
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
