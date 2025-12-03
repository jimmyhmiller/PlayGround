package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.concurrent.CompletableFuture;

/**
 * Utility to cache Acorn AST results for files.
 * This allows us to create tests from discovered failures/mismatches.
 */
public class AcornCacheBuilder {

    private static final ObjectMapper mapper = new ObjectMapper();
    private final Path cacheBaseDir;

    public AcornCacheBuilder(Path cacheBaseDir) {
        this.cacheBaseDir = cacheBaseDir;
    }

    /**
     * Parse a file with Acorn and cache the result
     * @param filePath Path to JavaScript file to parse
     * @param sourceType "script" or "module"
     * @return CacheResult containing success status and cache file path
     */
    public CacheResult cacheFile(Path filePath, String sourceType) throws IOException {
        String source = Files.readString(filePath);

        // Generate cache file path based on original file path
        Path cacheFilePath = generateCacheFilePath(filePath);

        // Ensure cache directory exists
        Files.createDirectories(cacheFilePath.getParent());

        // Parse with Acorn
        AcornResult result = parseWithAcorn(source, filePath, sourceType);

        if (!result.success) {
            return new CacheResult(false, null, result.error);
        }

        // Create cached AST with metadata
        ObjectNode cachedData = mapper.createObjectNode();

        // Add metadata
        ObjectNode metadata = mapper.createObjectNode();
        metadata.put("originalFile", filePath.toString());
        metadata.put("sourceType", sourceType);
        metadata.put("cacheDate", java.time.Instant.now().toString());
        metadata.put("fileHash", computeFileHash(source));
        cachedData.set("_metadata", metadata);

        // Add AST
        cachedData.set("ast", mapper.readTree(result.astJson));

        // Write to cache
        Files.writeString(cacheFilePath,
            mapper.writerWithDefaultPrettyPrinter().writeValueAsString(cachedData),
            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        return new CacheResult(true, cacheFilePath, null);
    }

    /**
     * Generate a cache file path for a given source file.
     * This creates a unique, filesystem-safe path in the cache directory.
     */
    private Path generateCacheFilePath(Path originalFile) {
        // Create a safe filename by replacing path separators and special chars
        String relativePath = originalFile.toString();

        // Replace path separators and special characters with underscores
        String safeName = relativePath
            .replace("/", "_")
            .replace("\\", "_")
            .replace("..", ".._")
            .replaceAll("[^a-zA-Z0-9._-]", "_");

        return cacheBaseDir.resolve(safeName + ".json");
    }

    /**
     * Compute SHA-256 hash of file contents for cache validation
     */
    private String computeFileHash(String content) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(content.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("SHA-256 not available", e);
        }
    }

    /**
     * Parse a file with Acorn and return the result
     */
    private AcornResult parseWithAcorn(String source, Path filePath, String sourceType) {
        try {
            // Write source to temp file
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
            CompletableFuture<String> errorFuture = CompletableFuture.supplyAsync(() -> {
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

    /**
     * Result of caching operation
     */
    public static class CacheResult {
        public final boolean success;
        public final Path cacheFilePath;
        public final String error;

        public CacheResult(boolean success, Path cacheFilePath, String error) {
            this.success = success;
            this.cacheFilePath = cacheFilePath;
            this.error = error;
        }
    }

    /**
     * Result from Acorn parsing
     */
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

    /**
     * Main method for manual cache generation
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 1) {
            System.err.println("Usage: java AcornCacheBuilder <file-path> [sourceType]");
            System.err.println("  sourceType: 'script' or 'module' (default: auto-detect)");
            System.exit(1);
        }

        Path filePath = Paths.get(args[0]);
        if (!Files.exists(filePath)) {
            System.err.println("File not found: " + filePath.toAbsolutePath());
            System.exit(1);
        }

        // Auto-detect source type if not provided
        String sourceType = "script";
        if (args.length > 1) {
            sourceType = args[1];
        } else {
            String source = Files.readString(filePath);
            if (filePath.toString().endsWith(".mjs") ||
                source.contains("import ") ||
                source.contains("export ")) {
                sourceType = "module";
            }
        }

        Path cacheDir = Paths.get("test-oracles/adhoc-cache");
        AcornCacheBuilder builder = new AcornCacheBuilder(cacheDir);

        System.out.println("Caching AST for: " + filePath);
        System.out.println("Source type: " + sourceType);

        CacheResult result = builder.cacheFile(filePath, sourceType);

        if (result.success) {
            System.out.println("✓ Cached to: " + result.cacheFilePath);
        } else {
            System.err.println("✗ Failed: " + result.error);
            System.exit(1);
        }
    }
}
