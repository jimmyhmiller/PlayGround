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
 * Utility to cache JavaScript source files for testing.
 * Instead of caching the massive AST JSON (which can be 250x larger than source),
 * we cache the JS source file and a small metadata file.
 * At test time, both Acorn and our parser parse the source and compare ASTs.
 */
public class AcornCacheBuilder {

    private static final ObjectMapper mapper = new ObjectMapper();
    private final Path cacheBaseDir;

    public AcornCacheBuilder(Path cacheBaseDir) {
        this.cacheBaseDir = cacheBaseDir;
    }

    /**
     * Cache a JavaScript file for later testing.
     * Copies the source file and creates a metadata file with sourceType info.
     * @param filePath Path to JavaScript file to cache
     * @param sourceType "script" or "module"
     * @return CacheResult containing success status and cache file path
     */
    public CacheResult cacheFile(Path filePath, String sourceType) throws IOException {
        String source = readFileWithFallbackEncoding(filePath);

        // Generate cache file paths
        Path cacheFilePath = generateCacheFilePath(filePath);
        Path metaFilePath = Paths.get(cacheFilePath.toString() + ".meta");

        // Ensure cache directory exists
        Files.createDirectories(cacheFilePath.getParent());

        // Verify Acorn can parse this file before caching
        AcornResult result = verifyAcornCanParse(source, filePath, sourceType);

        if (!result.success) {
            return new CacheResult(false, null, result.error);
        }

        // Copy the JS source to cache (much smaller than AST JSON!)
        Files.writeString(cacheFilePath, source,
            StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        // Write small metadata file
        ObjectNode metadata = mapper.createObjectNode();
        metadata.put("originalFile", filePath.toString());
        metadata.put("sourceType", sourceType);
        metadata.put("cacheDate", java.time.Instant.now().toString());
        metadata.put("fileHash", computeFileHash(source));
        metadata.put("fileSize", source.length());

        Files.writeString(metaFilePath,
            mapper.writerWithDefaultPrettyPrinter().writeValueAsString(metadata),
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

        // Keep the .js extension since we're caching JS source now
        return cacheBaseDir.resolve(safeName);
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
     * Verify that Acorn can parse the file (but don't return the full AST - it can be huge!)
     */
    private AcornResult verifyAcornCanParse(String source, Path filePath, String sourceType) {
        try {
            // Build Node.js command to parse with Acorn and just verify it succeeds
            String[] cmd = {
                "node",
                "-e",
                "const acorn = require('acorn'); " +
                "const fs = require('fs'); " +
                "const source = fs.readFileSync(process.argv[1], 'utf-8'); " +
                "try { " +
                "  const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: '" + sourceType + "'}); " +
                "  console.log('OK: ' + ast.body.length + ' statements'); " +
                "  process.exit(0); " +
                "} catch (e) { " +
                "  console.error('PARSE_ERROR: ' + e.message); " +
                "  process.exit(1); " +
                "}",
                filePath.toAbsolutePath().toString()
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
                return new AcornResult(false, null, null, errorOutput);
            }

            return new AcornResult(true, null, sourceType, null);
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
            String source = readFileWithFallbackEncoding(filePath);
            if (filePath.toString().endsWith(".mjs") ||
                source.contains("import ") ||
                source.contains("export ")) {
                sourceType = "module";
            }
        }

        Path cacheDir = Paths.get("test-oracles/adhoc-cache");
        AcornCacheBuilder builder = new AcornCacheBuilder(cacheDir);

        System.out.println("Caching JS source for: " + filePath);
        System.out.println("Source type: " + sourceType);

        CacheResult result = builder.cacheFile(filePath, sourceType);

        if (result.success) {
            System.out.println("✓ Cached source to: " + result.cacheFilePath);
            System.out.println("✓ Metadata at: " + result.cacheFilePath + ".meta");
        } else {
            System.err.println("✗ Failed: " + result.error);
            System.exit(1);
        }
    }

    /**
     * Read file contents, trying UTF-8 first then falling back to ISO-8859-1
     */
    private static String readFileWithFallbackEncoding(Path filePath) throws IOException {
        try {
            return Files.readString(filePath);
        } catch (java.nio.charset.MalformedInputException e) {
            return Files.readString(filePath, java.nio.charset.StandardCharsets.ISO_8859_1);
        }
    }
}
