package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

/**
 * Generate JUnit test files from cached JavaScript source files.
 * Scans the cache directory for .js files (with .meta files) and creates test classes.
 * Tests run both Acorn and our parser in real-time and compare ASTs using temp files.
 */
public class TestGeneratorFromCache {

    private static final ObjectMapper mapper = new ObjectMapper();
    private final Path cacheBaseDir;
    private final Path testOutputDir;

    public TestGeneratorFromCache(Path cacheBaseDir, Path testOutputDir) {
        this.cacheBaseDir = cacheBaseDir;
        this.testOutputDir = testOutputDir;
    }

    /**
     * Scan cache directory and generate test classes
     */
    public void generateTests() throws IOException {
        System.out.println("Scanning cache directory: " + cacheBaseDir);

        // Collect all cached JS files (those with .meta files)
        List<Path> cacheFiles = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(cacheBaseDir)) {
            paths.filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".js") &&
                             Files.exists(Paths.get(p.toString() + ".meta")))
                .forEach(cacheFiles::add);
        }

        System.out.println("Found " + cacheFiles.size() + " cached JS files");

        if (cacheFiles.isEmpty()) {
            System.out.println("No cache files found. Run DirectoryTester or AcornCacheBuilder first.");
            return;
        }

        // Group cache files by category for organized test generation
        Map<String, List<CacheEntry>> categorizedTests = new HashMap<>();

        for (Path cacheFile : cacheFiles) {
            try {
                CacheEntry entry = readCacheFile(cacheFile);
                String category = determineCategory(entry);

                categorizedTests.computeIfAbsent(category, k -> new ArrayList<>()).add(entry);
            } catch (Exception e) {
                System.err.println("Failed to read cache file " + cacheFile + ": " + e.getMessage());
            }
        }

        // Generate test classes for each category
        for (Map.Entry<String, List<CacheEntry>> categoryEntry : categorizedTests.entrySet()) {
            String category = categoryEntry.getKey();
            List<CacheEntry> entries = categoryEntry.getValue();

            generateTestClass(category, entries);
        }

        System.out.println("\n✓ Generated test classes in: " + testOutputDir);
    }

    /**
     * Read a cache file and extract metadata from the .meta file
     */
    private CacheEntry readCacheFile(Path cacheFile) throws IOException {
        Path metaFile = Paths.get(cacheFile.toString() + ".meta");
        String metaContent = Files.readString(metaFile);
        JsonNode metadata = mapper.readTree(metaContent);

        String originalFile = metadata.get("originalFile").asText();
        String sourceType = metadata.get("sourceType").asText("script");

        return new CacheEntry(cacheFile, originalFile, sourceType);
    }

    /**
     * Determine test category from file path
     */
    private String determineCategory(CacheEntry entry) {
        String path = entry.originalFile.toLowerCase();

        // Extract meaningful category from path
        if (path.contains("node_modules/monaco-editor")) {
            return "MonacoEditor";
        } else if (path.contains("node_modules/tailwindcss")) {
            return "Tailwindcss";
        } else if (path.contains("vscode")) {
            return "VSCode";
        } else if (path.contains("node_modules")) {
            return "NodeModules";
        } else if (path.contains("react")) {
            return "React";
        } else {
            return "AdHoc";
        }
    }

    /**
     * Generate a JUnit test class for a category
     */
    private void generateTestClass(String category, List<CacheEntry> entries) throws IOException {
        String className = "Generated" + category + "Test";
        Path outputFile = testOutputDir.resolve(className + ".java");

        StringBuilder sb = new StringBuilder();

        // Package and imports
        sb.append("package com.jsparser;\n\n");
        sb.append("import com.fasterxml.jackson.databind.JsonNode;\n");
        sb.append("import com.fasterxml.jackson.databind.ObjectMapper;\n");
        sb.append("import com.fasterxml.jackson.databind.SerializationFeature;\n");
        sb.append("import com.jsparser.ast.Program;\n");
        sb.append("import org.junit.jupiter.api.DisplayName;\n");
        sb.append("import org.junit.jupiter.api.Test;\n\n");
        sb.append("import java.io.*;\n");
        sb.append("import java.nio.file.Files;\n");
        sb.append("import java.nio.file.Path;\n");
        sb.append("import java.nio.file.Paths;\n");
        sb.append("import java.util.List;\n");
        sb.append("import java.util.Map;\n");
        sb.append("import java.util.Objects;\n");
        sb.append("import java.util.concurrent.CompletableFuture;\n\n");
        sb.append("import static org.junit.jupiter.api.Assertions.fail;\n\n");

        // Class javadoc
        sb.append("/**\n");
        sb.append(" * Auto-generated tests from cached JS source files.\n");
        sb.append(" * Category: ").append(category).append("\n");
        sb.append(" * Generated: ").append(java.time.Instant.now()).append("\n");
        sb.append(" * \n");
        sb.append(" * Tests parse with both Acorn (real-time) and our parser, streaming ASTs to temp files\n");
        sb.append(" * for memory-efficient byte-for-byte comparison.\n");
        sb.append(" */\n");
        sb.append("public class ").append(className).append(" {\n\n");
        sb.append("    private static final ObjectMapper mapper;\n");
        sb.append("    static {\n");
        sb.append("        mapper = new ObjectMapper();\n");
        sb.append("        // Configure to match Node.js JSON.stringify(obj, null, 2) format\n");
        sb.append("        mapper.enable(SerializationFeature.INDENT_OUTPUT);\n");
        sb.append("        mapper.setDefaultPrettyPrinter(new com.fasterxml.jackson.core.util.DefaultPrettyPrinter()\n");
        sb.append("            .withObjectIndenter(new com.fasterxml.jackson.core.util.DefaultIndenter(\"  \", \"\\n\"))\n");
        sb.append("            .withSeparators(com.fasterxml.jackson.core.util.Separators.createDefaultInstance()\n");
        sb.append("                .withObjectFieldValueSpacing(com.fasterxml.jackson.core.util.Separators.Spacing.NONE)));\n");
        sb.append("    }\n\n");

        // Generate test methods for ALL cached files
        int count = 0;
        for (CacheEntry entry : entries) {
            generateTestMethod(sb, entry, count);
            count++;
        }

        // Add helper methods
        appendHelperMethods(sb);

        sb.append("}\n");

        // Write test file
        Files.createDirectories(testOutputDir);
        Files.writeString(outputFile, sb.toString());

        System.out.println("Generated: " + outputFile + " (" + count + " tests)");
    }

    /**
     * Generate a single test method
     */
    private void generateTestMethod(StringBuilder sb, CacheEntry entry, int index) {
        String methodName = "test_" + sanitizeMethodName(entry.originalFile) + "_" + index;
        String displayName = shortenPath(entry.originalFile);

        sb.append("    @Test\n");
        sb.append("    @DisplayName(\"").append(escapeJava(displayName)).append("\")\n");
        sb.append("    void ").append(methodName).append("() throws Exception {\n");
        sb.append("        // Original file: ").append(entry.originalFile).append("\n");
        sb.append("        // Cached JS source: ").append(entry.cacheFile.getFileName()).append("\n");
        sb.append("        assertASTMatches(\n");
        sb.append("            \"").append(escapeJava(entry.cacheFile.toString())).append("\",\n");
        sb.append("            ").append(entry.sourceType.equals("module")).append("\n");
        sb.append("        );\n");
        sb.append("    }\n\n");
    }

    /**
     * Append helper methods to test class - memory-efficient streaming approach
     */
    private void appendHelperMethods(StringBuilder sb) {
        sb.append("    // Helper methods\n\n");

        // Main assertion method - streams both ASTs to temp files and uses hash comparison
        sb.append("    private void assertASTMatches(String cachedJsPath, boolean isModule) throws Exception {\n");
        sb.append("        // Read cached JS source\n");
        sb.append("        String source = Files.readString(Paths.get(cachedJsPath));\n\n");
        sb.append("        Path acornTmp = Files.createTempFile(\"acorn-ast-\", \".json\");\n");
        sb.append("        Path ourTmp = Files.createTempFile(\"our-ast-\", \".json\");\n\n");
        sb.append("        try {\n");
        sb.append("            // 1. Stream Acorn AST to temp file (via Node.js subprocess)\n");
        sb.append("            runAcornToFile(cachedJsPath, isModule, acornTmp);\n\n");
        sb.append("            // 2. Stream our AST to temp file\n");
        sb.append("            Program ourAst = Parser.parse(source, isModule);\n");
        sb.append("            try (var out = new BufferedOutputStream(Files.newOutputStream(ourTmp))) {\n");
        sb.append("                mapper.writeValue(out, ourAst);\n");
        sb.append("            }\n\n");
        sb.append("            // 3. Compare files via hash (ignoring whitespace)\n");
        sb.append("            String acornHash = hashJsonContent(acornTmp);\n");
        sb.append("            String ourHash = hashJsonContent(ourTmp);\n\n");
        sb.append("            if (!acornHash.equals(ourHash)) {\n");
        sb.append("                // Hashes differ - show first difference\n");
        sb.append("                showFirstDifference(acornTmp, ourTmp, cachedJsPath);\n");
        sb.append("                fail(\"AST mismatch for \" + cachedJsPath);\n");
        sb.append("            }\n");
        sb.append("        } finally {\n");
        sb.append("            Files.deleteIfExists(acornTmp);\n");
        sb.append("            Files.deleteIfExists(ourTmp);\n");
        sb.append("        }\n");
        sb.append("    }\n\n");

        // Method to run Acorn and write to file using streaming JSON for large files
        sb.append("    private void runAcornToFile(String jsPath, boolean isModule, Path outputFile) throws Exception {\n");
        sb.append("        String sourceType = isModule ? \"module\" : \"script\";\n");
        sb.append("        // Use streaming JSON writer to avoid Node.js string length limits\n");
        sb.append("        String script = \"\"\"\n");
        sb.append("            const acorn = require('acorn');\n");
        sb.append("            const fs = require('fs');\n");
        sb.append("            const source = fs.readFileSync(process.argv[1], 'utf-8');\n");
        sb.append("            const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: '%s'});\n");
        sb.append("            // Streaming JSON writer to avoid string length limits\n");
        sb.append("            const out = fs.createWriteStream(process.argv[2]);\n");
        sb.append("            function writeJson(obj, indent) {\n");
        sb.append("                if (obj === null) { out.write('null'); return; }\n");
        sb.append("                if (typeof obj === 'undefined') { out.write('null'); return; }\n");
        sb.append("                if (typeof obj === 'boolean') { out.write(obj.toString()); return; }\n");
        sb.append("                if (typeof obj === 'number') { out.write(Number.isFinite(obj) ? obj.toString() : 'null'); return; }\n");
        sb.append("                if (typeof obj === 'bigint') { out.write('null'); return; }\n");
        sb.append("                if (typeof obj === 'string') { out.write(JSON.stringify(obj)); return; }\n");
        sb.append("                if (Array.isArray(obj)) {\n");
        sb.append("                    out.write('[');\n");
        sb.append("                    for (let i = 0; i < obj.length; i++) {\n");
        sb.append("                        if (i > 0) out.write(',');\n");
        sb.append("                        out.write('\\\\n' + indent + '  ');\n");
        sb.append("                        writeJson(obj[i], indent + '  ');\n");
        sb.append("                    }\n");
        sb.append("                    if (obj.length > 0) out.write('\\\\n' + indent);\n");
        sb.append("                    out.write(']');\n");
        sb.append("                    return;\n");
        sb.append("                }\n");
        sb.append("                out.write('{');\n");
        sb.append("                const keys = Object.keys(obj);\n");
        sb.append("                for (let i = 0; i < keys.length; i++) {\n");
        sb.append("                    if (i > 0) out.write(',');\n");
        sb.append("                    out.write('\\\\n' + indent + '  ');\n");
        sb.append("                    out.write(JSON.stringify(keys[i]) + ': ');\n");
        sb.append("                    writeJson(obj[keys[i]], indent + '  ');\n");
        sb.append("                }\n");
        sb.append("                if (keys.length > 0) out.write('\\\\n' + indent);\n");
        sb.append("                out.write('}');\n");
        sb.append("            }\n");
        sb.append("            writeJson(ast, '');\n");
        sb.append("            out.write('\\\\n');\n");
        sb.append("            out.end(() => process.exit(0));\n");
        sb.append("            \"\"\".formatted(sourceType);\n\n");
        sb.append("        String[] cmd = { \"node\", \"--max-old-space-size=8192\", \"-e\", script, jsPath, outputFile.toString() };\n");
        sb.append("        ProcessBuilder pb = new ProcessBuilder(cmd);\n");
        sb.append("        Process process = pb.start();\n\n");
        sb.append("        // Capture stderr\n");
        sb.append("        CompletableFuture<String> errorFuture = CompletableFuture.supplyAsync(() -> {\n");
        sb.append("            try {\n");
        sb.append("                return new String(process.getErrorStream().readAllBytes());\n");
        sb.append("            } catch (IOException e) { return \"\"; }\n");
        sb.append("        });\n\n");
        sb.append("        int exitCode = process.waitFor();\n");
        sb.append("        if (exitCode != 0) {\n");
        sb.append("            throw new RuntimeException(\"Acorn failed: \" + errorFuture.get());\n");
        sb.append("        }\n");
        sb.append("    }\n\n");

        // Method to compute a normalized hash of JSON using Jackson tree traversal
        sb.append("    private String hashJsonContent(Path input) throws Exception {\n");
        sb.append("        // Parse JSON tree and hash with sorted keys using Jackson\n");
        sb.append("        ObjectMapper localMapper = new ObjectMapper();\n");
        sb.append("        com.fasterxml.jackson.databind.JsonNode tree = localMapper.readTree(input.toFile());\n");
        sb.append("        java.security.MessageDigest digest = java.security.MessageDigest.getInstance(\"SHA-256\");\n");
        sb.append("        hashNode(tree, digest);\n");
        sb.append("        return java.util.HexFormat.of().formatHex(digest.digest());\n");
        sb.append("    }\n\n");
        sb.append("    private void hashNode(com.fasterxml.jackson.databind.JsonNode node, java.security.MessageDigest digest) {\n");
        sb.append("        if (node.isNull()) { digest.update(\"null\".getBytes()); return; }\n");
        sb.append("        if (node.isBoolean()) { digest.update(node.asText().getBytes()); return; }\n");
        sb.append("        if (node.isNumber()) { digest.update(node.asText().getBytes()); return; }\n");
        sb.append("        if (node.isTextual()) {\n");
        sb.append("            try { digest.update(mapper.writeValueAsBytes(node.asText())); }\n");
        sb.append("            catch (Exception e) { throw new RuntimeException(e); }\n");
        sb.append("            return;\n");
        sb.append("        }\n");
        sb.append("        if (node.isArray()) {\n");
        sb.append("            digest.update((byte)'[');\n");
        sb.append("            boolean first = true;\n");
        sb.append("            for (var elem : node) {\n");
        sb.append("                if (!first) digest.update((byte)',');\n");
        sb.append("                first = false;\n");
        sb.append("                hashNode(elem, digest);\n");
        sb.append("            }\n");
        sb.append("            digest.update((byte)']');\n");
        sb.append("            return;\n");
        sb.append("        }\n");
        sb.append("        if (node.isObject()) {\n");
        sb.append("            digest.update((byte)'{');\n");
        sb.append("            var fields = new java.util.ArrayList<String>();\n");
        sb.append("            node.fieldNames().forEachRemaining(fields::add);\n");
        sb.append("            java.util.Collections.sort(fields);\n");
        sb.append("            boolean first = true;\n");
        sb.append("            for (String field : fields) {\n");
        sb.append("                if (!first) digest.update((byte)',');\n");
        sb.append("                first = false;\n");
        sb.append("                try { digest.update(mapper.writeValueAsBytes(field)); }\n");
        sb.append("                catch (Exception e) { throw new RuntimeException(e); }\n");
        sb.append("                digest.update((byte)':');\n");
        sb.append("                hashNode(node.get(field), digest);\n");
        sb.append("            }\n");
        sb.append("            digest.update((byte)'}');\n");
        sb.append("        }\n");
        sb.append("    }\n\n");

        // Method to show first difference
        sb.append("    private void showFirstDifference(Path acornFile, Path ourFile, String jsPath) throws Exception {\n");
        sb.append("        // Run diff to get actual differences\n");
        sb.append("        ProcessBuilder pb = new ProcessBuilder(\"diff\", \"-u\",\n");
        sb.append("            acornFile.toString(), ourFile.toString());\n");
        sb.append("        Process p = pb.start();\n");
        sb.append("        String diff = new String(p.getInputStream().readAllBytes());\n");
        sb.append("        p.waitFor();\n\n");
        sb.append("        System.err.println(\"\\n=== AST MISMATCH: \" + jsPath + \" ===\");\n");
        sb.append("        // Show first 50 lines of diff\n");
        sb.append("        String[] lines = diff.split(\"\\\\n\");\n");
        sb.append("        for (int i = 0; i < Math.min(50, lines.length); i++) {\n");
        sb.append("            System.err.println(lines[i]);\n");
        sb.append("        }\n");
        sb.append("        if (lines.length > 50) {\n");
        sb.append("            System.err.println(\"... (\" + (lines.length - 50) + \" more lines)\");\n");
        sb.append("        }\n");
        sb.append("    }\n");
    }

    /**
     * Sanitize a file path to create a valid Java method name
     */
    private String sanitizeMethodName(String path) {
        String sanitized = path.replaceAll("[^a-zA-Z0-9]", "_")
                               .replaceAll("_{2,}", "_");
        return sanitized.substring(0, Math.min(sanitized.length(), 80));
    }

    /**
     * Shorten a path for display
     */
    private String shortenPath(String path) {
        if (path.length() <= 100) {
            return path;
        }
        return "..." + path.substring(path.length() - 97);
    }

    /**
     * Escape Java string
     */
    private String escapeJava(String str) {
        return str.replace("\\", "\\\\")
                 .replace("\"", "\\\"")
                 .replace("\n", "\\n")
                 .replace("\r", "\\r")
                 .replace("\t", "\\t");
    }

    /**
     * Cache entry data
     */
    private static class CacheEntry {
        final Path cacheFile;
        final String originalFile;
        final String sourceType;

        CacheEntry(Path cacheFile, String originalFile, String sourceType) {
            this.cacheFile = cacheFile;
            this.originalFile = originalFile;
            this.sourceType = sourceType;
        }
    }

    /**
     * Main entry point
     */
    public static void main(String[] args) throws IOException {
        Path cacheDir = Paths.get("test-oracles/adhoc-cache");
        Path testOutputDir = Paths.get("src/test/java/com/jsparser");

        if (args.length > 0) {
            cacheDir = Paths.get(args[0]);
        }
        if (args.length > 1) {
            testOutputDir = Paths.get(args[1]);
        }

        if (!Files.exists(cacheDir)) {
            System.err.println("Cache directory not found: " + cacheDir);
            System.err.println("Run DirectoryTester first to create cache files.");
            System.exit(1);
        }

        TestGeneratorFromCache generator = new TestGeneratorFromCache(cacheDir, testOutputDir);
        generator.generateTests();
    }
}
