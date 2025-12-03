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
 * Generate JUnit test files from cached Acorn ASTs.
 * Scans the cache directory and creates test classes for failures and mismatches.
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

        // Collect all cache files
        List<Path> cacheFiles = new ArrayList<>();
        try (Stream<Path> paths = Files.walk(cacheBaseDir)) {
            paths.filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith(".json") && !p.toString().endsWith("_summary.json"))
                .forEach(cacheFiles::add);
        }

        System.out.println("Found " + cacheFiles.size() + " cached AST files");

        if (cacheFiles.isEmpty()) {
            System.out.println("No cache files found. Run DirectoryTester first to cache results.");
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
     * Read a cache file and extract metadata
     */
    private CacheEntry readCacheFile(Path cacheFile) throws IOException {
        String content = Files.readString(cacheFile);
        JsonNode root = mapper.readTree(content);

        // Check if file has new format (with _metadata) or old format (just AST)
        JsonNode metadata = root.get("_metadata");
        String originalFile;
        String sourceType;

        if (metadata != null && metadata.get("originalFile") != null) {
            // New format with metadata
            originalFile = metadata.get("originalFile").asText();
            sourceType = metadata.get("sourceType").asText("script");
        } else {
            // Old format - derive info from cache file path
            String fileName = cacheFile.getFileName().toString();
            // Remove .json extension
            if (fileName.endsWith(".json")) {
                fileName = fileName.substring(0, fileName.length() - 5);
            }
            // Reconstruct original path from encoded filename
            originalFile = fileName.replace(".._", "../").replace("_", "/");
            // Guess source type - default to script
            sourceType = "script";
            if (originalFile.endsWith(".mjs")) {
                sourceType = "module";
            }
        }

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
        sb.append("import com.jsparser.ast.Program;\n");
        sb.append("import org.junit.jupiter.api.DisplayName;\n");
        sb.append("import org.junit.jupiter.api.Test;\n");
        sb.append("import org.junit.jupiter.api.Disabled;\n\n");
        sb.append("import java.nio.file.Files;\n");
        sb.append("import java.nio.file.Path;\n");
        sb.append("import java.nio.file.Paths;\n");
        sb.append("import java.util.List;\n");
        sb.append("import java.util.Map;\n");
        sb.append("import java.util.Objects;\n\n");
        sb.append("import static org.junit.jupiter.api.Assertions.assertTrue;\n\n");

        // Class javadoc
        sb.append("/**\n");
        sb.append(" * Auto-generated tests from Acorn AST cache.\n");
        sb.append(" * Category: ").append(category).append("\n");
        sb.append(" * Generated: ").append(java.time.Instant.now()).append("\n");
        sb.append(" * \n");
        sb.append(" * These tests verify that the Java parser produces the same AST as Acorn.\n");
        sb.append(" * Tests are initially @Disabled and will fail until parser bugs are fixed.\n");
        sb.append(" */\n");
        sb.append("@Disabled(\"Auto-generated tests - enable when ready to fix\")\n");
        sb.append("public class ").append(className).append(" {\n\n");
        sb.append("    private static final ObjectMapper mapper = new ObjectMapper();\n\n");

        // Generate test methods (limit to first 50 to avoid huge files)
        int count = 0;
        for (CacheEntry entry : entries) {
            if (count >= 50) {
                sb.append("    // ... ").append(entries.size() - 50).append(" more tests omitted\n\n");
                break;
            }

            generateTestMethod(sb, entry, count);
            count++;
        }

        // Add helper methods
        appendHelperMethods(sb);

        sb.append("}\n");

        // Write test file
        Files.createDirectories(testOutputDir);
        Files.writeString(outputFile, sb.toString());

        System.out.println("Generated: " + outputFile + " (" + Math.min(count, entries.size()) + " tests)");
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
        sb.append("        // Cache file: ").append(entry.cacheFile.getFileName()).append("\n");
        sb.append("        assertASTMatches(\n");
        sb.append("            \"").append(escapeJava(entry.cacheFile.toString())).append("\",\n");
        sb.append("            \"").append(escapeJava(entry.originalFile)).append("\",\n");
        sb.append("            ").append(entry.sourceType.equals("module")).append("\n");
        sb.append("        );\n");
        sb.append("    }\n\n");
    }

    /**
     * Append helper methods to test class
     */
    private void appendHelperMethods(StringBuilder sb) {
        sb.append("    // Helper methods\n\n");
        sb.append("    private void assertASTMatches(String cacheFilePath, String originalFilePath, boolean isModule) throws Exception {\n");
        sb.append("        // Read cached Acorn AST\n");
        sb.append("        String cacheContent = Files.readString(Paths.get(cacheFilePath));\n");
        sb.append("        JsonNode cacheRoot = mapper.readTree(cacheContent);\n");
        sb.append("        // Handle both old format (just AST) and new format (with _metadata)\n");
        sb.append("        JsonNode expectedAst = cacheRoot.has(\"ast\") ? cacheRoot.get(\"ast\") : cacheRoot;\n\n");
        sb.append("        // Read original source file\n");
        sb.append("        Path originalFile = Paths.get(originalFilePath);\n");
        sb.append("        if (!Files.exists(originalFile)) {\n");
        sb.append("            // File might have been moved/deleted - skip test\n");
        sb.append("            System.out.println(\"Skipping - file not found: \" + originalFilePath);\n");
        sb.append("            return;\n");
        sb.append("        }\n\n");
        sb.append("        String source = Files.readString(originalFile);\n\n");
        sb.append("        // Parse with Java parser\n");
        sb.append("        Program actual = Parser.parse(source, isModule);\n");
        sb.append("        String actualJson = mapper.writeValueAsString(actual);\n\n");
        sb.append("        // Compare ASTs\n");
        sb.append("        String expectedJson = mapper.writeValueAsString(expectedAst);\n");
        sb.append("        Object expectedObj = mapper.readValue(expectedJson, Object.class);\n");
        sb.append("        Object actualObj = mapper.readValue(actualJson, Object.class);\n\n");
        sb.append("        // Apply normalizations\n");
        sb.append("        normalizeRegexValues(expectedObj, actualObj);\n");
        sb.append("        normalizeBigIntValues(expectedObj, actualObj);\n\n");
        sb.append("        if (!Objects.deepEquals(expectedObj, actualObj)) {\n");
        sb.append("            System.out.println(\"\\nAST mismatch for: \" + originalFilePath);\n");
        sb.append("            // Could add diff printing here\n");
        sb.append("        }\n\n");
        sb.append("        assertTrue(Objects.deepEquals(expectedObj, actualObj),\n");
        sb.append("            \"AST mismatch for \" + originalFilePath);\n");
        sb.append("    }\n\n");

        // Add normalization methods
        sb.append("    @SuppressWarnings(\"unchecked\")\n");
        sb.append("    private void normalizeRegexValues(Object expected, Object actual) {\n");
        sb.append("        if (expected instanceof Map && actual instanceof Map) {\n");
        sb.append("            Map<String, Object> expMap = (Map<String, Object>) expected;\n");
        sb.append("            Map<String, Object> actMap = (Map<String, Object>) actual;\n");
        sb.append("            if (\"Literal\".equals(expMap.get(\"type\")) && expMap.containsKey(\"regex\")) {\n");
        sb.append("                if (expMap.get(\"value\") == null && actMap.get(\"value\") instanceof Map) {\n");
        sb.append("                    Map<?, ?> actValue = (Map<?, ?>) actMap.get(\"value\");\n");
        sb.append("                    if (actValue.isEmpty()) { actMap.put(\"value\", null); }\n");
        sb.append("                }\n");
        sb.append("            }\n");
        sb.append("            for (String key : expMap.keySet()) {\n");
        sb.append("                if (actMap.containsKey(key)) {\n");
        sb.append("                    normalizeRegexValues(expMap.get(key), actMap.get(key));\n");
        sb.append("                }\n");
        sb.append("            }\n");
        sb.append("        } else if (expected instanceof List && actual instanceof List) {\n");
        sb.append("            List<Object> expList = (List<Object>) expected;\n");
        sb.append("            List<Object> actList = (List<Object>) actual;\n");
        sb.append("            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {\n");
        sb.append("                normalizeRegexValues(expList.get(i), actList.get(i));\n");
        sb.append("            }\n");
        sb.append("        }\n");
        sb.append("    }\n\n");

        sb.append("    @SuppressWarnings(\"unchecked\")\n");
        sb.append("    private void normalizeBigIntValues(Object expected, Object actual) {\n");
        sb.append("        if (expected instanceof Map && actual instanceof Map) {\n");
        sb.append("            Map<String, Object> expMap = (Map<String, Object>) expected;\n");
        sb.append("            Map<String, Object> actMap = (Map<String, Object>) actual;\n");
        sb.append("            if (\"Literal\".equals(expMap.get(\"type\")) && expMap.containsKey(\"bigint\")) {\n");
        sb.append("                Object expBigint = expMap.get(\"bigint\");\n");
        sb.append("                Object actBigint = actMap.get(\"bigint\");\n");
        sb.append("                if (expBigint != null && actBigint != null) {\n");
        sb.append("                    String expStr = expBigint.toString();\n");
        sb.append("                    String actStr = actBigint.toString();\n");
        sb.append("                    if (!expStr.equals(actStr)) {\n");
        sb.append("                        try {\n");
        sb.append("                            java.math.BigInteger expBI = new java.math.BigInteger(expStr);\n");
        sb.append("                            java.math.BigInteger actBI = new java.math.BigInteger(actStr);\n");
        sb.append("                            if (expBI.equals(actBI)) { actMap.put(\"bigint\", expStr); }\n");
        sb.append("                        } catch (NumberFormatException e) {}\n");
        sb.append("                    }\n");
        sb.append("                }\n");
        sb.append("            }\n");
        sb.append("            for (String key : expMap.keySet()) {\n");
        sb.append("                if (actMap.containsKey(key)) {\n");
        sb.append("                    normalizeBigIntValues(expMap.get(key), actMap.get(key));\n");
        sb.append("                }\n");
        sb.append("            }\n");
        sb.append("        } else if (expected instanceof List && actual instanceof List) {\n");
        sb.append("            List<Object> expList = (List<Object>) expected;\n");
        sb.append("            List<Object> actList = (List<Object>) actual;\n");
        sb.append("            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {\n");
        sb.append("                normalizeBigIntValues(expList.get(i), actList.get(i));\n");
        sb.append("            }\n");
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
