package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Test a single JavaScript file against Acorn and write ASTs to files.
 */
public class SingleFileTester {

    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.err.println("Usage: java SingleFileTester <file-path>");
            System.err.println("");
            System.err.println("Examples:");
            System.err.println("  java SingleFileTester ../my-project/index.js");
            System.err.println("  java SingleFileTester ~/code/app/cli.js");
            System.exit(1);
        }

        String filePath = args[0];
        Path file = Paths.get(filePath);

        if (!Files.exists(file)) {
            System.err.println("File not found: " + file.toAbsolutePath());
            System.exit(1);
        }

        System.out.println("Testing file: " + file.toAbsolutePath());
        System.out.println("");

        String source = Files.readString(file);

        // Parse with Acorn
        System.out.println("Parsing with Acorn...");
        AcornResult acornResult = parseWithAcorn(source, file);

        if (!acornResult.success) {
            System.err.println("Acorn failed to parse:");
            System.err.println(acornResult.error);
            System.exit(1);
        }

        System.out.println("Acorn parse successful, source type: " + acornResult.sourceType);

        // Parse with Java
        System.out.println("Parsing with Java...");
        boolean isModule = acornResult.sourceType.equals("module");

        Program javaAst = null;
        String javaError = null;

        try {
            javaAst = Parser.parse(source, isModule);
            System.out.println("Java parse successful");
        } catch (ParseException e) {
            System.err.println("Java parser failed:");
            System.err.println(e.getMessage());

            Token token = e.getToken();
            if (token != null) {
                System.err.println("Line: " + token.line() + " Column: " + token.column());
                System.err.println("Token: " + token);
            }
            System.exit(1);
        } catch (Exception e) {
            System.err.println("Java parser failed with exception:");
            System.err.println(e.getMessage());
            System.exit(1);
        }

        // Compare ASTs
        System.out.println("\nComparing ASTs...");
        String acornJson = acornResult.astJson;
        String javaJson = mapper.writeValueAsString(javaAst);

        // Parse both for structural comparison
        Object acornObj = mapper.readValue(acornJson, Object.class);
        Object javaObj = mapper.readValue(javaJson, Object.class);

        // Normalize
        normalizeRegexValues(acornObj, javaObj);
        normalizeBigIntValues(acornObj, javaObj);

        // Write ASTs to files
        Path acornOutputPath = Paths.get("/tmp/acorn-cli-ast.json");
        Path javaOutputPath = Paths.get("/tmp/java-cli-ast.json");

        // Write directly to files to avoid memory issues with large ASTs
        mapper.writerWithDefaultPrettyPrinter().writeValue(acornOutputPath.toFile(), acornObj);
        mapper.writerWithDefaultPrettyPrinter().writeValue(javaOutputPath.toFile(), javaObj);

        System.out.println("Acorn AST written to: " + acornOutputPath);
        System.out.println("Java AST written to: " + javaOutputPath);

        if (Objects.deepEquals(acornObj, javaObj)) {
            System.out.println("\n✓ ASTs match!");
        } else {
            System.out.println("\n⚠ ASTs differ!");
            System.out.println("Use a diff tool to compare the files:");
            System.out.println("  diff /tmp/acorn-cli-ast.json /tmp/java-cli-ast.json");
        }
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
