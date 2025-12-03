package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Compare a single file's AST between Acorn and Java parser
 */
public class CompareFile {
    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
            .enable(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT);

    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            System.err.println("Usage: java CompareFile <file-path>");
            System.exit(1);
        }

        Path filePath = Paths.get(args[0]);
        if (!Files.exists(filePath)) {
            System.err.println("File not found: " + filePath.toAbsolutePath());
            System.exit(1);
        }

        String source = Files.readString(filePath);

        // Determine if module
        boolean isModule = filePath.toString().endsWith(".mjs") ||
                          source.contains("import ") ||
                          source.contains("export ");

        // Parse with Java
        System.out.println("=== Java Parser AST ===");
        try {
            Program javaAst = Parser.parse(source, isModule);
            String javaJson = mapper.writeValueAsString(javaAst);
            System.out.println(javaJson);
        } catch (Exception e) {
            System.err.println("Java parser failed: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("\n\n=== Acorn AST ===");

        // Parse with Acorn
        String sourceType = isModule ? "module" : "script";
        Path tempFile = Files.createTempFile("acorn-ast-", ".json");

        String[] cmd = {
            "node",
            "-e",
            "const acorn = require('acorn'); " +
            "const fs = require('fs'); " +
            "const source = fs.readFileSync(process.argv[1], 'utf-8'); " +
            "try { " +
            "  const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: '" + sourceType + "'}); " +
            "  fs.writeFileSync(process.argv[2], JSON.stringify(ast, (k,v) => typeof v === 'bigint' ? null : v, 2)); " +
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
        int exitCode = process.waitFor();

        if (exitCode == 0) {
            String acornJson = Files.readString(tempFile);
            System.out.println(acornJson);
        } else {
            String error = new String(process.getErrorStream().readAllBytes());
            System.err.println("Acorn failed: " + error);
        }

        Files.deleteIfExists(tempFile);
    }
}
