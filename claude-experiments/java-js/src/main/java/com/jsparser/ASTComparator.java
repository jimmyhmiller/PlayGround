package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * Compare Java parser AST with Acorn AST and show differences
 */
public class ASTComparator {
    private static final ObjectMapper mapper = new ObjectMapper()
            .enable(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS);

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.err.println("Usage: java ASTComparator <file-path>");
            System.exit(1);
        }

        String filePath = args[0];
        String source = Files.readString(Paths.get(filePath));

        // Parse with Acorn
        String acornJson = parseWithAcorn(source, filePath);
        if (acornJson == null) {
            System.err.println("Failed to parse with Acorn");
            System.exit(1);
        }

        // Parse with Java
        Program javaAst;
        try {
            // Try as module first (most Next.js files are modules)
            javaAst = Parser.parse(source, true);
        } catch (Exception e) {
            try {
                // Fallback to script
                javaAst = Parser.parse(source, false);
            } catch (Exception e2) {
                System.err.println("Failed to parse with Java parser: " + e2.getMessage());
                System.exit(1);
                return;
            }
        }

        String javaJson = mapper.writerWithDefaultPrettyPrinter().writeValueAsString(javaAst);

        // Print both for comparison
        System.out.println("=== Acorn AST ===");
        System.out.println(acornJson);
        System.out.println("\n=== Java AST ===");
        System.out.println(javaJson);

        // Check if they match
        Object acornObj = mapper.readValue(acornJson, Object.class);
        Object javaObj = mapper.readValue(javaJson, Object.class);

        if (acornObj.equals(javaObj)) {
            System.out.println("\n✓ ASTs match!");
        } else {
            System.out.println("\n✗ ASTs differ!");
        }
    }

    private static String parseWithAcorn(String source, String filePath) {
        try {
            ProcessBuilder pb = new ProcessBuilder("node", "scripts/parse-with-acorn.js", filePath);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                return output.toString();
            }
            return null;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
