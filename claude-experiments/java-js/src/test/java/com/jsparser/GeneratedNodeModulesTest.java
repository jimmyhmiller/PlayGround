package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * Auto-generated tests from cached JS source files.
 * Category: NodeModules
 * Generated: 2025-12-09T05:12:56.270440Z
 * 
 * Tests parse with both Acorn (real-time) and our parser, streaming ASTs to temp files
 * for memory-efficient byte-for-byte comparison.
 */
public class GeneratedNodeModulesTest {

    private static final ObjectMapper mapper;
    static {
        mapper = new ObjectMapper();
        // Configure to match Node.js JSON.stringify(obj, null, 2) format
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.setDefaultPrettyPrinter(new com.fasterxml.jackson.core.util.DefaultPrettyPrinter()
            .withObjectIndenter(new com.fasterxml.jackson.core.util.DefaultIndenter("  ", "\n"))
            .withSeparators(com.fasterxml.jackson.core.util.Separators.createDefaultInstance()
                .withObjectFieldValueSpacing(com.fasterxml.jackson.core.util.Separators.Spacing.NONE)));
    }

    @Test
    @DisplayName(".../jimmyhmiller/Documents/Code/poll-app/frontend/node_modules/next/dist/compiled/webpack/bundle5.js")
    void test__Users_jimmyhmiller_Documents_Code_poll_app_frontend_node_modules_next_dist_comp_0() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/poll-app/frontend/node_modules/next/dist/compiled/webpack/bundle5.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_poll-app_frontend_node_modules_next_dist_compiled_webpack_bundle5.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_poll-app_frontend_node_modules_next_dist_compiled_webpack_bundle5.js",
            true
        );
    }

    // Helper methods

    private void assertASTMatches(String cachedJsPath, boolean isModule) throws Exception {
        // Read cached JS source
        String source = Files.readString(Paths.get(cachedJsPath));

        Path acornTmp = Files.createTempFile("acorn-ast-", ".json");
        Path ourTmp = Files.createTempFile("our-ast-", ".json");

        try {
            // 1. Stream Acorn AST to temp file (via Node.js subprocess)
            runAcornToFile(cachedJsPath, isModule, acornTmp);

            // 2. Stream our AST to temp file
            Program ourAst = Parser.parse(source, isModule);
            try (var out = new BufferedOutputStream(Files.newOutputStream(ourTmp))) {
                mapper.writeValue(out, ourAst);
            }

            // 3. Compare files via hash (ignoring whitespace)
            String acornHash = hashJsonContent(acornTmp);
            String ourHash = hashJsonContent(ourTmp);

            if (!acornHash.equals(ourHash)) {
                // Hashes differ - show first difference
                showFirstDifference(acornTmp, ourTmp, cachedJsPath);
                fail("AST mismatch for " + cachedJsPath);
            }
        } finally {
            Files.deleteIfExists(acornTmp);
            Files.deleteIfExists(ourTmp);
        }
    }

    private void runAcornToFile(String jsPath, boolean isModule, Path outputFile) throws Exception {
        String sourceType = isModule ? "module" : "script";
        // Use streaming JSON writer to avoid Node.js string length limits
        String script = """
            const acorn = require('acorn');
            const fs = require('fs');
            const source = fs.readFileSync(process.argv[1], 'utf-8');
            const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: '%s'});
            // Streaming JSON writer to avoid string length limits
            const out = fs.createWriteStream(process.argv[2]);
            function writeJson(obj, indent) {
                if (obj === null) { out.write('null'); return; }
                if (typeof obj === 'undefined') { out.write('null'); return; }
                if (typeof obj === 'boolean') { out.write(obj.toString()); return; }
                if (typeof obj === 'number') { out.write(obj.toString()); return; }
                if (typeof obj === 'bigint') { out.write('\"' + obj.toString() + '\"'); return; }
                if (typeof obj === 'string') { out.write(JSON.stringify(obj)); return; }
                if (Array.isArray(obj)) {
                    out.write('[');
                    for (let i = 0; i < obj.length; i++) {
                        if (i > 0) out.write(',');
                        out.write('\\n' + indent + '  ');
                        writeJson(obj[i], indent + '  ');
                    }
                    if (obj.length > 0) out.write('\\n' + indent);
                    out.write(']');
                    return;
                }
                out.write('{');
                const keys = Object.keys(obj);
                for (let i = 0; i < keys.length; i++) {
                    if (i > 0) out.write(',');
                    out.write('\\n' + indent + '  ');
                    out.write(JSON.stringify(keys[i]) + ': ');
                    writeJson(obj[keys[i]], indent + '  ');
                }
                if (keys.length > 0) out.write('\\n' + indent);
                out.write('}');
            }
            writeJson(ast, '');
            out.write('\\n');
            out.end(() => process.exit(0));
            """.formatted(sourceType);

        String[] cmd = { "node", "--max-old-space-size=8192", "-e", script, jsPath, outputFile.toString() };
        ProcessBuilder pb = new ProcessBuilder(cmd);
        Process process = pb.start();

        // Capture stderr
        CompletableFuture<String> errorFuture = CompletableFuture.supplyAsync(() -> {
            try {
                return new String(process.getErrorStream().readAllBytes());
            } catch (IOException e) { return ""; }
        });

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new RuntimeException("Acorn failed: " + errorFuture.get());
        }
    }

    private String hashJsonContent(Path input) throws Exception {
        java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
        try (BufferedReader reader = Files.newBufferedReader(input)) {
            boolean inString = false;
            boolean escape = false;
            int ch;
            while ((ch = reader.read()) != -1) {
                if (escape) {
                    digest.update((byte) ch);
                    escape = false;
                } else if (ch == '\\' && inString) {
                    digest.update((byte) ch);
                    escape = true;
                } else if (ch == '"') {
                    digest.update((byte) ch);
                    inString = !inString;
                } else if (inString) {
                    digest.update((byte) ch);
                } else if (!Character.isWhitespace(ch)) {
                    digest.update((byte) ch);
                }
            }
        }
        return java.util.HexFormat.of().formatHex(digest.digest());
    }

    private void showFirstDifference(Path acornFile, Path ourFile, String jsPath) throws Exception {
        // Run diff to get actual differences
        ProcessBuilder pb = new ProcessBuilder("diff", "-u",
            acornFile.toString(), ourFile.toString());
        Process p = pb.start();
        String diff = new String(p.getInputStream().readAllBytes());
        p.waitFor();

        System.err.println("\n=== AST MISMATCH: " + jsPath + " ===");
        // Show first 50 lines of diff
        String[] lines = diff.split("\\n");
        for (int i = 0; i < Math.min(50, lines.length); i++) {
            System.err.println(lines[i]);
        }
        if (lines.length > 50) {
            System.err.println("... (" + (lines.length - 50) + " more lines)");
        }
    }
}
