package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.fail;

/**
 * Test for React development build parsing.
 * Uses cached file from test-oracles/adhoc-cache.
 */
public class DebugReactPosition {

    private static final ObjectMapper mapper;
    static {
        mapper = new ObjectMapper();
        mapper.enable(SerializationFeature.INDENT_OUTPUT);
        mapper.setDefaultPrettyPrinter(new com.fasterxml.jackson.core.util.DefaultPrettyPrinter()
            .withObjectIndenter(new com.fasterxml.jackson.core.util.DefaultIndenter("  ", "\n"))
            .withSeparators(com.fasterxml.jackson.core.util.Separators.createDefaultInstance()
                .withObjectFieldValueSpacing(com.fasterxml.jackson.core.util.Separators.Spacing.NONE)));
    }

    @Test
    @DisplayName("React experimental development build")
    public void testReactDevelopmentBuild() throws Exception {
        // Uses cached file from test-oracles/adhoc-cache
        assertASTMatches(
            "test-oracles/adhoc-cache/_react.development.js",
            false
        );
    }

    private void assertASTMatches(String cachedJsPath, boolean isModule) throws Exception {
        String source = Files.readString(Paths.get(cachedJsPath));

        Path acornTmp = Files.createTempFile("acorn-ast-", ".json");
        Path ourTmp = Files.createTempFile("our-ast-", ".json");

        try {
            // 1. Stream Acorn AST to temp file
            runAcornToFile(cachedJsPath, isModule, acornTmp);

            // 2. Stream our AST to temp file
            Program ourAst = Parser.parse(source, isModule);
            try (var out = new BufferedOutputStream(Files.newOutputStream(ourTmp))) {
                mapper.writeValue(out, ourAst);
            }

            // 3. Compare files via hash
            String acornHash = hashJsonContent(acornTmp);
            String ourHash = hashJsonContent(ourTmp);

            if (!acornHash.equals(ourHash)) {
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
        String script = """
            const acorn = require('acorn');
            const fs = require('fs');
            const source = fs.readFileSync(process.argv[1], 'utf-8');
            const ast = acorn.parse(source, {ecmaVersion: 2025, locations: true, sourceType: '%s'});
            const out = fs.createWriteStream(process.argv[2]);
            function writeJson(obj, indent) {
                if (obj === null) { out.write('null'); return; }
                if (typeof obj === 'undefined') { out.write('null'); return; }
                if (typeof obj === 'boolean') { out.write(obj.toString()); return; }
                if (typeof obj === 'number') { out.write(Number.isFinite(obj) ? obj.toString() : 'null'); return; }
                if (typeof obj === 'bigint') { out.write('"' + obj.toString() + '"'); return; }
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
        ObjectMapper localMapper = new ObjectMapper();
        com.fasterxml.jackson.databind.JsonNode tree = localMapper.readTree(input.toFile());
        java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
        hashNode(tree, digest);
        return java.util.HexFormat.of().formatHex(digest.digest());
    }

    private void hashNode(com.fasterxml.jackson.databind.JsonNode node, java.security.MessageDigest digest) {
        if (node.isNull()) { digest.update("null".getBytes()); return; }
        if (node.isBoolean()) { digest.update(node.asText().getBytes()); return; }
        if (node.isNumber()) { digest.update(node.asText().getBytes()); return; }
        if (node.isTextual()) {
            try { digest.update(mapper.writeValueAsBytes(node.asText())); }
            catch (Exception e) { throw new RuntimeException(e); }
            return;
        }
        if (node.isArray()) {
            digest.update((byte)'[');
            boolean first = true;
            for (var elem : node) {
                if (!first) digest.update((byte)',');
                first = false;
                hashNode(elem, digest);
            }
            digest.update((byte)']');
            return;
        }
        if (node.isObject()) {
            digest.update((byte)'{');
            var fields = new java.util.ArrayList<String>();
            node.fieldNames().forEachRemaining(fields::add);
            java.util.Collections.sort(fields);
            boolean first = true;
            for (String field : fields) {
                if (!first) digest.update((byte)',');
                first = false;
                try { digest.update(mapper.writeValueAsBytes(field)); }
                catch (Exception e) { throw new RuntimeException(e); }
                digest.update((byte)':');
                hashNode(node.get(field), digest);
            }
            digest.update((byte)'}');
        }
    }

    private void showFirstDifference(Path acornFile, Path ourFile, String jsPath) throws Exception {
        ProcessBuilder pb = new ProcessBuilder("diff", "-u",
            acornFile.toString(), ourFile.toString());
        Process p = pb.start();
        String diff = new String(p.getInputStream().readAllBytes());
        p.waitFor();

        System.err.println("\n=== AST MISMATCH: " + jsPath + " ===");
        String[] lines = diff.split("\\n");
        for (int i = 0; i < Math.min(50, lines.length); i++) {
            System.err.println(lines[i]);
        }
        if (lines.length > 50) {
            System.err.println("... (" + (lines.length - 50) + " more lines)");
        }
    }
}
