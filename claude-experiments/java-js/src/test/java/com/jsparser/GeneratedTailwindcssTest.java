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
 * Category: Tailwindcss
 * Generated: 2025-12-12T14:33:31.530275Z
 * 
 * Tests parse with both Acorn (real-time) and our parser, streaming ASTs to temp files
 * for memory-efficient byte-for-byte comparison.
 */
public class GeneratedTailwindcssTest {

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
    @DisplayName("../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/lib/util/bigSign.js")
    void test__PlayGround_claude_experiments_hlc_simulation_node_modules_tailwindcss_lib_util__0() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/lib/util/bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_util_bigSign.js",
            false
        );
    }

    @Test
    @DisplayName("../../../bingo-generator/node_modules/tailwindcss/lib/lib/offsets.js")
    void test__bingo_generator_node_modules_tailwindcss_lib_lib_offsets_js_1() throws Exception {
        // Original file: ../../../bingo-generator/node_modules/tailwindcss/lib/lib/offsets.js
        // Cached JS source: ..__..__..__bingo-generator_node_modules_tailwindcss_lib_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_lib_offsets.js",
            false
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/lib/lib/offsets.js")
    void test__PlayGround_claude_experiments_hlc_simulation_node_modules_tailwindcss_lib_lib_o_2() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/lib/lib/offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_offsets.js",
            false
        );
    }

    @Test
    @DisplayName("../../../PlayGround/meal-prep/node_modules/tailwindcss/src/lib/offsets.js")
    void test__PlayGround_meal_prep_node_modules_tailwindcss_src_lib_offsets_js_3() throws Exception {
        // Original file: ../../../PlayGround/meal-prep/node_modules/tailwindcss/src/lib/offsets.js
        // Cached JS source: ..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_lib_offsets.js",
            true
        );
    }

    @Test
    @DisplayName("../../../spandrel/node_modules/tailwindcss/src/lib/remap-bitfield.js")
    void test__spandrel_node_modules_tailwindcss_src_lib_remap_bitfield_js_4() throws Exception {
        // Original file: ../../../spandrel/node_modules/tailwindcss/src/lib/remap-bitfield.js
        // Cached JS source: ..__..__..__spandrel_node_modules_tailwindcss_src_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_lib_remap-bitfield.js",
            true
        );
    }

    @Test
    @DisplayName("..../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/lib/lib/remap-bitfield.js")
    void test__PlayGround_claude_experiments_hlc_simulation_node_modules_tailwindcss_lib_lib_r_5() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/lib/lib/remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_remap-bitfield.js",
            false
        );
    }

    @Test
    @DisplayName("../../../bingo-generator/node_modules/tailwindcss/lib/lib/remap-bitfield.js")
    void test__bingo_generator_node_modules_tailwindcss_lib_lib_remap_bitfield_js_6() throws Exception {
        // Original file: ../../../bingo-generator/node_modules/tailwindcss/lib/lib/remap-bitfield.js
        // Cached JS source: ..__..__..__bingo-generator_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_lib_remap-bitfield.js",
            false
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/src/lib/offsets.js")
    void test__PlayGround_claude_experiments_hlc_simulation_node_modules_tailwindcss_src_lib_o_7() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/src/lib/offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_offsets.js",
            true
        );
    }

    @Test
    @DisplayName("../../../bingo-generator/node_modules/tailwindcss/src/lib/offsets.js")
    void test__bingo_generator_node_modules_tailwindcss_src_lib_offsets_js_8() throws Exception {
        // Original file: ../../../bingo-generator/node_modules/tailwindcss/src/lib/offsets.js
        // Cached JS source: ..__..__..__bingo-generator_node_modules_tailwindcss_src_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_lib_offsets.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/src/util/bigSign.js")
    void test__PlayGround_claude_experiments_hlc_simulation_node_modules_tailwindcss_src_util__9() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/src/util/bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_util_bigSign.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/meal-prep/node_modules/tailwindcss/lib/lib/remap-bitfield.js")
    void test__PlayGround_meal_prep_node_modules_tailwindcss_lib_lib_remap_bitfield_js_10() throws Exception {
        // Original file: ../../../PlayGround/meal-prep/node_modules/tailwindcss/lib/lib/remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_remap-bitfield.js",
            false
        );
    }

    @Test
    @DisplayName("...ments/simple-nextjs-demo/simple-nextjs-demo/simple-ecommerce/node_modules/tailwindcss/dist/lib.js")
    void test__PlayGround_claude_experiments_simple_nextjs_demo_simple_nextjs_demo_simple_ecom_11() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/simple-nextjs-demo/simple-nextjs-demo/simple-ecommerce/node_modules/tailwindcss/dist/lib.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules_tailwindcss_dist_lib.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules_tailwindcss_dist_lib.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/meal-prep/node_modules/tailwindcss/lib/lib/offsets.js")
    void test__PlayGround_meal_prep_node_modules_tailwindcss_lib_lib_offsets_js_12() throws Exception {
        // Original file: ../../../PlayGround/meal-prep/node_modules/tailwindcss/lib/lib/offsets.js
        // Cached JS source: ..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_offsets.js",
            false
        );
    }

    @Test
    @DisplayName("../../../spandrel/node_modules/tailwindcss/src/lib/offsets.js")
    void test__spandrel_node_modules_tailwindcss_src_lib_offsets_js_13() throws Exception {
        // Original file: ../../../spandrel/node_modules/tailwindcss/src/lib/offsets.js
        // Cached JS source: ..__..__..__spandrel_node_modules_tailwindcss_src_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_lib_offsets.js",
            true
        );
    }

    @Test
    @DisplayName("../../../spandrel/node_modules/tailwindcss/lib/util/bigSign.js")
    void test__spandrel_node_modules_tailwindcss_lib_util_bigSign_js_14() throws Exception {
        // Original file: ../../../spandrel/node_modules/tailwindcss/lib/util/bigSign.js
        // Cached JS source: ..__..__..__spandrel_node_modules_tailwindcss_lib_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_util_bigSign.js",
            false
        );
    }

    @Test
    @DisplayName("../../../bingo-generator/node_modules/tailwindcss/lib/util/bigSign.js")
    void test__bingo_generator_node_modules_tailwindcss_lib_util_bigSign_js_15() throws Exception {
        // Original file: ../../../bingo-generator/node_modules/tailwindcss/lib/util/bigSign.js
        // Cached JS source: ..__..__..__bingo-generator_node_modules_tailwindcss_lib_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_util_bigSign.js",
            false
        );
    }

    @Test
    @DisplayName("../../../PlayGround/meal-prep/node_modules/tailwindcss/lib/util/bigSign.js")
    void test__PlayGround_meal_prep_node_modules_tailwindcss_lib_util_bigSign_js_16() throws Exception {
        // Original file: ../../../PlayGround/meal-prep/node_modules/tailwindcss/lib/util/bigSign.js
        // Cached JS source: ..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_util_bigSign.js",
            false
        );
    }

    @Test
    @DisplayName("../../../PlayGround/meal-prep/node_modules/tailwindcss/src/lib/remap-bitfield.js")
    void test__PlayGround_meal_prep_node_modules_tailwindcss_src_lib_remap_bitfield_js_17() throws Exception {
        // Original file: ../../../PlayGround/meal-prep/node_modules/tailwindcss/src/lib/remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_lib_remap-bitfield.js",
            true
        );
    }

    @Test
    @DisplayName("...und/claude-experiments/simple-nextjs-demo/simple-nextjs-demo/node_modules/tailwindcss/dist/lib.js")
    void test__PlayGround_claude_experiments_simple_nextjs_demo_simple_nextjs_demo_node_module_18() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/simple-nextjs-demo/simple-nextjs-demo/node_modules/tailwindcss/dist/lib.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules_tailwindcss_dist_lib.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules_tailwindcss_dist_lib.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/meal-prep/node_modules/tailwindcss/src/util/bigSign.js")
    void test__PlayGround_meal_prep_node_modules_tailwindcss_src_util_bigSign_js_19() throws Exception {
        // Original file: ../../../PlayGround/meal-prep/node_modules/tailwindcss/src/util/bigSign.js
        // Cached JS source: ..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_util_bigSign.js",
            true
        );
    }

    @Test
    @DisplayName("../../../spandrel/node_modules/tailwindcss/lib/lib/offsets.js")
    void test__spandrel_node_modules_tailwindcss_lib_lib_offsets_js_20() throws Exception {
        // Original file: ../../../spandrel/node_modules/tailwindcss/lib/lib/offsets.js
        // Cached JS source: ..__..__..__spandrel_node_modules_tailwindcss_lib_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_lib_offsets.js",
            false
        );
    }

    @Test
    @DisplayName("../../../bingo-generator/node_modules/tailwindcss/src/lib/remap-bitfield.js")
    void test__bingo_generator_node_modules_tailwindcss_src_lib_remap_bitfield_js_21() throws Exception {
        // Original file: ../../../bingo-generator/node_modules/tailwindcss/src/lib/remap-bitfield.js
        // Cached JS source: ..__..__..__bingo-generator_node_modules_tailwindcss_src_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_lib_remap-bitfield.js",
            true
        );
    }

    @Test
    @DisplayName("../../../spandrel/node_modules/tailwindcss/src/util/bigSign.js")
    void test__spandrel_node_modules_tailwindcss_src_util_bigSign_js_22() throws Exception {
        // Original file: ../../../spandrel/node_modules/tailwindcss/src/util/bigSign.js
        // Cached JS source: ..__..__..__spandrel_node_modules_tailwindcss_src_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_util_bigSign.js",
            true
        );
    }

    @Test
    @DisplayName("../../../spandrel/node_modules/tailwindcss/lib/lib/remap-bitfield.js")
    void test__spandrel_node_modules_tailwindcss_lib_lib_remap_bitfield_js_23() throws Exception {
        // Original file: ../../../spandrel/node_modules/tailwindcss/lib/lib/remap-bitfield.js
        // Cached JS source: ..__..__..__spandrel_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_lib_remap-bitfield.js",
            false
        );
    }

    @Test
    @DisplayName("..../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/src/lib/remap-bitfield.js")
    void test__PlayGround_claude_experiments_hlc_simulation_node_modules_tailwindcss_src_lib_r_24() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/hlc-simulation/node_modules/tailwindcss/src/lib/remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_remap-bitfield.js",
            true
        );
    }

    @Test
    @DisplayName("../../../bingo-generator/node_modules/tailwindcss/src/util/bigSign.js")
    void test__bingo_generator_node_modules_tailwindcss_src_util_bigSign_js_25() throws Exception {
        // Original file: ../../../bingo-generator/node_modules/tailwindcss/src/util/bigSign.js
        // Cached JS source: ..__..__..__bingo-generator_node_modules_tailwindcss_src_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_util_bigSign.js",
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
                if (typeof obj === 'number') { out.write(Number.isFinite(obj) ? obj.toString() : 'null'); return; }
                if (typeof obj === 'bigint') { out.write('null'); return; }
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
        // Parse JSON tree and hash with sorted keys using Jackson
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
