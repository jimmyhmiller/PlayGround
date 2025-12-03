package com.jsparser;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Disabled;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Auto-generated tests from Acorn AST cache.
 * Category: MonacoEditor
 * Generated: 2025-12-03T05:10:12.105139Z
 *
 * These tests verify that the Java parser produces the same AST as Acorn.
 */
public class GeneratedMonacoEditorTest {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("../ai-dashboard2/node_modules/monaco-editor/min/vs/editor.api-CalNCsUg.js")
    void test__ai_dashboard2_node_modules_monaco_editor_min_vs_editor_api_CalNCsUg_js_0() throws Exception {
        // Original file: ../ai-dashboard2/node_modules/monaco-editor/min/vs/editor.api-CalNCsUg.js
        // Cache file: ..__ai-dashboard2_node_modules_monaco-editor_min_vs_editor.api-CalNCsUg.js.json
        assertASTMatches(
            "test-oracles/adhoc-cache/..__ai-dashboard2_node_modules_monaco-editor_min_vs_editor.api-CalNCsUg.js.json",
            "../ai-dashboard2/node_modules/monaco-editor/min/vs/editor.api-CalNCsUg.js",
            true
        );
    }

    @Test
    @DisplayName("../ai-dashboard2/node_modules/monaco-editor/esm/vs/platform/contextkey/common/contextkey.js")
    void test__ai_dashboard2_node_modules_monaco_editor_esm_vs_platform_contextkey_common_cont_1() throws Exception {
        // Original file: ../ai-dashboard2/node_modules/monaco-editor/esm/vs/platform/contextkey/common/contextkey.js
        // Cache file: ..__ai-dashboard2_node_modules_monaco-editor_esm_vs_platform_contextkey_common_contextkey.js.json
        assertASTMatches(
            "test-oracles/adhoc-cache/..__ai-dashboard2_node_modules_monaco-editor_esm_vs_platform_contextkey_common_contextkey.js.json",
            "../ai-dashboard2/node_modules/monaco-editor/esm/vs/platform/contextkey/common/contextkey.js",
            true
        );
    }

    @Test
    @DisplayName("../ai-dashboard2/node_modules/monaco-editor/dev/vs/editor.api-CykLys8L.js")
    void test__ai_dashboard2_node_modules_monaco_editor_dev_vs_editor_api_CykLys8L_js_2() throws Exception {
        // Original file: ../ai-dashboard2/node_modules/monaco-editor/dev/vs/editor.api-CykLys8L.js
        // Cache file: ..__ai-dashboard2_node_modules_monaco-editor_dev_vs_editor.api-CykLys8L.js.json
        assertASTMatches(
            "test-oracles/adhoc-cache/..__ai-dashboard2_node_modules_monaco-editor_dev_vs_editor.api-CykLys8L.js.json",
            "../ai-dashboard2/node_modules/monaco-editor/dev/vs/editor.api-CykLys8L.js",
            true
        );
    }

    @Test
    @DisplayName("...-dashboard2/node_modules/monaco-editor/esm/vs/editor/browser/viewParts/lineNumbers/lineNumbers.js")
    void test__ai_dashboard2_node_modules_monaco_editor_esm_vs_editor_browser_viewParts_lineNu_3() throws Exception {
        // Original file: ../ai-dashboard2/node_modules/monaco-editor/esm/vs/editor/browser/viewParts/lineNumbers/lineNumbers.js
        // Cache file: ..__ai-dashboard2_node_modules_monaco-editor_esm_vs_editor_browser_viewParts_lineNumbers_lineNumbers.js.json
        assertASTMatches(
            "test-oracles/adhoc-cache/..__ai-dashboard2_node_modules_monaco-editor_esm_vs_editor_browser_viewParts_lineNumbers_lineNumbers.js.json",
            "../ai-dashboard2/node_modules/monaco-editor/esm/vs/editor/browser/viewParts/lineNumbers/lineNumbers.js",
            true
        );
    }

    // Helper methods

    private void assertASTMatches(String cacheFilePath, String originalFilePath, boolean isModule) throws Exception {
        // Read cached Acorn AST
        String cacheContent = Files.readString(Paths.get(cacheFilePath));
        JsonNode cacheRoot = mapper.readTree(cacheContent);
        // Handle both old format (just AST) and new format (with _metadata)
        JsonNode expectedAst = cacheRoot.has("ast") ? cacheRoot.get("ast") : cacheRoot;

        // Read original source file
        Path originalFile = Paths.get(originalFilePath);
        if (!Files.exists(originalFile)) {
            // File might have been moved/deleted - skip test
            System.out.println("Skipping - file not found: " + originalFilePath);
            return;
        }

        String source = Files.readString(originalFile);

        // Parse with Java parser
        Program actual = Parser.parse(source, isModule);
        String actualJson = mapper.writeValueAsString(actual);

        // Compare ASTs
        String expectedJson = mapper.writeValueAsString(expectedAst);
        Object expectedObj = mapper.readValue(expectedJson, Object.class);
        Object actualObj = mapper.readValue(actualJson, Object.class);

        // Apply normalizations
        normalizeRegexValues(expectedObj, actualObj);
        normalizeBigIntValues(expectedObj, actualObj);

        if (!Objects.deepEquals(expectedObj, actualObj)) {
            System.out.println("\nAST mismatch for: " + originalFilePath);
            // Could add diff printing here
        }

        assertTrue(Objects.deepEquals(expectedObj, actualObj),
            "AST mismatch for " + originalFilePath);
    }

    @SuppressWarnings("unchecked")
    private void normalizeRegexValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;
            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("regex")) {
                if (expMap.get("value") == null && actMap.get("value") instanceof Map) {
                    Map<?, ?> actValue = (Map<?, ?>) actMap.get("value");
                    if (actValue.isEmpty()) { actMap.put("value", null); }
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
    private void normalizeBigIntValues(Object expected, Object actual) {
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
                            if (expBI.equals(actBI)) { actMap.put("bigint", expStr); }
                        } catch (NumberFormatException e) {}
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
}
