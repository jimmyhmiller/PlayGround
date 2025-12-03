package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for AST mismatches related to await identifier handling.
 * These tests document known bugs where the Java parser produces different ASTs than Acorn.
 * Each test will FAIL until the underlying parser bug is fixed.
 */
public class AwaitIdentifierMismatchTest {

    private static final ObjectMapper mapper = new ObjectMapper();

    @Test
    @DisplayName("new-await-script-code.js - await in script mode")
    void testNewAwaitScriptCode() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/module-code/top-level-await/new-await-script-code.js", false);
    }

    @Test
    @DisplayName("await-identifier.js - await in dynamic import")
    void testAwaitIdentifierDynamicImport() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/dynamic-import/assignment-expression/await-identifier.js", false);
    }

    @Test
    @DisplayName("2nd-param-await-ident.js - await in import attributes")
    void testAwaitIdentImportAttributes() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/dynamic-import/import-attributes/2nd-param-await-ident.js", false);
    }

    @Test
    @DisplayName("private-field-rhs-await-absent.js - await in 'in' expression")
    void testPrivateFieldAwaitAbsent() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/in/private-field-rhs-await-absent.js", false);
    }

    @Test
    @DisplayName("static-init-await-reference.js (function) - await in static init")
    void testStaticInitAwaitReferenceFunction() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/function/static-init-await-reference.js", false);
    }

    @Test
    @DisplayName("await-in-nested-function.js - await identifier in nested function")
    void testAwaitInNestedFunction() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/await/await-in-nested-function.js", false);
    }

    @Test
    @DisplayName("await-in-global.js - await identifier in global scope")
    void testAwaitInGlobal() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/await/await-in-global.js", false);
    }

    @Test
    @DisplayName("await-BindingIdentifier-in-global.js - await as binding identifier in global")
    void testAwaitBindingIdentifierInGlobal() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/await/await-BindingIdentifier-in-global.js", false);
    }

    @Test
    @DisplayName("await-in-nested-generator.js - await in nested generator")
    void testAwaitInNestedGenerator() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/await/await-in-nested-generator.js", false);
    }

    @Test
    @DisplayName("static-init-await-reference-normal.js - await in object method static init")
    void testStaticInitAwaitReferenceNormal() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/object/method-definition/static-init-await-reference-normal.js", false);
    }

    @Test
    @DisplayName("static-init-await-reference-generator.js - await in object method generator")
    void testStaticInitAwaitReferenceGenerator() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/object/method-definition/static-init-await-reference-generator.js", false);
    }

    @Test
    @DisplayName("static-init-await-reference.js (class) - await in class static init")
    void testStaticInitAwaitReferenceClass() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/class/static-init-await-reference.js", false);
    }

    @Test
    @DisplayName("static-init-await-reference.js (generators) - await in generator static init")
    void testStaticInitAwaitReferenceGenerators() throws Exception {
        assertASTMatches("test-oracles/test262/test/language/expressions/generators/static-init-await-reference.js", false);
    }

    // Helper method to compare ASTs
    private void assertASTMatches(String testFilePath, boolean isModule) throws Exception {
        // Read source file
        String source = Files.readString(Path.of(testFilePath));

        // Determine if module from Test262 frontmatter or filename
        boolean shouldUseModule = isModule || Parser.hasModuleFlag(source) ||
            testFilePath.endsWith("_FIXTURE.js");

        // Load expected AST from cache
        String cacheFilePath = testFilePath.replace("test-oracles/test262/test/", "test-oracles/test262-cache/") + ".json";
        String expectedJson = Files.readString(Path.of(cacheFilePath));

        // Parse with Java parser
        Program actual = Parser.parse(source, shouldUseModule);
        String actualJson = mapper.writeValueAsString(actual);

        // Parse both JSONs to Objects for structural comparison
        Object expectedObj = mapper.readValue(expectedJson, Object.class);
        Object actualObj = mapper.readValue(actualJson, Object.class);

        // Remove metadata field from expected
        if (expectedObj instanceof Map) {
            ((Map<?, ?>)expectedObj).remove("_metadata");
        }

        // Apply normalizations (same as Test262Runner)
        normalizeRegexValues(expectedObj, actualObj);
        normalizeBigIntValues(expectedObj, actualObj);

        // Assert - this will FAIL until the bug is fixed
        if (!Objects.deepEquals(expectedObj, actualObj)) {
            // Find and print differences for debugging
            List<String> diffs = findDifferences(expectedObj, actualObj, "");
            System.out.println("\nAST differences for " + testFilePath + ":");
            diffs.stream().limit(10).forEach(System.out::println);
        }

        assertTrue(Objects.deepEquals(expectedObj, actualObj),
            "AST mismatch for " + testFilePath + " (see console output for details)");
    }

    // Normalization methods copied from Test262Runner
    @SuppressWarnings("unchecked")
    private void normalizeRegexValues(Object expected, Object actual) {
        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            if ("Literal".equals(expMap.get("type")) && expMap.containsKey("regex")) {
                if (expMap.get("value") == null && actMap.get("value") instanceof Map) {
                    Map<?, ?> actValue = (Map<?, ?>) actMap.get("value");
                    if (actValue.isEmpty()) {
                        actMap.put("value", null);
                    }
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

    // Helper to find and report differences
    @SuppressWarnings("unchecked")
    private List<String> findDifferences(Object expected, Object actual, String path) {
        List<String> diffs = new java.util.ArrayList<>();

        if (expected == null && actual == null) {
            return diffs;
        }

        if (expected == null) {
            diffs.add(path + ": expected null but got " + actual);
            return diffs;
        }

        if (actual == null) {
            diffs.add(path + ": expected " + expected + " but got null");
            return diffs;
        }

        if (expected instanceof Map && actual instanceof Map) {
            Map<String, Object> expMap = (Map<String, Object>) expected;
            Map<String, Object> actMap = (Map<String, Object>) actual;

            for (String key : expMap.keySet()) {
                if (!actMap.containsKey(key)) {
                    diffs.add(path + "." + key + ": missing in actual");
                } else {
                    diffs.addAll(findDifferences(expMap.get(key), actMap.get(key), path + "." + key));
                }
            }

            for (String key : actMap.keySet()) {
                if (!expMap.containsKey(key)) {
                    diffs.add(path + "." + key + ": extra in actual");
                }
            }
        } else if (expected instanceof List && actual instanceof List) {
            List<Object> expList = (List<Object>) expected;
            List<Object> actList = (List<Object>) actual;

            if (expList.size() != actList.size()) {
                diffs.add(path + ": size mismatch (expected " + expList.size() + ", got " + actList.size() + ")");
            }

            for (int i = 0; i < Math.min(expList.size(), actList.size()); i++) {
                diffs.addAll(findDifferences(expList.get(i), actList.get(i), path + "[" + i + "]"));
            }
        } else if (!expected.equals(actual)) {
            diffs.add(path + ": expected " + expected + " but got " + actual);
        }

        return diffs;
    }
}
