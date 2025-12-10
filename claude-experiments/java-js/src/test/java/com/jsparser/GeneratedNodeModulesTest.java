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
 * Generated: 2025-12-09T05:16:43.026658Z
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

    // motion-dom tests

    @Test
    @DisplayName("motion-dom.dev.js")
    void test_motionDom_dev() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_motion-dom.dev.js",
            false
        );
    }

    @Test
    @DisplayName("motion-dom parse-transform.mjs")
    void test_motionDom_parseTransform() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_es_render_dom_parse-transform.mjs",
            true
        );
    }

    @Test
    @DisplayName("motion-dom.js")
    void test_motionDom() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_motion-dom.js",
            false
        );
    }

    @Test
    @DisplayName("motion-dom cjs/index.js")
    void test_motionDom_cjs() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_cjs_index.js",
            false
        );
    }

    // framer-motion tests

    @Test
    @DisplayName("framer-motion size-rollup-dom-max.js")
    void test_framerMotion_sizeRollupDomMax() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-dom-max.js",
            true
        );
    }

    @Test
    @DisplayName("framer-motion.js")
    void test_framerMotion() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_framer-motion.js",
            false
        );
    }

    @Test
    @DisplayName("framer-motion.dev.js")
    void test_framerMotion_dev() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_framer-motion.dev.js",
            false
        );
    }

    @Test
    @DisplayName("framer-motion size-rollup-animate.js")
    void test_framerMotion_sizeRollupAnimate() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-animate.js",
            true
        );
    }

    @Test
    @DisplayName("framer-motion size-rollup-motion.js")
    void test_framerMotion_sizeRollupMotion() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-motion.js",
            true
        );
    }

    @Test
    @DisplayName("framer-motion dom.js")
    void test_framerMotion_dom() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_dom.js",
            false
        );
    }

    @Test
    @DisplayName("framer-motion size-rollup-dom-animation.js")
    void test_framerMotion_sizeRollupDomAnimation() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-dom-animation.js",
            true
        );
    }

    // ==================== AST Mismatch Tests ====================

    @Test
    @DisplayName("prefer-reflect-apply.js")
    void test_preferReflectApply() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_ink_node_modules_eslint-plugin-unicorn_rules_prefer-reflect-apply.js",
            false
        );
    }

    @Test
    @DisplayName("binaryIntegerLiteral.js")
    void test_binaryIntegerLiteral() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_binaryIntegerLiteral.js",
            false
        );
    }

    @Test
    @DisplayName("octalIntegerLiteral.js")
    void test_octalIntegerLiteral() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_octalIntegerLiteral.js",
            false
        );
    }

    @Test
    @DisplayName("octalIntegerLiteralES6.js")
    void test_octalIntegerLiteralES6() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_octalIntegerLiteralES6.js",
            false
        );
    }

    @Test
    @DisplayName("binaryIntegerLiteralES6.js")
    void test_binaryIntegerLiteralES6() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_binaryIntegerLiteralES6.js",
            false
        );
    }

    @Test
    @DisplayName("parenthesizedExpressionInternalComments.js")
    void test_parenthesizedExpressionInternalComments() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_parenthesizedExpressionInternalComments.js",
            false
        );
    }

    @Test
    @DisplayName("swc-8253.js")
    void test_swc8253() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_oxc_tasks_coverage_misc_pass_swc-8253.js",
            false
        );
    }

    @Test
    @DisplayName("property-name.js (BigInt)")
    void test_propertyNameBigInt() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_staging_sm_BigInt_property-name.js",
            false
        );
    }

    @Test
    @DisplayName("function-name-computed-01.js")
    void test_functionNameComputed01() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_staging_sm_Function_function-name-computed-01.js",
            false
        );
    }

    @Test
    @DisplayName("function-name-computed-02.js")
    void test_functionNameComputed02() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_staging_sm_Function_function-name-computed-02.js",
            false
        );
    }

    @Test
    @DisplayName("11.1.5-01.js (expressions)")
    void test_expressions11_1_5_01() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_staging_sm_expressions_11.1.5-01.js",
            false
        );
    }

    // ==================== Parse Failure Tests ====================

    @Test
    @DisplayName("watchpack.js (poll-app)")
    void test_watchpackPollApp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_poll-app_frontend_node_modules_watchpack_lib_watchpack.js",
            false
        );
    }

    @Test
    @DisplayName("watchpack.js (jimmyhmiller.github.io)")
    void test_watchpackJimmyhmiller() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_watchpack_lib_watchpack.js",
            false
        );
    }

    @Test
    @DisplayName("watchpack.js (ink)")
    void test_watchpackInk() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_ink_node_modules_watchpack_lib_watchpack.js",
            false
        );
    }

    @Test
    @DisplayName("doNotEmitDetachedCommentsAtStartOfLambdaFunction.js")
    void test_doNotEmitDetachedComments() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_doNotEmitDetachedCommentsAtStartOfLambdaFunction.js",
            false
        );
    }

    @Test
    @DisplayName("parserForStatement9.js")
    void test_parserForStatement9() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_parserForStatement9.js",
            false
        );
    }

    @Test
    @DisplayName("parserForOfStatement25.js")
    void test_parserForOfStatement25() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_TypeScript_tests_baselines_reference_parserForOfStatement25.js",
            false
        );
    }

    @Test
    @DisplayName("babel-16776-m.js")
    void test_babel16776m() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_oxc_tasks_coverage_misc_pass_babel-16776-m.js",
            true
        );
    }

    @Test
    @DisplayName("oxc-255.js")
    void test_oxc255() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_oxc_tasks_coverage_misc_pass_oxc-255.js",
            false
        );
    }

    @Test
    @DisplayName("object.js (oxc assignments)")
    void test_oxcAssignmentsObject() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_oxc_crates_oxc_semantic_tests_fixtures_oxc_js_assignments_object.js",
            false
        );
    }

    @Test
    @DisplayName("nested-assignment.js (oxc)")
    void test_oxcNestedAssignment() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_oxc_crates_oxc_semantic_tests_fixtures_oxc_js_assignments_nested-assignment.js",
            false
        );
    }

    @Test
    @DisplayName("object-pattern-emulates-undefined.js (annexB)")
    void test_annexBObjectPatternEmulatesUndefined() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_annexB_language_expressions_assignment_dstr_object-pattern-emulates-undefined.js",
            false
        );
    }

    @Test
    @DisplayName("rest-parameter-names.js")
    void test_restParameterNames() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_staging_sm_Function_rest-parameter-names.js",
            false
        );
    }

    @Test
    @DisplayName("destructuring-object-__proto__-2.js")
    void test_destructuringObjectProto2() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_staging_sm_expressions_destructuring-object-__proto__-2.js",
            false
        );
    }

    // ==================== Top-Level Await Tests (modules) ====================

    @Test
    @DisplayName("for-await-await-expr-regexp.js")
    void test_forAwaitAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_for-await-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("for-in-await-expr-regexp.js")
    void test_forInAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_for-in-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("export-dflt-assign-expr-await-expr-regexp.js")
    void test_exportDfltAssignExprAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_export-dflt-assign-expr-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("try-await-expr-regexp.js")
    void test_tryAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_try-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("while-await-expr-regexp.js")
    void test_whileAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_while-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("export-dft-class-decl-await-expr-regexp.js")
    void test_exportDftClassDeclAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_export-dft-class-decl-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("for-await-expr-regexp.js")
    void test_forAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_for-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("export-lex-decl-await-expr-regexp.js")
    void test_exportLexDeclAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_export-lex-decl-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("if-expr-await-expr-regexp.js")
    void test_ifExprAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_if-expr-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("if-block-await-expr-regexp.js")
    void test_ifBlockAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_if-block-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("export-var-await-expr-regexp.js")
    void test_exportVarAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_export-var-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("typeof-await-expr-regexp.js")
    void test_typeofAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_typeof-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("void-await-expr-regexp.js")
    void test_voidAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_void-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("export-class-decl-await-expr-regexp.js")
    void test_exportClassDeclAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_export-class-decl-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("top-level-await-expr-regexp.js")
    void test_topLevelAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_top-level-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("block-await-expr-regexp.js")
    void test_blockAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_block-await-expr-regexp.js",
            true
        );
    }

    @Test
    @DisplayName("for-of-await-expr-regexp.js")
    void test_forOfAwaitExprRegexp() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_module-code_top-level-await_syntax_for-of-await-expr-regexp.js",
            true
        );
    }

    // ==================== For-Await-Of Tests (scripts) ====================

    @Test
    @DisplayName("async-func-decl-dstr-obj-id-init-fn-name-cover.js")
    void test_asyncFuncDeclDstrObjIdInitFnNameCover() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_statements_for-await-of_async-func-decl-dstr-obj-id-init-fn-name-cover.js",
            false
        );
    }

    @Test
    @DisplayName("async-func-decl-dstr-obj-id-init-simple-no-strict.js")
    void test_asyncFuncDeclDstrObjIdInitSimpleNoStrict() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_statements_for-await-of_async-func-decl-dstr-obj-id-init-simple-no-strict.js",
            false
        );
    }

    @Test
    @DisplayName("async-gen-decl-dstr-obj-id-init-evaluation.js")
    void test_asyncGenDeclDstrObjIdInitEvaluation() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_statements_for-await-of_async-gen-decl-dstr-obj-id-init-evaluation.js",
            false
        );
    }

    @Test
    @DisplayName("async-func-decl-dstr-obj-id-init-fn-name-fn.js")
    void test_asyncFuncDeclDstrObjIdInitFnNameFn() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_statements_for-await-of_async-func-decl-dstr-obj-id-init-fn-name-fn.js",
            false
        );
    }

    @Test
    @DisplayName("async-gen-decl-dstr-obj-id-init-in.js")
    void test_asyncGenDeclDstrObjIdInitIn() throws Exception {
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_PlayGround_claude-experiments_java-js_test-oracles_test262_test_language_statements_for-await-of_async-gen-decl-dstr-obj-id-init-in.js",
            false
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
