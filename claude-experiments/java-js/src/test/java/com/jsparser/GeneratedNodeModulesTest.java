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
 * Generated: 2025-12-12T05:21:03.119864Z
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
    @DisplayName("...ller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/size-rollup-dom-max.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_framer_mo_0() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/size-rollup-dom-max.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-dom-max.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-dom-max.js",
            true
        );
    }

    @Test
    @DisplayName("...../PlayGround/claude-experiments/webgpu-renderer/node_modules/rollup/dist/es/shared/node-entry.js")
    void test__PlayGround_claude_experiments_webgpu_renderer_node_modules_rollup_dist_es_share_1() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/webgpu-renderer/node_modules/rollup/dist/es/shared/node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("../../../open-source/iongraph-standalone/node_modules/chai/lib/chai/core/assertions.js")
    void test__open_source_iongraph_standalone_node_modules_chai_lib_chai_core_assertions_js_2() throws Exception {
        // Original file: ../../../open-source/iongraph-standalone/node_modules/chai/lib/chai/core/assertions.js
        // Cached JS source: ..__..__..__open-source_iongraph-standalone_node_modules_chai_lib_chai_core_assertions.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_chai_lib_chai_core_assertions.js",
            true
        );
    }

    @Test
    @DisplayName(".../adhoc-cache/..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_chai_index.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_3() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_chai_index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_webgpu-renderer_node_modules_chai_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_webgpu-renderer_node_modules_chai_index.js",
            true
        );
    }

    @Test
    @DisplayName("...ava-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_util_bigSign.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_spandrel_node_mo_4() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_util_bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_src_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_src_util_bigSign.js",
            true
        );
    }

    @Test
    @DisplayName("...de-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules__tailwindcss_node_dist_index.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_5() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules__tailwindcss_node_dist_index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules__tailwindcss_node_dist_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules__tailwindcss_node_dist_index.js",
            false
        );
    }

    @Test
    @DisplayName("...ache/..__..__..__PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_shared_rollup.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_motio_6() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_shared_rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName(".....__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_util_bigSign.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_7() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_util_bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_util_bigSign.js",
            true
        );
    }

    @Test
    @DisplayName("...s/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_lib_remap-bitfield.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_meal__8() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_lib_remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_src_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_src_lib_remap-bitfield.js",
            true
        );
    }

    @Test
    @DisplayName("...und_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules_tailwindcss_dist_lib.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_9() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules_tailwindcss_dist_lib.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules_tailwindcss_dist_lib.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules_tailwindcss_dist_lib.js",
            true
        );
    }

    @Test
    @DisplayName("...mple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules__tailwindcss_node_dist_index.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_10() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules__tailwindcss_node_dist_index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules__tailwindcss_node_dist_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules__tailwindcss_node_dist_index.js",
            false
        );
    }

    @Test
    @DisplayName("...-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_offsets.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_meal__11() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_offsets.js",
            false
        );
    }

    @Test
    @DisplayName(".../test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_lib_remap-bitfield.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_spandrel_node_mo_12() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_lib_lib_remap-bitfield.js",
            false
        );
    }

    @Test
    @DisplayName(".../jimmyhmiller/Documents/Code/poll-app/frontend/node_modules/next/dist/compiled/webpack/bundle5.js")
    void test__Users_jimmyhmiller_Documents_Code_poll_app_frontend_node_modules_next_dist_comp_13() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/poll-app/frontend/node_modules/next/dist/compiled/webpack/bundle5.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_poll-app_frontend_node_modules_next_dist_compiled_webpack_bundle5.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_poll-app_frontend_node_modules_next_dist_compiled_webpack_bundle5.js",
            true
        );
    }

    @Test
    @DisplayName("...test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_util_bigSign.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_bingo_generator__14() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_util_bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_lib_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_lib_util_bigSign.js",
            false
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/grid/node_modules/rollup/dist/shared/rollup.js")
    void test__PlayGround_claude_experiments_grid_node_modules_rollup_dist_shared_rollup_js_15() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/grid/node_modules/rollup/dist/shared/rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_grid_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_grid_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("....__PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_es_shared_node-entry.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_16() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_es_shared_node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("../../../open-source/iongraph-standalone/node_modules/chai/index.js")
    void test__open_source_iongraph_standalone_node_modules_chai_index_js_17() throws Exception {
        // Original file: ../../../open-source/iongraph-standalone/node_modules/chai/index.js
        // Cached JS source: ..__..__..__open-source_iongraph-standalone_node_modules_chai_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_chai_index.js",
            true
        );
    }

    @Test
    @DisplayName("...e/..__..__..__PlayGround_claude-experiments_grid_node_modules_rollup_dist_es_shared_node-entry.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_18() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_grid_node_modules_rollup_dist_es_shared_node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_grid_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_grid_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("/Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/motion-dom/dist/cjs/index.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_motion_do_19() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/motion-dom/dist/cjs/index.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_cjs_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_cjs_index.js",
            false
        );
    }

    @Test
    @DisplayName("...r/Documents/Code/open-source/ink/node_modules/eslint-plugin-unicorn/rules/prefer-reflect-apply.js")
    void test__Users_jimmyhmiller_Documents_Code_open_source_ink_node_modules_eslint_plugin_un_20() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/open-source/ink/node_modules/eslint-plugin-unicorn/rules/prefer-reflect-apply.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_open-source_ink_node_modules_eslint-plugin-unicorn_rules_prefer-reflect-apply.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_ink_node_modules_eslint-plugin-unicorn_rules_prefer-reflect-apply.js",
            false
        );
    }

    @Test
    @DisplayName("...s/test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_chai_index.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_iong_21() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_chai_index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_iongraph-standalone_node_modules_chai_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_iongraph-standalone_node_modules_chai_index.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/ai-dashboard/node_modules/rollup/dist/es/shared/node-entry.js")
    void test__PlayGround_claude_experiments_ai_dashboard_node_modules_rollup_dist_es_shared_n_22() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/ai-dashboard/node_modules/rollup/dist/es/shared/node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("...ments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules_tailwindcss_dist_lib.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_23() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules_tailwindcss_dist_lib.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules_tailwindcss_dist_lib.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules_tailwindcss_dist_lib.js",
            true
        );
    }

    @Test
    @DisplayName("...test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_util_bigSign.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_bingo_generator__24() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_util_bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_src_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_src_util_bigSign.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/ai-dashboard2/node_modules/rollup/dist/shared/rollup.js")
    void test__PlayGround_claude_experiments_ai_dashboard2_node_modules_rollup_dist_shared_rol_25() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/ai-dashboard2/node_modules/rollup/dist/shared/rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("...racles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_lib_remap-bitfield.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_bingo_generator__26() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_lib_lib_remap-bitfield.js",
            false
        );
    }

    @Test
    @DisplayName("...jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/motion-dom/dist/motion-dom.dev.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_motion_do_27() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/motion-dom/dist/motion-dom.dev.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_motion-dom.dev.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_motion-dom.dev.js",
            false
        );
    }

    @Test
    @DisplayName("...iller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/size-rollup-motion.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_framer_mo_28() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/size-rollup-motion.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-motion.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-motion.js",
            true
        );
    }

    @Test
    @DisplayName("...mo/simple-nextjs-demo/node_modules/next/dist/compiled/react-experimental/cjs/react.development.js")
    void test__simple_nextjs_demo_simple_nextjs_demo_node_modules_next_dist_compiled_react_exp_29() throws Exception {
        // Original file: ../simple-nextjs-demo/simple-nextjs-demo/node_modules/next/dist/compiled/react-experimental/cjs/react.development.js
        // Cached JS source: _react.development.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_react.development.js",
            false
        );
    }

    @Test
    @DisplayName("../../../open-source/ink/node_modules/@faker-js/faker/dist/chunk-YQYVFZYE.js")
    void test__open_source_ink_node_modules_faker_js_faker_dist_chunk_YQYVFZYE_js_30() throws Exception {
        // Original file: ../../../open-source/ink/node_modules/@faker-js/faker/dist/chunk-YQYVFZYE.js
        // Cached JS source: ..__..__..__open-source_ink_node_modules__faker-js_faker_dist_chunk-YQYVFZYE.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules__faker-js_faker_dist_chunk-YQYVFZYE.js",
            true
        );
    }

    @Test
    @DisplayName("...ache/..__..__..__open-source_iongraph-standalone_node_modules_rollup_dist_es_shared_node-entry.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_iong_31() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_rollup_dist_es_shared_node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_iongraph-standalone_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_iongraph-standalone_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("...__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_remap-bitfield.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_32() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_remap-bitfield.js",
            true
        );
    }

    @Test
    @DisplayName("...ocuments/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/size-rollup-dom-animation.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_framer_mo_33() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/size-rollup-dom-animation.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-dom-animation.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-dom-animation.js",
            true
        );
    }

    @Test
    @DisplayName("...miller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/framer-motion.dev.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_framer_mo_34() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/framer-motion.dev.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_framer-motion.dev.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_framer-motion.dev.js",
            false
        );
    }

    @Test
    @DisplayName("...ava-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_util_bigSign.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_spandrel_node_mo_35() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_util_bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_lib_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_lib_util_bigSign.js",
            false
        );
    }

    @Test
    @DisplayName("...de-experiments/simple-nextjs-demo/simple-nextjs-demo/node_modules/@tailwindcss/node/dist/index.js")
    void test__PlayGround_claude_experiments_simple_nextjs_demo_simple_nextjs_demo_node_module_36() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/simple-nextjs-demo/simple-nextjs-demo/node_modules/@tailwindcss/node/dist/index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules__tailwindcss_node_dist_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_node_modules__tailwindcss_node_dist_index.js",
            false
        );
    }

    @Test
    @DisplayName("...-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_lib_offsets.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_meal__37() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_lib_offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_src_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_src_lib_offsets.js",
            true
        );
    }

    @Test
    @DisplayName(".....__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_util_bigSign.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_38() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_util_bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_util_bigSign.js",
            false
        );
    }

    @Test
    @DisplayName("...racles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_lib_remap-bitfield.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_bingo_generator__39() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_lib_remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_src_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_src_lib_remap-bitfield.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/webgpu-renderer/node_modules/rollup/dist/shared/rollup.js")
    void test__PlayGround_claude_experiments_webgpu_renderer_node_modules_rollup_dist_shared_r_40() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/webgpu-renderer/node_modules/rollup/dist/shared/rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("../../../open-source/iongraph-standalone/node_modules/rollup/dist/shared/rollup.js")
    void test__open_source_iongraph_standalone_node_modules_rollup_dist_shared_rollup_js_41() throws Exception {
        // Original file: ../../../open-source/iongraph-standalone/node_modules/rollup/dist/shared/rollup.js
        // Cached JS source: ..__..__..__open-source_iongraph-standalone_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("/Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/dom.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_framer_mo_42() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/dom.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_dom.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_dom.js",
            false
        );
    }

    @Test
    @DisplayName(".../test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_lib_offsets.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_bingo_generator__43() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_src_lib_offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_src_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_src_lib_offsets.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/motion-canvas/first-attempt/node_modules/rollup/dist/es/shared/node-entry.js")
    void test__PlayGround_motion_canvas_first_attempt_node_modules_rollup_dist_es_shared_node__44() throws Exception {
        // Original file: ../../../PlayGround/motion-canvas/first-attempt/node_modules/rollup/dist/es/shared/node-entry.js
        // Cached JS source: ..__..__..__PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName(".....__..__PlayGround_claude-experiments_hlc-simulation_node_modules__sinclair_typebox_value_cast.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_45() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules__sinclair_typebox_value_cast.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules__sinclair_typebox_value_cast.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules__sinclair_typebox_value_cast.js",
            false
        );
    }

    @Test
    @DisplayName(".....__..__..__PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_shared_rollup.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_46() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_shared_rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("...ller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/size-rollup-animate.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_framer_mo_47() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/size-rollup-animate.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-animate.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_size-rollup-animate.js",
            true
        );
    }

    @Test
    @DisplayName("...va-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_lib_verbose_info.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_ink__48() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_lib_verbose_info.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_execa_lib_verbose_info.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_execa_lib_verbose_info.js",
            true
        );
    }

    @Test
    @DisplayName("/Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/watchpack/lib/watchpack.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_watchpack_49() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/watchpack/lib/watchpack.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_watchpack_lib_watchpack.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_watchpack_lib_watchpack.js",
            false
        );
    }

    @Test
    @DisplayName("../../../open-source/ink/node_modules/execa/node_modules/pretty-ms/index.js")
    void test__open_source_ink_node_modules_execa_node_modules_pretty_ms_index_js_50() throws Exception {
        // Original file: ../../../open-source/ink/node_modules/execa/node_modules/pretty-ms/index.js
        // Cached JS source: ..__..__..__open-source_ink_node_modules_execa_node_modules_pretty-ms_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_node_modules_pretty-ms_index.js",
            true
        );
    }

    @Test
    @DisplayName("...__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_shared_rollup.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_51() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_shared_rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_webgpu-renderer_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("...mmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/framer-motion.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_framer_mo_52() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/framer-motion/dist/framer-motion.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_framer-motion.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_framer-motion_dist_framer-motion.js",
            false
        );
    }

    @Test
    @DisplayName("...__..__..__PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_es_shared_node-entry.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_motio_53() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_es_shared_node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("../../../open-source/iongraph-standalone/node_modules/rollup/dist/es/shared/node-entry.js")
    void test__open_source_iongraph_standalone_node_modules_rollup_dist_es_shared_node_entry_j_54() throws Exception {
        // Original file: ../../../open-source/iongraph-standalone/node_modules/rollup/dist/es/shared/node-entry.js
        // Cached JS source: ..__..__..__open-source_iongraph-standalone_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/grid/node_modules/rollup/dist/es/shared/node-entry.js")
    void test__PlayGround_claude_experiments_grid_node_modules_rollup_dist_es_shared_node_entr_55() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/grid/node_modules/rollup/dist/es/shared/node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_grid_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_grid_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("...java-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_lib_ipc_strict.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_ink__56() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_lib_ipc_strict.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_execa_lib_ipc_strict.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_execa_lib_ipc_strict.js",
            true
        );
    }

    @Test
    @DisplayName("/Users/jimmyhmiller/Documents/Code/open-source/ink/node_modules/watchpack/lib/watchpack.js")
    void test__Users_jimmyhmiller_Documents_Code_open_source_ink_node_modules_watchpack_lib_wa_57() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/open-source/ink/node_modules/watchpack/lib/watchpack.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_open-source_ink_node_modules_watchpack_lib_watchpack.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_open-source_ink_node_modules_watchpack_lib_watchpack.js",
            false
        );
    }

    @Test
    @DisplayName("..._..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_offsets.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_58() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_src_lib_offsets.js",
            true
        );
    }

    @Test
    @DisplayName("...__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_remap-bitfield.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_59() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_remap-bitfield.js",
            false
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/webgpu-renderer/node_modules/chai/index.js")
    void test__PlayGround_claude_experiments_webgpu_renderer_node_modules_chai_index_js_60() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/webgpu-renderer/node_modules/chai/index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_chai_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_webgpu-renderer_node_modules_chai_index.js",
            true
        );
    }

    @Test
    @DisplayName("../../../open-source/ink/node_modules/execa/lib/verbose/info.js")
    void test__open_source_ink_node_modules_execa_lib_verbose_info_js_61() throws Exception {
        // Original file: ../../../open-source/ink/node_modules/execa/lib/verbose/info.js
        // Cached JS source: ..__..__..__open-source_ink_node_modules_execa_lib_verbose_info.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_lib_verbose_info.js",
            true
        );
    }

    @Test
    @DisplayName("...c-cache/..__..__..__open-source_iongraph-standalone_node_modules_chai_lib_chai_core_assertions.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_iong_62() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_chai_lib_chai_core_assertions.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_iongraph-standalone_node_modules_chai_lib_chai_core_assertions.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_iongraph-standalone_node_modules_chai_lib_chai_core_assertions.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/ai-dashboard/node_modules/rollup/dist/shared/rollup.js")
    void test__PlayGround_claude_experiments_ai_dashboard_node_modules_rollup_dist_shared_roll_63() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/ai-dashboard/node_modules/rollup/dist/shared/rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("...__..__PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_es_shared_node-entry.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_64() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_es_shared_node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("../../../open-source/ink/node_modules/execa/node_modules/parse-ms/index.js")
    void test__open_source_ink_node_modules_execa_node_modules_parse_ms_index_js_65() throws Exception {
        // Original file: ../../../open-source/ink/node_modules/execa/node_modules/parse-ms/index.js
        // Cached JS source: ..__..__..__open-source_ink_node_modules_execa_node_modules_parse-ms_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_node_modules_parse-ms_index.js",
            true
        );
    }

    @Test
    @DisplayName("...oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_util_bigSign.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_meal__66() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_src_util_bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_src_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_src_util_bigSign.js",
            true
        );
    }

    @Test
    @DisplayName("...java-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_lib_offsets.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_spandrel_node_mo_67() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_lib_lib_offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_lib_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_lib_lib_offsets.js",
            false
        );
    }

    @Test
    @DisplayName("...acles/adhoc-cache/..__..__..__open-source_ink_node_modules__faker-js_faker_dist_chunk-YQYVFZYE.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_ink__68() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules__faker-js_faker_dist_chunk-YQYVFZYE.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules__faker-js_faker_dist_chunk-YQYVFZYE.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules__faker-js_faker_dist_chunk-YQYVFZYE.js",
            true
        );
    }

    @Test
    @DisplayName("..._..__PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_es_shared_node-entry.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_69() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_es_shared_node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName("...s/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_remap-bitfield.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_meal__70() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_lib_lib_remap-bitfield.js",
            false
        );
    }

    @Test
    @DisplayName("...oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_node_modules_parse-ms_index.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_ink__71() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_node_modules_parse-ms_index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_execa_node_modules_parse-ms_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_execa_node_modules_parse-ms_index.js",
            true
        );
    }

    @Test
    @DisplayName("...oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_util_bigSign.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_meal__72() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_meal-prep_node_modules_tailwindcss_lib_util_bigSign.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_lib_util_bigSign.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_meal-prep_node_modules_tailwindcss_lib_util_bigSign.js",
            false
        );
    }

    @Test
    @DisplayName(".../..__..__..__PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_shared_rollup.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_73() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_shared_rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_ai-dashboard_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("..../../PlayGround/claude-experiments/ai-dashboard2/node_modules/rollup/dist/es/shared/node-entry.js")
    void test__PlayGround_claude_experiments_ai_dashboard2_node_modules_rollup_dist_es_shared__74() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/ai-dashboard2/node_modules/rollup/dist/es/shared/node-entry.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_es_shared_node-entry.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_ai-dashboard2_node_modules_rollup_dist_es_shared_node-entry.js",
            true
        );
    }

    @Test
    @DisplayName(".../test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_lib_offsets.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_bingo_generator__75() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__bingo-generator_node_modules_tailwindcss_lib_lib_offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_lib_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___bingo-generator_node_modules_tailwindcss_lib_lib_offsets.js",
            false
        );
    }

    @Test
    @DisplayName("/Users/jimmyhmiller/Documents/Code/poll-app/frontend/node_modules/watchpack/lib/watchpack.js")
    void test__Users_jimmyhmiller_Documents_Code_poll_app_frontend_node_modules_watchpack_lib__76() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/poll-app/frontend/node_modules/watchpack/lib/watchpack.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_poll-app_frontend_node_modules_watchpack_lib_watchpack.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_poll-app_frontend_node_modules_watchpack_lib_watchpack.js",
            false
        );
    }

    @Test
    @DisplayName("...oc-cache/..__..__..__PlayGround_claude-experiments_grid_node_modules_rollup_dist_shared_rollup.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_77() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_grid_node_modules_rollup_dist_shared_rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_grid_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_grid_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("...adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_rollup_dist_shared_rollup.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_iong_78() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_iongraph-standalone_node_modules_rollup_dist_shared_rollup.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_iongraph-standalone_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_iongraph-standalone_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("...java-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_lib_offsets.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_spandrel_node_mo_79() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_lib_offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_src_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_src_lib_offsets.js",
            true
        );
    }

    @Test
    @DisplayName("../../../PlayGround/motion-canvas/first-attempt/node_modules/rollup/dist/shared/rollup.js")
    void test__PlayGround_motion_canvas_first_attempt_node_modules_rollup_dist_shared_rollup_j_80() throws Exception {
        // Original file: ../../../PlayGround/motion-canvas/first-attempt/node_modules/rollup/dist/shared/rollup.js
        // Cached JS source: ..__..__..__PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_shared_rollup.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_motion-canvas_first-attempt_node_modules_rollup_dist_shared_rollup.js",
            true
        );
    }

    @Test
    @DisplayName("...s/java-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_p-queue_dist_index.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_ink__81() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_p-queue_dist_index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_p-queue_dist_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_p-queue_dist_index.js",
            true
        );
    }

    @Test
    @DisplayName("/Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/motion-dom/dist/motion-dom.js")
    void test__Users_jimmyhmiller_Documents_Code_jimmyhmiller_github_io_node_modules_motion_do_82() throws Exception {
        // Original file: /Users/jimmyhmiller/Documents/Code/jimmyhmiller.github.io/node_modules/motion-dom/dist/motion-dom.js
        // Cached JS source: _Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_motion-dom.js
        assertASTMatches(
            "test-oracles/adhoc-cache/_Users_jimmyhmiller_Documents_Code_jimmyhmiller.github.io_node_modules_motion-dom_dist_motion-dom.js",
            false
        );
    }

    @Test
    @DisplayName("../../../PlayGround/claude-experiments/hlc-simulation/node_modules/@sinclair/typebox/value/cast.js")
    void test__PlayGround_claude_experiments_hlc_simulation_node_modules_sinclair_typebox_valu_83() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/hlc-simulation/node_modules/@sinclair/typebox/value/cast.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules__sinclair_typebox_value_cast.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules__sinclair_typebox_value_cast.js",
            false
        );
    }

    @Test
    @DisplayName("../../../open-source/ink/node_modules/execa/lib/ipc/strict.js")
    void test__open_source_ink_node_modules_execa_lib_ipc_strict_js_84() throws Exception {
        // Original file: ../../../open-source/ink/node_modules/execa/lib/ipc/strict.js
        // Cached JS source: ..__..__..__open-source_ink_node_modules_execa_lib_ipc_strict.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_lib_ipc_strict.js",
            true
        );
    }

    @Test
    @DisplayName("...mple-nextjs-demo/simple-nextjs-demo/simple-ecommerce/node_modules/@tailwindcss/node/dist/index.js")
    void test__PlayGround_claude_experiments_simple_nextjs_demo_simple_nextjs_demo_simple_ecom_85() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/simple-nextjs-demo/simple-nextjs-demo/simple-ecommerce/node_modules/@tailwindcss/node/dist/index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules__tailwindcss_node_dist_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_simple-nextjs-demo_simple-nextjs-demo_simple-ecommerce_node_modules__tailwindcss_node_dist_index.js",
            false
        );
    }

    @Test
    @DisplayName("..._..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_offsets.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_PlayGround_claud_86() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_offsets.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_offsets.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___PlayGround_claude-experiments_hlc-simulation_node_modules_tailwindcss_lib_lib_offsets.js",
            false
        );
    }

    @Test
    @DisplayName(".../test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_lib_remap-bitfield.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_spandrel_node_mo_87() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__spandrel_node_modules_tailwindcss_src_lib_remap-bitfield.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_src_lib_remap-bitfield.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___spandrel_node_modules_tailwindcss_src_lib_remap-bitfield.js",
            true
        );
    }

    @Test
    @DisplayName("../../../open-source/ink/node_modules/p-queue/dist/index.js")
    void test__open_source_ink_node_modules_p_queue_dist_index_js_88() throws Exception {
        // Original file: ../../../open-source/ink/node_modules/p-queue/dist/index.js
        // Cached JS source: ..__..__..__open-source_ink_node_modules_p-queue_dist_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_p-queue_dist_index.js",
            true
        );
    }

    @Test
    @DisplayName("...racles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_node_modules_pretty-ms_index.js")
    void test__PlayGround_claude_experiments_java_js_test_oracles_adhoc_cache_open_source_ink__89() throws Exception {
        // Original file: ../../../PlayGround/claude-experiments/java-js/test-oracles/adhoc-cache/..__..__..__open-source_ink_node_modules_execa_node_modules_pretty-ms_index.js
        // Cached JS source: ..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_execa_node_modules_pretty-ms_index.js
        assertASTMatches(
            "test-oracles/adhoc-cache/..__..__..__PlayGround_claude-experiments_java-js_test-oracles_adhoc-cache_..___..___..___open-source_ink_node_modules_execa_node_modules_pretty-ms_index.js",
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
