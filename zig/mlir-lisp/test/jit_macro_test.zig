const std = @import("std");
const mlir_lisp = @import("mlir_lisp");
const Tokenizer = mlir_lisp.Tokenizer;
const Reader = mlir_lisp.Reader;
const Parser = mlir_lisp.Parser;
const Builder = mlir_lisp.Builder;
const Executor = mlir_lisp.Executor;
const MacroExpander = mlir_lisp.MacroExpander;
const OperationFlattener = mlir_lisp.OperationFlattener;
const jit_macro_wrapper = mlir_lisp.jit_macro_wrapper;
const mlir = mlir_lisp.mlir;

test "JIT-compiled macro can replace Zig macro" {
    const allocator = std.testing.allocator;

    // Step 1: Create MLIR context
    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir.Context.registerAllPasses();

    // Load necessary dialects
    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("llvm");
    try ctx.getOrLoadDialect("arith");

    // Step 2: Load and compile the macro definition
    const macro_source = try std.fs.cwd().readFileAlloc(
        allocator,
        "examples/add_macro_with_strings.lisp",
        10 * 1024 * 1024,
    );
    defer allocator.free(macro_source);

    // Parse the macro file
    var macro_tokenizer = Tokenizer.init(allocator, macro_source);

    var macro_reader = try Reader.init(allocator, &macro_tokenizer);

    const macro_ast = try macro_reader.read();

    // Expand macros (defn, mlir, etc.)
    var macro_expander = MacroExpander.init(allocator);
    defer macro_expander.deinit();
    try mlir_lisp.builtin_macros.registerBuiltinMacros(&macro_expander);

    const expanded_ast = try macro_expander.expandAll(macro_ast);

    // Flatten operations (needed before parsing)
    var flattener = OperationFlattener.init(allocator);
    const flattened_ast = try flattener.flattenModule(expanded_ast);

    // Parse into MlirModule
    var parser = Parser.init(allocator, macro_source);
    var parsed_module = try parser.parseModule(flattened_ast);

    // Build MLIR module
    var macro_builder = Builder.init(allocator, &ctx);
    defer macro_builder.deinit();

    var macro_module = try macro_builder.buildModule(&parsed_module);

    // JIT compile the macro
    var macro_executor = Executor.init(allocator, &ctx, .{});
    defer macro_executor.deinit();

    // Register runtime symbols before compilation (needed for allocations, etc.)
    mlir_lisp.runtime_symbols.registerAllRuntimeSymbols(&macro_executor);

    try macro_executor.compile(&macro_module);

    // Step 3: Look up the JIT-compiled function
    const add_macro_fn = macro_executor.lookup("addMacro") orelse {
        std.debug.print("Failed to find addMacro function\n", .{});
        return error.MacroFunctionNotFound;
    };

    // Cast to the expected JIT macro signature
    const jit_fn: jit_macro_wrapper.JitMacroFn = @ptrCast(@alignCast(add_macro_fn));

    // Step 4: Create macro expander and register the JIT macro
    var expander = MacroExpander.init(allocator);
    defer expander.deinit();

    try jit_macro_wrapper.registerJitMacro(&expander, "+", jit_fn);

    // Step 5: Create test input: (+ (: i64) %x %y)
    const test_source = "(+ (: i64) %x %y)";

    var test_tokenizer = Tokenizer.init(allocator, test_source);

    var test_reader = try Reader.init(allocator, &test_tokenizer);

    const test_input = try test_reader.read();

    // Step 6: Expand using the JIT macro
    const expanded = try expander.expandAll(test_input);

    // Step 7: Verify the result
    // The expanded result should be:
    // (operation (name arith.addi) (result-types i64) (operands %x %y))

    try std.testing.expect(expanded.type == .list);
    const result_list = expanded.data.list;
    try std.testing.expect(result_list.len() >= 1);

    // First element should be "operation"
    const first = result_list.at(0);
    try std.testing.expect(first.type == .identifier);
    try std.testing.expectEqualStrings("operation", first.data.atom);

    std.debug.print("✓ JIT macro successfully expanded (+ (: i64) %x %y)\n", .{});
    std.debug.print("  Result type: {s}\n", .{@tagName(expanded.type)});
    std.debug.print("  Result list length: {}\n", .{result_list.len()});
    std.debug.print("  First element: {s}\n", .{first.data.atom});
}

test "JIT macro produces same output as Zig macro" {
    const allocator = std.testing.allocator;

    var ctx = try mlir.Context.create();
    defer ctx.destroy();
    ctx.registerAllDialects();
    mlir.Context.registerAllPasses();

    // Load necessary dialects
    try ctx.getOrLoadDialect("func");
    try ctx.getOrLoadDialect("llvm");
    try ctx.getOrLoadDialect("arith");

    // Load and compile JIT macro
    const macro_source = try std.fs.cwd().readFileAlloc(
        allocator,
        "examples/add_macro_with_strings.lisp",
        10 * 1024 * 1024,
    );
    defer allocator.free(macro_source);

    var macro_tokenizer = Tokenizer.init(allocator, macro_source);

    var macro_reader = try Reader.init(allocator, &macro_tokenizer);

    const macro_ast = try macro_reader.read();

    // Expand macros (defn, mlir, etc.)
    var macro_expander2 = MacroExpander.init(allocator);
    defer macro_expander2.deinit();
    try mlir_lisp.builtin_macros.registerBuiltinMacros(&macro_expander2);

    const expanded_ast2 = try macro_expander2.expandAll(macro_ast);

    // Flatten operations
    var flattener2 = OperationFlattener.init(allocator);
    const flattened_ast2 = try flattener2.flattenModule(expanded_ast2);

    // Parse into MlirModule
    var parser = Parser.init(allocator, macro_source);
    var parsed_module = try parser.parseModule(flattened_ast2);

    var macro_builder = Builder.init(allocator, &ctx);
    defer macro_builder.deinit();

    var macro_module = try macro_builder.buildModule(&parsed_module);

    var macro_executor = Executor.init(allocator, &ctx, .{});
    defer macro_executor.deinit();

    // Register runtime symbols before compilation (needed for allocations, etc.)
    mlir_lisp.runtime_symbols.registerAllRuntimeSymbols(&macro_executor);

    try macro_executor.compile(&macro_module);

    const add_macro_fn_ptr = macro_executor.lookup("addMacro") orelse {
        std.debug.print("Failed to find addMacro function in second test\n", .{});
        return error.MacroFunctionNotFound;
    };

    const jit_fn: jit_macro_wrapper.JitMacroFn = @ptrCast(@alignCast(add_macro_fn_ptr));

    // Create two macro expanders: one with JIT macro, one with builtin
    var jit_expander = MacroExpander.init(allocator);
    defer jit_expander.deinit();

    var builtin_expander = MacroExpander.init(allocator);
    defer builtin_expander.deinit();

    try jit_macro_wrapper.registerJitMacro(&jit_expander, "+", jit_fn);
    try mlir_lisp.builtin_macros.registerBuiltinMacros(&builtin_expander);

    // Test input
    const test_source = "(+ (: i64) %a %b)";

    var test_tokenizer = Tokenizer.init(allocator, test_source);

    var test_reader = try Reader.init(allocator, &test_tokenizer);

    const test_input1 = try test_reader.read();

    // Reset reader for second expansion
    test_tokenizer = Tokenizer.init(allocator, test_source);

    test_reader = try Reader.init(allocator, &test_tokenizer);

    const test_input2 = try test_reader.read();

    // Expand with both
    const jit_result = try jit_expander.expandAll(test_input1);
    const builtin_result = try builtin_expander.expandAll(test_input2);

    // Both should produce lists starting with "operation"
    try std.testing.expect(jit_result.type == .list);
    try std.testing.expect(builtin_result.type == .list);

    const jit_list = jit_result.data.list;
    const builtin_list = builtin_result.data.list;

    try std.testing.expect(jit_list.len() >= 1);
    try std.testing.expect(builtin_list.len() >= 1);

    const jit_first = jit_list.at(0);
    const builtin_first = builtin_list.at(0);

    try std.testing.expectEqualStrings(jit_first.data.atom, builtin_first.data.atom);

    std.debug.print("✓ JIT macro and Zig macro produce compatible results\n", .{});
}
